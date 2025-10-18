import pandas as pd
import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.models.g_retriever import LLM
from torch_geometric.nn import GCN
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import psutil
import time

parser = argparse.ArgumentParser(description="Run G-Retriever Training")
parser.add_argument('--llm_name', type=str, default='google/gemma-2b', help='Name of the Hugging Face language model to use.')
parser.add_argument('--data_limit', type=int, default=0, help='Limit data samples for testing. Set to 0 for all data.')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
args = parser.parse_args()

DATA_LIMIT = args.data_limit if args.data_limit > 0 else None
RUN_DIR = './experiment_runs/run_2025-10-11_19-13-00/'
CHECKPOINT_PATH = os.path.join(RUN_DIR, 'g_retriever_manual_checkpoint.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"--- Memory Usage [{stage}]: {mem_info.rss / (1024**3):.2f} GB ---")

print(f"🚀 Starting the G-Retriever script (Device: {DEVICE})...")
try:
    print(f"Loading graph from '{RUN_DIR}final_graph.pt'...")
    graph = torch.load(f'{RUN_DIR}final_graph.pt', weights_only=False)
    
    print(f"Loading node metadata from '{RUN_DIR}nodes_df.pkl'...")
    nodes_df = pd.read_pickle(f'{RUN_DIR}nodes_df.pkl')
    print(f"Loading task mapping from '{RUN_DIR}task_to_idx.json'...")
    with open(f'{RUN_DIR}task_to_idx.json', 'r') as f:
        task_to_idx = json.load(f)
    idx_to_task = {v: k for k, v in task_to_idx.items()}
    print(f"Loading node index map from '{RUN_DIR}old_to_new_idx.json'...")
    with open(f'{RUN_DIR}old_to_new_idx.json', 'r') as f:
        old_to_new_idx = json.load(f)
except FileNotFoundError as e:
    print(f"❌ Error: A file was not found: {e}")
    exit()
print("\n✅ Data loaded successfully!")
print(f"   - Graph attributes: {graph}")
print(f"   - Graph num_features: {graph.num_features}")

print_memory_usage("------After Data Loading------")

print("\nPreparing text documents...")
kept_indices = [int(i) for i in old_to_new_idx.keys()]
nodes_df_filtered = nodes_df.iloc[kept_indices].copy()
nodes_df_filtered['new_idx'] = list(old_to_new_idx.values())
nodes_df_filtered = nodes_df_filtered.sort_values('new_idx').set_index('new_idx')
node_texts = nodes_df_filtered['description'].fillna('No description available.').tolist()
assert len(node_texts) == graph.num_nodes, "Mismatch!"
print(f"✅ Documents prepared and filtered. Final count: {len(node_texts)}")

print_memory_usage("------After Document Preparation------")

print(f"\nInitializing GRetriever with model: {args.llm_name}...")
print(f"   - Pre-loading '{args.llm_name}' to count parameters...")
if 't5' in args.llm_name:
    temp_model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_name)
else:
    temp_model = AutoModelForCausalLM.from_pretrained(args.llm_name)
num_params = sum(p.numel() for p in temp_model.parameters())
print(f"     - Model has {num_params} parameters.")
del temp_model

print(f"   - Initializing LLM wrapper...")
llm_wrapper = LLM(model_name=args.llm_name, num_params=num_params)

print(f"   - Creating GCN model...")
GNN_HIDDEN_CHANNELS = 256
GNN_OUT_CHANNELS = 256
gnn_model = GCN(
    in_channels=graph.num_features,
    hidden_channels=GNN_HIDDEN_CHANNELS,
    num_layers=2,
    out_channels=GNN_OUT_CHANNELS,
)

retriever = GRetriever(llm=llm_wrapper, gnn=gnn_model, mlp_out_channels=2048) # Ensure retriever is on the correct device
print(f"✅ GRetriever model initialized successfully!")

print_memory_usage("------After Model Initialization------")

print("\nPreparing DataLoaders...")
class QADataset(Dataset):
    def __init__(self, queries, answers, node_indices):
        self.queries = queries
        self.answers = answers
        self.node_indices = node_indices
    def __len__(self):
        return len(self.queries)
    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "answer": self.answers[idx],
            "node_idx": self.node_indices[idx].item()
        }

def get_indices(mask, limit=None):
    indices = mask.nonzero(as_tuple=True)[0]
    if limit:
        indices = indices[:limit]
    return indices

train_indices = get_indices(graph.train_mask, limit=DATA_LIMIT)
val_indices = get_indices(graph.val_mask, limit=DATA_LIMIT)
test_indices = get_indices(graph.test_mask, limit=DATA_LIMIT)

def prepare_qa_pairs(indices):
    queries, answers = [], []
    for node_idx in tqdm(indices, desc="Preparing Q/A pairs"):
        node_idx = node_idx.item()
        node_info = nodes_df_filtered.loc[node_idx]
        node_name = node_info['id']
        query = f"What are the tasks for the model or dataset named '{node_name}'?"
        true_label_indices = graph.y[node_idx].nonzero(as_tuple=True)[0]
        true_tasks = [idx_to_task[i.item()] for i in true_label_indices]
        answer = ", ".join(sorted(true_tasks))
        queries.append(query)
        answers.append(answer)
    return queries, answers

train_queries, train_answers = prepare_qa_pairs(train_indices)
val_queries, val_answers = prepare_qa_pairs(val_indices)
test_queries, test_answers = prepare_qa_pairs(test_indices)

train_dataset = QADataset(train_queries, train_answers, train_indices)
val_dataset = QADataset(val_queries, val_answers, val_indices)
test_dataset = QADataset(test_queries, test_answers, test_indices)

def simple_collate_fn(batch):
    queries = [item['query'] for item in batch]
    answers = [item['answer'] for item in batch]

    local_edge_index = graph.edge_index
    if isinstance(local_edge_index, tuple) and len(local_edge_index) == 2:
        local_edge_index = torch.stack(local_edge_index, dim=0)
    elif not isinstance(local_edge_index, torch.Tensor):
         raise TypeError(f"Collate: Unexpected type for edge_index: {type(local_edge_index)}")

    if local_edge_index.dim() == 2 and local_edge_index.shape[0] != 2:
        local_edge_index = local_edge_index.t()

    batch_x = graph.x.to(DEVICE)
    batch_edge_index = local_edge_index.to(DEVICE)
    batch_vector = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

    return {
        "query": queries,
        "answer": answers,
        "x": batch_x,
        "edge_index": batch_edge_index,
        "batch": batch_vector
    }

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=simple_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=simple_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=simple_collate_fn)
print(f"✅ DataLoaders prepared.")

print_memory_usage("------After DataLoader Preparation------")

optimizer = AdamW(retriever.parameters(), lr=args.lr)
best_val_loss = float('inf')

print("\n🚀 Starting Training Loop...")
for epoch in range(args.epochs):
    start_time = time.time()
    retriever.train()
    total_train_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
        optimizer.zero_grad()
        loss = retriever(
            question=batch['query'],
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
            label=batch['answer']
        )
        print_memory_usage(f"------Epoch {epoch+1} Batch {i+1} After Forward Pass------")
        if loss is not None:
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        else:
            print("Warning: Received None loss from model forward pass.")
    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

    retriever.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
            loss = retriever(
                question=batch['query'],
                x=batch['x'],
                edge_index=batch['edge_index'],
                batch=batch['batch'],
                label=batch['answer']
            )
            if loss is not None:
                total_val_loss += loss.item()
            else:
                 print("Warning: Received None loss during validation.")
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch+1}/{args.epochs} | Duration: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(retriever.state_dict(), CHECKPOINT_PATH)
        print(f"   ✨ New best validation loss. Checkpoint saved to {CHECKPOINT_PATH}")

print("\n✅ Training complete.")

print("\nLoading best model for final evaluation...")
if os.path.exists(CHECKPOINT_PATH):
    retriever.load_state_dict(torch.load(CHECKPOINT_PATH))
    retriever.eval()
    print("Evaluating model on the test set...")
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Test Predictions"):
             batch_predictions = retriever.query(
                 query=batch['query'],
                 x=graph.x,
                 edge_index=graph.edge_index
             )
             predictions.extend(batch_predictions)

    if len(predictions) == len(test_answers):
        all_labels = list(task_to_idx.keys())
        mlb = MultiLabelBinarizer(classes=all_labels)
        valid_test_answers = [ans for ans in test_answers if ans]
        valid_predictions = [pred for pred in predictions if pred]
        min_len = min(len(valid_test_answers), len(valid_predictions))
        y_true = mlb.fit_transform([ans.split(', ') for ans in valid_test_answers[:min_len]])
        y_pred = mlb.transform([pred.split(', ') for pred in valid_predictions[:min_len]])
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        print("\n--- Final Test Results ---")
        print(f"📊 Test Micro-F1 Score: {micro_f1:.4f}")
        print("--------------------------")
    else:
        print(f"❌ Error: Number of predictions ({len(predictions)}) does not match number of test answers ({len(test_answers)}). Cannot calculate F1 score.")
else:
    print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}. Cannot evaluate.")

print("\nScript Finished.")