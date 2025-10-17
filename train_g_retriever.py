import pandas as pd
import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.models.g_retriever import LLM
from torch_geometric.nn import GCN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Run G-Retriever Training")
parser.add_argument('--llm_name', type=str, default='distilgpt2', help='Name of the Hugging Face language model to use.')
parser.add_argument('--data_limit', type=int, default=200, help='Limit data samples for testing. Set to 0 for all data.')
args = parser.parse_args()

# --- Configuration ---
DATA_LIMIT = args.data_limit if args.data_limit > 0 else None
RUN_DIR = './experiment_runs/run_2025-10-11_19-13-00/'

# --- 2. Load Data ---
print("🚀 Starting the G-Retriever script...")
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

# --- 3. Prepare Documents for Retriever ---
print("\nPreparing text documents...")
kept_indices = [int(i) for i in old_to_new_idx.keys()]
nodes_df_filtered = nodes_df.iloc[kept_indices].copy()
nodes_df_filtered['new_idx'] = list(old_to_new_idx.values())
nodes_df_filtered = nodes_df_filtered.sort_values('new_idx').set_index('new_idx')
node_texts = nodes_df_filtered['description'].fillna('No description available.').tolist()
assert len(node_texts) == graph.num_nodes, "Mismatch!"
print(f"✅ Documents prepared and filtered. Final count: {len(node_texts)}")

# --- 4. Initialize the G-Retriever Model Components ---
print(f"\nInitializing GRetriever with model: {args.llm_name}...")

# 4.1 Pre-load the model to count its parameters
print(f"   - Pre-loading '{args.llm_name}' to count parameters...")
if 't5' in args.llm_name:
    temp_model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_name)
else:
    temp_model = AutoModelForCausalLM.from_pretrained(args.llm_name)
num_params = sum(p.numel() for p in temp_model.parameters())
print(f"     - Model has {num_params} parameters.")
del temp_model

# 4.2 Initialize the LLM wrapper
print(f"   - Initializing LLM wrapper...")
llm_wrapper = LLM(model_name=args.llm_name, num_params=num_params)

# 4.3 Create a proper PyG GNN model (GCN)
print(f"   - Creating GCN model...")
GNN_HIDDEN_CHANNELS = 256
GNN_OUT_CHANNELS = 256
gnn_model = GCN(
    in_channels=graph.num_features,
    hidden_channels=GNN_HIDDEN_CHANNELS,
    num_layers=2, # Example: 2 layers
    out_channels=GNN_OUT_CHANNELS,
).to('cuda')

# 4.4 Initialize GRetriever by passing the LLM and GNN objects
retriever = GRetriever(llm=llm_wrapper, gnn=gnn_model, mlp_out_channels=4096)
print(f"✅ GRetriever model initialized successfully!")

# --- 5. Prepare Training & Evaluation Data ---
# ... (Rest of the script is unchanged)
print("\nPreparing training, validation, and test datasets...")
def prepare_data(mask, limit=None):
    queries, answers = [], []
    indices = mask.nonzero(as_tuple=True)[0]
    if limit:
        indices = indices[:limit]
    for idx in tqdm(indices, desc="Preparing data"):
        idx = idx.item()
        node_info = nodes_df_filtered.loc[idx]
        node_name = node_info['id']
        query = f"What are the tasks for the model or dataset named '{node_name}'?"
        true_label_indices = graph.y[idx].nonzero(as_tuple=True)[0]
        true_tasks = [idx_to_task[i.item()] for i in true_label_indices]
        answer = ", ".join(sorted(true_tasks))
        queries.append(query)
        answers.append(answer)
    return queries, answers

train_queries, train_answers = prepare_data(graph.train_mask, limit=DATA_LIMIT)
val_queries, val_answers = prepare_data(graph.val_mask, limit=DATA_LIMIT)
test_queries, test_answers = prepare_data(graph.test_mask, limit=DATA_LIMIT)
print(f"✅ Datasets prepared: {len(train_queries)} train, {len(val_queries)} val, {len(test_queries)} test examples.")

# --- 6. Fine-Tune the G-Retriever Model ---
print(f"\nSkipping fine-tuning for testing purposes. Load existing checkpoint instead.")
# print("\nFine-tuning G-Retriever model... This will take a while.")
# checkpoint_path = os.path.join(RUN_DIR, 'g_retriever_checkpoint.pt')
# retriever.fine_tune(
#     train_queries=train_queries,
#     train_answers=train_answers,
#     val_queries=val_queries,
#     val_answers=val_answers,
#     x=graph.x,
#     edge_index=graph.edge_index,
#     epochs=3,
#     batch_size=8,
#     save_path=checkpoint_path,
# )
# print(f"✅ Fine-tuning complete. Best model saved to {checkpoint_path}")

# --- 7. Evaluate on the Test Set ---
# print("\nEvaluating model on the test set...")
# retriever.load_fine_tuned(checkpoint_path)
# predictions = []
# for query in tqdm(test_queries, desc="Generating predictions"):
#     predictions.append(retriever.query(query, x=graph.x, edge_index=graph.edge_index))
# all_labels = list(task_to_idx.keys())
# mlb = MultiLabelBinarizer(classes=all_labels)
# y_true = mlb.fit_transform([ans.split(', ') for ans in test_answers if ans])
# y_pred = mlb.transform([pred.split(', ') for pred in predictions if pred])
# micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
# print("\n--- Final Test Results ---")
# print(f"📊 Test Micro-F1 Score: {micro_f1:.4f}")
# print("--------------------------")

print("\Script Finished")