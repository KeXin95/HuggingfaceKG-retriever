import pandas as pd
import time
import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.models.g_retriever import LLM
from torch_geometric.nn import GCN
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run G-Retriever Zero-Shot Inference")
parser.add_argument('--llm_name', type=str, default='google/gemma-2b', help='Name of the Hugging Face language model.')
parser.add_argument('--data_limit', type=int, default=10, help='Number of test samples to query.')
args = parser.parse_args()

DATA_LIMIT = args.data_limit if args.data_limit > 0 else 10
RUN_DIR = './experiment_runs/run_2025-10-11_19-13-00/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 Starting G-Retriever Zero-Shot Inference (Device: {DEVICE})...")
try:
    print(f"Loading graph from '{RUN_DIR}final_graph.pt'...")
    graph = torch.load(f'{RUN_DIR}final_graph.pt', weights_only=False)

    print(f"   - Initial type of graph.edge_index: {type(graph.edge_index)}")
    local_edge_index = graph.edge_index

    if isinstance(local_edge_index, tuple) and len(local_edge_index) == 2 and all(isinstance(t, torch.Tensor) for t in local_edge_index):
        print("   - Converting edge_index from tuple to tensor...")
        local_edge_index = torch.stack(local_edge_index, dim=0)
    elif not isinstance(local_edge_index, torch.Tensor):
         raise TypeError(f"Loaded edge_index is not a Tensor or convertible tuple: {type(local_edge_index)}")
    else:
        print("   - edge_index is already a tensor.")

    if local_edge_index.dim() == 2 and local_edge_index.shape[0] != 2:
        print("   - Transposing edge_index to shape [2, num_edges]...")
        local_edge_index = local_edge_index.t().contiguous()

    graph.edge_index = local_edge_index
    print(f"   - Final edge_index type: {type(graph.edge_index)}")
    print(f"   - Final edge_index shape: {graph.edge_index.shape}")


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
except Exception as e:
    print(f"❌ An unexpected error occurred during data loading: {e}")
    exit()
print("\n✅ Data loaded successfully!")

print("\nPreparing text documents...")
kept_indices = [int(i) for i in old_to_new_idx.keys()]
nodes_df_filtered = nodes_df.iloc[kept_indices].copy()
nodes_df_filtered['new_idx'] = list(old_to_new_idx.values())
nodes_df_filtered = nodes_df_filtered.sort_values('new_idx').set_index('new_idx')
print(f"✅ Documents prepared.")

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
gnn_model = GCN(
    in_channels=graph.num_features, hidden_channels=256, num_layers=2, out_channels=256
).to(DEVICE)

retriever = GRetriever(llm=llm_wrapper, gnn=gnn_model, mlp_out_channels=2048).to(DEVICE)
print(f"✅ GRetriever model initialized successfully!")

print(f"\nPreparing {DATA_LIMIT} sample queries from the test set...")
def get_indices(mask, limit=None):
    indices = mask.nonzero(as_tuple=True)[0]
    if limit:
        limit = min(limit, len(indices))
        indices = indices[:limit]
    return indices

test_indices = get_indices(graph.test_mask, limit=DATA_LIMIT)

queries_to_run = []
true_answers = []
if len(test_indices) == 0:
    print("⚠️ Warning: No test indices found within the data limit or in the test mask.")
else:
    for node_idx in test_indices:
        node_idx = node_idx.item()
        node_info = nodes_df_filtered.loc[node_idx]
        node_name = node_info['id']
        query = f"What are the tasks for the model or dataset named '{node_name}'?"
        true_label_indices = graph.y[node_idx].nonzero(as_tuple=True)[0]
        true_tasks = [idx_to_task[i.item()] for i in true_label_indices]
        answer = ", ".join(sorted(true_tasks))
        queries_to_run.append(query)
        true_answers.append(answer)
    print(f"✅ Sample queries prepared.")

print("\nRunning zero-shot inference...")
retriever.eval() 
predictions = []
graph_x_device = graph.x.to(DEVICE)
graph_edge_index_device = graph.edge_index.to(DEVICE)
batch_vector_device = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

start_time = time.time()
if queries_to_run:
    with torch.inference_mode():
        for i in tqdm(range(len(queries_to_run)), desc="Generating Predictions"):
            query = queries_to_run[i]
            try:
                prediction = retriever.inference(
                    question=[query],
                    x=graph_x_device,
                    edge_index=graph_edge_index_device,
                    batch=batch_vector_device
                )
                predictions.append(prediction[0].strip() if prediction else "Model returned no prediction")
            except Exception as e:
                print(f"\n❌ Error during inference for query {i}: {e}")
                predictions.append(f"Error during inference: {e}")

    end_time = time.time()
    inference_duration = end_time - start_time
    print(f"\n✅ Inference complete. Duration: {inference_duration:.2f}s")

    print("\n--- Zero-Shot Inference Results ---")
    for i in range(len(queries_to_run)):
        print(f"\nQuery: {queries_to_run[i]}")
        print(f"True Answer:  {true_answers[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 20)
else:
    print("No queries to run for inference.")


print("\nScript Finished.")