import torch
import os
import pandas as pd
import json
import numpy as np
import re
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from torch_geometric.nn import GAT
from torch_geometric.llm.models import LLM, GRetriever

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
SAVED_MODEL_DIR = "g_retriever_multilabel"
BASE_EXP_DIR = 'experiment_runs/run_2025-10-11_19-13-00' 
GRAPH_FILE = 'graph_data/final_graph.pt'
NODES_DF_PATH = os.path.join(BASE_EXP_DIR, 'nodes_df.pkl')
BRIDGE_FILE = os.path.join(BASE_EXP_DIR, 'old_to_new_idx.json')
OUTPUT_FILE = "evaluation_results.json" 

TASK_LIST = [
    'text-generation', 'question-answering', 'text-to-video', 'image-to-video', 
    'image-to-3d', 'robotics', 'translation', 'feature-extraction', 'text-to-3d', 
    'text-to-speech', 'automatic-speech-recognition', 'image-classification', 
    'table-question-answering', 'fill-mask', 'multiple-choice', 
    'visual-question-answering', 'summarization', 'image-to-text', 
    'image-feature-extraction', 'text-to-image', 'text-to-audio', 
    'reinforcement-learning', 'image-text-to-text', 'text-classification', 
    'sentence-similarity', 'zero-shot-classification', 'text-retrieval', 
    'token-classification', 'object-detection', 'audio-classification', 
    'image-segmentation', 'time-series-forecasting', 'video-classification', 
    'zero-shot-image-classification', 'any-to-any', 'image-to-image', 
    'depth-estimation', 'tabular-classification', 'tabular-regression', 
    'table-to-text', 'video-text-to-text', 'audio-to-audio', 
    'voice-activity-detection', 'audio-text-to-text', 
    'document-question-answering', 'visual-document-retrieval', 'text-ranking', 
    'graph-ml', 'tabular-to-text', 'unconditional-image-generation', 
    'mask-generation', 'keypoint-detection', 'zero-shot-object-detection', 
    'video-to-video'
]
TASK_LIST_PROMPT = f"""Here is a list of possible tasks: {TASK_LIST}.
Please predict all relevant tasks for this model. Output only the indices, separated by commas."""

class MyGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GAT(in_channels, hidden_channels, num_layers=1, edge_dim=edge_dim)
        self.conv2 = GAT(hidden_channels, out_channels, num_layers=1, edge_dim=edge_dim)
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

def load_data():
    print("Loading graph and data...")
    raw_graph = torch.load(GRAPH_FILE, weights_only=False)
    x = raw_graph.x.float()
    
    if isinstance(raw_graph.edge_index, tuple):
        src, dst = raw_graph.edge_index
        if isinstance(src, np.ndarray): src = torch.from_numpy(src)
        if isinstance(dst, np.ndarray): dst = torch.from_numpy(dst)
        src = src.long() if isinstance(src, torch.Tensor) else torch.tensor(src).long()
        dst = dst.long() if isinstance(dst, torch.Tensor) else torch.tensor(dst).long()
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = raw_graph.edge_index.long()
        
    graph = Data(x=x, edge_index=edge_index)
    graph.num_nodes = raw_graph.num_nodes
    del raw_graph

    nodes_df = pd.read_pickle(NODES_DF_PATH)
    with open(BRIDGE_FILE, 'r') as f:
        old_to_new_idx_dict = json.load(f)
    new_to_old_idx = {value: int(key) for key, value in old_to_new_idx_dict.items()}
    
    dataset_list = []
    for new_idx, old_idx in new_to_old_idx.items():
        if new_idx >= graph.num_nodes: continue
        node_data = nodes_df.iloc[old_idx]
        labels = node_data['y']
        if node_data['type'] == 'model' and len(labels) > 0:
            dataset_list.append({
                "name": node_data['id'],
                "label": labels,
                "graph_id": new_idx
            })
            
    dataset = Dataset.from_list(dataset_list)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    
    return graph, splits['train'], splits['test']

def create_collate_fn(graph):
    _graph = graph
    def collate_fn(batch):
        questions = []
        true_labels = []
        data_list = []
        
        for item in batch:
            prompt = f"{TASK_LIST_PROMPT}\n\nFor this node, here's the info:\n{item['name']}"
            questions.append(f"[INST] {prompt} [/INST]")
            true_labels.append(item['label']) 
            
            node_subset, edge_index_sub, _, _ = k_hop_subgraph(
                item['graph_id'], num_hops=1, edge_index=_graph.edge_index, 
                relabel_nodes=True, num_nodes=_graph.num_nodes
            )
            data_list.append(Data(x=_graph.x[node_subset], edge_index=edge_index_sub))
            
        graph_batch = Batch.from_data_list(data_list)
        return questions, true_labels, graph_batch
    return collate_fn

def load_trained_model(input_dim):
    print("Loading model components...")
    llm = LLM(model_name=f"{SAVED_MODEL_DIR}/lora_adapters", n_gpus=1)
    
    gnn = MyGraphEncoder(input_dim, 256, 256, None)
    gnn.load_state_dict(torch.load(f"{SAVED_MODEL_DIR}/gnn.pt"))
    
    model = GRetriever(llm, gnn, use_lora=False)
    model.projector.load_state_dict(torch.load(f"{SAVED_MODEL_DIR}/projector.pt"))
    model.eval()
    return model

def parse_prediction(text):
    nums = re.findall(r'\d+', text)
    return [int(n) for n in nums if int(n) < len(TASK_LIST)]

def run_inference_loop(model, dataset, graph, desc_name):
    print(f"\nRunning inference on {desc_name} set ({len(dataset)} samples)...")
    
    collate_fn = create_collate_fn(graph)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for questions, labels, graph_batch in tqdm(loader):
            preds_text = model.inference(
                question=questions,
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
                max_out_tokens=20
            )
            
            for pred_text, true_label in zip(preds_text, labels):
                pred_indices = parse_prediction(pred_text)
                predictions.append(pred_indices)
                ground_truth.append(true_label)
                
    return predictions, ground_truth

def main():
    graph, train_dataset, test_dataset = load_data()
    
    print("Calculating training class frequencies...")
    train_counts = [0] * len(TASK_LIST)
    for item in tqdm(train_dataset):
        for label in item['label']:
            train_counts[label] += 1
            
    model = load_trained_model(graph.x.shape[1])
    
    test_preds, test_labels = run_inference_loop(model, test_dataset, graph, "TEST")
    
    # train_preds, train_labels = run_inference_loop(model, train_dataset, graph, "TRAIN")
    
    results = {
        "train_counts": train_counts,
        "test": {
            "predictions": test_preds,
            "ground_truth": test_labels
        },
    }
    
    print(f"Saving combined results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)
    print("Done.")

if __name__ == "__main__":
    main()