import torch
import os
import warnings
import pandas as pd
import json
import numpy as np
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from torch_geometric.nn import GAT
from torch_geometric.llm.models import LLM, GRetriever
from typing import List, Optional

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

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
NEW_MODEL_NAME = "g_retriever_multilabel" 
BASE_EXP_DIR = 'experiment_runs/run_2025-10-11_19-13-00' 
GRAPH_FILE = 'graph_data/final_graph.pt'
NODES_DF_PATH = os.path.join(BASE_EXP_DIR, 'nodes_df.pkl')
BRIDGE_FILE = os.path.join(BASE_EXP_DIR, 'old_to_new_idx.json')

BGE_EMBED_DIM = 768 
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 256
GNN_EDGE_DIM = None 

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

formatted_task_list = "\n".join([f"{i}: {task}" for i, task in enumerate(TASK_LIST)])
TASK_LIST_PROMPT = f"""Below is the list of valid tasks and their corresponding IDs:
{formatted_task_list}

Please predict the relevant tasks for this model.
Output ONLY a comma-separated list of the corresponding IDs (e.g. 0, 5, 12)."""

def load_full_data():
    print("Loading all data sources...")
    print(f"Loading graph from: {GRAPH_FILE}")
    
    raw_graph = torch.load(GRAPH_FILE, weights_only=False)
    
    print("Converting raw graph to clean PyG Data object...")
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

    graph = Data(
        x=x, 
        edge_index=edge_index,
        train_mask=raw_graph.train_mask,
        test_mask=raw_graph.test_mask
    )
    graph.num_nodes = raw_graph.num_nodes
    del raw_graph
    
    nodes_df = pd.read_pickle(NODES_DF_PATH)
    with open(BRIDGE_FILE, 'r') as f:
        old_to_new_idx_dict = json.load(f)
    new_to_old_idx = {value: int(key) for key, value in old_to_new_idx_dict.items()}
    
    train_list = []
    test_list = []
    
    print("Filtering data using Graph Masks...")
    
    for new_idx, old_idx in new_to_old_idx.items():
        if new_idx >= graph.num_nodes: continue
        
        node_data = nodes_df.iloc[old_idx]
        labels = node_data['y']
        
        if len(labels) > 0:
            data_item = {
                "name": node_data['id'],
                "label": labels,
                "graph_id": new_idx
            }

            if graph.train_mask[new_idx].item():
                train_list.append(data_item)
            elif graph.test_mask[new_idx].item():
                test_list.append(data_item)
            
    print(f"  Train Samples: {len(train_list)}")
    print(f"  Test Samples:  {len(test_list)}")

    train_dataset = Dataset.from_list(train_list)
    eval_dataset = Dataset.from_list(test_list)
    
    return graph, train_dataset, eval_dataset

def create_collate_fn(graph):
    _graph = graph
    
    def collate_fn(batch: List[dict]):
        questions = []
        labels = []
        data_list = []
        
        for item in batch:
            raw_labels = item['label']
            
            if len(raw_labels) == len(TASK_LIST): 
                active_indices = [i for i, val in enumerate(raw_labels) if val == 1]
            else:
                active_indices = raw_labels

            label_string = ", ".join(map(str, active_indices))
            
            prompt = f"{TASK_LIST_PROMPT}\n\nFor this node, here's the info:\n{item['name']}"
            
            questions.append(f"[INST] {prompt} [/INST]")
            labels.append(f" {label_string}")
            
            node_subset, edge_index_sub, _, _ = k_hop_subgraph(
                item['graph_id'], num_hops=1, edge_index=_graph.edge_index, 
                relabel_nodes=True, num_nodes=_graph.num_nodes
            )
            data_list.append(Data(x=_graph.x[node_subset], edge_index=edge_index_sub))
            
        graph_batch = Batch.from_data_list(data_list)
        return questions, labels, graph_batch
        
    return collate_fn

def save_checkpoint(model, step, base_dir):
    ckpt_dir = os.path.join(base_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving safety checkpoint to {ckpt_dir}...")
    torch.save(model.gnn.state_dict(), f"{ckpt_dir}/gnn.pt")
    torch.save(model.projector.state_dict(), f"{ckpt_dir}/projector.pt")
    
    if hasattr(model, 'llm_generator'):
        model.llm_generator.save_pretrained(f"{ckpt_dir}/lora_adapters")
    else:
        model.llm.llm.save_pretrained(f"{ckpt_dir}/lora_adapters")

def main_train():
    graph, train_dataset, eval_dataset = load_full_data()
    
    global BGE_EMBED_DIM
    BGE_EMBED_DIM = graph.x.shape[1]
    
    print(f"Loading base model: {MODEL_NAME}...")
    llm = LLM(model_name=MODEL_NAME, n_gpus=1)
    
    print("Initializing GNN...")
    gnn = MyGraphEncoder(
        in_channels=BGE_EMBED_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
        edge_dim=GNN_EDGE_DIM
    )

    model = GRetriever(llm=llm, gnn=gnn, use_lora=True, mlp_out_tokens=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    collate_fn = create_collate_fn(graph)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                            collate_fn=collate_fn, num_workers=0)
    
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(5):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (questions, labels, graph_batch) in enumerate(pbar):
            optimizer.zero_grad()
            loss = model(
                question=questions,
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch, 
                label=labels
            )
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            
            if (step + 1) % 5000 == 0:
                save_checkpoint(model, step + 1, NEW_MODEL_NAME)

    print("Training finished.")
    os.makedirs(NEW_MODEL_NAME, exist_ok=True)
    
    print("Saving FINAL model...")
    torch.save(model.gnn.state_dict(), f"./{NEW_MODEL_NAME}/gnn.pt")
    torch.save(model.projector.state_dict(), f"./{NEW_MODEL_NAME}/projector.pt")
    
    print("Merging LoRA adapters...")
    if hasattr(model, 'llm_generator'):
        target_model = model.llm_generator
    else:
        target_model = model.llm.llm
        
    if hasattr(target_model, "merge_and_unload"):
        merged_model = target_model.merge_and_unload()
        merged_model.save_pretrained(f"./{NEW_MODEL_NAME}/lora_adapters")
    else:
        target_model.save_pretrained(f"./{NEW_MODEL_NAME}/lora_adapters")
        
    model.llm.tokenizer.save_pretrained(f"./{NEW_MODEL_NAME}/lora_adapters")
    print("Done.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main_train()