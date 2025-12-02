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
from torch.cuda.amp import autocast, GradScaler

from model_utils import GraphEncoderGAT, GraphEncoderGATConv2, GraphEncoderSAGE
from torch_geometric.llm.models import LLM, GRetriever
from typing import List, Optional
import glob
import shutil
import joblib
from sklearn.preprocessing import StandardScaler

from huggingface_hub import login
import os

# ==========================================
# 2. Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BASE_EXP_DIR = './experiment_runs/run_2025-10-11_13-12-14/'
GRAPH_FILE = os.path.join(BASE_EXP_DIR, 'final_graph.pt')
NODES_DF_PATH = os.path.join(BASE_EXP_DIR, 'nodes_df.parquet')
BRIDGE_FILE = os.path.join(BASE_EXP_DIR, 'old_to_new_idx.json')
SCALER_FILE = os.path.join(BASE_EXP_DIR, 'scaler.pkl')
MODEL_TYPE = 'GATV2'#'GATV2' #Or 'GAT'
VERSION = 1
NEW_MODEL_NAME = f"g_retriever_multilabel_{MODEL_TYPE}_v{VERSION}"
IS_SCALE = False

BGE_EMBED_DIM = 768 
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 256
GNN_EDGE_DIM = None 

model_dict = {
    'GAT': GraphEncoderGAT,
    'GATV2': GraphEncoderGATConv2,
    'SAGE': GraphEncoderSAGE
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

TASK_LIST_SORTED = sorted(TASK_LIST)  # ['any-to-any', 'audio-classification', ...]
TASK_TO_IDX = {task: idx for idx, task in enumerate(TASK_LIST)}  # Original index mapping

TASK_LIST_PROMPT = f"""Predict the task categories for this model.
Available tasks: {TASK_LIST_SORTED}

Output ONLY the relevant task names in alphabetical order, separated by commas."""

# ==========================================
# 3. Data Loading (THE FIX)
# ==========================================
def load_full_data():
    print("Loading all data sources...")
    print(f"Loading graph from: {GRAPH_FILE}")
    
    raw_graph = torch.load(GRAPH_FILE, weights_only=False)
    
    print("Converting raw graph to clean PyG Data object...")

    if IS_SCALE:
        scaler = StandardScaler()
        scaler.fit(raw_graph.x[raw_graph.train_mask])
        joblib.dump(scaler, SCALER_FILE)
        x = torch.from_numpy(scaler.transform(raw_graph.x)).float()
    else:
        x = raw_graph.x.float()    
    
    if isinstance(raw_graph.edge_index, tuple):
        src, dst = raw_graph.edge_index
        if isinstance(src, np.ndarray): src = torch.from_numpy(src)
        if isinstance(dst, np.ndarray): dst = torch.from_numpy(dst)
        src = src.long() if isinstance(src, torch.Tensor) else torch.tensor(src).long()
        dst = dst.long() if isinstance(dst, torch.Tensor) else torch.tensor(dst).long()
        edge_index = torch.stack([src, dst], dim=0)
    elif isinstance(raw_graph.edge_index, torch.Tensor):
        edge_index = raw_graph.edge_index.long()
    else:
        raise TypeError(f"Unknown edge_index type: {type(raw_graph.edge_index)}")

    # Attach masks to the graph object so we can access them
    graph = Data(
        x=x, 
        edge_index=edge_index,
        train_mask=raw_graph.train_mask,
        test_mask=raw_graph.val_mask
    )
    graph.num_nodes = raw_graph.num_nodes
    del raw_graph
    print("  Clean PyG graph created.")

    print(f"Loading nodes_df from: {NODES_DF_PATH}")
    nodes_df = pd.read_parquet(NODES_DF_PATH)
    
    print(f"Loading bridge file from: {BRIDGE_FILE}")
    with open(BRIDGE_FILE, 'r') as f:
        old_to_new_idx_dict = json.load(f)
    
    new_to_old_idx = {value: int(key) for key, value in old_to_new_idx_dict.items()}
    
    train_list = []
    test_list = []
    
    print("Filtering data using Graph Masks (Allowing ALL node types)...")
    
    for new_idx, old_idx in new_to_old_idx.items():
        # if new_idx >= graph.num_nodes: continue
        
        node_data = nodes_df.iloc[old_idx]
        labels = node_data['y']
        
        # --- THE FIX IS HERE ---
        # We removed: node_data['type'] == 'model'
        # We keep: len(labels) > 0 (We still need ground truth to train/eval)
        if len(labels) > 0:
            
            data_item = {
                "name": node_data['id'],
                "label": labels,
                "graph_id": new_idx
            }

            # Check the MASKS from the graph file
            if graph.train_mask[new_idx].item():
                train_list.append(data_item)
            elif graph.test_mask[new_idx].item():
                test_list.append(data_item)
            
    print(f"  Train Samples: {len(train_list)}")
    print(f"  Test Samples:  {len(test_list)}")

    train_dataset = Dataset.from_list(train_list)
    eval_dataset = Dataset.from_list(test_list)
    
    print("Graph loaded and kept on CPU.")
    return graph, train_dataset, eval_dataset

# ==========================================
# 4. Collate Function
# ==========================================
def create_collate_fn(graph):
    _graph = graph
   
    def collate_fn(batch: List[dict]):
        questions = []
        labels = []
        data_list = []
       
        for item in batch:
            # Convert multi-hot [0,1,1,0,...] to task names
            label_indices = [i for i, v in enumerate(item['label']) if v == 1]
            task_names = [TASK_LIST[i] for i in label_indices]
            
            # Sort alphabetically for consistent ordering
            task_names_sorted = sorted(task_names)
            label_string = ", ".join(task_names_sorted)
            
            prompt = f"{TASK_LIST_PROMPT}\n\nModel: {item['name']}"
           
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            questions.append(formatted_prompt)
            labels.append(label_string)
            
            node_subset, edge_index_sub, _, _ = k_hop_subgraph(
                item['graph_id'], num_hops=1, edge_index=_graph.edge_index,
                relabel_nodes=True, num_nodes=_graph.num_nodes
            )
            data_list.append(Data(x=_graph.x[node_subset], edge_index=edge_index_sub))
           
        graph_batch = Batch.from_data_list(data_list)
        return questions, labels, graph_batch
       
    return collate_fn

# ==========================================
# 5. Training Loop & Helpers
# ==========================================

def save_checkpoint(model, step: int, save_dir: str, is_best: bool = False, keep_last_n: int = 2):
    """Save model checkpoint, keeping only the last N checkpoints."""
    if not is_best:
        return
        
    ckpt_dir = os.path.join(save_dir, "best")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}...")
    
    torch.save(model.gnn.state_dict(), f"{ckpt_dir}/gnn.pt")
    torch.save(model.projector.state_dict(), f"{ckpt_dir}/projector.pt")
    
    if hasattr(model, 'llm_generator'):
        model.llm_generator.save_pretrained(f"{ckpt_dir}/lora_adapters")
    else:
        model.llm.llm.save_pretrained(f"{ckpt_dir}/lora_adapters")
    model.llm.tokenizer.save_pretrained(f"./{ckpt_dir}/lora_adapters")
    


def evaluate(model, eval_loader, device, use_amp: bool = True):
    """
    Evaluate model on validation/test set.
    
    Returns:
        Average loss over the evaluation set
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for questions, labels, graph_batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            if use_amp:
                with autocast():
                    loss = model(
                        question=questions,
                        x=graph_batch.x.to(device),
                        edge_index=graph_batch.edge_index.to(device),
                        batch=graph_batch.batch.to(device),
                        label=labels
                    )
            else:
                loss = model(
                    question=questions,
                    x=graph_batch.x.to(device),
                    edge_index=graph_batch.edge_index.to(device),
                    batch=graph_batch.batch.to(device),
                    label=labels
                )
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)

def main_train():
    # Load data using the NEW mask-based function
    graph, train_dataset, eval_dataset = load_full_data()
    
    global BGE_EMBED_DIM
    BGE_EMBED_DIM = graph.x.shape[1]
    print(f"Detected BGE_EMBED_DIM: {BGE_EMBED_DIM}")
    
    print(f"Loading base model: {MODEL_NAME}...")
    llm = LLM(model_name=MODEL_NAME, n_gpus=7)
    
    print("Initializing GNN...")
    gnn = model_dict[MODEL_TYPE](
        in_channels=BGE_EMBED_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
        edge_dim=GNN_EDGE_DIM
    )

    model = GRetriever(llm=llm, gnn=gnn, use_lora=True, mlp_out_tokens=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    collate_fn = create_collate_fn(graph)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                            collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)

    best_eval_loss = float('inf')  # Initialize to infinity
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(3):
        total_loss = 0
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
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            if (step + 1) % 5000 == 0:
                # Evaluate to check if this is the best checkpoint
                model.eval()
                eval_loss = evaluate(model, eval_loader, device)
                model.train()
                
                # Check if this is the best so far
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                    print(f"🎉 New best model! Eval loss: {eval_loss:.4f}")
                
                save_checkpoint(model, step + 1, NEW_MODEL_NAME, is_best=is_best)

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
        print("Merged model saved.")
    else:
        print("WARNING: Could not find merge_and_unload. Saving adapters only.")
        target_model.save_pretrained(f"./{NEW_MODEL_NAME}/lora_adapters")
        
    model.llm.tokenizer.save_pretrained(f"./{NEW_MODEL_NAME}/lora_adapters")
    print("Done.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main_train()

#  nohup python gretrieval.py >> gretrieval.log 2>&1&
