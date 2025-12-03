import os
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import re
import argparse
import joblib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from huggingface_hub import snapshot_download

from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import GAT
from torch_geometric.llm.models import LLM, GRetriever
from torch.utils.data import DataLoader
from model_utils import GraphEncoderGAT, GraphEncoderGATConv2, GraphEncoderSAGE

from sklearn.metrics import f1_score, average_precision_score

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BASE_EXP_DIR = '../experiment_runs/run_2025-10-11_13-12-14/'
GRAPH_FILE = os.path.join(BASE_EXP_DIR, 'final_graph.pt')
NODES_DF_PATH = os.path.join(BASE_EXP_DIR, 'nodes_df.parquet')
BRIDGE_FILE = os.path.join(BASE_EXP_DIR, 'old_to_new_idx.json')
SCALER_FILE = os.path.join(BASE_EXP_DIR, 'scaler.pkl')

BGE_EMBED_DIM = 768+54 #BGE with BM25 
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 256
GNN_EDGE_DIM = None 

model_dict = {
    'GAT': GraphEncoderGAT,
    'GATV2': GraphEncoderGATConv2,
    'SAGE': GraphEncoderSAGE
}

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
NUM_CLASSES = len(TASK_LIST)
# TASK_LIST_PROMPT = f"""Here is a list of possible tasks: {TASK_LIST}.
# Please predict all relevant tasks for this model. Output only the indices, separated by commas."""

TASK_LIST_SORTED = sorted(TASK_LIST)  # ['any-to-any', 'audio-classification', ...]
TASK_TO_IDX = {task: idx for idx, task in enumerate(TASK_LIST)}  # Original index mapping

TASK_LIST_PROMPT = f"""Predict the task categories for this model.
Available tasks: {TASK_LIST_SORTED}

Output ONLY the relevant task names in alphabetical order, separated by commas."""

def load_data():
    print("Loading graph and data...")
    raw_graph = torch.load(GRAPH_FILE, weights_only=False)
    if os.path.exists(SCALER_FILE):
        print('scaling input...')
        scaler = joblib.load(SCALER_FILE)
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
    else:
        edge_index = raw_graph.edge_index.long()
        
    graph = Data(
        x=x, 
        edge_index=edge_index,
        train_mask=raw_graph.train_mask,
        val_mask=getattr(raw_graph, 'val_mask', None),
        test_mask=raw_graph.test_mask
    )
    graph.num_nodes = raw_graph.num_nodes
    del raw_graph

    nodes_df = pd.read_parquet(NODES_DF_PATH)
    with open(BRIDGE_FILE, 'r') as f:
        old_to_new_idx_dict = json.load(f)
    new_to_old_idx = {value: int(key) for key, value in old_to_new_idx_dict.items()}
    
    train_list = []
    val_list = []
    test_list = []

    print("Filtering data using Graph Masks (Allowing ALL node types)...")
    for new_idx, old_idx in new_to_old_idx.items():
        
        node_data = nodes_df.iloc[old_idx]
        labels = node_data['y']
        
        if len(labels) > 0:
            data_item = {
                "name": node_data['id'],
                "label": labels,
                "graph_id": new_idx
            }
            
            # Bucket into splits
            if graph.train_mask[new_idx].item():
                train_list.append(data_item)
            elif graph.test_mask[new_idx].item():
                test_list.append(data_item)
            elif graph.val_mask is not None and graph.val_mask[new_idx].item():
                val_list.append(data_item)
            
    print(f"  Train Samples (for Stats): {len(train_list)}")
    print(f"  Val Samples:               {len(val_list)}")
    print(f"  Test Samples (for Eval):   {len(test_list)}")

    train_dataset = Dataset.from_list(train_list)
    val_dataset = Dataset.from_list(val_list) if val_list else None
    test_dataset = Dataset.from_list(test_list)
    
    return graph, train_dataset, val_dataset, test_dataset

def create_collate_fn(graph):
    _graph = graph
    def collate_fn(batch):
        questions = []
        true_labels = []
        data_list = []
        
        for item in batch:
            prompt = f"{TASK_LIST_PROMPT}\n\nFor this node, here's the info:\n{item['name']}"
            questions.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            #questions.append(f"[INST] {prompt}")
            true_labels.append(item['label']) 
            
            node_subset, edge_index_sub, _, _ = k_hop_subgraph(
                item['graph_id'], num_hops=1, edge_index=_graph.edge_index, 
                relabel_nodes=True, num_nodes=_graph.num_nodes
            )
            data_list.append(Data(x=_graph.x[node_subset], edge_index=edge_index_sub))
            
        graph_batch = Batch.from_data_list(data_list)
        return questions, true_labels, graph_batch
    return collate_fn

def load_checkpoint(model_type, base_dir, checkpoint_name, input_dim):
    print(f"\n--- Loading Checkpoint: {checkpoint_name} ---")
    
    if checkpoint_name == "final":
        ckpt_path = base_dir
        lora_path = os.path.join(base_dir, "lora_adapters")
    else:
        ckpt_path = os.path.join(base_dir, checkpoint_name)
        lora_path = os.path.join(ckpt_path, "lora_adapters")

    gnn_path = os.path.join(ckpt_path, "gnn.pt")
    proj_path = os.path.join(ckpt_path, "projector.pt")

    print(f"Loading LLM/LoRA from: {lora_path}")
    llm = LLM(model_name=lora_path, n_gpus=1)

    print(f"Loading GNN from: {gnn_path}")
    gnn = model_dict[model_type](
        in_channels=BGE_EMBED_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
        edge_dim=GNN_EDGE_DIM
    )
    gnn.load_state_dict(torch.load(gnn_path))

    model = GRetriever(llm, gnn, use_lora=False) 
    
    print(f"Loading Projector from: {proj_path}")
    model.projector.load_state_dict(torch.load(proj_path))
    
    model.eval()
    return model

def parse_prediction(text):
    # cleaned = re.sub(r'\[/?s\]|</s>', '', text).strip()
    # task_name_ls = [t.strip().lower() for t in cleaned.split(',')]
    # return task_name_ls

    pattern = '|'.join(re.escape(t) for t in TASK_LIST)  # escape special chars
    
    # Find all matches (case insensitive)
    matches = re.findall(pattern, text.lower(), re.IGNORECASE)
    
    return matches

def to_multi_hot(task_lists, num_classes):
    batch_size = len(task_lists)
    multi_hot = torch.zeros((batch_size, num_classes))
    for i, task_list in enumerate(task_lists):
        for task_name in task_list:
            if task_name in TASK_LIST:
                multi_hot[i, TASK_LIST.index(task_name)] = 1.0
    return multi_hot

def analyze_long_tail_performance(train_counts, test_y_true, test_y_pred, save_name):
    print("\n=== Long-Tail Distribution Analysis ===")
    
    per_class_f1 = f1_score(test_y_true, test_y_pred, average=None, zero_division=0)
    
    class_counts = np.array(train_counts)
    
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_counts = class_counts[sorted_indices]
    sorted_f1 = per_class_f1[sorted_indices]
    
    head_indices = sorted_indices[:10]
    tail_indices = sorted_indices[10:]
    
    head_f1_avg = np.mean(per_class_f1[head_indices])
    tail_f1_avg = np.mean(per_class_f1[tail_indices])
    
    print(f"Head Labels (Top 10 freq): Avg F1 = {head_f1_avg:.4f}")
    print(f"Tail Labels (Rest):        Avg F1 = {tail_f1_avg:.4f}")
    print(f"Gap (Head - Tail):         {head_f1_avg - tail_f1_avg:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(class_counts, per_class_f1, alpha=0.6, c='blue', edgecolors='w', s=80)
    plt.xscale('log')
    plt.title(f"The 'Smoking Gun': Test F1 vs. Training Frequency\n({save_name})")
    plt.xlabel("Number of Training Samples (Log Scale)")
    plt.ylabel("Test F1 Score")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=head_f1_avg, color='g', linestyle='--', label=f'Head: {head_f1_avg:.2f}')
    plt.axhline(y=tail_f1_avg, color='r', linestyle='--', label=f'Tail: {tail_f1_avg:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_name}_smoking_gun.png")
    print(f"Visualization saved to {save_name}_smoking_gun.png")

def plot_class_breakdown(df, save_name):
    top_10 = df.head(10)
    bottom_10 = df.tail(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].barh(top_10['Task_Name'], top_10['F1_Score'], color='forestgreen')
    axes[0].set_title('Top 10 Performing Tasks')
    axes[0].set_xlabel('F1 Score')
    axes[0].set_xlim(0, 1.05)
    axes[0].invert_yaxis() 
    
    axes[1].barh(bottom_10['Task_Name'], bottom_10['F1_Score'], color='salmon')
    axes[1].set_title('Bottom 10 Performing Tasks')
    axes[1].set_xlabel('F1 Score')
    axes[1].set_xlim(0, 1.05)
    axes[1].invert_yaxis()
    
    plt.suptitle(f"Per-Class Performance Breakdown ({save_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_name}_top_bottom_bars.png")
    print(f"Bar charts saved to {save_name}_top_bottom_bars.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='final', 
                        help="Folder name of checkpoint (e.g. 'checkpoint-10000') or 'final'")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help="Which split to evaluate on (val or test)")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help=(
            "Path to checkpoint directory. Can be:\n"
            "  - Local directory path (e.g., '/path/to/gretriever-HFKG_GATV2' or './g_retriever_multilabel_GATV2_v1')\n"
            "  - Hugging Face repo ID (e.g., 'Chloe/gretriever-HFKG_GATV2'). "
            "If using HF repo, also provide --hf_dir."
        ),
    )
    parser.add_argument(
        '--hf_dir',
        type=str,
        default=None,
        help=(
            "Local directory to download HF repo into. "
            "Required when --checkpoint_path is a Hugging Face repo ID."
        ),
    )
    parser.add_argument('--model', type=str, default='GAT', choices=['GAT', 'GATV2', 'SAGE'],
                        help="Specify the model type (used for GNN architecture selection).")
    args = parser.parse_args()

    graph, train_dataset, val_dataset, test_dataset = load_data()

    model_type = args.model
    checkpoint_path = args.checkpoint_path

    # Determine if checkpoint_path is a local directory or an HF repo ID
    # Check if it's a local path that exists, or if it looks like an HF repo (contains '/' and doesn't exist as path)
    is_local_path = os.path.exists(checkpoint_path) or os.path.isabs(checkpoint_path) or checkpoint_path.startswith('./') or checkpoint_path.startswith('../')
    is_hf_repo = '/' in checkpoint_path and not is_local_path

    if is_hf_repo:
        # Download from Hugging Face Hub
        if "HF_TOKEN" not in os.environ:
            raise EnvironmentError(
                "HF_TOKEN environment variable is not set. "
                "Please export your Hugging Face token, e.g. `export HF_TOKEN=xxx`, "
                "before using a Hugging Face repo."
            )
        
        if args.hf_dir is None or args.hf_dir.strip() == "":
            raise ValueError(
                "--hf_dir is required when --checkpoint_path is a Hugging Face repo ID. "
                "Please provide a local directory path, e.g., --hf_dir /path/to/download"
            )

        repo_id = checkpoint_path
        print(f"\n--- Downloading checkpoint from HF Hub: {repo_id} ---")
        print(f"Downloading to: {args.hf_dir}")
        
        download_kwargs = {
            "repo_id": repo_id,
            "local_dir": args.hf_dir,
            "local_dir_use_symlinks": False
        }
        
        snapshot_root = snapshot_download(**download_kwargs)
        new_model_name = snapshot_root

        print(f"Downloaded to: {snapshot_root}")
        print(f"Using checkpoint directory: {new_model_name}")
    else:
        # Use local directory path
        new_model_name = checkpoint_path
        if not os.path.exists(new_model_name):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {new_model_name}. "
                "Please provide a valid local path or Hugging Face repo ID."
            )
        print(f"Using local checkpoint directory: {new_model_name}")
    
    if args.split == 'val':
        target_dataset = val_dataset
        print("\n[INFO] Evaluating on VALIDATION set.")
    else:
        target_dataset = test_dataset
        print("\n[INFO] Evaluating on TEST set.")

    if target_dataset is None or len(target_dataset) == 0:
        print(f"[ERROR] The {args.split} dataset is empty! Check your masks.")
        return

    print("Calculating training class frequencies...")
    train_counts = np.zeros(NUM_CLASSES)
    for item in tqdm(train_dataset, desc="Scanning Train Data"):
        for label in item['label']:
            train_counts[label] += 1
            
    model = load_checkpoint(model_type, new_model_name, args.ckpt, graph.x.shape[1])
    
    print(f"\nRunning Inference on {args.split.upper()} set using {args.ckpt}...")
    collate_fn = create_collate_fn(graph)
    loader = DataLoader(target_dataset, batch_size=16, collate_fn=collate_fn, num_workers=0)
    
    all_preds_labels = []
    all_true_labels = []
    
    with torch.no_grad():
        for questions, labels, graph_batch in tqdm(loader, desc="Inference"):
            preds_text = model.inference(
                question=questions,
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
                # max_out_tokens=132
            )
            
            for pred_text, true_label in zip(preds_text, labels):
                pred_names = parse_prediction(pred_text)
                all_preds_labels.append(pred_names)
                all_true_labels.append(true_label)
                

    print("\nConverting Generative Output to Multi-Hot Vectors...")
    y_pred = to_multi_hot(all_preds_labels, NUM_CLASSES).numpy()
    y_true = all_true_labels

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "="*40)
    print(f"RESULTS FOR: {args.ckpt} | Split: {args.split.upper()}")
    print("-" * 40)
    print(f"Micro F1:   {micro_f1:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print("="*40)

    analyze_long_tail_performance(train_counts, y_true, y_pred, save_name=f"{args.ckpt}_{args.split}")

    print("\nGenerating Per-Class Breakdown...")
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    df_results = pd.DataFrame({
        "Task_Name": TASK_LIST,
        "F1_Score": per_class_f1,
        "Training_Samples": train_counts
    })
    
    df_results = df_results.sort_values(by="F1_Score", ascending=False)
    
    csv_name = f"{args.ckpt}_{args.split}_class_performance.csv"
    df_results.to_csv(csv_name, index=False)
    print(f"Detailed CSV saved to {csv_name}")
    
    plot_class_breakdown(df_results, save_name=f"{args.ckpt}_{args.split}")

if __name__ == "__main__":
    main()
    
# export HF_TOKEN=<your_token>
# For Hugging Face repo:
# python g_retrieval_eval_w_labels_qwen.py \
#   --checkpoint_path Chloe/gretriever-HFKG_GATV2 \
#   --hf_dir /path/to/download \
#   --model GATV2 \
#   --ckpt final \
#   --split test

# For local directory:
# python g_retrieval_eval_w_labels_qwen.py \
#   --checkpoint_path ./gretriever-HFKG_GATV2 \
#   --model GATV2 \
#   --ckpt final \
#   --split test
