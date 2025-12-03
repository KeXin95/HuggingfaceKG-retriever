import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import argparse
from torch_geometric.data import Data
import joblib
import pandas as pd
import json
import numpy as np
from model_utils import GCN, GAT, GATv2, GraphTransformer, SAGE
from eval_helper import print_multilabel_classification_report
from utils import fix_graph_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "GATV2": GATv2,
    "TRANS": GraphTransformer,
    "SAGE": SAGE,
}

def load_graph(graph_path, model_type=None, device='cuda'):
    # Load graph (map to specified device to handle CPU/GPU compatibility)
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    graph = fix_graph_data(graph)
    graph = graph.to(device)
    return graph

def load_model(graph, config_dict, model_path, device):
    """
    Load model from config dictionary.
    
    Args:
        graph: Graph data object
        config_dict: Dictionary containing model configuration
        model_path: Path to the saved model state dict
        device: Device to load model on
    
    Config dict should have:
        - model_type: str (e.g., "GCN", "GAT", etc.)
        - model_config: dict with model hyperparameters
    """
    model_type = config_dict["model_type"].upper()
    
    # Get model class from registry
    if model_type.upper() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_REGISTRY.keys())}")
    
    ModelClass = MODEL_REGISTRY[model_type.upper()]
    model_config_dict = config_dict["model_config"]
    
    # Update dynamic dimensions from graph
    model_config_dict = model_config_dict.copy()
    model_config_dict["in_feats"] = graph.x.size(1)
    model_config_dict["out_feats"] = graph.y.size(1)
    
    
    # Create model instance
    model = ModelClass(**model_config_dict)
    model = model.to(device)
    print(model)
    
    # Load state dict (map to specified device to handle CPU/GPU compatibility)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False) # Should be true
    
    model.eval()
    
    return model

def inference_graph(graph, model, scaler, new_to_old_idx, model_type):
    new_features_scaled = scaler.transform(graph.x.cpu())
    new_features_tensor = torch.from_numpy(new_features_scaled).float().to(device)
    
    # Handle different model forward signatures
    model_type_upper = model_type.upper()
    # Models that require edge_attr: GATV2, TRANS
    if model_type_upper in ["GATV2", "GATV2CONV", "TRANS", "TRANSFORMER"]:
        # TODO: TRANS accept edge_attr as None
        logits = model(new_features_tensor, graph.edge_index, graph.edge_attr)
        
    else:
        # Models that only need x and edge_index (GCN, GAT, SAGE)
        logits = model(new_features_tensor, graph.edge_index)
    
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    preds_lst = preds.to('cpu').numpy().tolist()
    
    # Convert three mask columns into one column indicating "train", "val", or "test"
    train_mask = graph.train_mask.cpu().numpy()
    val_mask = graph.val_mask.cpu().numpy()
    test_mask = graph.test_mask.cpu().numpy()
    split_type = np.where(train_mask, 'train',
                          np.where(val_mask, 'val',
                          np.where(test_mask, 'test', 'Not in train/val/test')))

    nodes_id_df = pd.DataFrame(
        {
            'x': [row.tolist() for row in graph.x.cpu().numpy()],
            'node_id': [i for i in range(len(graph.x))],
            'ori_id': [new_to_old_idx[i] for i in range(len(graph.x))],
            'y': [graph.y[i].cpu().numpy() for i in range(len(graph.y))],
            'pred': preds_lst,
            'train_type': split_type
        })

    return nodes_id_df

def combine_nodes_id_df(nodes_id_df, nodes_df, idx_to_task):
    # Convert ori_id to match the index type (int64)
    nodes_id_df['ori_id'] = nodes_id_df['ori_id'].astype(int)
    nodes_df = nodes_df.reset_index(names='ori_id')
    merged_df = nodes_df.merge(nodes_id_df, 
                on='ori_id', how='left',
                suffixes=('', '_graph'))
    merged_df['y_multi_lab_text'] = merged_df['y_multi_lab'].apply(lambda x:[idx_to_task[x_] for x_ in x])
    merged_df['pred_label'] = merged_df['pred'].fillna('').apply(lambda x: np.where(x)[0] if x!='' else [])
    merged_df['pred_text'] = merged_df['pred_label'].apply(lambda x: [idx_to_task[x_] for x_ in x])
    
    merged_df['train_type'] = merged_df['train_type'].fillna('Not in graph')
    
    return merged_df

def load_utils(run_id):
    nodes_df = pd.read_parquet(f'../experiment_runs/{run_id}/nodes_df.parquet')
    # edges_df = pd.read_parquet(f'experiment_runs/{run_id}/edges_df.parquet')

    old_to_new_idx = json.load(open(f'../experiment_runs/{run_id}/old_to_new_idx.json','r'))
    new_to_old_idx = dict((v,k) for k,v in old_to_new_idx.items())
    
    task_to_idx = json.load(open(f'../experiment_runs/{run_id}/task_to_idx.json','r'))
    idx_to_task = dict((v,k) for k,v in task_to_idx.items())

    return nodes_df, new_to_old_idx, idx_to_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for graph neural network models")
    parser.add_argument('--run_id', type=str, required=True, help="Experiment run ID")
    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to JSON config file containing all model configurations")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['GCN', 'GAT', 'GATV2', 'TRANS', 'SAGE'],
                        help="Model type to use (must exist in config file)")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Output path for the parquet file")
    args = parser.parse_args()
    
    # Load configuration from JSON file
    with open(args.config_path, 'r') as f:
        all_configs = json.load(f)
    
    # Get the specific model configuration
    model_type = args.model_type.upper()
    if model_type not in all_configs:
        raise ValueError(f"Model type '{model_type}' not found in config file. Available: {list(all_configs.keys())}")
    
    config = all_configs[model_type]
    
    # Verify model_type matches
    if config.get("model_type", "").upper() != model_type:
        print(f"Warning: Config 'model_type' field ({config.get('model_type')}) doesn't match --model_type ({model_type}). Using --model_type.")
        config["model_type"] = model_type
    
    run_id = args.run_id
    scaler_filename = config.get("scaler_filename", "scaler.pkl")
    model_filename = config.get("model_filename", "trained_model.pt")
    
    graph_path = f'../experiment_runs/{run_id}/final_graph.pt'
    scaler_path = f'../experiment_runs/{run_id}/{scaler_filename}'
    model_path = f'../experiment_runs/{run_id}/{model_filename}'
    output_path = args.output_path

    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path, model_type=model_type, device=device)
    print(f"Loading utilities for run_id: {run_id}...")
    nodes_df, new_to_old_idx, idx_to_task = load_utils(run_id)
    
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    print(f"Loading model: {model_type} from {model_path}...")
    model = load_model(graph, config, model_path, device)
    
    print("Running inference...")
    nodes_id_df = inference_graph(graph, model, scaler, new_to_old_idx, model_type)
    
    merged_df = combine_nodes_id_df(nodes_id_df, nodes_df, idx_to_task)
    
    for train_type in ['train', 'val', 'test']:
        print(f"Evaluating {train_type} set...")
        eval_df = merged_df[merged_df['train_type'] == train_type]
        print_multilabel_classification_report(eval_df['y_multi_lab_text'], eval_df['pred_text'])
        print("\n\n")
        
    merged_df.to_parquet(output_path, index=False)
    print(f'✅ Saved to {output_path}')
    
# Example usage:
# python inference_graph_to_df.py \
#   --run_id=run_2025-10-11_13-12-14 \
#   --config_path=model_config_example.json \
#   --model_type=GCN \
#   --output_path=results_inferences/GCN_run_2025-10-11_13-12-14.parquet
#
# python inference_graph_to_df.py \
#   --run_id=run_2025-10-11_13-12-14 \
#   --config_path=model_config_example.json \
#   --model_type=GATV2 \
#   --output_path=results_inferences/GATV2_run_2025-10-11_13-12-14.parquet
# 
# python inference_graph_to_df.py \
#   --run_id=run_2025-10-11_13-12-14 \
#   --config_path=model_config_example.json \
#   --model_type=GAT \
#   --output_path=results_inferences/GAT_run_2025-10-11_13-12-14.parquet
# 
# python inference_graph_to_df.py \
#   --run_id=run_2025-10-11_13-12-14 \
#   --config_path=model_config_example.json \
#   --model_type=TRANS \
#   --output_path=results_inferences/TRANS_run_2025-10-11_13-12-14.parquet
# 
# python inference_graph_to_df.py \
#   --run_id=run_2025-10-11_13-12-14 \
#   --config_path=model_config_example.json \
#   --model_type=SAGE \
#   --output_path=results_inferences/SAGE_run_2025-10-11_13-12-14.parquet #
#
# The config file should contain all model configurations in a single JSON object:
# {
#   "GCN": { ... },
#   "GAT": { ... },
#   "GATV2": { ... },
#   "TRANS": { ... },
#   "SAGE": { ... }
# }


