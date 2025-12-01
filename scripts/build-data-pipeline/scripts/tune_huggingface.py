import torch
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, generate_random_graph
from cogdl.experiments import experiment
from cogdl.utils import BCEWithLogitsLoss, MultiLabelMicroF1
from sklearn.preprocessing import StandardScaler
import argparse
import os

class HuggingfaceNodeDataset(NodeDataset):
    def __init__(self, path):
        self.path = path
        super(HuggingfaceNodeDataset, self).__init__(path, scale_feat=True, metric="multilabel_f1")

# automl usage
def search_space(trial):
    return {
        "num_layers": trial.suggest_categorical("num_layers", [2]),
        "lr": trial.suggest_categorical("lr", [0.001, 1e-4, 1e-5]),  # config
        "hidden_size": trial.suggest_categorical("hidden_size", [256]),
        "epochs": trial.suggest_categorical("epochs", [500]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw"]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0]),
        "batch_size": trial.suggest_categorical("batch_size", [4096]),
        "patience": trial.suggest_categorical("patience", [100]),
    }


if __name__ == "__main__":
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Run GNN experiments on HuggingFace KG data.")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing the graph data file(s).")
                        
    parser.add_argument('--graph_files', nargs='+', default=['final_graph.pt'],
                        help="Name of the graph file(s) inside the data directory.")
                        
    parser.add_argument('--save_path', type=str, required=True,
                        help="Full path where the trained model will be saved (e.g., './trained_gcn.pt').")
                        
    parser.add_argument('--models', nargs='+', default=['gcn'],
                        help="List of GNN models to train (e.g., gcn gat graphsage).")
                        
    parser.add_argument('--devices', nargs='+', type=int, default=[0],
                        help="List of GPU device IDs to use.")
                        
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help="List of random seeds for multiple runs.")
                        
    args = parser.parse_args()

    # --- 2. Load Dataset(s) ---
    datasets = []
    print(f"Loading graph(s) from directory: {args.data_dir}")
    for file_name in args.graph_files:
        # Construct the full, safe path to the graph file
        full_path = os.path.join(args.data_dir, file_name)
        if not os.path.exists(full_path):
            print(f"Error: Graph file not found at {full_path}")
            exit()
        
        print(f"  - Loading {file_name}")
        dataset = HuggingfaceNodeDataset(path=full_path)
        datasets.append(dataset)
    
    # --- 3. Run Experiment ---
    print(f"\nStarting experiment with models: {args.models}")
    print(f"Models will be saved to: {args.save_path}")
    
    experiment(
        dataset=datasets,
        model=args.models,
        devices=args.devices,
        seed=args.seeds,
        search_space=search_space,
        saved_model_path=args.save_path
    )
