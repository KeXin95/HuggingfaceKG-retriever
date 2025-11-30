# scripts/utils.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv, GATv2Conv
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def fix_graph_data(graph):
    """Convert CogDL graph to PyTorch Geometric Data format"""
    
    # Extract features and labels
    x = graph.x  # Node features
    y = graph.y  # Node labels (multi-label)
    
    # Extract edge information and ensure it's in the correct format
    
    edge_index = torch.stack(list(graph.edge_index), dim=0)
    
    # Extract masks
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    
    print(f"Converted edge_index shape: {edge_index.shape}")
    print(f"Converted edge_index dtype: {edge_index.dtype}")
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data

def initialize_bias(model):
    print("\n--- Initializing Bias ---")
    try:
        # 1. Grab the final layer
        last_layer = model.conv2
        
        pi = 0.01
        bias_value = -np.log((1 - pi) / pi)
        
        # 2. Check for Standard Bias (GCN, SAGE, Linear)
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            torch.nn.init.constant_(last_layer.bias, bias_value)
            print(f"Initialized standard bias to {bias_value:.4f}")
            
        # 3. Check for Transformer Bias (Hidden in lin_skip)
        elif isinstance(last_layer, TransformerConv):
            if hasattr(last_layer, 'lin_skip') and last_layer.lin_skip is not None:
                torch.nn.init.constant_(last_layer.lin_skip.bias, bias_value)
                print(f"Initialized Transformer lin_skip bias to {bias_value:.4f}")
            else:
                print("WARNING: TransformerConv has no lin_skip (root_weight=False?). Cannot init bias.")
                
        # 4. Check for GAT Bias (Often None, but check just in case)
        elif isinstance(last_layer, GATConv) or isinstance(last_layer, GATConv2):
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                 torch.nn.init.constant_(last_layer.bias, bias_value)
                 print(f"Initialized GAT bias to {bias_value:.4f}")
            else:
                print("Note: GATConv usually has no bias by default. Skipping.")

        else:
            print(f"WARNING: Could not find a bias parameter in {type(last_layer).__name__}.")

    except AttributeError:
        print("WARNING: Model does not have attribute 'conv2'.")
    print("-------------------------\n")

def analyze_long_tail_performance(train_y, test_per_class_f1, save_path="f1_vs_support.png"):
    """
    Generates the 'Smoking Gun' visualization and Head/Tail stats.
    train_y: The labels from the TRAINING set (to determine frequency).
    test_per_class_f1: The F1 scores from the TEST set (to determine performance).
    """
    print("\n=== Long-Tail Distribution Analysis ===")
    
    # 1. Calculate Support (Frequency) per class in the Training Set
    # Assuming train_y is [num_nodes, num_classes]
    class_counts = train_y.sum(axis=0).cpu().numpy() # Shape: [num_classes]
    
    # 2. Sort classes by frequency (Most frequent first)
    sorted_indices = np.argsort(class_counts)[::-1]
    
    # 3. Define Head (Top 10) and Tail (The rest)
    head_indices = sorted_indices[:10]
    tail_indices = sorted_indices[10:]
    
    head_f1_avg = np.mean(test_per_class_f1[head_indices])
    tail_f1_avg = np.mean(test_per_class_f1[tail_indices])
    
    print(f"Head Labels (Top 10 freq): Avg F1 = {head_f1_avg:.4f}")
    print(f"Tail Labels (Bottom {len(tail_indices)}):   Avg F1 = {tail_f1_avg:.4f}")
    print(f"Gap (Head - Tail):         {head_f1_avg - tail_f1_avg:.4f}")
    
    # 4. The 'Smoking Gun' Plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(class_counts, test_per_class_f1, alpha=0.6, c='blue', edgecolors='w', s=80)
    
    # Log scale usually looks better for long-tail counts
    plt.xscale('log') 
    
    plt.title("The 'Smoking Gun': Test F1 Score vs. Training Class Frequency")
    plt.xlabel("Number of Training Samples (Log Scale)")
    plt.ylabel("Test F1 Score")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=head_f1_avg, color='g', linestyle='--', label=f'Head Avg F1: {head_f1_avg:.2f}')
    plt.axhline(y=tail_f1_avg, color='r', linestyle='--', label=f'Tail Avg F1: {tail_f1_avg:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    print("========================================\n")

def print_df_stats(df):
    print(f"""Total nodes: {df.shape[0]}
            train: {df.train_mask.sum()}
            val:  {df.val_mask.sum()}
            test: {df.test_mask.sum()}
    """)

def print_stats(graph_data):
    print(f"""Total nodes: {graph_data.num_nodes}
            train: {graph_data.train_mask.sum()}
            val:  {graph_data.val_mask.sum()}
            test: {graph_data.test_mask.sum()}
    """)

def convert2group(cur_rel, src_col, dst_col):
    """
    Groups a list of model relationships into a dictionary without using explicit loops.

    Args:
        cur_rel: A list of dictionaries.
        src_col: The column name for the source of the relationship (e.g., 'base_model_id').
        dst_col: The column name for the destination of the relationship (e.g., 'model_id').

    Returns:
        A dictionary mapping each source ID to a list of its associated destination IDs.
    """
    if not cur_rel:
        return {}
        
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(cur_rel)
    
    return df.groupby(src_col)[dst_col].apply(list).to_dict()

def encode_onehot(temp_df):
    """
    Efficiently creates a one-hot encoded matrix for multi-label data.
    """
    # Ensure every entry in the column is a list, converting None/NaN to []
    cleaned_labels = temp_df['y_multi_lab'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create coordinate lists for the sparse matrix
    rows = []
    cols = []
    for i, sublist in enumerate(cleaned_labels):
        for item in sublist:
            # Check if the item is not None/NaN and is a valid integer index
            if pd.notna(item) and isinstance(item, (int, float)) and not isinstance(item, bool):
                try:
                    cols.append(int(item))
                    rows.append(i)
                except (ValueError, TypeError):
                    # Skip items that cannot be converted to an int
                    pass
    
    # Determine the size of the one-hot vector
    num_classes = max(cols) + 1 if cols else 0
    
    # --- Vectorized One-Hot Matrix Creation ---
    num_rows = len(temp_df)
    one_hot_matrix = np.zeros((num_rows, num_classes), dtype=int)
    
    # Use NumPy's fast indexing if there are labels to encode
    if rows:
        one_hot_matrix[rows, cols] = 1
    
    # Return the result as a list of numpy arrays
    return list(one_hot_matrix)
    
