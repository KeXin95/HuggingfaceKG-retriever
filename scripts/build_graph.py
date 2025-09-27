import torch
import pandas as pd
import json
import argparse
from cogdl.data import Graph
import os
import numpy as np
import pytz
from configs import main_config as config
from .utils import print_stats


def prepocess_nodes_df(nodes_df_, JSON_FOLDER, timefilter=True):
    
    # # Create a mapping from original ID to a new integer index (0, 1, 2, ...)
    # nodes_df_['node_idx'] = range(len(nodes_df_))
    # id_to_idx = pd.Series(nodes_df_.node_idx.values, index=nodes_df_.id).to_dict()
    if timefilter:
        # Define date boundaries as pandas Timestamps with UTC timezone
        val_start_model = pd.Timestamp('2024-9-15 00:00:00', tz=pytz.UTC)
        val_end_model = pd.Timestamp('2024-10-15 00:00:00', tz=pytz.UTC)
        test_start_model = pd.Timestamp('2024-10-15 00:00:00', tz=pytz.UTC)
        test_end_model = pd.Timestamp('2024-12-15 00:00:00', tz=pytz.UTC)
        
        val_start_ds = pd.Timestamp('2024-04-15 00:00:00', tz=pytz.UTC)
        val_end_ds = pd.Timestamp('2024-08-15 00:00:00', tz=pytz.UTC)
        test_start_ds = pd.Timestamp('2024-08-15 00:00:00', tz=pytz.UTC)
        test_end_ds = pd.Timestamp('2024-12-15 00:00:00', tz=pytz.UTC)
        
        # Parse the createdAt_str column to datetime
        nodes_df_['createdAt'] = pd.to_datetime(nodes_df_['createdAt'], utc=True)
        
        # Vectorized filtering
        mask_model = nodes_df_['type'] == 'model'
        mask_dataset = nodes_df_['type'] == 'dataset'
        
        train_mask = ((mask_model & (nodes_df_['createdAt'] < val_start_model)) |
                      (mask_dataset & (nodes_df_['createdAt'] < val_start_ds)))
        val_mask = ((mask_model & (val_start_model <= nodes_df_['createdAt']) & (nodes_df_['createdAt'] < test_start_model)) |
                    (mask_dataset & (val_start_ds <= nodes_df_['createdAt']) & (nodes_df_['createdAt'] < test_start_ds)))
        test_mask = ((mask_model & (test_start_model <= nodes_df_['createdAt']) & (nodes_df_['createdAt'] <= test_end_model)) |
                     (mask_dataset & (test_start_ds <= nodes_df_['createdAt']) & (nodes_df_['createdAt'] <= test_end_ds)))
    else:
        # --- Create random splits: 70% train, 15% validation, 15% test ---
        
        # Set a seed for reproducible random splits
        np.random.seed(42)
        
        num_nodes = len(nodes_df_)
        
        # Create a shuffled array of indices from 0 to num_nodes - 1
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        # Calculate the exact index where each split ends
        train_end = int(0.7 * num_nodes)
        val_end = int(0.85 * num_nodes)  # 70% + 15%
        
        # Assign indices to each split
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create boolean masks of the same size as the number of nodes
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        # Set the corresponding indices to True for each mask
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
    # Add masks to the DataFrame
    nodes_df_['train_mask'] = train_mask
    nodes_df_['val_mask'] = val_mask
    nodes_df_['test_mask'] = test_mask

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    
    # Print the number of nodes in each set
    print(f"Training set size: {train_mask.sum()}")
    print(f"Validation set size: {val_mask.sum()}")
    print(f"Test set size: {test_mask.sum()}")
    print(f"Total nodes: {len(nodes_df_)}")
    print(f"Unassigned nodes: {(~(train_mask | val_mask | test_mask)).sum()}")
    return nodes_df_, train_mask, val_mask, test_mask

# The function now correctly accepts all the tensors it needs
def construct_graph(nodes_df_, GRAPH_OUTPUT_FILE,
                    train_mask, val_mask, test_mask,
                    x, y, edge_index, edge_attr, drop_dormant_nodes=False
                   ):
    
    # --- Corrected Filtering and Re-indexing Section ---
    print("\nStep 6: Filtering isolated nodes and re-indexing graph...")
    
    # Find nodes that have at least one label
    labeled_nodes = y.sum(dim=1) > 0
    
    # Find nodes that participate in at least one edge
    num_nodes = len(nodes_df_)
    connected_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    # Ensure edge_index is not empty before flattening
    if edge_index.numel() > 0:
        unique_nodes_in_edges = torch.unique(edge_index.flatten())
        connected_nodes[unique_nodes_in_edges] = True
    
    if drop_dormant_nodes:
        final_node_mask = labeled_nodes & connected_nodes
    else:
        # The final set of nodes we want to keep are those that are either labeled OR connected
        final_node_mask = labeled_nodes | connected_nodes
    final_indices = final_node_mask.nonzero(as_tuple=True)[0]
    
    print(f"Initial nodes: {num_nodes}. Final nodes after removing isolates: {len(final_indices)}")
    
    # Filter the features, labels, and masks
    numpy_mask = final_node_mask.cpu().numpy()
    final_x = x[final_node_mask]
    final_y = y[final_node_mask]
    # Note: train_mask is already a tensor from the preprocessing step
    final_train_mask = train_mask[final_node_mask]
    final_val_mask = val_mask[final_node_mask]
    final_test_mask = test_mask[final_node_mask]

    ###############################################################temp only
    print(f"current nodes count={len(nodes_df_)}")
    print(f"Keeping nodes that are in *defined_for.json only (following paper)")
    model_task_df = pd.read_json(os.path.join(config.JSON_PATH, 'model_definedFor_task.json'))
    dataset_task_df = pd.read_json(os.path.join(config.JSON_PATH, 'dataset_definedFor_task.json'))
    model_nodes = pd.unique(model_task_df['model_id'])
    dataset_nodes = pd.unique(dataset_task_df['dataset_id'])
    
    nodes_df_ = nodes_df_[nodes_df_['id'].isin(np.concatenate([model_nodes,dataset_nodes]))]
    print(f"After cleaning, nodes count={len(nodes_df_)}")
    
    # Preparing idx to nodes
    nodes_df_ = nodes_df_[~nodes_df_['description'].isna()]
    print(f"Removing nodes where description is empty, nodes count={len(nodes_df_)}")
    
    nodes_df_ = nodes_df_.reset_index(drop=True)
    #####################################################################temp only
    
    # --- Re-indexing Edges ---
    
    # 1. Create a boolean mask for the EDGES.
    source_nodes = edge_index[0]
    dest_nodes = edge_index[1]
    edge_mask = final_node_mask[source_nodes] & final_node_mask[dest_nodes]
    
    # 2. Filter both the edge_index and edge_attr using this mask.
    filtered_edge_index = edge_index[:, edge_mask]
    filtered_edge_attr = edge_attr[edge_mask]
    
    # 3. Create a fast lookup tensor for mapping old indices to new ones.
    lookup = torch.full((num_nodes,), -1, dtype=torch.long)
    lookup[final_indices] = torch.arange(len(final_indices))
    
    # 4. Apply the mapping to the filtered edges in a single, vectorized operation.
    final_edge_index = lookup[filtered_edge_index]
    final_edge_attr = filtered_edge_attr
    
    # (Optional) Create the dictionary if you still need to return it
    old_to_new_idx = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(final_indices)}

    # --- Verification ---
    print(f"Original edges: {edge_index.shape[1]}")
    print(f"Final edges after re-indexing: {final_edge_index.shape[1]}")
    print(f"Final edge attributes: {final_edge_attr.shape[0]}")
    
    # --- Create and save the final graph object ---
    graph_data = Graph(
        x=final_x,
        y=final_y,
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
        train_mask=final_train_mask,
        val_mask=final_val_mask,
        test_mask=final_test_mask
    )
    
    torch.save(graph_data, GRAPH_OUTPUT_FILE)
    print("\nDone!")
    print(f"Final graph object saved to '{GRAPH_OUTPUT_FILE}'.")
    print(graph_data)
    print_stats(graph_data)
    return graph_data, old_to_new_idx


def main():
    # --- Update argparse to accept run_dir and strategy ---
    parser = argparse.ArgumentParser(description="Build the final graph object from pre-processed data.")
    parser.add_argument('--run_dir', type=str, required=True, help="The directory for this specific run's inputs and outputs.")
    parser.add_argument('--split_strategy', type=str, default='time', choices=['time', 'random'], help="The splitting strategy to use.")
    args = parser.parse_args()
    # ----------------------------------------------------

    print(f"\n--- STAGE 3: Building Final Graph (Split Strategy: {args.split_strategy}) ---")

    # --- Construct all necessary paths ---
    nodes_df_path = os.path.join(args.run_dir, config.NODES_DF_FILENAME)
    edges_df_path = os.path.join(args.run_dir, config.EDGES_DF_FILENAME)
    features_path = os.path.join(args.run_dir, config.FEATURES_FILENAME)
    graph_output_path = os.path.join(args.run_dir, config.GRAPH_OUTPUT_FILENAME)
    index_map_path = os.path.join(args.run_dir, config.INDEX_MAP_FILENAME)
    # -------------------------------------
    
    # Load all inputs for this stage
    print("Loading pre-processed data...")
    nodes_df = pd.read_pickle(nodes_df_path)
    edges_df = pd.read_pickle(edges_df_path)
    x = torch.load(features_path)

    # 1. Create splits
    use_time_filter = (args.split_strategy == 'time')
    nodes_df, train_mask, val_mask, test_mask = prepocess_nodes_df(nodes_df, config.JSON_PATH, timefilter=use_time_filter)

    # 2. Prepare labels and edges
    y = torch.tensor(nodes_df['y'].to_list())
    id_to_idx = dict(zip(nodes_df['id'], range(len(nodes_df))))
    
    # Map string IDs to integer indices
    src_idx = edges_df['source_node'].map(id_to_idx)
    dst_idx = edges_df['dest_node'].map(id_to_idx)
    
    # Combine into a single DataFrame and drop edges with missing nodes
    edges = pd.DataFrame({
        'source_node': src_idx,
        'dest_node': dst_idx,
        'edge_attr': edges_df['edge_attr']
    }).dropna().astype(int)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges[['source_node', 'dest_node']].values).T
    edge_attr = torch.tensor(edges['edge_attr'].values)
    
    # 3. Construct the final graph, passing the correct output path
    graph_data, old_to_new_idx = construct_graph(
        nodes_df, graph_output_path,  # Pass the constructed path here
        train_mask, val_mask, test_mask,
        x, y, edge_index, edge_attr
    )

    # 4. Save final mapping file
    with open(index_map_path, 'w') as f:
        json.dump(old_to_new_idx, f)
        
    print(f"Index map saved to {index_map_path}")
    print("--- STAGE 3 COMPLETE ---")

if __name__ == '__main__':
    main()
