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
                    x, y, edge_index, edge_attr, drop_dormant_nodes=False,
                    remove_isolated_nodes=True, isolated_strategy='connected_only'
                   ):
    
    # --- Additional filtering BEFORE isolated node detection ---
    print(f"current nodes count={len(nodes_df_)}")
    print(f"Keeping nodes that are in *defined_for.json only (following paper)")
    model_task_df = pd.read_json(os.path.join(config.JSON_PATH, 'model_definedFor_task.json'))
    dataset_task_df = pd.read_json(os.path.join(config.JSON_PATH, 'dataset_definedFor_task.json'))
    model_nodes = pd.unique(model_task_df['model_id'])
    dataset_nodes = pd.unique(dataset_task_df['dataset_id'])
    
    # Create mask for nodes that are in the defined_for files
    defined_for_mask = nodes_df_['id'].isin(np.concatenate([model_nodes,dataset_nodes]))
    nodes_df_ = nodes_df_[defined_for_mask]
    print(f"After cleaning by defined_for, nodes count={len(nodes_df_)}")
    
    # Remove nodes with empty descriptions
    desc_mask = ~nodes_df_['description'].isna()
    nodes_df_ = nodes_df_[desc_mask]
    print(f"Removing nodes where description is empty, nodes count={len(nodes_df_)}")
    
    # Reset index after filtering
    nodes_df_ = nodes_df_.reset_index(drop=True)
    
    # Update all tensors to match the filtered nodes_df
    # Create a mapping from old indices to new indices
    old_indices = np.arange(len(defined_for_mask))
    old_indices = old_indices[defined_for_mask]  # Filter by defined_for
    old_indices = old_indices[desc_mask[defined_for_mask]]  # Filter by description
    
    # Create mapping from old index to new index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}
    
    # Update tensors
    final_x = x[old_indices]
    final_y = y[old_indices]
    final_train_mask = train_mask[old_indices]
    final_val_mask = val_mask[old_indices]
    final_test_mask = test_mask[old_indices]
    
    # Update edge_index to use new indices
    if edge_index.numel() > 0:
        # Filter edges where both source and dest are in the kept nodes
        source_mask = torch.tensor([idx in old_to_new for idx in edge_index[0].numpy()])
        dest_mask = torch.tensor([idx in old_to_new for idx in edge_index[1].numpy()])
        edge_mask = source_mask & dest_mask
        
        filtered_edge_index = edge_index[:, edge_mask]
        filtered_edge_attr = edge_attr[edge_mask]
        
        # Remap edge indices to new node indices
        new_source = torch.tensor([old_to_new[idx.item()] for idx in filtered_edge_index[0]])
        new_dest = torch.tensor([old_to_new[idx.item()] for idx in filtered_edge_index[1]])
        edge_index = torch.stack([new_source, new_dest])
        edge_attr = filtered_edge_attr
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)
    
    # --- Now detect and remove isolated nodes (if requested) ---
    if remove_isolated_nodes:
        print("\nStep 6: Filtering isolated nodes and re-indexing graph...")
        
        # Find nodes that have at least one label
        labeled_nodes = final_y.sum(dim=1) > 0
        
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
            if isolated_strategy == 'connected_only':
                # Keep only connected nodes (remove all isolated nodes)
                final_node_mask = connected_nodes
                print(f"Strategy: Keeping only connected nodes")
            else:  # labeled_or_connected
                # Keep nodes that are either labeled OR connected
                final_node_mask = labeled_nodes | connected_nodes
                print(f"Strategy: Keeping labeled OR connected nodes")
        final_indices = final_node_mask.nonzero(as_tuple=True)[0]
        
        print(f"After additional filtering: {num_nodes} nodes. Final nodes after removing isolates: {len(final_indices)}")
        print(f"Labeled nodes: {labeled_nodes.sum()}")
        print(f"Connected nodes: {connected_nodes.sum()}")
        print(f"Nodes to keep (labeled OR connected): {final_node_mask.sum()}")
        
        # Filter the features, labels, and masks based on isolated node detection
        print(f"Before filtering - final_x shape: {final_x.shape}")
        print(f"Before filtering - final_y shape: {final_y.shape}")
        print(f"final_node_mask shape: {final_node_mask.shape}")
        print(f"final_node_mask sum: {final_node_mask.sum()}")
        
        final_x = final_x[final_node_mask]
        final_y = final_y[final_node_mask]
        final_train_mask = final_train_mask[final_node_mask]
        final_val_mask = final_val_mask[final_node_mask]
        final_test_mask = final_test_mask[final_node_mask]
        
        print(f"After filtering - final_x shape: {final_x.shape}")
        print(f"After filtering - final_y shape: {final_y.shape}")
    else:
        print("\nStep 6: Skipping isolated node removal...")
        num_nodes = len(nodes_df_)
        final_node_mask = torch.ones(num_nodes, dtype=torch.bool)
        final_indices = torch.arange(num_nodes)
    
    # --- Re-indexing Edges ---
    
    if remove_isolated_nodes:
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
    else:
        # If not removing isolated nodes, just use the existing edge_index
        final_edge_index = edge_index
        final_edge_attr = edge_attr
    
    # Create the final mapping dictionary (combining both filtering steps)
    final_old_to_new_idx = {}
    for new_idx, old_idx in enumerate(final_indices):
        # Map from the original node index to the final node index
        original_idx = old_indices[old_idx.item()]
        final_old_to_new_idx[original_idx.item()] = new_idx

    # --- Verification ---
    print(f"Original edges: {edge_index.shape[1]}")
    print(f"Final edges after re-indexing: {final_edge_index.shape[1]}")
    print(f"Final edge attributes: {final_edge_attr.shape[0]}")
    
    # Additional verification for isolated nodes
    if remove_isolated_nodes and final_edge_index.numel() > 0:
        all_nodes_in_edges = torch.unique(final_edge_index.flatten())
        total_nodes = len(final_x)
        connected_nodes_count = len(all_nodes_in_edges)
        
        # Check which nodes have labels
        labeled_nodes_count = (final_y.sum(dim=1) > 0).sum()
        
        # Find nodes that are neither labeled nor connected (truly isolated)
        labeled_mask = final_y.sum(dim=1) > 0
        connected_mask = torch.zeros(total_nodes, dtype=torch.bool)
        connected_mask[all_nodes_in_edges] = True
        
        truly_isolated_mask = ~(labeled_mask | connected_mask)
        truly_isolated_count = truly_isolated_mask.sum()
        
        print(f"Total nodes in final graph: {total_nodes}")
        print(f"Connected nodes: {connected_nodes_count}")
        print(f"Labeled nodes: {labeled_nodes_count}")
        print(f"Truly isolated nodes (neither labeled nor connected): {truly_isolated_count}")
        
        if truly_isolated_count > 0:
            print("WARNING: There are still truly isolated nodes in the graph!")
            # Find which nodes are truly isolated
            truly_isolated_indices = truly_isolated_mask.nonzero().flatten()
            print(f"First 10 truly isolated node indices: {truly_isolated_indices[:10].tolist()}")
        else:
            print("SUCCESS: No truly isolated nodes remaining in the graph!")
            print("Note: Some nodes may be labeled but not connected, which is expected.")
    
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
    return graph_data, final_old_to_new_idx


def main():
    # --- Update argparse to accept run_dir and strategy ---
    parser = argparse.ArgumentParser(description="Build the final graph object from pre-processed data.")
    parser.add_argument('--run_dir', type=str, required=True, help="The directory for this specific run's inputs and outputs.")
    parser.add_argument('--split_strategy', type=str, default='time', choices=['time', 'random'], help="The splitting strategy to use.")
    parser.add_argument('--remove_isolated', action='store_true', default=True, help="Remove isolated nodes from the graph.")
    parser.add_argument('--keep_isolated', action='store_true', help="Keep isolated nodes in the graph (overrides --remove_isolated).")
    parser.add_argument('--isolated_strategy', type=str, default='connected_only', 
                       choices=['connected_only', 'labeled_or_connected'], 
                       help="Strategy for handling isolated nodes: 'connected_only' removes all unconnected nodes, 'labeled_or_connected' keeps labeled nodes even if unconnected.")
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
    
    # Determine whether to remove isolated nodes
    remove_isolated = args.remove_isolated and not args.keep_isolated
    
    # 3. Construct the final graph, passing the correct output path
    graph_data, old_to_new_idx = construct_graph(
        nodes_df, graph_output_path,  # Pass the constructed path here
        train_mask, val_mask, test_mask,
        x, y, edge_index, edge_attr,
        remove_isolated_nodes=remove_isolated,
        isolated_strategy=args.isolated_strategy
    )

    # 4. Save final mapping file
    with open(index_map_path, 'w') as f:
        json.dump(old_to_new_idx, f)
        
    print(f"Index map saved to {index_map_path}")
    print("--- STAGE 3 COMPLETE ---")

if __name__ == '__main__':
    main()
