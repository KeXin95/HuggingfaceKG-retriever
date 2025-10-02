import os
import json
import pandas as pd
import numpy as np
import torch
import pytz
from datetime import datetime
import argparse
from configs import main_config as config
from .utils import convert2group, encode_onehot
from .feas_author import add_author_col


def process_folder2df(json_dir):
    """
    Loads and processes all node and edge files from a directory into a single DataFrame.
    """
    print("Step 1: Loading and consolidating nodes and edges...")
    
    # --- Node Loading ---
    models_df = pd.read_json(os.path.join(json_dir, 'models.json'))
    tasks_df = pd.read_json(os.path.join(json_dir, 'tasks.json'))
    task_to_label = dict(zip(tasks_df['label'], tasks_df['id']))
    
    models_df['y_multi_lab'] = models_df['pipeline_tag'].apply(
        lambda x: [task_to_label.get(tag) for tag in x if tag in task_to_label] if isinstance(x, list) else []
    )
    models_df['type'] = 'model'

    datasets_df = pd.read_json(os.path.join(json_dir, 'datasets.json'))
    datasets_df['y_multi_lab'] = datasets_df['tasks']
    datasets_df['type'] = 'dataset'
    
    # --- Node Consolidation ---
    temp_df = pd.concat([
        models_df[['id', 'description', 'createdAt', 'type', 'y_multi_lab']],
        datasets_df[['id', 'description', 'createdAt', 'type', 'y_multi_lab']]
    ], ignore_index=True)
    
    temp_df = temp_df.dropna(subset=['description']).reset_index(drop=True)
    temp_df['relationships'] = ''

    print(f"current nodes count={len(temp_df)}")
    print(f"Keeping nodes that are in *defined_for.json only (following paper)")
    model_task_df = pd.read_json(f'{json_dir}/model_definedFor_task.json')
    dataset_task_df = pd.read_json(f'{json_dir}/dataset_definedFor_task.json')
    model_nodes = pd.unique(model_task_df['model_id'])
    dataset_nodes = pd.unique(dataset_task_df['dataset_id'])
    
    temp_df = temp_df[temp_df['id'].isin(np.concatenate([model_nodes,dataset_nodes]))]
    print(f"After cleaning, nodes count={len(temp_df)}")
    
    ## Extract author column
    temp_df['author'] = add_author_col(temp_df)
    print(f"Author column added, nodes count={len(temp_df)}")
    
    # Preparing idx to nodes
    temp_df = temp_df[~temp_df['description'].isna()]
    print(f"Removing nodes where description is empty, nodes count={len(temp_df)}")
    
    temp_df = temp_df.reset_index(drop=True)
    
    # --- Label/Task Mapping ---
    dataset_tasks_df = pd.read_json(os.path.join(json_dir, 'dataset_definedFor_task.json'))
    all_task_ids = pd.unique(list(dataset_tasks_df['task_id']) + list(tasks_df['id']))
    task_to_idx = {task_id: i for i, task_id in enumerate(all_task_ids)}
    
    temp_df['y_multi_lab'] = temp_df['y_multi_lab'].apply(
        lambda lst: [task_to_idx.get(item) for item in lst if item in task_to_idx] if isinstance(lst, list) else []
    )

    # --- One-Hot Encoding ---
    temp_df['y'] = encode_onehot(temp_df)

    # --- Edge Processing ---
    edge_files = {
        'model_finetune_model.json': ('base_model_id', 'model_id'),
        'model_trainedOrFineTunedOn_dataset.json': ('model_id', 'dataset_id'),
        'model_merge_model.json': ('base_model_id', 'model_id'),
        'model_quantized_model.json': ('base_model_id', 'model_id'),
        'model_adapter_model.json': ('base_model_id', 'model_id'),
    }

    edge_type_to_id = {}
    idx = 0
    for edge_type, (filename, (src_col, dst_col)) in enumerate(edge_files.items()):
        task = filename.split('.')[0]
        print(f"  Processing {filename} as edge type {edge_type}...")
        # Note: Assuming JSON files are line-delimited. Adjust if not.
        f = open(os.path.join(json_dir, filename),'r')
        cur_rel = json.load(f)
        # cur_rel = dict(zip([x[src_col] for x in cur_rel ], [x[dst_col] for x in cur_rel ]))
        cur_rel = convert2group(cur_rel, src_col, dst_col)
        temp = [f'{task}:{", ".join(x)}' if type(x) is list else '' for x in temp_df['id'].apply(lambda x:cur_rel.get(x, '')).to_list() ]
        temp = [f"{temp_df['relationships'].iloc[i]}, {temp[i]}" for i in range(len(temp_df))]
        temp = [x.strip(', ') for x in temp]
        temp = ['' if x==', ' else x for x in temp ]
        temp_df['relationships'] = temp
        edge_type_to_id[task] = idx
        idx+=1
    print(f'Total nodes after loading from JSON: {len(temp_df):,}')
            
    return temp_df, edge_type_to_id, task_to_idx

def create_edges_df_vectorized_v2(temp_df: pd.DataFrame, edge_type_to_id) -> pd.DataFrame:
    """
    Parses the 'relationships' column using a robust, no-loop approach
    that avoids the explicit join error.
    """
    # Define the regex pattern (same as before)
    pattern = r'([^,:]+:)(.*?)(?=[^,:]+:|$)'
    print(f"Dataframe snapshot: {len(temp_df):,} records")
    # Filter out empty rows
    df = temp_df[temp_df['relationships'].notna() & (temp_df['relationships'] != '')].copy()
    print(f"Processing {len(df):,} records")

    # --- REVISED LOGIC ---
    # 1. Set 'id' as the index BEFORE extraction. 
    #    This makes the join implicit and avoids the error.
    s = df.set_index('id')['relationships']

    # 2. Extract from the Series. The resulting MultiIndex will have 'id' as level 0.
    edges = s.str.extractall(pattern)
    edges.columns = ['edge_type', 'dest_nodes_str']

    # 3. Rename the index level 'id' to 'source_node' for clarity
    edges.index = edges.index.rename(['source_node', 'match'])
    
    # 4. Clean the edge_type and split the dest_nodes_str into lists
    edges['edge_type'] = edges['edge_type'].str.strip(':')
    edges['dest_node'] = edges['dest_nodes_str'].str.split(',')

    # 5. Explode the lists in the 'dest_node' column into separate rows
    edges_df = edges.explode('dest_node')

    # 6. Final cleanup
    edges_df['dest_node'] = edges_df['dest_node'].str.strip()
    edges_df.dropna(subset=['dest_node'], inplace=True)
    edges_df = edges_df[edges_df['dest_node'] != '']
    
    # Reset the index to turn 'source_node' back into a column

    edges_df = edges_df.reset_index(level='source_node')[['source_node', 'dest_node', 'edge_type']]
    edges_df['edge_attr'] = edges_df['edge_type'].apply(lambda x:edge_type_to_id[x.strip()])

    print(f"Total {len(edges_df):,} edges")
    return edges_df


def main():
    # --- Add argparse to accept the run directory ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="The output directory for this specific run.")
    args = parser.parse_args()

    print("--- STAGE 1: Building Base DataFrames from JSON ---")
    
    # Create the main output directory for this run
    os.makedirs(args.run_dir, exist_ok=True)
    
    nodes_df, edge_type_to_id, task_to_idx = process_folder2df(config.JSON_PATH)
    
    print("\nCreating Edges DataFrame...")
    edges_df = create_edges_df_vectorized_v2(nodes_df, edge_type_to_id)

    nodes_df_path = os.path.join(args.run_dir, config.NODES_DF_FILENAME)
    edges_df_path = os.path.join(args.run_dir, config.EDGES_DF_FILENAME)
    task_map_path = os.path.join(args.run_dir, config.TASK_MAP_FILENAME)

    # Save outputs of this stage
    nodes_df.to_pickle(nodes_df_path)
    edges_df.to_pickle(edges_df_path)
    with open(task_map_path, 'w') as f:
        json.dump(task_to_idx, f)
        
    print(f"Successfully saved outputs to: {args.run_dir}")
    print("--- STAGE 1 COMPLETE ---")

if __name__ == '__main__':
    main()
