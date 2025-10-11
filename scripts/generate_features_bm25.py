import os
import pandas as pd
from tqdm.auto import tqdm 
import bm25s
import argparse
import json
import numpy as np
import torch

from configs import main_config as config
from scripts.generate_features import generate_bge_embeddings

def bm25_get_emb(desc, task_to_idx):
    # Create your corpus here
    corpus = list(task_to_idx.keys())
    
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    
    # Query the corpus and get top-k results
    query = desc
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=len(corpus))
    
    original_array = np.zeros_like(scores)

    # 4. Use the shuffle_indices to place the elements back in their original spots
    # This single line performs the entire mapping operation
    final_res = []
    for idx, score in enumerate(scores):
        idx_rearrange = [task_to_idx[str(x)] for x in results[idx]]
        original_array[idx, idx_rearrange] = score

    return original_array

def combine_feas(desc, task_to_idx):
    bm25_feas = bm25_get_emb(desc, task_to_idx)
    ori_emb = generate_bge_embeddings(desc)
    combined = torch.cat((ori_emb, torch.Tensor(bm25_feas)), dim=1)
    return combined
    

def main():
    import time
    start = time.time()
    # --- Add argparse to accept the run directory ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="The directory for this specific run's inputs and outputs.")
    args = parser.parse_args()

    print("\n--- STAGE 2: Generating Node Features ---")

    # --- Construct full paths using the passed-in run_dir ---
    nodes_df_path = os.path.join(args.run_dir, config.NODES_DF_FILENAME)
    features_path = os.path.join(args.run_dir, config.FEATURES_FILENAME)
    features_df_path = os.path.join(args.run_dir, config.FEATURES_FILENAME.replace('.pt', '_df.pkl'))

    print(f"Loading nodes DataFrame from {nodes_df_path}")
    nodes_df = pd.read_pickle(nodes_df_path)

    task_to_idx = json.load(
        open(
            os.path.join(args.run_dir, 'task_to_idx.json'), 'r'
        )
    )

    # Generate embeddings
    x_features = combine_feas(nodes_df['description'], task_to_idx)
    # Note: make sure x_features also in torch type (numpy array isnt accepted)
    # assert type(x_features)==torch.Tensor
    
    # Save the feature tensor
    torch.save(torch.Tensor(x_features), features_path)
    print(f"Node features tensor saved to {features_path}")

    nodes_df['ori_bm25_emb'] = x_features.cpu().numpy().tolist()#x_features.tolist()
    nodes_df.to_pickle(features_df_path)
    print(f"Node df with features tensor saved to {features_df_path}")
    
    print("--- STAGE 2 COMPLETE ---")
    print(f'{time.time()-start:.2f} seconds')

if __name__ == '__main__':
    main()
