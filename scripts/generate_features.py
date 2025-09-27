# scripts/generate_features.py
import os
import pandas as pd

import torch
import pandas as pd
from FlagEmbedding import FlagModel
from huggingface_hub import login
import argparse

from configs import main_config as config


os.environ["HF_TOKEN"] = config.HF_TOKEN

# Let's rename it for clarity
def generate_bge_embeddings(descriptions: pd.Series) -> torch.Tensor:
    print(f"Total nodes to embed: {len(descriptions)}")
    login(token=config.HF_TOKEN)
    
    model = FlagModel(config.EMBEDDING_MODEL, use_fp16=True, token=config.HF_TOKEN, device='cuda')
    
    # Handle missing descriptions
    desc_list = descriptions.fillna('').tolist()
    
    print(f"Encoding {len(desc_list)} descriptions...")
    features = model.encode(desc_list, batch_size=64)
    x = torch.tensor(features, dtype=torch.float)
    print(f"Feature matrix 'x' created with shape: {x.shape}")
    return x

def main():
    # --- Add argparse to accept the run directory ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="The directory for this specific run's inputs and outputs.")
    args = parser.parse_args()

    print("\n--- STAGE 2: Generating Node Features ---")

    # --- Construct full paths using the passed-in run_dir ---
    nodes_df_path = os.path.join(args.run_dir, config.NODES_DF_FILENAME)
    features_path = os.path.join(args.run_dir, config.FEATURES_FILENAME)

    print(f"Loading nodes DataFrame from {nodes_df_path}")
    nodes_df = pd.read_pickle(nodes_df_path)

    # Generate embeddings
    x_features = generate_bge_embeddings(nodes_df['description'])

    # Save the feature tensor
    torch.save(x_features, features_path)
    print(f"Node features tensor saved to {features_path}")
    print("--- STAGE 2 COMPLETE ---")

if __name__ == '__main__':
    main()
