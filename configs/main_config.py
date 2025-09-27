# configs/main_config.py
import os
from datetime import datetime

# --- INPUT ---
JSON_PATH = '../../HuggingKG_V20250916155543'
HF_TOKEN = ''

# --- FILENAMES (Relative names, not full paths) ---
# The full path will be constructed by joining RUN_DIR + FILENAME
NODES_DF_FILENAME = 'nodes_df.pkl'
EDGES_DF_FILENAME = 'edges_df.pkl'
TASK_MAP_FILENAME = 'task_to_idx.json'
FEATURES_FILENAME = 'node_features.pt'
GRAPH_OUTPUT_FILENAME = 'final_graph.pt'
INDEX_MAP_FILENAME = 'old_to_new_idx.json'

# --- MODEL/SPLIT PARAMS ---
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
