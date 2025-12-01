# Data Pipeline for Hugging Face Knowledge Graph

This directory contains the complete data processing pipeline that transforms raw Hugging Face Hub JSON data into a graph neural network-ready format.

## Overview

The pipeline consists of three main stages that process Hugging Face Hub data into a heterogeneous knowledge graph:

1. **Stage 1: Build Base Data** - Extract nodes and edges from JSON files
2. **Stage 2: Generate Features** - Create node embeddings (BGE + optional BM25)
3. **Stage 3: Build Graph** - Construct final graph object with train/val/test splits

## Pipeline Architecture

```
Raw JSON Files
    ↓
[Stage 1] build_base_data.py
    ├── nodes_df.pkl (nodes with labels)
    ├── edges_df.pkl (relationships)
    └── task_to_idx.json (task mapping)
    ↓
[Stage 2] generate_features.py / generate_features_bm25.py
    └── node_features.pt (BGE embeddings or BGE+BM25)
    ↓
[Stage 3] build_graph.py
    ├── final_graph.pt (CogDL Graph object)
    └── old_to_new_idx.json (node reindexing map)
```

## Quick Start

### Run Complete Pipeline

Execute all three stages in sequence:

```bash
bash run_pipeline.sh
```

This will:
1. Create a unique run directory with timestamp
2. Process all JSON files
3. Generate embeddings
4. Build the final graph
5. Optionally start training (if configured)

### Manual Stage-by-Stage Execution

Run each stage individually for more control:

```bash
# Create a run directory
RUN_DIR="../../experiment_runs/my_custom_run"
mkdir -p "$RUN_DIR"

# Stage 1: Build base data
python -m scripts.build_base_data --run_dir "$RUN_DIR"

# Stage 2: Generate features (choose one)
python -m scripts.generate_features --run_dir "$RUN_DIR"          # BGE only
python -m scripts.generate_features_bm25 --run_dir "$RUN_DIR"     # BGE + BM25

# Stage 3: Build graph
python -m scripts.build_graph \
    --run_dir "$RUN_DIR" \
    --split_strategy time \
    --remove_isolated \
    --isolated_strategy connected_only
```

## Stage Details

### Stage 1: Build Base Data (`build_base_data.py`)

**Purpose**: Extract and consolidate nodes and edges from JSON files.

**Input**: 
- `models.json` - Model metadata
- `datasets.json` - Dataset metadata
- `tasks.json` - Task definitions
- Relationship JSON files (fine-tuning, training, merging, etc.)

**Processing Steps**:
1. Load models and datasets as nodes
2. Extract task labels and create multi-label encoding
3. Filter nodes to only those in `*_definedFor_task.json` files
4. Extract author information
5. Process edge relationships (fine-tuning, training, merging, quantization, adapters)
6. Create one-hot encoded task labels

**Output**:
- `nodes_df.pkl` - DataFrame with node information, labels, and masks
- `edges_df.pkl` - DataFrame with source, destination, and edge type
- `task_to_idx.json` - Mapping from task IDs to indices

**Key Features**:
- Filters nodes to only those with task definitions (following paper methodology)
- Removes nodes with empty descriptions
- Creates edge type mappings for heterogeneous graph

### Stage 2: Generate Features

#### Option A: BGE Embeddings Only (`generate_features.py`)

**Purpose**: Generate BGE (BAAI General Embedding) embeddings for node descriptions.

**Input**: `nodes_df.pkl` from Stage 1

**Processing**:
- Uses `BAAI/bge-base-en-v1.5` model
- Encodes all node descriptions
- Outputs 768-dimensional embeddings

**Output**: `node_features.pt` - Tensor of shape `[num_nodes, 768]`

#### Option B: BGE + BM25 Features (`generate_features_bm25.py`)

**Purpose**: Combine BGE embeddings with BM25 retrieval scores.

**Input**: `nodes_df.pkl` and `task_to_idx.json` from Stage 1

**Processing**:
- Generates BGE embeddings (768 dim)
- Computes BM25 scores between descriptions and task names (54 dim)
- Concatenates features: `[BGE (768) | BM25 (54)] = 822 dim`

**Output**: `node_features.pt` - Tensor of shape `[num_nodes, 822]`

**Why BM25?**: Task names often appear directly in model/dataset descriptions, providing a useful signal for classification.

### Stage 3: Build Graph (`build_graph.py`)

**Purpose**: Construct the final graph object with train/validation/test splits.

**Input**: 
- `nodes_df.pkl` from Stage 1
- `edges_df.pkl` from Stage 1
- `node_features.pt` from Stage 2
- `task_to_idx.json` from Stage 1

**Processing Steps**:
1. **Create splits** (time-based or random)
2. **Filter nodes** (remove isolated nodes if requested)
3. **Re-index** nodes and edges
4. **Construct graph** object

**Output**:
- `final_graph.pt` - CogDL Graph object
- `old_to_new_idx.json` - Mapping from original to final node indices

## Configuration

Edit `configs/main_config.py` to configure:

```python
# Input data path
JSON_PATH = '../../HuggingKG_V20250916155543'

# Hugging Face token (required for BGE embeddings)
HF_TOKEN = 'your_token_here'

# Embedding model
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'

# Output filenames (relative to run_dir)
NODES_DF_FILENAME = 'nodes_df.pkl'
EDGES_DF_FILENAME = 'edges_df.pkl'
TASK_MAP_FILENAME = 'task_to_idx.json'
FEATURES_FILENAME = 'node_features.pt'
GRAPH_OUTPUT_FILENAME = 'final_graph.pt'
INDEX_MAP_FILENAME = 'old_to_new_idx.json'
```

## Command-Line Arguments

### Stage 1: `build_base_data.py`

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--run_dir` | str | Yes | Output directory for this run |

### Stage 2: `generate_features.py` / `generate_features_bm25.py`

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--run_dir` | str | Yes | Directory containing Stage 1 outputs |

### Stage 3: `build_graph.py`

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--run_dir` | str | Yes | - | Directory containing Stage 1 & 2 outputs |
| `--split_strategy` | str | No | `time` | Split strategy: `time` or `random` |
| `--remove_isolated` | flag | No | True | Remove isolated nodes from graph |
| `--keep_isolated` | flag | No | False | Keep isolated nodes (overrides `--remove_isolated`) |
| `--isolated_strategy` | str | No | `connected_only` | Strategy: `connected_only` or `labeled_or_connected` |

## Split Strategies

### Time-Based Split (Default)

Splits nodes based on creation date to simulate real-world temporal evaluation:

- **Training**: Models before 2024-09-15, Datasets before 2024-04-15
- **Validation**: Models 2024-09-15 to 2024-10-15, Datasets 2024-04-15 to 2024-08-15
- **Test**: Models 2024-10-15 to 2024-12-15, Datasets 2024-08-15 to 2024-12-15

### Random Split

Random 70/15/15 split across all nodes (seed=42 for reproducibility).

## Isolated Node Handling

The pipeline can filter out isolated nodes (nodes with no edges):

- **`connected_only`**: Keep only nodes that participate in at least one edge
- **`labeled_or_connected`**: Keep nodes that are either labeled OR connected

This is important because:
- Isolated nodes provide no graph signal
- They can skew evaluation metrics
- The paper methodology removes all isolated nodes

## Output Structure

Each run creates a directory with:

```
experiment_runs/run_YYYY-MM-DD_HH-MM-SS/
├── nodes_df.pkl              # Stage 1: Node DataFrame
├── edges_df.pkl              # Stage 1: Edge DataFrame
├── task_to_idx.json         # Stage 1: Task ID mapping
├── node_features.pt         # Stage 2: Feature tensor
├── final_graph.pt           # Stage 3: Final graph object
└── old_to_new_idx.json      # Stage 3: Node reindexing map
```

## Graph Object Format

The final graph object (`final_graph.pt`) is a CogDL `Graph` object with:

```python
graph = torch.load('final_graph.pt')
# Data(
#     x=[num_nodes, feature_dim],      # Node features (768 or 822)
#     y=[num_nodes, num_tasks],        # Multi-label task assignments
#     edge_index=[2, num_edges],        # Edge connectivity
#     edge_attr=[num_edges],            # Edge type IDs
#     train_mask=[num_nodes],           # Training set mask
#     val_mask=[num_nodes],             # Validation set mask
#     test_mask=[num_nodes]              # Test set mask
# )
```

## Edge Types

The pipeline processes 5 types of relationships:

1. **Fine-tuning** (`model_finetune_model.json`)
2. **Training** (`model_trainedOrFineTunedOn_dataset.json`)
3. **Merging** (`model_merge_model.json`)
4. **Quantization** (`model_quantized_model.json`)
5. **Adapters** (`model_adapter_model.json`)

Each edge type is assigned a unique integer ID stored in `edge_attr`.

## Requirements

```bash
pip install torch pandas numpy pytz FlagEmbedding huggingface_hub bm25s cogdl scikit-learn
```

**Note**: 
- Requires a Hugging Face token for downloading BGE models
- GPU recommended for BGE embedding generation (uses FP16)
- CogDL must be installed for graph construction

## Troubleshooting

### Missing Hugging Face Token
Set `HF_TOKEN` in `configs/main_config.py` or export as environment variable.

### Out of Memory
- Reduce batch size in `generate_features.py` (default: 64)
- Use BGE-only features instead of BGE+BM25
- Process in smaller chunks

### Isolated Nodes Warning
If you see warnings about isolated nodes, check:
- `--isolated_strategy` setting
- Whether nodes have edges in the input JSON files
- Edge filtering logic in Stage 1

### Edge Index Errors
Ensure all edges reference nodes that exist in `nodes_df.pkl`. The pipeline automatically filters invalid edges.

## Performance Notes

- **Stage 1**: Fast (seconds to minutes, depends on JSON file size)
- **Stage 2**: Slow (minutes to hours, depends on GPU and number of nodes)
  - BGE-only: ~1-2 hours for 100K nodes on GPU
  - BGE+BM25: ~2-3 hours for 100K nodes
- **Stage 3**: Fast (seconds to minutes)

## Next Steps

After running the pipeline, use the generated graph for training:

```bash
cd ../models
python train.py \
    --model_type gcn \
    --graph_path ../experiment_runs/run_YYYY-MM-DD_HH-MM-SS/final_graph.pt \
    --save_dir ./results
```

