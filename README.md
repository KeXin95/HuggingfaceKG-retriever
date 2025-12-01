# HuggingfaceKG-retriever

A knowledge graph-based retrieval system for Hugging Face models and datasets. This project constructs a heterogeneous knowledge graph from Hugging Face Hub data and provides tools for graph-based model and dataset recommendation using Graph Neural Networks (GNNs) and Large Language Models (LLMs).

## Overview

This project processes Hugging Face Hub data to build a comprehensive knowledge graph containing:
- **Models** and **Datasets** as nodes
- **Relationships** between models and datasets as edges (fine-tuning, training, quantization, merging, adapters)
- **Task labels** for multi-label classification
- **Text embeddings** using BGE (BAAI General Embedding) models
- **BM25 features** for enhanced task classification

The system supports multiple approaches:
1. **GNN-based classification**: Train Graph Neural Networks for multi-label task prediction
2. **GRetriever**: Fine-tune LLMs with graph-aware retrieval for task classification and link prediction

## Project Structure

```
HuggingfaceKG-retriever/
├── scripts/
│   ├── build-data-pipeline/          # Data processing pipeline
│   │   ├── configs/
│   │   │   └── main_config.py        # Configuration settings
│   │   ├── scripts/
│   │   │   ├── build_base_data.py    # Stage 1: Process JSON → DataFrames
│   │   │   ├── generate_features.py  # Stage 2a: Generate BGE embeddings
│   │   │   ├── generate_features_bm25.py  # Stage 2b: BGE + BM25 features
│   │   │   ├── build_graph.py        # Stage 3: Construct graph object
│   │   │   └── utils.py              # Utility functions
│   │   ├── run_pipeline.sh           # Complete pipeline execution
│   │   └── readme.md                 # Detailed pipeline documentation
│   │
│   ├── GNN_training/                 # GNN model training
│   │   ├── train.py                  # Main training script
│   │   ├── model_utils.py            # Model utilities
│   │   └── README.md                 # Training documentation
│   │
│   ├── gretriever/                   # GRetriever (LLM + Graph)
│   │   ├── gretriever.py             # GRetriever implementation
│   │   ├── finetune_llm_taskclass.py # Fine-tune for task classification
│   │   ├── finetune_llm_linkpred.py  # Fine-tune for link prediction
│   │   ├── gret_eval.py              # Evaluation script
│   │   └── create_classification_evalset.py  # Create evaluation sets
│   │
│   ├── notebooks/                    # Jupyter notebooks
│   │   ├── exploration.ipynb         # Data exploration
│   │   ├── EDA-ori.ipynb             # Original data EDA
│   │   └── midterm-analysis.ipynb    # Analysis notebooks
│   │
│   ├── eval.py                       # Evaluation utilities
│   ├── plot_eval.py                  # Plotting evaluation results
│   ├── eda_analysis.py               # EDA analysis script
│   └── data/                         # Evaluation results and plots
│
├── CogDL-master/                     # CogDL library (for graph construction)
├── experiment_runs/                  # Output directory (generated)
│   └── run_YYYY-MM-DD_HH-MM-SS/      # Individual experiment runs
│       ├── nodes_df.pkl              # Processed node DataFrame
│       ├── edges_df.pkl              # Processed edge DataFrame
│       ├── node_features.pt         # BGE embeddings tensor
│       ├── final_graph.pt           # Final graph object
│       ├── task_to_idx.json         # Task ID mapping
│       ├── old_to_new_idx.json      # Node reindexing mapping
│       └── trained_*.pt             # Trained model checkpoints
│
├── HuggingKG_V20250916155543/        # Input data folder (large, not in repo)
│   ├── models.json                   # Model metadata
│   ├── datasets.json                 # Dataset metadata
│   ├── tasks.json                    # Task definitions
│   ├── model_definedFor_task.json
│   ├── dataset_definedFor_task.json
│   ├── model_finetune_model.json
│   ├── model_trainedOrFineTunedOn_dataset.json
│   ├── model_merge_model.json
│   ├── model_quantized_model.json
│   └── model_adapter_model.json
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd HuggingfaceKG-retriever
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch`, `torch-geometric` - Deep learning frameworks
- `transformers` - Hugging Face transformers
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `FlagEmbedding` - BGE embeddings
- `bm25s` - BM25 retrieval
- `cogdl` - Graph neural network library

### 3. Install CogDL (Required for Graph Construction)

```bash
cd CogDL-master
pip install -e .
cd ..
```

**Note**: CogDL is included as a subdirectory in this repository. If you prefer to install from source:

```bash
git clone https://github.com/THUDM/CogDL.git
cd CogDL
pip install -e .
cd ..
```

## Input Data Setup

### Obtaining the Data Folder

The pipeline requires a large input data folder `HuggingKG_V20250916155543/` containing JSON files with Hugging Face Hub data. This folder is **not included in the repository** due to its large size.

### Required JSON Files

The input folder must contain these files:
- `models.json` - Model metadata and descriptions
- `datasets.json` - Dataset metadata and descriptions  
- `tasks.json` - Task definitions and labels
- `model_definedFor_task.json` - Model-to-task relationships
- `dataset_definedFor_task.json` - Dataset-to-task relationships
- `model_finetune_model.json` - Fine-tuning relationships
- `model_trainedOrFineTunedOn_dataset.json` - Training relationships
- `model_merge_model.json` - Model merging relationships
- `model_quantized_model.json` - Quantization relationships
- `model_adapter_model.json` - Adapter relationships

### Data Folder Size
- **Estimated size**: 1-5 GB (depending on data completeness)
- **File count**: ~10 JSON files
- **Format**: Line-delimited JSON (JSONL) for relationship files

### Pre-computed Experiment Results

Pre-computed experiment results are available for download:

**🔗 [Download Experiment Results from Google Drive](https://drive.google.com/drive/u/1/folders/1tk03BEFGyRhB9Sn7UzvXO1yAdc3MdopH)**

**Note**: The experiment results are large (several GB) and contain all the processed data from the pipeline stages.

## Quick Start

### Complete Pipeline Execution

Run the complete data processing pipeline:

```bash
cd scripts/build-data-pipeline
bash run_pipeline.sh
```

This will:
1. Create a unique run directory with timestamp
2. Process JSON data into nodes and edges DataFrames
3. Generate BGE embeddings (and optional BM25 features)
4. Build the final graph with train/val/test splits
5. Save all outputs to `experiment_runs/run_YYYY-MM-DD_HH-MM-SS/`

### Manual Pipeline Execution

Run each stage individually for more control:

```bash
cd scripts/build-data-pipeline

# Create a run directory
RUN_DIR="../../experiment_runs/my_custom_run"
mkdir -p "$RUN_DIR"

# Stage 1: Build base data
python -m scripts.build_base_data --run_dir "$RUN_DIR"

# Stage 2: Generate features (choose one)
python -m scripts.generate_features --run_dir "$RUN_DIR"          # BGE only (768 dim)
python -m scripts.generate_features_bm25 --run_dir "$RUN_DIR"     # BGE + BM25 (822 dim)

# Stage 3: Build graph
python -m scripts.build_graph \
    --run_dir "$RUN_DIR" \
    --split_strategy time \
    --remove_isolated \
    --isolated_strategy connected_only
```

For detailed pipeline documentation, see [`scripts/build-data-pipeline/readme.md`](scripts/build-data-pipeline/readme.md).

### Training GNN Models

Train Graph Neural Networks for multi-label task classification:

```bash
cd scripts/GNN_training

python train.py \
    --model_type gcn \
    --graph_path ../experiment_runs/run_2025-10-11_13-12-14/final_graph.pt \
    --save_dir ./results/gcn_run
```

**Supported models**: `gcn`, `gat`, `sage`, `transformer`, `gatv2`

**Options**:
- `--use_focal` - Use Focal Loss for class imbalance
- `--exclude_bm25` - Use only BGE embeddings (drop BM25 features)

For detailed training documentation, see [`scripts/GNN_training/README.md`](scripts/GNN_training/README.md).

### GRetriever (LLM + Graph)

Fine-tune LLMs with graph-aware retrieval:

```bash
cd scripts/gretriever

# Task classification
python finetune_llm_taskclass.py

# Link prediction
python finetune_llm_linkpred.py
```

## Configuration

Edit `scripts/build-data-pipeline/configs/main_config.py` to configure:

```python
# Input data path
JSON_PATH = '../../HuggingKG_V20250916155543'

# Hugging Face token (required for BGE embeddings)
HF_TOKEN = 'your_token_here'

# Embedding model
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
```

## Features

### Node Features

1. **BGE Embeddings** (768 dimensions)
   - Uses `BAAI/bge-base-en-v1.5` to encode model/dataset descriptions
   - Provides semantic representations of text content

2. **BM25 Features** (54 dimensions)
   - Computes BM25 scores between descriptions and task names
   - Captures explicit task mentions in descriptions
   - Combined with BGE: `[BGE (768) | BM25 (54)] = 822 dim`

### Graph Structure and Data Format

The knowledge graph is represented as a PyTorch Geometric `Data` object:

```python
graph = torch.load('experiment_runs/run_2025-10-11_13-12-14/final_graph.pt')
# Data(
#     x=[107694, 822],              # Node features (BGE + BM25)
#     edge_index=[2, 124056],       # Edge connectivity
#     edge_attr=[124056],           # Edge type IDs
#     y=[107694, 54],               # Multi-label task assignments
#     train_mask=[107694],          # Training set mask
#     val_mask=[107694],            # Validation set mask
#     test_mask=[107694]            # Test set mask
# )
```

**Components:**
- **Nodes** (`x`): Models and Datasets (filtered to those with task definitions)
  - Features: BGE embeddings (columns 0-767) + BM25 scores (columns 768-821)
- **Edges** (`edge_index`, `edge_attr`): 5 relationship types
  - Fine-tuning (`model_finetune_model`) - ID: 0
  - Training (`model_trainedOrFineTunedOn_dataset`) - ID: 1
  - Merging (`model_merge_model`) - ID: 2
  - Quantization (`model_quantized_model`) - ID: 3
  - Adapters (`model_adapter_model`) - ID: 4
- **Labels** (`y`): Multi-label task assignments (54 task classes)
  - Binary vectors where `1` indicates task association
  - Task mapping: Check `task_to_idx.json` in each run directory
- **Splits** (`train_mask/val_mask/test_mask`): Boolean masks for train/validation/test sets

## Troubleshooting

### Missing Hugging Face Token
Set `HF_TOKEN` in `scripts/build-data-pipeline/configs/main_config.py` or export as environment variable:
```bash
export HF_TOKEN='your_token_here'
```

### Out of Memory
- Reduce batch size in `generate_features.py` (default: 64)
- Use BGE-only features instead of BGE+BM25
- Process in smaller chunks

### Isolated Nodes Warning
If you see warnings about isolated nodes:
- Check `--isolated_strategy` setting
- Verify nodes have edges in the input JSON files
- Review edge filtering logic in Stage 1

### Edge Index Errors
Ensure all edges reference nodes that exist in `nodes_df.pkl`. The pipeline automatically filters invalid edges.

## Documentation

- **Data Pipeline**: [`scripts/build-data-pipeline/readme.md`](scripts/build-data-pipeline/readme.md)
- **GNN Training**: [`scripts/GNN_training/README.md`](scripts/GNN_training/README.md)



