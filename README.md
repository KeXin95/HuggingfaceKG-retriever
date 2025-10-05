# HuggingfaceKG-retriever

A knowledge graph-based retrieval system for Hugging Face models and datasets. This project constructs a heterogeneous knowledge graph from Hugging Face Hub data and provides tools for graph-based model and dataset recommendation.

## Overview

This project processes Hugging Face Hub data to build a comprehensive knowledge graph containing:
- **Models** and **Datasets** as nodes
- **Relationships** between models and datasets as edges (fine-tuning, training, quantization, merging, etc.)
- **Task labels** for multi-label classification
- **Text embeddings** using BGE (BAAI General Embedding) models

## Project Structure

```
HuggingfaceKG-retriever/
├── configs/
│   └── main_config.py          # Configuration settings
├── scripts/
│   ├── build_base_data.py    # Stage 1: Process JSON data into DataFrames
│   ├── generate_features.py  # Stage 2: Generate BGE embeddings
│   ├── build_graph.py        # Stage 3: Construct final graph object
│   └── utils.py                # Utility functions
├── tune_huggingface.py         # GNN training script
├── run_pipeline.sh             # Complete pipeline execution
├── exploration.ipynb           # Data exploration notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── experiment_runs/            # Output directory (generated, not in repo)
│   └── run_YYYY-MM-DD_HH-MM-SS/  # Individual experiment runs
│       ├── nodes_df.pkl           # Processed node DataFrame
│       ├── edges_df.pkl           # Processed edge DataFrame
│       ├── node_features.pt       # BGE embeddings tensor
│       ├── final_graph.pt         # Final graph object
│       ├── task_to_idx.json       # Task ID mapping
│       ├── old_to_new_idx.json    # Node reindexing mapping
│       └── trained_*.pt           # Trained model checkpoints
└── HuggingKG_V20250916155543/  # Input data folder (large, not in repo)
    ├── models.json             # Model metadata
    ├── datasets.json           # Dataset metadata
    ├── tasks.json              # Task definitions
    ├── model_definedFor_task.json
    ├── dataset_definedFor_task.json
    ├── model_finetune_model.json
    ├── model_trainedOrFineTunedOn_dataset.json
    ├── model_merge_model.json
    ├── model_quantized_model.json
    └── model_adapter_model.json
```


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HuggingfaceKG-retriever
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install CogDL (required for graph neural networks):
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

**🔗 [Download Experiment Results from Google Drive](https://drive.google.com/drive/u/1/folders/1z4z75BCGtIfGcgeztAWTcP2yNiY1nNUK)**

**Note**: The experiment results are large (several GB) and contain all the processed data from the pipeline stages.

## Data Storage

### Large Files and Experiment Results

The pipeline generates large files that may exceed GitHub's file size limits:

- **Experiment runs** (`experiment_runs/`): Contains processed data, embeddings, and trained models
- **Input data** (`HuggingKG_V20250916155543/`): Raw JSON data from Hugging Face Hub
- **Model checkpoints**: Trained GNN models (`.pt` files)


## Configuration

Edit `configs/main_config.py` to configure:
- Input data path (`JSON_PATH`)
- Hugging Face token (`HF_TOKEN`)
- Embedding model (`EMBEDDING_MODEL`)
- Output filenames

## Usage

### Quick Start

Run the complete pipeline:
```bash
bash run_pipeline.sh
```

This will:
1. Process JSON data into nodes and edges DataFrames
2. Generate BGE embeddings for all descriptions
3. Build the final graph with train/val/test splits
4. Train a GCN model on the graph

### Manual Pipeline Execution

Run each stage individually:

```bash
# Stage 1: Build base data
python -m scripts.build_base_data --run_dir ./experiment_runs/my_run

# Stage 2: Generate features
python -m scripts.generate_features --run_dir ./experiment_runs/my_run

# Stage 3: Build graph
python -m scripts.build_graph --run_dir ./experiment_runs/my_run --split_strategy time
```

### Training GNN Models

Train graph neural networks using CogDL:

```bash
python tune_huggingface.py \
    --data_dir ./experiment_runs/my_run \
    --save_path ./experiment_runs/my_run/trained_model.pt \
    --models gcn gat graphsage \
    --devices 0 \
    --seeds 0 1 2
```

### Data Exploration

Use the provided Jupyter notebook for data exploration:
```bash
jupyter notebook exploration.ipynb
```

### CHANGELOG
1. 2025-10-04: 
    1. Replace Standard scaler to use L2 norm
    2. Add EDA on original graph from paper
    3. Add GAT training script

### Performance Tracking
#### 2025-10-04:
> CUDA_VISIBLE_DEVICES=4 python train_GCN.py --graph_path ./experiment_runs/run_2025-10-04_21-45-35/final_graph.pt --save_path ./experiment_runs/run_2025-10-04_21-45-35/trained_gcn.pt
```
Training finished!
🏆 Best Validation Micro-F1 Score: 0.8162

Loading best model and evaluating on the test set...
-----------------------------------------
Final Test Results:
  - Test Loss: 0.0695
  - Test Micro-F1: 0.4004
-----------------------------------------
```

> CUDA_VISIBLE_DEVICES=4 python train_GAT.py --graph_path ./experiment_runs/run_2025-10-04_21-45-35/final_graph.pt --save_path ./experiment_runs/run_2025-10-04_21-45-35/trained_gat.pt
```
Training finished!
🏆 Best Validation Micro-F1 Score: 0.8203

Loading best model and evaluating on the test set...
-----------------------------------------
Final Test Results:
  - Test Loss: 0.0522
  - Test Micro-F1: 0.4073
-----------------------------------------
```