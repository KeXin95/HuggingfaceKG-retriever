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

**🔗 [Download Experiment Results from Google Drive](https://drive.google.com/drive/u/1/folders/1tk03BEFGyRhB9Sn7UzvXO1yAdc3MdopH)**

**Note**: The experiment results are large (several GB) and contain all the processed data from the pipeline stages.

## Data Storage

### Large Files and Experiment Results

The pipeline generates large files that may exceed GitHub's file size limits:

- **Experiment runs** (`experiment_runs/`): Contains processed data, embeddings, and trained models
- **Input data** (`HuggingKG_V20250916155543/`): Raw JSON data from Hugging Face Hub
- **Model checkpoints**: Trained GNN models (`trained_*.pt` files)


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
python -m scripts.generate_features_bm25 --run_dir ./experiment_runs/my_run

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
### Features
1. text embedding: use bge-base-en to get embedding from description
2. bm25 score: use model/dataset description as query and list of tasks name as corpus, we create this feature as a lot of times the user has put the task name directly on the description, so that we can use it to enhance model performance.

### Graph data field explanation
```
graph = torch.load('experiment_runs/run_2025-09-26_21-25-12/final_graph.pt')
graph
> Data(x=[107694, 822], edge_index=[2, 124056], y=[107694, 54], train_mask=[107694], val_mask=[107694], test_mask=[107694])
```
- x: BGE-en-base (column 1-768) embedding and BM25 retriever (column 768-822) of 107694 records
- edge_index: edge pairs
- y: target label (Multilabel), example: [0, 0, 0, 1, 0, ...0], each entries in this lists is 1 indicates that the user has labeled its model/dataset to this task, in this example, a "1" at index 3 indicates that this model/dataset is associated to image-to-video (check `task_to_idx.json` in each run result)
- train_mask/val_mask/test_mask: a list of True/False indicating the nodes is for training/validation/testing

### CHANGELOG
1. 2025-10-04: 
    1. Replace Standard scaler to use L2 norm
    2. Add EDA on original graph from paper
    3. Add GAT training script
    4. Removed **ALL** isolated nodes from graph
2. 2025-10-11: 
    1. Use bm25 as features on top of original BGE embeddings
    2. Regenerate dataset with dates closer to our own collections dates (latest data can be found [here](https://drive.google.com/drive/u/1/folders/1tk03BEFGyRhB9Sn7UzvXO1yAdc3MdopH))
    3. Train SAGECONV and Graph Transformer model and achieve much better micro-f1 result.

### Performance Tracking

| Date | Model | Run ID | Best Val Micro-F1 | Test Loss | Test Micro-F1 | Notes | Model filename |
|------|-------|--------|-------------------|-----------|---------------|-------| -------------- | 
| 2025-10-04 | GCN | run_2025-10-04_21-45-35 | 0.8162 | 0.0695 | 0.4004 | Standard scaler | trained_gcn.pt |
| 2025-10-04 | GAT | run_2025-10-04_21-45-35 | 0.8203 | 0.0522 | 0.4073 | Standard scaler | trained_gat.pt |
| 2025-10-11 | GCN | run_2025-10-11_13-12-14 | 0.8918 | 0.0720 | 0.4912 | With own scaler + BM25 features | trained_gcn_own_scaler.pt |
| 2025-10-11 | SAGECONV | run_2025-10-11_13-12-14 | 0.9284 | 0.0306 | 0.5528 | With own scaler + BM25 features | trained_sageconv_own_scaler.pt |
| 2025-10-11 | Graph Transformer | run_2025-10-11_13-12-14 | 0.9268 | 0.0100 | 0.6032 | With own scaler + BM25 features | trained_trans_own_scaler.pt |

#### Key Improvements:
- **2025-10-11**: Significant improvement in both validation and test Micro-F1 scores
- **BM25 Features**: Addition of BM25 features on top of BGE embeddings improved performance
- **Custom Scaler**: Using own scaler instead of standard scaler contributed to better results
