# Model Training and Evaluation

This directory contains all model training and evaluation code for the Hugging Face Knowledge Graph project. It includes both pure Graph Neural Network (GNN) approaches and hybrid LLM + Graph approaches (GRetriever).

## Directory Structure

```
models/
├── model_utils.py                      # Shared GNN model definitions (GCN, GAT, SAGE, etc.)
├── utils.py                            # Shared utilities (graph loading, training helpers)
├── train.py                            # Main GNN training script
├── inference_graph_to_df.py            # Generate predictions for all nodes and save to parquet
├── model_config_example.json           # Example config file for inference
├── eval_helper.py                      # Helper functions for evaluation
├── g_retrieval_w_labels_qwen.py        # Fine-tune GRetriever with Qwen2.5-3B-Instruct
├── g_retrieval_eval_w_labels_qwen.py   # Evaluate GRetriever models
├── gretriever-mistral/                 # LLM + Graph hybrid (GRetriever) with Mistral
│   ├── gretriever.py                   # GRetriever implementation
│   ├── gret_eval.py                    # GRetriever evaluation script
│   └── rag_building_exploration.ipynb  # Exploration notebook
├── results_inferences/                 # Output directory for inference results
└── README.md                           # This file
```

## Shared Components

### Models (`model_utils.py`)

All GNN architectures are defined in a single shared module:

- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **SAGE**: GraphSAGE
- **GraphTransformer**: Transformer-based GNN
- **GATv2**: GATv2 with edge attributes

### Utilities (`utils.py`)

Shared utility functions used across training and evaluation:

- **`fix_graph_data()`**: Convert CogDL graph to PyTorch Geometric Data format
- **`initialize_bias()`**: Initialize bias for Focal Loss training
- **`analyze_long_tail_performance()`**: Generate long-tail performance visualization
- **`print_stats()`**: Print graph statistics (train/val/test splits)
- **`convert2group()`**: Group relationship data into dictionaries
- **`encode_onehot()`**: Create one-hot encoded matrices for multi-label data

## Usage

### GNN Training

Train pure GNN models directly from the `models/` directory:

```bash
cd models
python train.py \
    --model_type gcn \
    --graph_path ../experiment_runs/run_2025-10-11_13-12-14/final_graph.pt \
    --save_dir ./results/gcn_run
```

**Supported models**: `gcn`, `gat`, `sage`, `transformer`, `gatv2`

**Options**:
- `--use_focal`: Use Focal Loss for class imbalance
- `--exclude_bm25`: Use only BGE embeddings (drop BM25 features)

**Trained GNN checkpoints** can be downloaded from Google Drive:  
[Download trained GNN models](https://drive.google.com/drive/folders/1-VkIPmk1hcd772ofoF0tR8VlBozRemQ7?usp=drive_link)

### Inference

Run inference on a trained model to generate predictions for all nodes in the graph:

```bash
cd models

python inference_graph_to_df.py \
    --run_id run_2025-10-11_13-12-14 \
    --config_path model_config_example.json \
    --model_type GATV2 \
    --output_path results_inferences/GATV2_predictions.parquet
```

**Arguments**:
- `--run_id`: Experiment run ID (directory name in `experiment_runs/`)
- `--config_path`: Path to JSON config file with model configurations
- `--model_type`: Model type to use (`GCN`, `GAT`, `GATV2`, `TRANS`, `SAGE`)
- `--output_path`: Output path for the parquet file with predictions

**Configuration File** (`model_config_example.json`):
The config file should contain model-specific settings:
- `model_type`: Model architecture name
- `model_config`: Hyperparameters (hidden_size, dropout, etc.)
- `scaler_filename`: Path to the scaler file
- `model_filename`: Path to the trained model checkpoint

**Output**:
The script generates a Parquet file containing:
- Node IDs and metadata
- Predicted task probabilities for all 54 task classes

#### Performance Results

All results are averaged over 10 random seeds with mean ± standard deviation reported.

| Model | Loss | Features | Val Micro-F1 | Test Micro-F1 | Test Macro-F1 | Test PR-AUC | Test Head F1 (Top 10) | Test Tail F1 (Rest) | Test Gap |
|-------|------|----------|--------------|---------------|---------------|-------------|------------------|----------------|-----|
| GAT | BCE | BGE Only (768) | 0.8939 ± 0.0019 | 0.5526 ± 0.0529 | 0.2187 ± 0.0115 | 0.4052 ± 0.0166 | 0.5726 | 0.1471 | 0.4256 |
| GAT | BCE | BGE + BM25 (822) | 0.8938 ± 0.0016 | 0.5613 ± 0.0691 | 0.2383 ± 0.0149 | 0.4391 ± 0.0218 | 0.6533 | 0.1663 | 0.4870 |
| GAT | Focal | BGE + BM25 (822) | 0.8824 ± 0.0044 | 0.6086 ± 0.0146 | 0.1354 ± 0.0119 | 0.4968 ± 0.0234 | 0.4952 | 0.0600 | 0.4353 |
| GCN | BCE | BGE Only (768) | 0.8892 ± 0.0005 | 0.4844 ± 0.0042 | 0.1368 ± 0.0101 | 0.3600 ± 0.0094 | 0.4870 | 0.0791 | 0.4079 |
| GCN | BCE | BGE + BM25 (822) | 0.8905 ± 0.0022 | 0.4832 ± 0.0023 | 0.1665 ± 0.0122 | 0.3954 ± 0.0081 | 0.5147 | 0.0857 | 0.4290 |
| GCN | Focal | BGE + BM25 (822) | 0.8748 ± 0.0098 | 0.4733 ± 0.0253 | 0.1787 ± 0.0157 | 0.4424 ± 0.0195 | 0.4623 | 0.1404 | 0.3218 |
| SAGE | BCE | BGE Only (768) | 0.9239 ± 0.0009 | 0.5753 ± 0.0250 | 0.1961 ± 0.0052 | 0.4835 ± 0.0137 | 0.5963 | 0.1078 | 0.4885 |
| SAGE | BCE | BGE + BM25 (822) | 0.9273 ± 0.0009 | 0.5834 ± 0.0254 | 0.2026 ± 0.0038 | 0.5499 ± 0.0064 | 0.6282 | 0.0993 | 0.5289 |
| SAGE | Focal | BGE + BM25 (822) | 0.9152 ± 0.0007 | 0.5923 ± 0.0490 | 0.1673 ± 0.0124 | 0.5219 ± 0.0173 | 0.5740 | 0.0544 | 0.5196 |
| TRANSFORMER | BCE | BGE Only (768) | 0.9229 ± 0.0008 | 0.5783 ± 0.0147 | 0.1914 ± 0.0056 | 0.4740 ± 0.0088 | 0.6380 | 0.0784 | 0.5596 |
| TRANSFORMER | BCE | BGE + BM25 (822) | 0.9265 ± 0.0008 | 0.6302 ± 0.0483 | 0.2101 ± 0.0047 | 0.5789 ± 0.0130 | 0.6137 | 0.1107 | 0.5030 |
| TRANSFORMER | Focal | BGE + BM25 (822) | 0.9261 ± 0.0005 | **0.7174 ± 0.0037** | **0.2608 ± 0.0131** | **0.6791 ± 0.0068** | 0.6624 | 0.1636 | 0.4989 |
| GATV2 | BCE | BGE Only (768) | 0.9170 ± 0.0019 | 0.7065 ± 0.0077 | 0.1749 ± 0.0041 | 0.4146 ± 0.0256 | 0.6623 | 0.0610 | 0.6014 |
| GATV2 | BCE | BGE + BM25 (822) | 0.9191 ± 0.0011 | **0.7124 ± 0.0051** | 0.1945 ± 0.0089 | 0.4753 ± 0.0111 | 0.6469 | 0.0738 | 0.5731 |
| GATV2 | Focal | BGE + BM25 (822) | 0.9212 ± 0.0015 | 0.7085 ± 0.0082 | 0.2140 ± 0.0168 | 0.6290 ± 0.0226 | 0.6504 | 0.1004 | 0.5500 |

#### Performance Visualizations

![Micro-F1 Comparison](../../data/micro_f1_comparison.png)

*Comparison of Test Micro-F1 scores across different models and configurations*

![PR-AUC Ranking](../../data/pr_auc_ranking.png)

*Ranking of models by Test PR-AUC scores*

#### Key Findings

1. **Best Test Micro-F1**: TRANSFORMER with Focal Loss achieves **0.7174 ± 0.0037** (highest overall)
2. **Best Test PR-AUC**: TRANSFORMER with Focal Loss achieves **0.6791 ± 0.0068**
3. **Best Test Macro-F1**: TRANSFORMER with Focal Loss achieves **0.2608 ± 0.0131**
4. **Most Stable Test Micro-F1**: GATV2 with BCE Loss achieves **0.7124 ± 0.0051** (lowest variance)
5. **BM25 Impact**: Adding BM25 features generally improves performance across most models (compare BGE-only vs BGE+BM25 variants)
6. **Focal Loss**: Generally improves PR-AUC and Macro-F1, and can improve Micro-F1 for TRANSFORMER and GAT models
7. **Long-tail Performance**: Head F1 (top 10 tasks) is consistently higher than Tail F1 (remaining tasks), indicating class imbalance challenges

**Note**: Head F1 refers to performance on the top 10 most frequent task classes, while Tail F1 refers to the remaining task classes. The Gap metric measures the performance difference between head and tail classes.

### GRetriever (LLM + Graph)

We release three GRetriever checkpoints on Hugging Face, each combining a different GNN encoder with Qwen2.5-3B-Instruct:

- **SAGE-based GRetriever**: [`Chloe/gretriever-HFKG_SAGE`](https://huggingface.co/Chloe/gretriever-HFKG_SAGE)
- **GAT-based GRetriever**: [`Chloe/gretriever-HFKG_GAT`](https://huggingface.co/Chloe/gretriever-HFKG_GAT)
- **GATv2-based GRetriever**: [`Chloe/gretriever-HFKG_GATV2`](https://huggingface.co/Chloe/gretriever-HFKG_GATV2)

All three models use **BGE + BM25 (822-dim)** node features and are trained for **multi-label task classification** over 54 tasks on the Hugging Face KG.

#### Evaluation

To evaluate a GRetriever model using `g_retrieval_eval_w_labels_qwen.py`, you can either use a **local checkpoint directory** or download directly from **Hugging Face**.

**Option 1: Using a local checkpoint directory**

If you've already downloaded a checkpoint folder locally:

```bash
python g_retrieval_eval_w_labels_qwen.py \
  --checkpoint_path ./gretriever_ckpts/gretriever-HFKG_GATV2 \
  --model GATV2 \
  --ckpt final \
  --split test
```

**Option 2: Downloading from Hugging Face automatically**

The script can automatically download and use a checkpoint from Hugging Face:

```bash
export HF_TOKEN=your_huggingface_token

python g_retrieval_eval_w_labels_qwen.py \
  --checkpoint_path Chloe/gretriever-HFKG_GATV2 \
  --hf_dir /path/to/download/gretriever-HFKG_GATV2 \
  --model GATV2 \
  --ckpt final \
  --split test
```

**Arguments:**
- `--checkpoint_path`: Either a local directory path or a Hugging Face repo ID (e.g., `Chloe/gretriever-HFKG_GATV2`)
- `--hf_dir`: Required when using a Hugging Face repo ID. Specifies where to download the checkpoint.
- `--model`: Model type (`GAT`, `GATV2`, or `SAGE`) - used for GNN architecture selection
- `--ckpt`: Checkpoint name within the directory (usually `final`)
- `--split`: Evaluation split (`val` or `test`)

**Validation and test performance:**

| Model | Features | Val Micro-F1 | Test Micro-F1 | Test Macro-F1 |
|-------|----------|--------------|---------------|---------------|
| SAGE + Qwen2.5-3B-Instruct | BGE + BM25 (822) | 0.5817 | 0.3238 | 0.1159 |
| GAT + Mistral 7B | BGE + BM25 (822) | 0.6457 | 0.3571 | 0.1942 |
| GAT + Qwen2.5-3B-Instruct | BGE + BM25 (822) | 0.7719 | 0.4104 | 0.1964 |
| **GATV2 + Qwen2.5-3B-Instruct** | BGE + BM25 (822) | **0.8497** | **0.4515** | **0.1369** |

![Generative Models - Comparison](../../data/comparison_mistral_vs_qwen.png)

### Evaluation Scripts

After training models, use the evaluation scripts to analyze performance:

```bash
# Evaluate fine-tuned LLM models
python eval.py

# Examine evaluation results
python examine_eval.py

# Generate evaluation visualizations
python plot_eval.py
```

**Evaluation Scripts**:
- **`eval.py`**: Evaluates fine-tuned LLM models on task classification, generates predictions and metrics
- **`examine_eval.py`**: Analyzes evaluation results from JSON files, prints detailed classification reports
- **`plot_eval.py`**: Creates visualizations (F1 vs frequency scatter plots, confusion matrices) from evaluation results

**Final Eval**

Comparing select Generative and Pure-GNN models:

![Comparison](../../data/full_model_comparison.png)