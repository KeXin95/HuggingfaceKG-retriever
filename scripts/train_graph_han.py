import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score
import argparse
from torch_geometric.data import Data
import joblib
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_curve

def tune_thresholds(y_true, y_probs):
    """
    Finds a SINGLE threshold for ALL classes to maximize Micro-F1.
    """
    # Flatten everything to treat it as one giant binary classification problem
    y_true_flat = y_true.flatten()
    y_probs_flat = y_probs.flatten()
    
    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_probs_flat)
    
    # Calculate F1 for all global thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return best_threshold


# --- Set Random Seed ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculate Standard BCE Loss (this is -log(pt))
        # reduction='none' is crucial so we can weight per-element
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 2. Calculate pt (probability of the true class)
        # Since bce_loss = -log(pt), then pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)
        
        # 3. Calculate Alpha Factor
        # If target=1, use alpha. If target=0, use (1-alpha)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # 4. Calculate Focal Loss
        # Formula: alpha * (1-pt)^gamma * BCE
        focal_loss = alpha_factor * ((1 - pt) ** self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Model Classes (Unchanged) ---
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hidden_size, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * heads, out_feats, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.elu(h)
        h = self.conv2(h, edge_index)
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

class GraphTransformer(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_feats, hidden_size, edge_dim=1)
        self.conv2 = TransformerConv(hidden_size, out_feats, edge_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1).float()
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        return h

# --- Helper Functions ---
def fix_graph_data(graph):
    data = Data(
            x=graph.x,
            edge_index=torch.stack(list(graph.edge_index), dim=0),
            y=graph.y,
            train_mask=graph.train_mask,
            val_mask=graph.val_mask,
            test_mask=graph.test_mask,
            edge_attr=getattr(graph, 'edge_attr', None)
    )
    return data

def evaluate(model, graph, mask, criterion, model_type, thresholds=None):
    model.eval()
    with torch.no_grad():
        if model_type == 'transformer':
            logits = model(graph.x, graph.edge_index, graph.edge_attr)
        else:
            logits = model(graph.x, graph.edge_index)
        
        eval_logits = logits[mask]
        eval_labels = graph.y[mask].float()
        
        loss = criterion(eval_logits, eval_labels)
        
        probs = torch.sigmoid(eval_logits)

        # --- THRESHOLD LOGIC ---
        if thresholds is not None:
            # If custom thresholds provided, use them per-class
            # specific threshold for specific column
            thresholds_tensor = torch.tensor(thresholds, device=probs.device).float()
            preds = (probs > thresholds_tensor).int()
        else:
            # Default standard 0.5 cutoff
            preds = (probs > 0.5).int()
        # -----------------------

        y_true = eval_labels.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # --- THE FIX STARTS HERE ---
        # 1. Identify classes that have at least one positive instance in this batch
        valid_classes = y_true.sum(axis=0) > 0
        
        # 2. If valid_classes is empty (rare edge case), return 0
        if valid_classes.sum() == 0:
            pr_auc = 0.0
        else:
            # 3. Only calculate AUC for the columns where valid_classes is True
            # This ignores the "Ghost Classes" so they don't crash the metric
            pr_auc = average_precision_score(
                y_true[:, valid_classes], 
                y_probs[:, valid_classes], 
                average="macro"
            )
        # --- THE FIX ENDS HERE ---
        
    return loss.item(), micro_f1, macro_f1, pr_auc, per_class_f1, y_true, y_probs

def initialize_bias(model):
    print("\n--- Initializing Bias ---")
    try:
        # 1. Grab the final layer
        last_layer = model.conv2
        
        pi = 0.01
        bias_value = -np.log((1 - pi) / pi)
        
        # 2. Check for Standard Bias (GCN, SAGE, Linear)
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            torch.nn.init.constant_(last_layer.bias, bias_value)
            print(f"✅ Initialized standard bias to {bias_value:.4f}")
            
        # 3. Check for Transformer Bias (Hidden in lin_skip)
        elif isinstance(last_layer, TransformerConv):
            if hasattr(last_layer, 'lin_skip') and last_layer.lin_skip is not None:
                torch.nn.init.constant_(last_layer.lin_skip.bias, bias_value)
                print(f"✅ Initialized Transformer lin_skip bias to {bias_value:.4f}")
            else:
                print("⚠️ WARNING: TransformerConv has no lin_skip (root_weight=False?). Cannot init bias.")
                
        # 4. Check for GAT Bias (Often None, but check just in case)
        elif isinstance(last_layer, GATConv):
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                 torch.nn.init.constant_(last_layer.bias, bias_value)
                 print(f"✅ Initialized GAT bias to {bias_value:.4f}")
            else:
                print("ℹ️ Note: GATConv usually has no bias by default. Skipping.")

        else:
            print(f"⚠️ WARNING: Could not find a bias parameter in {type(last_layer).__name__}.")

    except AttributeError:
        print("⚠️ WARNING: Model does not have attribute 'conv2'.")
    print("-------------------------\n")

def analyze_long_tail_performance(train_y, test_per_class_f1, save_path="f1_vs_support.png"):
    """
    Generates the 'Smoking Gun' visualization and Head/Tail stats.
    train_y: The labels from the TRAINING set (to determine frequency).
    test_per_class_f1: The F1 scores from the TEST set (to determine performance).
    """
    print("\n=== Long-Tail Distribution Analysis ===")
    
    # 1. Calculate Support (Frequency) per class in the Training Set
    # Assuming train_y is [num_nodes, num_classes]
    class_counts = train_y.sum(axis=0).cpu().numpy() # Shape: [num_classes]
    
    # 2. Sort classes by frequency (Most frequent first)
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_counts = class_counts[sorted_indices]
    sorted_f1 = test_per_class_f1[sorted_indices]
    
    # 3. Define Head (Top 10) and Tail (The rest)
    head_indices = sorted_indices[:10]
    tail_indices = sorted_indices[10:]
    
    head_f1_avg = np.mean(test_per_class_f1[head_indices])
    tail_f1_avg = np.mean(test_per_class_f1[tail_indices])
    
    print(f"Head Labels (Top 10 freq): Avg F1 = {head_f1_avg:.4f}")
    print(f"Tail Labels (Bottom {len(tail_indices)}):   Avg F1 = {tail_f1_avg:.4f}")
    print(f"Gap (Head - Tail):         {head_f1_avg - tail_f1_avg:.4f}")
    
    # 4. The 'Smoking Gun' Plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(class_counts, test_per_class_f1, alpha=0.6, c='blue', edgecolors='w', s=80)
    
    # Log scale usually looks better for long-tail counts
    plt.xscale('log') 
    
    plt.title("The 'Smoking Gun': Test F1 Score vs. Training Class Frequency")
    plt.xlabel("Number of Training Samples (Log Scale)")
    plt.ylabel("Test F1 Score")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=head_f1_avg, color='g', linestyle='--', label=f'Head Avg F1: {head_f1_avg:.2f}')
    plt.axhline(y=tail_f1_avg, color='r', linestyle='--', label=f'Tail Avg F1: {tail_f1_avg:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    print("========================================\n")


# --- Training Logic ---
def train_session(seed, args, device, data):
    print(f"\n--- Starting Run with Seed {seed} ---")
    set_seed(seed)

    # --- FEATURE TOGGLE LOGIC ---
    if args.exclude_bm25:
        # Keep only the first 768 columns (BGE), drop the rest (BM25)
        # Check to ensure we don't slice if it's already sliced (safety check)
        if data.x.shape[1] > 768:
            data.x = data.x[:, :768]

    # ----------------------------
    
    # Reload data
    # data = torch.load(args.graph_path, weights_only=False)

    # Preprocessing
    if args.model_type == 'gat':
        data = fix_graph_data(data)
        data.x = F.normalize(data.x, p=2, dim=1)
    else:
        scaler = StandardScaler()
        scaler.fit(data.x[data.train_mask])
        data.x = torch.from_numpy(scaler.transform(data.x)).float()
        data = fix_graph_data(data)

    data = data.to(device)

    # Model Init
    if args.model_type == 'gcn':
        model = GCN(data.x.size(1), 256, data.y.size(1))
    elif args.model_type == 'gat':
        model = GAT(data.x.size(1), 32, data.y.size(1))
    elif args.model_type == 'sage':
        model = SAGE(data.x.size(1), 256, data.y.size(1))
    elif args.model_type == 'transformer':
        model = GraphTransformer(data.x.size(1), 256, data.y.size(1))

    model = model.to(device)

    # --- INSERT THIS BLOCK HERE ---
    if args.use_focal:
        # Only initialize bias if using Focal Loss, as standard BCE doesn't strictly require it
        initialize_bias(model)
    # ------------------------------
    
    # Loss Selection
    if args.use_focal:
        print("Using Focal Loss")
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    
    # Training Loop
    best_val_metric = 0 # monitoring macro-f1 for early stopping
    patience = 100
    patience_counter = 0
    best_state = None

    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        
        if args.model_type == 'transformer':
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
        
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            _, _, val_macro, _, _, _, _ = evaluate(model, data, data.val_mask, criterion, args.model_type)
            
            if val_macro > best_val_metric:
                best_val_metric = val_macro
                patience_counter = 0
                best_state = model.state_dict()
            else:
                # CRITICAL FIX: Increment by 10 (the stride), not 1
                patience_counter += 10 
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
# --- FINAL EVALUATION & THRESHOLD TUNING ---
    model.load_state_dict(best_state)
    
    # 1. Evaluate on Validation Set
    # CHANGED: We now capture val_micro, val_macro, val_prauc instead of using underscores (_)
    _, val_micro, val_macro, val_prauc, _, val_true, val_probs = evaluate(
        model, data, data.val_mask, criterion, args.model_type
    )
    
    # 2. Calculate Optimal Thresholds based on Validation Data
    optimal_thresholds = tune_thresholds(val_true, val_probs)
    
    # 3. Evaluate on Test Set using Standard (0.5) Thresholds
    test_loss_std, test_micro_std, test_macro_std, test_prauc_std, test_per_class_f1_std, _, _ = evaluate(
        model, data, data.test_mask, criterion, args.model_type, thresholds=None
    )

    # 4. Evaluate on Test Set using Optimized Thresholds
    test_loss_opt, test_micro_opt, test_macro_opt, test_prauc_opt, test_per_class_f1_opt, _, _ = evaluate(
        model, data, data.test_mask, criterion, args.model_type, thresholds=optimal_thresholds
    )

    print(f"Seed {seed} [VAL] Macro: {val_macro:.4f} | Micro: {val_micro:.4f} | "
      f"[TEST] Macro: {test_macro_std:.4f} | Micro: {test_micro_std:.4f} | "
      f"PR-AUC: {test_prauc_std:.4f} | Loss: {test_loss_std:.4f}")
    
    # Save visualization for seed 42
    if seed == 42:
        torch.save(best_state, args.save_path)
        save_path = f"smoking_gun_{args.model_type}{'_FOCAL' if args.use_focal else ''}.png"
        analyze_long_tail_performance(data.y[data.train_mask], test_per_class_f1_opt, save_path=save_path)
        
    return {
        # --- VALIDATION METRICS (Standard 0.5 Threshold) ---
        "val_macro_f1": val_macro,
        "val_micro_f1": val_micro,
        "val_pr_auc":   val_prauc,

        # --- TEST METRICS (Optimized Threshold) ---
        "test_macro_opt": test_macro_opt,
        "test_micro_opt": test_micro_opt,
        
        # --- TEST METRICS (Standard 0.5 Threshold) ---
        "test_macro_std": test_macro_std,
        "test_micro_std": test_micro_std,
        "test_prauc_std": test_prauc_std,
        "test_loss_std": test_loss_std,  # Add this
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn', 'gat', 'sage', 'transformer'])
    parser.add_argument('--graph_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='best_model.pt')
    parser.add_argument('--scaler_path', type=str, default='scaler.gz') # kept for compatibility
    parser.add_argument('--use_focal', action='store_true', help="Use Focal Loss instead of BCE")
    parser.add_argument('--exclude_bm25', action='store_true', help="Drop the last 54 BM25 features, using only the first 768 BGE embeddings.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    SEEDS = [42, 100, 2023, 123, 999] #, 41, 99, 2022, 122, 998]

    # 1. Load Data ONCE
    print(f"Loading data from {args.graph_path}...")
    raw_data = torch.load(args.graph_path, weights_only=False)

    results_list = []

    # 2. Single Loop to Run and Collect
    for seed in SEEDS:
        # Clone to ensure thread safety / no leakage between runs
        # (Though we re-process inside, this is a good safety habit)
        current_data = raw_data.clone()
        
        # Pass the pre-loaded data into the function
        result = train_session(seed, args, device, data=current_data)
        results_list.append(result)

    # 3. Aggregate
    agg_results = defaultdict(list)
    for res in results_list:
        for k, v in res.items():
            agg_results[k].append(v)

    print("\n" + "="*40)
    print(f"Final Aggregated Results ({len(SEEDS)} runs)")
    print(f"Model: {args.model_type.upper()} | Loss: {'Focal' if args.use_focal else 'BCE'}")
    print(f"Features: {'BGE Only (768)' if args.exclude_bm25 else 'BGE + BM25 (822)'}")
    print("-" * 40)
    
    # Sort keys to keep Val and Test grouped together in the output
    for metric in sorted(agg_results.keys()):
        values = agg_results[metric]
        print(f"{metric.upper().ljust(20)}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("="*40)