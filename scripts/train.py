import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score
import argparse
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from model_utils import GCN, GAT, SAGE, GraphTransformer, GATv2
from utils import initialize_bias, analyze_long_tail_performance, fix_graph_data
import joblib

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

def evaluate(model, graph, mask, criterion, model_type, thresholds=None):
    model.eval()
    with torch.no_grad():
        if model_type in ['transformer', 'gatv2']:
            logits = model(graph.x, graph.edge_index, graph.edge_attr)
        else:
            logits = model(graph.x, graph.edge_index)
        
        eval_logits = logits[mask]
        eval_labels = graph.y[mask].float()
        
        loss = criterion(eval_logits, eval_labels)
        
        probs = torch.sigmoid(eval_logits)

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



# --- Training Logic ---
def train_session(seed, args, device, data):
    print(f"\n--- Starting Run with Seed {seed} ---")
    set_seed(seed)
    
    # Model Init
    if args.model_type == 'gcn':
        model = GCN(data.x.size(1), args.hidden_size, data.y.size(1), args.dropout)
    elif args.model_type == 'gat':
        model = GAT(data.x.size(1), args.hidden_size, data.y.size(1), args.dropout)
    elif args.model_type == 'sage':
        model = SAGE(data.x.size(1), args.hidden_size, data.y.size(1), args.dropout)
    elif args.model_type == 'transformer':
        model = GraphTransformer(data.x.size(1), args.hidden_size, data.y.size(1), args.dropout)
    elif args.model_type == 'gatv2':
        model = GATv2(data.x.size(1), args.hidden_size, data.y.size(1), args.dropout)

    model = model.to(device)

    # --- INSERT THIS BLOCK HERE ---
    if args.use_focal:
        # Only initialize bias if using Focal Loss, as standard BCE doesn't strictly require it
        initialize_bias(model)
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
        
        if args.model_type in ['transformer', 'gatv2']:
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
    
    # 3. Evaluate on Test Set using Standard (0.5) Thresholds
    test_loss_std, test_micro_std, test_macro_std, test_prauc_std, test_per_class_f1_std, _, _ = evaluate(
        model, data, data.test_mask, criterion, args.model_type, thresholds=None
    )

    print(f"Seed {seed} [VAL] Macro: {val_macro:.4f} | Micro: {val_micro:.4f} | "
      f"[TEST] Macro: {test_macro_std:.4f} | Micro: {test_micro_std:.4f} | "
      f"PR-AUC: {test_prauc_std:.4f} | Loss: {test_loss_std:.4f}")
        
    res = {
        # --- VALIDATION METRICS (Standard 0.5 Threshold) ---
        "val_macro_f1": val_macro,
        "val_micro_f1": val_micro,
        "val_pr_auc":   val_prauc,
        # --- TEST METRICS (Standard 0.5 Threshold) ---
        "test_macro_std": test_macro_std,
        "test_micro_std": test_micro_std,
        "test_prauc_std": test_prauc_std,
        "test_loss_std": test_loss_std,  # Add this
    }
        
    return res, best_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn', 'gat', 'sage', 'transformer', 'gatv2'])
    parser.add_argument('--graph_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_focal', action='store_true', help="Use Focal Loss instead of BCE")
    parser.add_argument('--exclude_bm25', action='store_true', help="Drop the last 54 BM25 features, using only the first 768 BGE embeddings.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    SEEDS = [42, 100, 2023, 123, 999, 41, 99, 2022, 122, 998]

    # 1. Load Data ONCE
    print(f"Loading data from {args.graph_path}...")
    raw_data = torch.load(args.graph_path, weights_only=False)

    results_list = []
    os.makedirs(args.save_dir, exist_ok=True)

    # --- FEATURE TOGGLE LOGIC ---
    if args.exclude_bm25:
        # Keep only the first 768 columns (BGE), drop the rest (BM25)
        # Check to ensure we don't slice if it's already sliced (safety check)
        if raw_data.x.shape[1] > 768:
            raw_data.x = data.x[:, :768]
    
    scaler = StandardScaler()
    scaler.fit(raw_data.x[raw_data.train_mask])
    raw_data.x = torch.from_numpy(scaler.transform(raw_data.x)).float()
    raw_data = fix_graph_data(raw_data)

    # 2. Single Loop to Run and Collect
    for seed in SEEDS:
        # Clone to ensure thread safety / no leakage between runs
        # (Though we re-process inside, this is a good safety habit)
        current_data = raw_data.clone()
        current_data = current_data.to(device)
        
        # Pass the pre-loaded data into the function

        result, best_state = train_session(seed, args, device, data=current_data)
       

        # Save visualization for seed 42
        if seed == 42:
            torch.save(best_state, os.path.join(args.save_dir, 'model.pt'))
            joblib.dump(scaler, os.path.join(args.save_dir, 'scaler.pkl'))
            save_path = os.path.join(args.save_dir, f"smoking_gun_{args.model_type}{'_FOCAL' if args.use_focal else ''}.png")
            analyze_long_tail_performance(data.y[data.train_mask], test_per_class_f1_std, save_path=save_path)
            
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