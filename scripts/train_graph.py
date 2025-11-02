import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import argparse
from torch_geometric.data import Data
import joblib
import torch.nn.functional as F

# --- Model-specific Imports ---
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv

class GCN(nn.Module):
    """ Graph Convolutional Network (GCN) """
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
    """ Graph Attention Network (GAT) """
    def __init__(self, in_feats, hidden_size, out_feats, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hidden_size, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * heads, out_feats, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.elu(h)
        # Dropout is already included in GATConv layers
        h = self.conv2(h, edge_index)
        return h

class SAGE(nn.Module):
    """ GraphSAGE (SAGEConv) """
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
    """ Graph Transformer (TransformerConv) """
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_feats, hidden_size)
        self.conv2 = TransformerConv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Pass edge_attr (which can be None)
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        return h

def fix_graph_data(graph):
    """Convert CogDL (or similar) graph to PyTorch Geometric Data format"""
    
    x = graph.x  # Node features
    y = graph.y  # Node labels (multi-label)
    
    # Extract edge information and ensure it's in the correct format
    edge_index = torch.stack(list(graph.edge_index), dim=0)
    
    # Extract masks
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    
    # --- MODIFICATION ---
    # Handle edge_attr if it exists, otherwise set to None
    edge_attr = getattr(graph, 'edge_attr', None)
    
    # Create PyTorch Geometric Data object
    data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_attr=edge_attr  # Add edge_attr
    )
    
    return data

def evaluate(model, graph, mask, criterion, model_type):
    """
    Evaluates the model on a given data mask.
    Handles different model forward pass requirements.
    """
    model.eval()
    with torch.no_grad():
        # --- NEW: Conditional forward pass based on model type ---
        if model_type == 'transformer':
            logits = model(graph.x, graph.edge_index, graph.edge_attr)
        else:
            logits = model(graph.x, graph.edge_index)
        
        eval_logits = logits[mask]
        eval_labels = graph.y[mask].float()
        
        loss = criterion(eval_logits, eval_labels)
        
        probs = torch.sigmoid(eval_logits)
        preds = (probs > 0.5).int()
        
        f1 = f1_score(
                eval_labels.cpu().numpy(), 
                preds.cpu().numpy(), 
                average="micro", 
                zero_division=0
        )
            
    return loss.item(), f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GNN Training for Multi-Label Classification")
    
    # Model selection argument
    parser.add_argument(
            '--model_type', 
            type=str, 
            required=True, 
            choices=['gcn', 'gat', 'sage', 'transformer'],
            help="Type of GNN model to train."
    )
    
    parser.add_argument('--graph_path', type=str, required=True, help="Path to the final_graph.pt file")
    parser.add_argument('--save_path', type=str, default='best_model.pt', help="Path to save the best trained model")
    parser.add_argument('--scaler_path', type=str, default='scaler.gz', help="Path to save the fitted scaler")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected Model: {args.model_type.upper()}")

    print(f"Loading graph from {args.graph_path}...")
    # Allow loading pickled data (which StandardScaler is)
    data = torch.load(args.graph_path, weights_only=False) 

    # Model-specific preprocessing
    if args.model_type == 'gat':
        print("Applying L2 normalization to node features (for GAT)...")
        # GAT script used fix_graph_data first
        data = fix_graph_data(data)
        data.x = F.normalize(data.x, p=2, dim=1)
    else:
        # GCN, SAGE, Transformer used StandardScaler
        print("Scaling node features (for GCN/SAGE/Transformer)...")
        scaler = StandardScaler()
        scaler.fit(data.x[data.train_mask])
        joblib.dump(scaler, args.scaler_path)
        print(f"Scaler saved to {args.scaler_path}")
        
        data.x = torch.from_numpy(scaler.transform(data.x)).float()
        data = fix_graph_data(data)

    
    print("Moving preprocessed data to GPU...")
    data = data.to(device)

    #  Model Instantiation Logic
    model = None
    if args.model_type == 'gcn':
        model = GCN(
            in_feats=data.x.size(1),
            hidden_size=256,
            out_feats=data.y.size(1),
            dropout=0.5)
    elif args.model_type == 'gat':
        model = GAT(
            in_feats=data.x.size(1),
            hidden_size=32,  
            out_feats=data.y.size(1),
            heads=8,
            dropout=0.5)
    elif args.model_type == 'sage':
        model = SAGE(
            in_feats=data.x.size(1),
            hidden_size=256,
            out_feats=data.y.size(1),
            dropout=0.5)
    elif args.model_type == 'transformer':
        model = GraphTransformer(
            in_feats=data.x.size(1),
            hidden_size=256,
            out_feats=data.y.size(1),
            dropout=0.5)

    model = model.to(device)
    print(f"\nModel Architecture ({args.model_type.upper()}):")
    print(model)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    
    epochs = 500
    patience = 100
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # --- NEW: Conditional forward pass ---
        if args.model_type == 'transformer':
                logits = model(data.x, data.edge_index, data.edge_attr)
        else:
                logits = model(data.x, data.edge_index)
        
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        
        loss.backward()
        optimizer.step()
        
        # Pass model_type to evaluate function
        val_loss, val_f1 = evaluate(model, data, data.val_mask, criterion, args.model_type)
        
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Micro-F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"✨ New best validation F1: {best_val_f1:.4f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print(f"Stopping early after {patience} epochs with no improvement.")
                    break
    
    print("\nTraining finished!")
    print(f"Best Validation Micro-F1 Score: {best_val_f1:.4f}")

    if best_model_state:
        print("\nLoading best model and evaluating on the test set...")
        model.load_state_dict(best_model_state)
        
        # Pass model_type to evaluate function
        test_loss, test_f1 = evaluate(model, data, data.test_mask, criterion, args.model_type)
        
        print("-----------------------------------------")
        print(f"Final Test Results ({args.model_type.upper()}):")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - Test Micro-F1: {test_f1:.4f}")
        print("-----------------------------------------")
        
        torch.save(best_model_state, args.save_path)
        print(f"Best model saved to {args.save_path}")
        
# CUDA_VISIBLE_DEVICES=4 python train_graph.py --model_type gcn --graph_path ./experiment_runs/run_2025-10-02_14-47-01/final_graph.pt --save_path ../notebooks/trained_gcn.pt