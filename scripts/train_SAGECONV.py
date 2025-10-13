import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import argparse
from torch_geometric.data import Data
import joblib
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

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

def evaluate(model, graph, mask, criterion):
    """Evaluates the model on a given data mask."""
    model.eval()
    with torch.no_grad():
        # **UPDATED: Pass both x and edge_index**
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


def fix_graph_data(graph):
    """Convert CogDL graph to PyTorch Geometric Data format"""
    
    # Extract features and labels
    x = graph.x  # Node features
    y = graph.y  # Node labels (multi-label)
    
    # Extract edge information and ensure it's in the correct format
    
    edge_index = torch.stack(list(graph.edge_index), dim=0)
    
    # Extract masks
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    
    print(f"Converted edge_index shape: {edge_index.shape}")
    print(f"Converted edge_index dtype: {edge_index.dtype}")
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch SAGECONV Training for Multi-Label Classification")
    parser.add_argument('--graph_path', type=str, required=True, help="Path to the final_graph.pt file")
    parser.add_argument('--save_path', type=str, default='best_model.pt', help="Path to save the best trained model")
    parser.add_argument('--scaler_path', type=str, default='scaler.gz', help="Path to save the fitted scaler")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading graph from {args.graph_path}...")
    data = torch.load(args.graph_path, weights_only=False)
    # 1. Perform all CPU-based operations FIRST.
    print("Scaling node features...")
    # # import pdb; pdb.set_trace()
    scaler = StandardScaler()
    scaler.fit(data.x[data.train_mask])
    joblib.dump(scaler, args.scaler_path)
    print(f"Scaler saved to {args.scaler_path}")
    
    data.x = torch.from_numpy(scaler.transform(data.x)).float()
    # import pdb; pdb.set_trace()
    # print("Applying L2 normalization to node features...")
    # data.x = F.normalize(data.x, p=2, dim=1) # Add this line

    data = fix_graph_data(data)

    
    # # 2. Correctly handle the edge_index format (also on CPU).
    # # This block robustly handles both tuple and tensor formats.
    # print("Correcting edge_index format to torch.long...")
    # if isinstance(data.edge_index, tuple):
    #     # If it's a tuple like (sources, targets), stack them
    #     data.edge_index = torch.stack([data.edge_index[0], data.edge_index[1]], dim=0).long()
    # else:
    #     # Otherwise, just ensure the existing tensor is long
    #     data.edge_index = data.edge_index.long()
    # import pdb; pdb.set_trace()
    # 3. NOW, after all preprocessing, move the data to the GPU.
    print("Moving preprocessed data to GPU...")
    data = data.to(device)

    model = SAGE(
        in_feats=data.x.size(1),
        hidden_size=256,
        out_feats=data.y.size(1),
        dropout=0.5
    ).to(device)

    print("\nModel Architecture (True SAGECONV):")
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
        
        # **UPDATED: The forward pass now requires x and edge_index**
        logits = model(data.x, data.edge_index)
        
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        
        loss.backward()
        optimizer.step()
        
        val_loss, val_f1 = evaluate(model, data, data.val_mask, criterion)
        
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
    print(f"🏆 Best Validation Micro-F1 Score: {best_val_f1:.4f}")

    if best_model_state:
        print("\nLoading best model and evaluating on the test set...")
        model.load_state_dict(best_model_state)
        test_loss, test_f1 = evaluate(model, data, data.test_mask, criterion)
        print("-----------------------------------------")
        print(f"Final Test Results:")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - Test Micro-F1: {test_f1:.4f}")
        print("-----------------------------------------")
        
        torch.save(best_model_state, args.save_path)
        print(f"Best model saved to {args.save_path}")

