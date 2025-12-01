"""
Shared GNN Model Definitions

This module contains all Graph Neural Network architectures used across
both pure GNN training and GRetriever (LLM + Graph) approaches.
"""

from torch_geometric.nn import GraphSAGE, GAT, GCNConv, GATConv, SAGEConv, TransformerConv, GATv2Conv
import torch.nn as nn

class GraphEncoderSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=1)
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=1)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class GraphEncoderGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.conv1 = GAT(in_channels, hidden_channels, num_layers=1, edge_dim=edge_dim)
        self.conv2 = GAT(hidden_channels, out_channels, num_layers=1, edge_dim=edge_dim)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class GraphEncoderGATConv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class GATv2(nn.Module):
    """ Graph Attention Network v2 """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(in_feats, hidden_size)
        self.conv2 = GATv2Conv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        return h

class GCN(nn.Module):
    """ Graph Convolutional Network (GCN) """
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
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

class GATKG(nn.Module):
    """ Graph Attention Network (GAT) """
    def __init__(self, in_feats, hidden_size, out_feats, dropout, heads=8):
        super(GATKG, self).__init__()
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
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
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
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
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