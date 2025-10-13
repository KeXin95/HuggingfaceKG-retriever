#!/usr/bin/env python3
"""
EDA Analysis Script for HuggingfaceKG Dataset

This script performs comprehensive exploratory data analysis on the HuggingfaceKG dataset,
including label distribution analysis, graph visualization, node feature analysis,
and edge connectivity analysis.

Usage:
    python eda_analysis.py --dataset_path /path/to/dataset.pt --output_dir /path/to/output
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import argparse
import os
import warnings
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from .utils import fix_graph_structure

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_task_mappings(task_mapping_path):
    """
    Load task to index mappings from JSON file.
    
    Args:
        task_mapping_path: Path to task_to_idx.json file
        
    Returns:
        tuple: (task_to_idx, idx_to_task) dictionaries
    """
    try:
        with open(task_mapping_path, 'r') as f:
            task_to_idx = json.load(f)
        idx_to_task = {v: k for k, v in task_to_idx.items()}
        print(f"Loaded {len(task_to_idx)} task labels from {task_mapping_path}")
        return task_to_idx, idx_to_task
    except FileNotFoundError:
        print(f"Task mapping file not found: {task_mapping_path}")
        print("Using numeric indices instead")
        return {}, {}


def analyze_label_distribution(data, idx_to_task=None, output_dir=None):
    """
    Perform comprehensive label distribution analysis.
    
    Args:
        data: PyTorch Geometric Data object
        idx_to_task: Dictionary mapping label indices to task names
        output_dir: Directory to save plots and results
    """
    print("=" * 60)
    print("📊 LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Convert labels to numpy for analysis
    labels = data.y.cpu().numpy()
    train_labels = data.y[data.train_mask].cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()
    test_labels = data.y[data.test_mask].cpu().numpy()
    
    # Calculate label statistics
    total_labels = labels.sum(axis=0)
    train_label_counts = train_labels.sum(axis=0)
    val_label_counts = val_labels.sum(axis=0)
    test_label_counts = test_labels.sum(axis=0)
    
    # Create label distribution dataframe
    label_df = pd.DataFrame({
        'Label_Index': range(len(total_labels)),
        'Total_Count': total_labels,
        'Train_Count': train_label_counts,
        'Val_Count': val_label_counts,
        'Test_Count': test_label_counts
    })
    
    # Calculate percentages
    label_df['Total_Percentage'] = (label_df['Total_Count'] / label_df['Total_Count'].sum()) * 100
    label_df['Train_Percentage'] = (label_df['Train_Count'] / label_df['Train_Count'].sum()) * 100
    label_df['Val_Percentage'] = (label_df['Val_Count'] / label_df['Val_Count'].sum()) * 100
    label_df['Test_Percentage'] = (label_df['Test_Count'] / label_df['Test_Count'].sum()) * 100
    
    # Print statistics
    print(f"Total number of unique labels: {len(total_labels)}")
    print(f"Total label instances: {total_labels.sum()}")
    print(f"Average labels per node: {total_labels.sum() / data.num_nodes:.2f}")
    print(f"Max labels per node: {labels.sum(axis=1).max()}")
    print(f"Min labels per node: {labels.sum(axis=1).min()}")
    
    # Show top 10 most frequent labels
    print("\n🏆 Top 10 Most Frequent Labels:")
    top_labels = label_df.nlargest(10, 'Total_Count')[['Label_Index', 'Total_Count', 'Total_Percentage']].copy()
    
    if idx_to_task:
        top_labels['Task_Name'] = top_labels['Label_Index'].map(idx_to_task)
        display_cols = ['Label_Index', 'Task_Name', 'Total_Count', 'Total_Percentage']
    else:
        display_cols = ['Label_Index', 'Total_Count', 'Total_Percentage']
    
    print(top_labels[display_cols].to_string(index=False))
    
    # Create visualizations
    if output_dir:
        create_label_visualizations(label_df, idx_to_task, train_label_counts, 
                                  val_label_counts, test_label_counts, labels, 
                                  total_labels, output_dir)
    
    return label_df


def create_label_visualizations(label_df, idx_to_task, train_label_counts, 
                              val_label_counts, test_label_counts, labels, 
                              total_labels, output_dir):
    """
    Create and save label distribution visualizations.
    
    Args:
        label_df: DataFrame with label statistics
        idx_to_task: Dictionary mapping label indices to task names
        train_label_counts, val_label_counts, test_label_counts: Label counts per split
        labels: Full label matrix
        total_labels: Total label counts
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Label Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Top 20 most frequent labels
    ax1 = axes[0, 0]
    top_20 = label_df.nlargest(20, 'Total_Count')
    bars1 = ax1.bar(range(len(top_20)), top_20['Total_Count'], color='skyblue', alpha=0.7)
    ax1.set_title('Top 20 Most Frequent Labels', fontweight='bold')
    ax1.set_xlabel('Label Index')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Add task names to x-axis if available
    if idx_to_task:
        x_labels = [f"{idx}\n{idx_to_task.get(idx, 'Unknown')}" for idx in top_20['Label_Index']]
        ax1.set_xticks(range(len(top_20)))
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    # Plot 2: Label distribution across splits
    ax2 = axes[0, 1]
    splits = ['Train', 'Val', 'Test']
    counts = [train_label_counts.sum(), val_label_counts.sum(), test_label_counts.sum()]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars2 = ax2.bar(splits, counts, color=colors, alpha=0.7)
    ax2.set_title('Label Distribution Across Data Splits', fontweight='bold')
    ax2.set_ylabel('Total Label Count')
    
    # Add percentage labels
    total_count = sum(counts)
    for i, (bar, count) in enumerate(zip(bars2, counts)):
        height = bar.get_height()
        percentage = (count / total_count) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(count)}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Labels per node distribution
    ax3 = axes[1, 0]
    labels_per_node = labels.sum(axis=1)
    ax3.hist(labels_per_node, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_title('Distribution of Labels per Node', fontweight='bold')
    ax3.set_xlabel('Number of Labels per Node')
    ax3.set_ylabel('Frequency')
    ax3.axvline(labels_per_node.mean(), color='red', linestyle='--', 
               label=f'Mean: {labels_per_node.mean():.2f}')
    ax3.legend()
    
    # Plot 4: Label frequency distribution (log scale)
    ax4 = axes[1, 1]
    ax4.hist(total_labels, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title('Label Frequency Distribution', fontweight='bold')
    ax4.set_xlabel('Label Frequency')
    ax4.set_ylabel('Number of Labels')
    ax4.set_yscale('log')
    ax4.axvline(total_labels.mean(), color='red', linestyle='--', 
               label=f'Mean: {total_labels.mean():.2f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'label_distribution_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Label distribution plot saved to: {plot_path}")
    plt.close()


def sample_subgraph(data, sample_size=1000, seed=42):
    """
    Sample a subgraph for visualization purposes.
    
    Args:
        data: PyTorch Geometric Data object
        sample_size: Number of nodes to sample
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (subgraph, sampled_node_indices)
    """
    np.random.seed(seed)
    
    # Get all nodes
    all_nodes = torch.arange(data.num_nodes)
    
    # Sample nodes
    sampled_nodes = torch.from_numpy(np.random.choice(all_nodes.cpu().numpy(), 
                                                    size=min(sample_size, data.num_nodes), 
                                                    replace=False))
    
    # Create node mapping
    node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
    
    # Filter edges that connect sampled nodes
    edge_mask = torch.isin(data.edge_index[0], sampled_nodes) & torch.isin(data.edge_index[1], sampled_nodes)
    filtered_edges = data.edge_index[:, edge_mask]
    
    # Remap edge indices to new node indices
    new_edge_index = torch.zeros_like(filtered_edges)
    for i in range(filtered_edges.shape[1]):
        new_edge_index[0, i] = node_mapping[filtered_edges[0, i].item()]
        new_edge_index[1, i] = node_mapping[filtered_edges[1, i].item()]
    
    # Create subgraph data
    subgraph = Data(
        x=data.x[sampled_nodes],
        edge_index=new_edge_index,
        y=data.y[sampled_nodes],
        train_mask=data.train_mask[sampled_nodes],
        val_mask=data.val_mask[sampled_nodes],
        test_mask=data.test_mask[sampled_nodes]
    )
    
    return subgraph, sampled_nodes


def analyze_graph_structure(data, output_dir=None, sample_size=500):
    """
    Analyze graph structure and create visualizations.
    
    Args:
        data: PyTorch Geometric Data object
        output_dir: Directory to save plots
        sample_size: Size of subgraph to sample for visualization
    """
    print("=" * 60)
    print("🕸️ SAMPLE GRAPH VISUALIZATION")
    print("=" * 60)
    
    # Sample a subgraph for visualization
    subgraph, sampled_node_indices = sample_subgraph(data, sample_size=sample_size)
    print(f"Sampled subgraph: {subgraph.num_nodes} nodes, {subgraph.num_edges} edges")
    
    # Convert to NetworkX
    G = to_networkx(subgraph, to_undirected=True)
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Calculate basic graph statistics
    print(f"\n📊 Graph Statistics:")
    print(f"   • Number of connected components: {nx.number_connected_components(G)}")
    print(f"   • Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"   • Density: {nx.density(G):.4f}")
    print(f"   • Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    # Get the largest connected component for better visualization
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc)
    print(f"   • Largest connected component: {G_largest.number_of_nodes()} nodes, {G_largest.number_of_edges()} edges")
    
    if output_dir:
        create_graph_visualizations(G_largest, subgraph, output_dir)


def create_graph_visualizations(G_largest, subgraph, output_dir):
    """
    Create and save graph structure visualizations.
    
    Args:
        G_largest: Largest connected component NetworkX graph
        subgraph: Sampled subgraph
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Sample Graph Visualizations', fontsize=16, fontweight='bold')
    
    # Plot 1: Network visualization with degree-based node sizes
    ax1 = axes[0]
    pos = nx.spring_layout(G_largest, k=1, iterations=50, seed=42)
    degrees = dict(G_largest.degree())
    node_sizes = [degrees[node] * 20 + 50 for node in G_largest.nodes()]
    
    nx.draw(G_largest, pos, 
            node_size=node_sizes,
            node_color='lightblue',
            edge_color='gray',
            alpha=0.7,
            with_labels=False,
            ax=ax1)
    ax1.set_title(f'Network Structure\n({G_largest.number_of_nodes()} nodes, {G_largest.number_of_edges()} edges)', 
                  fontweight='bold')
    
    # Plot 2: Degree distribution
    ax2 = axes[1]
    degree_sequence = sorted([d for n, d in G_largest.degree()], reverse=True)
    ax2.hist(degree_sequence, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Degree Distribution', fontweight='bold')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(degree_sequence), color='red', linestyle='--', 
               label=f'Mean: {np.mean(degree_sequence):.2f}')
    ax2.legend()
    
    # Plot 3: Node label distribution in the sample
    ax3 = axes[2]
    sample_labels = subgraph.y.cpu().numpy()
    sample_label_counts = sample_labels.sum(axis=0)
    non_zero_labels = sample_label_counts[sample_label_counts > 0]
    
    ax3.bar(range(len(non_zero_labels)), non_zero_labels, alpha=0.7, color='lightgreen')
    ax3.set_title('Label Distribution in Sample', fontweight='bold')
    ax3.set_xlabel('Label Index')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'graph_structure_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"🕸️ Graph structure plot saved to: {plot_path}")
    plt.close()


def analyze_node_features(data, output_dir=None):
    """
    Analyze node features and create visualizations.
    
    Args:
        data: PyTorch Geometric Data object
        output_dir: Directory to save plots
    """
    print("=" * 60)
    print("🔬 NODE FEATURE ANALYSIS")
    print("=" * 60)
    
    # Convert features to numpy for analysis
    features = data.x.cpu().numpy()
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature data type: {features.dtype}")
    print(f"Feature value range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Basic feature statistics
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_medians = np.median(features, axis=0)
    
    print(f"\n📊 Feature Statistics:")
    print(f"   • Mean of feature means: {np.mean(feature_means):.4f}")
    print(f"   • Mean of feature stds: {np.mean(feature_stds):.4f}")
    print(f"   • Features with zero variance: {(feature_stds == 0).sum()}")
    print(f"   • Features with near-zero variance (< 0.001): {(feature_stds < 0.001).sum()}")
    
    if output_dir:
        create_feature_visualizations(features, feature_means, feature_stds, output_dir)


def create_feature_visualizations(features, feature_means, feature_stds, output_dir):
    """
    Create and save node feature visualizations.
    
    Args:
        features: Feature matrix
        feature_means: Mean of each feature
        feature_stds: Standard deviation of each feature
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Node Feature Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature value distribution (sample of features)
    ax1 = axes[0, 0]
    sample_features = features[:, :min(100, features.shape[1])]  # Sample first 100 features
    ax1.hist(sample_features.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Feature Value Distribution (Sample)', fontweight='bold')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(sample_features.mean(), color='red', linestyle='--', 
               label=f'Mean: {sample_features.mean():.4f}')
    ax1.legend()
    
    # Plot 2: Feature means distribution
    ax2 = axes[0, 1]
    ax2.hist(feature_means, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Distribution of Feature Means', fontweight='bold')
    ax2.set_xlabel('Feature Mean')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(feature_means), color='red', linestyle='--', 
               label=f'Mean: {np.mean(feature_means):.4f}')
    ax2.legend()
    
    # Plot 3: Feature standard deviations distribution
    ax3 = axes[0, 2]
    ax3.hist(feature_stds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Distribution of Feature Standard Deviations', fontweight='bold')
    ax3.set_xlabel('Feature Std')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(feature_stds), color='red', linestyle='--', 
               label=f'Mean: {np.mean(feature_stds):.4f}')
    ax3.legend()
    
    # Plot 4: Feature correlation heatmap (sample)
    ax4 = axes[1, 0]
    sample_size = min(50, features.shape[1])
    sample_features_corr = features[:, :sample_size]
    correlation_matrix = np.corrcoef(sample_features_corr.T)
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title(f'Feature Correlation Matrix (First {sample_size} features)', fontweight='bold')
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Feature variance distribution
    ax5 = axes[1, 1]
    feature_vars = np.var(features, axis=0)
    ax5.hist(feature_vars, bins=50, alpha=0.7, color='gold', edgecolor='black')
    ax5.set_title('Distribution of Feature Variances', fontweight='bold')
    ax5.set_xlabel('Feature Variance')
    ax5.set_ylabel('Frequency')
    ax5.set_yscale('log')
    ax5.axvline(np.mean(feature_vars), color='red', linestyle='--', 
               label=f'Mean: {np.mean(feature_vars):.4f}')
    ax5.legend()
    
    # Plot 6: Feature sparsity
    ax6 = axes[1, 2]
    feature_sparsity = (features == 0).sum(axis=0) / features.shape[0]
    ax6.hist(feature_sparsity, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.set_title('Feature Sparsity Distribution', fontweight='bold')
    ax6.set_xlabel('Sparsity (fraction of zeros)')
    ax6.set_ylabel('Frequency')
    ax6.axvline(np.mean(feature_sparsity), color='red', linestyle='--', 
               label=f'Mean: {np.mean(feature_sparsity):.4f}')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'node_feature_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"🔬 Node feature plot saved to: {plot_path}")
    plt.close()


def analyze_edge_connectivity(data):
    """
    Analyze edge connectivity and graph structure.
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        dict: Dictionary containing connectivity statistics
    """
    print("=" * 60)
    print("🔗 EDGE ANALYSIS & GRAPH CONNECTIVITY")
    print("=" * 60)
    
    # Edge analysis
    edge_index = data.edge_index.cpu().numpy()
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Edge index dtype: {edge_index.dtype}")
    
    # Check for self-loops and duplicate edges
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    
    self_loops = (source_nodes == target_nodes).sum()
    print(f"Number of self-loops: {self_loops}")
    
    # Check for duplicate edges (considering both directions)
    edges_set = set()
    duplicate_edges = 0
    for i in range(edge_index.shape[1]):
        edge = tuple(sorted([source_nodes[i], target_nodes[i]]))
        if edge in edges_set:
            duplicate_edges += 1
        else:
            edges_set.add(edge)
    
    print(f"Number of duplicate edges: {duplicate_edges}")
    
    # Graph connectivity analysis
    print(f"\n📊 Graph Connectivity Analysis:")
    print(f"   • Total nodes: {data.num_nodes}")
    print(f"   • Total edges: {data.num_edges}")
    print(f"   • Graph density: {2 * data.num_edges / (data.num_nodes * (data.num_nodes - 1)):.6f}")
    print(f"   • Average degree: {2 * data.num_edges / data.num_nodes:.2f}")
    
    # Analyze degree distribution
    degrees = np.zeros(data.num_nodes)
    for i in range(edge_index.shape[1]):
        degrees[source_nodes[i]] += 1
        degrees[target_nodes[i]] += 1
    
    print(f"   • Max degree: {degrees.max()}")
    print(f"   • Min degree: {degrees.min()}")
    print(f"   • Mean degree: {degrees.mean():.2f}")
    print(f"   • Median degree: {np.median(degrees):.2f}")
    print(f"   • Std degree: {degrees.std():.2f}")
    
    # Analyze isolated nodes
    isolated_nodes = (degrees == 0).sum()
    print(f"   • Isolated nodes (degree 0): {isolated_nodes} ({isolated_nodes/data.num_nodes*100:.2f}%)")
    
    # Analyze high-degree nodes
    high_degree_threshold = np.percentile(degrees, 95)
    high_degree_nodes = (degrees >= high_degree_threshold).sum()
    print(f"   • High-degree nodes (≥95th percentile): {high_degree_nodes} ({high_degree_nodes/data.num_nodes*100:.2f}%)")
    
    return {
        'self_loops': self_loops,
        'duplicate_edges': duplicate_edges,
        'degrees': degrees,
        'isolated_nodes': isolated_nodes,
        'high_degree_nodes': high_degree_nodes
    }


def generate_summary_report(data, label_df, idx_to_task, connectivity_stats, output_dir=None):
    """
    Generate a comprehensive summary report of the EDA analysis.
    
    Args:
        data: PyTorch Geometric Data object
        label_df: DataFrame with label statistics
        idx_to_task: Dictionary mapping label indices to task names
        connectivity_stats: Dictionary with connectivity statistics
        output_dir: Directory to save the report
    """
    print("=" * 60)
    print("📋 EDA SUMMARY & KEY INSIGHTS")
    print("=" * 60)
    
    # Calculate additional statistics
    total_labels = label_df['Total_Count'].values
    features = data.x.cpu().numpy()
    feature_stds = np.std(features, axis=0)
    feature_sparsity = (features == 0).sum(axis=0) / features.shape[0]
    
    print("🎯 Dataset Overview:")
    print(f"   • Total nodes: {data.num_nodes:,}")
    print(f"   • Total edges: {data.num_edges:,}")
    print(f"   • Feature dimension: {data.x.shape[1]}")
    print(f"   • Number of labels: {data.y.shape[1]}")
    print(f"   • Graph density: {2 * data.num_edges / (data.num_nodes * (data.num_nodes - 1)):.6f}")
    
    print(f"\n🏷️ Label Analysis:")
    print(f"   • Total label instances: {total_labels.sum():,}")
    print(f"   • Average labels per node: {total_labels.sum() / data.num_nodes:.2f}")
    most_freq_idx = total_labels.argmax()
    most_freq_name = idx_to_task.get(most_freq_idx, f"Index {most_freq_idx}")
    print(f"   • Most frequent label: {most_freq_name} ({total_labels.max()} instances)")
    print(f"   • Labels with zero instances: {(total_labels == 0).sum()}")
    print(f"   • Label distribution CV: {total_labels.std() / total_labels.mean():.2f}")
    
    # Show top 5 task categories
    if idx_to_task:
        print(f"\n📊 Top 5 Task Categories:")
        top_5_tasks = label_df.nlargest(5, 'Total_Count')
        for _, row in top_5_tasks.iterrows():
            task_name = idx_to_task.get(row['Label_Index'], f"Index {row['Label_Index']}")
            print(f"   • {task_name}: {row['Total_Count']} instances ({row['Total_Percentage']:.1f}%)")
    
    print(f"\n🔗 Graph Structure:")
    print(f"   • Average degree: {connectivity_stats['degrees'].mean():.2f}")
    print(f"   • Max degree: {connectivity_stats['degrees'].max()}")
    print(f"   • Isolated nodes: {connectivity_stats['isolated_nodes']} ({connectivity_stats['isolated_nodes']/data.num_nodes*100:.2f}%)")
    print(f"   • Self-loops: {connectivity_stats['self_loops']}")
    print(f"   • Duplicate edges: {connectivity_stats['duplicate_edges']}")
    
    print(f"\n🔬 Feature Quality:")
    print(f"   • Feature value range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   • Average feature sparsity: {np.mean(feature_sparsity):.4f}")
    print(f"   • Features with zero variance: {(feature_stds == 0).sum()}")
    print(f"   • High sparsity features (>90% zeros): {(feature_sparsity > 0.9).sum()}")
    
    print(f"\n📊 Data Splits:")
    print(f"   • Train nodes: {data.train_mask.sum():,} ({data.train_mask.sum()/data.num_nodes*100:.1f}%)")
    print(f"   • Val nodes: {data.val_mask.sum():,} ({data.val_mask.sum()/data.num_nodes*100:.1f}%)")
    print(f"   • Test nodes: {data.test_mask.sum():,} ({data.test_mask.sum()/data.num_nodes*100:.1f}%)")
    
    print(f"\n💡 Key Insights:")
    print(f"   • This is a {data.num_nodes:,}-node graph with {data.num_edges:,} edges")
    print(f"   • The graph is {'sparse' if 2 * data.num_edges / (data.num_nodes * (data.num_nodes - 1)) < 0.01 else 'dense'}")
    print(f"   • Label distribution is {'balanced' if total_labels.std() / total_labels.mean() < 1.0 else 'imbalanced'}")
    print(f"   • Node connectivity shows {'high' if connectivity_stats['degrees'].mean() > 10 else 'low'} average degree")
    print(f"   • Feature quality is {'good' if (feature_stds == 0).sum() < 10 else 'needs attention'}")
    
    print(f"\nEDA Analysis Complete!")
    print("=" * 60)
    
    # Save summary to file if output directory is provided
    if output_dir:
        summary_path = os.path.join(output_dir, 'eda_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("EDA Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Overview:\n")
            f.write(f"  Total nodes: {data.num_nodes:,}\n")
            f.write(f"  Total edges: {data.num_edges:,}\n")
            f.write(f"  Feature dimension: {data.x.shape[1]}\n")
            f.write(f"  Number of labels: {data.y.shape[1]}\n")
            f.write(f"  Graph density: {2 * data.num_edges / (data.num_nodes * (data.num_nodes - 1)):.6f}\n\n")
            
            f.write(f"Label Analysis:\n")
            f.write(f"  Total label instances: {total_labels.sum():,}\n")
            f.write(f"  Average labels per node: {total_labels.sum() / data.num_nodes:.2f}\n")
            f.write(f"  Most frequent label: {most_freq_name} ({total_labels.max()} instances)\n")
            f.write(f"  Label distribution CV: {total_labels.std() / total_labels.mean():.2f}\n\n")
            
            f.write(f"Graph Structure:\n")
            f.write(f"  Average degree: {connectivity_stats['degrees'].mean():.2f}\n")
            f.write(f"  Max degree: {connectivity_stats['degrees'].max()}\n")
            f.write(f"  Isolated nodes: {connectivity_stats['isolated_nodes']}\n")
            f.write(f"  Self-loops: {connectivity_stats['self_loops']}\n")
            f.write(f"  Duplicate edges: {connectivity_stats['duplicate_edges']}\n\n")
            
            f.write(f"Feature Quality:\n")
            f.write(f"  Feature value range: [{features.min():.4f}, {features.max():.4f}]\n")
            f.write(f"  Average feature sparsity: {np.mean(feature_sparsity):.4f}\n")
            f.write(f"  Features with zero variance: {(feature_stds == 0).sum()}\n")
        
        print(f"📄 Summary report saved to: {summary_path}")


def main():
    """Main function to run EDA analysis."""
    parser = argparse.ArgumentParser(description="Comprehensive EDA Analysis for HuggingfaceKG Dataset")
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help="Path to the dataset .pt file")
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./eda_results',
                       help="Directory to save analysis results and plots (default: ./eda_results)")
    parser.add_argument('--task_mapping_path', type=str, default=None,
                       help="Path to task_to_idx.json file for label mapping")
    parser.add_argument('--sample_size', type=int, default=500,
                       help="Size of subgraph to sample for visualization (default: 500)")
    parser.add_argument('--fix_graph', action='store_true', default=True,
                       help="Fix graph structure by correcting edge_index format")
    parser.add_argument('--no_fix_graph', action='store_true',
                       help="Skip graph structure fixing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Starting EDA Analysis...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.dataset_path}...")
    try:
        data = torch.load(args.dataset_path)
        print(f"Dataset loaded successfully")
        print(f"Graph info: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Feature dimension: {data.x.shape[1]}")
        print(f"Number of labels: {data.y.shape[1]}")
        print(f"Device: {data.x.device}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Fix graph structure if requested
    if args.fix_graph and not args.no_fix_graph:
        data = fix_graph_structure(data)
    
    # Load task mappings if provided
    task_to_idx, idx_to_task = {}, {}
    if args.task_mapping_path:
        task_to_idx, idx_to_task = load_task_mappings(args.task_mapping_path)
    
    # Run EDA analysis
    print(f"\n📋 Loading task label mappings...")
    if idx_to_task:
        print(f"Sample mappings: {dict(list(idx_to_task.items())[:5])}")
    else:
        print("Using numeric indices")
    
    # 1. Label Distribution Analysis
    label_df = analyze_label_distribution(data, idx_to_task, args.output_dir)
    
    # 2. Graph Structure Analysis
    analyze_graph_structure(data, args.output_dir, args.sample_size)
    
    # 3. Node Feature Analysis
    analyze_node_features(data, args.output_dir)
    
    # 4. Edge Connectivity Analysis
    connectivity_stats = analyze_edge_connectivity(data)
    
    # 5. Generate Summary Report
    generate_summary_report(data, label_df, idx_to_task, connectivity_stats, args.output_dir)
    
    print(f"\n🎉 EDA Analysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# python -m scripts.eda_analysis --dataset_path ./experiment_runs/run_2025-09-26_22-12-13/final_graph.pt --task_mapping_path ./experiment_runs/run_2025-09-26_22-12-13/task_to_idx.json --sample_size 1000 --fix_graph
# python -m scripts.eda_analysis --dataset_path ../task_classification/data/huggingface_bge.pt --task_mapping_path ../task_classification/data/task_id.json --sample_size 1000 --fix_graph


