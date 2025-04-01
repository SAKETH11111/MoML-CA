"""
Visualization utilities for molecular graphs and hierarchical representations.

This module provides functions for visualizing molecular graphs with different
highlighting options and hierarchical graph structures.

This module is focused on PyTorch Geometric graphs, while molecule_visualization.py
is focused on RDKit molecule objects.
"""

import os
import torch
from typing import Dict, Optional


def visualize_molecular_graph(graph, output_file: Optional[str] = None, highlight_feature: str = 'fluorine'):
    """
    Visualize a molecular graph with highlighted features.
    
    Args:
        graph: PyTorch Geometric Data object
        output_file: Path to save the visualization
        highlight_feature: Feature to highlight ('fluorine', 'partial_charge', 'functional_group', 'head_group')
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError:
        print("Visualization requires matplotlib, networkx, and PyTorch Geometric.")
        return
    
    G = to_networkx(graph, to_undirected=True)
    
    plt.figure(figsize=(12, 10))
    
    # Use 3D positions for layout if available
    if hasattr(graph, 'pos') and graph.pos is not None:
        pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.num_nodes)}
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Determine node colors based on highlight feature
    node_colors = []
    node_sizes = []
    
    for i in range(graph.num_nodes):
        size = 500  # Default node size
        
        if highlight_feature == 'fluorine':
            # Highlight fluorine atoms and carbons bonded to fluorine
            is_f = graph.x[i][8].item() > 0.5  # Fluorine flag
            is_cf = graph.x[i][9].item() > 0.5  # Carbon-fluorine flag
            
            if is_f:
                node_colors.append('green')  # Fluorine atoms
                size = 700
            elif is_cf:
                node_colors.append('orange')  # Carbon atoms bonded to fluorine
                size = 600
            else:
                node_colors.append('lightblue')  # Other atoms
        
        elif highlight_feature == 'partial_charge':
            # Color based on partial charge (if available)
            # Assuming partial charge is the last feature in each node's feature vector
            charge_idx = -1
            if graph.x.shape[1] > 10:  # Check if we have enough features
                charge = graph.x[i][charge_idx].item()
                # Use a red-white-blue colormap for charges
                if charge > 0:
                    intensity = min(1.0, charge / 0.5)  # Scale to [0, 1]
                    node_colors.append((1.0, 1.0 - intensity, 1.0 - intensity))  # Red for positive
                else:
                    intensity = min(1.0, abs(charge) / 0.5)  # Scale to [0, 1]
                    node_colors.append((1.0 - intensity, 1.0 - intensity, 1.0))  # Blue for negative
            else:
                node_colors.append('gray')  # Default if no charge info
        
        elif highlight_feature == 'functional_group':
            # Highlight functional groups if PFAS features are available
            idx_offset = 11  # Index where PFAS-specific features start
            
            if graph.x.shape[1] > idx_offset + 2:
                is_carboxylic = graph.x[i][idx_offset].item() > 0.5
                is_sulfonic = graph.x[i][idx_offset + 1].item() > 0.5
                is_phosphonic = graph.x[i][idx_offset + 2].item() > 0.5
                
                if is_carboxylic:
                    node_colors.append('red')
                    size = 800
                elif is_sulfonic:
                    node_colors.append('purple')
                    size = 800
                elif is_phosphonic:
                    node_colors.append('brown')
                    size = 800
                else:
                    node_colors.append('lightgray')
            else:
                node_colors.append('lightgray')
        
        elif highlight_feature == 'head_group':
            # Highlight head group vs fluorinated tail
            idx_offset = 14  # Index where head_group feature is expected
            
            if graph.x.shape[1] > idx_offset:
                is_head_group = graph.x[i][idx_offset].item() > 0.5
                
                if is_head_group:
                    node_colors.append('blue')
                    size = 600
                else:
                    node_colors.append('yellow')
            else:
                node_colors.append('lightgray')
        
        else:
            # Default coloring
            node_colors.append('lightblue')
        
        node_sizes.append(size)
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            width=2.0,
            with_labels=True,
            font_size=10,
            font_weight='bold')
    
    # Add title based on highlight feature
    plt.title(f'Molecular Graph - {highlight_feature.replace("_", " ").title()} Highlight', fontsize=16)
    
    # Save or display the visualization
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Graph visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def print_graph_statistics(graph):
    """
    Print statistics about a molecular graph.
    
    Args:
        graph: PyTorch Geometric Data object
    """
    print("\nGraph Statistics:")
    print(f"  Number of nodes (atoms): {graph.num_nodes}")
    print(f"  Number of edges (bonds): {graph.edge_index.shape[1] // 2}")  # Divide by 2 because graph is undirected
    print(f"  Node feature dimension: {graph.x.shape[1]}")
    print(f"  Edge feature dimension: {graph.edge_attr.shape[1]}")
    
    # Count atom types if features follow the expected format
    if graph.x.shape[1] >= 10:
        fluorine_count = sum(1 for i in range(graph.num_nodes) if graph.x[i][8].item() > 0.5)
        carbon_fluorine_count = sum(1 for i in range(graph.num_nodes) if graph.x[i][9].item() > 0.5)
        print(f"  Fluorine atoms: {fluorine_count}")
        print(f"  Carbon atoms bonded to fluorine: {carbon_fluorine_count}")
    
    # Print global features
    if hasattr(graph, 'y') and graph.y is not None:
        print("\nGlobal Molecular Features:")
        print(f"  Molecular weight: {graph.y[0].item():.2f}")
        print(f"  Topological polar surface area: {graph.y[1].item():.2f}")
        print(f"  H-bond donors: {graph.y[2].item():.0f}")
        print(f"  H-bond acceptors: {graph.y[3].item():.0f}")
        print(f"  LogP (octanol-water partition coefficient): {graph.y[4].item():.2f}")
        print(f"  Total atoms: {graph.y[5].item():.0f}")
        print(f"  Fluorine atoms: {graph.y[6].item():.0f}")
        
        # PFAS-specific global features (if available)
        if len(graph.y) > 7:
            print(f"  CF3 groups: {graph.y[7].item():.0f}")
        if len(graph.y) > 10:
            print(f"  Carboxylic acid groups: {graph.y[8].item():.0f}")
            print(f"  Sulfonic acid groups: {graph.y[9].item():.0f}")
            print(f"  Phosphonic acid groups: {graph.y[10].item():.0f}")
        if len(graph.y) > 13:
            print(f"  CF groups: {graph.y[11].item():.0f}")
            print(f"  CF2 groups: {graph.y[12].item():.0f}")
            print(f"  CF3 groups: {graph.y[13].item():.0f}")


def visualize_hierarchical_graphs(hierarchical_graphs: Dict[str, str], output_dir: Optional[str] = None):
    """
    Visualize hierarchical graphs at each level.
    
    Args:
        hierarchical_graphs: Dictionary mapping level names to graph file paths
        output_dir: Directory to save visualizations
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for level, graph_path in hierarchical_graphs.items():
        try:
            graph = torch.load(graph_path)
            
            if output_dir:
                base_name = os.path.splitext(os.path.basename(graph_path))[0]
                viz_path = os.path.join(output_dir, f"{base_name}_{level}_visualization.png")
            else:
                viz_path = None
            
            visualize_molecular_graph(graph, viz_path, highlight_feature='functional_group')
            print(f"Visualized {level} graph from {graph_path}")
            
            # Print statistics for this level
            print(f"\n{level.replace('_', ' ').title()} Graph:")
            print_graph_statistics(graph)
            
        except Exception as e:
            print(f"Error visualizing {level} graph: {e}") 