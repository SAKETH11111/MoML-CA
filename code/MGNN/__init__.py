"""
Molecular Graph Neural Network (MGNN) Package

This package provides tools for generating and analyzing molecular graphs
with a specific focus on PFAS compounds and their quantum properties.
"""

# Core components
from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder

# Graph generation tools
from code.MGNN.utils.unified_graph_generator import (
    create_graph_from_smiles,
    create_graph_from_orca,
    batch_create_graphs_from_orca
)

# Visualization tools
from code.MGNN.utils.visualization import visualize_molecular_graph, print_graph_statistics

# Graph coarsening (hierarchical analysis)
from code.MGNN.architectures.graph_coarsening import GraphCoarsener, FunctionalGroupIdentifier

__version__ = "1.0.0"
__all__ = [
    # Core classes
    'MolecularGraphBuilder',
    
    # Graph generation
    'create_graph_from_smiles',
    'create_graph_from_orca',
    'batch_create_graphs_from_orca',
    
    # Visualization
    'visualize_molecular_graph',
    'print_graph_statistics',
    
    # Hierarchical analysis
    'GraphCoarsener',
    'FunctionalGroupIdentifier'
] 