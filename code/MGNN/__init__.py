"""
MoML-CA: MGNN - Molecular Graph Neural Networks for PFAS Analysis

This module provides functionality for creating, visualizing, and analyzing 
molecular graph representations of PFAS compounds with quantum chemical properties.
"""

# Core architectures
from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder
from code.MGNN.architectures.graph_coarsening import GraphCoarsener, FunctionalGroupIdentifier

# Utility functions
from code.MGNN.utils.mol_graph_generator import create_graph_from_orca_data, batch_create_graphs_from_orca
from code.MGNN.utils.graph_coarsening_utils import create_hierarchical_graphs_from_orca, batch_create_hierarchical_graphs
from code.MGNN.utils.visualization import visualize_molecular_graph, print_graph_statistics, visualize_hierarchical_graphs
from code.MGNN.utils.orca_parser import parse_orca_output, extract_mulliken_charges, extract_loewdin_charges

# Main pipeline
from code.MGNN.pfas_pipeline import run_pipeline

__version__ = "1.0.0"
__all__ = [
    # Core classes
    'MolecularGraphBuilder',
    'GraphCoarsener',
    'FunctionalGroupIdentifier',
    
    # Graph generation
    'create_graph_from_orca_data',
    'batch_create_graphs_from_orca',
    'create_hierarchical_graphs_from_orca',
    'batch_create_hierarchical_graphs',
    
    # Visualization
    'visualize_molecular_graph',
    'print_graph_statistics',
    'visualize_hierarchical_graphs',
    
    # ORCA parsing
    'parse_orca_output',
    'extract_mulliken_charges',
    'extract_loewdin_charges',
    
    # Pipeline
    'run_pipeline'
] 