"""
Utilities for Molecular Graph Neural Networks

This package provides utility functions for processing PFAS molecules,
creating graph representations, and extracting quantum properties from ORCA outputs.
"""

# ORCA parsing utilities
from code.MGNN.utils.orca_parser import (
    parse_orca_output,
    extract_mulliken_charges,
    extract_loewdin_charges,
    extract_homo_lumo_gap,
    extract_homo_lumo_contributions,
    extract_electrostatic_potential,
    extract_orbital_contributions_from_orca,
    extract_electrostatic_potential_from_orca,
    batch_extract_charges
)

# Graph generation utilities
from code.MGNN.utils.mol_graph_generator import (
    create_graph_from_orca_data,
    batch_create_graphs_from_orca
)

# Graph coarsening utilities
from code.MGNN.utils.graph_coarsening_utils import (
    create_hierarchical_graphs_from_orca,
    batch_create_hierarchical_graphs,
    visualize_hierarchical_graphs
)

# Visualization utilities
from code.MGNN.utils.visualization import (
    visualize_molecular_graph,
    print_graph_statistics,
    visualize_hierarchical_graphs
)

__all__ = [
    # ORCA parsing
    'parse_orca_output',
    'extract_mulliken_charges',
    'extract_loewdin_charges',
    'extract_homo_lumo_gap',
    'extract_homo_lumo_contributions',
    'extract_electrostatic_potential',
    'extract_orbital_contributions_from_orca',
    'extract_electrostatic_potential_from_orca',
    'batch_extract_charges',
    
    # Graph generation
    'create_graph_from_orca_data',
    'batch_create_graphs_from_orca',
    
    # Graph coarsening
    'create_hierarchical_graphs_from_orca',
    'batch_create_hierarchical_graphs',
    
    # Visualization 
    'visualize_molecular_graph',
    'print_graph_statistics',
    'visualize_hierarchical_graphs'
] 