"""
MoML: Molecular Machine Learning Framework

A comprehensive framework for molecular representation, visualization, 
and machine learning for PFAS compounds and other molecular structures.
"""

__version__ = '0.1.0'

# Core molecular graph functionality
from moml.core.molecular_graph import (
    MolecularGraphProcessor,
    create_graph_processor,
    mol_file_to_graph,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules,
    read_charges_from_file,
    find_charges_file
)

# Data handling
from moml.data.dataset import MolecularGraphDataset
from moml.data.mol_processors import process_mol_file, batch_process_molecules

# Molecular visualization
from moml.utils.visualization.visualization import (
    draw_molecule,
    save_molecule_image,
    generate_molecule_grid,
    save_molecule_grid,
    visualize_dataset,
    load_molecules_from_pickle,
    visualize_molecular_graph,
    print_graph_statistics,
    visualize_hierarchical_graphs
)

# Molecular descriptors
from moml.core.molecular_descriptors import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor
)

# Expose commonly used submodules
__all__ = [
    # Classes
    'MolecularGraphProcessor',
    'MolecularGraphDataset',
    'FunctionalGroupDetector',
    'MolecularFeatureExtractor',
    
    # Graph generation
    'create_graph_processor',
    'mol_file_to_graph',
    'create_molecular_graph_json',
    'batch_create_graphs_from_molecules',
    'find_charges_file',
    'read_charges_from_file',
    
    # Data processing
    'process_mol_file',
    'batch_process_molecules',
    
    # Visualization
    'draw_molecule',
    'save_molecule_image',
    'generate_molecule_grid',
    'save_molecule_grid',
    'visualize_dataset',
    'visualize_molecular_graph',
    'print_graph_statistics',
    'visualize_hierarchical_graphs',
    'load_molecules_from_pickle'
]
