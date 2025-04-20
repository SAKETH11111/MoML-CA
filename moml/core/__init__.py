"""
MoML Core Package

Public API:
- MolecularGraphProcessor: build and configure graph processors
- mol_file_to_graph, create_molecular_graph_json, batch_create_graphs_from_molecules: file‑based graph builders
- collate_graphs, graph_to_device, find_charges_file, read_charges_from_file: graph utilities
- GraphCoarsener: hierarchical coarsening of PFAS graphs
- FunctionalGroupDetector, MolecularFeatureExtractor: feature‑extraction helpers
- calculate_molecular_descriptors, extract_fingerprints: single‑molecule descriptor utilities
"""

# Import core processing classes and functions
from .molecular_graph_processor import (
    MolecularGraphProcessor,
    create_graph_processor,
    mol_file_to_graph,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules,
    collate_graphs,
    graph_to_device,
    find_charges_file,
    read_charges_from_file,
)

# Hierarchical graph coarsening
from .hierarchical_graph_coarsener import GraphCoarsener

# Feature extraction and descriptor utilities
from .molecular_feature_extraction import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor,
    calculate_molecular_descriptors,
    extract_fingerprints,
    # validate_smiles is provided in utils
)

__all__ = [
    'MolecularGraphProcessor',
    'create_graph_processor',
    'mol_file_to_graph',
    'create_molecular_graph_json',
    'batch_create_graphs_from_molecules',
    'collate_graphs',
    'graph_to_device',
    'find_charges_file',
    'read_charges_from_file',
    'GraphCoarsener',
    'FunctionalGroupDetector',
    'MolecularFeatureExtractor',
    'calculate_molecular_descriptors',
    'extract_fingerprints',
    # 'validate_smiles', removed as part of core, use utils.validate_smiles
]
