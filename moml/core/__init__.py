"""
MoML Core Package

This package contains core functionality for molecular representation,
including graph generation, coarsening, and molecular descriptors.
"""

# Molecular Graph Processing
from moml.core.molecular_graph import (
    MolecularGraphProcessor, 
    create_graph_processor,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules,
    collate_graphs
)

# Graph Coarsening
from moml.core.graph_coarsening import (
    GraphCoarsener,
)

# Molecular Descriptors
from moml.core.molecular_descriptors import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor,
    calculate_molecular_descriptors,
    extract_fingerprints,
    validate_smiles
)

__all__ = [
    # Molecular Graph Processing
    "MolecularGraphProcessor",
    "create_graph_processor",
    "create_molecular_graph_json",
    "batch_create_graphs_from_molecules",
    "collate_graphs",
    
    # Graph Coarsening
    "GraphCoarsener",
    "FunctionalGroupDetector",
    "StructuralMotifDetector",
    
    # Molecular Descriptors
    "MolecularFeatureExtractor",
    "calculate_molecular_descriptors",
    "extract_fingerprints",
    "validate_smiles"
]
