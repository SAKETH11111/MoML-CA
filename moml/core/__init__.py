"""
moml/core/__init__.py

Core package for molecular graph processing and feature extraction in MoML-CA.
"""

from typing import Any


def _create_dummy_class(name: str, error_msg: str) -> type:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    
    return type(name, (), {"__init__": __init__})


def _create_dummy_function(name: str, error_msg: str) -> Any:
    def dummy(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    return dummy


try:
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
except ImportError:
    # Create dummy implementations
    MolecularGraphProcessor = _create_dummy_class("MolecularGraphProcessor", "torch_geometric, rdkit")  # type: ignore
    create_graph_processor = _create_dummy_function("create_graph_processor", "torch_geometric, rdkit")
    mol_file_to_graph = _create_dummy_function("mol_file_to_graph", "torch_geometric, rdkit")
    create_molecular_graph_json = _create_dummy_function("create_molecular_graph_json", "torch_geometric, rdkit")
    batch_create_graphs_from_molecules = _create_dummy_function("batch_create_graphs_from_molecules", "torch_geometric, rdkit")
    collate_graphs = _create_dummy_function("collate_graphs", "torch_geometric, rdkit")
    graph_to_device = _create_dummy_function("graph_to_device", "torch_geometric, rdkit")
    find_charges_file = _create_dummy_function("find_charges_file", "torch_geometric, rdkit")
    read_charges_from_file = _create_dummy_function("read_charges_from_file", "torch_geometric, rdkit")


# Hierarchical graph coarsening
try:
    from .hierarchical_graph_coarsener import GraphCoarsener
except ImportError:
    GraphCoarsener = _create_dummy_class("GraphCoarsener", "torch_geometric, rdkit")  # type: ignore


# Feature extraction and descriptor utilities
try:
    from .molecular_feature_extraction import (
        FunctionalGroupDetector,
        MolecularFeatureExtractor,
        calculate_molecular_descriptors,
        extract_fingerprints,
    )
except ImportError:
    FunctionalGroupDetector = _create_dummy_class("FunctionalGroupDetector", "rdkit")  # type: ignore
    MolecularFeatureExtractor = _create_dummy_class("MolecularFeatureExtractor", "rdkit")  # type: ignore
    calculate_molecular_descriptors = _create_dummy_function("calculate_molecular_descriptors", "rdkit")
    extract_fingerprints = _create_dummy_function("extract_fingerprints", "rdkit")


__all__ = [
    "MolecularGraphProcessor",
    "create_graph_processor",
    "mol_file_to_graph",
    "create_molecular_graph_json",
    "batch_create_graphs_from_molecules",
    "collate_graphs",
    "graph_to_device",
    "find_charges_file",
    "read_charges_from_file",
    "GraphCoarsener",
    "FunctionalGroupDetector",
    "MolecularFeatureExtractor",
    "calculate_molecular_descriptors",
    "extract_fingerprints",
]
