"""
moml/data/__init__.py

MoML Data Package - Molecular data processing and dataset utilities.

This package provides comprehensive tools for loading, processing, and managing
molecular datasets for machine learning applications. It includes dataset classes,
data loaders, splitting utilities, and graph processing functions.

Public API:
    Dataset Classes:
        - MolecularGraphDataset: Standard molecular graph dataset
        - HierarchicalGraphDataset: Multi-level molecular representations  
        - PFASDataset: Specialized PFAS compound dataset
    
    Data Loading:
        - load_dataset: Load dataset from directory
        - load_datasets_from_splits: Load pre-split datasets
        - load_hierarchical_dataset: Load hierarchical graph datasets
    
    Dataset Splitting:
        - split_dataset: Random dataset splitting
        - stratified_split_dataset: Stratified splitting by labels
        - scaffold_split_dataset: Scaffold-based molecular splitting
    
    PyTorch DataLoaders:
        - prepare_dataloaders: Create PyTorch DataLoader objects
        - create_dataloaders_from_directory: End-to-end data loading
        - create_stratified_dataloaders: Stratified DataLoader creation
    
    Molecule Processing:
        - process_mol_file: Single molecule file processing
        - process_mol_file_to_graph: File to graph conversion
        - batch_process_molecules: Batch molecule processing
        - process_dataset: CSV dataset processing
        - save_processed_molecules: Save processed data
        - batch_process_molecules_dataset: End-to-end dataset processing
        - graph_batch_process: Pipeline-optimized batch processing
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from .datasets import (
        MolecularGraphDataset as _MolecularGraphDataset,
        HierarchicalGraphDataset as _HierarchicalGraphDataset,
        PFASDataset as _PFASDataset,
    )
    from .dataset_loader import (
        load_dataset as _load_dataset,
        load_datasets_from_splits as _load_datasets_from_splits,
        load_hierarchical_dataset as _load_hierarchical_dataset,
    )
    from .dataset_splitter import (
        split_dataset as _split_dataset,
        stratified_split_dataset as _stratified_split_dataset,
        scaffold_split_dataset as _scaffold_split_dataset,
    )
    from .pytorch_data_loader import (
        prepare_dataloaders as _prepare_dataloaders,
        create_dataloaders_from_directory as _create_dataloaders_from_directory,
        create_stratified_dataloaders as _create_stratified_dataloaders,
    )
    from .molecule_processors import (
        process_mol_file as _process_mol_file,
        process_mol_file_to_graph as _process_mol_file_to_graph,
        batch_process_molecules as _batch_process_molecules,
        process_dataset as _process_dataset,
        save_processed_molecules as _save_processed_molecules,
        batch_process_molecules_dataset as _batch_process_molecules_dataset,
        graph_batch_process as _graph_batch_process,
    )

logger = logging.getLogger(__name__)

# Import errors to handle gracefully
_IMPORT_ERRORS: Dict[str, str] = {}

# Helper functions for dynamic dummy creation
def _create_dummy_class(name: str, error_msg: str) -> type:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    
    return type(name, (), {"__init__": __init__})

def _create_dummy_function(name: str, error_msg: str) -> Callable[..., Any]:
    def dummy(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    return dummy

# Dataset classes
try:
    from .datasets import (
        MolecularGraphDataset,
        HierarchicalGraphDataset, 
        PFASDataset,
    )
except ImportError as e:
    _IMPORT_ERRORS['datasets'] = str(e)
    
    MolecularGraphDataset = _create_dummy_class("MolecularGraphDataset", _IMPORT_ERRORS['datasets'])  # type: ignore
    HierarchicalGraphDataset = _create_dummy_class("HierarchicalGraphDataset", _IMPORT_ERRORS['datasets'])  # type: ignore
    PFASDataset = _create_dummy_class("PFASDataset", _IMPORT_ERRORS['datasets'])  # type: ignore

# Dataset loader functions
try:
    from .dataset_loader import (
        load_dataset,
        load_datasets_from_splits,
        load_hierarchical_dataset,
    )
except ImportError as e:
    _IMPORT_ERRORS['dataset_loader'] = str(e)
    
    load_dataset = _create_dummy_function("load_dataset", _IMPORT_ERRORS['dataset_loader'])
    load_datasets_from_splits = _create_dummy_function("load_datasets_from_splits", _IMPORT_ERRORS['dataset_loader'])
    load_hierarchical_dataset = _create_dummy_function("load_hierarchical_dataset", _IMPORT_ERRORS['dataset_loader'])

# Dataset splitting functions
try:
    from .dataset_splitter import (
        split_dataset,
        stratified_split_dataset,
        scaffold_split_dataset,
    )
except ImportError as e:
    _IMPORT_ERRORS['dataset_splitter'] = str(e)
    
    split_dataset = _create_dummy_function("split_dataset", _IMPORT_ERRORS['dataset_splitter'])
    stratified_split_dataset = _create_dummy_function("stratified_split_dataset", _IMPORT_ERRORS['dataset_splitter'])
    scaffold_split_dataset = _create_dummy_function("scaffold_split_dataset", _IMPORT_ERRORS['dataset_splitter'])

# PyTorch DataLoader utilities
try:
    from .pytorch_data_loader import (
        prepare_dataloaders,
        create_dataloaders_from_directory,
        create_stratified_dataloaders,
    )
except ImportError as e:
    _IMPORT_ERRORS['pytorch_data_loader'] = str(e)
    
    prepare_dataloaders = _create_dummy_function("prepare_dataloaders", _IMPORT_ERRORS['pytorch_data_loader'])
    create_dataloaders_from_directory = _create_dummy_function("create_dataloaders_from_directory", _IMPORT_ERRORS['pytorch_data_loader'])
    create_stratified_dataloaders = _create_dummy_function("create_stratified_dataloaders", _IMPORT_ERRORS['pytorch_data_loader'])

# Molecule processing utilities
try:
    from .molecule_processors import (
        process_mol_file,
        process_mol_file_to_graph,
        batch_process_molecules,
        process_dataset,
        save_processed_molecules,
        batch_process_molecules_dataset,
        graph_batch_process,
    )
except ImportError as e:
    _IMPORT_ERRORS['molecule_processors'] = str(e)
    
    process_mol_file = _create_dummy_function("process_mol_file", _IMPORT_ERRORS['molecule_processors'])
    process_mol_file_to_graph = _create_dummy_function("process_mol_file_to_graph", _IMPORT_ERRORS['molecule_processors'])
    batch_process_molecules = _create_dummy_function("batch_process_molecules", _IMPORT_ERRORS['molecule_processors'])
    process_dataset = _create_dummy_function("process_dataset", _IMPORT_ERRORS['molecule_processors'])
    save_processed_molecules = _create_dummy_function("save_processed_molecules", _IMPORT_ERRORS['molecule_processors'])
    batch_process_molecules_dataset = _create_dummy_function("batch_process_molecules_dataset", _IMPORT_ERRORS['molecule_processors'])
    graph_batch_process = _create_dummy_function("graph_batch_process", _IMPORT_ERRORS['molecule_processors'])

# Public API
__all__ = [
    # Dataset classes
    "MolecularGraphDataset",
    "HierarchicalGraphDataset", 
    "PFASDataset",
    # Dataset loading
    "load_dataset",
    "load_datasets_from_splits",
    "load_hierarchical_dataset",
    # Dataset splitting
    "split_dataset",
    "stratified_split_dataset", 
    "scaffold_split_dataset",
    # PyTorch DataLoaders
    "prepare_dataloaders",
    "create_dataloaders_from_directory",
    "create_stratified_dataloaders",
    # Molecule processing
    "process_mol_file",
    "process_mol_file_to_graph",
    "batch_process_molecules",
    "process_dataset",
    "save_processed_molecules",
    "batch_process_molecules_dataset",
    "graph_batch_process",
]
