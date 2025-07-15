"""
MoML Data Package

Public API:
- MolecularGraphDataset, HierarchicalGraphDataset, PFASDataset: dataset classes
- load_dataset, load_datasets_from_splits, load_hierarchical_dataset: data loaders
- split_dataset, stratified_split_dataset, scaffold_split_dataset: dataset splitting utilities
- prepare_dataloaders, create_dataloaders_from_directory, create_stratified_dataloaders: PyTorch DataLoader helpers
- process_mol_file, process_mol_file_to_graph, batch_process_molecules: molecule‑to‑graph utilities
- process_dataset, save_processed_molecules, batch_process_molecules_dataset: CSV‑based dataset processors
"""

# Dataset classes with conditional imports
try:
    from .datasets import (
        MolecularGraphDataset,
        HierarchicalGraphDataset,
        PFASDataset,
    )
except ImportError as e:
    # Create dummy classes when dependencies are not available
    class MolecularGraphDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MolecularGraphDataset requires additional dependencies: {e}")
    
    class HierarchicalGraphDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"HierarchicalGraphDataset requires additional dependencies: {e}")
    
    class PFASDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PFASDataset requires additional dependencies: {e}")

# Loader functions
try:
    from .dataset_loader import (
        load_dataset,
        load_datasets_from_splits,
        load_hierarchical_dataset,
    )
except ImportError as e:
    def load_dataset(*args, **kwargs):
        raise ImportError(f"load_dataset requires additional dependencies: {e}")
    
    def load_datasets_from_splits(*args, **kwargs):
        raise ImportError(f"load_datasets_from_splits requires additional dependencies: {e}")
    
    def load_hierarchical_dataset(*args, **kwargs):
        raise ImportError(f"load_hierarchical_dataset requires additional dependencies: {e}")

# Splitting functions
try:
    from .dataset_splitter import (
        split_dataset,
        stratified_split_dataset,
        scaffold_split_dataset,
    )
except ImportError as e:
    def split_dataset(*args, **kwargs):
        raise ImportError(f"split_dataset requires additional dependencies: {e}")
    
    def stratified_split_dataset(*args, **kwargs):
        raise ImportError(f"stratified_split_dataset requires additional dependencies: {e}")
    
    def scaffold_split_dataset(*args, **kwargs):
        raise ImportError(f"scaffold_split_dataset requires additional dependencies: {e}")

# DataLoader utilities
try:
    from .pytorch_data_loader import (
        prepare_dataloaders,
        create_dataloaders_from_directory,
        create_stratified_dataloaders,
    )
except ImportError as e:
    def prepare_dataloaders(*args, **kwargs):
        raise ImportError(f"prepare_dataloaders requires additional dependencies: {e}")
    
    def create_dataloaders_from_directory(*args, **kwargs):
        raise ImportError(f"create_dataloaders_from_directory requires additional dependencies: {e}")
    
    def create_stratified_dataloaders(*args, **kwargs):
        raise ImportError(f"create_stratified_dataloaders requires additional dependencies: {e}")

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
    def process_mol_file(*args, **kwargs):
        raise ImportError(f"process_mol_file requires additional dependencies: {e}")
    
    def process_mol_file_to_graph(*args, **kwargs):
        raise ImportError(f"process_mol_file_to_graph requires additional dependencies: {e}")
    
    def batch_process_molecules(*args, **kwargs):
        raise ImportError(f"batch_process_molecules requires additional dependencies: {e}")
    
    def process_dataset(*args, **kwargs):
        raise ImportError(f"process_dataset requires additional dependencies: {e}")
    
    def save_processed_molecules(*args, **kwargs):
        raise ImportError(f"save_processed_molecules requires additional dependencies: {e}")
    
    def batch_process_molecules_dataset(*args, **kwargs):
        raise ImportError(f"batch_process_molecules_dataset requires additional dependencies: {e}")
    
    def graph_batch_process(*args, **kwargs):
        raise ImportError(f"graph_batch_process requires additional dependencies: {e}")

__all__ = [
    "MolecularGraphDataset",
    "HierarchicalGraphDataset",
    "PFASDataset",
    "load_dataset",
    "load_datasets_from_splits",
    "load_hierarchical_dataset",
    "split_dataset",
    "stratified_split_dataset",
    "scaffold_split_dataset",
    "prepare_dataloaders",
    "create_dataloaders_from_directory",
    "create_stratified_dataloaders",
    "process_mol_file",
    "process_mol_file_to_graph",
    "batch_process_molecules",
    "process_dataset",
    "save_processed_molecules",
    "batch_process_molecules_dataset",
    "graph_batch_process",
]
