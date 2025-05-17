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

# Dataset classes
from .datasets import (
    MolecularGraphDataset,
    HierarchicalGraphDataset,
    PFASDataset,
)

# Loader functions
from .dataset_loader import (
    load_dataset,
    load_datasets_from_splits,
    load_hierarchical_dataset,
)

# Splitting functions
from .dataset_splitter import (
    split_dataset,
    stratified_split_dataset,
    scaffold_split_dataset,
)

# DataLoader utilities
from .pytorch_data_loader import (
    prepare_dataloaders,
    create_dataloaders_from_directory,
    create_stratified_dataloaders,
)

# Molecule processing utilities
from .molecule_processors import (
    process_mol_file,
    process_mol_file_to_graph,
    batch_process_molecules,
    process_dataset,
    save_processed_molecules,
    batch_process_molecules_dataset,
    graph_batch_process,
)

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
