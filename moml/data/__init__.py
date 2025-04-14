"""
MoML Data Package

This package provides tools for working with molecular datasets,
including dataset classes, data loading, splitting, and processing.
"""

# Dataset classes
from moml.data.dataset import MolecularGraphDataset, HierarchicalGraphDataset, PFASDataset

# Dataset loading
from moml.data.dataset_loader import load_dataset, load_datasets_from_splits, load_hierarchical_dataset

# Dataset splitting
from moml.data.dataset_splitter import split_dataset, stratified_split_dataset, scaffold_split_dataset

# DataLoader utilities
from moml.data.pytorch_dataloader import (
    prepare_dataloaders,
    create_dataloaders_from_directory,
    create_stratified_dataloaders
)

# Processor utilities
from moml.data.mol_processors import (
    process_mol_file,
    process_mol_file_to_graph,
    batch_process_molecules,
    process_dataset,
    save_processed_molecules,
    batch_process_molecules_dataset
)

__all__ = [
    # Dataset classes
    "MolecularGraphDataset",
    "HierarchicalGraphDataset",
    "PFASDataset",
    
    # Loading functions
    "load_dataset",
    "load_datasets_from_splits",
    "load_hierarchical_dataset",
    
    # Splitting functions
    "split_dataset",
    "stratified_split_dataset",
    "scaffold_split_dataset",
    
    # DataLoader utilities
    "prepare_dataloaders",
    "create_dataloaders_from_directory",
    "create_stratified_dataloaders",
    
    # Processor utilities
    "process_mol_file",
    "process_mol_file_to_graph",
    "batch_process_molecules",
    "process_dataset",
    "save_processed_molecules",
    "batch_process_molecules_dataset"
]