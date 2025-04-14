# MoML Data Module

This module provides tools for working with molecular datasets in the MoML framework. The data module is organized into several components for improved maintainability and separation of concerns.

## Structure

- **dataset.py**: Contains dataset classes for different types of molecular data

  - `MolecularGraphDataset`: For working with molecular graphs
  - `HierarchicalGraphDataset`: For hierarchical molecular representations
  - `PFASDataset`: Specialized dataset for PFAS compounds

- **dataset_loader.py**: Functions for loading datasets from files and directories

  - `load_dataset`: Load dataset from a directory of molecule files
  - `load_datasets_from_splits`: Load datasets from pre-split directories
  - `load_hierarchical_dataset`: Load hierarchical graph datasets

- **splitting.py**: Utilities for splitting datasets into train/val/test sets

  - `split_dataset`: Basic random splitting
  - `stratified_split_dataset`: Splitting while preserving label distributions
  - `scaffold_split_dataset`: Splitting based on molecular scaffolds

- **pytorch_dataloader.py**: Tools for creating PyTorch DataLoader objects

  - `prepare_dataloaders`: Create dataloaders from existing dataset splits
  - `create_dataloaders_from_directory`: One-step loading and dataloader creation
  - `create_stratified_dataloaders`: Create dataloaders with stratified splitting

- **processors.py**: Functions for processing molecule files into graph representations
  - `process_mol_file`: Process a molecule file into a PyTorch Geometric data object
  - `process_mol_file_to_graph`: Process a molecule file into a graph representation

## Usage Examples

### Loading and Splitting a Dataset

```python
from moml.data import load_dataset, split_dataset, prepare_dataloaders

# Load a dataset from a directory
dataset = load_dataset(
    data_dir="path/to/molecules",
    labels_file="path/to/labels.csv",
    file_pattern="*.mol"
)

# Split the dataset
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

# Create dataloaders
dataloaders = prepare_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32
)
```

### Creating DataLoaders from a Directory (One-Step)

```python
from moml.data import create_dataloaders_from_directory

# Create dataloaders directly from a directory
dataloaders = create_dataloaders_from_directory(
    data_dir="path/to/molecules",
    labels_file="path/to/labels.csv",
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Working with Hierarchical Graphs

```python
from moml.data import load_hierarchical_dataset

# Load a hierarchical dataset
dataset = load_hierarchical_dataset(
    data_dir="path/to/hierarchical_molecules",
    levels=["atom", "functional_group", "structural_motif"]
)
```
