"""
PyTorch DataLoader Utilities

This module provides utilities for creating and managing PyTorch DataLoader objects.
"""

import os
from typing import Dict, Optional, Any

from torch.utils.data import Dataset, DataLoader

from moml.core.molecular_graph_processor import collate_graphs
from moml.data.dataset_loader import load_dataset
from moml.data.dataset_splitter import split_dataset


def prepare_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Dict[str, DataLoader]:
    """
    Prepare dataloaders for training, validation, and testing.

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the training data

    Returns:
        Dictionary of dataloaders for each set
    """
    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_graphs
        )
    }

    # Validation dataloader
    if val_dataset is not None:
        dataloaders["val"] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graphs
        )

    # Test dataloader
    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graphs
        )

    return dataloaders


def create_dataloaders_from_directory(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = "*.mol",
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders from a directory of files.

    This is a convenience function that combines loading a dataset,
    splitting it, and creating dataloaders.

    Args:
        data_dir: Directory containing molecule files
        labels_file: Optional path to file containing labels
        file_pattern: Pattern to match molecule files
        config: Configuration for graph processing
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of dataloaders for each set
    """
    # First, check if we have pre-split directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if os.path.isdir(train_dir):
        # We have pre-split directories, load each separately
        train_dataset = load_dataset(train_dir, labels_file, file_pattern, config)
        val_dataset = None
        test_dataset = None

        if os.path.isdir(val_dir):
            val_dataset = load_dataset(val_dir, labels_file, file_pattern, config)

        if os.path.isdir(test_dir):
            test_dataset = load_dataset(test_dir, labels_file, file_pattern, config)

        # Create dataloaders
        return prepare_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    else:
        # We need to load and split the dataset
        dataset = load_dataset(data_dir, labels_file, file_pattern, config)

        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
        )

        # Create dataloaders
        return prepare_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def create_stratified_dataloaders(
    dataset: Dataset,
    labels: list,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders with stratified splitting.

    Args:
        dataset: The dataset to split
        labels: List of labels for stratification
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of dataloaders for each set
    """
    from moml.data.dataset_splitter import stratified_split_dataset

    # Split dataset
    train_dataset, val_dataset, test_dataset = stratified_split_dataset(
        dataset, labels, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
    )

    # Create dataloaders
    return prepare_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
