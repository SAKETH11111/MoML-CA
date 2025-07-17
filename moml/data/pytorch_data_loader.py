"""
moml/data/pytorch_data_loader.py

PyTorch DataLoader utilities for molecular graph data.

This module provides comprehensive utilities for creating and managing PyTorch
DataLoader objects specifically designed for molecular graph datasets. It includes
functions for preparing dataloaders from datasets, loading from directories with
automatic splitting, and creating stratified splits for classification tasks.

Functions:
    prepare_dataloaders: Create dataloaders from existing datasets
    create_dataloaders_from_directory: Load and split data from directory
    create_stratified_dataloaders: Create dataloaders with stratified splitting

Example:
    Basic usage for creating dataloaders:
    
    >>> from moml.data.pytorch_data_loader import create_dataloaders_from_directory
    >>> dataloaders = create_dataloaders_from_directory(
    ...     data_dir='molecules/',
    ...     batch_size=64,
    ...     train_ratio=0.8
    ... )
    >>> train_loader = dataloaders['train']
    >>> for batch in train_loader:
    ...     # Process batch of molecular graphs
    ...     pass
"""

import logging
import os
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

# Constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
DEFAULT_FILE_PATTERN = "*.mol"

# Configure module logger
logger = logging.getLogger(__name__)

try:
    from moml.core.molecular_graph_processor import collate_graphs  # type: ignore
    HAS_COLLATE = True
except ImportError:
    HAS_COLLATE = False
    logger.warning("Molecular graph collate function not available")
    
    def collate_graphs(batch):  # type: ignore
        """Dummy collate function when not available."""
        return batch

try:
    from moml.data.dataset_loader import load_dataset  # type: ignore
    from moml.data.dataset_splitter import split_dataset  # type: ignore
    HAS_DATASET_UTILS = True
except ImportError:
    HAS_DATASET_UTILS = False
    logger.warning("Dataset utilities not available")
    
    def load_dataset(*args, **kwargs):  # type: ignore
        """Dummy load function when not available."""
        raise ImportError("Dataset loading utilities not available")
    
    def split_dataset(*args, **kwargs):  # type: ignore
        """Dummy split function when not available."""
        raise ImportError("Dataset splitting utilities not available")


def prepare_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: bool = True,
) -> Dict[str, DataLoader]:
    """
    Prepare PyTorch DataLoaders for training, validation, and testing.
    
    This function creates DataLoader objects for the provided datasets with
    appropriate configurations for each split. Training data is shuffled by
    default, while validation and test data are not shuffled to ensure
    reproducible evaluation.

    Args:
        train_dataset: Training dataset (required)
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data (validation/test never shuffled)

    Returns:
        Dictionary containing DataLoader objects for each split:
        - 'train': Training DataLoader (always present)
        - 'val': Validation DataLoader (if val_dataset provided)
        - 'test': Test DataLoader (if test_dataset provided)
    
    Example:
        >>> train_ds = MyDataset('train')
        >>> val_ds = MyDataset('val')
        >>> loaders = prepare_dataloaders(train_ds, val_ds, batch_size=64)
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
    
    Raises:
        ValueError: If batch_size or num_workers are invalid
        ImportError: If collate function is not available
    """
    # Validate parameters
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    
    if train_dataset is None:
        raise ValueError("train_dataset is required")
    
    logger.info(
        f"Preparing dataloaders: batch_size={batch_size}, "
        f"num_workers={num_workers}, shuffle={shuffle}"
    )
    
    # Create training dataloader
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_graphs
        )
    }
    
    logger.debug(f"Created training dataloader with {len(train_dataset)} samples")  # type: ignore

    # Create validation dataloader if dataset provided
    if val_dataset is not None:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle validation data
            num_workers=num_workers,
            collate_fn=collate_graphs
        )
        logger.debug(f"Created validation dataloader with {len(val_dataset)} samples")  # type: ignore

    # Create test dataloader if dataset provided
    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle test data
            num_workers=num_workers,
            collate_fn=collate_graphs
        )
        logger.debug(f"Created test dataloader with {len(test_dataset)} samples")  # type: ignore

    return dataloaders


def create_dataloaders_from_directory(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from a directory of molecular files.

    This convenience function handles the complete pipeline from directory
    to DataLoaders: loading datasets, splitting data, and creating loaders.
    It supports both pre-split directory structures and automatic splitting.
    
    Directory Structure Options:
    1. Pre-split: data_dir/{train,val,test}/ subdirectories
    2. Single directory: All files in data_dir, automatically split

    Args:
        data_dir: Directory containing molecule files or train/val/test subdirs
        labels_file: Optional path to file containing molecular labels
        file_pattern: Glob pattern to match molecule files
        config: Optional configuration dictionary for graph processing
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        train_ratio: Fraction of data for training (ignored if pre-split)
        val_ratio: Fraction of data for validation (ignored if pre-split)
        test_ratio: Fraction of data for testing (ignored if pre-split)
        random_seed: Random seed for reproducible splitting
    
    Returns:
        Dictionary containing DataLoader objects for available splits
    
    Example:
        Pre-split directory structure:
        >>> loaders = create_dataloaders_from_directory(
        ...     'data/molecules/',  # Contains train/, val/, test/ subdirs
        ...     batch_size=64
        ... )
        
        Single directory with automatic splitting:
        >>> loaders = create_dataloaders_from_directory(
        ...     'data/all_molecules/',
        ...     train_ratio=0.7,
        ...     val_ratio=0.2,
        ...     test_ratio=0.1
        ... )
    
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If split ratios don't sum to 1.0
        ImportError: If dataset utilities are not available
    """
    if not HAS_DATASET_UTILS:
        raise ImportError("Dataset utilities required for directory loading")
    
    # Validate directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data path is not a directory: {data_dir}")
    
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    
    logger.info(f"Creating dataloaders from directory: {data_dir}")
    
    # Check for pre-split directory structure
    if _has_presplit_structure(data_dir):
        return _create_dataloaders_presplit(
            data_dir, labels_file, file_pattern, config, 
            batch_size, num_workers
        )
    else:
        return _create_dataloaders_with_splitting(
            data_dir, labels_file, file_pattern, config,
            batch_size, num_workers, train_ratio, val_ratio, 
            test_ratio, random_seed
        )


def _has_presplit_structure(data_dir: str) -> bool:
    """
    Check if directory has pre-split structure (train/ subdirectory exists).
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if train/ subdirectory exists
    """
    train_dir = os.path.join(data_dir, "train")
    return os.path.isdir(train_dir)


def _create_dataloaders_presplit(
    data_dir: str,
    labels_file: Optional[str],
    file_pattern: str,
    config: Optional[Dict[str, Any]],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders from pre-split directory structure.
    
    Args:
        data_dir: Root directory containing train/, val/, test/ subdirs
        labels_file: Optional labels file path
        file_pattern: File pattern for matching molecules
        config: Graph processing configuration
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        Dictionary of dataloaders
    """
    logger.info("Detected pre-split directory structure")
    
    # Define subdirectory paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Load training dataset (required)
    train_dataset = load_dataset(train_dir, labels_file, file_pattern, config)
    logger.info(f"Loaded training dataset: {len(train_dataset)} samples")
    
    # Load validation dataset if directory exists
    val_dataset = None
    if os.path.isdir(val_dir):
        val_dataset = load_dataset(val_dir, labels_file, file_pattern, config)
        logger.info(f"Loaded validation dataset: {len(val_dataset)} samples")

    # Load test dataset if directory exists
    test_dataset = None
    if os.path.isdir(test_dir):
        test_dataset = load_dataset(test_dir, labels_file, file_pattern, config)
        logger.info(f"Loaded test dataset: {len(test_dataset)} samples")

    # Create and return dataloaders
    return prepare_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def _create_dataloaders_with_splitting(
    data_dir: str,
    labels_file: Optional[str],
    file_pattern: str,
    config: Optional[Dict[str, Any]],
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders with automatic dataset splitting.
    
    Args:
        data_dir: Directory containing all molecule files
        labels_file: Optional labels file path
        file_pattern: File pattern for matching molecules
        config: Graph processing configuration
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        random_seed: Random seed for splitting
        
    Returns:
        Dictionary of dataloaders
    """
    logger.info("Loading dataset for automatic splitting")
    
    # Load complete dataset
    dataset = load_dataset(data_dir, labels_file, file_pattern, config)
    logger.info(f"Loaded complete dataset: {len(dataset)} samples")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(  # type: ignore
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Log split sizes
    logger.info(
        f"Split dataset - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset) if val_dataset else 0}, "
        f"Test: {len(test_dataset) if test_dataset else 0}"
        )

    # Create and return dataloaders
    return prepare_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def create_stratified_dataloaders(
    dataset: Dataset,
    labels: List[Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders with stratified dataset splitting.
    
    This function creates dataloaders using stratified splitting to ensure
    that the distribution of class labels is preserved across train, validation,
    and test sets. This is particularly important for classification tasks with
    imbalanced datasets.

    Args:
        dataset: The complete dataset to split
        labels: List of labels corresponding to dataset samples for stratification
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducible stratified splitting

    Returns:
        Dictionary containing DataLoader objects for each split
    
    Example:
        >>> dataset = MyClassificationDataset()
        >>> labels = [0, 1, 0, 1, 1, 0, 1, 0]  # Binary classification
        >>> loaders = create_stratified_dataloaders(
        ...     dataset, labels,
        ...     batch_size=32,
        ...     train_ratio=0.7
        ... )
        >>> # Class distribution preserved in each split
    
    Raises:
        ValueError: If labels length doesn't match dataset size
        ValueError: If split ratios don't sum to 1.0
        ImportError: If stratified splitting utilities are not available
    """
    # Validate inputs
    if len(labels) != len(dataset):  # type: ignore
        raise ValueError(
            f"Labels length ({len(labels)}) must match dataset size ({len(dataset)})"  # type: ignore
        )
    
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    
    logger.info(
        f"Creating stratified dataloaders for {len(dataset)} samples "  # type: ignore
        f"with {len(set(labels))} unique labels"
    )
    
    # Import stratified splitting function
    try:
        from moml.data.dataset_splitter import stratified_split_dataset  # type: ignore 
    except ImportError:
        raise ImportError(
            "Stratified splitting utilities not available. "
            "Please install required dependencies."
        )
    
    # Perform stratified split
    train_dataset, val_dataset, test_dataset = stratified_split_dataset(
        dataset,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Log split information
    logger.info(
        f"Stratified split completed - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset) if val_dataset else 0}, "
        f"Test: {len(test_dataset) if test_dataset else 0}"
    )

    # Create and return dataloaders
    return prepare_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )