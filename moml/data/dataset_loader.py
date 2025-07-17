"""
moml/data/dataset_loader.py

Dataset loading utilities for molecular machine learning applications.

This module provides functions for loading molecular datasets from files and
directories, supporting various data formats and label configurations. It
integrates with the MoML dataset classes to create standardized molecular
graph datasets for training and evaluation.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from moml.data.datasets import HierarchicalGraphDataset, MolecularGraphDataset

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FILE_PATTERNS = ["*.mol", "*.sdf", "*.pdb", "*.mol2"]
DEFAULT_FILE_PATTERN = "*.mol"


def load_dataset(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    config: Optional[Dict[str, Any]] = None,
) -> MolecularGraphDataset:
    """
    Load a molecular dataset from a directory of molecule files.

    Searches for molecule files matching the specified pattern and optionally
    loads corresponding labels from a CSV file. Creates a MolecularGraphDataset
    instance ready for machine learning applications.

    Args:
        data_dir: Directory containing molecule files (.mol, .sdf, etc.).
        labels_file: Optional path to CSV file containing labels. The file
            should have a 'filename' column and at least one label column.
        file_pattern: Pattern to match molecule files (e.g., '*.mol', '*.sdf').
            Can be comma-separated for multiple patterns.
        config: Optional configuration dictionary for graph processing,
            passed to the MolecularGraphDataset constructor.

    Returns:
        MolecularGraphDataset instance containing the loaded molecules
        and labels.

    Raises:
        ValueError: If no files are found matching the pattern or if
            required columns are missing from the labels file.
        FileNotFoundError: If the data directory doesn't exist.
        pd.errors.ParserError: If the labels file cannot be parsed.

    Example:
        >>> dataset = load_dataset(
        ...     data_dir="data/molecules/",
        ...     labels_file="data/labels.csv",
        ...     file_pattern="*.sdf",
        ...     config={"use_3d": True}
        ... )
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all molecule files matching the pattern(s)
    mol_files = []
    patterns = [p.strip() for p in file_pattern.split(",")]
    
    for pattern in patterns:
        matching_files = list(data_path.glob(pattern))
        mol_files.extend([str(f) for f in matching_files])

    if not mol_files:
        raise ValueError(
            f"No files found in {data_dir} matching pattern(s): {file_pattern}"
        )

    logger.info(f"Found {len(mol_files)} molecule files in {data_dir}")

    # Load labels if provided
    labels = None
    if labels_file and Path(labels_file).exists():
        labels = _load_labels_from_csv(labels_file, data_dir)

    # Create and return dataset
    return MolecularGraphDataset(
        mol_files=mol_files,
        labels=labels,
        config=config
    )


def load_datasets_from_splits(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, MolecularGraphDataset]:
    """
    Load molecular datasets from pre-split directories.

    Expects the data directory to contain subdirectories named 'train',
    'val' (optional), and 'test' (optional). Each subdirectory should
    contain molecule files for that split.

    Args:
        data_dir: Root directory containing split subdirectories
            (train/, val/, test/).
        labels_file: Optional path to CSV file containing labels for
            all splits.
        file_pattern: Pattern to match molecule files in each split
            directory.
        config: Optional configuration dictionary for graph processing.

    Returns:
        Dictionary mapping split names ('train', 'val', 'test') to
        MolecularGraphDataset instances.

    Raises:
        ValueError: If the training directory is not found (required).
        FileNotFoundError: If the root data directory doesn't exist.

    Example:
        >>> datasets = load_datasets_from_splits(
        ...     data_dir="data/",
        ...     labels_file="data/all_labels.csv",
        ...     file_pattern="*.mol"
        ... )
        >>> train_data = datasets['train']
        >>> val_data = datasets.get('val')  # May be None
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    datasets = {}

    # Define split directories
    split_dirs = {
        "train": data_path / "train",
        "val": data_path / "val",
        "test": data_path / "test",
    }

    # Load training dataset (required)
    train_dir = split_dirs["train"]
    if not train_dir.is_dir():
        raise ValueError(f"Training directory not found: {train_dir}")

    datasets["train"] = load_dataset(
        data_dir=str(train_dir),
        labels_file=labels_file,
        file_pattern=file_pattern,
        config=config
    )

    # Load validation dataset (optional)
    val_dir = split_dirs["val"]
    if val_dir.is_dir():
        datasets["val"] = load_dataset(
            data_dir=str(val_dir),
            labels_file=labels_file,
            file_pattern=file_pattern,
            config=config
        )

    # Load test dataset (optional)
    test_dir = split_dirs["test"]
    if test_dir.is_dir():
        datasets["test"] = load_dataset(
            data_dir=str(test_dir),
            labels_file=labels_file,
            file_pattern=file_pattern,
            config=config
        )

    logger.info(
        f"Loaded datasets with splits: {list(datasets.keys())} "
        f"from {data_dir}"
    )

    return datasets


def load_hierarchical_dataset(
    data_dir: str,
    levels: Optional[List[str]] = None,
    labels_file: Optional[str] = None,
) -> Dict[str, HierarchicalGraphDataset]:
    """
    Load hierarchical molecular graph datasets from a directory.

    Creates datasets for different hierarchy levels of molecular
    representations (e.g., atom-level, functional group-level,
    structural motif-level).

    Args:
        data_dir: Directory containing hierarchical graph files.
            Should have subdirectories for each hierarchy level.
        levels: List of hierarchy level names to include. Defaults to
            ['atom', 'functional_group', 'structural_motif'].
        labels_file: Optional path to CSV file containing labels.
            Should have molecule ID and label columns.

    Returns:
        Dictionary mapping hierarchy level names to
        HierarchicalGraphDataset instances.

    Raises:
        ValueError: If no valid hierarchy levels are found.
        FileNotFoundError: If the data directory doesn't exist.

    Example:
        >>> datasets = load_hierarchical_dataset(
        ...     data_dir="data/hierarchical/",
        ...     levels=["atom", "functional_group"],
        ...     labels_file="data/labels.csv"
        ... )
        >>> atom_dataset = datasets['atom']
        >>> fg_dataset = datasets['functional_group']
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if levels is None:
        levels = ["atom", "functional_group", "structural_motif"]

    # Load labels if provided
    labels = None
    if labels_file and Path(labels_file).exists():
        labels = _load_hierarchical_labels_from_csv(labels_file)

    # Create datasets for each level
    datasets = {}
    for level in levels:
        try:
            dataset = HierarchicalGraphDataset(
                data_dir=data_dir,
                labels=labels,
                levels=[level],  # Only include this specific level
                transform=None
            )
            datasets[level] = dataset
            logger.info(f"Created hierarchical dataset for level: {level}")
        except Exception as e:
            logger.warning(f"Failed to create dataset for level {level}: {e}")

    if not datasets:
        raise ValueError(f"No valid hierarchy levels found in {data_dir}")

    return datasets


def _load_labels_from_csv(
    labels_file: str, 
    data_dir: str
) -> Dict[str, float]:
    """
    Load labels from a CSV file and map them to molecule file paths.

    Args:
        labels_file: Path to the CSV file containing labels.
        data_dir: Directory containing the molecule files.

    Returns:
        Dictionary mapping full file paths to label values.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(labels_file)
        logger.info(f"Loaded labels file with {len(df)} entries")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing labels file {labels_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading labels file {labels_file}: {e}")
        raise

    # Validate required columns
    if "filename" not in df.columns:
        raise ValueError("Labels file must contain a 'filename' column")

    if len(df.columns) < 2:
        raise ValueError("Labels file must contain at least one label column")

    # Determine label column (first non-filename column)
    label_columns = [col for col in df.columns if col != "filename"]
    label_col = label_columns[0]

    labels = {}
    data_path = Path(data_dir)

    for _, row in df.iterrows():
        filename = row["filename"]
        label_value = row[label_col]

        # Create full path to molecule file
        full_path = data_path / filename
        if full_path.exists():
            labels[str(full_path)] = float(label_value)
        else:
            logger.warning(f"Label file references non-existent file: {filename}")

    logger.info(f"Successfully mapped {len(labels)} labels to molecule files")
    return labels


def _load_hierarchical_labels_from_csv(labels_file: str) -> Dict[str, float]:
    """
    Load labels for hierarchical datasets from a CSV file.

    Args:
        labels_file: Path to the CSV file containing labels.

    Returns:
        Dictionary mapping molecule IDs to label values.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(labels_file)
        logger.info(f"Loaded hierarchical labels file with {len(df)} entries")
    except Exception as e:
        logger.error(f"Error loading hierarchical labels file {labels_file}: {e}")
        raise

    if len(df.columns) < 2:
        raise ValueError(
            "Hierarchical labels file must contain at least "
            "molecule ID and label columns"
        )

    # Use first column as molecule ID and second as label
    id_col = df.columns[0]
    label_col = df.columns[1]

    labels = {}
    for _, row in df.iterrows():
        mol_id = str(row[id_col])
        label_value = float(row[label_col])
        labels[mol_id] = label_value

    logger.info(f"Successfully loaded {len(labels)} hierarchical labels")
    return labels
