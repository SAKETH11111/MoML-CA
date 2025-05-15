"""
Dataset Splitting Utilities

This module provides functions for splitting datasets into training, validation, and test sets.
"""

import logging
from typing import List, Tuple
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into training, validation, and test sets.

    Args:
        dataset: The dataset to split
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating-point errors
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0. Current sum: {total_ratio}")

    # Get indices for the full dataset
    indices = list(range(len(dataset)))

    # First split into (train+val, test)
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=random_seed)

    # Then split train_val into (train, val)
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=adjusted_val_ratio, random_state=random_seed
    )

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def stratified_split_dataset(
    dataset: Dataset,
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into training, validation, and test sets with stratification.

    Args:
        dataset: The dataset to split
        labels: List of integer labels for stratification
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating-point errors
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0. Current sum: {total_ratio}")

    # Check that labels match dataset length
    if len(labels) != len(dataset):
        raise ValueError(f"Length of labels ({len(labels)}) must match length of dataset ({len(dataset)})")

    # Get indices for the full dataset
    indices = list(range(len(dataset)))

    # First split into (train+val, test)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, stratify=[labels[i] for i in indices], random_state=random_seed
    )

    # Then split train_val into (train, val)
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_ratio,
        stratify=[labels[i] for i in train_val_indices],
        random_state=random_seed,
    )

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def scaffold_split_dataset(
    dataset: Dataset,
    smiles_list: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset based on molecular scaffolds.

    Args:
        dataset: The dataset to split
        smiles_list: List of SMILES strings corresponding to dataset entries
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise ImportError("RDKit is required for scaffold splitting")

    # Check that SMILES list matches dataset length
    if len(smiles_list) != len(dataset):
        raise ValueError(f"Length of SMILES list ({len(smiles_list)}) must match length of dataset ({len(dataset)})")

    # Generate scaffolds
    scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = ""
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception as e:
                logger.warning(f"Could not generate Murcko scaffold for SMILES {smiles}: {e}")
                scaffold = ""

        if scaffold not in scaffolds:
            scaffolds[scaffold] = [i]
        else:
            scaffolds[scaffold].append(i)

    # Sort scaffolds by size (largest first) to get more balanced splits
    scaffold_sets = [scaffold_indices for (scaffold, scaffold_indices) in scaffolds.items()]
    scaffold_sets.sort(key=len, reverse=True)

    # Calculate split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Assign molecules to each set
    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_size:
            train_indices.extend(scaffold_set)
        elif len(val_indices) + len(scaffold_set) <= val_size:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
