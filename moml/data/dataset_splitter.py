"""
moml/data/dataset_splitter.py

Dataset splitting utilities for molecular machine learning applications.

This module provides functions for splitting datasets into training, validation,
and test sets using various strategies including random splitting, stratified
splitting, and scaffold-based molecular splitting for fair evaluation of
molecular property prediction models.
"""

import logging
from typing import Any, List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
RATIO_TOLERANCE = 0.001  # Tolerance for floating-point ratio validation


def split_dataset(
    dataset: Dataset[Any],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """
    Split a dataset into training, validation, and test sets using random sampling.

    Performs a random split of the dataset while maintaining the specified
    proportions. Useful for general machine learning tasks where data points
    are independent and identically distributed.

    Args:
        dataset: The PyTorch dataset to split.
        train_ratio: Fraction of data for training (0.0 to 1.0).
        val_ratio: Fraction of data for validation (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).
        random_seed: Random seed for reproducible splits.

    Returns:
        Tuple containing (train_dataset, val_dataset, test_dataset)
        as PyTorch Subset objects.

    Raises:
        ValueError: If ratios don't sum to 1.0 or if dataset is empty.

    Example:
        >>> train_data, val_data, test_data = split_dataset(
        ...     dataset=my_dataset,
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15,
        ...     random_seed=42
        ... )
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    
    if len(dataset) == 0:  # type: ignore
        raise ValueError("Cannot split an empty dataset")

    # Get indices for the full dataset
    indices = list(range(len(dataset)))  # type: ignore

    # First split: (train+val) vs test
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=test_ratio, 
        random_state=random_seed
    )

    # Second split: train vs val from the remaining data
    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=adjusted_val_ratio,
            random_state=random_seed
        )
    else:
        train_indices = train_val_indices
        val_indices = []

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(
        f"Dataset split complete: {len(train_indices)} train, "
        f"{len(val_indices)} val, {len(test_indices)} test samples"
    )

    return train_dataset, val_dataset, test_dataset


def stratified_split_dataset(
    dataset: Dataset[Any],
    labels: List[int],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """
    Split a dataset using stratified sampling to maintain label distributions.

    Ensures that each split (train/val/test) maintains approximately the same
    proportion of samples for each class as the original dataset. Particularly
    useful for imbalanced datasets or classification tasks.

    Args:
        dataset: The PyTorch dataset to split.
        labels: List of integer labels for stratification. Must have the
            same length as the dataset.
        train_ratio: Fraction of data for training (0.0 to 1.0).
        val_ratio: Fraction of data for validation (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).
        random_seed: Random seed for reproducible splits.

    Returns:
        Tuple containing (train_dataset, val_dataset, test_dataset)
        as PyTorch Subset objects.

    Raises:
        ValueError: If ratios don't sum to 1.0, if labels length doesn't
            match dataset length, or if dataset is empty.

    Example:
        >>> labels = [0, 1, 0, 1, 1, 0]  # Binary classification labels
        >>> train_data, val_data, test_data = stratified_split_dataset(
        ...     dataset=my_dataset,
        ...     labels=labels,
        ...     train_ratio=0.8,
        ...     val_ratio=0.1,
        ...     test_ratio=0.1
        ... )
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    
    if len(dataset) == 0:  # type: ignore
        raise ValueError("Cannot split an empty dataset")

    if len(labels) != len(dataset):  # type: ignore
        raise ValueError(
            f"Length of labels ({len(labels)}) must match "
            f"length of dataset ({len(dataset)})"  # type: ignore
        )

    # Get indices for the full dataset
    indices = list(range(len(dataset)))  # type: ignore

    # First split: (train+val) vs test with stratification
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=[labels[i] for i in indices],
        random_state=random_seed
    )

    # Second split: train vs val with stratification
    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=adjusted_val_ratio,
            stratify=[labels[i] for i in train_val_indices],
            random_state=random_seed
        )
    else:
        train_indices = train_val_indices
        val_indices = []

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(
        f"Stratified dataset split complete: {len(train_indices)} train, "
        f"{len(val_indices)} val, {len(test_indices)} test samples"
    )

    return train_dataset, val_dataset, test_dataset


def scaffold_split_dataset(
    dataset: Dataset[Any],
    smiles_list: List[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """
    Split a dataset based on molecular scaffolds for fair molecular evaluation.

    Uses Murcko scaffolds to ensure that molecules with similar structural
    frameworks are not split across training and test sets. This provides
    a more realistic evaluation of model generalization to new molecular
    scaffolds.

    Args:
        dataset: The PyTorch dataset to split.
        smiles_list: List of SMILES strings corresponding to dataset entries.
            Must have the same length as the dataset.
        train_ratio: Fraction of data for training (0.0 to 1.0).
        val_ratio: Fraction of data for validation (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).
        random_seed: Random seed for reproducible splits (currently unused
            but kept for API consistency).

    Returns:
        Tuple containing (train_dataset, val_dataset, test_dataset)
        as PyTorch Subset objects.

    Raises:
        ImportError: If RDKit is not available for scaffold generation.
        ValueError: If ratios don't sum to 1.0, if SMILES list length
            doesn't match dataset length, or if dataset is empty.

    Example:
        >>> smiles = ["CCO", "CCC", "c1ccccc1", "c1cccnc1"]
        >>> train_data, val_data, test_data = scaffold_split_dataset(
        ...     dataset=my_dataset,
        ...     smiles_list=smiles,
        ...     train_ratio=0.8,
        ...     val_ratio=0.1,
        ...     test_ratio=0.1
        ... )
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except ImportError as e:
        raise ImportError(
            "RDKit is required for scaffold splitting. "
            "Please install RDKit: conda install -c conda-forge rdkit"
        ) from e

    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    
    if len(dataset) == 0:  # type: ignore
        raise ValueError("Cannot split an empty dataset")

    if len(smiles_list) != len(dataset):  # type: ignore
        raise ValueError(
            f"Length of SMILES list ({len(smiles_list)}) must match "
            f"length of dataset ({len(dataset)})"  # type: ignore
        )

    # Generate scaffolds for each molecule
    scaffolds = {}
    invalid_count = 0

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES at index {i}: '{smiles}'")
            scaffold = f"invalid_smiles_{i}"
            invalid_count += 1
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except Exception as e:
                logger.warning(
                    f"Error generating scaffold for SMILES at index {i} "
                    f"('{smiles}'): {e}"
                )
                scaffold = f"scaffold_error_{i}"

        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)

    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid SMILES strings")

    # Sort scaffold groups by size (largest first) for better balance
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    # Calculate target sizes for each split
    total_size = len(dataset)  # type: ignore
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Assign scaffolds to splits using greedy algorithm
    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold_indices in scaffold_groups:
        if len(train_indices) + len(scaffold_indices) <= train_size:
            train_indices.extend(scaffold_indices)
        elif len(val_indices) + len(scaffold_indices) <= val_size:
            val_indices.extend(scaffold_indices)
        else:
            test_indices.extend(scaffold_indices)

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(
        f"Scaffold-based dataset split complete: "
        f"{len(train_indices)} train ({len(train_indices)/total_size:.1%}), "
        f"{len(val_indices)} val ({len(val_indices)/total_size:.1%}), "
        f"{len(test_indices)} test ({len(test_indices)/total_size:.1%}) samples"
    )
    logger.info(f"Generated {len(scaffolds)} unique molecular scaffolds")

    return train_dataset, val_dataset, test_dataset


def _validate_split_ratios(
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float
) -> None:
    """
    Validate that split ratios are valid and sum to 1.0.

    Args:
        train_ratio: Training data fraction.
        val_ratio: Validation data fraction.
        test_ratio: Test data fraction.

    Raises:
        ValueError: If ratios don't sum to 1.0 within tolerance or if
            any ratio is negative.
    """
    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be non-negative")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not (1.0 - RATIO_TOLERANCE <= total_ratio <= 1.0 + RATIO_TOLERANCE):
        raise ValueError(
            f"Train, validation, and test ratios must sum to 1.0. "
            f"Current sum: {total_ratio:.4f}"
        )