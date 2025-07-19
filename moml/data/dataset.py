"""
moml/data/dataset.py

Unified dataset interface for molecular machine learning applications.

This module provides a simple factory function to load different molecular
datasets with a consistent interface. It supports QM9, SPICE, and PFAS
datasets with unified parameter handling and pre-transform capabilities.
"""

from typing import Callable, Optional, Union

from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

from .feature_transforms import CreateEdges, FeaturizeNodes
from .pfas_sdf_dataset import PFASSDFDataset
from .spice_dataset import SpiceDataset

# Type alias for supported datasets
MolecularDataset = Union[QM9, SpiceDataset, PFASSDFDataset]

# Constants
SUPPORTED_DATASETS = {"qm9", "spice", "pfas"}
DEFAULT_ROOT = "data/"


def get_dataset(
    dataset_name: str,
    root: str = DEFAULT_ROOT,
    split: Optional[str] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    **kwargs
) -> Dataset:
    """
    Get a molecular dataset by name with a unified interface.

    This factory function provides a consistent way to load different
    molecular datasets while handling their specific parameter requirements
    and directory structures.

    Args:
        dataset_name: Name of the dataset to load. Supported values:
            'qm9', 'spice', 'pfas'.
        root: Root directory for datasets. Each dataset will be stored
            in a subdirectory named after the dataset.
        split: Dataset split to load ('train', 'val', 'test'). Not all
            datasets support splitting.
        pre_transform: Optional transform to apply to data objects before
            saving to disk. Applied only once during preprocessing.
        force_reload: Whether to force reloading/reprocessing of the
            dataset even if it exists.
        **kwargs: Additional dataset-specific parameters passed to the
            dataset constructor.

    Returns:
        Dataset instance for the requested molecular dataset.

    Raises:
        ValueError: If an unsupported dataset name is provided.
        RuntimeError: If dataset loading fails due to missing dependencies
            or corrupted data.

    Example:
        >>> # Load QM9 dataset with feature transforms
        >>> dataset = get_dataset(
        ...     dataset_name='qm9',
        ...     root='data/',
        ...     pre_transform=FeaturizeNodes()
        ... )
        >>> print(f"Loaded QM9 dataset with {len(dataset)} molecules")

        >>> # Load SPICE training split
        >>> train_data = get_dataset(
        ...     dataset_name='spice',
        ...     split='train',
        ...     root='data/'
        ... )

        >>> # Load PFAS dataset with custom parameters
        >>> pfas_data = get_dataset(
        ...     dataset_name='pfas',
        ...     root='data/',
        ...     transform=CreateEdges(cutoff=3.0)
        ... )
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )

    if dataset_name_lower == 'qm9':
        return _load_qm9_dataset(
            root=root,
            pre_transform=pre_transform,
            force_reload=force_reload,
            **kwargs
        )
    
    elif dataset_name_lower == 'spice':
        return _load_spice_dataset(
            root=root,
            split=split,
            pre_transform=pre_transform,
            force_reload=force_reload,
            **kwargs
        )
    
    elif dataset_name_lower == 'pfas':
        return _load_pfas_dataset(
            root=root,
            split=split,
            pre_transform=pre_transform,
            force_reload=force_reload,
            **kwargs
        )
    
    else:
        # This should never be reached due to the check above,
        # but included for completeness
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_qm9_dataset(
    root: str,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    **kwargs
) -> QM9:
    """
    Load the QM9 quantum chemistry dataset.

    Args:
        root: Root directory for the dataset.
        pre_transform: Optional preprocessing transform.
        force_reload: Whether to force reload the dataset.
        **kwargs: Additional parameters (split is filtered out as QM9
            doesn't support splitting).

    Returns:
        QM9 dataset instance.
    """
    # QM9 doesn't support split parameter, so filter it out
    qm9_kwargs = {k: v for k, v in kwargs.items() if k != 'split'}
    
    try:
        dataset = QM9(
            root=f"{root}/qm9",
            pre_transform=pre_transform,
            **qm9_kwargs
        )
        print(f"Loaded QM9 dataset with {len(dataset)} molecules")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load QM9 dataset: {e}") from e


def _load_spice_dataset(
    root: str,
    split: Optional[str] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    **kwargs
) -> SpiceDataset:
    """
    Load the SPICE quantum chemistry dataset.

    Args:
        root: Root directory for the dataset.
        split: Dataset split to load.
        pre_transform: Optional preprocessing transform.
        force_reload: Whether to force reload the dataset.
        **kwargs: Additional parameters for SpiceDataset.

    Returns:
        SpiceDataset instance.
    """
    try:
        dataset = SpiceDataset(
            root=f"{root}/spice",
            split=split,
            pre_transform=pre_transform,
            **kwargs
        )
        print(f"Loaded SPICE dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load SPICE dataset: {e}") from e


def _load_pfas_dataset(
    root: str,
    split: Optional[str] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    **kwargs
) -> PFASSDFDataset:
    """
    Load the PFAS molecular dataset.

    Args:
        root: Root directory for the dataset.
        split: Dataset split to load.
        pre_transform: Optional preprocessing transform.
        force_reload: Whether to force reload the dataset.
        **kwargs: Additional parameters for PFASSDFDataset.

    Returns:
        PFASSDFDataset instance.
    """
    try:
        dataset = PFASSDFDataset(
            root=f"{root}/pfas",
            split=split,
            pre_transform=pre_transform,
            **kwargs
        )
        print(f"Loaded PFAS dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load PFAS dataset: {e}") from e