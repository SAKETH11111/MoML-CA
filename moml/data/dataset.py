from typing import Optional, Union, Callable
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9 
from torch.utils.data import DataLoader

from .spice_dataset import SpiceDataset
from .pfas_sdf_dataset import PFASSDFDataset
from .feature_transforms import CreateEdges, FeaturizeNodes


def get_dataset(
    dataset_name: str,
    root: str = "data/",
    split: Optional[str] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    **kwargs
) -> Dataset:
    """
    Get dataset by name with unified interface.
    
    Args:
        dataset_name: Name of dataset ('qm9', 'spice')
        root: Root directory for datasets
        split: Dataset split ('train', 'val', 'test') - not supported by all datasets
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Dataset instance
    """
    
    if dataset_name.lower() == 'qm9':
        # QM9 doesn't support split parameter, so filter it out
        qm9_kwargs = {k: v for k, v in kwargs.items() if k != 'split'}
        dataset = QM9(root=f"{root}/qm9", pre_transform=pre_transform, **qm9_kwargs)
        print(f"Loaded QM9 dataset with {len(dataset)} molecules")
        return dataset
        
    elif dataset_name.lower() == 'spice':
        dataset = SpiceDataset(root=f"{root}/spice", split=split, pre_transform=pre_transform, **kwargs)
        print(f"Loaded SPICE dataset with {len(dataset)} samples")
        return dataset

    elif dataset_name.lower() == 'pfas':
        dataset = PFASSDFDataset(root=f"{root}/pfas", split=split, pre_transform=pre_transform, **kwargs)
        print(f"Loaded PFAS dataset with {len(dataset)} samples")
        return dataset
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'qm9', 'spice', 'pfas'")