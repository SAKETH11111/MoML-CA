from typing import Dict, Any, Optional, Union
import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9
from torch.utils.data import ConcatDataset

from .spice_dataset import SpiceDataset
from .pfas_sdf_dataset import PFASSDFDataset


def get_dataset(
    dataset_name: str,
    root: str = "data/",
    split: Optional[str] = None,
    **kwargs
) -> Union[Dataset, ConcatDataset]:
    """
    Get dataset by name with unified interface.
    
    Args:
        dataset_name: Name of dataset ('qm9', 'spice', 'pfas', 'qm9+pfas')
        root: Root directory for datasets
        split: Dataset split ('train', 'val', 'test') - not supported by all datasets
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Dataset instance or ConcatDataset for combined datasets
    """
    
    if dataset_name.lower() == 'qm9':
        # QM9 doesn't support split parameter, so filter it out
        qm9_kwargs = {k: v for k, v in kwargs.items() if k != 'split'}
        dataset = QM9(root=f"{root}/qm9", **qm9_kwargs)
        print(f"Loaded QM9 dataset with {len(dataset)} molecules")
        return dataset
        
    elif dataset_name.lower() == 'spice':
        dataset = SpiceDataset(root=f"{root}/spice", split=split, **kwargs)
        print(f"Loaded SPICE dataset with {len(dataset)} samples")
        return dataset
        
    elif dataset_name.lower() == 'pfas':
        dataset = PFASSDFDataset(root=f"{root}/diverse_pfas_sdf_batch", split=split, **kwargs)
        print(f"Loaded PFAS dataset with {len(dataset)} molecules")
        return dataset
        
    elif dataset_name.lower() == 'qm9+pfas':
        # Combine QM9 and PFAS for graph-level tasks
        qm9_kwargs = {k: v for k, v in kwargs.items() if k != 'split'}
        qm9_dataset = QM9(root=f"{root}/qm9", **qm9_kwargs)
        pfas_dataset = PFASSDFDataset(root=f"{root}/diverse_pfas_sdf_batch", split=split, **kwargs)
        
        combined_dataset = ConcatDataset([qm9_dataset, pfas_dataset])
        print(f"Combined dataset: QM9 ({len(qm9_dataset)}) + PFAS ({len(pfas_dataset)}) = {len(combined_dataset)} total")
        return combined_dataset
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'qm9', 'spice', 'pfas', 'qm9+pfas'") 