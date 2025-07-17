"""
moml/utils/dataset_utils.py

This module provides custom dataset utilities, including wrappers that adapt
standard PyTorch datasets for compatibility with other libraries like
PyTorch Geometric.
"""

from torch.utils.data import Subset
from torch_geometric.data import Dataset


class SubsetWrapper(Dataset):
    """
    A wrapper to make a `torch.utils.data.Subset` compatible with PyG DataLoaders.

    PyTorch Geometric's DataLoader expects a `torch_geometric.data.Dataset` object.
    This wrapper takes a standard `torch.utils.data.Subset` and exposes the
    necessary interface (`__len__` and `__getitem__`) to make it work seamlessly
    with PyG's data loading and batching system.

    Attributes:
        subset (Subset): The `torch.utils.data.Subset` instance being wrapped.
    """

    def __init__(self, subset: Subset):
        """
        Initializes the SubsetWrapper.

        Args:
            subset (Subset): The PyTorch Subset to be wrapped.
        """
        super().__init__()
        self.subset = subset

    def __len__(self) -> int:
        """Returns the number of samples in the subset."""
        return len(self.subset)

    def __getitem__(self, idx: int):
        """Retrieves the sample at the specified index from the subset."""
        return self.subset[idx]