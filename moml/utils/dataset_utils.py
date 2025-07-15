from torch_geometric.data import Dataset

class SubsetWrapper(Dataset):
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]