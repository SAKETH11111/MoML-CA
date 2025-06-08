import os
import sys
import yaml
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset

def compute_stats():
    """
    Computes the mean and standard deviation of the PFAS dataset targets
    and updates the target_stats.yaml file.
    """
    print("Loading PFAS dataset to compute target statistics...")
    dataset = get_dataset("pfas", root="data")
    loader = GraphDataLoader(dataset, batch_size=512, shuffle=False)

    all_targets = []
    for batch in loader:
        all_targets.append(batch.y)

    if not all_targets:
        print("Dataset is empty. No stats to compute.")
        return

    # Concatenate all target tensors
    all_targets_tensor = torch.cat(all_targets, dim=0)

    # Compute mean and std
    mean = torch.mean(all_targets_tensor, dim=0)
    std = torch.std(all_targets_tensor, dim=0)

    # Ensure std is not zero to avoid division by zero
    std[std == 0] = 1.0

    print(f"Computed Mean: {mean.tolist()}")
    print(f"Computed Std Dev: {std.tolist()}")

    # Update the stats file
    stats_path = "data/target_stats.yaml"
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
    else:
        stats = {}

    stats['pfas'] = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, indent=4)

    print(f"Successfully updated {stats_path} with PFAS target statistics.")

if __name__ == "__main__":
    compute_stats()