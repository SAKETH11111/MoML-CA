import os
import sys
import argparse
import yaml
import torch
import glob
import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from torchvision.transforms import Compose
from torch_geometric.data import Dataset

class SubsetWrapper(Dataset):
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

def evaluate_checkpoint(ckpt_path, config, device, val_loader, stats):
    """Evaluates a single checkpoint and returns its scaled loss."""
    mgnn_config = config.get('mgnn', {})
    model = DJMGNN(
        in_node_dim=29,
        in_edge_dim=mgnn_config.get('in_edge_dim', 0),
        node_output_dims=mgnn_config.get('node_output_dims', 3),
        graph_output_dims=mgnn_config.get('graph_output_dims', 19),
        energy_output_dims=mgnn_config.get('energy_output_dims', 1),
        hidden_dim=mgnn_config.get('hidden_channels', 128),
        n_blocks=mgnn_config.get('num_layers', 4)
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            all_preds_scaled.append(out['graph_pred'])
            all_targets_scaled.append(batch.y)

    all_preds_scaled = torch.cat(all_preds_scaled, dim=0)
    all_targets_scaled = torch.cat(all_targets_scaled, dim=0)
    
    scaled_loss = torch.nn.functional.mse_loss(all_preds_scaled, all_targets_scaled).item()
    return scaled_loss

def main():
    parser = argparse.ArgumentParser(description='Find the best DJMGNN checkpoint on the QM9 validation set.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (cuda/cpu).')
    parser.add_argument('--config_path', type=str, default='config/training_config.template.yaml', help='Path to training config YAML file')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Dataset ---
    print(f"Loading QM9 {args.split} split...")
    transform = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name="qm9")])
    full_dataset = get_dataset("qm9", root="MoML-CA/data", transform=transform)
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    val_indices = shuffled_indices[train_size : train_size + val_size]
    val_subset = torch.utils.data.Subset(full_dataset, val_indices.tolist())
    val_dataset = SubsetWrapper(val_subset)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Load Target Statistics ---
    stats_path = "data/target_stats.yaml"
    with open(stats_path, 'r') as f:
        stats = yaml.safe_load(f)

    # --- Find and Evaluate All Checkpoints ---
    checkpoint_paths = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_step_*.pt'))
    checkpoint_paths.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
    if not checkpoint_paths:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    best_loss = float('inf')
    best_ckpt_path = None

    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate.")
    for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        step = int(ckpt_path.split('_')[-1].split('.')[0])
        loss = evaluate_checkpoint(ckpt_path, config, device, val_loader, stats)
        print(f"  - Step {step:6d}: Scaled Loss (MSE) = {loss:.6f}")
        if loss < best_loss:
            best_loss = loss
            best_ckpt_path = ckpt_path

    print("\n--- Best Checkpoint Found ---")
    print(f"Path: {best_ckpt_path}")
    print(f"Scaled Loss (MSE): {best_loss:.6f}")
    print("-----------------------------")

if __name__ == "__main__":
    main()