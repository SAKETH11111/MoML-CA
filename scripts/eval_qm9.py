import os
import sys
import argparse
import yaml
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from torchvision.transforms import Compose
from moml.utils.dataset_utils import SubsetWrapper # Import SubsetWrapper

def main():
    parser = argparse.ArgumentParser(description='Evaluate DJMGNN on QM9 validation set.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint.')
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

    # --- Load Model ---
    print(f"Loading model from checkpoint: {args.ckpt}")
    mgnn_config = config.get('mgnn', {})
    model = DJMGNN(
        in_node_dim=29,  # Hardcoded to 29 as per our unified feature dimension
        in_edge_dim=mgnn_config.get('in_edge_dim', 0),
        node_output_dims=mgnn_config.get('node_output_dims', 3),
        graph_output_dims=mgnn_config.get('graph_output_dims', 19),
        energy_output_dims=mgnn_config.get('energy_output_dims', 1),
        hidden_dim=mgnn_config.get('hidden_channels', 128),
        n_blocks=mgnn_config.get('num_layers', 4)
    )
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # --- Load Dataset ---
    print(f"Loading QM9 {args.split} split...")
    # Note: We use the same transforms as in training for consistency
    transform = Compose([
        CreateEdges(),
        FeaturizeNodes(),
        StandardizeTargets(dataset_name="qm9")
    ])
    # The QM9 dataset in PyG doesn't have official splits, so we create one.
    # This is a simplified approach for validation.
    full_dataset = get_dataset("qm9", root="data", transform=transform)
    # Use a fixed random seed for reproducibility of splits
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    
    if args.split == 'val':
        split_indices = shuffled_indices[train_size : train_size + val_size]
    elif args.split == 'test':
        split_indices = shuffled_indices[train_size + val_size:]
    else: # train
        split_indices = shuffled_indices[:train_size]

    subset = torch.utils.data.Subset(full_dataset, split_indices.tolist())
    dataset = SubsetWrapper(subset)
    
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Load Target Statistics for Un-scaling ---
    stats_path = "data/target_stats.yaml"
    with open(stats_path, 'r') as f:
        stats = yaml.safe_load(f)
    mean = torch.tensor(stats['qm9']['mean'], device=device)
    std = torch.tensor(stats['qm9']['std'], device=device)

    # --- Evaluation Loop ---
    all_preds_scaled = []
    all_targets_scaled = []
    all_preds_unscaled = []
    all_targets_unscaled = []

    print(f"Running evaluation on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch
            )
            preds_scaled = out['graph_pred']
            targets_scaled = batch.y

            # Store scaled values
            all_preds_scaled.append(preds_scaled)
            all_targets_scaled.append(targets_scaled)

            # Un-scale for MAE calculation
            preds_unscaled = preds_scaled * std + mean
            targets_unscaled = targets_scaled * std + mean
            all_preds_unscaled.append(preds_unscaled)
            all_targets_unscaled.append(targets_unscaled)

    # --- Calculate Metrics ---
    all_preds_scaled = torch.cat(all_preds_scaled, dim=0)
    all_targets_scaled = torch.cat(all_targets_scaled, dim=0)
    all_preds_unscaled = torch.cat(all_preds_unscaled, dim=0)
    all_targets_unscaled = torch.cat(all_targets_unscaled, dim=0)

    # Scaled Loss (MSE)
    scaled_loss = torch.nn.functional.mse_loss(all_preds_scaled, all_targets_scaled).item()

    # Graph MAE (un-scaled)
    graph_mae = torch.nn.functional.l1_loss(all_preds_unscaled, all_targets_unscaled).item()

    print("\n--- QM9 Validation Results ---")
    print(f"Graph MAE (un-scaled): {graph_mae:.6f}")
    print(f"Scaled Loss (MSE):     {scaled_loss:.6f}")
    print("----------------------------")

if __name__ == "__main__":
    main()