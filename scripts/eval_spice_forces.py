import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes
from torchvision.transforms import Compose

def main():
    parser = argparse.ArgumentParser(description='Evaluate DJMGNN on SPICE forces.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
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
    print(f"Loading SPICE {args.split} split...")
    # For SPICE, we don't need to standardize the node-level force targets
    transform = Compose([CreateEdges(), FeaturizeNodes()])
    dataset = get_dataset("spice", root="MoML-CA/data", split=args.split, transform=transform)
    
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Evaluation Loop ---
    all_preds = []
    all_targets = []

    print(f"Running evaluation on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch
            )
            # We are evaluating the node-level predictions (forces)
            preds = out['node_pred']
            targets = batch.node_y

            all_preds.append(preds)
            all_targets.append(targets)

    # --- Calculate Metrics ---
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Forces RMSE
    # The conversion factor from Hartree/Bohr to kcal/mol/Angstrom is ~51.42
    # Assuming the model outputs in Hartree/Bohr, we convert to the target units.
    conversion_factor = 51.42 
    forces_rmse = torch.sqrt(torch.mean((all_preds - all_targets)**2)) * conversion_factor

    print("\n--- SPICE Forces Validation Results ---")
    print(f"Forces RMSE (kcal/mol/Å): {forces_rmse.item():.6f}")
    print("-------------------------------------")

if __name__ == "__main__":
    main()