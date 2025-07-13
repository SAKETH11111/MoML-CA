import os
import sys
import argparse
import yaml
import glob
import torch
import glob
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned DJMGNN checkpoints on the PFAS dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on (val/test).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (cuda/cpu).')
    parser.add_argument('--config_path', type=str, default='config/training_config.template.yaml', help='Path to training config YAML file')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    mgnn_config = config.get('mgnn', {})
    
    # --- Load PFAS Validation/Test Dataset ---
    print(f"Loading PFAS {args.split} dataset...")
    transform = Compose([
        CreateEdges(),
        FeaturizeNodes(),
        StandardizeTargets(dataset_name="pfas"),
    ])
    dataset = get_dataset("pfas", root="data", split=args.split, transform=transform)
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Find all checkpoints ---
    checkpoints = sorted(glob.glob(os.path.join(args.ckpt_dir, 'finetuned_pfas_8k_step_*.pt')))
    if not checkpoints:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    results = []
    best_mae = float('inf')
    best_ckpt_path = None

    for ckpt_path in checkpoints:
        print(f"\n--- Evaluating checkpoint: {os.path.basename(ckpt_path)} ---")
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

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating on {args.split} split"):
                batch = batch.to(device)
                out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                preds = out['graph_pred']
                targets = batch.y.view(preds.shape)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        mae_scaled = torch.nn.functional.l1_loss(all_preds, all_targets).item()
        results.append({'ckpt': os.path.basename(ckpt_path), 'mae': mae_scaled})

        print(f"Standardized MAE: {mae_scaled:.6f}")

        if mae_scaled < best_mae:
            best_mae = mae_scaled
            best_ckpt_path = ckpt_path

    print("\n--- Overall Results ---")
    for res in results:
        print(f"Checkpoint: {res['ckpt']}, MAE: {res['mae']:.6f}")
    
    if best_ckpt_path:
        print(f"\nBest checkpoint: {os.path.basename(best_ckpt_path)} with MAE: {best_mae:.6f}")
        # Save the best model
        best_model_save_path = os.path.join(args.ckpt_dir, 'finetuned_pfas_best.pt')
        torch.save(torch.load(best_ckpt_path), best_model_save_path)
        print(f"Best model saved to {best_model_save_path}")

if __name__ == "__main__":
    main()