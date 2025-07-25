"""
scripts/find_best_checkpoint.py

This script automates the process of finding the best model checkpoint from a
training run. It iterates through all checkpoint files in a specified directory,
evaluates each one on a validation dataset (e.g., QM9 validation split), and
identifies the checkpoint that yields the lowest validation loss.

This is a crucial step for model selection, ensuring that the model used for
final evaluation or deployment is the one that generalized best to unseen data
during training.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN
from moml.utils.dataset_utils import SubsetWrapper

NODE_FEATURE_DIM = 29

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    ckpt_path: str, config: Dict[str, Any], device: torch.device, val_loader: GraphDataLoader
) -> float:
    """Evaluates a single checkpoint and returns its scaled validation loss."""
    mgnn_config = config.get("mgnn", {})
    model = DJMGNN(
        in_node_dim=NODE_FEATURE_DIM,
        in_edge_dim=mgnn_config.get("in_edge_dim", 0),
        node_output_dims=mgnn_config.get("node_output_dims", 3),
        graph_output_dims=mgnn_config.get("graph_output_dims", 19),
        energy_output_dims=mgnn_config.get("energy_output_dims", 1),
        hidden_dim=mgnn_config.get("hidden_channels", 160),
        n_blocks=mgnn_config.get("num_layers", 4),
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds_scaled, all_targets_scaled = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch: Batch = batch.to(device)  # type: ignore[attr-defined]
            out = model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch  # type: ignore[attr-defined]
            )
            all_preds_scaled.append(out["graph_pred"])
            all_targets_scaled.append(batch.y)  # type: ignore[attr-defined]

    preds_tensor = torch.cat(all_preds_scaled, dim=0)
    targets_tensor = torch.cat(all_targets_scaled, dim=0)
    
    return torch.nn.functional.mse_loss(preds_tensor, targets_tensor).item()


def main():
    """Main function to find the best checkpoint."""
    parser = argparse.ArgumentParser(
        description="Find the best DJMGNN checkpoint on the QM9 validation set."
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing model checkpoints.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda/cpu).")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config YAML file.")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config_path}...")
    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Dataset ---
    logger.info(f"Loading QM9 {args.split} split...")
    transform = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name="qm9")])
    full_dataset = get_dataset("qm9", root="data", transform=transform)
    
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    val_indices = shuffled_indices[train_size : train_size + val_size]
    
    val_subset = torch.utils.data.Subset(full_dataset, val_indices.tolist())
    val_dataset = SubsetWrapper(val_subset)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Find and Evaluate Checkpoints ---
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "checkpoint_step_*.pt")
    checkpoint_paths = glob.glob(checkpoint_pattern)
    
    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {args.checkpoint_dir} matching '{checkpoint_pattern}'")
        return

    # Sort checkpoints by step number to evaluate them in order.
    checkpoint_paths.sort(key=lambda p: int(Path(p).stem.split("_")[-1]))

    best_loss = float("inf")
    best_ckpt_path = None

    logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate.")
    for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        step = int(Path(ckpt_path).stem.split("_")[-1])
        try:
            loss = evaluate_checkpoint(ckpt_path, config, device, val_loader)
            logger.info(f"  - Step {step:6d}: Scaled Loss (MSE) = {loss:.6f}")
            if loss < best_loss:
                best_loss = loss
                best_ckpt_path = ckpt_path
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Could not evaluate checkpoint {ckpt_path}: {e}")
            continue

    if best_ckpt_path:
        logger.info("\n--- Best Checkpoint Found ---")
        logger.info(f"Path: {best_ckpt_path}")
        logger.info(f"Scaled Loss (MSE): {best_loss:.6f}")
        logger.info("-----------------------------")
    else:
        logger.warning("Could not determine the best checkpoint.")


if __name__ == "__main__":
    main()