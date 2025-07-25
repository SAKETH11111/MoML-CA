"""
scripts/eval_pfas.py

This script evaluates one or more fine-tuned DJMGNN models on the PFAS dataset.
It is designed to iterate through all matching checkpoint files in a directory,
evaluate each one on a specified dataset split (validation or test), and report
the Mean Absolute Error (MAE) for each.

The script identifies the best-performing checkpoint based on the lowest MAE and
saves a copy of it as `finetuned_pfas_best.pt` for easy access. This is useful
for automated model selection after a series of fine-tuning runs.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from moml.models.mgnn.djmgnn import DJMGNN

NODE_FEATURE_DIM = 29

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    ckpt_path: Path, config: Dict[str, Any], loader: GraphDataLoader, device: torch.device
) -> float:
    """Evaluates a single model checkpoint and returns the scaled MAE."""
    logger.info(f"--- Evaluating checkpoint: {ckpt_path.name} ---")
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
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {loader.dataset.split}"):  # type: ignore
            batch: Batch = batch.to(device)  # type: ignore[attr-defined]
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)  # type: ignore[attr-defined]
            preds = out["graph_pred"]
            targets = batch.y.view(preds.shape)  # type: ignore[attr-defined]
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    mae_scaled = torch.nn.functional.l1_loss(preds_tensor, targets_tensor).item()
    logger.info(f"Standardized MAE: {mae_scaled:.6f}")
    return mae_scaled


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DJMGNN checkpoints on the PFAS dataset."
    )
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing model checkpoints.")
    parser.add_argument("--ckpt_pattern", type=str, default="finetuned_pfas_*.pt", help="Pattern to match checkpoint files.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda/cpu).")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config.")
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Dataset
    logger.info(f"Loading PFAS {args.split} dataset...")
    transform = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name="pfas")])
    dataset = get_dataset("pfas", root="data", split=args.split, transform=transform)
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Find and Evaluate Checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob(args.ckpt_pattern))
    if not checkpoints:
        logger.error(f"No checkpoints found in {ckpt_dir} matching '{args.ckpt_pattern}'")
        return

    results: List[Tuple[Path, float]] = []
    for ckpt_path in checkpoints:
        mae = evaluate_checkpoint(ckpt_path, config, loader, device)
        results.append((ckpt_path, mae))

    # Report and Save Best
    if not results:
        logger.warning("No checkpoints were successfully evaluated.")
        return
        
    results.sort(key=lambda x: x[1])  # Sort by MAE
    best_ckpt_path, best_mae = results[0]

    logger.info("\n--- Overall Results ---")
    for ckpt_path, mae in results:
        logger.info(f"Checkpoint: {ckpt_path.name}, MAE: {mae:.6f}")

    logger.info(f"\nBest checkpoint: {best_ckpt_path.name} with MAE: {best_mae:.6f}")
    best_model_save_path = ckpt_dir / "finetuned_pfas_best.pt"
    
    try:
        torch.save(torch.load(str(best_ckpt_path)), str(best_model_save_path))
        logger.info(f"Best model saved to {best_model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save best model: {e}")


if __name__ == "__main__":
    main()