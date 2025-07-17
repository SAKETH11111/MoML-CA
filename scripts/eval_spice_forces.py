"""
scripts/eval_spice_forces.py

This script evaluates a trained DJMGNN model on the task of predicting atomic
forces using the SPICE dataset. It loads a model checkpoint, processes a
specified dataset split (e.g., validation or test), and computes the Root Mean
Squared Error (RMSE) for the predicted forces.

The script assumes the model outputs forces in units of Hartree/Bohr and
converts them to kcal/mol/Angstrom for the final metric calculation, which is a
common unit for force evaluation in molecular modeling.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes
from moml.models.mgnn.djmgnn import DJMGNN

NODE_FEATURE_DIM = 29
HARTREE_BOHR_TO_KCAL_MOL_A = 51.422067

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, config: Dict[str, Any], device: torch.device) -> DJMGNN:
    """Loads a DJMGNN model from a checkpoint."""
    logger.info(f"Loading model from checkpoint: {ckpt_path}")
    mgnn_config = config.get("mgnn", {})
    model = DJMGNN(
        in_node_dim=NODE_FEATURE_DIM,
        in_edge_dim=mgnn_config.get("in_edge_dim", 0),
        node_output_dims=mgnn_config.get("node_output_dims", 3),
        graph_output_dims=mgnn_config.get("graph_output_dims", 19),
        energy_output_dims=mgnn_config.get("energy_output_dims", 1),
        hidden_dim=mgnn_config.get("hidden_channels", 128),
        n_blocks=mgnn_config.get("num_layers", 4),
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_evaluation(model: DJMGNN, loader: GraphDataLoader, device: torch.device) -> float:
    """Runs the evaluation loop and returns the forces RMSE."""
    all_preds, all_targets = [], []
    logger.info(f"Running evaluation on {len(loader.dataset)} samples...")  # type: ignore
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch: Batch = batch.to(device)  # type: ignore[attr-defined]
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)  # type: ignore[attr-defined]
            all_preds.append(out["node_pred"])
            all_targets.append(batch.node_y)  # type: ignore[attr-defined]

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    # Calculate RMSE and convert units.
    rmse = torch.sqrt(torch.mean((preds_tensor - targets_tensor) ** 2))
    return rmse.item() * HARTREE_BOHR_TO_KCAL_MOL_A


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate DJMGNN on SPICE forces.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda/cpu).")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config YAML file.")
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model
    model = load_model(args.ckpt, config, device)

    # Load Dataset
    logger.info(f"Loading SPICE {args.split} split...")
    transform = Compose([CreateEdges(), FeaturizeNodes()])
    dataset = get_dataset("spice", root="data", split=args.split, transform=transform)
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluation
    forces_rmse = run_evaluation(model, loader, device)

    logger.info("\n--- SPICE Forces Validation Results ---")
    logger.info(f"Forces RMSE (kcal/mol/Å): {forces_rmse:.6f}")
    logger.info("-------------------------------------")


if __name__ == "__main__":
    main()