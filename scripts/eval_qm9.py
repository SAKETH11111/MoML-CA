"""
scripts/eval_qm9.py

This script evaluates a trained Dense Junction Molecular Graph Neural Network
(DJMGNN) on the QM9 dataset. It loads a model checkpoint, processes a specified
dataset split (train, validation, or test), and computes the Mean Absolute Error
(MAE) on the un-scaled graph properties.

This serves as a standard benchmark for model performance on the QM9 dataset,
which is a common baseline for graph-based molecular property prediction.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN
from moml.utils.dataset_utils import SubsetWrapper

# Add project root to Python path to allow for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Standard feature dimension for nodes after featurization.
NODE_FEATURE_DIM = 29

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, config: Dict[str, Any], device: torch.device) -> DJMGNN:
    """
    Load a DJMGNN model from a checkpoint file.

    Args:
        ckpt_path (str): Path to the model checkpoint (.pt file).
        config (Dict[str, Any]): The model configuration dictionary.
        device (torch.device): The device (CPU or CUDA) to load the model onto.

    Returns:
        DJMGNN: The loaded and initialized model, set to evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint file cannot be found.
        KeyError: If the checkpoint is missing the 'model_state_dict'.
        RuntimeError: If there's an error loading the model's state dictionary.
    """
    logger.info(f"Loading model from checkpoint: {ckpt_path}")
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
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {ckpt_path}")
        raise
    except KeyError:
        logger.error(f"Checkpoint is missing 'model_state_dict': {ckpt_path}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load model state from {ckpt_path}: {e}")
        raise


def prepare_data_loader(
    split: str, batch_size: int, root_dir: str
) -> GraphDataLoader:
    """
    Prepare the QM9 data loader for a specific dataset split.

    Args:
        split (str): The dataset split to load ('train', 'val', or 'test').
        batch_size (int): The number of graphs per batch.
        root_dir (str): The root directory where the dataset is stored.

    Returns:
        GraphDataLoader: The configured data loader for the specified split.
    """
    logger.info(f"Loading QM9 {split} split...")
    transform = Compose(
        [CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name="qm9")]
    )
    full_dataset = get_dataset("qm9", root=root_dir, transform=transform)

    # Use a fixed random seed for reproducible splits.
    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(len(full_dataset), generator=generator)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))

    if split == "val":
        split_indices = shuffled_indices[train_size : train_size + val_size]
    elif split == "test":
        split_indices = shuffled_indices[train_size + val_size :]
    else:  # 'train'
        split_indices = shuffled_indices[:train_size]

    dataset_subset = torch.utils.data.Subset(full_dataset, split_indices.tolist())
    # Wrap the subset for PyG DataLoader compatibility.
    wrapped_dataset = SubsetWrapper(dataset_subset)

    return GraphDataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False)


def get_target_statistics(stats_path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load target statistics (mean and std) for un-scaling predictions.

    Args:
        stats_path (str): Path to the YAML file with target statistics.
        device (torch.device): The device to move the tensors to.

    Returns:
        A tuple containing the mean and standard deviation tensors.

    Raises:
        FileNotFoundError: If the statistics file is not found.
        KeyError: If the 'qm9' key is missing from the file.
    """
    try:
        with open(stats_path, "r") as f:
            stats = yaml.safe_load(f)
        mean = torch.tensor(stats["qm9"]["mean"], device=device)
        std = torch.tensor(stats["qm9"]["std"], device=device)
        return mean, std
    except FileNotFoundError:
        logger.error(f"Target statistics file not found: {stats_path}")
        raise
    except KeyError:
        logger.error(f"Statistics file is missing 'qm9' key: {stats_path}")
        raise


def run_evaluation(
    model: DJMGNN, loader: GraphDataLoader, mean: torch.Tensor, std: torch.Tensor, device: torch.device
) -> tuple[float, float]:
    """
    Run the evaluation loop and compute metrics.

    Args:
        model (DJMGNN): The model to evaluate.
        loader (GraphDataLoader): The data loader for the evaluation set.
        mean (torch.Tensor): The mean of the target values for un-scaling.
        std (torch.Tensor): The standard deviation for un-scaling.
        device (torch.device): The device to run evaluation on.

    Returns:
        A tuple containing the un-scaled Mean Absolute Error (MAE) and the
        scaled Mean Squared Error (MSE).
    """
    all_preds_scaled, all_targets_scaled = [], []
    all_preds_unscaled, all_targets_unscaled = [], []

    logger.info(f"Running evaluation on {len(loader.dataset)} samples...")  # type: ignore
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            preds_scaled = out["graph_pred"]
            targets_scaled = batch.y

            all_preds_scaled.append(preds_scaled)
            all_targets_scaled.append(targets_scaled)

            # Un-scale for MAE calculation.
            preds_unscaled = preds_scaled * std + mean
            targets_unscaled = targets_scaled * std + mean
            all_preds_unscaled.append(preds_unscaled)
            all_targets_unscaled.append(targets_unscaled)

    # Concatenate all batch results.
    preds_scaled = torch.cat(all_preds_scaled, dim=0)
    targets_scaled = torch.cat(all_targets_scaled, dim=0)
    preds_unscaled = torch.cat(all_preds_unscaled, dim=0)
    targets_unscaled = torch.cat(all_targets_unscaled, dim=0)

    # Calculate metrics.
    scaled_mse = torch.nn.functional.mse_loss(preds_scaled, targets_scaled).item()
    unscaled_mae = torch.nn.functional.l1_loss(preds_unscaled, targets_unscaled).item()

    return unscaled_mae, scaled_mse


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DJMGNN model on the QM9 dataset."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/training_config.template.yaml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory for storing datasets.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="data/target_stats.yaml",
        help="Path to the target statistics file for un-scaling.",
    )
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model, data, and statistics.
        model = load_model(args.ckpt, config, device)
        loader = prepare_data_loader(args.split, args.batch_size, args.data_dir)
        mean, std = get_target_statistics(args.stats_path, device)

        # Run evaluation.
        mae, mse = run_evaluation(model, loader, mean, std, device)

        # --- Print Results ---
        logger.info("\n--- QM9 Validation Results ---")
        logger.info(f"Graph MAE (un-scaled): {mae:.6f}")
        logger.info(f"Scaled Loss (MSE):     {mse:.6f}")
        logger.info("----------------------------")

    except (FileNotFoundError, KeyError, RuntimeError) as e:
        logger.error(f"A critical error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()