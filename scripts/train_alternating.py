"""
scripts/train_alternating.py

This script implements an alternating training scheme for the DJMGNN model.
It is designed to train on two different types of tasks in an alternating
fashion: a primary graph-level task (e.g., predicting molecular properties
on the QM9 dataset) and a secondary node-level task (e.g., predicting atomic
forces on the SPICE dataset).

This approach allows the model to learn from both macroscopic graph-level
properties and microscopic node-level details, potentially leading to more
robust and accurate representations. The script handles data loading, model
initialization, checkpointing, and the alternating training loop.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN

DEFAULT_NODE_FEATURE_DIM = 29
LOG_FILE_NAME = "alternating_training.log"

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging to output to both console and a file."""
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler
        try:
            log_file_path = Path(LOG_FILE_NAME).resolve()
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception:
            logger.error(f"Failed to initialize file logger.", exc_info=True)


def create_cycle_iterator(dataloader: Optional[GraphDataLoader]) -> Optional[Iterator[Batch]]:
    """Create an infinite iterator from a DataLoader."""
    if dataloader is None:
        return None
    while True:
        for batch in dataloader:
            yield batch


def compute_losses(
    model: DJMGNN, batch: Batch, device: torch.device, task_type: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute and return the losses for a given batch and task type."""
    batch = batch.to(device)  # type: ignore[attr-defined]
    if not hasattr(batch, "x") or batch.x is None:  # type: ignore[attr-defined]
        logger.warning("Batch is missing 'x' attribute. Skipping loss computation.")
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    out = model(
        x=batch.x,  # type: ignore[attr-defined]
        edge_index=batch.edge_index,  # type: ignore[attr-defined]
        edge_attr=getattr(batch, "edge_attr", None),
        batch=getattr(batch, "batch", None),
        dist=getattr(batch, "dist", None),
    )

    node_loss, graph_loss, energy_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # Node-level force prediction loss (e.g., for SPICE)
    if task_type == "node" and "node_pred" in out and hasattr(batch, "node_y"):
        if batch.node_y.numel() > 0 and out["node_pred"].shape == batch.node_y.shape:  # type: ignore[attr-defined]
            node_loss = nn.MSELoss()(out["node_pred"], batch.node_y)  # type: ignore[attr-defined]

    # Graph-level property prediction loss (e.g., for QM9)
    if task_type == "graph" and "graph_pred" in out and hasattr(batch, "y"):
        if batch.y.numel() > 0 and out["graph_pred"].shape == batch.y.shape:  # type: ignore[attr-defined]
            graph_loss = nn.MSELoss()(out["graph_pred"], batch.y)  # type: ignore[attr-defined]

    # Graph-level energy prediction loss (e.g., for SPICE)
    if task_type == "node" and "energy_pred" in out and hasattr(batch, "y_graph"):
        if batch.y_graph.numel() > 0:  # type: ignore[attr-defined]
            targets = batch.y_graph.view(-1, 1)  # type: ignore[attr-defined]
            if out["energy_pred"].shape == targets.shape:
                energy_loss = nn.MSELoss()(out["energy_pred"], targets)

    return node_loss, graph_loss, energy_loss


def train_step(
    model: DJMGNN,
    optimizer: optim.Optimizer,
    batch: Batch,
    device: torch.device,
    task_type: str,
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()
    optimizer.zero_grad()

    node_loss, graph_loss, energy_loss = compute_losses(model, batch, device, task_type)

    total_loss = (
        weights["loss_node"] * node_loss
        + weights["loss_graph"] * graph_loss
        + weights["loss_energy"] * energy_loss
    )

    if torch.isfinite(total_loss) and total_loss > 0:
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    elif not torch.isfinite(total_loss):
        logger.warning(f"Non-finite loss detected ({total_loss}). Skipping step.")

    return {
        "total_loss": total_loss.item(),
        "node_loss": node_loss.item(),
        "graph_loss": graph_loss.item(),
        "energy_loss": energy_loss.item(),
    }


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, step: int, loss: float, ckpt_dir: str
):
    """Save a training checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, ckpt_path: str
) -> int:
    """Load a training checkpoint."""
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0)


def main():
    """Main function to run the alternating training script."""
    parser = argparse.ArgumentParser(
        description="Alternating training for DJMGNN on graph and node-level tasks."
    )
    parser.add_argument("--max_steps", type=int, default=8000, help="Maximum training steps.")
    parser.add_argument("--batch_graph", type=int, default=8, help="Batch size for graph-level tasks.")
    parser.add_argument("--batch_node", type=int, default=4, help="Batch size for node-level tasks.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--loss_node_weight", type=float, default=1000.0, help="Weight for node-level losses (forces).")
    parser.add_argument("--loss_energy_weight", type=float, default=1.0, help="Weight for node-level energy loss.")
    parser.add_argument("--loss_graph_weight", type=float, default=1.0, help="Weight for graph-level loss.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--save_every", type=int, default=1000, help="Frequency of saving checkpoints.")
    parser.add_argument("--log_every", type=int, default=20, help="Frequency of logging training progress.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cpu', 'cuda').")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config.")
    parser.add_argument("--fresh_start", action="store_true", help="Start fresh, ignoring existing checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--graph_dataset", type=str, default="qm9", help="Dataset for graph-level tasks.")
    parser.add_argument("--node_dataset", type=str, default="spice", help="Dataset for node-level tasks ('none' to disable).")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting alternating training for DJMGNN.")
    logger.info(f"Arguments: {vars(args)}")

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config_path}", exc_info=True)
        sys.exit(1)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- DataLoaders ---
    transform_graph = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name=args.graph_dataset)])
    ds_graph = get_dataset(args.graph_dataset, root="data", transform=transform_graph)
    graph_loader = GraphDataLoader(ds_graph, batch_size=args.batch_graph, shuffle=True, num_workers=4)
    graph_cycle = create_cycle_iterator(graph_loader)
    logger.info(f"Loaded {args.graph_dataset.upper()} dataset with {len(ds_graph)} samples.")

    node_cycle = None
    if args.node_dataset.lower() != "none":
        try:
            transform_node = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name=args.node_dataset)])
            ds_node = get_dataset(args.node_dataset, root="data", split="train", transform=transform_node)
            node_loader = GraphDataLoader(ds_node, batch_size=args.batch_node, shuffle=True, num_workers=4)
            node_cycle = create_cycle_iterator(node_loader)
            logger.info(f"Loaded {args.node_dataset.upper()} dataset with {len(ds_node)} samples.")
        except Exception as e:
            logger.warning(f"Could not load {args.node_dataset.upper()} dataset: {e}. Node-level tasks will be skipped.")

    # Model and Optimizer
    mgnn_config = config.get("mgnn", {})
    model = DJMGNN(
        in_node_dim=DEFAULT_NODE_FEATURE_DIM,
        in_edge_dim=mgnn_config.get("in_edge_dim", 0),
        node_output_dims=mgnn_config.get("node_output_dims", 3),
        graph_output_dims=mgnn_config.get("graph_output_dims", 19),
        energy_output_dims=mgnn_config.get("energy_output_dims", 1),
        hidden_dim=mgnn_config.get("hidden_channels", 128),
        n_blocks=mgnn_config.get("num_layers", 4),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if not args.fresh_start:
        ckpt_path = args.resume_from_checkpoint
        if not ckpt_path:
            # Find the latest checkpoint if not specified.
            checkpoints = sorted(
                Path(args.checkpoint_dir).glob("checkpoint_step_*.pt"),
                key=os.path.getmtime,
                reverse=True,
            )
            if checkpoints:
                ckpt_path = checkpoints[0]
        if ckpt_path and os.path.exists(ckpt_path):
            start_step = load_checkpoint(model, optimizer, str(ckpt_path))

    # Training Loop
    loss_weights = {
        "loss_node": args.loss_node_weight,
        "loss_energy": args.loss_energy_weight,
        "loss_graph": args.loss_graph_weight,
    }

    start_time = time.time()
    for step in range(start_step, args.max_steps):
        # Alternate between graph and node-level tasks.
        if node_cycle and step % 2 == 0:
            task_type = "node"
            batch = next(node_cycle)
        elif graph_cycle:
            task_type = "graph"
            batch = next(graph_cycle)
        else:
            logger.error("No data loaders available. Exiting.")
            break

        losses = train_step(model, optimizer, batch, device, task_type, loss_weights)

        if step % args.log_every == 0:
            elapsed_time = time.time() - start_time
            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"Task: {task_type.capitalize()} | "
                f"Total Loss: {losses['total_loss']:.4f} | "
                f"Node: {losses['node_loss']:.4f} | "
                f"Graph: {losses['graph_loss']:.4f} | "
                f"Energy: {losses['energy_loss']:.4f} | "
                f"Time/Step: {elapsed_time/args.log_every:.2f}s"
            )
            start_time = time.time()

        if step % args.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, losses["total_loss"], args.checkpoint_dir)


if __name__ == "__main__":
    main()
