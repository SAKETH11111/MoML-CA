"""
scripts/train_alternating_optimized.py

Optimized alternating training for DJMGNN based on o3-pro analysis.
Implements fixes for scaler serialization, dynamic loss weighting, 
cycle-based scheduling, and deterministic training.
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

# Import proper GradNorm implementation
from gradnorm_pytorch import GradNormLossWeighter

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN

DEFAULT_NODE_FEATURE_DIM = 29
LOG_FILE_NAME = "alternating_training_optimized.log"

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


def set_deterministic_training(seed: int = 1337):
    """Set all seeds and enable deterministic algorithms for reproducibility."""
    # Set Python seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set CUDA deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
        logger.info(f"Enabled deterministic training with seed {seed}")
    except RuntimeError as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# Note: Using proper lucidrains GradNormLossWeighter instead of custom implementation


class CycleIterator:
    """Cycle-based task iterator that alternates between node and graph tasks."""
    
    def __init__(
        self, 
        node_loader: Optional[GraphDataLoader], 
        graph_loader: Optional[GraphDataLoader],
        node_steps: int = 1,
        graph_steps: int = 3
    ):
        self.node_loader = node_loader
        self.graph_loader = graph_loader
        self.node_steps = node_steps
        self.graph_steps = graph_steps
        
        # Create infinite iterators
        self.node_iter = self._create_infinite_iterator(node_loader) if node_loader else None
        self.graph_iter = self._create_infinite_iterator(graph_loader) if graph_loader else None
        
        # Track cycle state
        self.cycle_position = 0
        self.total_cycle_length = node_steps + graph_steps
        
    def _create_infinite_iterator(self, dataloader: GraphDataLoader) -> Iterator[Batch]:
        """Create an infinite iterator from a DataLoader."""
        while True:
            for batch in dataloader:
                yield batch
                
    def __next__(self) -> tuple[Batch, str]:
        """Get next batch and task type."""
        position_in_cycle = self.cycle_position % self.total_cycle_length
        
        if position_in_cycle < self.node_steps and self.node_iter:
            task_type = "node"
            batch = next(self.node_iter)
        elif self.graph_iter:
            task_type = "graph" 
            batch = next(self.graph_iter)
        else:
            raise StopIteration("No data loaders available")
            
        self.cycle_position += 1
        return batch, task_type


def compute_losses(
    model: DJMGNN, batch: Batch, device: torch.device, task_type: str
) -> Dict[str, torch.Tensor]:
    """Compute and return the losses for a given batch and task type."""
    batch = batch.to(device)
    if not hasattr(batch, "x") or batch.x is None:
        logger.warning("Batch is missing 'x' attribute. Skipping loss computation.")
        return {
            "node_loss": torch.tensor(0.0, device=device),
            "graph_loss": torch.tensor(0.0, device=device), 
            "energy_loss": torch.tensor(0.0, device=device)
        }

    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=getattr(batch, "batch", None),
        dist=getattr(batch, "dist", None),
    )

    losses = {
        "node_loss": torch.tensor(0.0, device=device),
        "graph_loss": torch.tensor(0.0, device=device),
        "energy_loss": torch.tensor(0.0, device=device)
    }

    # Node-level force prediction loss (e.g., for SPICE)
    if task_type == "node" and "node_pred" in out and hasattr(batch, "node_y"):
        if batch.node_y.numel() > 0 and out["node_pred"].shape == batch.node_y.shape:
            losses["node_loss"] = nn.MSELoss()(out["node_pred"], batch.node_y)

    # Graph-level property prediction loss (e.g., for QM9)  
    if task_type == "graph" and "graph_pred" in out and hasattr(batch, "y"):
        if batch.y.numel() > 0 and out["graph_pred"].shape == batch.y.shape:
            losses["graph_loss"] = nn.MSELoss()(out["graph_pred"], batch.y)

    # Graph-level energy prediction loss (e.g., for SPICE)
    if task_type == "node" and "energy_pred" in out and hasattr(batch, "y_graph"):
        if batch.y_graph.numel() > 0:
            targets = batch.y_graph.view(-1, 1)
            if out["energy_pred"].shape == targets.shape:
                losses["energy_loss"] = nn.MSELoss()(out["energy_pred"], targets)

    return losses


def train_step(
    model: DJMGNN,
    optimizer: optim.Optimizer,
    loss_weighter: GradNormLossWeighter,
    batch: Batch,
    device: torch.device,
    task_type: str,
    shared_parameters: nn.Parameter,
) -> Dict[str, float]:
    """Perform a single training step with proper GradNorm loss balancing."""
    model.train()
    optimizer.zero_grad()

    losses_dict = compute_losses(model, batch, device, task_type)
    
    # Convert to tensor format expected by GradNormLossWeighter
    losses_tensor = torch.stack([
        losses_dict["node_loss"],
        losses_dict["graph_loss"], 
        losses_dict["energy_loss"]
    ])
    
    # Use proper GradNorm implementation
    weighted_loss = loss_weighter(
        losses_tensor, 
        shared_parameters=shared_parameters
    )

    if torch.isfinite(weighted_loss) and weighted_loss > 0:
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    elif not torch.isfinite(weighted_loss):
        logger.warning(f"Non-finite loss detected ({weighted_loss}). Skipping step.")

    return {
        "total_loss": weighted_loss.item(),
        "node_loss": losses_dict["node_loss"].item(),
        "graph_loss": losses_dict["graph_loss"].item(),
        "energy_loss": losses_dict["energy_loss"].item(),
        "weights": loss_weighter.weights.detach().cpu().tolist()
    }


def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    scaler_graph: StandardizeTargets,
    scaler_node: Optional[StandardizeTargets],
    step: int, 
    loss: float, 
    seed: int,
    ckpt_dir: str
):
    """Save a training checkpoint with scalers and metadata."""
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
    
    # Prepare scaler state
    scaler_state = {
        "graph_scaler": {
            "mean": scaler_graph.mean,
            "std": scaler_graph.std,
            "dataset_name": scaler_graph.dataset_name
        }
    }
    
    if scaler_node:
        scaler_state["node_scaler"] = {
            "mean": scaler_node.mean,
            "std": scaler_node.std, 
            "dataset_name": scaler_node.dataset_name
        }
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state": scaler_state,
            "step": step,
            "loss": loss,
            "seed": seed,
            "timestamp": time.time()
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved to {checkpoint_path} with scalers")


def load_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    ckpt_path: str
) -> tuple[int, Dict]:
    """Load a training checkpoint."""
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    scaler_state = checkpoint.get("scaler_state", {})
    step = checkpoint.get("step", 0)
    seed = checkpoint.get("seed", None)
    
    if seed:
        logger.info(f"Restored training seed: {seed}")
        set_deterministic_training(seed)
    
    return step, scaler_state


def main():
    """Main function to run the optimized alternating training script."""
    parser = argparse.ArgumentParser(
        description="Optimized alternating training for DJMGNN with o3-pro fixes."
    )
    parser.add_argument("--max_steps", type=int, default=40000, help="Maximum training steps (increased for 95% target).")
    parser.add_argument("--batch_graph", type=int, default=32, help="Batch size for graph-level tasks.")
    parser.add_argument("--batch_node", type=int, default=16, help="Batch size for node-level tasks.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument("--node_cycle_steps", type=int, default=1, help="Node steps per cycle.")
    parser.add_argument("--graph_cycle_steps", type=int, default=3, help="Graph steps per cycle.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_optimized", help="Directory to save checkpoints.")
    parser.add_argument("--save_every", type=int, default=2000, help="Frequency of saving checkpoints.")
    parser.add_argument("--log_every", type=int, default=50, help="Frequency of logging training progress.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cpu', 'cuda').")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config.")
    parser.add_argument("--fresh_start", action="store_true", help="Start fresh, ignoring existing checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--graph_dataset", type=str, default="qm9", help="Dataset for graph-level tasks.")
    parser.add_argument("--node_dataset", type=str, default="spice", help="Dataset for node-level tasks.")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting optimized alternating training for DJMGNN.")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set deterministic training
    set_deterministic_training(args.seed)

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config_path}")
        sys.exit(1)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- DataLoaders with Scalers ---
    transform_graph = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name=args.graph_dataset)])
    ds_graph = get_dataset(args.graph_dataset, root="data", transform=transform_graph)
    graph_loader = GraphDataLoader(ds_graph, batch_size=args.batch_graph, shuffle=True, num_workers=2)
    
    # Store scaler for later serialization  
    graph_scaler = transform_graph.transforms[-1]  # StandardizeTargets is last
    logger.info(f"Loaded {args.graph_dataset.upper()} dataset with {len(ds_graph)} samples.")

    node_scaler = None
    node_loader = None
    if args.node_dataset.lower() != "none":
        try:
            transform_node = Compose([CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name=args.node_dataset)])
            ds_node = get_dataset(args.node_dataset, root="data", split="train", transform=transform_node)
            node_loader = GraphDataLoader(ds_node, batch_size=args.batch_node, shuffle=True, num_workers=2)
            node_scaler = transform_node.transforms[-1]
            logger.info(f"Loaded {args.node_dataset.upper()} dataset with {len(ds_node)} samples.")
        except Exception as e:
            logger.warning(f"Could not load {args.node_dataset.upper()} dataset: {e}")

    # Create cycle iterator
    cycle_iter = CycleIterator(
        node_loader, graph_loader,
        node_steps=args.node_cycle_steps,
        graph_steps=args.graph_cycle_steps
    )

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
    
    # AdamW optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=5e-6
    )
    
    # Proper GradNorm loss balancer using lucidrains implementation
    loss_weighter = GradNormLossWeighter(
        num_losses=3,  # node_loss, graph_loss, energy_loss
        learning_rate=1e-4,
        restoring_force_alpha=0.5  # o3-pro recommended alpha
    )

    # Resume from checkpoint if needed
    start_step = 0
    if not args.fresh_start:
        ckpt_path = args.resume_from_checkpoint
        if not ckpt_path:
            checkpoints = sorted(
                Path(args.checkpoint_dir).glob("checkpoint_step_*.pt"),
                key=os.path.getmtime,
                reverse=True,
            )
            if checkpoints:
                ckpt_path = checkpoints[0]
        if ckpt_path and os.path.exists(ckpt_path):
            start_step, _ = load_checkpoint(model, optimizer, str(ckpt_path))

    # Get shared parameter for GradNorm (use the first layer of the model)
    shared_parameter = next(model.parameters())
    
    # Training Loop
    start_time = time.time()
    best_loss = float('inf')
    
    for step in range(start_step, args.max_steps):
        try:
            batch, task_type = next(cycle_iter)
        except StopIteration:
            logger.error("Data iterator exhausted")
            break

        # Perform training step  
        losses = train_step(
            model, optimizer, loss_weighter, batch, device, task_type, shared_parameter
        )
        
        # Update scheduler
        scheduler.step()

        if step % args.log_every == 0:
            elapsed_time = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"Task: {task_type.capitalize()} | "
                f"Total Loss: {losses['total_loss']:.4f} | "
                f"Node: {losses['node_loss']:.4f} | "
                f"Graph: {losses['graph_loss']:.4f} | "
                f"Energy: {losses['energy_loss']:.4f} | "
                f"Weights: {[f'{w:.2f}' for w in losses['weights']]} | "
                f"LR: {lr:.2e} | "
                f"Time/Step: {elapsed_time/args.log_every:.2f}s"
            )
            start_time = time.time()

        # Save checkpoints
        if step % args.save_every == 0 and step > 0:
            save_checkpoint(
                model, optimizer, graph_scaler, node_scaler,
                step, losses["total_loss"], args.seed, args.checkpoint_dir
            )
            
            # Save best checkpoint
            if losses["total_loss"] < best_loss:
                best_loss = losses["total_loss"]
                best_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pt")
                save_checkpoint(
                    model, optimizer, graph_scaler, node_scaler,
                    step, losses["total_loss"], args.seed, args.checkpoint_dir.replace("checkpoint_step_", "best_checkpoint")
                )
                logger.info(f"New best checkpoint saved with loss {best_loss:.4f}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()