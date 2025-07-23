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
import csv
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

# Rich imports for enhanced logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

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


class EnhancedTrainingLogger:
    """Enhanced training logger with rich progress bars and live metrics."""
    
    def __init__(self, max_steps: int, log_every: int = 10, log_dir: str = "."):
        self.console = Console()
        self.max_steps = max_steps
        self.log_every = log_every
        self.log_dir = Path(log_dir)
        
        # Setup CSV logging
        self._setup_csv_logging()
        
        # Metrics tracking
        self.metrics_history = {
            'node': deque(maxlen=100),
            'graph': deque(maxlen=100), 
            'energy': deque(maxlen=100),
            'total': deque(maxlen=100),
            'lr': deque(maxlen=100),
            'steps_per_sec': deque(maxlen=10)
        }
        
        # Phase tracking
        self.phase_counts = {'node': 0, 'graph': 0}
        self.current_phase = None
        self.phase_history = deque(maxlen=20)  # Track recent phases
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=4
        )
        
        # Create main progress task
        self.progress_task = self.progress.add_task(
            "[cyan]Training Progress", 
            total=max_steps
        )
        
        # Start progress display
        self.live = Live(
            self._create_layout(), 
            console=self.console, 
            refresh_per_second=2,
            vertical_overflow="visible"
        )
        
        self.start_time = time.time()
        self.last_log_time = time.time()
    
    def _setup_csv_logging(self):
        """Setup CSV logging for metrics tracking."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Main metrics log
        self.metrics_csv_path = self.log_dir / f"training_metrics_{timestamp}.csv"
        self.metrics_csv_headers = [
            'step', 'timestamp', 'phase', 'total_loss', 'node_loss', 'graph_loss', 
            'energy_loss', 'learning_rate', 'steps_per_sec', 'elapsed_time',
            'weight_node', 'weight_graph', 'weight_energy'
        ]
        
        # Phase summary log  
        self.phase_csv_path = self.log_dir / f"phase_summary_{timestamp}.csv"
        self.phase_csv_headers = [
            'step', 'timestamp', 'phase', 'phase_count_node', 'phase_count_graph',
            'avg_loss_last_10', 'best_loss_so_far', 'time_in_phase'
        ]
        
        # Initialize CSV files with headers
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics_csv_headers)
            
        with open(self.phase_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.phase_csv_headers)
            
        # Track best loss for CSV logging
        self.best_loss_so_far = float('inf')
        self.phase_start_time = time.time()
        self.last_phase = None
        
    def start(self):
        """Start the live display."""
        self.live.start()
        
    def stop(self):
        """Stop the live display."""
        self.live.stop()
        
    def _create_layout(self):
        """Create the main layout for the live display."""
        # Progress bar
        progress_panel = Panel(
            self.progress,
            title="[bold blue]🚀 DJMGNN Training Progress",
            border_style="blue"
        )
        
        # Metrics table
        metrics_table = self._create_metrics_table()
        metrics_panel = Panel(
            metrics_table,
            title="[bold green]📊 Live Metrics",
            border_style="green"
        )
        
        # Phase tracking
        phase_panel = self._create_phase_panel()
        
        return Columns([
            Panel(
                progress_panel,
                expand=True
            ),
            Panel(
                Columns([metrics_panel, phase_panel]),
                expand=True
            )
        ])
    
    def _create_metrics_table(self):
        """Create metrics table showing current and recent values."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Current", style="yellow", width=10)
        table.add_column("Avg (10)", style="green", width=10)
        table.add_column("Trend", style="blue", width=8)
        
        # Helper function to get trend indicator
        def get_trend(history):
            if len(history) < 5:
                return "📊"
            recent = list(history)[-5:]
            if len(recent) < 2:
                return "📊"
            trend = sum(recent[-2:]) / 2 - sum(recent[:2]) / 2
            if trend > 0.01:
                return "📈"
            elif trend < -0.01:
                return "📉"
            else:
                return "➡️"
        
        # Add metrics rows
        for name, history in self.metrics_history.items():
            if len(history) > 0:
                current = history[-1]
                avg = sum(list(history)[-10:]) / min(len(history), 10)
                trend = get_trend(history)
                
                if name == 'lr':
                    table.add_row(f"LR", f"{current:.2e}", f"{avg:.2e}", trend)
                elif name == 'steps_per_sec':
                    table.add_row(f"Speed", f"{current:.1f}/s", f"{avg:.1f}/s", trend)
                else:
                    table.add_row(f"{name.title()}", f"{current:.4f}", f"{avg:.4f}", trend)
        
        return table
    
    def _create_phase_panel(self):
        """Create phase tracking panel."""
        phase_text = Text()
        
        # Current phase
        if self.current_phase:
            phase_emoji = "🧬" if self.current_phase == "node" else "📈"
            phase_text.append(f"Current: {phase_emoji} {self.current_phase.upper()}\n", style="bold white")
        
        # Phase counts
        phase_text.append(f"NODE: {self.phase_counts['node']:,} steps\n", style="cyan")
        phase_text.append(f"GRAPH: {self.phase_counts['graph']:,} steps\n", style="green")
        
        # Recent phase history (last 20 steps)
        if self.phase_history:
            phase_text.append("\nRecent Pattern:\n", style="bold")
            pattern = ""
            for phase in list(self.phase_history)[-20:]:
                pattern += "N" if phase == "node" else "G"
            phase_text.append(pattern, style="dim")
        
        return Panel(phase_text, title="[bold yellow]⚡ Phase Status", border_style="yellow")
    
    def update(self, step: int, task_type: str, losses: Dict[str, float], lr: float):
        """Update the logger with new step information."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Update progress
        self.progress.update(self.progress_task, completed=step)
        
        # Track phase changes
        if self.last_phase != task_type:
            self.phase_start_time = current_time
            self.last_phase = task_type
        
        # Track current phase
        self.current_phase = task_type
        self.phase_counts[task_type] += 1
        self.phase_history.append(task_type)
        
        # Update metrics
        self.metrics_history['node'].append(losses['node_loss'])
        self.metrics_history['graph'].append(losses['graph_loss'])
        self.metrics_history['energy'].append(losses['energy_loss'])
        self.metrics_history['total'].append(losses['total_loss'])
        self.metrics_history['lr'].append(lr)
        
        # Calculate steps per second
        if hasattr(self, '_last_update_time'):
            time_diff = current_time - self._last_update_time
            if time_diff > 0:
                self.metrics_history['steps_per_sec'].append(1.0 / time_diff)
        self._last_update_time = current_time
        
        # Track best loss
        if losses['total_loss'] < self.best_loss_so_far:
            self.best_loss_so_far = losses['total_loss']
        
        # Write to CSV every step (for complete data)
        self._log_to_csv(step, task_type, losses, lr, elapsed_time, current_time)
        
        # Update live layout
        self.live.update(self._create_layout())
        
        # Log to file every log_every steps
        if step % self.log_every == 0:
            eta_seconds = (self.max_steps - step) / (step / elapsed_time) if step > 0 else 0
            
            # File logging (traditional format)
            progress_pct = (step / self.max_steps) * 100
            log_msg = (
                f"🎯 {progress_pct:5.1f}% ({step:,}/{self.max_steps:,}) | "
                f"📋 {task_type.upper():5s} | "
                f"📊 Loss: {losses['total_loss']:7.4f} | "
                f"🧬 Node: {losses['node_loss']:6.4f} | "
                f"📈 Graph: {losses['graph_loss']:6.4f} | "
                f"⚡ Energy: {losses['energy_loss']:6.4f} | "
                f"⚖️ W: {[f'{w:.2f}' for w in losses.get('weights', [])]} | "
                f"📚 LR: {lr:8.2e} | "
                f"🕐 ETA: {eta_seconds/3600:.1f}h"
            )
            logger.info(log_msg)
    
    def _log_to_csv(self, step: int, task_type: str, losses: Dict[str, float], lr: float, elapsed_time: float, current_time: float):
        """Log metrics to CSV files."""
        try:
            # Get current steps per second
            current_sps = self.metrics_history['steps_per_sec'][-1] if self.metrics_history['steps_per_sec'] else 0.0
            
            # Get weights (handle missing weights gracefully)
            weights = losses.get('weights', [1.0, 1.0, 1.0])
            weight_node = weights[0] if len(weights) > 0 else 1.0
            weight_graph = weights[1] if len(weights) > 1 else 1.0  
            weight_energy = weights[2] if len(weights) > 2 else 1.0
            
            # Main metrics CSV
            metrics_row = [
                step, current_time, task_type, losses['total_loss'], losses['node_loss'],
                losses['graph_loss'], losses['energy_loss'], lr, current_sps, elapsed_time,
                weight_node, weight_graph, weight_energy
            ]
            
            with open(self.metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metrics_row)
            
            # Phase summary CSV (log every 10 steps or phase change)
            if step % 10 == 0 or self.last_phase != task_type:
                avg_loss_last_10 = sum(list(self.metrics_history['total'])[-10:]) / min(len(self.metrics_history['total']), 10)
                time_in_current_phase = current_time - self.phase_start_time
                
                phase_row = [
                    step, current_time, task_type, self.phase_counts['node'],
                    self.phase_counts['graph'], avg_loss_last_10, self.best_loss_so_far,
                    time_in_current_phase
                ]
                
                with open(self.phase_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(phase_row)
                    
        except Exception as e:
            # Don't let CSV logging break training
            logger.warning(f"CSV logging failed: {e}")
    
    def log_completion(self, total_time: float, final_loss: float, checkpoint_dir: str):
        """Log training completion."""
        self.stop()
        
        # Create completion panel
        completion_text = Text()
        completion_text.append("🎉 TRAINING COMPLETED! 🎉\n\n", style="bold green")
        completion_text.append(f"⏱️  Total Time: {total_time/3600:.2f} hours\n", style="cyan")
        completion_text.append(f"📊 Final Loss: {final_loss:.4f}\n", style="yellow")
        completion_text.append(f"🧬 NODE Steps: {self.phase_counts['node']:,}\n", style="cyan")
        completion_text.append(f"📈 GRAPH Steps: {self.phase_counts['graph']:,}\n", style="green")
        completion_text.append(f"📁 Checkpoints: {checkpoint_dir}\n", style="blue")
        completion_text.append(f"📊 CSV Logs: {self.metrics_csv_path.name}\n", style="magenta")
        completion_text.append(f"📈 Phase Log: {self.phase_csv_path.name}\n", style="magenta")
        
        panel = Panel(
            completion_text,
            title="[bold green]✨ Training Summary",
            border_style="green",
            expand=False
        )
        
        self.console.print("\n")
        self.console.print(panel)
        self.console.print("🎯 Ready to validate accuracy target!", style="bold green")
        
        # Log final summary to CSV
        try:
            summary_csv_path = self.log_dir / f"training_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(summary_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['total_time_hours', total_time/3600])
                writer.writerow(['final_loss', final_loss])
                writer.writerow(['node_steps', self.phase_counts['node']])
                writer.writerow(['graph_steps', self.phase_counts['graph']])
                writer.writerow(['best_loss', self.best_loss_so_far])
                writer.writerow(['checkpoint_dir', checkpoint_dir])
                writer.writerow(['metrics_csv', str(self.metrics_csv_path)])
                writer.writerow(['phase_csv', str(self.phase_csv_path)])
        except Exception as e:
            logger.warning(f"Failed to write training summary CSV: {e}")
    
    def log_error(self, message: str):
        """Log an error message."""
        self.console.print(f"❌ ERROR: {message}", style="bold red")
        logger.error(message)


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
    
    # Check for finite losses
    if not torch.isfinite(losses_tensor).all():
        logger.warning(f"Non-finite losses detected: {losses_tensor}. Skipping step.")
        return {
            "total_loss": float('nan'),
            "node_loss": losses_dict["node_loss"].item(),
            "graph_loss": losses_dict["graph_loss"].item(),
            "energy_loss": losses_dict["energy_loss"].item(),
            "weights": loss_weighter.loss_weights.detach().cpu().tolist()
        }
    
    # Use proper GradNorm backward implementation
    loss_weighter.backward(losses_tensor, retain_graph=False)
    
    # Clip gradients and step optimizer
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": torch.sum(losses_tensor * loss_weighter.loss_weights).item(),
        "node_loss": losses_dict["node_loss"].item(),
        "graph_loss": losses_dict["graph_loss"].item(),
        "energy_loss": losses_dict["energy_loss"].item(),
        "weights": loss_weighter.loss_weights.detach().cpu().tolist()
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
    logger.info("🚀 STARTING OPTIMIZED ALTERNATING TRAINING FOR DJMGNN")
    logger.info("=" * 80)
    logger.info(f"📋 Training Configuration:")
    logger.info(f"   • Max Steps: {args.max_steps:,}")
    logger.info(f"   • Batch Sizes: Graph={args.batch_graph}, Node={args.batch_node}")
    logger.info(f"   • Learning Rate: {args.lr}")
    logger.info(f"   • Checkpoint Dir: {args.checkpoint_dir}")
    logger.info(f"   • Logging Every: {args.log_every} steps")
    logger.info(f"   • Saving Every: {args.save_every} steps")
    logger.info("=" * 80)
    
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
    
    # Get shared parameter for GradNorm (backbone parameter from last block)
    backbone_parameter = model.blocks[-1].transition_layers[-1].weight  # Last transition layer weight
    
    # Proper GradNorm loss balancer using lucidrains implementation
    loss_weighter = GradNormLossWeighter(
        num_losses=3,  # node_loss, graph_loss, energy_loss
        learning_rate=1e-4,
        restoring_force_alpha=0.5,  # o3-pro recommended alpha
        grad_norm_parameters=backbone_parameter
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

    # Training Loop
    logger.info("🎯 BEGINNING TRAINING LOOP")
    logger.info(f"   • QM9 Dataset: {len(ds_graph):,} samples")
    logger.info(f"   • SPICE Dataset: {len(ds_node):,} samples") 
    logger.info(f"   • Total Training Data: {len(ds_graph) + len(ds_node):,} samples")
    logger.info("=" * 80)
    
    # Initialize enhanced logger with checkpoint directory for CSV logs
    enhanced_logger = EnhancedTrainingLogger(
        max_steps=args.max_steps, 
        log_every=args.log_every,
        log_dir=args.checkpoint_dir
    )
    enhanced_logger.start()
    
    print(f"\n🚀 STARTING 40K TRAINING RUN - TARGET: 95% ACCURACY")
    print(f"📊 Datasets: QM9 ({len(ds_graph):,}) + SPICE ({len(ds_node):,}) = {len(ds_graph) + len(ds_node):,} total")
    print(f"⏱️  Expected Runtime: ~3-4 hours\n")
    print("✨ Enhanced logging with live progress bars enabled!")
    print(f"📊 CSV logs will be saved to: {enhanced_logger.metrics_csv_path}")
    print(f"📈 Phase logs will be saved to: {enhanced_logger.phase_csv_path}\n")
    
    start_time = time.time()
    best_loss = float('inf')
    
    try:
        for step in range(start_step, args.max_steps):
            try:
                batch, task_type = next(cycle_iter)
            except StopIteration:
                logger.error("Data iterator exhausted")
                break

            # Perform training step  
            losses = train_step(
                model, optimizer, loss_weighter, batch, device, task_type
            )
            
            # Update scheduler
            scheduler.step()
            
            # Update enhanced logger with current step information
            lr = scheduler.get_last_lr()[0]
            try:
                enhanced_logger.update(step, task_type, losses, lr)
            except Exception as e:
                logger.warning(f"Enhanced logger update failed: {e}")
                # Fallback to basic logging
                if step % args.log_every == 0:
                    progress_pct = (step / args.max_steps) * 100
                    logger.info(f"🎯 {progress_pct:5.1f}% ({step:,}/{args.max_steps:,}) | 📋 {task_type.upper()} | Loss: {losses['total_loss']:.4f}")

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

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        enhanced_logger.log_error("Training interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        enhanced_logger.log_error(f"Training failed: {e}")
    finally:
        # Ensure enhanced logger is properly stopped
        try:
            enhanced_logger.stop()
        except:
            pass

    total_time = time.time() - start_time
    
    # Use enhanced logger for completion
    try:
        enhanced_logger.log_completion(
            total_time=total_time, 
            final_loss=losses.get('total_loss', 0.0), 
            checkpoint_dir=args.checkpoint_dir
        )
    except Exception as e:
        logger.warning(f"Enhanced logger completion failed: {e}")
        # Fallback to traditional logging
        total_hours = total_time / 3600
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"   • Total Steps: {args.max_steps:,}")
        logger.info(f"   • Total Time: {total_hours:.2f} hours ({total_time:.0f}s)")
        logger.info(f"   • Final Loss: {losses.get('total_loss', 'N/A')}")
        logger.info(f"   • Checkpoints Saved: {args.checkpoint_dir}")
        logger.info("=" * 80)
        print(f"\n🎉 SUCCESS! Training completed in {total_hours:.1f} hours")
        print(f"📁 Checkpoints saved to: {args.checkpoint_dir}")
        print(f"🎯 Ready to validate 95% accuracy target!")


if __name__ == "__main__":
    main()