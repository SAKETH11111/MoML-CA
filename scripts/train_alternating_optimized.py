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
import torch.utils.data
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
                                          AddPositionalFeatures, StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN

DEFAULT_NODE_FEATURE_DIM = 33  # Updated: 29 (original) + 4 (positional features)
LOG_FILE_NAME = "alternating_training_optimized.log"

# 3-Phase Curriculum Constants
PHASE_1_END_STEP = 2000   # Train only PIMEH
PHASE_2_END_STEP = 6000   # Train base DJMGNN
# Phase 3 starts at PHASE_2_END_STEP and continues until max_steps (joint training)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting and save computational resources.
    
    Monitors validation loss and stops training when no improvement is observed
    for a specified number of consecutive validation checks (patience).
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        """
        Initialize EarlyStopping.
        
        Args:
            patience (int): Number of validation checks to wait after last improvement
            min_delta (float): Minimum change in validation loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_step = 0
        
    def __call__(self, val_loss: float, step: int) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
            step (int): Current training step
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check if this is an improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            logger.info(f"🎯 New best validation loss: {val_loss:.6f} at step {step}")
            return False
        else:
            self.counter += 1
            logger.info(f"⏳ No improvement for {self.counter}/{self.patience} checks (best: {self.best_loss:.6f} at step {self.best_step})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"🛑 Early stopping triggered! No improvement for {self.patience} consecutive validation checks")
                logger.info(f"📊 Best validation loss: {self.best_loss:.6f} achieved at step {self.best_step}")
                return True
                
        return False


class CurriculumManager:
    """
    Manages 3-phase curriculum training for DJMGNN with PIMEH integration.
    
    Phase 1 (0-2000 steps): Train only PIMEH, freeze base DJMGNN
    Phase 2 (2000-6000 steps): Train base DJMGNN, freeze PIMEH  
    Phase 3 (6000+ steps): Joint training of everything
    """
    
    def __init__(self, model: DJMGNN, enhanced_logger: 'EnhancedTrainingLogger'):
        self.model = model
        self.logger = enhanced_logger
        self.current_phase = None  # Initialize to None so first update always triggers change
        self.phase_weights = {
            1: {'physics': 2.0, 'others': 0.1},    # Phase 1: Focus on physics loss
            2: {'physics': 0.1, 'others': 1.0},    # Phase 2: Focus on other losses
            3: {'physics': 1.0, 'others': 1.0}     # Phase 3: Balanced via GradNorm
        }
        
        # Track parameter counts for logging
        self._count_parameters()
        
    def _count_parameters(self):
        """Count parameters in different model components."""
        self.param_counts = {
            'pimeh': sum(p.numel() for p in self.model.pimeh_head.parameters()),
            'base': sum(p.numel() for p in self.model.parameters()) - sum(p.numel() for p in self.model.pimeh_head.parameters()),
            'total': sum(p.numel() for p in self.model.parameters())
        }
        
    def get_current_phase(self, step: int) -> int:
        """Determine current curriculum phase based on step."""
        if step < PHASE_1_END_STEP:
            return 1
        elif step < PHASE_2_END_STEP:
            return 2
        else:
            return 3
    
    def freeze_base_model(self):
        """Freeze all DJMGNN parameters except PIMEH (Phase 1)."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if not name.startswith('pimeh_head'):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
        
        logger.info(f"Phase 1: Frozen base model parameters ({frozen_count:,}), active PIMEH ({self.param_counts['pimeh']:,})")
        return frozen_count, self.param_counts['pimeh']
    
    def freeze_pimeh(self):
        """Freeze only PIMEH parameters (Phase 2)."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if name.startswith('pimeh_head'):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
        
        logger.info(f"Phase 2: Frozen PIMEH parameters ({frozen_count:,}), active base model ({self.param_counts['base']:,})")
        return frozen_count, self.param_counts['base']
    
    def unfreeze_all(self):
        """Unfreeze all parameters (Phase 3)."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info(f"Phase 3: All parameters unfrozen ({self.param_counts['total']:,})")
        return 0, self.param_counts['total']
    
    def update_phase(self, step: int, optimizer: optim.Optimizer) -> bool:
        """
        Update training phase based on current step.
        
        Returns:
            bool: True if phase changed, False otherwise
        """
        new_phase = self.get_current_phase(step)
        
        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase
            
            # Apply parameter freezing based on new phase
            if new_phase == 1:
                frozen, active = self.freeze_base_model()
                phase_desc = "PIMEH Only Training"
                reason = f"Step {step} < {PHASE_1_END_STEP} (Phase 1 threshold)"
            elif new_phase == 2:
                frozen, active = self.freeze_pimeh()
                phase_desc = "Base DJMGNN Training"
                reason = f"Step {step} >= {PHASE_1_END_STEP} and < {PHASE_2_END_STEP} (Phase 2 threshold)"
            else:  # Phase 3
                frozen, active = self.unfreeze_all()
                phase_desc = "Joint Training"
                reason = f"Step {step} >= {PHASE_2_END_STEP} (Phase 3 threshold)"
            
            # Update optimizer parameter groups (important for momentum/adam states)
            self._update_optimizer_groups(optimizer)
            
            # Log phase transition
            physics_weight = self.phase_weights[new_phase]['physics']
            self.logger.log_curriculum_transition(
                step=step,
                phase=new_phase,
                phase_description=phase_desc,
                frozen_params=frozen,
                active_params=active,
                physics_weight=physics_weight,
                reason=reason
            )
            
            # Console notification
            logger.info("=" * 80)
            logger.info(f"CURRICULUM PHASE TRANSITION: {old_phase} -> {new_phase}")
            logger.info(f"   Description: {phase_desc}")
            logger.info(f"   Frozen Parameters: {frozen:,}")
            logger.info(f"   Active Parameters: {active:,}")
            logger.info(f"   Physics Loss Weight: {physics_weight:.1f}")
            logger.info(f"   Reason: {reason}")
            logger.info("=" * 80)
            
            return True
        
        return False
    
    def _update_optimizer_groups(self, optimizer: optim.Optimizer):
        """Update optimizer parameter groups after freezing/unfreezing."""
        # Clear momentum/adam states for frozen parameters
        # This prevents stale gradients from affecting training
        active_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                active_params.append(param)
        
        # Update optimizer's param_groups
        optimizer.param_groups[0]['params'] = active_params
        
        # Clear state for parameters that are no longer active
        # In PyTorch optimizers, state keys are the actual parameter tensors
        params_to_remove = []
        for param_tensor in optimizer.state.keys():
            if not param_tensor.requires_grad:
                params_to_remove.append(param_tensor)
        
        for param_tensor in params_to_remove:
            del optimizer.state[param_tensor]
    
    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Get phase-specific loss weights."""
        phase = self.get_current_phase(step)
        weights = self.phase_weights[phase]
        
        return {
            'physics_loss': weights['physics'],
            'node_loss': weights['others'],
            'graph_loss': weights['others'],
            'energy_loss': weights['others']
        }
    
    def should_skip_gradnorm(self, step: int) -> bool:
        """Determine if GradNorm should be skipped for this phase."""
        # Skip GradNorm in phases 1 and 2, only use in phase 3
        return self.get_current_phase(step) < 3


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
            'physics': deque(maxlen=100),  # New physics loss tracking
            'total': deque(maxlen=100),
            'lr': deque(maxlen=100),
            'steps_per_sec': deque(maxlen=10)
        }
        
        # Phase tracking
        self.phase_counts = {'node': 0, 'graph': 0}
        self.current_phase = None
        self.phase_history = deque(maxlen=20)  # Track recent phases
        
        # Curriculum phase tracking
        self.curriculum_phase = None  # Will be set on first update
        self.phase_transitions = []  # Track phase transitions with timestamps
        
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
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Main metrics log
        self.metrics_csv_path = self.log_dir / f"training_metrics_{timestamp}.csv"
        self.metrics_csv_headers = [
            'step', 'timestamp', 'phase', 'total_loss', 'node_loss', 'graph_loss', 
            'energy_loss', 'physics_loss', 'learning_rate', 'steps_per_sec', 'elapsed_time',
            'weight_node', 'weight_graph', 'weight_energy', 'weight_physics'
        ]
        
        # Phase summary log  
        self.phase_csv_path = self.log_dir / f"phase_summary_{timestamp}.csv"
        self.phase_csv_headers = [
            'step', 'timestamp', 'phase', 'phase_count_node', 'phase_count_graph',
            'avg_loss_last_10', 'best_loss_so_far', 'time_in_phase', 'curriculum_phase'
        ]
        
        # Curriculum phase log
        self.curriculum_csv_path = self.log_dir / f"curriculum_phases_{timestamp}.csv"
        self.curriculum_csv_headers = [
            'step', 'timestamp', 'curriculum_phase', 'phase_description', 'frozen_parameters', 
            'active_parameters', 'physics_loss_weight', 'transition_reason'
        ]
        
        # Initialize CSV files with headers
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics_csv_headers)
            
        with open(self.phase_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.phase_csv_headers)
            
        with open(self.curriculum_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.curriculum_csv_headers)
            
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
            phase_text.append(f"Current: {self.current_phase.upper()}\n", style="bold white")
        
        # Curriculum phase
        if self.curriculum_phase is not None:
            curriculum_info = {
                1: "PHASE 1: PIMEH Only",
                2: "PHASE 2: Base DJMGNN", 
                3: "PHASE 3: Joint Training"
            }
            curriculum_desc = curriculum_info.get(self.curriculum_phase, f"Phase {self.curriculum_phase}")
            phase_text.append(f"\n{curriculum_desc}\n", style="bold magenta")
        else:
            phase_text.append(f"\nCurriculum: Initializing\n", style="bold magenta")
        
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
    
    def log_curriculum_transition(self, step: int, phase: int, phase_description: str, 
                                 frozen_params: int, active_params: int, physics_weight: float, reason: str):
        """Log curriculum phase transition."""
        try:
            current_time = time.time()
            transition_row = [
                step, current_time, phase, phase_description, frozen_params,
                active_params, physics_weight, reason
            ]
            
            with open(self.curriculum_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(transition_row)
                
        except Exception as e:
            logger.warning(f"Curriculum transition logging failed: {e}")
    
    def update(self, step: int, task_type: str, losses: Dict[str, float], lr: float, curriculum_phase: int = None):
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
        
        # Update curriculum phase if provided
        if curriculum_phase is not None and curriculum_phase != self.curriculum_phase:
            self.phase_transitions.append({
                'step': step,
                'timestamp': current_time,
                'old_phase': self.curriculum_phase,
                'new_phase': curriculum_phase
            })
            self.curriculum_phase = curriculum_phase
        
        # Update metrics
        self.metrics_history['node'].append(losses['node_loss'])
        self.metrics_history['graph'].append(losses['graph_loss'])
        self.metrics_history['energy'].append(losses['energy_loss'])
        self.metrics_history['physics'].append(losses['physics_loss'])  # New physics loss
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
                f"🌀 Physics: {losses['physics_loss']:6.4f} | "
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
            
            # Get weights (handle missing weights gracefully) - now 4 weights
            weights = losses.get('weights', [1.0, 1.0, 1.0, 1.0])
            weight_node = weights[0] if len(weights) > 0 else 1.0
            weight_graph = weights[1] if len(weights) > 1 else 1.0  
            weight_energy = weights[2] if len(weights) > 2 else 1.0
            weight_physics = weights[3] if len(weights) > 3 else 1.0
            
            # Main metrics CSV
            metrics_row = [
                step, current_time, task_type, losses['total_loss'], losses['node_loss'],
                losses['graph_loss'], losses['energy_loss'], losses['physics_loss'], 
                lr, current_sps, elapsed_time, weight_node, weight_graph, weight_energy, weight_physics
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
                    time_in_current_phase, self.curriculum_phase if self.curriculum_phase is not None else 0
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
        completion_text.append(f"CSV Logs: {self.metrics_csv_path.name}\n", style="magenta")
        completion_text.append(f"Phase Log: {self.phase_csv_path.name}\n", style="magenta")
        completion_text.append(f"Curriculum Log: {self.curriculum_csv_path.name}\n", style="magenta")
        completion_text.append(f"Phase Transitions: {len(self.phase_transitions)}\n", style="cyan")
        
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


def validate_model(
    model: DJMGNN,
    graph_loader: GraphDataLoader,
    device: torch.device,
    num_batches: int = 50
) -> float:
    """
    Validate the model on a subset of the validation data.
    
    Args:
        model: The DJMGNN model to validate
        graph_loader: DataLoader for graph-level validation data
        device: Device to run validation on
        num_batches: Number of batches to use for validation
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        batch_count = 0
        for batch in graph_loader:
            if batch_count >= num_batches:
                break
                
            batch = batch.to(device)
            if not hasattr(batch, "x") or batch.x is None:
                continue
                
            # Forward pass
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=getattr(batch, "batch", None),
                dist=getattr(batch, "dist", None),
                pos=getattr(batch, "pos", None),
            )
            
            # Compute validation loss (focusing on graph predictions)
            if "graph_pred" in out and hasattr(batch, "y"):
                if batch.y.numel() > 0 and out["graph_pred"].shape == batch.y.shape:
                    # Compute total loss for all properties
                    batch_loss = nn.MSELoss()(out["graph_pred"], batch.y)
                    total_loss += batch_loss.item() * batch.y.size(0)
                    total_samples += batch.y.size(0)
            
            batch_count += 1
    
    model.train()
    return total_loss / total_samples if total_samples > 0 else float('inf')


def compute_losses(
    model: DJMGNN, batch: Batch, device: torch.device, task_type: str
) -> Dict[str, torch.Tensor]:
    """Compute and return the losses for a given batch and task type.
    
    Now includes separate loss computation for rotational constants:
    - Regular graph properties (indices 0-15): graph_loss
    - Rotational constants (indices 16-18): physics_loss (rotational_loss)
    """
    batch = batch.to(device)
    if not hasattr(batch, "x") or batch.x is None:
        logger.warning("Batch is missing 'x' attribute. Skipping loss computation.")
        return {
            "node_loss": torch.tensor(0.0, device=device),
            "graph_loss": torch.tensor(0.0, device=device), 
            "energy_loss": torch.tensor(0.0, device=device),
            "physics_loss": torch.tensor(0.0, device=device)  # New rotational constants loss
        }

    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch=getattr(batch, "batch", None),
        dist=getattr(batch, "dist", None),
        pos=getattr(batch, "pos", None),  # Add positions for PIMEH rotational constants
    )

    losses = {
        "node_loss": torch.tensor(0.0, device=device),
        "graph_loss": torch.tensor(0.0, device=device),
        "energy_loss": torch.tensor(0.0, device=device),
        "physics_loss": torch.tensor(0.0, device=device)  # New rotational constants loss
    }

    # Node-level force prediction loss (e.g., for SPICE)
    if task_type == "node" and "node_pred" in out and hasattr(batch, "node_y"):
        if batch.node_y.numel() > 0 and out["node_pred"].shape == batch.node_y.shape:
            losses["node_loss"] = nn.MSELoss()(out["node_pred"], batch.node_y)

    # Graph-level property prediction loss - SPLIT INTO TWO PARTS
    if task_type == "graph" and "graph_pred" in out and hasattr(batch, "y"):
        if batch.y.numel() > 0 and out["graph_pred"].shape == batch.y.shape:
            # Split predictions and targets
            pred_regular = out["graph_pred"][:, 0:16]  # Regular properties (indices 0-15)
            pred_rotational = out["graph_pred"][:, 16:19]  # Rotational constants (indices 16-18)
            
            target_regular = batch.y[:, 0:16]  # Regular property targets
            target_rotational = batch.y[:, 16:19]  # Rotational constant targets
            
            # Compute separate MSE losses
            losses["graph_loss"] = nn.MSELoss()(pred_regular, target_regular)
            losses["physics_loss"] = nn.MSELoss()(pred_rotational, target_rotational)
            
            # Optional: Apply scaling factor for rotational constants
            # Rotational constants are in GHz and may need different weighting
            # This is handled by GradNorm, but we can apply initial scaling if needed
            # losses["physics_loss"] = losses["physics_loss"] * 0.1  # Initial lower weight

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
    curriculum_manager: Optional[CurriculumManager] = None,
    step: int = 0,
) -> Dict[str, float]:
    """Perform a single training step with curriculum-aware loss balancing.
    
    Now handles 4 losses with 3-phase curriculum:
    - Phase 1: Focus on physics_loss (PIMEH training)
    - Phase 2: Focus on other losses (base DJMGNN training)  
    - Phase 3: Balanced GradNorm training
    """
    model.train()
    optimizer.zero_grad()

    losses_dict = compute_losses(model, batch, device, task_type)
    
    # Convert to tensor format expected by GradNormLossWeighter (now 4 losses)
    losses_tensor = torch.stack([
        losses_dict["node_loss"],
        losses_dict["graph_loss"], 
        losses_dict["energy_loss"],
        losses_dict["physics_loss"]  # New rotational constants loss
    ])
    
    # Apply phase-specific loss weighting if curriculum manager provided
    if curriculum_manager is not None:
        phase_weights = curriculum_manager.get_loss_weights(step)
        
        # Apply weights to losses
        weighted_losses = torch.stack([
            losses_dict["node_loss"] * phase_weights['node_loss'],
            losses_dict["graph_loss"] * phase_weights['graph_loss'],
            losses_dict["energy_loss"] * phase_weights['energy_loss'],
            losses_dict["physics_loss"] * phase_weights['physics_loss']
        ])
        
        # Use weighted losses for backward pass calculation
        final_losses = weighted_losses
    else:
        final_losses = losses_tensor
    
    # Check for finite losses
    if not torch.isfinite(final_losses).all():
        logger.warning(f"Non-finite losses detected: {final_losses}. Skipping step.")
        return {
            "total_loss": float('nan'),
            "node_loss": losses_dict["node_loss"].item(),
            "graph_loss": losses_dict["graph_loss"].item(),
            "energy_loss": losses_dict["energy_loss"].item(),
            "physics_loss": losses_dict["physics_loss"].item(),  # Include physics loss
            "weights": loss_weighter.loss_weights.detach().cpu().tolist() if curriculum_manager is None or not curriculum_manager.should_skip_gradnorm(step) else [1.0, 1.0, 1.0, 1.0]
        }
    
    # Use GradNorm or manual weighting based on curriculum phase
    if curriculum_manager is not None and curriculum_manager.should_skip_gradnorm(step):
        # Phases 1 & 2: Use manual weighting, skip GradNorm
        total_loss = final_losses.sum()
        
        # Check if total_loss requires gradients before calling backward()
        if not total_loss.requires_grad:
            logger.warning(f"Step {step}: Loss tensor does not require gradients (likely due to frozen parameters in curriculum phase {curriculum_manager.current_phase}). Skipping backward pass for task '{task_type}'.")
            return {
                "total_loss": total_loss.item(),
                "node_loss": losses_dict["node_loss"].item(),
                "graph_loss": losses_dict["graph_loss"].item(),
                "energy_loss": losses_dict["energy_loss"].item(),
                "physics_loss": losses_dict["physics_loss"].item(),
                "weights": [1.0, 1.0, 1.0, 1.0]  # Default weights when gradient computation is skipped
            }
        
        total_loss.backward()
        current_weights = [
            curriculum_manager.get_loss_weights(step)['node_loss'],
            curriculum_manager.get_loss_weights(step)['graph_loss'],
            curriculum_manager.get_loss_weights(step)['energy_loss'],
            curriculum_manager.get_loss_weights(step)['physics_loss']
        ]
    else:
        # Phase 3: Use GradNorm for automatic balancing
        # Check if any of the losses require gradients before calling GradNorm backward
        if not any(loss.requires_grad for loss in losses_tensor):
            logger.warning(f"Step {step}: No loss tensors require gradients. Skipping GradNorm backward pass for task '{task_type}'.")
            total_loss = torch.sum(losses_tensor)
            return {
                "total_loss": total_loss.item(),
                "node_loss": losses_dict["node_loss"].item(),
                "graph_loss": losses_dict["graph_loss"].item(),
                "energy_loss": losses_dict["energy_loss"].item(),
                "physics_loss": losses_dict["physics_loss"].item(),
                "weights": [1.0, 1.0, 1.0, 1.0]  # Default weights when gradient computation is skipped
            }
        
        loss_weighter.backward(losses_tensor, retain_graph=False)
        total_loss = torch.sum(losses_tensor * loss_weighter.loss_weights)
        current_weights = loss_weighter.loss_weights.detach().cpu().tolist()
    
    # Clip gradients and step optimizer
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "node_loss": losses_dict["node_loss"].item(),
        "graph_loss": losses_dict["graph_loss"].item(),
        "energy_loss": losses_dict["energy_loss"].item(),
        "physics_loss": losses_dict["physics_loss"].item(),
        "weights": current_weights
    }


def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    scaler_graph: StandardizeTargets,
    scaler_node: Optional[StandardizeTargets],
    step: int, 
    loss: float, 
    seed: int,
    ckpt_dir: str,
    is_best: bool = False,
    val_loss: Optional[float] = None
):
    """Save a training checkpoint with scalers and metadata."""
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if is_best:
        checkpoint_path = os.path.join(ckpt_dir, "best_checkpoint.pt")
    else:
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
    
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state": scaler_state,
        "step": step,
        "loss": loss,
        "seed": seed,
        "timestamp": time.time()
    }
    
    # Add validation loss if provided
    if val_loss is not None:
        checkpoint_data["val_loss"] = val_loss
    
    torch.save(checkpoint_data, checkpoint_path)
    
    if is_best:
        logger.info(f"🏆 Best checkpoint saved to {checkpoint_path} (val_loss: {val_loss:.6f})")
    else:
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
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (validation checks).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Minimum delta for early stopping.")
    parser.add_argument("--validate_every", type=int, default=500, help="Frequency of validation checks.")
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
        config_path = 'config/joint_training.yaml'
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- DataLoaders with Scalers ---
    transform_graph = Compose([CreateEdges(), FeaturizeNodes(), AddPositionalFeatures(), StandardizeTargets(dataset_name=args.graph_dataset)])
    ds_graph = get_dataset(args.graph_dataset, root="data", transform=transform_graph)
    
    # Split dataset for training and validation (80/20 split)
    train_size = int(0.8 * len(ds_graph))
    val_size = len(ds_graph) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        ds_graph, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    graph_loader = GraphDataLoader(train_dataset, batch_size=args.batch_graph, shuffle=True, num_workers=0)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_graph, shuffle=False, num_workers=0)
    
    # Store scaler for later serialization      
    graph_scaler = transform_graph.transforms[-1]  # StandardizeTargets is last
    logger.info(f"Loaded {args.graph_dataset.upper()} dataset: {train_size} train, {val_size} validation samples.")

    node_scaler = None
    node_loader = None
    if args.node_dataset.lower() != "none":
        try:
            logger.info(f"🔧 Creating transform pipeline for {args.node_dataset.upper()}...")
            transform_node = Compose([CreateEdges(), FeaturizeNodes(), AddPositionalFeatures(), StandardizeTargets(dataset_name=args.node_dataset)])
            logger.info("✅ Transform pipeline created")
            
            logger.info(f"📦 Loading {args.node_dataset.upper()} dataset...")
            logger.info("   This may take a few minutes for SPICE preprocessing...")
            
            ds_node = get_dataset(args.node_dataset, root="data", split="train", transform=transform_node)
            logger.info(f"✅ Dataset loaded: {len(ds_node)} samples")
            
            logger.info("🚛 Creating GraphDataLoader for node dataset...")
            logger.info(f"   • Dataset size: {len(ds_node)}")
            logger.info(f"   • Batch size: {args.batch_node}")
            logger.info(f"   • Shuffle: True")
            logger.info(f"   • Num workers: 0")
            
            node_loader = GraphDataLoader(ds_node, batch_size=args.batch_node, shuffle=True, num_workers=0)
            logger.info("✅ GraphDataLoader created successfully!")
            
            node_scaler = transform_node.transforms[-1]
            logger.info(f"Loaded {args.node_dataset.upper()} dataset with {len(ds_node)} samples.")
        except Exception as e:
            logger.warning(f"Could not load {args.node_dataset.upper()} dataset: {e}")
            logger.error(f"Full error: {str(e)}", exc_info=True)

    # Create cycle iterator
    logger.info("🔄 Creating cycle iterator...")
    cycle_iter = CycleIterator(
        node_loader, graph_loader,
        node_steps=args.node_cycle_steps,
        graph_steps=args.graph_cycle_steps
    )
    logger.info("✅ Cycle iterator created successfully")

    # Model and Optimizer
    logger.info("🏗️ Loading model configuration...")
    mgnn_config = config.get("mgnn", {})
    logger.info(f"📋 Model config loaded: {mgnn_config}")
    
    logger.info("🧠 Creating DJMGNN model...")
    logger.info(f"   • in_node_dim: {DEFAULT_NODE_FEATURE_DIM}")
    logger.info(f"   • in_edge_dim: {mgnn_config.get('in_edge_dim', 0)}")
    logger.info(f"   • node_output_dims: {mgnn_config.get('node_output_dims', 3)}")
    logger.info(f"   • graph_output_dims: {mgnn_config.get('graph_output_dims', 19)}")
    logger.info(f"   • energy_output_dims: {mgnn_config.get('energy_output_dims', 1)}")
    logger.info(f"   • hidden_dim: {mgnn_config.get('hidden_channels', 160)}")
    logger.info(f"   • n_blocks: {mgnn_config.get('num_layers', 4)}")
    
    model = DJMGNN(
        in_node_dim=DEFAULT_NODE_FEATURE_DIM,
        in_edge_dim=mgnn_config.get("in_edge_dim", 0),
        node_output_dims=mgnn_config.get("node_output_dims", 3),
        graph_output_dims=mgnn_config.get("graph_output_dims", 19),
        energy_output_dims=mgnn_config.get("energy_output_dims", 1),
        hidden_dim=mgnn_config.get("hidden_channels", 160),
        n_blocks=mgnn_config.get("num_layers", 4),
    )
    logger.info("✅ DJMGNN model created successfully")
    
    logger.info("🚀 Moving model to device...")
    model = model.to(device)
    logger.info("✅ Model moved to device successfully")
    
    # AdamW optimizer with cosine annealing
    logger.info("⚙️ Creating optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    logger.info("✅ Optimizer created successfully")
    
    logger.info("📅 Creating scheduler...")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=5e-6
    )
    logger.info("✅ Scheduler created successfully")
    
    # Get shared parameter for GradNorm (backbone parameter from last block)
    logger.info("🔗 Getting backbone parameter for GradNorm...")
    backbone_parameter = model.blocks[-1].transition_layers[-1].weight  # Last transition layer weight
    logger.info("✅ Backbone parameter obtained successfully")
    
    # Proper GradNorm loss balancer using lucidrains implementation
    # Updated to handle 4 losses: node_loss, graph_loss, energy_loss, physics_loss
    logger.info("⚖️ Creating GradNorm loss weighter...")
    loss_weighter = GradNormLossWeighter(
        num_losses=4,  # node_loss, graph_loss, energy_loss, physics_loss (rotational)
        learning_rate=1e-4,
        restoring_force_alpha=0.5,  # o3-pro recommended alpha
        grad_norm_parameters=backbone_parameter
    )
    logger.info("✅ GradNorm loss weighter created successfully")

    # Resume from checkpoint if needed
    logger.info("💾 Checking for checkpoint resume...")
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
    logger.info("✅ Checkpoint check completed")

    # Training Loop
    logger.info("🎯 BEGINNING TRAINING LOOP")
    logger.info(f"   • QM9 Dataset: {len(ds_graph):,} samples")
    logger.info(f"   • SPICE Dataset: {len(ds_node):,} samples") 
    logger.info(f"   • Total Training Data: {len(ds_graph) + len(ds_node):,} samples")
    logger.info("=" * 80)
    
    # Initialize enhanced logger with checkpoint directory for CSV logs
    logger.info("📊 Initializing enhanced logger...")
    enhanced_logger = EnhancedTrainingLogger(
        max_steps=args.max_steps, 
        log_every=args.log_every,
        log_dir=args.checkpoint_dir
    )
    logger.info("✅ Enhanced logger initialized")
    
    logger.info("🎬 Starting enhanced logger display...")
    enhanced_logger.start()
    logger.info("✅ Enhanced logger display started")
    
    # Initialize curriculum manager for 3-phase training
    logger.info("📚 Initializing curriculum manager...")
    curriculum_manager = CurriculumManager(model, enhanced_logger)
    logger.info("✅ Curriculum manager initialized")
    
    # Set initial phase (Phase 1: PIMEH only)
    logger.info("⚡ Setting initial curriculum phase...")
    curriculum_manager.update_phase(start_step, optimizer)
    logger.info("✅ Initial curriculum phase set")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta
    )
    logger.info(f"🛑 Early stopping initialized: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    
    print(f"\n🚀 STARTING 40K TRAINING RUN - TARGET: 95% ACCURACY")
    print(f"📊 Datasets: QM9 ({len(ds_graph):,}) + SPICE ({len(ds_node):,}) = {len(ds_graph) + len(ds_node):,} total")
    print(f"⏱️  Expected Runtime: ~3-4 hours\n")
    print("✨ Enhanced logging with live progress bars enabled!")
    print(f"📊 CSV logs will be saved to: {enhanced_logger.metrics_csv_path}")
    print(f"📈 Phase logs will be saved to: {enhanced_logger.phase_csv_path}\n")
    
    start_time = time.time()
    best_loss = float('inf')
    
    # Initialize losses dictionary to prevent UnboundLocalError in finally block
    losses = {
        'total_loss': 0.0,
        'node_loss': 0.0,
        'graph_loss': 0.0,
        'energy_loss': 0.0,
        'physics_loss': 0.0
    }
    
    try:
        for step in range(start_step, args.max_steps):
            try:
                batch, task_type = next(cycle_iter)
            except StopIteration:
                logger.error("Data iterator exhausted")
                break

            # Update curriculum phase if needed
            phase_changed = curriculum_manager.update_phase(step, optimizer)
            
            # Perform training step with curriculum management
            losses = train_step(
                model, optimizer, loss_weighter, batch, device, task_type,
                curriculum_manager=curriculum_manager, step=step
            )
            
            # Update scheduler
            scheduler.step()
            
            # Update enhanced logger with current step information
            lr = scheduler.get_last_lr()[0]
            try:
                enhanced_logger.update(step, task_type, losses, lr, curriculum_manager.current_phase)
            except Exception as e:
                logger.warning(f"Enhanced logger update failed: {e}")
                # Fallback to basic logging
                if step % args.log_every == 0:
                    progress_pct = (step / args.max_steps) * 100
                    logger.info(f"🎯 {progress_pct:5.1f}% ({step:,}/{args.max_steps:,}) | 📋 {task_type.upper()} | Loss: {losses['total_loss']:.4f}")

            # Validation and early stopping check
            if step % args.validate_every == 0 and step > 0:
                logger.info(f"🔍 Running validation at step {step}...")
                val_loss = validate_model(model, val_loader, device, num_batches=50)
                logger.info(f"📊 Validation loss: {val_loss:.6f}")
                
                # Check early stopping and save best model if improved
                # Note: early_stopping updates its best_loss internally when val_loss improves
                previous_best = early_stopping.best_loss
                should_stop = early_stopping(val_loss, step)
                
                # Save best model if validation improved (best_loss was updated)
                if early_stopping.best_loss < previous_best:
                    save_checkpoint(
                        model, optimizer, graph_scaler, node_scaler,
                        step, losses["total_loss"], args.seed, args.checkpoint_dir,
                        is_best=True, val_loss=val_loss
                    )
                
                # Stop training if early stopping triggered
                if should_stop:
                    logger.info("🛑 Early stopping triggered - terminating training")
                    break

            # Save regular checkpoints
            if step % args.save_every == 0 and step > 0:
                save_checkpoint(
                    model, optimizer, graph_scaler, node_scaler,
                    step, losses["total_loss"], args.seed, args.checkpoint_dir
                )

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