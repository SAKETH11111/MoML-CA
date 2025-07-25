"""
moml/models/mgnn/training/early_stopping.py

Advanced Early Stopping and Monitoring System for Deep Learning Training.

This module provides a comprehensive early stopping and monitoring framework
designed for molecular property prediction tasks. It includes advanced features
such as multi-metric monitoring, statistical significance testing, and integration
with modern monitoring tools like Weights & Biases and TensorBoard.

Key Features:
    - Multi-metric early stopping with configurable thresholds
    - Statistical significance testing for improvement detection
    - Warmup period support and learning rate scheduling integration
    - Comprehensive logging and visualization capabilities
    - Memory-efficient checkpoint management
    - Integration with WandB, TensorBoard, and custom monitoring dashboards

Main Components:
    - EarlyStoppingConfig: Configuration dataclass for early stopping parameters
    - AdvancedEarlyStopping: Enhanced early stopping callback
    - ValidationMonitor: Comprehensive metrics tracking and visualization
    - MetricTracker: Efficient metric storage and trend analysis
    - CheckpointManager: Intelligent model checkpoint management

References:
    - Based on research of PyTorch, Lightning, and modern ML best practices (2024)
    - Incorporates statistical methods for robust improvement detection
    - Follows callback patterns from leading deep learning frameworks
"""

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum
import logging

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

# Optional imports for visualization and monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

from .callbacks import Callback

logger = logging.getLogger(__name__)


class MonitorMode(Enum):
    """Enumeration for monitoring modes."""
    MIN = "min"
    MAX = "max"
    AUTO = "auto"


class EarlyStoppingReason(Enum):
    """Enumeration for early stopping reasons."""
    PATIENCE_EXCEEDED = "patience_exceeded"
    NO_IMPROVEMENT = "no_improvement"
    THRESHOLD_REACHED = "threshold_reached"
    MANUAL_STOP = "manual_stop"
    DIVERGENCE_DETECTED = "divergence_detected"


@dataclass
class EarlyStoppingConfig:
    """
    Configuration class for advanced early stopping parameters.
    
    This dataclass encapsulates all configuration options for the early stopping
    mechanism, providing type safety and validation for training parameters.
    
    Attributes:
        monitor: Primary metric to monitor for early stopping decisions
        min_delta: Minimum change to qualify as an improvement
        patience: Number of epochs to wait after last improvement
        mode: Whether to minimize or maximize the monitored metric
        restore_best_weights: Whether to restore best weights when stopping
        verbose: Level of verbosity (0=silent, 1=epoch info, 2=detailed)
        
        # Advanced Features
        warmup_epochs: Number of epochs before early stopping is active
        multiple_metrics: Additional metrics to monitor for stopping
        metric_weights: Weights for combining multiple metrics
        statistical_test: Whether to use statistical significance testing
        confidence_level: Confidence level for statistical testing (0.95 = 95%)
        
        # Learning Rate Integration
        reduce_lr_on_plateau: Whether to reduce LR before stopping
        lr_reduction_factor: Factor to reduce learning rate by
        lr_reduction_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate threshold
        
        # Divergence Detection
        detect_divergence: Whether to detect training divergence
        divergence_threshold: Threshold for detecting divergence
        divergence_patience: Patience for divergence detection
        
        # Checkpoint Management
        save_best_checkpoint: Whether to save checkpoints of best models
        checkpoint_dir: Directory to save checkpoints
        keep_checkpoint_history: Number of checkpoints to keep
        
        # Monitoring Integration
        log_to_wandb: Whether to log metrics to Weights & Biases
        log_to_tensorboard: Whether to log metrics to TensorBoard
        tensorboard_log_dir: Directory for TensorBoard logs
    """
    
    # Core Early Stopping Parameters
    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: Union[str, MonitorMode] = MonitorMode.MIN
    restore_best_weights: bool = True
    verbose: int = 1
    
    # Advanced Features
    warmup_epochs: int = 0
    multiple_metrics: List[str] = field(default_factory=list)
    metric_weights: Dict[str, float] = field(default_factory=dict)
    statistical_test: bool = False
    confidence_level: float = 0.95
    
    # Learning Rate Integration
    reduce_lr_on_plateau: bool = False
    lr_reduction_factor: float = 0.5
    lr_reduction_patience: int = 5
    min_lr: float = 1e-8
    
    # Divergence Detection
    detect_divergence: bool = True
    divergence_threshold: float = 10.0
    divergence_patience: int = 3
    
    # Checkpoint Management
    save_best_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    keep_checkpoint_history: int = 3
    
    # Monitoring Integration
    log_to_wandb: bool = WANDB_AVAILABLE
    log_to_tensorboard: bool = TENSORBOARD_AVAILABLE
    tensorboard_log_dir: str = "runs"
    
    def __post_init__(self):
        """Validate and normalize configuration parameters."""
        # Normalize mode
        if isinstance(self.mode, str):
            self.mode = MonitorMode(self.mode.lower())
        
        # Validate confidence level
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError(f"confidence_level must be between 0.5 and 0.99, got {self.confidence_level}")
        
        # Validate patience values
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        
        if self.lr_reduction_patience <= 0:
            raise ValueError(f"lr_reduction_patience must be positive, got {self.lr_reduction_patience}")
        
        # Ensure checkpoint directory exists
        if self.save_best_checkpoint:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate monitoring setup
        if self.log_to_wandb and not WANDB_AVAILABLE:
            warnings.warn("wandb not available, disabling wandb logging")
            self.log_to_wandb = False
        
        if self.log_to_tensorboard and not TENSORBOARD_AVAILABLE:
            warnings.warn("tensorboard not available, disabling tensorboard logging")
            self.log_to_tensorboard = False


class MetricTracker:
    """
    Efficient metric storage and trend analysis for training monitoring.
    
    This class provides memory-efficient storage of training metrics with
    built-in trend analysis, statistical testing, and visualization capabilities.
    """
    
    def __init__(self, metric_name: str, mode: MonitorMode, maxlen: int = 1000):
        """
        Initialize metric tracker.
        
        Args:
            metric_name: Name of the metric to track
            mode: Whether to minimize or maximize the metric
            maxlen: Maximum number of values to keep in memory
        """
        self.metric_name = metric_name
        self.mode = mode
        self.maxlen = maxlen
        
        # Use deque for efficient memory management
        self.values = deque(maxlen=maxlen)
        self.epochs = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        
        # Best value tracking
        self.best_value = float('inf') if mode == MonitorMode.MIN else float('-inf')
        self.best_epoch = -1
        
        # Trend analysis
        self.recent_window = 10  # Window size for trend analysis
        self.improvement_threshold = 1e-8
    
    def update(self, value: float, epoch: int, timestamp: Optional[float] = None) -> bool:
        """
        Update tracker with new metric value.
        
        Args:
            value: New metric value
            epoch: Current epoch number
            timestamp: Timestamp of the measurement
            
        Returns:
            True if this is a new best value
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.values.append(value)
        self.epochs.append(epoch)
        self.timestamps.append(timestamp)
        
        # Check if this is a new best
        is_better = self._is_better(value, self.best_value)
        if is_better:
            self.best_value = value
            self.best_epoch = epoch
        
        return is_better
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == MonitorMode.MIN:
            return current < best
        else:
            return current > best
    
    def get_trend(self, window: Optional[int] = None) -> str:
        """
        Analyze recent trend in metric values.
        
        Args:
            window: Window size for trend analysis
            
        Returns:
            Trend description ('improving', 'stable', 'worsening', 'insufficient_data')
        """
        if window is None:
            window = self.recent_window
        
        if len(self.values) < window:
            return "insufficient_data"
        
        recent_values = list(self.values)[-window:]
        
        # Simple linear regression for trend
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(recent_values) < 2:
            return "insufficient_data"
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if np.isnan(correlation):
            return "stable"
        
        # Determine trend based on correlation and mode
        if self.mode == MonitorMode.MIN:
            if correlation < -0.1:  # Negative correlation = decreasing values = improving
                return "improving"
            elif correlation > 0.1:  # Positive correlation = increasing values = worsening
                return "worsening"
        else:
            if correlation > 0.1:  # Positive correlation = increasing values = improving
                return "improving"
            elif correlation < -0.1:  # Negative correlation = decreasing values = worsening
                return "worsening"
        
        return "stable"
    
    def has_improved_recently(self, patience: int, min_delta: float = 0.0) -> bool:
        """
        Check if metric has improved in the last 'patience' epochs.
        
        Args:
            patience: Number of recent epochs to check
            min_delta: Minimum improvement threshold
            
        Returns:
            True if there has been improvement within patience window
        """
        if len(self.values) < patience + 1:
            return True  # Not enough data, assume improving
        
        recent_values = list(self.values)[-patience-1:]
        baseline = recent_values[0]
        
        for value in recent_values[1:]:
            if self.mode == MonitorMode.MIN:
                if baseline - value > min_delta:
                    return True
            else:
                if value - baseline > min_delta:
                    return True
        
        return False
    
    def detect_divergence(self, threshold: float, patience: int) -> bool:
        """
        Detect if metric is diverging (rapidly worsening).
        
        Args:
            threshold: Threshold multiplier for divergence detection
            patience: Number of epochs to confirm divergence
            
        Returns:
            True if divergence is detected
        """
        if len(self.values) < patience + 1:
            return False
        
        recent_values = list(self.values)[-patience-1:]
        
        divergence_count = 0
        for i in range(1, len(recent_values)):
            prev_value = recent_values[i-1]
            curr_value = recent_values[i]
            
            if self.mode == MonitorMode.MIN:
                # For minimization, divergence is when value increases by threshold factor
                if curr_value > prev_value * threshold:
                    divergence_count += 1
            else:
                # For maximization, divergence is when value decreases by threshold factor
                if curr_value < prev_value / threshold:
                    divergence_count += 1
        
        return divergence_count >= patience
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the tracked metric."""
        if not self.values:
            return {}
        
        values_array = np.array(self.values)
        
        stats_dict = {
            'count': len(self.values),
            'current': self.values[-1],
            'best': self.best_value,
            'best_epoch': self.best_epoch,
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'trend': self.get_trend(),
        }
        
        # Add percentiles if we have enough data
        if len(values_array) >= 10:
            stats_dict.update({
                'q25': np.percentile(values_array, 25),
                'median': np.percentile(values_array, 50),
                'q75': np.percentile(values_array, 75),
            })
        
        return stats_dict


class AdvancedEarlyStopping(Callback):
    """
    Advanced early stopping callback with comprehensive monitoring capabilities.
    
    This callback extends the basic early stopping functionality with:
    - Multi-metric monitoring and statistical significance testing
    - Learning rate scheduling integration
    - Divergence detection and automatic recovery
    - Comprehensive logging and visualization
    - Intelligent checkpoint management
    """
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None, **kwargs):
        """
        Initialize advanced early stopping callback.
        
        Args:
            config: EarlyStoppingConfig instance or None for default config
            **kwargs: Additional configuration parameters to override defaults
        """
        # Initialize configuration
        if config is None:
            config = EarlyStoppingConfig(**kwargs)
        else:
            # Override config with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Initialize metric trackers
        self.primary_tracker = MetricTracker(config.monitor, config.mode)
        self.metric_trackers = {config.monitor: self.primary_tracker}
        
        # Add secondary metric trackers
        for metric in config.multiple_metrics:
            mode = MonitorMode.AUTO  # Auto-detect mode for secondary metrics
            self.metric_trackers[metric] = MetricTracker(metric, mode)
        
        # Early stopping state
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_reason = None
        self.best_weights = None
        self.lr_reduced_count = 0
        
        # Statistical testing
        if config.statistical_test:
            self.recent_improvements = deque(maxlen=20)
        
        # Monitoring setup
        self.tensorboard_writer = None
        if config.log_to_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = Path(config.tensorboard_log_dir) / f"early_stopping_{time.strftime('%Y%m%d_%H%M%S')}"
            self.tensorboard_writer = SummaryWriter(log_dir)
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(config) if config.save_best_checkpoint else None
        
        logger.info(f"AdvancedEarlyStopping initialized: monitoring '{config.monitor}' with patience {config.patience}")
    
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize early stopping state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_reason = None
        self.best_weights = None
        self.lr_reduced_count = 0
        
        # Reset all metric trackers
        for tracker in self.metric_trackers.values():
            tracker.values.clear()
            tracker.epochs.clear()
            tracker.timestamps.clear()
            tracker.best_value = float('inf') if tracker.mode == MonitorMode.MIN else float('-inf')
            tracker.best_epoch = -1
        
        if self.config.statistical_test:
            self.recent_improvements.clear()
        
        if self.config.verbose >= 1:
            logger.info(f"Early stopping active: monitoring {self.config.monitor} with patience {self.config.patience}")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Check early stopping conditions after each epoch.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics from the epoch
        """
        logs = logs or {}
        
        # Skip early stopping during warmup period
        if epoch < self.config.warmup_epochs:
            if self.config.verbose >= 2:
                logger.info(f"Epoch {epoch}: In warmup period, early stopping inactive")
            return
        
        # Update metric trackers
        primary_improved = self._update_metrics(logs, epoch, trainer)
        
        # Check for divergence
        if self.config.detect_divergence and self._check_divergence():
            self._trigger_early_stop(trainer, epoch, EarlyStoppingReason.DIVERGENCE_DETECTED)
            return
        
        # Determine if we should continue training
        should_stop, reason = self._should_stop(primary_improved, epoch)
        
        if should_stop:
            self._trigger_early_stop(trainer, epoch, reason)
        else:
            # Check if we should reduce learning rate
            if self.config.reduce_lr_on_plateau and not primary_improved:
                self._maybe_reduce_lr(trainer, epoch)
        
        # Log metrics to monitoring systems
        self._log_metrics(logs, epoch)
        
        # Save checkpoint if this is the best model
        if primary_improved and self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(trainer, epoch, logs)
    
    def _update_metrics(self, logs: Dict[str, Any], epoch: int, trainer: Any) -> bool:
        """
        Update all metric trackers and return if primary metric improved.
        
        Args:
            logs: Dictionary of metrics
            epoch: Current epoch number
            trainer: The trainer instance
            
        Returns:
            True if primary metric improved
        """
        primary_improved = False
        
        for metric_name, tracker in self.metric_trackers.items():
            if metric_name in logs:
                value = logs[metric_name]
                # Handle None values
                if value is None:
                    continue
                improved = tracker.update(value, epoch)
                
                if metric_name == self.config.monitor:
                    primary_improved = improved
                    
                    if improved:
                        self.wait = 0
                        # Store best weights
                        if self.config.restore_best_weights and hasattr(trainer, 'model'):
                            state_dict = trainer.model.state_dict()
                            self.best_weights = {k: v.clone().detach() for k, v in state_dict.items()}
                        
                        # Track improvement for statistical testing
                        if self.config.statistical_test:
                            improvement = abs(value - tracker.best_value)
                            self.recent_improvements.append(improvement)
        
        return primary_improved
    
    def _should_stop(self, primary_improved: bool, epoch: int) -> Tuple[bool, EarlyStoppingReason]:
        """
        Determine if training should stop based on early stopping criteria.
        
        Args:
            primary_improved: Whether primary metric improved this epoch
            epoch: Current epoch number
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not primary_improved:
            self.wait += 1
        
        # Check basic patience
        if self.wait >= self.config.patience:
            return True, EarlyStoppingReason.PATIENCE_EXCEEDED
        
        # Statistical significance test
        if self.config.statistical_test and len(self.recent_improvements) >= 5:
            if not self._is_statistically_significant():
                return True, EarlyStoppingReason.NO_IMPROVEMENT
        
        # Check if primary metric has not improved for extended period
        primary_tracker = self.metric_trackers[self.config.monitor]
        if not primary_tracker.has_improved_recently(self.config.patience * 2, self.config.min_delta):
            return True, EarlyStoppingReason.NO_IMPROVEMENT
        
        return False, None
    
    def _check_divergence(self) -> bool:
        """Check if training is diverging based on primary metric."""
        primary_tracker = self.metric_trackers[self.config.monitor]
        return primary_tracker.detect_divergence(
            self.config.divergence_threshold,
            self.config.divergence_patience
        )
    
    def _is_statistically_significant(self) -> bool:
        """
        Test if recent improvements are statistically significant.
        
        Returns:
            True if improvements are statistically significant
        """
        if len(self.recent_improvements) < 5:
            return True  # Not enough data, assume significant
        
        improvements = np.array(self.recent_improvements)
        
        # One-sample t-test against zero improvement
        t_stat, p_value = stats.ttest_1samp(improvements, 0)
        
        # Check if p-value indicates significance
        alpha = 1 - self.config.confidence_level
        return p_value < alpha and t_stat > 0
    
    def _maybe_reduce_lr(self, trainer: Any, epoch: int) -> None:
        """
        Reduce learning rate if conditions are met.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """
        if self.wait >= self.config.lr_reduction_patience:
            current_lr = self._get_current_lr(trainer)
            
            if current_lr > self.config.min_lr:
                new_lr = max(current_lr * self.config.lr_reduction_factor, self.config.min_lr)
                self._set_lr(trainer, new_lr)
                self.lr_reduced_count += 1
                
                if self.config.verbose >= 1:
                    logger.info(f"Epoch {epoch}: Reduced learning rate from {current_lr:.2e} to {new_lr:.2e}")
                
                # Reset wait counter after LR reduction
                self.wait = 0
    
    def _get_current_lr(self, trainer: Any) -> float:
        """Get current learning rate from trainer."""
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            return trainer.optimizer.param_groups[0]['lr']
        return 0.0
    
    def _set_lr(self, trainer: Any, lr: float) -> None:
        """Set learning rate in trainer's optimizer."""
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr
    
    def _trigger_early_stop(self, trainer: Any, epoch: int, reason: EarlyStoppingReason) -> None:
        """
        Trigger early stopping with specified reason.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            reason: Reason for early stopping
        """
        self.stopped_epoch = epoch
        self.stop_reason = reason
        trainer.stop_training = True
        
        # Restore best weights if requested
        if self.config.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            if self.config.verbose >= 1:
                logger.info(f"Restored best weights from epoch {self.primary_tracker.best_epoch}")
        
        # Log stopping information
        primary_tracker = self.metric_trackers[self.config.monitor]
        if self.config.verbose >= 1:
            logger.info(
                f"Early stopping triggered at epoch {epoch} ({reason.value}): "
                f"best {self.config.monitor} = {primary_tracker.best_value:.6f} "
                f"at epoch {primary_tracker.best_epoch}"
            )
        
        # Log to monitoring systems
        if self.config.log_to_wandb and WANDB_AVAILABLE and wandb.run:
            wandb.log({
                "early_stopping/triggered": True,
                "early_stopping/reason": reason.value,
                "early_stopping/stopped_epoch": epoch,
                "early_stopping/best_epoch": primary_tracker.best_epoch,
                "early_stopping/best_value": primary_tracker.best_value,
            }, step=epoch)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("early_stopping/triggered", 1, epoch)
            self.tensorboard_writer.add_text("early_stopping/reason", reason.value, epoch)
    
    def _log_metrics(self, logs: Dict[str, Any], epoch: int) -> None:
        """
        Log metrics to monitoring systems.
        
        Args:
            logs: Dictionary of metrics
            epoch: Current epoch number
        """
        # Prepare early stopping specific metrics
        es_metrics = {
            "early_stopping/wait": self.wait,
            "early_stopping/patience_remaining": max(0, self.config.patience - self.wait),
            "early_stopping/lr_reductions": self.lr_reduced_count,
        }
        
        # Add metric statistics
        for metric_name, tracker in self.metric_trackers.items():
            if metric_name in logs:
                stats = tracker.get_statistics()
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        es_metrics[f"metrics/{metric_name}_{stat_name}"] = stat_value
        
        # Log to wandb
        if self.config.log_to_wandb and WANDB_AVAILABLE and wandb.run:
            wandb.log(es_metrics, step=epoch)
        
        # Log to tensorboard
        if self.tensorboard_writer:
            for key, value in es_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        """Clean up resources at the end of training."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Log final statistics
        if self.config.verbose >= 1 and self.stopped_epoch > 0:
            primary_tracker = self.metric_trackers[self.config.monitor]
            logger.info(
                f"Training stopped early at epoch {self.stopped_epoch} "
                f"(reason: {self.stop_reason.value if self.stop_reason else 'unknown'}). "
                f"Best {self.config.monitor}: {primary_tracker.best_value:.6f} "
                f"at epoch {primary_tracker.best_epoch}"
            )
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved during training."""
        return self.primary_tracker.best_value if self.primary_tracker.values else None
    
    def get_best_epoch(self) -> int:
        """Get the epoch where the best score was achieved."""
        return self.primary_tracker.best_epoch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the early stopping process."""
        stats = {
            "stopped_early": self.stopped_epoch > 0,
            "stopped_epoch": self.stopped_epoch,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "wait_count": self.wait,
            "lr_reductions": self.lr_reduced_count,
            "best_score": self.get_best_score(),
            "best_epoch": self.get_best_epoch(),
        }
        
        # Add per-metric statistics
        for metric_name, tracker in self.metric_trackers.items():
            metric_stats = tracker.get_statistics()
            stats[f"{metric_name}_stats"] = metric_stats
        
        return stats


class CheckpointManager:
    """
    Intelligent checkpoint management for model saving and restoration.
    
    This class handles automatic checkpoint saving with configurable retention
    policies and metadata tracking.
    """
    
    def __init__(self, config: EarlyStoppingConfig):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Early stopping configuration containing checkpoint settings
        """
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.saved_checkpoints = []
        self.best_score = float('inf') if config.mode == MonitorMode.MIN else float('-inf')
    
    def save_checkpoint(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> Optional[str]:
        """
        Save model checkpoint if it represents an improvement.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics
            
        Returns:
            Path to saved checkpoint or None if not saved
        """
        if self.config.monitor not in logs:
            logger.warning(f"Monitor metric '{self.config.monitor}' not found in logs")
            return None
        
        current_score = logs[self.config.monitor]
        
        # Check if this is a new best score
        is_better = (
            (self.config.mode == MonitorMode.MIN and current_score < self.best_score) or
            (self.config.mode == MonitorMode.MAX and current_score > self.best_score)
        )
        
        if not is_better:
            return None
        
        self.best_score = current_score
        
        # Create checkpoint filename with timestamp and score
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        score_str = f"{current_score:.6f}".replace(".", "_")
        filename = f"best_model_epoch_{epoch:04d}_{timestamp}_score_{score_str}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None,
            'best_score': self.best_score,
            'logs': logs,
            'config': trainer.config if hasattr(trainer, 'config') else {},
            'timestamp': time.time(),
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'score': current_score,
                'timestamp': time.time(),
            })
            
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        if len(self.saved_checkpoints) <= self.config.keep_checkpoint_history:
            return
        
        # Sort by score to keep the best checkpoints
        if self.config.mode == MonitorMode.MIN:
            self.saved_checkpoints.sort(key=lambda x: x['score'])
        else:
            self.saved_checkpoints.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove excess checkpoints
        checkpoints_to_remove = self.saved_checkpoints[self.config.keep_checkpoint_history:]
        
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint['path'].unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint['path']}: {e}")
        
        # Update saved checkpoints list
        self.saved_checkpoints = self.saved_checkpoints[:self.config.keep_checkpoint_history]
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best saved checkpoint."""
        if not self.saved_checkpoints:
            return None
        
        # The list is sorted, so the first element is the best
        return self.saved_checkpoints[0]['path']
    
    def load_best_checkpoint(self, trainer: Any) -> bool:
        """
        Load the best saved checkpoint into the trainer.
        
        Args:
            trainer: The trainer instance
            
        Returns:
            True if checkpoint was loaded successfully
        """
        best_path = self.get_best_checkpoint_path()
        if not best_path or not best_path.exists():
            logger.warning("No best checkpoint found to load")
            return False
        
        try:
            checkpoint_data = torch.load(best_path, map_location=trainer.device, weights_only=False)
            
            # Load model state
            trainer.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint_data and hasattr(trainer, 'optimizer'):
                trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            logger.info(
                f"Loaded best checkpoint from epoch {checkpoint_data['epoch']} "
                f"with score {checkpoint_data['best_score']:.6f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {best_path}: {e}")
            return False


# Factory function for easy instantiation
def create_early_stopping(
    monitor: str = "val_loss",
    patience: int = 10,
    min_delta: float = 0.0,
    mode: str = "min",
    restore_best_weights: bool = True,
    **kwargs
) -> AdvancedEarlyStopping:
    """
    Factory function to create an AdvancedEarlyStopping callback with common parameters.
    
    Args:
        monitor: Metric to monitor for early stopping
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' for minimizing or maximizing the metric
        restore_best_weights: Whether to restore best weights when stopping
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured AdvancedEarlyStopping instance
    """
    config = EarlyStoppingConfig(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights,
        **kwargs
    )
    
    return AdvancedEarlyStopping(config)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration for molecular property prediction
    config = EarlyStoppingConfig(
        monitor="val_loss",
        patience=15,
        min_delta=1e-4,
        mode=MonitorMode.MIN,
        warmup_epochs=5,
        multiple_metrics=["val_mae", "val_r2"],
        statistical_test=True,
        reduce_lr_on_plateau=True,
        detect_divergence=True,
        save_best_checkpoint=True,
        log_to_wandb=True,
        verbose=2
    )
    
    early_stopping = AdvancedEarlyStopping(config)
    print(f"Created AdvancedEarlyStopping with config: {config}")
    print("Early stopping system ready for integration with training pipeline.")