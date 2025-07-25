"""
moml/models/mgnn/training/enhanced_trainer.py

Enhanced Trainer with Advanced Early Stopping and Monitoring.

This module provides an enhanced trainer that integrates the advanced early stopping
and comprehensive monitoring systems with the existing MGNN training infrastructure.
It extends the base MGNNTrainer with modern MLOps practices including intelligent
checkpoint management, real-time monitoring, and configurable training workflows.

Key Features:
    - Integration with AdvancedEarlyStopping and ValidationMonitor
    - Configuration-driven training with Hydra/OmegaConf support
    - Comprehensive metrics tracking and visualization
    - Intelligent checkpoint management with retention policies
    - Learning rate scheduling with plateau detection
    - Integration with WandB, TensorBoard, and custom dashboards
    - Distributed training support preparation
    - Memory-efficient batch processing

Main Components:
    - EnhancedMGNNTrainer: Enhanced trainer with monitoring and early stopping
    - TrainingPipeline: High-level training pipeline with configuration management
    - ModelManager: Advanced model lifecycle management
    - TrainingUtils: Utilities for trainer creation and configuration

References:
    - Extends the base MGNNTrainer from trainer.py
    - Integrates with early_stopping.py and validation_monitor.py
    - Uses configuration system from config.py
"""

import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Optional imports for monitoring and configuration
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

# Import project components
from .trainer import MGNNTrainer
from .early_stopping import AdvancedEarlyStopping, EarlyStoppingConfig, create_early_stopping
from .validation_monitor import ValidationMonitor, DashboardConfig, create_validation_monitor
from .config import TrainingConfig, ConfigManager, create_molecular_config
from .callbacks import Callback

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Advanced model lifecycle management with intelligent saving and loading.
    
    This class handles model checkpointing, version management, and model
    artifact tracking for production training workflows.
    """
    
    def __init__(self, config: TrainingConfig, model: nn.Module):
        """
        Initialize model manager.
        
        Args:
            config: Training configuration
            model: Model to manage
        """
        self.config = config
        self.model = model
        self.checkpoint_dir = Path(config.checkpoint.dirpath)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.saved_checkpoints = []
        self.best_metrics = {}
        
        # Model metadata
        self.model_info = self._gather_model_info()
        
        logger.info(f"ModelManager initialized: {self.model_info['total_params']:,} parameters")
    
    def _gather_model_info(self) -> Dict[str, Any]:
        """Gather comprehensive model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_info = {
            'model_class': self.model.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture_summary': str(self.model),
        }
        
        # Add model-specific information if available
        if hasattr(self.model, 'config'):
            model_info['model_config'] = self.model.config
        
        return model_info
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float], 
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint or None if not saved
        """
        # Check if this checkpoint should be saved
        monitor_metric = self.config.checkpoint.monitor
        if monitor_metric not in metrics or metrics[monitor_metric] is None:
            logger.warning(f"Monitor metric '{monitor_metric}' not found or None in metrics")
            return None
        
        current_score = metrics[monitor_metric]
        if current_score is None:
            logger.warning(f"Monitor metric '{monitor_metric}' is None")
            return None
        
        # Determine if this is a new best
        mode = self.config.checkpoint.mode
        is_better = self._is_better_score(current_score, monitor_metric, mode)
        
        should_save = (
            not self.config.checkpoint.save_best_only or 
            is_better or 
            self.config.checkpoint.save_last
        )
        
        if not should_save:
            return None
        
        # Create checkpoint filename
        checkpoint_filename = self._create_checkpoint_filename(epoch, metrics)
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            # Model and training state
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': dict(self.best_metrics),
            
            # Configuration and metadata
            'config': self.config,
            'model_info': self.model_info,
            'timestamp': time.time(),
            'metadata': metadata or {},
            
            # Training context
            'random_state': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
            },
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        try:
            if self.config.checkpoint.enable_compression:
                self._save_compressed_checkpoint(checkpoint_data, checkpoint_path)
            else:
                torch.save(checkpoint_data, checkpoint_path)
            
            # Update tracking
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'metrics': metrics.copy(),
                'timestamp': time.time(),
                'is_best': is_better,
            })
            
            if is_better:
                self.best_metrics[monitor_metric] = current_score
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _is_better_score(self, current: float, metric: str, mode) -> bool:
        """Check if current score is better than the previous best."""
        if metric not in self.best_metrics:
            return True
        
        best = self.best_metrics[metric]
        
        if mode.value == "min":
            return current < best
        else:
            return current > best
    
    def _create_checkpoint_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Create checkpoint filename with formatting."""
        filename_template = self.config.checkpoint.filename
        
        # Format the filename with available variables
        format_vars = {
            'epoch': epoch,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        
        try:
            formatted_name = filename_template.format(**format_vars)
        except (KeyError, ValueError) as e:
            logger.warning(f"Error formatting checkpoint filename: {e}")
            formatted_name = f"checkpoint_epoch_{epoch:04d}"
        
        return f"{formatted_name}.pt"
    
    def _save_compressed_checkpoint(self, data: Dict[str, Any], path: Path) -> None:
        """Save checkpoint with compression."""
        import gzip
        import pickle
        
        compression_format = self.config.checkpoint.compression_format
        
        if compression_format == "gzip":
            with gzip.open(f"{path}.gz", 'wb') as f:
                pickle.dump(data, f)
        else:
            # Fallback to torch.save
            torch.save(data, path)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        if len(self.saved_checkpoints) <= self.config.checkpoint.save_top_k:
            return
        
        # Sort checkpoints by the monitored metric
        monitor_metric = self.config.checkpoint.monitor
        mode = self.config.checkpoint.mode
        reverse_sort = mode.value == "max"
        
        sorted_checkpoints = sorted(
            self.saved_checkpoints,
            key=lambda x: x['metrics'].get(monitor_metric, float('inf' if mode.value == 'min' else '-inf')),
            reverse=reverse_sort
        )
        
        # Keep the best checkpoints
        checkpoints_to_keep = sorted_checkpoints[:self.config.checkpoint.save_top_k]
        checkpoints_to_remove = [cp for cp in self.saved_checkpoints if cp not in checkpoints_to_keep]
        
        # Remove old checkpoint files
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint['path'].unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
        
        # Update saved checkpoints list
        self.saved_checkpoints = checkpoints_to_keep
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint and restore model state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to return optimizer state
            
        Returns:
            Dictionary with loaded checkpoint data
        """
        try:
            # Load checkpoint
            if checkpoint_path.endswith('.gz'):
                checkpoint_data = self._load_compressed_checkpoint(checkpoint_path)
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Update best metrics
            if 'best_metrics' in checkpoint_data:
                self.best_metrics.update(checkpoint_data['best_metrics'])
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def _load_compressed_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load compressed checkpoint."""
        import gzip
        import pickle
        
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if not self.saved_checkpoints:
            return None
        
        monitor_metric = self.config.checkpoint.monitor
        mode = self.config.checkpoint.mode
        
        best_checkpoint = None
        best_score = float('inf') if mode.value == 'min' else float('-inf')
        
        for checkpoint in self.saved_checkpoints:
            score = checkpoint['metrics'].get(monitor_metric)
            if score is None:
                continue
            
            is_better = (
                (mode.value == 'min' and score < best_score) or
                (mode.value == 'max' and score > best_score)
            )
            
            if is_better:
                best_score = score
                best_checkpoint = checkpoint
        
        return str(best_checkpoint['path']) if best_checkpoint else None


class EnhancedMGNNTrainer(MGNNTrainer):
    """
    Enhanced trainer with advanced early stopping and comprehensive monitoring.
    
    This trainer extends the base MGNNTrainer with modern MLOps practices
    including intelligent early stopping, real-time monitoring, and
    configuration-driven training workflows.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[Dict[str, Any], TrainingConfig],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        early_stopping_config: Optional[EarlyStoppingConfig] = None,
        monitoring_config: Optional[DashboardConfig] = None,
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: Model to train
            config: Training configuration (dict or TrainingConfig)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (will be created from config if None)
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Training device
            callbacks: Additional callbacks
            early_stopping_config: Early stopping configuration
            monitoring_config: Monitoring configuration
        """
        # Convert config to TrainingConfig if needed
        if isinstance(config, dict):
            # Try to convert dict to TrainingConfig
            try:
                self.training_config = TrainingConfig(**config)
            except Exception as e:
                logger.warning(f"Failed to convert config dict to TrainingConfig: {e}")
                self.training_config = TrainingConfig()
                # Keep original dict for backward compatibility
                config_dict = config
        else:
            self.training_config = config
            config_dict = config.__dict__
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=config_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            callbacks=callbacks or []
        )
        
        # Store enhanced components (after parent init so self.optimizer exists)
        self.scheduler = scheduler or self._setup_scheduler()
        self.model_manager = ModelManager(self.training_config, model)
        
        # Setup monitoring and early stopping
        self._setup_monitoring_and_early_stopping(early_stopping_config, monitoring_config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_start_time = None
        
        # Metrics tracking
        self.metrics_history = {}
        self.best_metrics = {}
        
        logger.info("EnhancedMGNNTrainer initialized with advanced monitoring and early stopping")
    
    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler from configuration."""
        if not hasattr(self.training_config, 'scheduler'):
            return None
        
        scheduler_config = self.training_config.scheduler
        scheduler_type = scheduler_config.type
        
        if scheduler_type.value == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        elif scheduler_type.value == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
        elif scheduler_type.value == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=getattr(scheduler_config, 'mode', 'min'),
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
                threshold=scheduler_config.threshold,
                min_lr=scheduler_config.min_lr
            )
        elif scheduler_type.value == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.gamma
            )
        elif scheduler_type.value == "cyclic":
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=scheduler_config.base_lr,
                max_lr=scheduler_config.max_lr,
                step_size_up=scheduler_config.step_size_up,
                mode=scheduler_config.mode
            )
        elif scheduler_type.value == "one_cycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.max_lr_one_cycle,
                epochs=self.training_config.max_epochs,
                steps_per_epoch=len(self.train_loader) if self.train_loader else 100,
                pct_start=scheduler_config.pct_start,
                anneal_strategy=scheduler_config.anneal_strategy,
                div_factor=scheduler_config.div_factor,
                final_div_factor=scheduler_config.final_div_factor
            )
        
        return None
    
    def _setup_monitoring_and_early_stopping(
        self, 
        early_stopping_config: Optional[EarlyStoppingConfig],
        monitoring_config: Optional[DashboardConfig]
    ) -> None:
        """Setup monitoring and early stopping callbacks."""
        # Create early stopping callback if enabled
        if self.training_config.early_stopping.enabled:
            if early_stopping_config is None:
                early_stopping_config = self.training_config.early_stopping
            
            early_stopping = AdvancedEarlyStopping(early_stopping_config)
            self.callbacks.append(early_stopping)
            self.early_stopping = early_stopping
        else:
            self.early_stopping = None
        
        # Create validation monitor if enabled
        if self.training_config.monitoring.enabled:
            if monitoring_config is None:
                # Convert monitoring config to dashboard config
                monitoring_cfg = self.training_config.monitoring
                monitoring_config = DashboardConfig(
                    primary_metrics=monitoring_cfg.primary_metrics,
                    secondary_metrics=monitoring_cfg.secondary_metrics,
                    enable_alerts=monitoring_cfg.enable_alerts,
                    alert_thresholds=monitoring_cfg.alert_thresholds,
                    export_dir=monitoring_cfg.export_dir,
                    auto_export=monitoring_cfg.auto_export,
                    log_to_wandb=monitoring_cfg.log_to_wandb,
                    log_to_tensorboard=monitoring_cfg.log_to_tensorboard,
                )
            
            validation_monitor = ValidationMonitor(monitoring_config)
            self.callbacks.append(validation_monitor)
            self.validation_monitor = validation_monitor
        else:
            self.validation_monitor = None
        
        # Setup WandB if enabled
        if self.training_config.monitoring.log_to_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
    
    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=self.training_config.monitoring.wandb_project or self.training_config.experiment_name,
                entity=self.training_config.monitoring.wandb_entity,
                name=self.training_config.run_name,
                config=self.training_config,
                tags=self.training_config.tags + self.training_config.monitoring.wandb_tags,
                notes=self.training_config.notes or self.training_config.monitoring.wandb_notes,
                reinit=True
            )
            
            # Log model info
            wandb.config.update(self.model_manager.model_info)
            
            logger.info(f"Initialized WandB: {wandb.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.training_config.monitoring.log_to_wandb = False
    
    def train_epoch(self) -> float:
        """Enhanced training epoch with comprehensive metrics tracking."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Metrics for this epoch
        epoch_metrics = {
            'batch_losses': [],
            'learning_rates': [],
            'grad_norms': [],
        }
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.training_config.max_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Call batch begin callbacks
            self._call_callbacks("on_batch_begin", batch_idx)
            
            # Move batch to device
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=getattr(batch, "batch", None),
                pos=getattr(batch, "pos", None),  # Add support for 3D coordinates
            )
            
            # Prepare targets
            targets = self._prepare_targets(batch, outputs)
            
            # Compute loss
            loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            grad_norm = None
            if self.training_config.model.gradient_clipping > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.model.gradient_clipping
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate scheduler (if step-based)
            if self.scheduler and hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Track metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1
            
            # Store batch metrics
            epoch_metrics['batch_losses'].append(batch_loss)
            epoch_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            if grad_norm is not None:
                epoch_metrics['grad_norms'].append(grad_norm.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": batch_loss,
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "grad_norm": f"{grad_norm:.2f}" if grad_norm else "N/A"
            })
            
            # Call batch end callbacks
            batch_logs = {
                "loss": batch_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "global_step": self.global_step,
            }
            if grad_norm is not None:
                batch_logs["grad_norm"] = grad_norm.item()
            
            self._call_callbacks("on_batch_end", batch_idx, logs=batch_logs)
            
            # Log to WandB (step-level)
            if self.training_config.monitoring.log_to_wandb and WANDB_AVAILABLE and wandb.run:
                if self.global_step % self.training_config.monitoring.log_every_n_steps == 0:
                    wandb.log(batch_logs, step=self.global_step)
        
        # Calculate epoch averages
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Store additional epoch metrics
        if epoch_metrics['grad_norms']:
            self.history.setdefault('grad_norm', []).append(np.mean(epoch_metrics['grad_norms']))
        
        self.history.setdefault('learning_rate', []).append(
            np.mean(epoch_metrics['learning_rates'])
        )
        
        return epoch_loss
    
    def _prepare_targets(self, batch, outputs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced target preparation with support for multiple target types."""
        targets = {}
        
        # Graph-level targets
        if hasattr(batch, "y"):
            if isinstance(outputs, dict) and "graph_pred" in outputs:
                targets["graph_targets"] = batch.y
            else:
                return batch.y
        
        # Node-level targets
        if hasattr(batch, "node_y"):
            if isinstance(outputs, dict) and "node_pred" in outputs:
                targets["node_targets"] = batch.node_y
        
        # Energy targets (for force field predictions)
        if hasattr(batch, "energy"):
            targets["energy_targets"] = batch.energy
        
        # PFAS-specific targets
        if hasattr(batch, "pfas_y"):
            targets["pfas_targets"] = batch.pfas_y
        
        # Treatment efficacy targets
        if hasattr(batch, "treatment_y"):
            targets["treatment_targets"] = batch.treatment_y
        
        return targets if targets else torch.tensor(0.0, device=self.device)
    
    def validate(self) -> Dict[str, float]:
        """Enhanced validation with comprehensive metrics."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Additional metrics tracking
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, "edge_attr", None),
                    batch=getattr(batch, "batch", None),
                    pos=getattr(batch, "pos", None),
                )
                
                # Prepare targets
                targets = self._prepare_targets(batch, outputs)
                
                # Compute loss
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for additional metrics
                if isinstance(outputs, dict) and "graph_pred" in outputs:
                    predictions.append(outputs["graph_pred"].cpu())
                    if isinstance(targets, dict) and "graph_targets" in targets:
                        targets_list.append(targets["graph_targets"].cpu())
                elif isinstance(outputs, torch.Tensor):
                    predictions.append(outputs.cpu())
                    if isinstance(targets, torch.Tensor):
                        targets_list.append(targets.cpu())
        
        # Calculate validation metrics
        val_metrics = {}
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_metrics['val_loss'] = val_loss
        
        # Calculate additional metrics if we have predictions and targets
        if predictions and targets_list:
            all_preds = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            
            # Mean Absolute Error
            mae = torch.mean(torch.abs(all_preds - all_targets)).item()
            val_metrics['val_mae'] = mae
            
            # Root Mean Square Error
            rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
            val_metrics['val_rmse'] = rmse
            
            # R² Score (coefficient of determination)
            ss_res = torch.sum((all_targets - all_preds) ** 2)
            ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            val_metrics['val_r2'] = r2.item() if isinstance(r2, torch.Tensor) else r2
        
        return val_metrics
    
    def train(self, epochs: Optional[int] = None, resume_from_checkpoint: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Enhanced training loop with comprehensive monitoring.
        
        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        # Setup training
        epochs = epochs or self.training_config.max_epochs
        self.training_start_time = time.time()
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_data = self.model_manager.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint_data['epoch'] + 1
            
            # Restore optimizer and scheduler states
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint_data and self.scheduler:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Restore history
            if 'history' in checkpoint_data:
                self.history.update(checkpoint_data['history'])
            
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        # Initialize training
        self.stop_training = False
        print(f"Training on device: {self.device}")
        print(f"Model: {self.model_manager.model_info['total_params']:,} parameters")
        
        # Call train begin callbacks
        self._call_callbacks("on_train_begin")
        
        # Training loop
        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                break
            
            self.current_epoch = epoch
            
            # Call epoch begin callbacks
            self._call_callbacks("on_epoch_begin", epoch)
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            for metric_name, value in val_metrics.items():
                self.history.setdefault(metric_name, []).append(value)
            
            # Update best metrics
            self._update_best_metrics(val_metrics)
            
            # Prepare epoch logs
            epoch_logs = {"train_loss": train_loss}
            epoch_logs.update(val_metrics)
            epoch_logs["epoch"] = epoch
            epoch_logs["learning_rate"] = self.optimizer.param_groups[0]['lr']
            
            # Step scheduler (epoch-based)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation loss for plateau scheduler
                    monitor_metric = val_metrics.get('val_loss', train_loss)
                    self.scheduler.step(monitor_metric)
                elif hasattr(self.scheduler, 'step') and not hasattr(self.scheduler, '_step_count'):
                    # Epoch-based schedulers that haven't been stepped in train_epoch
                    self.scheduler.step()
            
            # Log progress
            self._log_epoch_progress(epoch, epochs, epoch_logs)
            
            # Save checkpoint
            self.model_manager.save_checkpoint(
                epoch, epoch_logs, self.optimizer, self.scheduler,
                metadata={"global_step": self.global_step}
            )
            
            # Log to WandB (epoch-level)
            if self.training_config.monitoring.log_to_wandb and WANDB_AVAILABLE and wandb.run:
                wandb.log(epoch_logs, step=epoch)
            
            # Call epoch end callbacks
            self._call_callbacks("on_epoch_end", epoch, logs=epoch_logs)
        
        # Training completed
        training_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Call train end callbacks
        self._call_callbacks("on_train_end")
        
        # Close WandB run
        if self.training_config.monitoring.log_to_wandb and WANDB_AVAILABLE and wandb.run:
            wandb.finish()
        
        return self.history
    
    def _update_best_metrics(self, metrics: Dict[str, float]) -> None:
        """Update best metrics tracking."""
        for metric_name, value in metrics.items():
            # Skip None values
            if value is None:
                continue
                
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = value
            else:
                # Skip if current best is None
                if self.best_metrics[metric_name] is None:
                    self.best_metrics[metric_name] = value
                else:
                    # Assume lower is better for loss metrics, higher for others
                    if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                        if value < self.best_metrics[metric_name]:
                            self.best_metrics[metric_name] = value
                    else:
                        if value > self.best_metrics[metric_name]:
                            self.best_metrics[metric_name] = value
    
    def _log_epoch_progress(self, epoch: int, total_epochs: int, logs: Dict[str, float]) -> None:
        """Log epoch progress with comprehensive metrics."""
        progress_msg = f"Epoch {epoch + 1}/{total_epochs}"
        
        # Add key metrics
        key_metrics = ['train_loss', 'val_loss', 'val_mae', 'val_r2']
        for metric in key_metrics:
            if metric in logs:
                progress_msg += f", {metric}: {logs[metric]:.6f}"
        
        # Add learning rate
        progress_msg += f", lr: {logs.get('learning_rate', 0):.2e}"
        
        print(progress_msg)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        summary = {
            'model_info': self.model_manager.model_info,
            'training_config': self.training_config,
            'training_time_seconds': training_time,
            'epochs_completed': self.current_epoch + 1,
            'global_steps': self.global_step,
            'best_metrics': self.best_metrics,
            'final_metrics': {k: v[-1] if v else None for k, v in self.history.items()},
            'history_length': {k: len(v) for k, v in self.history.items()},
        }
        
        # Add early stopping info if available
        if self.early_stopping:
            summary['early_stopping'] = self.early_stopping.get_statistics()
        
        # Add monitoring info if available
        if self.validation_monitor:
            summary['monitoring'] = self.validation_monitor.get_current_statistics()
        
        return summary


class TrainingPipeline:
    """
    High-level training pipeline with configuration management.
    
    This class provides a simplified interface for setting up and running
    training workflows with automatic configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file or directory
        """
        self.config_manager = ConfigManager(config_path)
        self.trainer = None
        self.config = None
        
    def setup_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config_name: str = "base_config",
        config_overrides: Optional[List[str]] = None
    ) -> EnhancedMGNNTrainer:
        """
        Setup training with configuration.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config_name: Configuration name to load
            config_overrides: Configuration overrides
            
        Returns:
            Configured trainer
        """
        # Load configuration
        self.config = self.config_manager.load_config(config_name, config_overrides)
        
        # Validate configuration
        issues = self.config_manager.validate_config(self.config)
        if issues:
            logger.warning("Configuration validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Create trainer
        self.trainer = EnhancedMGNNTrainer(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        return self.trainer
    
    def run_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the training pipeline.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training summary
        """
        if self.trainer is None:
            raise RuntimeError("Training not setup. Call setup_training() first.")
        
        # Run training
        history = self.trainer.train(epochs)
        
        # Get summary
        summary = self.trainer.get_training_summary()
        summary['history'] = history
        
        return summary


# Utility functions for easy trainer creation
def create_enhanced_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Union[Dict[str, Any], TrainingConfig]] = None,
    **kwargs
) -> EnhancedMGNNTrainer:
    """
    Factory function to create an enhanced trainer with sensible defaults.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured EnhancedMGNNTrainer
    """
    if config is None:
        config = create_molecular_config()
    
    trainer = EnhancedMGNNTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        **kwargs
    )
    
    return trainer


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a molecular property prediction trainer
    from moml.models.mgnn import DJMGNN
    
    # Create model (placeholder)
    model = DJMGNN(
        in_node_dim=29,
        hidden_dim=128,
        n_blocks=4,
        layers_per_block=2,
        graph_output_dims=1,
    )
    
    # Create configuration
    config = create_molecular_config(
        dataset_name="qm9",
        model_type="djmgnn",
        hidden_dim=128,
        batch_size=32,
        max_epochs=10
    )
    
    print("Enhanced trainer configuration:")
    print(f"  Model: {config.model.model_type.value}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Early stopping: {config.early_stopping.enabled}")
    print(f"  Monitoring: {config.monitoring.enabled}")
    print(f"  Checkpointing: {config.checkpoint.enabled}")
    
    print("\nEnhanced MGNN training system ready for use!")