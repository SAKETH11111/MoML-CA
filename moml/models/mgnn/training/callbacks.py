"""
moml/models/mgnn/training/callbacks.py

Callbacks for training graph neural networks.

This module provides callback classes that can be used to customize the
training process by adding functionality like early stopping, model 
checkpointing, and learning rate scheduling. These callbacks follow the 
standard callback pattern used in deep learning frameworks.

Available Callbacks:
    - Callback: Base callback class defining the interface
    - EarlyStopping: Halt training when monitored metric stops improving
    - ModelCheckpoint: Save model checkpoints during training
    - LearningRateScheduler: Adjust learning rate during training
"""

import os
import torch
from typing import Optional, Callable, Any, Dict, Literal


class Callback:
    """
    Base callback class.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        pass

    def on_batch_begin(self, trainer: Any, batch: Any) -> None:
        """Called at the beginning of a batch."""
        pass

    def on_batch_end(self, trainer: Any, batch: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to halt training when a monitored metric
    stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, trainer: Any) -> None:
        """Reset early stopping state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_weights = None

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if training should be stopped after this epoch."""
        logs = logs or {}

        if self.monitor not in logs:
            if self.verbose:
                print(f"Warning: Early stopping monitored metric '{self.monitor}' not available in logs")
            return

        current = logs[self.monitor]

        # Determine if current value is better than best
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            # Reset counters and update best value
            self.best_value = current
            self.wait = 0
            # Deep-copy state_dict to freeze best_weights
            state = trainer.model.state_dict()
            self.best_weights = {k: v.clone().detach() for k, v in state.items()} if state else None
        else:
            # Increment wait counter
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True

                if self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)

                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch} with best {self.monitor} = {self.best_value:.6f}")


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        period: int = 1,
        verbose: bool = True,
        save_optimizer: bool = True,
    ):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path to save checkpoint file
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model
            mode: 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
            period: Interval (number of epochs) between checkpoints
            verbose: Whether to print messages
            save_optimizer: Whether to save optimizer state
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.period = period
        self.verbose = verbose
        self.save_optimizer = save_optimizer

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.epochs_since_last_save = 0

    def on_train_begin(self, trainer: Any) -> None:
        """Reset checkpoint state at the beginning of training."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.epochs_since_last_save = 0

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint if conditions are met."""
        logs = logs or {}
        self.epochs_since_last_save += 1

        # Check if it's time to save based on period
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            if self.save_best_only:
                # Check if current value is an improvement
                if self.monitor not in logs:
                    if self.verbose:
                        print(f"Warning: Monitored metric '{self.monitor}' not available in logs")
                    return

                current = logs[self.monitor]

                if self.mode == "min":
                    improved = current < self.best_value
                else:
                    improved = current > self.best_value

                if improved:
                    # Update best value and save
                    if self.verbose:
                        print(
                            f"Epoch {epoch}: {self.monitor} improved from {self.best_value:.6f} to {current:.6f}, saving model to {self.filepath}"
                        )

                    self.best_value = current

                    # Save checkpoint
                    self._save_checkpoint(trainer, epoch, logs)
            else:
                # Save regardless of improvement
                filepath = self.filepath.format(epoch=epoch, **logs)
                if self.verbose:
                    print(f"Epoch {epoch}: saving model to {filepath}")

                # Save checkpoint
                self._save_checkpoint(trainer, epoch, logs)

    def _save_checkpoint(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]]) -> None:
        """Save the actual checkpoint."""
        # Format filepath if it contains placeholders
        filepath = self.filepath.format(epoch=epoch, **(logs or {}))

        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "logs": logs,
            "config": trainer.config,
        }

        # Add optimizer state if requested
        if self.save_optimizer and trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

        # Save checkpoint
        torch.save(checkpoint, filepath)


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.
    """

    def __init__(
        self,
        schedule: Callable[[int], float],
        monitor: Optional[str] = None,
        mode: Literal['min', 'max'] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0,
        verbose: bool = True,
    ):
        """
        Initialize learning rate scheduler callback.

        Args:
            schedule: Learning rate schedule function or 'reduce_on_plateau'
            monitor: Metric to monitor (required if schedule is 'reduce_on_plateau')
            mode: 'min' or 'max' (for ReduceLROnPlateau)
            factor: Factor by which to reduce learning rate (for ReduceLROnPlateau)
            patience: Number of epochs with no improvement (for ReduceLROnPlateau)
            threshold: Threshold for measuring improvement (for ReduceLROnPlateau)
            cooldown: Cooldown periods after a reduction (for ReduceLROnPlateau)
            min_lr: Minimum learning rate (for ReduceLROnPlateau)
            verbose: Whether to print messages
        """
        self.schedule = schedule
        self.monitor = monitor
        self.mode: Literal['min', 'max'] = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose

        # Check for built-in schedulers
        if isinstance(schedule, str) and schedule == "reduce_on_plateau":
            if monitor is None:
                raise ValueError("monitor must be specified for 'reduce_on_plateau' scheduler")

            self.is_callable = False
        else:
            self.is_callable: bool = True

    def on_train_begin(self, trainer: Any) -> None:
        """Initialize scheduler at the beginning of training."""
        if not self.is_callable and hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            # Create ReduceLROnPlateau scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                cooldown=self.cooldown,
                min_lr=self.min_lr,
            )

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update learning rate at the end of the epoch."""
        logs = logs or {}

        if not self.is_callable and hasattr(self, "scheduler"):
            # For ReduceLROnPlateau
            if self.monitor not in logs:
                if self.verbose:
                    print(f"Warning: Monitored metric '{self.monitor}' not available in logs")
                return

            # Step the scheduler with the monitored metric
            self.scheduler.step(logs[self.monitor])
        else:
            # For callable scheduler
            # Call the schedule function to get the new learning rate
            new_lr = self.schedule(epoch)

            # Update learning rate for each parameter group
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = new_lr

            if self.verbose:
                print(f"Epoch {epoch}: Learning rate set to {new_lr}")
