"""
moml/models/mgnn/training/trainer.py

Trainer module for molecular graph neural networks.

This module provides a comprehensive trainer class for handling the training 
and evaluation of molecular graph neural network models. It includes support
for various loss functions, optimizers, callbacks, and both graph-level and 
node-level predictions.

Main Components:
    - MGNNTrainer: Main trainer class for MGNN model training and validation
    - Training utilities: Helper functions for model creation and training loops
    - Support for custom callbacks and training monitoring
"""

import os
from typing import Union, Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from moml.core import create_graph_processor
from moml.models.mgnn import DJMGNN
from moml.models.mgnn.evaluation import MGNNPredictor

class MGNNTrainer:
    """
    Trainer for molecular graph neural network models.

    This class handles training, validation, and model management.
    For prediction functionality, use :py:meth:`get_predictor` to obtain an
    :class:`~moml.models.mgnn.evaluation.predictor.MGNNPredictor` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[str] = None,
        callbacks: Optional[List] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            config: Configuration for training
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            loss_fn: Loss function
            device: Device to use for training
            callbacks: List of callbacks to use during training
        """
        # Store configuration
        self.config = config

        # Set device
        self.device = (
            device
            if device is not None
            else self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model
        self.model = model.to(self.device)

        # Set up training components
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or self._setup_optimizer()
        self.loss_fn = loss_fn or self._setup_loss_function()

        # Set up callbacks
        self.callbacks = callbacks or []

        # Training history
        self.history = {"train_loss": [], "val_loss": []}

        self.stop_training = False
        self.best_val_loss = float("inf")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Set up the optimizer based on configuration."""
        optimizer_type = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0)

        if optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _setup_loss_function(self) -> Callable:
        """Set up the loss function based on configuration."""
        task_type = self.config.get("task_type", "regression")

        if task_type == "regression":
            return nn.MSELoss()
        elif task_type == "classification":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _call_callbacks(self, hook_name, *args, **kwargs):
        """Call a hook method on all callbacks."""
        for callback in self.callbacks:
            hook_method = getattr(callback, hook_name, None)
            if hook_method is not None:
                hook_method(self, *args, **kwargs)

    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute loss for model outputs and targets.
        
        Supports both tensor and dictionary formats for multi-task learning.
        
        Args:
            outputs: Model predictions (tensor or dict with 'graph_pred'/'node_pred')
            targets: Target values (tensor or dict with 'graph_targets'/'node_targets')
            
        Returns:
            torch.Tensor: Computed loss value
            
        Raises:
            TypeError: If outputs and targets have mismatched types
        """
        if isinstance(outputs, dict) and isinstance(targets, dict):
            total_loss = torch.tensor(0.0, device=self.device)
            num_loss_components = 0
            
            # Compute graph-level loss if available
            if "graph_pred" in outputs and "graph_targets" in targets:
                total_loss += self.loss_fn(outputs["graph_pred"], targets["graph_targets"])
                num_loss_components += 1
                
            # Compute node-level loss if available  
            if "node_pred" in outputs and "node_targets" in targets:
                total_loss += self.loss_fn(outputs["node_pred"], targets["node_targets"])
                num_loss_components += 1

            # Handle case where no matching components found
            if num_loss_components == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return total_loss / num_loss_components if num_loss_components > 0 else total_loss
            
        elif isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            return self.loss_fn(outputs, targets)
            
        else:
            raise TypeError(
                f"Outputs and targets must both be dicts or both be tensors. "
                f"Got outputs type: {type(outputs)}, targets type: {type(targets)}"
            )

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

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
            )

            # Prepare targets
            targets = {}
            if hasattr(batch, "y"):
                if isinstance(outputs, dict) and "graph_pred" in outputs:
                    targets["graph_targets"] = batch.y
                else:
                    targets = batch.y

            if hasattr(batch, "node_y"):
                if isinstance(outputs, dict) and "node_pred" in outputs:
                    targets["node_targets"] = batch.node_y

            # Compute loss
            loss = self._compute_loss(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            # Call batch end callbacks
            batch_logs = {"loss": loss.item()}
            self._call_callbacks("on_batch_end", batch_idx, logs=batch_logs)

        # Calculate average loss
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return epoch_loss

    def validate(self) -> float:
        """
        Validate the model on the validation set.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, "edge_attr", None),
                    batch=getattr(batch, "batch", None),
                )

                # Prepare targets
                targets = {}
                if hasattr(batch, "y"):
                    if isinstance(outputs, dict) and "graph_pred" in outputs:
                        targets["graph_targets"] = batch.y
                    else:
                        targets = batch.y

                if hasattr(batch, "node_y"):
                    if isinstance(outputs, dict) and "node_pred" in outputs:
                        targets["node_targets"] = batch.node_y

                # Compute loss
                loss = self._compute_loss(outputs, targets)

                # Update statistics
                total_loss += loss.item()
                num_batches += 1

        # Calculate average loss
        val_loss = total_loss / num_batches if num_batches > 0 else 0

        return val_loss

    def train(self, epochs: Optional[int] = None, log_interval: int = 1) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train for (defaults to config value)
            log_interval: Interval for logging progress

        Returns:
            Training history
        """
        # Set number of epochs
        epochs = int(epochs or self.config.get("epochs", 100))

        # Initialize training
        self.stop_training = False
        print(f"Training on device: {self.device}")

        # Call train begin callbacks
        self._call_callbacks("on_train_begin")

        # Train for specified number of epochs
        for epoch in range(epochs):
            if self.stop_training:
                break

            # Call epoch begin callbacks
            self._call_callbacks("on_epoch_begin", epoch)

            # Train one epoch
            train_loss = self.train_epoch()

            # Validate if validation loader is available
            val_loss = self.validate() if self.val_loader is not None else None

            # Update history
            self.history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

                # Update best validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            # Print progress
            if (epoch + 1) % log_interval == 0:
                log_msg = f"Epoch {epoch + 1}/{epochs}, train_loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", val_loss: {val_loss:.6f}"
                print(log_msg)

            # Call epoch end callbacks
            epoch_logs = {"train_loss": train_loss}
            if val_loss is not None:
                epoch_logs["val_loss"] = val_loss
            self._call_callbacks("on_epoch_end", epoch, logs=epoch_logs)

        # Call train end callbacks
        self._call_callbacks("on_train_end")

        return self.history

    def save_model(self, filepath: str) -> str:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), filepath)

        return filepath

    def save_checkpoint(self, filepath: str) -> str:
        """
        Save a checkpoint with model and training state.

        Args:
            filepath: Path to save the checkpoint

        Returns:
            Path to the saved checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        # Save checkpoint
        torch.save(checkpoint, filepath)

        return filepath

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a checkpoint with model and training state.

        Args:
            filepath: Path to the checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training history if available
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        # Load best validation loss if available
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]

    def plot_training_curves(self, filepath: Optional[str] = None) -> str:
        """
        Plot training and validation loss curves.

        Args:
            filepath: Optional path to save the plot

        Returns:
            Path to the saved plot if filepath is provided, otherwise empty string
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")

        if "val_loss" in self.history and self.history["val_loss"]:
            plt.plot(self.history["val_loss"], label="Validation Loss")

        plt.title("Training Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if filepath:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save plot
            plt.savefig(filepath)
            plt.close()
            return filepath

        plt.show()
        return ""

    def get_predictor(self) -> MGNNPredictor:
        """
        Get a predictor for the trained model.

        Returns:
            MGNNPredictor instance for making predictions with the model
        """
        return MGNNPredictor(model=self.model, config=self.config, device=self.device)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    loss_fn: Callable,
    device: str,
) -> float:
    """Train a model for one epoch. (Implementation unchanged)"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=getattr(batch, "batch", None),
        )

        # Prepare targets
        targets = {}
        if hasattr(batch, "y"):
            if isinstance(outputs, dict) and "graph_pred" in outputs:
                targets["graph_targets"] = batch.y
            else:
                targets = batch.y

        if hasattr(batch, "node_y"):
            if isinstance(outputs, dict) and "node_pred" in outputs:
                targets["node_targets"] = batch.node_y

        # Compute loss with support for dict/tensor formats
        current_loss: torch.Tensor
        if isinstance(outputs, dict) and isinstance(targets, dict):
            total_loss_val = torch.tensor(0.0, device=device)
            num_loss_components = 0
            if "graph_pred" in outputs and "graph_targets" in targets:
                total_loss_val += loss_fn(outputs["graph_pred"], targets["graph_targets"])
                num_loss_components += 1
            if "node_pred" in outputs and "node_targets" in targets:
                total_loss_val += loss_fn(outputs["node_pred"], targets["node_targets"])
                num_loss_components += 1

            if num_loss_components == 0:
                # Fallback: if no components matched, but both are dicts, this is an issue.
                # Defaulting to 0 to avoid crash, but needs review for standalone usage.
                current_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                current_loss = total_loss_val / num_loss_components if num_loss_components > 0 else total_loss_val
        elif isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            current_loss = loss_fn(outputs, targets)
        else:
            raise TypeError(
                f"Standalone train_epoch: Outputs and targets must both be dicts or tensors. "
                f"Got outputs type: {type(outputs)}, targets type: {type(targets)}"
            )
        loss = current_loss  # Assign to 'loss' for backward pass

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Calculate average loss
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return epoch_loss


def _infer_in_dim_from_loader(loader: Optional[DataLoader]) -> Optional[int]:
    """Helper to infer input dimension from DataLoader."""
    if loader is None or not hasattr(loader, "dataset"):
        return None
    try:
        sample_data = loader.dataset[0]
        if hasattr(sample_data, "x") and sample_data.x is not None:
            return sample_data.x.shape[1]
        elif hasattr(sample_data, "node_feat") and sample_data.node_feat is not None: # For PyG Data
            return sample_data.node_feat.shape[1]
        elif isinstance(sample_data, tuple) and len(sample_data) > 0 and hasattr(sample_data[0], 'x') and sample_data[0].x is not None: # DGL case
            return sample_data[0].x.shape[1]
    except (IndexError, AttributeError, TypeError):
        return None
    return None
def create_trainer(
    config: Dict,
    train_loader: Optional[DataLoader] = None,
    model: Optional[nn.Module] = None,
) -> MGNNTrainer:
    """
    Create a trainer with a new model.

    Args:
        config: Configuration for training and model
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        callbacks: List of callbacks to use during training

    Returns:
        Configured trainer with initialized model
    """
    # Create graph processor to get dimensions
    create_graph_processor(config)

    # Prepare arguments for DJMGNN constructor
    model_kwargs = {
        "hidden_dim": config.get("hidden_dim", 64),
        "n_blocks": config.get("n_blocks", 3),
        "layers_per_block": config.get("layers_per_block", 2),
        "jk_mode": config.get("jk_mode", "cat"),
        "node_out_dim": config.get("node_out_dim", 1),
        "graph_out_dim": config.get("graph_out_dim", 1),
        "dropout": config.get("dropout", 0.2),
        "edge_attr_dim": config.get("edge_attr_dim", 0),
    }

    # Set input feature dimension
    if "in_dim" in config:
        model_kwargs["in_dim"] = config["in_dim"]
    else:
        inferred = _infer_in_dim_from_loader(train_loader)
        if inferred is not None:
            model_kwargs["in_dim"] = inferred
        else:
            raise ValueError(
                "`in_dim` is missing from `config` and could not be "
                "inferred from the provided DataLoader. Please supply it "
                "explicitly."
            )

    model = DJMGNN(**model_kwargs)

    # Setup optimizer and loss function
    optimizer_type = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    task_type = config.get("task_type", "regression").lower()
    if task_type == "regression":
        loss_fn = nn.MSELoss()
    elif task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return MGNNTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
