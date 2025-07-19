"""
moml/models/mgnn/training/joint_trainer.py

Joint trainer for coordinated DJMGNN and HMGNN training.

This module implements the joint training infrastructure for the hybrid molecular
machine learning framework, enabling coordinated optimization of both DJMGNN and
HMGNN models through alternating training, cross-model fusion, and multi-task learning.

Main Components:
    - JointMGNNTrainer: Extended trainer for joint model training
    - AlternatingTrainingStrategy: Coordinated training logic
    - Multi-task loss management and optimization
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from moml.models.mgnn.joint_mgnn import JointMGNN
from moml.models.mgnn.training.trainer import MGNNTrainer

logger = logging.getLogger(__name__)


class AlternatingTrainingStrategy:
    """
    Strategy class for alternating between DJMGNN and HMGNN training phases.
    
    Implements various alternating optimization schemes including fixed alternation,
    adaptive switching based on loss improvements, and phase-based training.
    """
    
    def __init__(
        self,
        strategy: str = "fixed_alternating",
        switch_frequency: int = 10,
        patience: int = 5,
        min_improvement: float = 1e-4
    ) -> None:
        """
        Initialize alternating training strategy.
        
        Args:
            strategy: Strategy type ('fixed_alternating', 'adaptive', 'phase_based')
            switch_frequency: Number of batches between switches (for fixed_alternating)
            patience: Number of steps to wait for improvement (for adaptive)
            min_improvement: Minimum loss improvement threshold (for adaptive)
        """
        self.strategy = strategy
        self.switch_frequency = switch_frequency
        self.patience = patience
        self.min_improvement = min_improvement
        
        # State tracking
        self.current_model = "djmgnn"
        self.step_count = 0
        self.last_switch_step = 0
        self.last_losses = {"djmgnn": float('inf'), "hmgnn": float('inf')}
        self.no_improvement_count = 0
    
    def should_switch_model(self, current_loss: float) -> bool:
        """
        Determine if we should switch between models.
        
        Args:
            current_loss: Current training loss
            
        Returns:
            True if should switch to the other model
        """
        self.step_count += 1
        
        if self.strategy == "fixed_alternating":
            return (self.step_count - self.last_switch_step) >= self.switch_frequency
        
        elif self.strategy == "adaptive":
            previous_loss = self.last_losses[self.current_model]
            improvement = previous_loss - current_loss
            
            if improvement < self.min_improvement:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            
            self.last_losses[self.current_model] = current_loss
            return self.no_improvement_count >= self.patience
        
        elif self.strategy == "phase_based":
            # Switch based on training phases (implementation can be extended)
            return (self.step_count - self.last_switch_step) >= self.switch_frequency
        
        return False
    
    def switch_model(self) -> str:
        """Switch to the other model and return the new current model."""
        self.current_model = "hmgnn" if self.current_model == "djmgnn" else "djmgnn"
        self.last_switch_step = self.step_count
        self.no_improvement_count = 0
        logger.info(f"Switched to {self.current_model.upper()} training")
        return self.current_model
    
    def get_current_model(self) -> str:
        """Get the current active model."""
        return self.current_model


class JointMGNNTrainer(MGNNTrainer):
    """
    Extended trainer for joint DJMGNN and HMGNN training.
    
    This trainer handles coordinated optimization of both models through various
    strategies including alternating training, joint optimization, and multi-task learning.
    """
    
    def __init__(
        self,
        joint_model: JointMGNN,
        config: Dict[str, Any],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        hierarchical_train_loader: Optional[DataLoader] = None,
        hierarchical_val_loader: Optional[DataLoader] = None,
        djmgnn_optimizer: Optional[optim.Optimizer] = None,
        hmgnn_optimizer: Optional[optim.Optimizer] = None,
        joint_optimizer: Optional[optim.Optimizer] = None,
        device: Optional[str] = None,
        callbacks: Optional[List] = None,
        training_strategy: str = "joint",
        alternating_config: Optional[Dict[str, Any]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the joint trainer.
        
        Args:
            joint_model: JointMGNN model to train
            config: Configuration for training
            train_loader: DataLoader for standard graph training data
            val_loader: DataLoader for standard graph validation data
            hierarchical_train_loader: DataLoader for hierarchical training data
            hierarchical_val_loader: DataLoader for hierarchical validation data
            djmgnn_optimizer: Optimizer for DJMGNN parameters
            hmgnn_optimizer: Optimizer for HMGNN parameters
            joint_optimizer: Optimizer for joint training
            device: Device to use for training
            callbacks: List of callbacks
            training_strategy: Training strategy ('joint', 'alternating', 'pretrain')
            alternating_config: Configuration for alternating training
            task_weights: Weights for different tasks in multi-task learning
        """
        # Initialize base trainer
        super().__init__(
            model=joint_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            callbacks=callbacks
        )
        
        self.joint_model = joint_model
        self.training_strategy = training_strategy
        self.task_weights = task_weights or {}
        
        # Data loaders
        self.hierarchical_train_loader = hierarchical_train_loader
        self.hierarchical_val_loader = hierarchical_val_loader
        
        # Optimizers
        self.djmgnn_optimizer = djmgnn_optimizer or self._setup_djmgnn_optimizer()
        self.hmgnn_optimizer = hmgnn_optimizer or self._setup_hmgnn_optimizer()
        self.joint_optimizer = joint_optimizer or self._setup_joint_optimizer()
        
        # Alternating training strategy
        if alternating_config is None:
            alternating_config = {}
        self.alternating_strategy = AlternatingTrainingStrategy(**alternating_config)
        
        # Training state
        self.phase = "pretraining"  # pretraining, joint_training, fine_tuning
        self.pretraining_epochs = config.get("pretraining_epochs", 10)
        self.joint_training_epochs = config.get("joint_training_epochs", 50)
        self.fine_tuning_epochs = config.get("fine_tuning_epochs", 20)
    
    def _setup_djmgnn_optimizer(self) -> optim.Optimizer:
        """Set up optimizer for DJMGNN parameters."""
        lr = self.config.get("djmgnn_lr", self.config.get("learning_rate", 0.001))
        weight_decay = self.config.get("djmgnn_weight_decay", self.config.get("weight_decay", 0))
        
        djmgnn_params = list(self.joint_model.djmgnn.parameters()) + \
                       list(self.joint_model.djmgnn_proj.parameters())
        
        optimizer_type = self.config.get("optimizer", "adam").lower()
        if optimizer_type == "adam":
            return optim.Adam(djmgnn_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(djmgnn_params, lr=lr, weight_decay=weight_decay)
        else:
            return optim.SGD(djmgnn_params, lr=lr, weight_decay=weight_decay)
    
    def _setup_hmgnn_optimizer(self) -> optim.Optimizer:
        """Set up optimizer for HMGNN parameters."""
        lr = self.config.get("hmgnn_lr", self.config.get("learning_rate", 0.001))
        weight_decay = self.config.get("hmgnn_weight_decay", self.config.get("weight_decay", 0))
        
        hmgnn_params = list(self.joint_model.hmgnn.parameters()) + \
                      list(self.joint_model.hmgnn_proj.parameters())
        
        optimizer_type = self.config.get("optimizer", "adam").lower()
        if optimizer_type == "adam":
            return optim.Adam(hmgnn_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(hmgnn_params, lr=lr, weight_decay=weight_decay)
        else:
            return optim.SGD(hmgnn_params, lr=lr, weight_decay=weight_decay)
    
    def _setup_joint_optimizer(self) -> optim.Optimizer:
        """Set up optimizer for joint training."""
        lr = self.config.get("joint_lr", self.config.get("learning_rate", 0.001))
        weight_decay = self.config.get("joint_weight_decay", self.config.get("weight_decay", 0))
        
        optimizer_type = self.config.get("optimizer", "adam").lower()
        if optimizer_type == "adam":
            return optim.Adam(self.joint_model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(self.joint_model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.SGD(self.joint_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _get_current_optimizer(self) -> optim.Optimizer:
        """Get the current optimizer based on training strategy and phase."""
        if self.training_strategy == "joint":
            return self.joint_optimizer
        elif self.training_strategy == "alternating":
            current_model = self.alternating_strategy.get_current_model()
            return self.djmgnn_optimizer if current_model == "djmgnn" else self.hmgnn_optimizer
        else:
            return self.joint_optimizer
    
    def train_step_joint(self, batch: Any, hierarchical_batch: Optional[Any] = None) -> Dict[str, float]:
        """
        Perform a joint training step on both models.
        
        Args:
            batch: Standard graph batch
            hierarchical_batch: Hierarchical graph batch for HMGNN
            
        Returns:
            Dictionary of loss values
        """
        self.joint_model.train()
        self.joint_optimizer.zero_grad()
        
        # Move batches to device
        batch = batch.to(self.device)
        if hierarchical_batch is not None:
            hierarchical_batch = self._move_hierarchical_batch_to_device(hierarchical_batch)
        
        # Prepare inputs
        inputs = self._prepare_joint_inputs(batch, hierarchical_batch)
        
        # Forward pass
        outputs = self.joint_model(**inputs, use_fusion=True, return_individual=True)
        
        # Prepare targets
        targets = self._prepare_targets(batch, hierarchical_batch)
        
        # Compute joint loss
        total_loss, individual_losses = self.joint_model.compute_joint_loss(
            outputs, targets, self.task_weights
        )
        
        # Backward pass
        if torch.isfinite(total_loss) and total_loss > 0:
            total_loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.joint_model.parameters(), max_grad_norm)
            
            self.joint_optimizer.step()
        
        # Prepare loss dict
        loss_dict = {"total_loss": total_loss.item()}
        loss_dict.update(individual_losses)
        
        return loss_dict
    
    def train_step_alternating(self, batch: Any, hierarchical_batch: Optional[Any] = None) -> Dict[str, float]:
        """
        Perform an alternating training step.
        
        Args:
            batch: Standard graph batch
            hierarchical_batch: Hierarchical graph batch
            
        Returns:
            Dictionary of loss values
        """
        current_model = self.alternating_strategy.get_current_model()
        current_optimizer = self._get_current_optimizer()
        
        self.joint_model.train()
        current_optimizer.zero_grad()
        
        # Move batches to device
        batch = batch.to(self.device)
        if hierarchical_batch is not None:
            hierarchical_batch = self._move_hierarchical_batch_to_device(hierarchical_batch)
        
        # Prepare inputs
        inputs = self._prepare_joint_inputs(batch, hierarchical_batch)
        
        # Forward pass with model-specific focus
        if current_model == "djmgnn":
            # Focus on DJMGNN training
            inputs['use_fusion'] = False  # Disable fusion for individual training
            outputs = self.joint_model(**inputs, return_individual=True)
            
            # Use only DJMGNN outputs for loss
            filtered_outputs = outputs['djmgnn_out']
        else:
            # Focus on HMGNN training
            inputs['use_fusion'] = False
            outputs = self.joint_model(**inputs, return_individual=True)
            
            # Use only HMGNN outputs for loss
            filtered_outputs = outputs['hmgnn_out']
        
        # Prepare targets
        targets = self._prepare_targets(batch, hierarchical_batch)
        
        # Compute model-specific loss
        if current_model == "djmgnn" and 'graph_targets' in targets:
            loss = nn.MSELoss()(filtered_outputs['graph_pred'], targets['graph_targets'])
        elif current_model == "hmgnn" and 'graph_targets' in targets:
            loss = nn.MSELoss()(filtered_outputs['graph_pred'], targets['graph_targets'])
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Backward pass
        if torch.isfinite(loss) and loss > 0:
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get("max_grad_norm", 1.0)
            if current_model == "djmgnn":
                torch.nn.utils.clip_grad_norm_(self.joint_model.djmgnn.parameters(), max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.joint_model.hmgnn.parameters(), max_grad_norm)
            
            current_optimizer.step()
        
        # Check if we should switch models
        if self.alternating_strategy.should_switch_model(loss.item()):
            self.alternating_strategy.switch_model()
        
        return {
            "total_loss": loss.item(),
            f"{current_model}_loss": loss.item(),
            "current_model": current_model
        }
    
    def train_epoch(self) -> float:
        """
        Train the joint model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.joint_model.train()
        total_loss = 0
        num_batches = 0
        
        # Create iterators for both data loaders
        train_iter = iter(self.train_loader)
        hierarchical_iter = iter(self.hierarchical_train_loader) if self.hierarchical_train_loader else None
        
        # Use the loader with more batches as the primary iterator
        primary_loader = self.train_loader
        if (self.hierarchical_train_loader and 
            len(self.hierarchical_train_loader) > len(self.train_loader)):
            primary_loader = self.hierarchical_train_loader
        
        progress_bar = tqdm(primary_loader, desc=f"Training ({self.training_strategy})", leave=False)
        
        for batch_idx, _ in enumerate(progress_bar):
            # Call batch begin callbacks
            self._call_callbacks("on_batch_begin", batch_idx)
            
            # Get batches from both loaders
            try:
                standard_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                standard_batch = next(train_iter)
            
            hierarchical_batch = None
            if hierarchical_iter:
                try:
                    hierarchical_batch = next(hierarchical_iter)
                except StopIteration:
                    hierarchical_iter = iter(self.hierarchical_train_loader)
                    hierarchical_batch = next(hierarchical_iter)
            
            # Perform training step based on strategy
            if self.training_strategy == "joint":
                losses = self.train_step_joint(standard_batch, hierarchical_batch)
            elif self.training_strategy == "alternating":
                losses = self.train_step_alternating(standard_batch, hierarchical_batch)
            else:
                losses = self.train_step_joint(standard_batch, hierarchical_batch)
            
            # Update statistics
            total_loss += losses["total_loss"]
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": losses["total_loss"]})
            
            # Call batch end callbacks
            self._call_callbacks("on_batch_end", batch_idx, logs=losses)
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """
        Validate the joint model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.joint_model.eval()
        total_loss = 0
        num_batches = 0
        
        val_iter = iter(self.val_loader)
        hierarchical_val_iter = iter(self.hierarchical_val_loader) if self.hierarchical_val_loader else None
        
        with torch.no_grad():
            for batch_idx, _ in enumerate(self.val_loader):
                # Get validation batches
                try:
                    standard_batch = next(val_iter)
                except StopIteration:
                    break
                
                hierarchical_batch = None
                if hierarchical_val_iter:
                    try:
                        hierarchical_batch = next(hierarchical_val_iter)
                    except StopIteration:
                        hierarchical_val_iter = iter(self.hierarchical_val_loader)
                        hierarchical_batch = next(hierarchical_val_iter)
                
                # Move to device
                standard_batch = standard_batch.to(self.device)
                if hierarchical_batch is not None:
                    hierarchical_batch = self._move_hierarchical_batch_to_device(hierarchical_batch)
                
                # Forward pass
                inputs = self._prepare_joint_inputs(standard_batch, hierarchical_batch)
                outputs = self.joint_model(**inputs, use_fusion=True)
                
                # Prepare targets and compute loss
                targets = self._prepare_targets(standard_batch, hierarchical_batch)
                loss, _ = self.joint_model.compute_joint_loss(outputs, targets, self.task_weights)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _move_hierarchical_batch_to_device(self, hierarchical_batch: Any) -> Any:
        """Move hierarchical batch data to device."""
        if isinstance(hierarchical_batch, list):
            return [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in scale_data.items()}
                for scale_data in hierarchical_batch
            ]
        else:
            return hierarchical_batch.to(self.device)
    
    def _prepare_joint_inputs(self, batch: Any, hierarchical_batch: Optional[Any] = None) -> Dict[str, Any]:
        """Prepare inputs for joint model forward pass."""
        inputs = {
            'x': batch.x,
            'edge_index': batch.edge_index,
            'edge_attr': getattr(batch, 'edge_attr', None),
            'batch': getattr(batch, 'batch', None),
            'dist': getattr(batch, 'dist', None)
        }
        
        if hierarchical_batch is not None:
            inputs['scale_data'] = hierarchical_batch
            # Add hierarchical-specific inputs if available
            inputs['maps'] = getattr(hierarchical_batch, 'maps', None)
            inputs['edge_pairs_cs'] = getattr(hierarchical_batch, 'edge_pairs_cs', None)
        
        return inputs
    
    def _prepare_targets(self, batch: Any, hierarchical_batch: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """Prepare target dictionary for loss computation."""
        targets = {}
        
        # Standard targets
        if hasattr(batch, 'y'):
            targets['qm9_properties'] = batch.y
            targets['graph_targets'] = batch.y
        
        if hasattr(batch, 'node_y'):
            targets['force_field'] = batch.node_y
            targets['node_targets'] = batch.node_y
        
        # PFAS-specific targets
        if hasattr(batch, 'pfas_y'):
            targets['pfas_properties'] = batch.pfas_y
        
        # Treatment efficacy targets
        if hasattr(batch, 'treatment_y'):
            targets['treatment_efficacy'] = batch.treatment_y
        
        return targets


def create_joint_trainer(
    djmgnn_config: Dict[str, Any],
    hmgnn_config: Dict[str, Any],
    joint_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    hierarchical_train_loader: Optional[DataLoader] = None,
    hierarchical_val_loader: Optional[DataLoader] = None,
    device: Optional[str] = None
) -> JointMGNNTrainer:
    """
    Factory function to create a joint trainer.
    
    Args:
        djmgnn_config: DJMGNN configuration
        hmgnn_config: HMGNN configuration
        joint_config: Joint training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        hierarchical_train_loader: Hierarchical training data loader
        hierarchical_val_loader: Hierarchical validation data loader
        device: Training device
        
    Returns:
        Configured JointMGNNTrainer instance
    """
    from moml.models.mgnn.joint_mgnn import create_joint_mgnn
    
    # Create joint model
    joint_model = create_joint_mgnn(djmgnn_config, hmgnn_config, joint_config)
    
    # Create trainer
    trainer = JointMGNNTrainer(
        joint_model=joint_model,
        config=joint_config,
        train_loader=train_loader,
        val_loader=val_loader,
        hierarchical_train_loader=hierarchical_train_loader,
        hierarchical_val_loader=hierarchical_val_loader,
        device=device,
        training_strategy=joint_config.get("training_strategy", "joint"),
        alternating_config=joint_config.get("alternating_config", {}),
        task_weights=joint_config.get("task_weights", {})
    )
    
    return trainer