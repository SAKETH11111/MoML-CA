"""
moml/models/mgnn/training/__init__.py

MGNN Training Module

This module provides comprehensive tools for training Molecular Graph Neural 
Network (MGNN) models. It includes training orchestration, callback systems,
and utilities for model training workflows.

Main Components:
    - MGNNTrainer: Main trainer class for MGNN model training and validation
    - Callbacks: Training control utilities (early stopping, checkpoints, scheduling)
    - Training utilities: Helper functions for creating trainers and training loops
"""

# Import trainer functionality
from moml.models.mgnn.training.trainer import (
    MGNNTrainer, 
    train_epoch, 
    create_trainer,
)

# Import joint trainer functionality
from moml.models.mgnn.training.joint_trainer import (
    JointMGNNTrainer,
    AlternatingTrainingStrategy,
    create_joint_trainer,
)

# Import callback functionality
from moml.models.mgnn.training.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateScheduler,
)

# Define public API
__all__ = [
    # Training classes and functions
    "MGNNTrainer",
    "train_epoch", 
    "create_trainer",
    # Joint training classes and functions
    "JointMGNNTrainer",
    "AlternatingTrainingStrategy",
    "create_joint_trainer",
    # Callback classes
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
]
