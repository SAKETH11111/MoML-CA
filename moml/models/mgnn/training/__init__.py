"""
Training module for DJMGNN.

This module provides utilities for training and evaluating models on
molecular graph datasets.
"""

from moml.models.mgnn.training.trainer import MGNNTrainer, train_epoch, create_trainer

from moml.models.mgnn.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

__all__ = [
    # Training classes
    "MGNNTrainer",
    "train_epoch",
    "create_trainer",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
]
