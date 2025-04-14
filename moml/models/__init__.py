"""
MoML Models Package

This package provides neural network models for molecular property prediction.
"""

# Import MGNN models
from moml.models.mgnn import (
    GraphConvLayer,
    DenseGNNBlock,
    JKAggregator,
    DJMGNN
)

# Import training components
from moml.models.mgnn.training import (
    MGNNTrainer,
    train_epoch,
    create_trainer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)

# Import evaluation tools
from moml.models.mgnn.evaluation import (
    MGNNPredictor,
    create_predictor,
    visualize_predictions,
    batch_predict_from_files
)

__all__ = [
    # Model components
    "GraphConvLayer",
    "DenseGNNBlock", 
    "JKAggregator",
    "DJMGNN",
    
    # Training components
    "MGNNTrainer",
    "train_epoch",
    "create_trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    
    # Evaluation tools
    "MGNNPredictor",
    "create_predictor",
    "batch_predict_from_files",
    "visualize_predictions"
]