"""
MoML Models Package

Public API:
- GraphConvLayer, DenseGNNBlock, JKAggregator, DJMGNN: graph neural network layers and model
- MGNNTrainer, train_epoch, create_trainer, EarlyStopping, ModelCheckpoint, LearningRateScheduler: training utilities
- MGNNPredictor, create_predictor, batch_predict_from_files, calculate_metrics, calculate_regression_metrics, calculate_classification_metrics, calculate_node_level_metrics, calculate_graph_level_metrics, visualize_predictions: evaluation utilities
"""

# Graph neural network layers and models
from .mgnn import (
    GraphConvLayer,
    DenseGNNBlock,
    JKAggregator,
    DJMGNN,
)

# Training utilities and callbacks
from .mgnn.training import (
    MGNNTrainer,
    train_epoch,
    create_trainer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)

# Evaluation and prediction utilities
from .mgnn.evaluation import (
    MGNNPredictor,
    create_predictor,
    batch_predict_from_files,
    calculate_metrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_node_level_metrics,
    calculate_graph_level_metrics,
    visualize_predictions,
)

__all__ = [
    'GraphConvLayer',
    'DenseGNNBlock',
    'JKAggregator',
    'DJMGNN',
    'MGNNTrainer',
    'train_epoch',
    'create_trainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'MGNNPredictor',
    'create_predictor',
    'batch_predict_from_files',
    'calculate_metrics',
    'calculate_regression_metrics',
    'calculate_classification_metrics',
    'calculate_node_level_metrics',
    'calculate_graph_level_metrics',
    'visualize_predictions',
]