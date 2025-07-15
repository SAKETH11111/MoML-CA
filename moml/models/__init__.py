"""
MoML Models Package

Public API:
- GraphConvLayer, DenseGNNBlock, JKAggregator, DJMGNN: graph neural network layers and model
- MGNNTrainer, train_epoch, create_trainer, EarlyStopping, ModelCheckpoint, LearningRateScheduler: training utilities
- MGNNPredictor, create_predictor, batch_predict_from_files, calculate_metrics, calculate_regression_metrics, calculate_classification_metrics, calculate_node_level_metrics, calculate_graph_level_metrics, visualize_predictions: evaluation utilities
"""

# Graph neural network layers and models
try:
    from .mgnn import (
        GraphConvLayer,
        DenseGNNBlock,
        JKAggregator,
        DJMGNN,
    )
except ImportError:
    # Create dummy classes when dependencies are not available
    class GraphConvLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GraphConvLayer requires additional dependencies")
    
    class DenseGNNBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"DenseGNNBlock requires additional dependencies")
    
    class JKAggregator:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"JKAggregator requires additional dependencies")
    
    class DJMGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"DJMGNN requires additional dependencies")

# Training utilities and callbacks
try:
    from .mgnn.training import (
        MGNNTrainer,
        train_epoch,
        create_trainer,
        EarlyStopping,
        ModelCheckpoint,
        LearningRateScheduler,
    )
except ImportError:
    # Create dummy classes when dependencies are not available
    class MGNNTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MGNNTrainer requires additional dependencies")
    
    def train_epoch(*args, **kwargs):
        raise ImportError(f"train_epoch requires additional dependencies")
    
    def create_trainer(*args, **kwargs):
        raise ImportError(f"create_trainer requires additional dependencies")
    
    class EarlyStopping:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"EarlyStopping requires additional dependencies")
    
    class ModelCheckpoint:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ModelCheckpoint requires additional dependencies")
    
    class LearningRateScheduler:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"LearningRateScheduler requires additional dependencies")

# Evaluation and prediction utilities
try:
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
except ImportError:
    # Create dummy classes when dependencies are not available
    class MGNNPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MGNNPredictor requires additional dependencies")
    
    def create_predictor(*args, **kwargs):
        raise ImportError(f"create_predictor requires additional dependencies")
    
    def batch_predict_from_files(*args, **kwargs):
        raise ImportError(f"batch_predict_from_files requires additional dependencies")
    
    def calculate_metrics(*args, **kwargs):
        raise ImportError(f"calculate_metrics requires additional dependencies")
    
    def calculate_regression_metrics(*args, **kwargs):
        raise ImportError(f"calculate_regression_metrics requires additional dependencies")
    
    def calculate_classification_metrics(*args, **kwargs):
        raise ImportError(f"calculate_classification_metrics requires additional dependencies")
    
    def calculate_node_level_metrics(*args, **kwargs):
        raise ImportError(f"calculate_node_level_metrics requires additional dependencies")
    
    def calculate_graph_level_metrics(*args, **kwargs):
        raise ImportError(f"calculate_graph_level_metrics requires additional dependencies")
    
    def visualize_predictions(*args, **kwargs):
        raise ImportError(f"visualize_predictions requires additional dependencies")

__all__ = [
    "GraphConvLayer",
    "DenseGNNBlock",
    "JKAggregator",
    "DJMGNN",
    "MGNNTrainer",
    "train_epoch",
    "create_trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "MGNNPredictor",
    "create_predictor",
    "batch_predict_from_files",
    "calculate_metrics",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    "calculate_node_level_metrics",
    "calculate_graph_level_metrics",
    "visualize_predictions",
]
