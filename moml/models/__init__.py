"""
moml/models/__init__.py

MoML Models Package - Comprehensive Machine Learning Models for Molecular Data

This package provides a complete suite of molecular machine learning models and utilities:

Core Components:
- Graph Neural Networks: Advanced GNN architectures including DJMGNN and HMGNN for molecular property prediction
- Training Infrastructure: Comprehensive training utilities with callbacks, early stopping, and checkpointing
- Evaluation Framework: Complete prediction and metrics calculation system with visualization tools

Models:
- DJMGNN: Dense Jumping Knowledge Molecular Graph Neural Network for enhanced feature propagation
- HMGNN: Hierarchical Molecular Graph Neural Network for multi-scale molecular representation learning

Key Features:
- Graceful dependency handling with informative error messages for missing optional dependencies
- Comprehensive training pipeline with advanced callbacks and scheduling
- Extensive evaluation metrics for both regression and classification tasks
- Production-ready architecture with proper error handling and logging

Public API:
    Graph Models:
        - GraphConvLayer, DenseGNNBlock, JKAggregator: Core GNN building blocks
        - DJMGNN: Dense Jumping Knowledge Molecular Graph Neural Network
        - HMGNN, create_hierarchical_mgnn: Hierarchical molecular modeling
    
    Training Infrastructure:
        - MGNNTrainer, train_epoch, create_trainer: Training utilities
        - EarlyStopping, ModelCheckpoint, LearningRateScheduler: Training callbacks
    
    Evaluation Framework:
        - MGNNPredictor, create_predictor: Prediction utilities
        - batch_predict_from_files: Batch prediction pipeline
        - calculate_metrics, calculate_regression_metrics, calculate_classification_metrics: Metrics
        - calculate_node_level_metrics, calculate_graph_level_metrics: Specialized metrics
        - visualize_predictions: Visualization utilities
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from .mgnn.djmgnn import (
        GraphConvLayer as _GraphConvLayer,
        DenseGNNBlock as _DenseGNNBlock,
        JKAggregator as _JKAggregator,
        DJMGNN as _DJMGNN,
    )
    from .mgnn.hmgnn import (
        HMGNN as _HMGNN,
        create_hierarchical_mgnn as _create_hierarchical_mgnn,
    )
    from .mgnn.training.trainer import (
        MGNNTrainer as _MGNNTrainer,
        train_epoch as _train_epoch,
        create_trainer as _create_trainer,
    )
    from .mgnn.training.callbacks import (
        EarlyStopping as _EarlyStopping,
        ModelCheckpoint as _ModelCheckpoint,
        LearningRateScheduler as _LearningRateScheduler,
    )
    from .mgnn.evaluation.predictor import (
        MGNNPredictor as _MGNNPredictor,
        create_predictor as _create_predictor,
        batch_predict_from_files as _batch_predict_from_files,
    )
    from .mgnn.evaluation.metrics import (
        calculate_metrics as _calculate_metrics,
        calculate_regression_metrics as _calculate_regression_metrics,
        calculate_classification_metrics as _calculate_classification_metrics,
        calculate_node_level_metrics as _calculate_node_level_metrics,
        calculate_graph_level_metrics as _calculate_graph_level_metrics,
        visualize_predictions as _visualize_predictions,
    )

logger = logging.getLogger(__name__)

# Import errors to handle gracefully
_IMPORT_ERRORS: Dict[str, str] = {}

# Helper functions for dynamic dummy creation
def _create_dummy_class(name: str, error_msg: str) -> type:
    """Create a dummy class that raises ImportError when instantiated."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    
    return type(name, (), {"__init__": __init__})

def _create_dummy_function(name: str, error_msg: str) -> Callable[..., Any]:
    """Create a dummy function that raises ImportError when called."""
    def dummy(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"{name} requires additional dependencies: {error_msg}")
    return dummy

# Graph Neural Network Models and Layers
try:
    from .mgnn.djmgnn import (
        GraphConvLayer,
        DenseGNNBlock,
        JKAggregator,
        DJMGNN,
    )
except ImportError as e:
    _IMPORT_ERRORS['mgnn'] = str(e)
    
    GraphConvLayer = _create_dummy_class("GraphConvLayer", _IMPORT_ERRORS['mgnn'])  # type: ignore
    DenseGNNBlock = _create_dummy_class("DenseGNNBlock", _IMPORT_ERRORS['mgnn'])  # type: ignore
    JKAggregator = _create_dummy_class("JKAggregator", _IMPORT_ERRORS['mgnn'])  # type: ignore
    DJMGNN = _create_dummy_class("DJMGNN", _IMPORT_ERRORS['mgnn'])  # type: ignore

# Hierarchical Graph Neural Network Models
try:
    from .mgnn.hmgnn import (
        HMGNN,
        create_hierarchical_mgnn,
    )
except ImportError as e:
    _IMPORT_ERRORS['hmgnn'] = str(e)
    
    HMGNN = _create_dummy_class("HMGNN", _IMPORT_ERRORS['hmgnn'])  # type: ignore
    create_hierarchical_mgnn = _create_dummy_function("create_hierarchical_mgnn", _IMPORT_ERRORS['hmgnn'])

# Training Infrastructure and Utilities
try:
    from .mgnn.training import (
        MGNNTrainer,
        train_epoch,
        create_trainer,
        EarlyStopping,
        ModelCheckpoint,
        LearningRateScheduler,
    )
except ImportError as e:
    _IMPORT_ERRORS['training'] = str(e)
    
    MGNNTrainer = _create_dummy_class("MGNNTrainer", _IMPORT_ERRORS['training'])  # type: ignore
    train_epoch = _create_dummy_function("train_epoch", _IMPORT_ERRORS['training'])
    create_trainer = _create_dummy_function("create_trainer", _IMPORT_ERRORS['training'])
    EarlyStopping = _create_dummy_class("EarlyStopping", _IMPORT_ERRORS['training'])  # type: ignore
    ModelCheckpoint = _create_dummy_class("ModelCheckpoint", _IMPORT_ERRORS['training'])  # type: ignore
    LearningRateScheduler = _create_dummy_class("LearningRateScheduler", _IMPORT_ERRORS['training'])  # type: ignore

# Evaluation and Prediction Infrastructure
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
except ImportError as e:
    _IMPORT_ERRORS['evaluation'] = str(e)
    
    MGNNPredictor = _create_dummy_class("MGNNPredictor", _IMPORT_ERRORS['evaluation'])  # type: ignore
    create_predictor = _create_dummy_function("create_predictor", _IMPORT_ERRORS['evaluation'])
    batch_predict_from_files = _create_dummy_function("batch_predict_from_files", _IMPORT_ERRORS['evaluation'])
    calculate_metrics = _create_dummy_function("calculate_metrics", _IMPORT_ERRORS['evaluation'])
    calculate_regression_metrics = _create_dummy_function("calculate_regression_metrics", _IMPORT_ERRORS['evaluation'])
    calculate_classification_metrics = _create_dummy_function("calculate_classification_metrics", _IMPORT_ERRORS['evaluation'])
    calculate_node_level_metrics = _create_dummy_function("calculate_node_level_metrics", _IMPORT_ERRORS['evaluation'])
    calculate_graph_level_metrics = _create_dummy_function("calculate_graph_level_metrics", _IMPORT_ERRORS['evaluation'])
    visualize_predictions = _create_dummy_function("visualize_predictions", _IMPORT_ERRORS['evaluation'])

# Public API Definition
__all__ = [
    # Graph Neural Network Models and Layers
    "GraphConvLayer",
    "DenseGNNBlock",
    "JKAggregator",
    "DJMGNN",
    "HMGNN",
    "create_hierarchical_mgnn",
    # Training Infrastructure
    "MGNNTrainer",
    "train_epoch",
    "create_trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    # Evaluation and Prediction
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
