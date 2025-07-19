"""
moml/models/mgnn/evaluation/__init__.py

MGNN Evaluation Module

This module provides comprehensive tools for evaluating and making predictions
with trained Molecular Graph Neural Network (MGNN) models. It includes 
functionality for model prediction, performance metrics calculation, and 
visualization of results.

Main Components:
    - MGNNPredictor: Class for loading trained models and making predictions
    - Metrics functions: Calculate regression/classification performance metrics
    - Visualization: Tools for plotting prediction results and model performance
"""

# Import predictor functionality
from moml.models.mgnn.evaluation.predictor import (
    MGNNPredictor,
    create_predictor,
    batch_predict_from_files,
)

# Import metrics functionality
from moml.models.mgnn.evaluation.metrics import (
    calculate_metrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_node_level_metrics,
    calculate_graph_level_metrics,
    visualize_predictions,
)

# Define public API
__all__ = [
    # Predictor classes and functions
    "MGNNPredictor",
    "create_predictor",
    "batch_predict_from_files",
    # Metrics calculation functions
    "calculate_metrics",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    "calculate_node_level_metrics",
    "calculate_graph_level_metrics",
    # Visualization functions
    "visualize_predictions",
]
