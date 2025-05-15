"""
MGNN Evaluation Module

This module provides tools for evaluating and making predictions
with trained MGNN models.
"""

from moml.models.mgnn.evaluation.predictor import MGNNPredictor, create_predictor, batch_predict_from_files

from moml.models.mgnn.evaluation.metrics import (
    calculate_metrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_node_level_metrics,
    calculate_graph_level_metrics,
    visualize_predictions,
)

__all__ = [
    # Predictor
    "MGNNPredictor",
    "create_predictor",
    "batch_predict_from_files",
    # Metrics
    "calculate_metrics",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    "calculate_node_level_metrics",
    "calculate_graph_level_metrics",
    "visualize_predictions",
]
