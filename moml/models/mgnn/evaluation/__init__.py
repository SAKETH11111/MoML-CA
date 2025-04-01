"""
MGNN Evaluation Module

This module provides tools for evaluating and making predictions
with trained MGNN models.
"""

from moml.models.mgnn.evaluation.predictor import (
    MGNNPredictor,
    create_predictor,
    batch_predict_from_files
)

__all__ = [
    "MGNNPredictor",
    "create_predictor",
    "batch_predict_from_files"
]