"""
moml/models/mgnn/evaluation/metrics.py

MGNN Model Evaluation Metrics Module

This module provides comprehensive evaluation metrics and visualization tools for 
Molecular Graph Neural Network (MGNN) models. It supports both regression and 
classification tasks with node-level and graph-level predictions.

The module includes functions for calculating standard machine learning metrics
such as RMSE, MAE, R², accuracy, precision, recall, F1-score, and AUC. It also
provides visualization capabilities for comparing predictions against ground truth.

Functions:
    calculate_metrics: General metrics calculation with task type detection
    calculate_regression_metrics: Regression-specific metrics (RMSE, MAE, R², etc.)
    calculate_classification_metrics: Classification metrics (accuracy, precision, etc.)
    calculate_node_level_metrics: Node-level prediction metrics with masking
    calculate_graph_level_metrics: Graph-level prediction metrics with masking
    visualize_predictions: Create plots comparing predictions vs ground truth
"""

import logging
import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target

# Constants
DEFAULT_THRESHOLD = 0.5
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_DPI = 300
EPSILON = 1e-8  # Small value to prevent division by zero
MIN_SAMPLES_FOR_METRICS = 1

# Setup logging
logger = logging.getLogger(__name__)


def calculate_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.

    This is the main entry point for metrics calculation. It automatically
    handles input conversion and dispatches to appropriate specialized functions
    based on the task type.

    Args:
        true_values: Ground truth values in various formats
        pred_values: Predicted values in various formats  
        task_type: Type of task ('regression' or 'classification')

    Returns:
        Dictionary containing calculated metrics with string keys and float values
    """
    # Convert inputs to numpy arrays with validation
    true_array = _convert_to_numpy(true_values, "true_values")
    pred_array = _convert_to_numpy(pred_values, "pred_values")

    # Validate array shapes
    _validate_array_shapes(true_array, pred_array)

    # Validate minimum sample size
    if true_array.size < MIN_SAMPLES_FOR_METRICS:
        logger.warning(f"Very small sample size: {true_array.size}")

    # Dispatch to appropriate metrics function
    if task_type == "regression":
        return calculate_regression_metrics(true_array, pred_array)
    elif task_type == "classification":
        return calculate_classification_metrics(true_array, pred_array)
    else:
        raise ValueError(
            f"Unsupported task_type: '{task_type}'. "
            f"Supported types are: 'regression', 'classification'"
        )


def _convert_to_numpy(
    values: Union[torch.Tensor, np.ndarray, List], 
    name: str
) -> np.ndarray:
    """
    Convert input values to numpy array with proper error handling.

    Args:
        values: Input values to convert
        name: Name of the variable for error messages

    Returns:
        Converted numpy array
    """
    try:
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        elif isinstance(values, list):
            array = np.array(values)
        elif isinstance(values, np.ndarray):
            array = values
        else:
            raise TypeError(f"Unsupported type for {name}: {type(values)}")

        # Validate the resulting array
        if array.size == 0:
            raise ValueError(f"{name} contains no elements")

        return array

    except ValueError:
        # Re-raise ValueError as-is (for empty arrays, NaN/Inf validation)
        raise
    except TypeError:
        # Re-raise TypeError as-is (for unsupported types)
        raise
    except Exception as e:
        logger.error(f"Failed to convert {name} to numpy array: {e}")
        raise TypeError(f"Could not convert {name} to numpy array: {e}")


def _validate_array_shapes(true_array: np.ndarray, pred_array: np.ndarray) -> None:
    """
    Validate that true and predicted arrays have compatible shapes.

    Args:
        true_array: Ground truth array
        pred_array: Predicted values array
    """
    if true_array.shape != pred_array.shape:
        raise ValueError(
            f"Shape mismatch: true_values shape {true_array.shape} "
            f"!= pred_values shape {pred_array.shape}"
        )


def calculate_regression_metrics(
    true_values: np.ndarray, 
    pred_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression evaluation metrics.

    Computes various regression metrics including error measures, correlation
    measures, and relative error metrics. Handles edge cases gracefully.

    Args:
        true_values: Ground truth values as numpy array
        pred_values: Predicted values as numpy array

    Returns:
        Dictionary with regression metrics:
        - rmse: Root Mean Square Error
        - mae: Mean Absolute Error  
        - r2: R-squared coefficient (clipped to [0, 1])
        - mre: Mean Relative Error
        - mape: Mean Absolute Percentage Error
        - medae: Median Absolute Error
    """
    # Validate inputs for NaN/Inf values
    if true_values.size == 0 or pred_values.size == 0:
        raise ValueError("Input arrays must not be empty")
    
    if np.isnan(true_values).any() or np.isinf(true_values).any():
        raise ValueError("true_values contains NaN or Inf values")
    
    if np.isnan(pred_values).any() or np.isinf(pred_values).any():
        raise ValueError("pred_values contains NaN or Inf values")
    
    try:
        # Core error metrics
        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, pred_values)

        # R-squared with bounds checking
        r2 = r2_score(true_values, pred_values)
        r2 = max(0.0, float(r2))  # Clip negative R² to 0

        # Relative error metrics with epsilon protection
        denominator_mre = np.abs(true_values) + EPSILON
        mre = np.mean(np.abs((true_values - pred_values) / denominator_mre))

        # Mean Absolute Percentage Error
        denominator_mape = np.abs(true_values) + EPSILON
        mape = np.mean(np.abs((true_values - pred_values) / denominator_mape) * 100)

        # Median Absolute Error
        medae = np.median(np.abs(true_values - pred_values))

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mre": float(mre),
            "mape": float(mape),
            "medae": float(medae),
        }

        logger.debug(f"Calculated regression metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}")
        # Return safe fallback values
        return {
            "rmse": float('inf'),
            "mae": float('inf'), 
            "r2": 0.0,
            "mre": float('inf'),
            "mape": float('inf'),
            "medae": float('inf'),
        }


def calculate_classification_metrics(
    true_values: np.ndarray, 
    pred_values: np.ndarray, 
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, float]:
    """
    Calculate comprehensive classification evaluation metrics.

    Handles both binary and multiclass classification with automatic detection
    of the classification type. Supports probability inputs and direct label inputs.

    Args:
        true_values: Ground truth class labels or one-hot encoded array
        pred_values: Predicted probabilities or class labels
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary with classification metrics:
        - accuracy: Overall accuracy
        - precision: Precision score (macro-averaged for multiclass)
        - recall: Recall score (macro-averaged for multiclass)  
        - f1: F1 score (macro-averaged for multiclass)
        - auc: AUC-ROC score (when applicable)
    """
    # Input validation
    _validate_classification_inputs(true_values, pred_values)

    try:
        # Standardize inputs to 1D class labels
        true_labels_1d = _standardize_true_labels(true_values)
        
        # Analyze target characteristics
        unique_classes = np.unique(true_labels_1d)
        num_classes = len(unique_classes)
        
        # Handle edge case: single class or empty
        if num_classes <= 1:
            return _handle_single_class_metrics(
                true_labels_1d, pred_values, threshold, num_classes
            )

        # Determine classification type and process predictions
        target_type = "binary" if num_classes == 2 else "multiclass"
        pred_labels_1d, probas_for_auc = _process_predictions(
            pred_values, target_type, threshold, unique_classes
        )

        # Validate processed arrays
        _validate_array_shapes(true_labels_1d, pred_labels_1d)

        # Calculate core metrics
        metrics = _calculate_core_classification_metrics(
            true_labels_1d, pred_labels_1d, target_type
        )

        # Add AUC if probabilities are available
        _add_auc_metrics(
            metrics, true_labels_1d, probas_for_auc, target_type, num_classes
        )

        logger.debug(f"Calculated {target_type} classification metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating classification metrics: {e}")
        return _get_fallback_classification_metrics()


def _validate_classification_inputs(
    true_values: np.ndarray, 
    pred_values: np.ndarray
) -> None:
    """
    Validate inputs for classification metrics calculation.

    Args:
        true_values: Ground truth array
        pred_values: Predicted values array
    """
    if true_values.size == 0 or pred_values.size == 0:
        raise ValueError("Input arrays must not be empty")
    
    if np.isnan(true_values).any() or np.isinf(true_values).any():
        raise ValueError("true_values contains NaN or Inf values")
    
    if np.isnan(pred_values).any() or np.isinf(pred_values).any():
        raise ValueError("pred_values contains NaN or Inf values")


def _standardize_true_labels(true_values: np.ndarray) -> np.ndarray:
    """
    Convert true values to standardized 1D integer labels.

    Args:
        true_values: Ground truth array (1D labels or 2D one-hot)
    """
    if true_values.ndim == 2:
        if true_values.shape[1] == 1:
            # Shape (N, 1) -> flatten
            return true_values.flatten().astype(int)
        else:
            # One-hot encoded -> argmax
            return np.argmax(true_values, axis=1)
    else:
        # Already 1D
        return true_values.astype(int)


def _handle_single_class_metrics(
    true_labels_1d: np.ndarray,
    pred_values: np.ndarray, 
    threshold: float,
    num_classes: int
) -> Dict[str, float]:
    """
    Handle edge case where only one class is present in true labels.

    Args:
        true_labels_1d: 1D true labels array
        pred_values: Prediction values
        threshold: Classification threshold
        num_classes: Number of unique classes

    Returns:
        Dictionary with edge case metrics
    """
    metrics = {
        "precision": 0.0,
        "recall": 0.0, 
        "f1": 0.0,
        "auc": 0.5 if num_classes == 1 else 0.0,
    }

    # Try to calculate accuracy if possible
    try:
        pred_labels = _convert_predictions_to_labels(
            pred_values, threshold, "single_class"
        )
        if pred_labels.shape == true_labels_1d.shape:
            metrics["accuracy"] = accuracy_score(true_labels_1d, pred_labels)
        else:
            metrics["accuracy"] = 0.0
    except Exception:
        metrics["accuracy"] = 0.0

    return metrics


def _process_predictions(
    pred_values: np.ndarray,
    target_type: str,
    threshold: float,
    unique_classes: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process prediction values into labels and extract probabilities for AUC.

    Args:
        pred_values: Raw prediction values
        target_type: 'binary' or 'multiclass'
        threshold: Classification threshold
        unique_classes: Array of unique class labels

    Returns:
        Tuple of (predicted_labels, probabilities_for_auc)
    """
    if target_type == "binary":
        return _process_binary_predictions(pred_values, threshold, unique_classes)
    else:
        return _process_multiclass_predictions(pred_values, len(unique_classes))


def _process_binary_predictions(
    pred_values: np.ndarray,
    threshold: float,
    unique_classes: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process binary classification predictions.

    Args:
        pred_values: Prediction values
        threshold: Classification threshold  
        unique_classes: Unique class labels

    Returns:
        Tuple of (predicted_labels, probabilities_for_auc)
    """
    probas_for_auc = None

    if pred_values.ndim == 1:
        # 1D predictions - could be probabilities or labels
        if _is_probability_array(pred_values):
            pred_labels = (pred_values > threshold).astype(int)
            probas_for_auc = pred_values
        else:
            pred_labels = pred_values.astype(int)
            
    elif pred_values.ndim == 2 and pred_values.shape[1] == 1:
        # Shape (N, 1) - flatten and process
        flat_preds = pred_values.flatten()
        if _is_probability_array(flat_preds):
            pred_labels = (flat_preds > threshold).astype(int)
            probas_for_auc = flat_preds
        else:
            pred_labels = flat_preds.astype(int)
            
    elif pred_values.ndim == 2 and pred_values.shape[1] == 2:
        # Shape (N, 2) - binary class probabilities
        pred_labels = np.argmax(pred_values, axis=1)
        # Use probability of positive class (assume class 1 if present)
        if 1 in unique_classes:
            probas_for_auc = pred_values[:, 1]
        else:
            probas_for_auc = pred_values[:, 1]  # Use second column as default
    else:
        # Fallback: treat as direct labels
        pred_labels = pred_values.astype(int).flatten()

    return pred_labels, probas_for_auc


def _process_multiclass_predictions(
    pred_values: np.ndarray,
    num_classes: int
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process multiclass classification predictions.

    Args:
        pred_values: Prediction values
        num_classes: Number of classes

    Returns:
        Tuple of (predicted_labels, probabilities_for_auc)
    """
    if pred_values.ndim == 2 and pred_values.shape[1] == num_classes:
        # Probability matrix (N, C)
        pred_labels = np.argmax(pred_values, axis=1)
        probas_for_auc = pred_values
    elif pred_values.ndim == 1:
        # Direct class labels
        pred_labels = pred_values.astype(int)
        probas_for_auc = None
    else:
        raise ValueError(
            f"Unsupported pred_values shape {pred_values.shape} "
            f"for multiclass with {num_classes} classes"
        )

    return pred_labels, probas_for_auc


def _is_probability_array(array: np.ndarray) -> bool:
    """
    Heuristic to determine if array contains probabilities.

    Args:
        array: Array to check

    Returns:
        True if array likely contains probabilities
    """
    return bool(
        np.issubdtype(array.dtype, np.floating) and
        np.all(array >= 0) and 
        np.all(array <= 1) and
        not np.all(np.isin(np.unique(array), [0, 1]))
    )


def _convert_predictions_to_labels(
    pred_values: np.ndarray,
    threshold: float,
    context: str
) -> np.ndarray:
    """
    Convert predictions to class labels based on context.

    Args:
        pred_values: Prediction values
        threshold: Classification threshold
        context: Context for conversion logic

    Returns:
        Array of predicted class labels
    """
    if pred_values.ndim == 1:
        if _is_probability_array(pred_values):
            return (pred_values > threshold).astype(int)
        else:
            return pred_values.astype(int)
    elif pred_values.ndim == 2:
        if pred_values.shape[1] == 1:
            flat_preds = pred_values.flatten()
            if _is_probability_array(flat_preds):
                return (flat_preds > threshold).astype(int)
            else:
                return flat_preds.astype(int)
        else:
            return np.argmax(pred_values, axis=1)
    else:
        return pred_values.astype(int).flatten()


def _calculate_core_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    target_type: str
) -> Dict[str, float]:
    """
    Calculate core classification metrics (accuracy, precision, recall, F1).

    Args:
        true_labels: True class labels
        pred_labels: Predicted class labels
        target_type: 'binary' or 'multiclass'

    Returns:
        Dictionary with core metrics
    """
    # Determine averaging strategy
    average_method = "binary" if target_type == "binary" else "macro"

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(
            true_labels, pred_labels, average=average_method, zero_division="warn"
        ),
        "recall": recall_score(
            true_labels, pred_labels, average=average_method, zero_division="warn"
        ),
        "f1": f1_score(
            true_labels, pred_labels, average=average_method, zero_division="warn"
        ),
    }

    return metrics


def _add_auc_metrics(
    metrics: Dict[str, float],
    true_labels: np.ndarray,
    probas_for_auc: Optional[np.ndarray],
    target_type: str,
    num_classes: int
) -> None:
    """
    Add AUC metrics to the metrics dictionary if probabilities are available.

    Args:
        metrics: Dictionary to add AUC metrics to
        true_labels: True class labels
        probas_for_auc: Probabilities for AUC calculation
        target_type: 'binary' or 'multiclass'
        num_classes: Number of classes
    """
    if probas_for_auc is None:
        return

    try:
        if target_type == "binary" and probas_for_auc.ndim == 1:
            if len(np.unique(true_labels)) >= 2:
                auc_score = roc_auc_score(true_labels, probas_for_auc)
                metrics["auc"] = 0.5 if np.isnan(auc_score) else float(auc_score)
            else:
                metrics["auc"] = 0.5
                
        elif target_type == "multiclass" and probas_for_auc.ndim == 2:
            if num_classes > 2 and probas_for_auc.shape[1] == num_classes:
                auc_score = roc_auc_score(
                    true_labels, probas_for_auc, 
                    multi_class="ovr", average="macro"
                )
                metrics["auc_macro_ovr"] = 0.5 if np.isnan(auc_score) else float(auc_score)
            else:
                metrics["auc_macro_ovr"] = 0.5
                
    except Exception as e:
        logger.debug(f"Could not calculate AUC: {e}")
        if target_type == "binary":
            metrics["auc"] = 0.5
        else:
            metrics["auc_macro_ovr"] = 0.5


def _get_fallback_classification_metrics() -> Dict[str, float]:
    """
    Get fallback metrics for error cases.

    Returns:
        Dictionary with safe fallback metric values
    """
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": 0.5,
    }


def calculate_node_level_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    node_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for node-level predictions with optional masking.

    This function is specialized for graph neural networks where predictions
    are made at the node level. It supports masking to include/exclude
    specific nodes from the evaluation.

    Args:
        true_values: Ground truth values for nodes
        pred_values: Predicted values for nodes
        task_type: Type of task ('regression' or 'classification')
        node_mask: Optional mask to filter nodes (1 for include, 0 for exclude)

    Returns:
        Dictionary with node-level metrics
    """
    # Convert inputs to numpy arrays
    true_array = _convert_to_numpy(true_values, "true_values")
    pred_array = _convert_to_numpy(pred_values, "pred_values")

    # Apply mask if provided
    if node_mask is not None:
        mask_array = _convert_to_numpy(node_mask, "node_mask").astype(bool)
        
        # Validate mask size
        if mask_array.size != true_array.size:
            raise ValueError(
                f"Node mask size {mask_array.size} does not match "
                f"true_values size {true_array.size}"
            )

        # Apply mask to filter nodes
        true_array = true_array[mask_array]
        pred_array = pred_array[mask_array]
        
        logger.debug(
            f"Applied node mask: {mask_array.sum()}/{mask_array.size} nodes selected"
        )

    # Calculate metrics using the general function
    return calculate_metrics(true_array, pred_array, task_type)


def calculate_graph_level_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    graph_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for graph-level predictions with optional masking.

    This function is specialized for graph neural networks where predictions
    are made at the graph level. It supports masking to include/exclude
    specific graphs from the evaluation.

    Args:
        true_values: Ground truth values for graphs
        pred_values: Predicted values for graphs
        task_type: Type of task ('regression' or 'classification')
        graph_mask: Optional mask to filter graphs (1 for include, 0 for exclude)

    Returns:
        Dictionary with graph-level metrics
    """
    # Convert inputs to numpy arrays
    true_array = _convert_to_numpy(true_values, "true_values")
    pred_array = _convert_to_numpy(pred_values, "pred_values")

    # Apply mask if provided
    if graph_mask is not None:
        mask_array = _convert_to_numpy(graph_mask, "graph_mask").astype(bool)
        
        # Validate mask size
        if mask_array.size != true_array.size:
            raise ValueError(
                f"Graph mask size {mask_array.size} does not match "
                f"true_values size {true_array.size}"
            )

        # Apply mask to filter graphs
        true_array = true_array[mask_array]
        pred_array = pred_array[mask_array]
        
        logger.debug(
            f"Applied graph mask: {mask_array.sum()}/{mask_array.size} graphs selected"
        )

    # Calculate metrics using the general function
    return calculate_metrics(true_array, pred_array, task_type)


def visualize_predictions(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    title: str = "Predictions vs Ground Truth",
    save_path: Optional[str] = None,
    show_metrics: bool = True,
) -> Figure:
    """
    Create visualization comparing model predictions to ground truth.

    This function generates appropriate plots based on the task type:
    - Regression: Scatter plot with identity line and error metrics
    - Classification: Confusion matrix with classification metrics

    Args:
        true_values: Ground truth values
        pred_values: Predicted values
        task_type: Type of task ('regression' or 'classification')
        title: Plot title text
        save_path: Optional path to save the plot image
        show_metrics: Whether to display metrics on the plot

    Returns:
        Matplotlib Figure object containing the visualization
    """
    # Convert inputs to numpy arrays
    true_array = _convert_to_numpy(true_values, "true_values")
    pred_array = _convert_to_numpy(pred_values, "pred_values")

    # Create figure with appropriate size
    fig = plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    try:
        if task_type == "regression":
            _create_regression_plot(true_array, pred_array, show_metrics)
        elif task_type == "classification":
            _create_classification_plot(true_array, pred_array, show_metrics)
        else:
            raise ValueError(f"Unsupported task_type for visualization: {task_type}")

        # Apply common formatting
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            _save_plot(fig, save_path)

        logger.debug(f"Created {task_type} visualization with title: {title}")
        return fig

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        plt.close(fig)
        raise


def _create_regression_plot(
    true_array: np.ndarray,
    pred_array: np.ndarray,
    show_metrics: bool
) -> None:
    """
    Create scatter plot for regression predictions.

    Args:
        true_array: Ground truth values
        pred_array: Predicted values
        show_metrics: Whether to show metrics
    """
    # Create scatter plot
    plt.scatter(true_array, pred_array, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    min_val = min(np.min(true_array), np.min(pred_array))
    max_val = max(np.max(true_array), np.max(pred_array))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    # Set labels
    plt.xlabel("Ground Truth", fontsize=12)
    plt.ylabel("Predictions", fontsize=12)
    plt.legend()

    # Add metrics if requested
    if show_metrics:
        try:
            metrics = calculate_regression_metrics(true_array, pred_array)
            metric_text = (
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"MAE: {metrics['mae']:.4f}\n"
                f"R²: {metrics['r2']:.4f}"
            )
            
            plt.text(
                0.05, 0.95, metric_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10
            )
        except Exception as e:
            logger.warning(f"Could not add metrics to regression plot: {e}")


def _create_classification_plot(
    true_array: np.ndarray,
    pred_array: np.ndarray,
    show_metrics: bool
) -> None:
    """
    Create confusion matrix plot for classification predictions.

    Args:
        true_array: Ground truth labels
        pred_array: Predicted values (probabilities or labels)
        show_metrics: Whether to show metrics
    """
    # Standardize true labels to 1D integer format (handle one-hot encoding)
    true_labels = _standardize_true_labels(true_array)
    
    # Convert predictions to labels if they are probabilities
    if (pred_array.ndim == 1 and 
        np.issubdtype(pred_array.dtype, np.floating) and
        np.all(pred_array >= 0) and np.all(pred_array <= 1) and
        pred_array.max() <= 1.0):
        pred_labels = (pred_array > DEFAULT_THRESHOLD).astype(int)
    elif pred_array.ndim == 2:
        pred_labels = np.argmax(pred_array, axis=1)
    else:
        pred_labels = pred_array.astype(int)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore
    plt.colorbar(im)

    # Determine class labels
    num_classes = cm.shape[0]
    if num_classes == 2:
        class_names = ["Negative", "Positive"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Set ticks and labels
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to cells
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12, fontweight='bold'
            )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Add metrics if requested
    if show_metrics:
        try:
            metrics = calculate_classification_metrics(true_array, pred_array)
            metric_text = (
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"F1: {metrics['f1']:.4f}"
            )
            
            if "auc" in metrics:
                metric_text += f"\nAUC: {metrics['auc']:.4f}"

            plt.text(
                1.05, 0.5, metric_text,
                transform=plt.gca().transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10
            )
        except Exception as e:
            logger.warning(f"Could not add metrics to classification plot: {e}")


def _save_plot(fig: Figure, save_path: str) -> None:
    """
    Save plot to file with proper directory creation.

    Args:
        fig: Matplotlib figure to save
        save_path: Path to save the plot

    Raises:
        OSError: If file cannot be saved
    """
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save with high quality
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor='white')
        logger.info(f"Saved plot to: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}")
        raise OSError(f"Could not save plot: {e}")
