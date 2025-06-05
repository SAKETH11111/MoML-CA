"""
Metrics module for evaluating models.

This module provides functions for calculating performance metrics
and visualizing predictions for molecular graph neural networks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union
import torch
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.

    Args:
        true_values: Ground truth values
        pred_values: Predicted values
        task_type: Type of task ('regression' or 'classification')

    Returns:
        Dictionary with calculated metrics
    """
    # Convert inputs to numpy arrays if they are tensors or lists
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
    elif isinstance(true_values, list):
        true_values = np.array(true_values)

    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    elif isinstance(pred_values, list):
        pred_values = np.array(pred_values)

    # Ensure arrays have the same shape
    if true_values.shape != pred_values.shape:
        raise ValueError(
            f"Shape mismatch: true_values shape {true_values.shape} != pred_values shape {pred_values.shape}"
        )

    # Calculate metrics based on task type
    if task_type == "regression":
        return calculate_regression_metrics(true_values, pred_values)
    elif task_type == "classification":
        return calculate_classification_metrics(true_values, pred_values)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def calculate_regression_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        true_values: Ground truth values
        pred_values: Predicted values

    Returns:
        Dictionary with regression metrics
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    mae = mean_absolute_error(true_values, pred_values)

    # R² can sometimes be negative, so we handle that case
    r2 = r2_score(true_values, pred_values)
    if r2 < 0:
        r2 = 0.0

    # Mean relative error
    mre = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-8)))

    # Mean absolute percentage error
    mape = np.mean(np.abs((true_values - pred_values) / (np.abs(true_values) + 1e-8)) * 100)

    # Median absolute error
    medae = np.median(np.abs(true_values - pred_values))

    return {"rmse": rmse, "mae": mae, "r2": r2, "mre": mre, "mape": mape, "medae": medae}


def calculate_classification_metrics(
    true_values: np.ndarray, pred_values: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics. Handles binary and multiclass tasks.

    Args:
        true_values: Ground truth values. Expected as 1D array of class labels
                     (e.g., [0, 1, 0] for binary; [0, 1, 2, 1] for multiclass)
                     or 2D one-hot encoded array for multiclass.
        pred_values: Predicted values.
                     For binary: 1D array of probabilities or direct labels.
                     For multiclass: 2D array of probabilities per class (n_samples, n_classes)
                                     or 1D array of direct class labels.
        threshold: Threshold for converting probabilities to binary labels.

    Returns:
        Dictionary with classification metrics.
    """
    metrics: Dict[str, float] = {}

    # Validate inputs for common issues before proceeding
    if true_values.size == 0 or pred_values.size == 0:
        raise ValueError("Input arrays true_values and pred_values must not be empty.")
    if np.isnan(true_values).any() or np.isinf(true_values).any():
        raise ValueError("true_values contains NaN or Inf values.")
    if np.isnan(pred_values).any() or np.isinf(pred_values).any():
        raise ValueError("pred_values contains NaN or Inf values.")

    # 1. Standardize true_values to 1D class labels (_true_labels_1d)
    _true_labels_1d: np.ndarray
    if true_values.ndim == 2:
        if true_values.shape[1] == 1:  # e.g., [[0], [1]]
            _true_labels_1d = true_values.flatten().astype(int)
        else:  # Assume one-hot encoded, e.g., [[1,0,0], [0,1,0]]
            _true_labels_1d = np.argmax(true_values, axis=1)
    else:  # Already 1D
        _true_labels_1d = true_values.astype(int)

    # 2. Determine target type and appropriate averaging method based on number of unique true classes
    _unique_true_classes = np.unique(_true_labels_1d)
    _num_true_classes = len(_unique_true_classes)
    _target_type: str  # To be explicitly set
    _sklearn_avg_method: str

    if _num_true_classes > 2:
        _target_type = "multiclass"
        _sklearn_avg_method = "macro"
    elif _num_true_classes == 2:
        _target_type = "binary"
        _sklearn_avg_method = "binary"
    elif _num_true_classes <= 1:
        # Handle single class case (metrics are defaulted)
        # This logic is to get some accuracy value if possible, other metrics are ill-defined.
        _pred_labels_1d_for_accuracy_only: np.ndarray
        if pred_values.ndim == 1:  # Check if pred_values are probabilities or labels
            is_proba_heuristic_single_class = np.issubdtype(pred_values.dtype, np.floating) and not np.all(
                np.isin(np.unique(pred_values), [0, 1])
            )
            _pred_labels_1d_for_accuracy_only = (
                (pred_values > threshold).astype(int) if is_proba_heuristic_single_class else pred_values.astype(int)
            )
        elif pred_values.ndim == 2 and pred_values.shape[1] == 1:  # (N,1) shape
            flat_preds_single_class = pred_values.flatten()
            is_proba_heuristic_single_class = np.issubdtype(flat_preds_single_class.dtype, np.floating) and not np.all(
                np.isin(np.unique(flat_preds_single_class), [0, 1])
            )
            _pred_labels_1d_for_accuracy_only = (
                (flat_preds_single_class > threshold).astype(int)
                if is_proba_heuristic_single_class
                else flat_preds_single_class.astype(int)
            )
        elif pred_values.ndim == 2:  # Assumed (N, C) probabilities for multiclass, take argmax
            _pred_labels_1d_for_accuracy_only = np.argmax(pred_values, axis=1)
        else:  # Fallback for unexpected shapes
            _pred_labels_1d_for_accuracy_only = pred_values.astype(int).flatten()

        if _pred_labels_1d_for_accuracy_only.shape == _true_labels_1d.shape:
            metrics["accuracy"] = accuracy_score(_true_labels_1d, _pred_labels_1d_for_accuracy_only)
        else:
            metrics["accuracy"] = 0.0  # Shape mismatch, cannot compute accuracy
        # For single class in true labels, other metrics are typically 0 or undefined.
        # AUC is conventionally 0.5 if only one class present. If zero classes (empty true_labels), then 0.0.
        metrics.update({"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5 if _num_true_classes == 1 else 0.0})
        return metrics
    else:  # Should ideally not be reached if _num_true_classes is always non-negative.
        # This case implies _num_true_classes is 0, which should have been caught by the empty array check earlier.
        raise ValueError(f"Internal logic error: Unexpected number of true classes: {_num_true_classes}")

    # 3. Determine predicted class labels (_pred_labels_1d)
    #    and identify if binary probabilities were provided for AUC.
    _pred_labels_1d: np.ndarray
    _binary_probas_for_auc: Optional[np.ndarray] = None
    _multiclass_probas_for_auc: Optional[np.ndarray] = None

    if _target_type == "binary":
        if pred_values.ndim == 1:
            is_proba_heuristic = np.issubdtype(pred_values.dtype, np.floating) and not np.all(
                np.isin(np.unique(pred_values), [0, 1])
            )
            if is_proba_heuristic and np.all(pred_values >= 0) and np.all(pred_values <= 1):
                _pred_labels_1d = (pred_values > threshold).astype(int)
                _binary_probas_for_auc = pred_values
            else:
                _pred_labels_1d = pred_values.astype(int)
        elif pred_values.ndim == 2 and pred_values.shape[1] == 1:  # (N,1)
            flat_preds = pred_values.flatten()
            is_proba_heuristic = np.issubdtype(flat_preds.dtype, np.floating) and not np.all(
                np.isin(np.unique(flat_preds), [0, 1])
            )
            if is_proba_heuristic and np.all(flat_preds >= 0) and np.all(flat_preds <= 1):
                _pred_labels_1d = (flat_preds > threshold).astype(int)
                _binary_probas_for_auc = flat_preds
            else:
                _pred_labels_1d = flat_preds.astype(int)
        elif pred_values.ndim == 2 and pred_values.shape[1] == 2:  # (N,2) probabilities
            _pred_labels_1d = np.argmax(pred_values, axis=1)
            # Determine positive class index for AUC. Assume class '1' is positive if present.
            positive_class_idx = 1
            if _num_true_classes == 2:  # Ensure we have two unique classes
                sorted_unique_classes = np.sort(_unique_true_classes)
                # If labels are not 0 and 1, map to 0 and 1 for roc_auc_score or pick one as positive
                # For simplicity, if 1 is present, use its probas. Otherwise, use probas of the larger label.
                if 1 in sorted_unique_classes:
                    # Find which column in pred_values corresponds to class 1
                    # This assumes pred_values columns are ordered like sorted unique classes,
                    # or that the second column is for the positive class if labels are 0,1.
                    # A common convention is [prob_class_0, prob_class_1].
                    _binary_probas_for_auc = pred_values[:, 1]  # Default to second column for class 1
                else:  # e.g. classes are -1, 1 or other pairs. Use prob of the class considered positive.
                    # This part might need more robust handling if classes are not {0,1}
                    # For now, if not {0,1}, AUC might be tricky without knowing positive label.
                    # Defaulting to prob of the second class if not 0,1.
                    _binary_probas_for_auc = pred_values[:, 1]

        else:  # Assumed to be direct binary labels if shape is unexpected for probabilities
            _pred_labels_1d = pred_values.astype(int).flatten()

    elif _target_type == "multiclass":
        if pred_values.ndim == 2 and pred_values.shape[1] == _num_true_classes:  # Probas (N, C)
            _pred_labels_1d = np.argmax(pred_values, axis=1)
            _multiclass_probas_for_auc = pred_values
        elif pred_values.ndim == 1:  # Direct class labels (N,)
            _pred_labels_1d = pred_values.astype(int)
        else:
            raise ValueError(
                f"Unsupported pred_values shape {pred_values.shape} for multiclass target "
                f"with {_num_true_classes} classes."
            )
    else:  # Should be caught by the initial _target_type check
        _pred_labels_1d = pred_values.astype(int).flatten()  # Fallback

    # Ensure _pred_labels_1d is 1D
    if _pred_labels_1d.ndim > 1 and _pred_labels_1d.shape[1] == 1:
        _pred_labels_1d = _pred_labels_1d.flatten()

    if _true_labels_1d.shape != _pred_labels_1d.shape:
        raise ValueError(
            f"Shape mismatch after processing: "
            f"true_labels_1d shape {_true_labels_1d.shape} != "
            f"pred_labels_1d shape {_pred_labels_1d.shape}"
        )

    # 4. Calculate metrics
    y_true_sklearn_type = type_of_target(_true_labels_1d)  # 'binary', 'multiclass', etc.

    # Determine the effective average method for precision, recall, f1
    if y_true_sklearn_type == "binary":
        effective_average_for_scores = "binary"
    elif y_true_sklearn_type == "multiclass":
        # For multiclass, 'macro' computes the metric independently for each class and then takes the average.
        # 'weighted' accounts for class imbalance. 'macro' is a common default.
        effective_average_for_scores = "macro"
    else:
        # This case should ideally not be reached if inputs are validated classification labels
        # and the single-class case (len(unique_labels) <= 1) is handled earlier.
        logger.warning(
            f"Unexpected y_true_sklearn_type '{y_true_sklearn_type}' encountered. "
            f"Defaulting 'average' parameter for scores to 'macro'. "
            f"True labels: {np.unique(_true_labels_1d)}"
        )
        effective_average_for_scores = "macro"

    metrics["accuracy"] = accuracy_score(_true_labels_1d, _pred_labels_1d)
    metrics["precision"] = precision_score(
        _true_labels_1d, _pred_labels_1d, average=effective_average_for_scores, zero_division=0
    )
    metrics["recall"] = recall_score(
        _true_labels_1d, _pred_labels_1d, average=effective_average_for_scores, zero_division=0
    )
    metrics["f1"] = f1_score(_true_labels_1d, _pred_labels_1d, average=effective_average_for_scores, zero_division=0)

    # 5. AUC Calculation
    if _target_type == "binary" and _binary_probas_for_auc is not None:
        # Ensure _binary_probas_for_auc is 1D
        if _binary_probas_for_auc.ndim > 1:
            if _binary_probas_for_auc.shape[1] == 1:
                _binary_probas_for_auc = _binary_probas_for_auc.flatten()
            else:  # Should not happen if logic above is correct for binary probas
                _binary_probas_for_auc = None

        if _binary_probas_for_auc is not None and _binary_probas_for_auc.shape == _true_labels_1d.shape:
            if len(np.unique(_true_labels_1d)) < 2:
                metrics["auc"] = 0.5
            else:
                try:
                    auc_val = roc_auc_score(_true_labels_1d, _binary_probas_for_auc)
                    metrics["auc"] = 0.5 if np.isnan(auc_val) else auc_val
                except ValueError:  # Catches "Only one class present in y_true" or other issues
                    metrics["auc"] = 0.5
                except Exception:  # General catch
                    metrics["auc"] = 0.5
        # else: Cannot calculate AUC if suitable probabilities are not found or shape mismatch

    elif _target_type == "multiclass" and _multiclass_probas_for_auc is not None:
        if _num_true_classes > 1 and _multiclass_probas_for_auc.shape == (_true_labels_1d.shape[0], _num_true_classes):
            try:
                auc_val = roc_auc_score(
                    _true_labels_1d,
                    _multiclass_probas_for_auc,
                    multi_class="ovr",  # or 'ovo'
                    average=(
                        _sklearn_avg_method if _sklearn_avg_method != "binary" else "macro"
                    ),  # Ensure valid average for roc_auc
                )
                metrics[f"auc_{_sklearn_avg_method}_ovr"] = 0.5 if np.isnan(auc_val) else auc_val
            except ValueError:  # e.g. "Only one class present in y_true"
                metrics[f"auc_{_sklearn_avg_method}_ovr"] = 0.5
            except Exception:
                metrics[f"auc_{_sklearn_avg_method}_ovr"] = 0.5
        # else: Cannot calculate multiclass AUC

    return metrics


def calculate_node_level_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    node_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for node-level predictions.

    Args:
        true_values: Ground truth values for nodes
        pred_values: Predicted values for nodes
        task_type: Type of task ('regression' or 'classification')
        node_mask: Optional mask to filter nodes (1 for nodes to include, 0 for nodes to exclude)

    Returns:
        Dictionary with node-level metrics
    """
    # Convert inputs to numpy arrays if they are tensors or lists
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
    elif isinstance(true_values, list):
        true_values = np.array(true_values)

    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    elif isinstance(pred_values, list):
        pred_values = np.array(pred_values)

    if node_mask is not None:
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.detach().cpu().numpy().astype(bool)
        elif isinstance(node_mask, list):
            node_mask = np.array(node_mask, dtype=bool)

        # Apply mask to include only certain nodes
        true_values = true_values[node_mask]
        pred_values = pred_values[node_mask]

    # Calculate metrics using the general function
    return calculate_metrics(true_values, pred_values, task_type)


def calculate_graph_level_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    graph_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for graph-level predictions.

    Args:
        true_values: Ground truth values for graphs
        pred_values: Predicted values for graphs
        task_type: Type of task ('regression' or 'classification')
        graph_mask: Optional mask to filter graphs (1 for graphs to include, 0 for graphs to exclude)

    Returns:
        Dictionary with graph-level metrics
    """
    # Convert inputs to numpy arrays if they are tensors or lists
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
    elif isinstance(true_values, list):
        true_values = np.array(true_values)

    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    elif isinstance(pred_values, list):
        pred_values = np.array(pred_values)

    if graph_mask is not None:
        if isinstance(graph_mask, torch.Tensor):
            graph_mask = graph_mask.detach().cpu().numpy().astype(bool)
        elif isinstance(graph_mask, list):
            graph_mask = np.array(graph_mask, dtype=bool)

        # Apply mask to include only certain graphs
        true_values = true_values[graph_mask]
        pred_values = pred_values[graph_mask]

    # Calculate metrics using the general function
    return calculate_metrics(true_values, pred_values, task_type)


def visualize_predictions(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = "regression",
    title: str = "Predictions vs Ground Truth",
    save_path: Optional[str] = None,
    show_metrics: bool = True,
) -> plt.Figure:
    """
    Visualize model predictions compared to ground truth.

    Args:
        true_values: Ground truth values
        pred_values: Predicted values
        task_type: Type of task ('regression' or 'classification')
        title: Plot title
        save_path: Optional path to save the plot
        show_metrics: Whether to show metrics on the plot

    Returns:
        Matplotlib figure
    """
    # Convert inputs to numpy arrays if they are tensors or lists
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
    elif isinstance(true_values, list):
        true_values = np.array(true_values)

    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    elif isinstance(pred_values, list):
        pred_values = np.array(pred_values)

    # Create figure
    fig = plt.figure(figsize=(10, 8))

    # Create different visualizations based on task type
    if task_type == "regression":
        # Scatter plot with identity line
        plt.scatter(true_values, pred_values, alpha=0.5)

        # Add identity line
        min_val = min(np.min(true_values), np.min(pred_values))
        max_val = max(np.max(true_values), np.max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")

        # Calculate and display metrics
        if show_metrics:
            metrics = calculate_regression_metrics(true_values, pred_values)

            metric_text = f"RMSE: {metrics['rmse']:.4f}\n" f"MAE: {metrics['mae']:.4f}\n" f"R²: {metrics['r2']:.4f}"

            plt.annotate(
                metric_text,
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )

    elif task_type == "classification":
        # For binary classification
        if len(true_values.shape) == 1 or true_values.shape[1] == 1:
            # Confusion matrix
            if pred_values.max() <= 1.0 and pred_values.min() >= 0.0:
                # Convert probabilities to binary predictions
                binary_preds = (pred_values > 0.5).astype(int)
                cm = confusion_matrix(true_values, binary_preds)
            else:
                # Predictions are already binary
                cm = confusion_matrix(true_values, pred_values)

            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()

            classes = ["Negative", "Positive"]
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            # Add text annotations to the confusion matrix cells
            threshold = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > threshold else "black",
                    )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # Calculate and display metrics
            if show_metrics:
                metrics = calculate_classification_metrics(true_values, pred_values)

                metric_text = (
                    f"Accuracy: {metrics['accuracy']:.4f}\n"
                    f"Precision: {metrics['precision']:.4f}\n"
                    f"Recall: {metrics['recall']:.4f}\n"
                    f"F1: {metrics['f1']:.4f}"
                )

                if "auc" in metrics:
                    metric_text += f"\nAUC: {metrics['auc']:.4f}"

                plt.annotate(
                    metric_text,
                    xy=(1.05, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                )

        # For multi-class classification
        else:
            # Convert to class indices if predictions are probabilities
            if pred_values.shape == true_values.shape:
                pred_classes = np.argmax(pred_values, axis=1)
                true_classes = np.argmax(true_values, axis=1)
            else:
                pred_classes = pred_values
                true_classes = true_values

            # Confusion matrix
            cm = confusion_matrix(true_classes, pred_classes)
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()

            # Adding labels
            num_classes = cm.shape[0]
            classes = [str(i) for i in range(num_classes)]
            tick_marks = np.arange(num_classes)
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            # Add text annotations to the confusion matrix cells
            threshold = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > threshold else "black",
                    )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # Calculate and display metrics
            if show_metrics:
                metrics = calculate_classification_metrics(true_classes, pred_classes)

                metric_text = (
                    f"Accuracy: {metrics['accuracy']:.4f}\n"
                    f"Precision: {metrics['precision']:.4f}\n"
                    f"Recall: {metrics['recall']:.4f}\n"
                    f"F1: {metrics['f1']:.4f}"
                )

                plt.annotate(
                    metric_text,
                    xy=(1.05, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                )

    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
