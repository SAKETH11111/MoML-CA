"""
Metrics module for evaluating models.

This module provides functions for calculating performance metrics
and visualizing predictions for molecular graph neural networks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)


def calculate_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = 'regression'
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
        raise ValueError(f"Shape mismatch: true_values shape {true_values.shape} != pred_values shape {pred_values.shape}")
    
    # Calculate metrics based on task type
    if task_type == 'regression':
        return calculate_regression_metrics(true_values, pred_values)
    elif task_type == 'classification':
        return calculate_classification_metrics(true_values, pred_values)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def calculate_regression_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray
) -> Dict[str, float]:
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
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mre': mre,
        'mape': mape,
        'medae': medae
    }


def calculate_classification_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        true_values: Ground truth values
        pred_values: Predicted values (probabilities for binary classification)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with classification metrics
    """
    # Handle binary classification case
    if len(true_values.shape) == 1 or true_values.shape[1] == 1:
        # Convert probability predictions to binary using threshold
        if pred_values.max() <= 1.0 and pred_values.min() >= 0.0:
            # Predictions are probabilities
            binary_preds = (pred_values > threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(true_values, binary_preds)
            precision = precision_score(true_values, binary_preds, zero_division=0)
            recall = recall_score(true_values, binary_preds, zero_division=0)
            f1 = f1_score(true_values, binary_preds, zero_division=0)
            
            # ROC AUC is calculated on probabilities
            try:
                auc = roc_auc_score(true_values, pred_values)
            except Exception:
                auc = 0.5  # Default value when ROC AUC calculation fails
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        else:
            # Predictions are already binary
            accuracy = accuracy_score(true_values, pred_values)
            precision = precision_score(true_values, pred_values, zero_division=0)
            recall = recall_score(true_values, pred_values, zero_division=0)
            f1 = f1_score(true_values, pred_values, zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    # Handle multi-class classification
    else:
        # Convert to class indices if predictions are probabilities
        if pred_values.shape == true_values.shape:
            pred_classes = np.argmax(pred_values, axis=1)
            true_classes = np.argmax(true_values, axis=1)
        else:
            pred_classes = pred_values
            true_classes = true_values
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        
        # Multi-class metrics with macro averaging
        precision = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
        recall = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
        f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def calculate_node_level_metrics(
    true_values: Union[torch.Tensor, np.ndarray, List],
    pred_values: Union[torch.Tensor, np.ndarray, List],
    task_type: str = 'regression',
    node_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None
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
    task_type: str = 'regression',
    graph_mask: Optional[Union[torch.Tensor, np.ndarray, List]] = None
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
    task_type: str = 'regression',
    title: str = 'Predictions vs Ground Truth',
    save_path: Optional[str] = None,
    show_metrics: bool = True
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
    if task_type == 'regression':
        # Scatter plot with identity line
        plt.scatter(true_values, pred_values, alpha=0.5)
        
        # Add identity line
        min_val = min(np.min(true_values), np.min(pred_values))
        max_val = max(np.max(true_values), np.max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        
        # Calculate and display metrics
        if show_metrics:
            metrics = calculate_regression_metrics(true_values, pred_values)
            
            metric_text = f"RMSE: {metrics['rmse']:.4f}\n" \
                         f"MAE: {metrics['mae']:.4f}\n" \
                         f"R²: {metrics['r2']:.4f}"
            
            plt.annotate(
                metric_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8)
            )
    
    elif task_type == 'classification':
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
            
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            
            classes = ['Negative', 'Positive']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations to the confusion matrix cells
            threshold = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > threshold else "black")
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Calculate and display metrics
            if show_metrics:
                metrics = calculate_classification_metrics(true_values, pred_values)
                
                metric_text = f"Accuracy: {metrics['accuracy']:.4f}\n" \
                             f"Precision: {metrics['precision']:.4f}\n" \
                             f"Recall: {metrics['recall']:.4f}\n" \
                             f"F1: {metrics['f1']:.4f}"
                
                if 'auc' in metrics:
                    metric_text += f"\nAUC: {metrics['auc']:.4f}"
                
                plt.annotate(
                    metric_text,
                    xy=(1.05, 0.5),
                    xycoords='axes fraction',
                    ha='left',
                    va='center',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8)
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
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > threshold else "black")
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Calculate and display metrics
            if show_metrics:
                metrics = calculate_classification_metrics(true_classes, pred_classes)
                
                metric_text = f"Accuracy: {metrics['accuracy']:.4f}\n" \
                             f"Precision: {metrics['precision']:.4f}\n" \
                             f"Recall: {metrics['recall']:.4f}\n" \
                             f"F1: {metrics['f1']:.4f}"
                
                plt.annotate(
                    metric_text,
                    xy=(1.05, 0.5),
                    xycoords='axes fraction',
                    ha='left',
                    va='center',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8)
                )
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 