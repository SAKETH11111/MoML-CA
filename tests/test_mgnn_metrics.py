"""
Unit tests for the metrics module in moml.models.mgnn.evaluation.metrics.
"""
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import tempfile
import shutil

from moml.models.mgnn.evaluation.metrics import (
    calculate_metrics,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_node_level_metrics,
    calculate_graph_level_metrics,
    visualize_predictions
)

# Regression Test Data
@pytest.fixture
def regression_data():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_good = np.array([1.1, 1.9, 3.2, 3.8, 5.3])
    pred_bad_r2 = np.array([3.0, 1.0, 5.0, 2.0, 4.0]) # Should give negative R2
    return {"true": true, "pred_perfect": pred_perfect, "pred_good": pred_good, "pred_bad_r2": pred_bad_r2}

# Binary Classification Test Data
@pytest.fixture
def binary_clf_data():
    true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    pred_proba_good = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.95, 0.15, 0.25, 0.6])
    pred_binary_good = (pred_proba_good > 0.5).astype(int)
    pred_proba_all_one_class = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.9, 0.7, 0.8, 0.9, 0.7]) # All predict 1
    true_all_one_class = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # For AUC error test
    return {
        "true": true, 
        "pred_proba_good": pred_proba_good, 
        "pred_binary_good": pred_binary_good,
        "pred_proba_all_one_class": pred_proba_all_one_class,
        "true_all_one_class": true_all_one_class
    }

# Multi-class Classification Test Data
@pytest.fixture
def multiclass_clf_data():
    true_indices = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    # Perfect predictions as indices
    pred_indices_perfect = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    # Good predictions as probabilities (one-hot like)
    pred_proba_good = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]
    ])
    # True values as one-hot for testing proba inputs
    true_one_hot = np.zeros((8, 3))
    true_one_hot[np.arange(8), true_indices] = 1

    return {
        "true_indices": true_indices,
        "pred_indices_perfect": pred_indices_perfect,
        "pred_proba_good": pred_proba_good,
        "true_one_hot": true_one_hot
    }

class TestCalculateRegressionMetrics:
    def test_perfect_prediction(self, regression_data):
        metrics = calculate_regression_metrics(regression_data["true"], regression_data["pred_perfect"])
        assert pytest.approx(metrics['rmse']) == 0.0
        assert pytest.approx(metrics['mae']) == 0.0
        assert pytest.approx(metrics['r2']) == 1.0
        assert pytest.approx(metrics['mre']) == 0.0
        assert pytest.approx(metrics['mape']) == 0.0
        assert pytest.approx(metrics['medae']) == 0.0

    def test_good_prediction(self, regression_data):
        metrics = calculate_regression_metrics(regression_data["true"], regression_data["pred_good"])
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert 0.0 <= metrics['r2'] < 1.0

    def test_negative_r2_capping(self, regression_data):
        metrics = calculate_regression_metrics(regression_data["true"], regression_data["pred_bad_r2"])
        # R2 should be capped at 0.0 if sklearn.metrics.r2_score would return negative
        from sklearn.metrics import r2_score
        raw_r2 = r2_score(regression_data["true"], regression_data["pred_bad_r2"])
        if raw_r2 < 0:
            assert pytest.approx(metrics['r2']) == 0.0
        else: # Should not happen with pred_bad_r2
            assert pytest.approx(metrics['r2']) == raw_r2


    def test_mre_mape_with_zeros(self):
        true = np.array([0.0, 1.0, 2.0])
        pred = np.array([0.1, 1.1, 2.1]) # Epsilon should prevent division by zero
        metrics = calculate_regression_metrics(true, pred)
        assert np.isfinite(metrics['mre'])
        assert np.isfinite(metrics['mape'])

class TestCalculateClassificationMetrics:
    def test_binary_perfect_proba(self, binary_clf_data):
        # Make perfect proba predictions
        pred_proba_perfect = np.array([0.01, 0.99, 0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.99])
        metrics = calculate_classification_metrics(binary_clf_data["true"], pred_proba_perfect)
        assert pytest.approx(metrics['accuracy']) == 1.0
        assert pytest.approx(metrics['precision']) == 1.0
        assert pytest.approx(metrics['recall']) == 1.0
        assert pytest.approx(metrics['f1']) == 1.0
        assert pytest.approx(metrics['auc']) == 1.0

    def test_binary_good_proba(self, binary_clf_data):
        metrics = calculate_classification_metrics(binary_clf_data["true"], binary_clf_data["pred_proba_good"])
        assert 0.0 < metrics['accuracy'] < 1.0
        assert 0.0 <= metrics['auc'] <= 1.0

    def test_binary_direct_labels(self, binary_clf_data):
        metrics = calculate_classification_metrics(binary_clf_data["true"], binary_clf_data["pred_binary_good"])
        assert 'auc' not in metrics # AUC not calculated for direct binary labels
        assert 0.0 < metrics['accuracy'] < 1.0
    
    def test_binary_zero_division(self):
        true = np.array([0, 0, 0])
        pred_proba = np.array([0.1, 0.2, 0.3]) # All predicted negative
        metrics = calculate_classification_metrics(true, pred_proba)
        assert pytest.approx(metrics['precision']) == 0.0 # zero_division=0
        assert pytest.approx(metrics['recall']) == 0.0    # zero_division=0
        assert pytest.approx(metrics['f1']) == 0.0        # zero_division=0

        true_all_pos = np.array([1,1,1])
        pred_all_neg_bin = np.array([0,0,0])
        metrics_bin = calculate_classification_metrics(true_all_pos, pred_all_neg_bin)
        assert pytest.approx(metrics_bin['precision']) == 0.0
        assert pytest.approx(metrics_bin['recall']) == 0.0


    def test_binary_auc_error_case(self, binary_clf_data):
        # Test AUC when true labels are all of one class
        metrics = calculate_classification_metrics(binary_clf_data["true_all_one_class"], binary_clf_data["pred_proba_good"])
        assert pytest.approx(metrics['auc']) == 0.5 # Default on error

    def test_multiclass_perfect_indices(self, multiclass_clf_data):
        metrics = calculate_classification_metrics(
            multiclass_clf_data["true_indices"], 
            multiclass_clf_data["pred_indices_perfect"]
            # task_type="classification" # Not needed, inferred by calculate_classification_metrics
        )
        assert pytest.approx(metrics['accuracy']) == 1.0
        assert pytest.approx(metrics['precision']) == 1.0 # Macro average
        assert pytest.approx(metrics['recall']) == 1.0
        assert pytest.approx(metrics['f1']) == 1.0

    def test_multiclass_good_proba(self, multiclass_clf_data):
        # Pass true values as one-hot to match shape of proba predictions
        metrics = calculate_classification_metrics(
            multiclass_clf_data["true_one_hot"], 
            multiclass_clf_data["pred_proba_good"]
        )
        assert 0.0 < metrics['accuracy'] < 1.0
        assert 0.0 < metrics['precision'] < 1.0

class TestCalculateMetricsDispatcher:
    def test_dispatch_regression(self, regression_data):
        metrics = calculate_metrics(regression_data["true"], regression_data["pred_good"], task_type='regression')
        assert 'rmse' in metrics
        assert 'auc' not in metrics

    def test_dispatch_classification_binary(self, binary_clf_data):
        metrics = calculate_metrics(binary_clf_data["true"], binary_clf_data["pred_proba_good"], task_type='classification')
        assert 'accuracy' in metrics
        assert 'auc' in metrics

    def test_dispatch_classification_multiclass(self, multiclass_clf_data):
        metrics = calculate_metrics(multiclass_clf_data["true_indices"], multiclass_clf_data["pred_indices_perfect"], task_type='classification')
        assert 'accuracy' in metrics
        assert 'auc' not in metrics # AUC not typically for multi-class direct indices

    def test_unsupported_task_type(self, regression_data):
        with pytest.raises(ValueError, match="Unsupported task_type"):
            calculate_metrics(regression_data["true"], regression_data["pred_good"], task_type='unknown_task')

    def test_input_conversion_torch_tensor(self, regression_data):
        true_tensor = torch.tensor(regression_data["true"])
        pred_tensor = torch.tensor(regression_data["pred_good"])
        metrics = calculate_metrics(true_tensor, pred_tensor, task_type='regression')
        assert 'rmse' in metrics

    def test_input_conversion_list(self, regression_data):
        true_list = regression_data["true"].tolist()
        pred_list = regression_data["pred_good"].tolist()
        metrics = calculate_metrics(true_list, pred_list, task_type='regression')
        assert 'rmse' in metrics
        
    def test_shape_mismatch(self, regression_data):
        true_short = regression_data["true"][:-1]
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_metrics(true_short, regression_data["pred_good"], task_type='regression')

    def test_empty_inputs_regression(self):
        true_empty = np.array([])
        pred_empty = np.array([])
        # sklearn metrics will raise ValueError for empty inputs
        with pytest.raises(ValueError):
            calculate_metrics(true_empty, pred_empty, task_type='regression')
        
        true_empty_dim = np.empty((0,1))
        pred_empty_dim = np.empty((0,1))
        with pytest.raises(ValueError):
            calculate_metrics(true_empty_dim, pred_empty_dim, task_type='regression')


    def test_empty_inputs_classification(self):
        true_empty = np.array([])
        pred_empty = np.array([])
        with pytest.raises(ValueError):
            calculate_metrics(true_empty, pred_empty, task_type='classification')

    def test_nan_inputs_regression(self, regression_data):
        true_nan = regression_data["true"].copy()
        true_nan[0] = np.nan
        with pytest.raises(ValueError): # sklearn metrics raise ValueError for NaNs
            calculate_metrics(true_nan, regression_data["pred_good"], task_type='regression')
        
        pred_nan = regression_data["pred_good"].copy()
        pred_nan[0] = np.nan
        with pytest.raises(ValueError):
            calculate_metrics(regression_data["true"], pred_nan, task_type='regression')

    def test_nan_inputs_classification(self, binary_clf_data):
        true_nan = binary_clf_data["true"].copy().astype(float)
        true_nan[0] = np.nan
        with pytest.raises(ValueError):
            calculate_metrics(true_nan, binary_clf_data["pred_proba_good"], task_type='classification')

        pred_nan = binary_clf_data["pred_proba_good"].copy()
        pred_nan[0] = np.nan
        with pytest.raises(ValueError):
            calculate_metrics(binary_clf_data["true"], pred_nan, task_type='classification')

    def test_inf_inputs_regression(self, regression_data):
        true_inf = regression_data["true"].copy()
        true_inf[0] = np.inf
        with pytest.raises(ValueError):
            calculate_metrics(true_inf, regression_data["pred_good"], task_type='regression')
        
        pred_inf = regression_data["pred_good"].copy()
        pred_inf[0] = np.inf
        with pytest.raises(ValueError):
            calculate_metrics(regression_data["true"], pred_inf, task_type='regression')

    def test_inf_inputs_classification(self, binary_clf_data):
        # For classification, inf in true values is less common but should be handled
        true_inf = binary_clf_data["true"].copy().astype(float)
        true_inf[0] = np.inf
        with pytest.raises(ValueError): # accuracy_score might handle it, but precision/recall might not
            calculate_metrics(true_inf, binary_clf_data["pred_proba_good"], task_type='classification')

        pred_inf = binary_clf_data["pred_proba_good"].copy()
        pred_inf[0] = np.inf
        # This will likely fail in the (pred_values > threshold) step or sklearn metrics
        with pytest.raises(ValueError): # roc_auc_score will fail
             calculate_metrics(binary_clf_data["true"], pred_inf, task_type='classification')


class TestMaskedMetrics:
    def test_node_level_metrics_with_mask(self, regression_data):
        true = regression_data["true"]
        pred = regression_data["pred_good"]
        mask = np.array([True, False, True, False, True]) # Select 1st, 3rd, 5th elements
        
        masked_true = true[mask]
        masked_pred = pred[mask]
        
        expected_metrics = calculate_regression_metrics(masked_true, masked_pred)
        actual_metrics = calculate_node_level_metrics(true, pred, task_type='regression', node_mask=mask)
        
        assert pytest.approx(actual_metrics['rmse']) == expected_metrics['rmse']
        assert pytest.approx(actual_metrics['mae']) == expected_metrics['mae']

    def test_graph_level_metrics_with_mask(self, binary_clf_data):
        true = binary_clf_data["true"]
        pred = binary_clf_data["pred_proba_good"]
        mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool)

        masked_true = true[mask.numpy()]
        masked_pred = pred[mask.numpy()]

        expected_metrics = calculate_classification_metrics(masked_true, masked_pred)
        actual_metrics = calculate_graph_level_metrics(true, pred, task_type='classification', graph_mask=mask)

        assert pytest.approx(actual_metrics['accuracy']) == expected_metrics['accuracy']
        assert pytest.approx(actual_metrics['auc']) == expected_metrics['auc']


@pytest.fixture(scope="module")
def temp_plot_dir():
    """Creates a temporary directory for plot files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

class TestVisualizePredictions:
    def test_visualize_regression(self, regression_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "regression_plot.png")
        fig = visualize_predictions(
            regression_data["true"], 
            regression_data["pred_good"], 
            task_type='regression',
            save_path=save_path,
            show_metrics=True
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        assert len(fig.axes[0].texts) > 0 # Check if metrics text is present
        plt.close(fig) # Close plot to free memory

    def test_visualize_regression_no_metrics_display(self, regression_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "regression_plot_no_metrics.png")
        fig = visualize_predictions(
            regression_data["true"],
            regression_data["pred_good"],
            task_type='regression',
            save_path=save_path,
            show_metrics=False  # Test this flag
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        # Check that no text annotations (metrics) are on the plot
        assert len(fig.axes[0].texts) == 0
        plt.close(fig)

    def test_visualize_binary_classification_proba(self, binary_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "binary_clf_proba_plot.png")
        fig = visualize_predictions(
            binary_clf_data["true"], 
            binary_clf_data["pred_proba_good"], 
            task_type='classification',
            save_path=save_path,
            show_metrics=True
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        assert len(fig.axes[0].texts) > 0 # Check if metrics text is present (in addition to CM numbers)
        plt.close(fig)

    def test_visualize_binary_classification_labels(self, binary_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "binary_clf_labels_plot.png")
        fig = visualize_predictions(
            binary_clf_data["true"], 
            binary_clf_data["pred_binary_good"], 
            task_type='classification',
            save_path=save_path,
            show_metrics=True
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        assert len(fig.axes[0].texts) > 0 
        plt.close(fig)

    def test_visualize_binary_classification_no_metrics_display(self, binary_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "binary_clf_proba_no_metrics_plot.png")
        fig = visualize_predictions(
            binary_clf_data["true"],
            binary_clf_data["pred_proba_good"],
            task_type='classification',
            save_path=save_path,
            show_metrics=False # Test this flag
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        # Metrics text is annotated outside the main CM, CM numbers are always there.
        # We expect CM number texts, but not the separate metrics block.
        # The metrics block is added as a single plt.annotate call.
        # If show_metrics is False, this annotation should not exist.
        # The CM cell values are added in a loop.
        # A simple check is that the number of texts should be cm.shape[0]*cm.shape[1] if no metrics
        cm_texts_count = 2*2 # For a 2x2 confusion matrix
        assert len(fig.axes[0].texts) == cm_texts_count
        plt.close(fig)


    def test_visualize_multiclass_classification_proba(self, multiclass_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "multiclass_clf_proba_plot.png")
        fig = visualize_predictions(
            multiclass_clf_data["true_one_hot"], # True as one-hot for proba preds
            multiclass_clf_data["pred_proba_good"], 
            task_type='classification',
            save_path=save_path,
            show_metrics=True
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        assert len(fig.axes[0].texts) > 0
        plt.close(fig)

    def test_visualize_multiclass_classification_indices(self, multiclass_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "multiclass_clf_indices_plot.png")
        fig = visualize_predictions(
            multiclass_clf_data["true_indices"], 
            multiclass_clf_data["pred_indices_perfect"], 
            task_type='classification',
            save_path=save_path,
            show_metrics=True
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        assert len(fig.axes[0].texts) > 0
        plt.close(fig)

    def test_visualize_multiclass_classification_no_metrics_display(self, multiclass_clf_data, temp_plot_dir):
        save_path = os.path.join(temp_plot_dir, "multiclass_clf_indices_no_metrics_plot.png")
        fig = visualize_predictions(
            multiclass_clf_data["true_indices"],
            multiclass_clf_data["pred_indices_perfect"],
            task_type='classification',
            save_path=save_path,
            show_metrics=False # Test this flag
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        num_classes = multiclass_clf_data["pred_proba_good"].shape[1]
        cm_texts_count = num_classes * num_classes
        assert len(fig.axes[0].texts) == cm_texts_count
        plt.close(fig)

    def test_visualize_no_save(self, regression_data):
        fig = visualize_predictions(regression_data["true"], regression_data["pred_good"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)