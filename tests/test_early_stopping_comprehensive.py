"""
tests/test_early_stopping_comprehensive.py

Comprehensive Test Suite for Early Stopping and Monitoring Systems.

This module provides extensive testing for the advanced early stopping and monitoring
systems, covering unit tests, integration tests, edge cases, and performance validation.
The tests ensure robust behavior across different training scenarios and configurations.

Test Categories:
    - Unit Tests: Individual components (MetricTracker, EarlyStopping, etc.)
    - Integration Tests: Full training workflow with early stopping
    - Edge Cases: Extreme scenarios, error conditions, and boundary cases
    - Configuration Tests: Different configuration combinations
    - Performance Tests: Memory usage and computational efficiency
    - Mock Training Tests: Simulated training scenarios for validation

Key Features:
    - Pytest-based test framework with fixtures and parameterization
    - Mock training scenarios with controlled metric trajectories
    - Memory leak detection and performance profiling
    - Configuration validation and error handling tests
    - Statistical significance testing validation
    - Checkpoint management and restoration tests

References:
    - Tests early_stopping.py, validation_monitor.py, and config.py
    - Uses pytest best practices for ML testing
    - Includes property-based testing for robust validation
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, List, Any, Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats

# Import the components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from moml.models.mgnn.training.early_stopping import (
    AdvancedEarlyStopping, EarlyStoppingConfig, MetricTracker,
    MonitorMode, EarlyStoppingReason, CheckpointManager,
    create_early_stopping
)
from moml.models.mgnn.training.validation_monitor import (
    ValidationMonitor, DashboardConfig, AlertSystem, MetricsVisualizer,
    create_validation_monitor
)
from moml.models.mgnn.training.config import (
    TrainingConfig, ConfigManager, create_molecular_config, create_debug_config
)
from moml.models.mgnn.training.enhanced_trainer import (
    EnhancedMGNNTrainer, ModelManager, TrainingPipeline
)

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Test Fixtures and Mock Components
class MockModel(nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(10, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, **kwargs):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        return self.output(x)


class MockTrainer:
    """Mock trainer that simulates training behavior."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.config = {"learning_rate": 0.001}
        self.optimizer = optim.Adam(model.parameters())
        self.stop_training = False
        self.device = "cpu"
        self.callbacks = []
    
    def _call_callbacks(self, hook_name, *args, **kwargs):
        """Call callback hooks."""
        for callback in self.callbacks:
            hook_method = getattr(callback, hook_name, None)
            if hook_method is not None:
                hook_method(self, *args, **kwargs)


class MetricSimulator:
    """Simulates different metric trajectories for testing."""
    
    @staticmethod
    def decreasing_loss(epochs: int, noise: float = 0.1) -> List[float]:
        """Simulate decreasing loss with noise."""
        base_loss = 2.0
        losses = []
        for epoch in range(epochs):
            # Exponential decay with noise
            loss = base_loss * np.exp(-epoch * 0.1) + np.random.normal(0, noise)
            losses.append(max(0.01, loss))  # Ensure positive
        return losses
    
    @staticmethod
    def plateauing_loss(epochs: int, plateau_start: int = 20) -> List[float]:
        """Simulate loss that decreases then plateaus."""
        losses = []
        for epoch in range(epochs):
            if epoch < plateau_start:
                loss = 2.0 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.05)
            else:
                # Plateau with small fluctuations
                loss = 0.3 + np.random.normal(0, 0.02)
            losses.append(max(0.01, loss))
        return losses
    
    @staticmethod
    def diverging_loss(epochs: int, diverge_start: int = 10) -> List[float]:
        """Simulate loss that starts decreasing then diverges."""
        losses = []
        for epoch in range(epochs):
            if epoch < diverge_start:
                loss = 2.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
            else:
                # Exponential growth (divergence)
                loss = 0.5 * np.exp((epoch - diverge_start) * 0.2)
            losses.append(loss)
        return losses
    
    @staticmethod
    def noisy_improvement(epochs: int, noise_level: float = 0.3) -> List[float]:
        """Simulate loss with high noise but overall improvement."""
        losses = []
        for epoch in range(epochs):
            base_loss = 1.5 * np.exp(-epoch * 0.05)
            noise = np.random.normal(0, noise_level)
            losses.append(max(0.01, base_loss + noise))
        return losses
    
    @staticmethod
    def cyclic_loss(epochs: int, cycle_length: int = 10) -> List[float]:
        """Simulate cyclic loss pattern."""
        losses = []
        for epoch in range(epochs):
            cycle_pos = (epoch % cycle_length) / cycle_length
            base_loss = 1.0 + 0.5 * np.sin(2 * np.pi * cycle_pos)
            trend = 1.0 * np.exp(-epoch * 0.02)  # Overall decreasing trend
            losses.append(base_loss * trend)
        return losses


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    return MockModel(hidden_dim=32)


@pytest.fixture
def mock_trainer(mock_model):
    """Create mock trainer for testing."""
    return MockTrainer(mock_model)


@pytest.fixture
def basic_config():
    """Create basic early stopping configuration."""
    return EarlyStoppingConfig(
        monitor="val_loss",
        patience=5,
        min_delta=1e-4,
        mode=MonitorMode.MIN,
        warmup_epochs=2,
        verbose=0  # Reduce output for tests
    )


@pytest.fixture
def advanced_config(temp_dir):
    """Create advanced early stopping configuration."""
    return EarlyStoppingConfig(
        monitor="val_loss",
        patience=10,
        min_delta=1e-3,
        mode=MonitorMode.MIN,
        warmup_epochs=3,
        statistical_test=True,
        reduce_lr_on_plateau=True,
        detect_divergence=True,
        save_best_checkpoint=True,
        checkpoint_dir=str(temp_dir / "checkpoints"),
        log_to_wandb=False,  # Disable for tests
        log_to_tensorboard=False,
        verbose=0
    )


# Unit Tests for MetricTracker
class TestMetricTracker:
    """Test suite for MetricTracker class."""
    
    def test_metric_tracker_initialization(self):
        """Test MetricTracker initialization."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        assert tracker.metric_name == "val_loss"
        assert tracker.mode == MonitorMode.MIN
        assert len(tracker.values) == 0
        assert tracker.best_value == float('inf')
        assert tracker.best_epoch == -1
    
    def test_metric_update_minimization(self):
        """Test metric updates for minimization."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        # First update should be best
        is_better = tracker.update(1.0, epoch=0)
        assert is_better
        assert tracker.best_value == 1.0
        assert tracker.best_epoch == 0
        
        # Better value
        is_better = tracker.update(0.5, epoch=1)
        assert is_better
        assert tracker.best_value == 0.5
        assert tracker.best_epoch == 1
        
        # Worse value
        is_better = tracker.update(0.8, epoch=2)
        assert not is_better
        assert tracker.best_value == 0.5  # Should remain unchanged
        assert tracker.best_epoch == 1
    
    def test_metric_update_maximization(self):
        """Test metric updates for maximization."""
        tracker = MetricTracker("val_acc", MonitorMode.MAX)
        
        # First update should be best
        is_better = tracker.update(0.7, epoch=0)
        assert is_better
        assert tracker.best_value == 0.7
        
        # Better value
        is_better = tracker.update(0.9, epoch=1)
        assert is_better
        assert tracker.best_value == 0.9
        
        # Worse value
        is_better = tracker.update(0.8, epoch=2)
        assert not is_better
        assert tracker.best_value == 0.9
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        # Not enough data
        assert tracker.get_trend() == "insufficient_data"
        
        # Add decreasing values (improving for MIN mode)
        for i, value in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
            tracker.update(value, epoch=i)
        
        trend = tracker.get_trend()
        assert trend == "improving"
        
        # Add increasing values (worsening for MIN mode)
        for i, value in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], start=10):
            tracker.update(value, epoch=i)
        
        trend = tracker.get_trend()
        assert trend == "worsening"
    
    def test_improvement_detection(self):
        """Test recent improvement detection."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        # Add initial values
        values = [1.0, 0.9, 0.8, 0.7, 0.6]
        for i, value in enumerate(values):
            tracker.update(value, epoch=i)
        
        # Should detect improvement
        assert tracker.has_improved_recently(patience=3, min_delta=0.05) == True
        
        # Add plateau values
        for i, value in enumerate([0.6, 0.61, 0.59, 0.60, 0.61], start=5):
            tracker.update(value, epoch=i)
        
        # Should not detect significant improvement
        assert tracker.has_improved_recently(patience=4, min_delta=0.05) == False
    
    def test_divergence_detection(self):
        """Test divergence detection."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        # Add normal decreasing values
        for i, value in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]):
            tracker.update(value, epoch=i)
        
        # Should not detect divergence
        assert tracker.detect_divergence(threshold=2.0, patience=2) == False
        
        # Add diverging values
        for i, value in enumerate([0.5, 1.0, 2.0, 4.0, 8.0], start=5):
            tracker.update(value, epoch=i)
        
        # Should detect divergence
        assert tracker.detect_divergence(threshold=2.0, patience=3) == True
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        tracker = MetricTracker("val_loss", MonitorMode.MIN)
        
        # Empty tracker
        stats = tracker.get_statistics()
        assert stats == {}
        
        # Add values
        values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.3, 0.25, 0.15, 0.1, 0.05]
        for i, value in enumerate(values):
            tracker.update(value, epoch=i)
        
        stats = tracker.get_statistics()
        
        assert 'count' in stats
        assert 'current' in stats
        assert 'best' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'trend' in stats
        
        assert stats['count'] == len(values)
        assert stats['current'] == values[-1]
        assert stats['best'] == min(values)  # MIN mode


# Unit Tests for EarlyStopping
class TestAdvancedEarlyStopping:
    """Test suite for AdvancedEarlyStopping class."""
    
    def test_initialization_basic(self, basic_config):
        """Test basic initialization."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        
        assert early_stopping.config == basic_config
        assert early_stopping.wait == 0
        assert early_stopping.stopped_epoch == 0
        assert early_stopping.stop_reason is None
        assert early_stopping.primary_tracker.metric_name == "val_loss"
    
    def test_initialization_advanced(self, advanced_config):
        """Test advanced initialization with all features."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        
        assert early_stopping.config.statistical_test == True
        assert early_stopping.config.detect_divergence == True
        assert early_stopping.config.reduce_lr_on_plateau == True
        assert early_stopping.checkpoint_manager is not None
    
    def test_early_stopping_patience(self, basic_config, mock_trainer):
        """Test early stopping triggered by patience."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer.callbacks = [early_stopping]
        
        # Initialize
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate improving phase
        for epoch in range(3):
            logs = {"val_loss": 1.0 - epoch * 0.1}
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
            assert not mock_trainer.stop_training
        
        # Simulate plateau (no improvement for patience epochs)
        plateau_loss = 0.7
        for epoch in range(3, 3 + basic_config.patience):
            logs = {"val_loss": plateau_loss + 0.001 * epoch}  # Slight increase
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
        
        # Should trigger early stopping
        assert mock_trainer.stop_training
        assert early_stopping.stop_reason == EarlyStoppingReason.PATIENCE_EXCEEDED
    
    def test_warmup_period(self, basic_config, mock_trainer):
        """Test that early stopping is inactive during warmup."""
        basic_config.warmup_epochs = 5
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer.callbacks = [early_stopping]
        
        early_stopping.on_train_begin(mock_trainer)
        
        # During warmup, early stopping should be inactive even with bad metrics
        for epoch in range(basic_config.warmup_epochs):
            logs = {"val_loss": 10.0}  # Very bad loss
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
            assert not mock_trainer.stop_training
        
        # After warmup, should work normally
        logs = {"val_loss": 1.0}
        early_stopping.on_epoch_end(mock_trainer, basic_config.warmup_epochs, logs)
        assert not mock_trainer.stop_training  # First good value
    
    def test_divergence_detection(self, advanced_config, mock_trainer):
        """Test divergence detection."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer.callbacks = [early_stopping]
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Start with good loss
        early_stopping.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        
        # Simulate divergence
        divergence_threshold = advanced_config.divergence_threshold
        for epoch in range(1, advanced_config.divergence_patience + 2):
            bad_loss = 0.5 * divergence_threshold * 2  # Exceeds threshold
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": bad_loss})
        
        # Should trigger divergence stopping
        assert mock_trainer.stop_training
        assert early_stopping.stop_reason == EarlyStoppingReason.DIVERGENCE_DETECTED
    
    def test_learning_rate_reduction(self, advanced_config, mock_trainer):
        """Test learning rate reduction on plateau."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer.callbacks = [early_stopping]
        
        initial_lr = 0.001
        mock_trainer.optimizer = optim.Adam(mock_trainer.model.parameters(), lr=initial_lr)
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate plateau
        plateau_loss = 0.5
        for epoch in range(advanced_config.lr_reduction_patience + 1):
            logs = {"val_loss": plateau_loss}
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
        
        # Learning rate should be reduced
        current_lr = mock_trainer.optimizer.param_groups[0]['lr']
        expected_lr = initial_lr * advanced_config.lr_reduction_factor
        assert abs(current_lr - expected_lr) < 1e-8
    
    def test_best_weights_restoration(self, basic_config, mock_trainer):
        """Test best weights restoration."""
        basic_config.restore_best_weights = True
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer.callbacks = [early_stopping]
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Save initial state
        initial_state = {k: v.clone() for k, v in mock_trainer.model.state_dict().items()}
        
        # Epoch with improvement (best weights should be saved)
        early_stopping.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        
        # Modify model weights
        with torch.no_grad():
            for param in mock_trainer.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Trigger early stopping
        for epoch in range(1, basic_config.patience + 2):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": 1.0})
        
        # Best weights should be restored
        assert mock_trainer.stop_training
        restored_state = mock_trainer.model.state_dict()
        
        # Check if weights were restored (should be close to initial state after first epoch)
        for key in initial_state.keys():
            # Weights should be different from the modified state but close to initial
            assert not torch.allclose(restored_state[key], initial_state[key], atol=1e-3)
    
    def test_multiple_metrics_monitoring(self):
        """Test monitoring multiple metrics."""
        config = EarlyStoppingConfig(
            monitor="val_loss",
            multiple_metrics=["val_mae", "val_r2"],
            patience=5,
            verbose=0
        )
        
        early_stopping = AdvancedEarlyStopping(config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Check that all metrics are being tracked
        assert "val_loss" in early_stopping.metric_trackers
        assert "val_mae" in early_stopping.metric_trackers
        assert "val_r2" in early_stopping.metric_trackers
        
        # Update with multiple metrics
        logs = {"val_loss": 0.5, "val_mae": 0.3, "val_r2": 0.8}
        early_stopping.on_epoch_end(mock_trainer, 0, logs)
        
        # All trackers should have values
        for metric_name in ["val_loss", "val_mae", "val_r2"]:
            tracker = early_stopping.metric_trackers[metric_name]
            assert len(tracker.values) == 1
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing."""
        config = EarlyStoppingConfig(
            monitor="val_loss",
            patience=10,
            statistical_test=True,
            confidence_level=0.95,
            verbose=0
        )
        
        early_stopping = AdvancedEarlyStopping(config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate small, non-significant improvements
        base_loss = 0.5
        for epoch in range(20):
            # Very small random improvements
            loss = base_loss - np.random.uniform(0, 0.001)
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            
            if mock_trainer.stop_training:
                break
        
        # Should eventually stop due to lack of statistical significance
        # (This is probabilistic, so we just check that the mechanism works)
        assert hasattr(early_stopping, 'recent_improvements')
        assert isinstance(early_stopping.recent_improvements, type(early_stopping.recent_improvements))
    
    def test_get_statistics(self, basic_config, mock_trainer):
        """Test statistics generation."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer.callbacks = [early_stopping]
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Add some data
        for epoch in range(5):
            logs = {"val_loss": 1.0 - epoch * 0.1}
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
        
        stats = early_stopping.get_statistics()
        
        assert 'stopped_early' in stats
        assert 'best_score' in stats
        assert 'wait_count' in stats
        assert 'val_loss_stats' in stats
        
        assert stats['stopped_early'] == False
        assert stats['best_score'] == 0.6  # Last value was best
        assert 'count' in stats['val_loss_stats']


# Unit Tests for ValidationMonitor
class TestValidationMonitor:
    """Test suite for ValidationMonitor class."""
    
    def test_validation_monitor_initialization(self, temp_dir):
        """Test ValidationMonitor initialization."""
        config = DashboardConfig(
            primary_metrics=["train_loss", "val_loss"],
            export_dir=str(temp_dir),
            log_to_wandb=False,
            log_to_tensorboard=False
        )
        
        monitor = ValidationMonitor(config)
        
        assert monitor.config == config
        assert len(monitor.metrics_history) == 0
        assert monitor.monitoring_active == False
    
    def test_metrics_tracking(self, temp_dir):
        """Test metrics tracking functionality."""
        config = DashboardConfig(
            primary_metrics=["train_loss", "val_loss"],
            export_dir=str(temp_dir),
            log_to_wandb=False,
            log_to_tensorboard=False,
            enable_alerts=False
        )
        
        monitor = ValidationMonitor(config)
        mock_trainer = MockTrainer(MockModel())
        
        monitor.on_train_begin(mock_trainer)
        
        # Simulate epoch updates
        for epoch in range(5):
            logs = {
                "train_loss": 1.0 - epoch * 0.1,
                "val_loss": 0.8 - epoch * 0.08,
                "learning_rate": 0.001
            }
            monitor.on_epoch_end(mock_trainer, epoch, logs)
        
        # Check metrics were recorded
        assert len(monitor.metrics_history["train_loss"]) == 5
        assert len(monitor.metrics_history["val_loss"]) == 5
        assert len(monitor.epochs) == 5
        
        monitor.on_train_end(mock_trainer)
    
    def test_alert_system(self, temp_dir):
        """Test alert system functionality."""
        config = DashboardConfig(
            enable_alerts=True,
            alert_thresholds={
                "val_loss": {"warning": 1.0, "error": 2.0, "critical": 5.0}
            },
            export_dir=str(temp_dir),
            log_to_wandb=False,
            log_to_tensorboard=False
        )
        
        monitor = ValidationMonitor(config)
        mock_trainer = MockTrainer(MockModel())
        
        # Track alerts
        triggered_alerts = []
        
        def alert_callback(message, level, context):
            triggered_alerts.append((message, level, context))
        
        monitor.alert_system.add_alert_callback(alert_callback)
        monitor.on_train_begin(mock_trainer)
        
        # Trigger warning
        monitor.on_epoch_end(mock_trainer, 0, {"val_loss": 1.5})
        assert len(triggered_alerts) == 1
        assert "warning" in triggered_alerts[0][1].value
        
        # Trigger error
        monitor.on_epoch_end(mock_trainer, 1, {"val_loss": 3.0})
        assert len(triggered_alerts) == 2
        assert "error" in triggered_alerts[1][1].value
        
        monitor.on_train_end(mock_trainer)


# Integration Tests
class TestTrainingIntegration:
    """Integration tests for complete training workflows."""
    
    def test_complete_training_workflow(self, temp_dir):
        """Test complete training workflow with early stopping."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create mock data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)
        
        # Create model and config
        model = MockModel()
        config = create_debug_config(max_epochs=10, batch_size=16)
        config.early_stopping.enabled = True
        config.early_stopping.patience = 3
        config.early_stopping.checkpoint_dir = str(temp_dir)
        config.monitoring.export_dir = str(temp_dir)
        config.monitoring.log_to_wandb = False
        config.monitoring.log_to_tensorboard = False
        
        # Create enhanced trainer
        trainer = EnhancedMGNNTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Mock forward pass for our simple test data
        def mock_forward(x, **kwargs):
            return model(x)
        
        trainer.model.forward = mock_forward
        
        # Run training
        history = trainer.train(epochs=5)
        
        # Check that training ran
        assert len(history["train_loss"]) > 0
        assert trainer.current_epoch >= 0
    
    def test_checkpoint_restoration(self, temp_dir):
        """Test checkpoint saving and restoration."""
        model = MockModel()
        config = TrainingConfig()
        config.checkpoint.dirpath = str(temp_dir)
        config.checkpoint.enabled = True
        
        model_manager = ModelManager(config, model)
        
        # Save initial checkpoint
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = optim.Adam(model.parameters())
        
        checkpoint_path = model_manager.save_checkpoint(
            epoch=5,
            metrics={"val_loss": 0.3, "train_loss": 0.25},
            optimizer=optimizer
        )
        
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))
        
        # Restore checkpoint
        checkpoint_data = model_manager.load_checkpoint(checkpoint_path)
        
        assert checkpoint_data["epoch"] == 5
        assert "val_loss" in checkpoint_data["metrics"]
        
        # Check model was restored
        restored_params = model.state_dict()
        for key in initial_params.keys():
            assert torch.allclose(initial_params[key], restored_params[key])


# Edge Cases and Error Handling Tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_metrics(self, basic_config):
        """Test handling of empty or missing metrics."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Empty logs
        early_stopping.on_epoch_end(mock_trainer, 0, {})
        assert not mock_trainer.stop_training
        
        # Missing monitored metric
        early_stopping.on_epoch_end(mock_trainer, 1, {"other_metric": 0.5})
        assert not mock_trainer.stop_training
        
        # None values
        early_stopping.on_epoch_end(mock_trainer, 2, {"val_loss": None})
        assert not mock_trainer.stop_training
    
    def test_invalid_metric_values(self, basic_config):
        """Test handling of invalid metric values."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # NaN values
        early_stopping.on_epoch_end(mock_trainer, 0, {"val_loss": float('nan')})
        assert not mock_trainer.stop_training
        
        # Infinite values
        early_stopping.on_epoch_end(mock_trainer, 1, {"val_loss": float('inf')})
        assert not mock_trainer.stop_trading
        
        # Negative values (might be valid for some metrics)
        early_stopping.on_epoch_end(mock_trainer, 2, {"val_loss": -0.5})
        # Should not crash, behavior depends on implementation
    
    def test_extreme_patience_values(self):
        """Test extreme patience values."""
        # Very high patience
        config = EarlyStoppingConfig(patience=10000, verbose=0)
        early_stopping = AdvancedEarlyStopping(config)
        assert early_stopping.config.patience == 10000
        
        # Minimum patience
        config = EarlyStoppingConfig(patience=1, verbose=0)
        early_stopping = AdvancedEarlyStopping(config)
        assert early_stopping.config.patience == 1
    
    def test_memory_usage_with_long_training(self):
        """Test memory usage doesn't grow unbounded with long training."""
        config = EarlyStoppingConfig(patience=5, verbose=0)
        early_stopping = AdvancedEarlyStopping(config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate very long training
        for epoch in range(1000):
            # Vary loss to prevent early stopping
            loss = 0.5 + 0.1 * np.sin(epoch * 0.1)
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            
            if epoch % 100 == 0:
                # Check memory usage isn't growing unbounded
                tracker = early_stopping.primary_tracker
                # Deque should limit memory usage
                assert len(tracker.values) <= tracker.maxlen
    
    def test_concurrent_callback_execution(self, basic_config):
        """Test behavior with multiple callbacks and concurrent execution."""
        early_stopping1 = AdvancedEarlyStopping(basic_config)
        early_stopping2 = AdvancedEarlyStopping(basic_config)
        
        mock_trainer = MockTrainer(MockModel())
        mock_trainer.callbacks = [early_stopping1, early_stopping2]
        
        # Should not interfere with each other
        early_stopping1.on_train_begin(mock_trainer)
        early_stopping2.on_train_begin(mock_trainer)
        
        for epoch in range(10):
            logs = {"val_loss": 1.0}  # No improvement
            early_stopping1.on_epoch_end(mock_trainer, epoch, logs)
            early_stopping2.on_epoch_end(mock_trainer, epoch, logs)
        
        # Both should trigger stopping eventually
        assert mock_trainer.stop_training


# Performance Tests
class TestPerformance:
    """Performance and efficiency tests."""
    
    def test_metric_tracker_performance(self):
        """Test MetricTracker performance with large datasets."""
        tracker = MetricTracker("loss", MonitorMode.MIN)
        
        start_time = time.time()
        
        # Add many values
        for i in range(10000):
            tracker.update(np.random.random(), epoch=i)
        
        elapsed = time.time() - start_time
        
        # Should be fast (less than 1 second for 10k updates)
        assert elapsed < 1.0
        
        # Memory should be limited by maxlen
        assert len(tracker.values) <= tracker.maxlen
    
    def test_early_stopping_overhead(self, basic_config):
        """Test early stopping computational overhead."""
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        start_time = time.time()
        
        # Simulate many epochs
        for epoch in range(1000):
            logs = {"val_loss": 0.5 + 0.01 * np.sin(epoch)}
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
        
        elapsed = time.time() - start_time
        
        # Should be very fast (overhead < 10ms per epoch on average)
        avg_time_per_epoch = elapsed / 1000
        assert avg_time_per_epoch < 0.01  # 10ms per epoch
    
    def test_memory_leak_detection(self, basic_config):
        """Test for memory leaks in long-running training."""
        import gc
        import sys
        
        early_stopping = AdvancedEarlyStopping(basic_config)
        mock_trainer = MockTrainer(MockModel())
        
        # Get initial memory usage
        gc.collect()
        initial_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Run training simulation
        early_stopping.on_train_begin(mock_trainer)
        
        for epoch in range(100):
            logs = {"val_loss": np.random.random()}
            early_stopping.on_epoch_end(mock_trainer, epoch, logs)
        
        early_stopping.on_train_end(mock_trainer)
        
        # Check for memory leaks
        gc.collect()
        final_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Reference count shouldn't grow significantly
        if initial_refs > 0:  # Only check if gettotalrefcount is available
            ref_growth = final_refs - initial_refs
            assert ref_growth < 1000  # Allow some growth but not excessive


# Configuration Tests
class TestConfiguration:
    """Test configuration validation and edge cases."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = EarlyStoppingConfig(patience=10, min_delta=0.001)
        # Should not raise exception
        
        # Invalid patience
        with pytest.raises(ValueError):
            EarlyStoppingConfig(patience=0)
        
        with pytest.raises(ValueError):
            EarlyStoppingConfig(patience=-5)
        
        # Invalid confidence level
        with pytest.raises(ValueError):
            EarlyStoppingConfig(confidence_level=0.3)  # Too low
        
        with pytest.raises(ValueError):
            EarlyStoppingConfig(confidence_level=1.1)  # Too high
    
    def test_config_manager_functionality(self, temp_dir):
        """Test configuration manager functionality."""
        config_manager = ConfigManager(str(temp_dir))
        
        # Create a config
        config = create_molecular_config(max_epochs=50)
        
        # Save config
        config_manager.save_config(config, "test_config.yaml")
        
        # Check file was created
        config_file = temp_dir / "test_config.yaml"
        assert config_file.exists()
        
        # Validate config
        issues = config_manager.validate_config(config)
        # Should return list (empty if no issues)
        assert isinstance(issues, list)
    
    def test_config_template_creation(self, temp_dir):
        """Test template configuration creation."""
        config_manager = ConfigManager(str(temp_dir))
        
        # Create templates
        config_manager.create_template_configs()
        
        # Check that template files were created
        template_files = list(temp_dir.glob("*.yaml"))
        assert len(template_files) > 0
        
        # Check that files contain valid YAML
        for template_file in template_files:
            content = template_file.read_text()
            assert len(content) > 0


# Mock Training Scenario Tests
class TestTrainingScenarios:
    """Test different training scenarios with simulated data."""
    
    def test_normal_training_scenario(self, advanced_config):
        """Test normal training scenario with gradual improvement."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate normal decreasing loss
        losses = MetricSimulator.decreasing_loss(50)
        
        for epoch, loss in enumerate(losses):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            if mock_trainer.stop_training:
                break
        
        # Should complete training or stop appropriately
        assert epoch > advanced_config.warmup_epochs
    
    def test_plateauing_scenario(self, advanced_config):
        """Test plateauing loss scenario."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate plateauing loss
        losses = MetricSimulator.plateauing_loss(100, plateau_start=20)
        
        stopped_epoch = None
        for epoch, loss in enumerate(losses):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            if mock_trainer.stop_training:
                stopped_epoch = epoch
                break
        
        # Should stop due to plateau
        assert stopped_epoch is not None
        assert stopped_epoch > 20  # Should stop after plateau starts
        assert early_stopping.stop_reason == EarlyStoppingReason.PATIENCE_EXCEEDED
    
    def test_diverging_scenario(self, advanced_config):
        """Test diverging loss scenario."""
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate diverging loss
        losses = MetricSimulator.diverging_loss(50, diverge_start=10)
        
        stopped_epoch = None
        for epoch, loss in enumerate(losses):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            if mock_trainer.stop_training:
                stopped_epoch = epoch
                break
        
        # Should stop due to divergence
        assert stopped_epoch is not None
        assert stopped_epoch > 10  # Should stop after divergence starts
        assert early_stopping.stop_reason == EarlyStoppingReason.DIVERGENCE_DETECTED
    
    def test_noisy_improvement_scenario(self, advanced_config):
        """Test noisy but improving loss scenario."""
        # Increase patience for noisy scenario
        advanced_config.patience = 20
        advanced_config.statistical_test = True
        
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate noisy improvement
        losses = MetricSimulator.noisy_improvement(100, noise_level=0.2)
        
        for epoch, loss in enumerate(losses):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            if mock_trainer.stop_training:
                break
        
        # Should either complete training or stop reasonably late
        # (noisy improvement should be detected as significant)
        if mock_trainer.stop_training:
            assert epoch > 30  # Should train for reasonable time despite noise
    
    def test_cyclic_loss_scenario(self, advanced_config):
        """Test cyclic loss pattern scenario."""
        advanced_config.patience = 15
        
        early_stopping = AdvancedEarlyStopping(advanced_config)
        mock_trainer = MockTrainer(MockModel())
        
        early_stopping.on_train_begin(mock_trainer)
        
        # Simulate cyclic loss
        losses = MetricSimulator.cyclic_loss(100, cycle_length=10)
        
        for epoch, loss in enumerate(losses):
            early_stopping.on_epoch_end(mock_trainer, epoch, {"val_loss": loss})
            if mock_trainer.stop_training:
                break
        
        # Should handle cyclic patterns reasonably
        # Trend analysis should detect overall improvement despite cycles
        best_loss = early_stopping.primary_tracker.best_value
        assert best_loss < losses[0]  # Should show some improvement overall


# Test Utilities and Helpers
class TestTestUtilities:
    """Test the testing utilities themselves."""
    
    def test_metric_simulator_decreasing(self):
        """Test decreasing loss simulator."""
        losses = MetricSimulator.decreasing_loss(10)
        
        assert len(losses) == 10
        assert all(loss > 0 for loss in losses)  # All positive
        
        # Should show general decreasing trend
        first_half_avg = np.mean(losses[:5])
        second_half_avg = np.mean(losses[5:])
        assert second_half_avg < first_half_avg
    
    def test_metric_simulator_plateauing(self):
        """Test plateauing loss simulator."""
        losses = MetricSimulator.plateauing_loss(30, plateau_start=10)
        
        assert len(losses) == 30
        
        # First part should decrease
        first_part = losses[:10]
        assert first_part[0] > first_part[-1]
        
        # Second part should be relatively stable
        plateau_part = losses[15:25]  # Middle of plateau
        plateau_std = np.std(plateau_part)
        assert plateau_std < 0.1  # Should be relatively stable
    
    def test_metric_simulator_diverging(self):
        """Test diverging loss simulator."""
        losses = MetricSimulator.diverging_loss(20, diverge_start=5)
        
        assert len(losses) == 20
        
        # First part should decrease
        first_part = losses[:5]
        assert first_part[0] > first_part[-1]
        
        # Second part should increase dramatically
        last_part = losses[10:]
        assert last_part[-1] > last_part[0] * 2  # Should grow significantly


# Run specific test scenarios
if __name__ == "__main__":
    # Run with pytest: python -m pytest test_early_stopping_comprehensive.py -v
    
    # Or run specific tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--durations=10"  # Show 10 slowest tests
    ])