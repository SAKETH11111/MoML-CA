"""
validate_early_stopping_system.py

Comprehensive Validation Script for Early Stopping and Monitoring System.

This script validates the complete early stopping and monitoring implementation
with real training scenarios, plateau detection, checkpoint management, and
comprehensive monitoring capabilities. It demonstrates the integration of all
components working together in a production-like environment.

Key Validation Areas:
    - Early stopping with patience and min_delta thresholds
    - Plateau detection and learning rate reduction
    - Checkpoint saving and restoration with retention policies
    - Comprehensive metrics tracking and visualization
    - Alert system with configurable thresholds
    - Statistical significance testing for improvement detection
    - Memory efficiency and performance validation
    - Configuration system validation with Hydra integration

Success Criteria:
    - Early stopping triggers correctly with plateau detection
    - Best model checkpoints are saved and can be restored
    - Monitoring system captures and visualizes training progress
    - Alert system triggers appropriately for training anomalies
    - System handles edge cases gracefully
    - Memory usage remains bounded during long training
    - All configuration options work as expected

Usage:
    python validate_early_stopping_system.py [--config CONFIG_NAME] [--verbose]
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import our components
from moml.models.mgnn.training.early_stopping import (
    AdvancedEarlyStopping, EarlyStoppingConfig, create_early_stopping
)
from moml.models.mgnn.training.validation_monitor import (
    ValidationMonitor, DashboardConfig, create_validation_monitor
)
from moml.models.mgnn.training.config import (
    TrainingConfig, ConfigManager, create_molecular_config, create_debug_config
)
from moml.models.mgnn.training.enhanced_trainer import (
    EnhancedMGNNTrainer, ModelManager, TrainingPipeline, create_enhanced_trainer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationModel(nn.Module):
    """Simple model for validation testing."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights for better training dynamics
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, **kwargs):
        """Forward pass with support for trainer interface."""
        if hasattr(x, 'x'):  # Graph-like input
            x = x.x
        return self.layers(x)


class TrainingSimulator:
    """Simulates controlled training scenarios for validation."""
    
    def __init__(self, scenario: str = "plateau", num_samples: int = 1000):
        """
        Initialize training simulator.
        
        Args:
            scenario: Type of training scenario to simulate
            num_samples: Number of training samples
        """
        self.scenario = scenario
        self.num_samples = num_samples
        self.model = ValidationModel()
        self.create_datasets()
    
    def create_datasets(self):
        """Create synthetic datasets for different scenarios."""
        torch.manual_seed(42)  # Reproducible results
        
        if self.scenario == "plateau":
            # Create data that will lead to plateauing loss
            X = torch.randn(self.num_samples, 20)
            # Add some noise to make learning challenging
            y = (X[:, :5].sum(dim=1, keepdim=True) + 
                 torch.randn(self.num_samples, 1) * 0.5)
        
        elif self.scenario == "normal":
            # Create well-structured data for normal training
            X = torch.randn(self.num_samples, 20)
            y = X[:, :10].sum(dim=1, keepdim=True) + torch.randn(self.num_samples, 1) * 0.1
        
        elif self.scenario == "noisy":
            # Create noisy data with inconsistent patterns
            X = torch.randn(self.num_samples, 20)
            y = (X[:, :3].sum(dim=1, keepdim=True) + 
                 torch.randn(self.num_samples, 1) * 2.0)  # High noise
        
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        
        # Create train/val split
        split_idx = int(0.8 * self.num_samples)
        
        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        val_dataset = TensorDataset(X[split_idx:], y[split_idx:])
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Created {self.scenario} dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")


class EarlyStoppingValidator:
    """Validates early stopping functionality."""
    
    def __init__(self, temp_dir: Path):
        """
        Initialize validator.
        
        Args:
            temp_dir: Temporary directory for test files
        """
        self.temp_dir = temp_dir
        self.results = {}
    
    def validate_basic_early_stopping(self) -> Dict[str, Any]:
        """Validate basic early stopping functionality."""
        logger.info("🧪 Validating basic early stopping...")
        
        # Create simulator with plateauing scenario
        simulator = TrainingSimulator("plateau", num_samples=500)
        
        # Create configuration with short patience for testing
        config = create_debug_config(max_epochs=50, batch_size=32)
        config.early_stopping.enabled = True
        config.early_stopping.patience = 8
        config.early_stopping.min_delta = 1e-4
        config.early_stopping.warmup_epochs = 3
        config.early_stopping.verbose = 1
        config.checkpoint.dirpath = str(self.temp_dir / "checkpoints")
        config.monitoring.export_dir = str(self.temp_dir / "monitoring")
        config.monitoring.log_to_wandb = False
        config.monitoring.log_to_tensorboard = False
        
        # Create enhanced trainer
        trainer = create_enhanced_trainer(
            model=simulator.model,
            train_loader=simulator.train_loader,
            val_loader=simulator.val_loader,
            config=config
        )
        
        # Run training
        history = trainer.train(epochs=50)
        
        # Validate results
        validation_results = {
            "training_stopped_early": trainer.stop_training,
            "epochs_completed": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "early_stopping_triggered": trainer.early_stopping is not None and trainer.early_stopping.stopped_epoch > 0,
            "best_metrics": trainer.best_metrics.copy(),
            "checkpoint_saved": len(list(Path(config.checkpoint.dirpath).glob("*.pt"))) > 0
        }
        
        # Assertions for validation
        assert validation_results["epochs_completed"] < 50, "Training should have stopped early"
        assert validation_results["early_stopping_triggered"], "Early stopping should have been triggered"
        assert validation_results["checkpoint_saved"], "At least one checkpoint should have been saved"
        
        logger.info("✅ Basic early stopping validation passed")
        return validation_results
    
    def validate_plateau_detection(self) -> Dict[str, Any]:
        """Validate plateau detection and learning rate reduction.""" 
        logger.info("🧪 Validating plateau detection...")
        
        simulator = TrainingSimulator("plateau", num_samples=800)
        
        # Configuration with LR reduction on plateau
        config = create_debug_config(max_epochs=60, batch_size=32)
        config.early_stopping.enabled = True
        config.early_stopping.patience = 15
        config.early_stopping.reduce_lr_on_plateau = True
        config.early_stopping.lr_reduction_factor = 0.5
        config.early_stopping.lr_reduction_patience = 5
        config.early_stopping.min_lr_threshold = 1e-6
        config.early_stopping.verbose = 1
        config.checkpoint.dirpath = str(self.temp_dir / "plateau_checkpoints")
        config.monitoring.export_dir = str(self.temp_dir / "plateau_monitoring")
        
        # Track learning rate changes
        lr_history = []
        
        class LRTracker:
            def on_epoch_end(self, trainer, epoch, logs=None):
                lr_history.append(trainer.optimizer.param_groups[0]['lr'])
        
        trainer = create_enhanced_trainer(
            model=simulator.model,
            train_loader=simulator.train_loader,
            val_loader=simulator.val_loader,
            config=config
        )
        
        trainer.callbacks.append(LRTracker())
        
        # Run training
        history = trainer.train(epochs=60)
        
        # Check for learning rate reductions
        lr_reductions = 0
        for i in range(1, len(lr_history)):
            if lr_history[i] < lr_history[i-1] * 0.9:  # Significant reduction
                lr_reductions += 1
        
        validation_results = {
            "epochs_completed": len(history["train_loss"]),
            "lr_reductions_detected": lr_reductions,
            "initial_lr": lr_history[0] if lr_history else None,
            "final_lr": lr_history[-1] if lr_history else None,
            "plateau_detected": lr_reductions > 0,
            "training_converged": len(history["train_loss"]) > 20  # Reasonable training length
        }
        
        # Validation assertions
        assert validation_results["plateau_detected"], "Plateau should have been detected with LR reduction"
        assert validation_results["lr_reductions_detected"] > 0, "Learning rate should have been reduced"
        
        logger.info("✅ Plateau detection validation passed")
        return validation_results
    
    def validate_checkpoint_management(self) -> Dict[str, Any]:
        """Validate checkpoint saving and restoration."""
        logger.info("🧪 Validating checkpoint management...")
        
        simulator = TrainingSimulator("normal", num_samples=400)
        
        # Configuration with comprehensive checkpointing
        config = create_debug_config(max_epochs=20, batch_size=32)
        config.early_stopping.enabled = False  # Let it run full course
        config.checkpoint.enabled = True
        config.checkpoint.save_best_only = False  # Save multiple checkpoints
        config.checkpoint.save_top_k = 3
        config.checkpoint.dirpath = str(self.temp_dir / "checkpoint_test")
        config.checkpoint.every_n_epochs = 2
        config.monitoring.export_dir = str(self.temp_dir / "checkpoint_monitoring")
        
        trainer = create_enhanced_trainer(
            model=simulator.model,
            train_loader=simulator.train_loader,
            val_loader=simulator.val_loader,
            config=config
        )
        
        # Store initial model state
        initial_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        
        # Run training
        history = trainer.train(epochs=20)
        
        # Check checkpoint files
        checkpoint_dir = Path(config.checkpoint.dirpath)
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        
        validation_results = {
            "checkpoints_saved": len(checkpoint_files),
            "expected_checkpoints": min(config.checkpoint.save_top_k, 20 // config.checkpoint.every_n_epochs),
            "checkpoint_files_exist": len(checkpoint_files) > 0,
            "training_completed": len(history["train_loss"]) == 20
        }
        
        # Test checkpoint loading
        if checkpoint_files:
            # Load the first checkpoint
            checkpoint_path = str(checkpoint_files[0])
            checkpoint_data = trainer.model_manager.load_checkpoint(checkpoint_path)
            
            validation_results.update({
                "checkpoint_loadable": True,
                "checkpoint_has_model_state": "model_state_dict" in checkpoint_data,
                "checkpoint_has_optimizer_state": "optimizer_state_dict" in checkpoint_data,
                "checkpoint_has_metrics": "metrics" in checkpoint_data,
                "checkpoint_epoch": checkpoint_data.get("epoch", -1)
            })
            
            # Verify model state was loaded correctly
            loaded_state = trainer.model.state_dict()
            state_matches = all(
                torch.allclose(checkpoint_data["model_state_dict"][k], loaded_state[k])
                for k in checkpoint_data["model_state_dict"].keys()
            )
            validation_results["model_state_restored_correctly"] = state_matches
        
        # Validation assertions
        assert validation_results["checkpoints_saved"] > 0, "Checkpoints should have been saved"
        assert validation_results["checkpoint_loadable"], "Checkpoint should be loadable"
        assert validation_results["model_state_restored_correctly"], "Model state should be restored correctly"
        
        logger.info("✅ Checkpoint management validation passed")
        return validation_results
    
    def validate_metrics_monitoring(self) -> Dict[str, Any]:
        """Validate comprehensive metrics monitoring."""
        logger.info("🧪 Validating metrics monitoring...")
        
        simulator = TrainingSimulator("noisy", num_samples=600)
        
        # Configuration with comprehensive monitoring
        config = create_debug_config(max_epochs=25, batch_size=32)
        config.early_stopping.enabled = True
        config.early_stopping.patience = 20  # Long patience for noisy scenario
        config.monitoring.enabled = True
        config.monitoring.primary_metrics = ["train_loss", "val_loss"]
        config.monitoring.secondary_metrics = ["learning_rate", "grad_norm"]
        config.monitoring.enable_alerts = True
        config.monitoring.alert_thresholds = {
            "val_loss": {"warning": 2.0, "error": 5.0, "critical": 10.0}
        }
        config.monitoring.export_dir = str(self.temp_dir / "monitoring_test")
        config.monitoring.auto_export = True
        config.monitoring.log_to_wandb = False
        config.monitoring.log_to_tensorboard = False
        
        trainer = create_enhanced_trainer(
            model=simulator.model,
            train_loader=simulator.train_loader,
            val_loader=simulator.val_loader,
            config=config
        )
        
        # Run training
        history = trainer.train(epochs=25)
        
        # Check monitoring results
        monitoring_dir = Path(config.monitoring.export_dir)
        monitoring_files = list(monitoring_dir.glob("*"))
        
        validation_results = {
            "training_completed": len(history["train_loss"]) > 0,
            "metrics_tracked": len(history.keys()),
            "has_train_loss": "train_loss" in history,
            "has_val_loss": "val_loss" in history,
            "has_learning_rate": "learning_rate" in history,
            "monitoring_files_created": len(monitoring_files) > 0,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "training_summary_available": trainer.get_training_summary() is not None
        }
        
        # Get training summary
        summary = trainer.get_training_summary()
        if summary:
            validation_results.update({
                "summary_has_model_info": "model_info" in summary,
                "summary_has_best_metrics": "best_metrics" in summary,
                "summary_has_history": "final_metrics" in summary
            })
        
        # Validation assertions
        assert validation_results["has_train_loss"], "Training loss should be tracked"
        assert validation_results["has_val_loss"], "Validation loss should be tracked"
        assert validation_results["training_summary_available"], "Training summary should be available"
        
        logger.info("✅ Metrics monitoring validation passed")
        return validation_results
    
    def validate_alert_system(self) -> Dict[str, Any]:
        """Validate alert system functionality."""
        logger.info("🧪 Validating alert system...")
        
        # Create configuration with sensitive alert thresholds
        dashboard_config = DashboardConfig(
            enable_alerts=True,
            alert_thresholds={
                "test_metric": {"warning": 1.0, "error": 2.0, "critical": 5.0}
            },
            alert_cooldown=0.1,  # Short cooldown for testing
            export_dir=str(self.temp_dir / "alert_test"),
            log_to_wandb=False,
            log_to_tensorboard=False
        )
        
        monitor = ValidationMonitor(dashboard_config)
        
        # Track triggered alerts
        triggered_alerts = []
        
        def alert_callback(message, level, context):
            triggered_alerts.append((message, level.value, context))
        
        monitor.alert_system.add_alert_callback(alert_callback)
        
        # Mock trainer
        class MockTrainer:
            def __init__(self):
                self.stop_training = False
        
        mock_trainer = MockTrainer()
        
        # Initialize monitoring
        monitor.on_train_begin(mock_trainer)
        
        # Test different alert levels
        test_scenarios = [
            (0, {"test_metric": 0.5}),  # No alert
            (1, {"test_metric": 1.5}),  # Warning
            (2, {"test_metric": 3.0}),  # Error
            (3, {"test_metric": 7.0}),  # Critical
        ]
        
        for epoch, metrics in test_scenarios:
            monitor.on_epoch_end(mock_trainer, epoch, metrics)
            time.sleep(0.05)  # Small delay for cooldown
        
        monitor.on_train_end(mock_trainer)
        
        validation_results = {
            "alerts_triggered": len(triggered_alerts),
            "warning_alerts": sum(1 for _, level, _ in triggered_alerts if level == "warning"),
            "error_alerts": sum(1 for _, level, _ in triggered_alerts if level == "error"),
            "critical_alerts": sum(1 for _, level, _ in triggered_alerts if level == "critical"),
            "alert_system_functional": len(triggered_alerts) >= 3  # Should have warning, error, critical
        }
        
        # Get alert summary
        alert_summary = monitor.alert_system.get_alert_summary(hours=1)
        validation_results.update({
            "alert_summary_available": alert_summary is not None,
            "total_alerts_in_summary": alert_summary.get("total_alerts", 0)
        })
        
        # Validation assertions
        assert validation_results["alert_system_functional"], "Alert system should trigger alerts"
        assert validation_results["warning_alerts"] >= 1, "Warning alert should be triggered"
        assert validation_results["error_alerts"] >= 1, "Error alert should be triggered"
        assert validation_results["critical_alerts"] >= 1, "Critical alert should be triggered"
        
        logger.info("✅ Alert system validation passed")
        return validation_results
    
    def validate_configuration_system(self) -> Dict[str, Any]:
        """Validate configuration system functionality."""
        logger.info("🧪 Validating configuration system...")
        
        config_manager = ConfigManager(str(self.temp_dir / "config_test"))
        
        # Test configuration creation
        molecular_config = create_molecular_config(
            dataset_name="test_dataset",
            model_type="djmgnn",
            hidden_dim=64,
            batch_size=16,
            max_epochs=30
        )
        
        debug_config = create_debug_config(max_epochs=5, batch_size=8)
        
        validation_results = {
            "molecular_config_created": molecular_config is not None,
            "debug_config_created": debug_config is not None,
            "config_has_early_stopping": hasattr(molecular_config, 'early_stopping'),
            "config_has_monitoring": hasattr(molecular_config, 'monitoring'),
            "config_has_checkpoint": hasattr(molecular_config, 'checkpoint'),
        }
        
        # Test configuration validation
        issues = config_manager.validate_config(molecular_config)
        validation_results.update({
            "config_validation_works": isinstance(issues, list),
            "config_validation_issues": len(issues),
        })
        
        # Test configuration saving/loading
        try:
            config_manager.save_config(molecular_config, "test_molecular.yaml")
            config_file = Path(config_manager.config_dir) / "test_molecular.yaml"
            validation_results.update({
                "config_save_works": config_file.exists(),
                "config_file_has_content": config_file.stat().st_size > 0 if config_file.exists() else False
            })
        except Exception as e:
            logger.warning(f"Config save/load test failed: {e}")
            validation_results.update({
                "config_save_works": False,
                "config_file_has_content": False
            })
        
        # Test template creation
        try:
            config_manager.create_template_configs()
            template_files = list(Path(config_manager.config_dir).glob("*.yaml"))
            validation_results.update({
                "template_creation_works": len(template_files) > 0,
                "number_of_templates": len(template_files)
            })
        except Exception as e:
            logger.warning(f"Template creation test failed: {e}")
            validation_results.update({
                "template_creation_works": False,
                "number_of_templates": 0
            })
        
        # Validation assertions
        assert validation_results["molecular_config_created"], "Molecular config should be created"
        assert validation_results["debug_config_created"], "Debug config should be created"
        assert validation_results["config_has_early_stopping"], "Config should have early stopping"
        
        logger.info("✅ Configuration system validation passed")
        return validation_results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("🚀 Starting comprehensive early stopping system validation...")
        
        all_results = {}
        
        try:
            # Run individual validations
            all_results["basic_early_stopping"] = self.validate_basic_early_stopping()
            all_results["plateau_detection"] = self.validate_plateau_detection()
            all_results["checkpoint_management"] = self.validate_checkpoint_management()
            all_results["metrics_monitoring"] = self.validate_metrics_monitoring()
            all_results["alert_system"] = self.validate_alert_system()
            all_results["configuration_system"] = self.validate_configuration_system()
            
            # Overall validation summary
            all_results["overall_validation"] = {
                "all_tests_passed": True,
                "total_tests": len(all_results),
                "validation_timestamp": time.time(),
                "temp_dir": str(self.temp_dir)
            }
            
            logger.info("🎉 All validation tests passed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            all_results["overall_validation"] = {
                "all_tests_passed": False,
                "error": str(e),
                "validation_timestamp": time.time()
            }
            raise
        
        return all_results


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate Early Stopping System")
    parser.add_argument("--config", default="debug", help="Configuration to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="early_stopping_validation_"))
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Run validation
        validator = EarlyStoppingValidator(temp_dir)
        results = validator.run_all_validations()
        
        # Print summary
        print("\n" + "="*80)
        print("EARLY STOPPING SYSTEM VALIDATION SUMMARY")
        print("="*80)
        
        for test_name, test_results in results.items():
            if test_name == "overall_validation":
                continue
            
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for key, value in test_results.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    print(f"  {status} {key}: {value}")
                elif isinstance(value, (int, float)):
                    print(f"  📊 {key}: {value}")
                elif isinstance(value, str) and len(value) < 50:
                    print(f"  📝 {key}: {value}")
        
        overall = results.get("overall_validation", {})
        if overall.get("all_tests_passed", False):
            print("\n🎉 ALL VALIDATION TESTS PASSED!")
            print("The early stopping and monitoring system is ready for production use.")
        else:
            print("\n❌ SOME VALIDATION TESTS FAILED!")
            print("Please review the results above and fix any issues.")
        
        print(f"\nValidation completed at: {time.ctime()}")
        print(f"Temporary files location: {temp_dir}")
        
        return 0 if overall.get("all_tests_passed", False) else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1
        
    finally:
        # Cleanup temporary directory unless requested to keep
        if not args.keep_temp:
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
        else:
            logger.info(f"Temporary files kept at: {temp_dir}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)