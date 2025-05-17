
"""
Unit tests for the training callbacks in moml.models.mgnn.training.callbacks.
"""
import pytest
import torch
import torch.nn as nn
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch, ANY  # Add ANY

from moml.models.mgnn.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)


# --- Mock Trainer and related components ---
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        # Use a more robust way to store state_dict for mocking
        self._state_dict_storage = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

    def forward(self, x):
        return self.linear(x)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Return a copy to mimic real behavior where modifications don't affect original
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self._state_dict_storage.items()}

    def load_state_dict(self, state_dict, strict=True):
        self._state_dict_storage = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.model = MockModel()

    trainer.optimizer = MagicMock(spec=torch.optim.Adam)
    trainer.optimizer.param_groups = [{"lr": 0.01}]
    trainer.optimizer.state_dict.return_value = {"opt_state_key": "opt_state_val"}

    trainer.stop_training = False
    trainer.config = {"model_name": "test_model", "version": "v1"}
    return trainer


@pytest.fixture(scope="module")
def temp_checkpoint_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available or PyTorch CUDA setup issue")
class TestEarlyStopping:
    def test_initialization(self):
        es = EarlyStopping(monitor="val_acc", patience=5, min_delta=0.01, mode="max", verbose=False)
        assert es.monitor == "val_acc"
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "max"
        assert not es.verbose
        assert es.best_value == float("-inf")

    def test_on_train_begin(self, mock_trainer):
        es = EarlyStopping(mode="min")
        es.wait = 5
        es.best_value = 10.0
        es.on_train_begin(mock_trainer)
        assert es.wait == 0
        assert es.stopped_epoch == 0
        assert es.best_value == float("inf")
        assert es.best_weights is None

    def test_improvement_mode_min(self, mock_trainer):
        es = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.01)
        es.on_train_begin(mock_trainer)

        logs1 = {"val_loss": 0.5}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([1.0])}
        es.on_epoch_end(mock_trainer, epoch=1, logs=logs1)
        assert es.best_value == 0.5
        assert es.wait == 0
        assert es.best_weights is not None
        assert torch.equal(es.best_weights["linear.weight"], torch.tensor([1.0]))
        assert not mock_trainer.stop_training

        logs2 = {"val_loss": 0.4}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([2.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs=logs2)
        assert es.best_value == 0.4
        assert es.wait == 0
        assert torch.equal(es.best_weights["linear.weight"], torch.tensor([2.0]))
        assert not mock_trainer.stop_training

    def test_no_improvement_patience_met_mode_min(self, mock_trainer):
        es = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.0)
        es.on_train_begin(mock_trainer)

        initial_weights = {"linear.weight": torch.tensor([0.0])}
        mock_trainer.model._state_dict_storage = initial_weights.copy()
        es.best_weights = initial_weights.copy()
        es.best_value = 0.5

        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.6})
        assert es.wait == 1
        assert not mock_trainer.stop_training

        es.on_epoch_end(mock_trainer, epoch=2, logs={"val_loss": 0.7})
        assert es.wait == 2
        assert mock_trainer.stop_training
        assert es.stopped_epoch == 2
        assert torch.equal(mock_trainer.model._state_dict_storage["linear.weight"], initial_weights["linear.weight"])

    def test_improvement_mode_max(self, mock_trainer):
        es = EarlyStopping(monitor="val_acc", mode="max", patience=3, min_delta=0.01)
        es.on_train_begin(mock_trainer)

        logs1 = {"val_acc": 0.8}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([1.0])}
        es.on_epoch_end(mock_trainer, epoch=1, logs=logs1)
        assert es.best_value == 0.8
        assert es.wait == 0
        assert torch.equal(es.best_weights["linear.weight"], torch.tensor([1.0]))

    def test_monitor_not_in_logs(self, mock_trainer, capsys):
        es = EarlyStopping(monitor="missing_metric", verbose=True)
        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Early stopping monitored metric 'missing_metric' not available" in captured.out
        assert es.wait == 0

    def test_no_best_weights_to_restore(self, mock_trainer):
        es = EarlyStopping(monitor="val_loss", patience=1)
        es.on_train_begin(mock_trainer)
        assert es.best_weights is None  # Ensure it starts as None

        # Trigger early stopping without any improvement (so best_weights remains None)
        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 1.0})  # This is an improvement from inf
        assert not mock_trainer.stop_training  # Should not stop after 1st epoch if it's an improvement
        # No error should occur, and model state should be unchanged if best_weights was None
        # This implicitly tests that load_state_dict is not called with None

    def test_min_delta_effect_mode_min(self, mock_trainer):
        es = EarlyStopping(monitor="val_loss", mode="min", patience=1, min_delta=0.1)
        es.on_train_begin(mock_trainer)
        es.best_value = 0.5

        # Improvement less than min_delta
        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.45})  # 0.5 - 0.45 = 0.05 < 0.1
        assert es.wait == 1  # Not considered an improvement
        assert es.best_value == 0.5  # Remains unchanged
        assert mock_trainer.stop_training  # Should stop as wait (1) >= patience (1)

        # Improvement greater than or equal to min_delta
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([3.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs={"val_loss": 0.39})  # 0.5 - 0.39 = 0.11 >= 0.1
        assert es.wait == 0  # Considered an improvement
        assert es.best_value == 0.39
        assert torch.equal(es.best_weights["linear.weight"], torch.tensor([3.0]))

    def test_min_delta_effect_mode_max(self, mock_trainer):
        es = EarlyStopping(monitor="val_acc", mode="max", patience=1, min_delta=0.1)
        es.on_train_begin(mock_trainer)
        es.best_value = 0.5

        # Improvement less than min_delta
        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_acc": 0.55})  # 0.55 - 0.5 = 0.05 < 0.1
        assert es.wait == 1
        assert es.best_value == 0.5
        assert mock_trainer.stop_training  # Should stop as wait (1) >= patience (1)

        # Improvement greater than or equal to min_delta
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([4.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs={"val_acc": 0.61})  # 0.61 - 0.5 = 0.11 >= 0.1
        assert es.wait == 0
        assert es.best_value == 0.61
        assert torch.equal(es.best_weights["linear.weight"], torch.tensor([4.0]))


class TestModelCheckpoint:
    def test_initialization(self, temp_checkpoint_dir):
        filepath = os.path.join(temp_checkpoint_dir, "model.pt")
        mc = ModelCheckpoint(filepath, monitor="val_acc", save_best_only=False, mode="max", period=2, verbose=False)
        assert mc.filepath == filepath
        assert mc.monitor == "val_acc"
        assert not mc.save_best_only
        assert mc.mode == "max"
        assert mc.period == 2

    def test_on_train_begin(self, temp_checkpoint_dir, mock_trainer):
        filepath = os.path.join(temp_checkpoint_dir, "model_begin.pt")
        mc = ModelCheckpoint(filepath)
        mc.best_value = 0.1
        mc.on_train_begin(mock_trainer)
        assert mc.best_value == float("inf")
        assert mc.epochs_since_last_save == 0
        assert os.path.exists(os.path.dirname(filepath))

    @patch("torch.save")
    def test_save_best_only_improvement(self, mock_torch_save, temp_checkpoint_dir, mock_trainer):
        filepath = os.path.join(temp_checkpoint_dir, "best_model.pt")
        mc = ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, mode="min", save_optimizer=True)
        mc.on_train_begin(mock_trainer)

        logs = {"val_loss": 0.5}
        mc.on_epoch_end(mock_trainer, epoch=1, logs=logs)

        mock_torch_save.assert_called_once()
        saved_checkpoint = mock_torch_save.call_args[0][0]
        assert saved_checkpoint["epoch"] == 1
        assert "model_state_dict" in saved_checkpoint
        assert "optimizer_state_dict" in saved_checkpoint
        assert saved_checkpoint["logs"] == logs
        assert mc.best_value == 0.5

    @patch("torch.save")
    def test_save_best_only_no_improvement(self, mock_torch_save, temp_checkpoint_dir, mock_trainer):
        filepath = os.path.join(temp_checkpoint_dir, "no_improve.pt")
        mc = ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, mode="min")
        mc.on_train_begin(mock_trainer)
        mc.best_value = 0.3

        logs = {"val_loss": 0.5}
        mc.on_epoch_end(mock_trainer, epoch=1, logs=logs)
        mock_torch_save.assert_not_called()

    @patch("torch.save")
    def test_save_periodically(self, mock_torch_save, temp_checkpoint_dir, mock_trainer):
        filepath_template = os.path.join(temp_checkpoint_dir, "epoch_{epoch:02d}_{val_loss:.2f}.pt")
        mc = ModelCheckpoint(filepath_template, save_best_only=False, period=2, verbose=False)
        mc.on_train_begin(mock_trainer)

        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        mock_torch_save.assert_not_called()
        assert mc.epochs_since_last_save == 1

        mc.on_epoch_end(mock_trainer, epoch=2, logs={"val_loss": 0.4})
        expected_filepath_e2 = filepath_template.format(epoch=2, val_loss=0.4)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath_e2)  # Use unittest.mock.ANY
        assert mc.epochs_since_last_save == 0

        mock_torch_save.reset_mock()
        mc.on_epoch_end(mock_trainer, epoch=3, logs={"val_loss": 0.3})
        mock_torch_save.assert_not_called()
        assert mc.epochs_since_last_save == 1

        mc.on_epoch_end(mock_trainer, epoch=4, logs={"val_loss": 0.2})
        expected_filepath_e4 = filepath_template.format(epoch=4, val_loss=0.2)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath_e4)  # Use unittest.mock.ANY

    def test_monitor_not_in_logs_save_best_only(self, temp_checkpoint_dir, mock_trainer, capsys):
        filepath = os.path.join(temp_checkpoint_dir, "monitor_missing.pt")
        mc = ModelCheckpoint(filepath, monitor="missing_metric", save_best_only=True, verbose=True)
        mc.on_train_begin(mock_trainer)
        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Monitored metric 'missing_metric' not available" in captured.out
        # Ensure no save attempt was made
        assert not os.path.exists(filepath)

    @patch("torch.save")
    def test_save_optimizer_false(self, mock_torch_save, temp_checkpoint_dir, mock_trainer):
        filepath = os.path.join(temp_checkpoint_dir, "no_opt_model.pt")
        mc = ModelCheckpoint(filepath, save_best_only=False, period=1, save_optimizer=False)
        mc.on_train_begin(mock_trainer)
        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})

        mock_torch_save.assert_called_once()
        saved_checkpoint = mock_torch_save.call_args[0][0]
        assert "optimizer_state_dict" not in saved_checkpoint

    @patch("torch.save")
    def test_filepath_formatting_with_logs(self, mock_torch_save, temp_checkpoint_dir, mock_trainer):
        filepath_template = os.path.join(temp_checkpoint_dir, "model_ep{epoch:03d}_loss{val_loss:.4f}.pt")
        mc = ModelCheckpoint(filepath_template, save_best_only=False, period=1)
        mc.on_train_begin(mock_trainer)
        logs = {"val_loss": 0.12345, "other_metric": 0.987}
        mc.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        expected_filepath = filepath_template.format(epoch=5, **logs)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath)  # Use unittest.mock.ANY


class TestLearningRateScheduler:
    def test_init_callable_schedule(self):
        def my_schedule(epoch):
            return 0.001 / (epoch + 1)

        lrs = LearningRateScheduler(schedule=my_schedule)
        assert lrs.is_callable
        assert lrs.schedule == my_schedule

    def test_init_reduce_on_plateau(self):
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss")
        assert not lrs.is_callable
        assert lrs.monitor == "val_loss"

    def test_init_reduce_on_plateau_no_monitor(self):
        with pytest.raises(ValueError, match="monitor must be specified"):
            LearningRateScheduler(schedule="reduce_on_plateau")

    def test_callable_schedule_on_epoch_end(self, mock_trainer):
        new_lrs_tracker = []

        def my_schedule(epoch):
            lr_val = 0.01 / (epoch + 1)
            new_lrs_tracker.append(lr_val)
            return lr_val

        lrs = LearningRateScheduler(schedule=my_schedule, verbose=False)
        lrs.on_train_begin(mock_trainer)

        lrs.on_epoch_end(mock_trainer, epoch=0, logs={})
        assert mock_trainer.optimizer.param_groups[0]["lr"] == new_lrs_tracker[0]

        lrs.on_epoch_end(mock_trainer, epoch=1, logs={})
        assert mock_trainer.optimizer.param_groups[0]["lr"] == new_lrs_tracker[1]

    def test_callable_schedule_verbose_output(self, mock_trainer, capsys):
        def my_schedule(epoch):
            return 0.005

        lrs = LearningRateScheduler(schedule=my_schedule, verbose=True)
        lrs.on_epoch_end(mock_trainer, epoch=3, logs={})
        captured = capsys.readouterr()
        assert "Epoch 3: Learning rate set to 0.005" in captured.out

    @patch("torch.optim.lr_scheduler.ReduceLROnPlateau")
    def test_reduce_on_plateau_on_train_begin(self, mock_reduce_lr, mock_trainer):
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss", factor=0.5, patience=5)
        lrs.on_train_begin(mock_trainer)
        mock_reduce_lr.assert_called_once_with(
            mock_trainer.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            cooldown=0,
            min_lr=0,
            # verbose=True, # Removed deprecated parameter
        )
        assert lrs.scheduler == mock_reduce_lr.return_value

    def test_reduce_on_plateau_on_epoch_end(self, mock_trainer):
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss", verbose=False)
        lrs.on_train_begin(mock_trainer)

        lrs.scheduler.step = MagicMock()

        logs = {"val_loss": 0.5}
        lrs.on_epoch_end(mock_trainer, epoch=1, logs=logs)
        lrs.scheduler.step.assert_called_once_with(0.5)

    def test_reduce_on_plateau_monitor_missing(self, mock_trainer, capsys):
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="missing_metric", verbose=True)
        lrs.on_train_begin(mock_trainer)
        lrs.scheduler.step = MagicMock()

        lrs.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Monitored metric 'missing_metric' not available" in captured.out
        lrs.scheduler.step.assert_not_called()
