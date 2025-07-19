"""
tests/test_mgnn_callbacks.py

Unit tests for the training callbacks in moml.models.mgnn.training.callbacks.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Generator, List

import pytest
import torch
import torch.nn as nn
from unittest.mock import ANY, MagicMock, patch

from moml.models.mgnn.training.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)


class MockModel(nn.Module):
    """
    A mock PyTorch model for testing callbacks.
    """

    def __init__(self) -> None:
        """
        Initializes the MockModel with a linear layer and a state dictionary.
        """
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self._state_dict_storage = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mock model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.linear(x)

    def state_dict(self, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns the state dictionary of the model.

        Args:
            destination (Any, optional): If provided, the state_dict will be saved into it. Defaults to None.
            prefix (str, optional): Prefix for parameter names. Defaults to "".
            keep_vars (bool, optional): If True, the variables are kept. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: The state dictionary.
        """
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self._state_dict_storage.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """
        Loads a state dictionary into the model.

        Args:
            state_dict (Dict[str, torch.Tensor]): A dictionary containing parameters and persistent buffers.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match
                                     the keys returned by this module's state_dict() function. Defaults to True.
        """
        self._state_dict_storage = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


@pytest.fixture
def mock_trainer() -> MagicMock:
    """
    Provides a mock trainer instance for testing callbacks.

    Returns:
        MagicMock: A mock object simulating a trainer.
    """
    trainer = MagicMock()
    trainer.model = MockModel()

    trainer.optimizer = MagicMock(spec=torch.optim.Adam)
    trainer.optimizer.param_groups = [{"lr": 0.01}]
    trainer.optimizer.state_dict.return_value = {"opt_state_key": "opt_state_val"}

    trainer.stop_training = False
    trainer.config = {"model_name": "test_model", "version": "v1"}
    return trainer


@pytest.fixture(scope="module")
def temp_checkpoint_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for checkpoint files.

    Yields:
        str: Path to the temporary directory.
    """
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


class TestEarlyStopping:
    """
    Test suite for the EarlyStopping callback.
    """

    def test_initialization(self) -> None:
        """
        Test EarlyStopping initialization with various parameters.
        """
        es = EarlyStopping(monitor="val_acc", patience=5, min_delta=0.01, mode="max", verbose=False)
        assert es.monitor == "val_acc"
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "max"
        assert not es.verbose
        assert es.best_value == float("-inf")

    def test_on_train_begin(self, mock_trainer: MagicMock) -> None:
        """
        Test on_train_begin method resets state.
        """
        es = EarlyStopping(mode="min")
        es.wait = 5
        es.best_value = 10.0
        es.on_train_begin(mock_trainer)
        assert es.wait == 0
        assert es.stopped_epoch == 0
        assert es.best_value == float("inf")
        assert es.best_weights is None

    def test_improvement_mode_min(self, mock_trainer: MagicMock) -> None:
        """
        Test on_epoch_end with improvement in 'min' mode.
        """
        es = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.01)
        es.on_train_begin(mock_trainer)

        logs1 = {"val_loss": 0.5}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([1.0])}
        es.on_epoch_end(mock_trainer, epoch=1, logs=logs1)
        assert es.best_value == 0.5
        assert es.wait == 0
        assert es.best_weights is not None
        if es.best_weights is not None:
            assert torch.equal(es.best_weights["linear.weight"], torch.tensor([1.0]))
        assert not mock_trainer.stop_training

        logs2 = {"val_loss": 0.4}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([2.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs=logs2)
        assert es.best_value == 0.4
        assert es.wait == 0
        if es.best_weights is not None:
            assert torch.equal(es.best_weights["linear.weight"], torch.tensor([2.0]))
        assert not mock_trainer.stop_training

    def test_no_improvement_patience_met_mode_min(self, mock_trainer: MagicMock) -> None:
        """
        Test on_epoch_end stops training when patience is met in 'min' mode.
        """
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
        if mock_trainer.model._state_dict_storage is not None:
            assert torch.equal(mock_trainer.model._state_dict_storage["linear.weight"], initial_weights["linear.weight"])

    def test_improvement_mode_max(self, mock_trainer: MagicMock) -> None:
        """
        Test on_epoch_end with improvement in 'max' mode.
        """
        es = EarlyStopping(monitor="val_acc", mode="max", patience=3, min_delta=0.01)
        es.on_train_begin(mock_trainer)

        logs1 = {"val_acc": 0.8}
        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([1.0])}
        es.on_epoch_end(mock_trainer, epoch=1, logs=logs1)
        assert es.best_value == 0.8
        assert es.wait == 0
        if es.best_weights is not None:
            assert torch.equal(es.best_weights["linear.weight"], torch.tensor([1.0]))

    def test_monitor_not_in_logs(self, mock_trainer: MagicMock, capsys: Any) -> None:
        """
        Test on_epoch_end handles missing monitored metric.
        """
        es = EarlyStopping(monitor="missing_metric", verbose=True)
        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Early stopping monitored metric 'missing_metric' not available" in captured.out
        assert es.wait == 0

    def test_no_best_weights_to_restore(self, mock_trainer: MagicMock) -> None:
        """
        Test on_epoch_end when no best weights are available to restore.
        """
        es = EarlyStopping(monitor="val_loss", patience=1)
        es.on_train_begin(mock_trainer)
        assert es.best_weights is None

        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 1.0})
        assert not mock_trainer.stop_training

    def test_min_delta_effect_mode_min(self, mock_trainer: MagicMock) -> None:
        """
        Test min_delta threshold in 'min' mode.
        """
        es = EarlyStopping(monitor="val_loss", mode="min", patience=1, min_delta=0.1)
        es.on_train_begin(mock_trainer)
        es.best_value = 0.5

        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.45})
        assert es.wait == 1
        assert es.best_value == 0.5
        assert mock_trainer.stop_training

        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([3.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs={"val_loss": 0.39})
        assert es.wait == 0
        assert es.best_value == 0.39
        if es.best_weights is not None:
            assert torch.equal(es.best_weights["linear.weight"], torch.tensor([3.0]))

    def test_min_delta_effect_mode_max(self, mock_trainer: MagicMock) -> None:
        """
        Test min_delta threshold in 'max' mode.
        """
        es = EarlyStopping(monitor="val_acc", mode="max", patience=1, min_delta=0.1)
        es.on_train_begin(mock_trainer)
        es.best_value = 0.5

        es.on_epoch_end(mock_trainer, epoch=1, logs={"val_acc": 0.55})
        assert es.wait == 1
        assert es.best_value == 0.5
        assert mock_trainer.stop_training

        mock_trainer.model._state_dict_storage = {"linear.weight": torch.tensor([4.0])}
        es.on_epoch_end(mock_trainer, epoch=2, logs={"val_acc": 0.61})
        assert es.wait == 0
        assert es.best_value == 0.61
        if es.best_weights is not None:
            assert torch.equal(es.best_weights["linear.weight"], torch.tensor([4.0]))


class TestModelCheckpoint:
    """
    Test suite for the ModelCheckpoint callback.
    """

    def test_initialization(self, temp_checkpoint_dir: str) -> None:
        """
        Test ModelCheckpoint initialization.
        """
        filepath = os.path.join(temp_checkpoint_dir, "model.pt")
        mc = ModelCheckpoint(filepath, monitor="val_acc", save_best_only=False, mode="max", period=2, verbose=False)
        assert mc.filepath == filepath
        assert mc.monitor == "val_acc"
        assert not mc.save_best_only
        assert mc.mode == "max"
        assert mc.period == 2

    def test_on_train_begin(self, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test on_train_begin method creates output directory and resets state.
        """
        filepath = os.path.join(temp_checkpoint_dir, "model_begin.pt")
        mc = ModelCheckpoint(filepath)
        mc.best_value = 0.1
        mc.on_train_begin(mock_trainer)
        assert mc.best_value == float("inf")
        assert mc.epochs_since_last_save == 0
        assert os.path.exists(os.path.dirname(filepath))

    @patch("torch.save")
    def test_save_best_only_improvement(self, mock_torch_save: MagicMock, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test saving checkpoint when there is an improvement in 'save_best_only' mode.
        """
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
    def test_save_best_only_no_improvement(self, mock_torch_save: MagicMock, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test that no checkpoint is saved when there is no improvement in 'save_best_only' mode.
        """
        filepath = os.path.join(temp_checkpoint_dir, "no_improve.pt")
        mc = ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, mode="min")
        mc.on_train_begin(mock_trainer)
        mc.best_value = 0.3

        logs = {"val_loss": 0.5}
        mc.on_epoch_end(mock_trainer, epoch=1, logs=logs)
        mock_torch_save.assert_not_called()

    @patch("torch.save")
    def test_save_periodically(self, mock_torch_save: MagicMock, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test saving checkpoints periodically.
        """
        filepath_template = os.path.join(temp_checkpoint_dir, "epoch_{epoch:02d}_{val_loss:.2f}.pt")
        mc = ModelCheckpoint(filepath_template, save_best_only=False, period=2, verbose=False)
        mc.on_train_begin(mock_trainer)

        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        mock_torch_save.assert_not_called()
        assert mc.epochs_since_last_save == 1

        mc.on_epoch_end(mock_trainer, epoch=2, logs={"val_loss": 0.4})
        expected_filepath_e2 = filepath_template.format(epoch=2, val_loss=0.4)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath_e2)
        assert mc.epochs_since_last_save == 0

        mock_torch_save.reset_mock()
        mc.on_epoch_end(mock_trainer, epoch=3, logs={"val_loss": 0.3})
        mock_torch_save.assert_not_called()
        assert mc.epochs_since_last_save == 1

        mc.on_epoch_end(mock_trainer, epoch=4, logs={"val_loss": 0.2})
        expected_filepath_e4 = filepath_template.format(epoch=4, val_loss=0.2)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath_e4)

    def test_monitor_not_in_logs_save_best_only(self, temp_checkpoint_dir: str, mock_trainer: MagicMock, capsys: Any) -> None:
        """
        Test handling of missing monitored metric in 'save_best_only' mode.
        """
        filepath = os.path.join(temp_checkpoint_dir, "monitor_missing.pt")
        mc = ModelCheckpoint(filepath, monitor="missing_metric", save_best_only=True, verbose=True)
        mc.on_train_begin(mock_trainer)
        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Monitored metric 'missing_metric' not available" in captured.out
        assert not os.path.exists(filepath)

    @patch("torch.save")
    def test_save_optimizer_false(self, mock_torch_save: MagicMock, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test that optimizer state is not saved when save_optimizer is False.
        """
        filepath = os.path.join(temp_checkpoint_dir, "no_opt_model.pt")
        mc = ModelCheckpoint(filepath, save_best_only=False, period=1, save_optimizer=False)
        mc.on_train_begin(mock_trainer)
        mc.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})

        mock_torch_save.assert_called_once()
        saved_checkpoint = mock_torch_save.call_args[0][0]
        assert "optimizer_state_dict" not in saved_checkpoint

    @patch("torch.save")
    def test_filepath_formatting_with_logs(self, mock_torch_save: MagicMock, temp_checkpoint_dir: str, mock_trainer: MagicMock) -> None:
        """
        Test filepath formatting with log values.
        """
        filepath_template = os.path.join(temp_checkpoint_dir, "model_ep{epoch:03d}_loss{val_loss:.4f}.pt")
        mc = ModelCheckpoint(filepath_template, save_best_only=False, period=1)
        mc.on_train_begin(mock_trainer)
        logs = {"val_loss": 0.12345, "other_metric": 0.987}
        mc.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        expected_filepath = filepath_template.format(epoch=5, **logs)
        mock_torch_save.assert_called_once_with(ANY, expected_filepath)


class TestLearningRateScheduler:
    """
    Test suite for the LearningRateScheduler callback.
    """

    def test_init_callable_schedule(self) -> None:
        """
        Test initialization with a callable schedule.
        """

        def my_schedule(epoch: int) -> float:
            return 0.001 / (epoch + 1)

        lrs = LearningRateScheduler(schedule=my_schedule)
        assert lrs.is_callable
        assert lrs.schedule == my_schedule

    def test_init_reduce_on_plateau(self) -> None:
        """
        Test initialization with 'reduce_on_plateau' schedule.
        """
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss")  # type: ignore
        assert not lrs.is_callable
        assert lrs.monitor == "val_loss"

    def test_init_reduce_on_plateau_no_monitor(self) -> None:
        """
        Test initialization with 'reduce_on_plateau' without a monitor raises error.
        """
        with pytest.raises(ValueError, match="monitor must be specified"):
            LearningRateScheduler(schedule="reduce_on_plateau")  # type: ignore

    def test_callable_schedule_on_epoch_end(self, mock_trainer: MagicMock) -> None:
        """
        Test callable schedule updates learning rate on epoch end.
        """
        new_lrs_tracker: List[float] = []

        def my_schedule(epoch: int) -> float:
            lr_val = 0.01 / (epoch + 1)
            new_lrs_tracker.append(lr_val)
            return lr_val

        lrs = LearningRateScheduler(schedule=my_schedule, verbose=False)
        lrs.on_train_begin(mock_trainer)

        lrs.on_epoch_end(mock_trainer, epoch=0, logs={})
        assert mock_trainer.optimizer.param_groups[0]["lr"] == new_lrs_tracker[0]

        lrs.on_epoch_end(mock_trainer, epoch=1, logs={})
        assert mock_trainer.optimizer.param_groups[0]["lr"] == new_lrs_tracker[1]

    def test_callable_schedule_verbose_output(self, mock_trainer: MagicMock, capsys: Any) -> None:
        """
        Test verbose output for callable schedule.
        """

        def my_schedule(epoch: int) -> float:
            return 0.005

        lrs = LearningRateScheduler(schedule=my_schedule, verbose=True)
        lrs.on_epoch_end(mock_trainer, epoch=3, logs={})
        captured = capsys.readouterr()
        assert "Epoch 3: Learning rate set to 0.005" in captured.out

    @patch("torch.optim.lr_scheduler.ReduceLROnPlateau")
    def test_reduce_on_plateau_on_train_begin(self, mock_reduce_lr: MagicMock, mock_trainer: MagicMock) -> None:
        """
        Test ReduceLROnPlateau scheduler is initialized on train begin.
        """
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss", factor=0.5, patience=5)  # type: ignore
        lrs.on_train_begin(mock_trainer)
        mock_reduce_lr.assert_called_once_with(
            mock_trainer.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            cooldown=0,
            min_lr=0,
        )
        assert lrs.scheduler == mock_reduce_lr.return_value

    def test_reduce_on_plateau_on_epoch_end(self, mock_trainer: MagicMock) -> None:
        """
        Test ReduceLROnPlateau scheduler steps on epoch end.
        """
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="val_loss", verbose=False)  # type: ignore
        lrs.on_train_begin(mock_trainer)

        lrs.scheduler.step = MagicMock()

        logs = {"val_loss": 0.5}
        lrs.on_epoch_end(mock_trainer, epoch=1, logs=logs)
        lrs.scheduler.step.assert_called_once_with(0.5)

    def test_reduce_on_plateau_monitor_missing(self, mock_trainer: MagicMock, capsys: Any) -> None:
        """
        Test ReduceLROnPlateau handles missing monitored metric.
        """
        lrs = LearningRateScheduler(schedule="reduce_on_plateau", monitor="missing_metric", verbose=True)  # type: ignore
        lrs.on_train_begin(mock_trainer)
        lrs.scheduler.step = MagicMock()

        lrs.on_epoch_end(mock_trainer, epoch=1, logs={"val_loss": 0.5})
        captured = capsys.readouterr()
        assert "Warning: Monitored metric 'missing_metric' not available" in captured.out
        lrs.scheduler.step.assert_not_called()
