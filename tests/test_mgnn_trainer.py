"""
tests/test_mgnn_trainer.py

Unit tests for the MGNNTrainer class and related functions
in moml.models.mgnn.training.trainer.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch_geometric.data import Batch, Data

from moml.models.mgnn.evaluation.predictor import MGNNPredictor
from moml.models.mgnn.training.callbacks import Callback
from moml.models.mgnn.training.trainer import (
    MGNNTrainer,
    create_trainer,
    train_epoch as standalone_train_epoch,
)

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")


class MockSimpleModel(nn.Module):
    """
    A simple mock PyTorch model for testing the trainer.
    """

    def __init__(self, in_features: int = 5, out_features_node: int = 1, out_features_graph: int = 1) -> None:
        """
        Initializes the MockSimpleModel.

        Args:
            in_features (int, optional): Input feature dimension. Defaults to 5.
            out_features_node (int, optional): Output dimension for node predictions. Defaults to 1.
            out_features_graph (int, optional): Output dimension for graph predictions. Defaults to 1.
        """
        super().__init__()
        self.linear_node = nn.Linear(in_features, out_features_node)
        self.linear_graph = nn.Linear(in_features, out_features_graph)
        self.out_features_node = out_features_node
        self.out_features_graph = out_features_graph

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the mock model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge connectivity.
            edge_attr (Optional[torch.Tensor], optional): Edge features. Defaults to None.
            batch (Optional[torch.Tensor], optional): Batch assignment vector. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing node and graph predictions.
        """
        node_pred = self.linear_node(x)

        if batch is None:
            if x.numel() == 0:
                graph_x = torch.zeros(
                    (0, x.size(1) if x.dim() > 1 else self.linear_graph.in_features), device=x.device
                )
            else:
                graph_x = x.mean(dim=0, keepdim=True)

        else:
            graph_x_list = []
            if x.numel() > 0:
                for i in range(batch.max().item() + 1):
                    graph_x_list.append(x[batch == i].mean(dim=0))

            if not graph_x_list:
                graph_x = torch.zeros(
                    (0, x.size(1) if x.dim() > 1 and x.size(1) > 0 else self.linear_graph.in_features), device=x.device
                )
            else:
                graph_x = torch.stack(graph_x_list)

        if graph_x.numel() == 0 and self.out_features_graph > 0:
            graph_pred = torch.zeros((0, self.out_features_graph), device=x.device)
        elif self.out_features_graph == 0:
            graph_pred = torch.empty((graph_x.shape[0], 0), device=x.device)
        else:
            graph_pred = self.linear_graph(graph_x)

        return {"node_pred": node_pred, "graph_pred": graph_pred}


class MockPyGDataset(TorchDataset):
    """
    A mock PyTorch Geometric Dataset for testing.
    """

    def __init__(
        self,
        num_samples: int = 10,
        in_features: int = 5,
        num_nodes: int = 3,
        node_out_dim: int = 1,
        graph_out_dim: int = 1,
        is_empty: bool = False,
    ) -> None:
        """
        Initializes the MockPyGDataset.

        Args:
            num_samples (int, optional): Number of samples in the dataset. Defaults to 10.
            in_features (int, optional): Input feature dimension for nodes. Defaults to 5.
            num_nodes (int, optional): Number of nodes per graph. Defaults to 3.
            node_out_dim (int, optional): Output dimension for node-level targets. Defaults to 1.
            graph_out_dim (int, optional): Output dimension for graph-level targets. Defaults to 1.
            is_empty (bool, optional): If True, creates an empty dataset. Defaults to False.
        """
        self.num_samples = 0 if is_empty else num_samples
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.node_out_dim = node_out_dim
        self.graph_out_dim = graph_out_dim

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a single data sample (graph).

        Args:
            idx (int): Index of the sample.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        if self.num_nodes == 0:
            x = torch.empty(0, self.in_features)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            if self.graph_out_dim > 0:
                y_graph = (
                    torch.randn(int(self.graph_out_dim)).unsqueeze(0)
                    if self.graph_out_dim == 1
                    else torch.randn(int(self.graph_out_dim))
                )
            else:
                y_graph = torch.empty(0)
            y_node = torch.empty(0, self.node_out_dim)
        else:
            x = torch.randn(self.num_nodes, self.in_features)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long) % self.num_nodes
            edge_index = edge_index[:, edge_index.max() < self.num_nodes]
            if edge_index.nelement() == 0 and self.num_nodes > 1:
                edge_index = torch.tensor([[0], [1]], dtype=torch.long) % self.num_nodes
            elif self.num_nodes == 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            if self.graph_out_dim > 0:
                y_g = torch.randn(int(self.graph_out_dim))
                y_graph = y_g.unsqueeze(0) if self.graph_out_dim == 1 and y_g.ndim == 0 else y_g
                if y_graph.ndim == 1 and self.graph_out_dim == 1:
                    y_graph = y_graph.unsqueeze(0)

            else:
                y_graph = torch.empty(0)

            if self.node_out_dim > 0:
                y_n = torch.randn(self.num_nodes, int(self.node_out_dim))
                y_node = y_n
            else:
                y_node = torch.empty(self.num_nodes, 0)

        return Data(x=x, edge_index=edge_index, y=y_graph, node_y=y_node)


def mock_collate_fn(batch_list: List[Data]) -> Batch:
    """
    A mock collate function for PyTorch Geometric Data objects.

    Args:
        batch_list (List[Data]): List of Data objects.

    Returns:
        Batch: A PyTorch Geometric Batch object.
    """
    return Batch.from_data_list(batch_list)  # type: ignore


@pytest.fixture
def dummy_config() -> Dict[str, Any]:
    """
    Provides a dummy configuration dictionary for the trainer.
    """
    return {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "task_type": "regression",
        "epochs": 3,
        "device": "cpu",
        "in_dim": 5,
        "hidden_dim": 10,
        "edge_attr_dim": 0,
        "node_out_dim": 1,
        "graph_out_dim": 1,
    }


@pytest.fixture
def mock_model(dummy_config: Dict[str, Any]) -> MockSimpleModel:
    """
    Provides a mock model instance.
    """
    return MockSimpleModel(
        in_features=dummy_config["in_dim"],
        out_features_node=dummy_config["node_out_dim"],
        out_features_graph=dummy_config["graph_out_dim"],
    )


@pytest.fixture
def mock_train_loader(dummy_config: Dict[str, Any]) -> DataLoader:
    """
    Provides a mock training DataLoader.
    """
    dataset = MockPyGDataset(
        num_samples=4,
        in_features=dummy_config["in_dim"],
        node_out_dim=dummy_config["node_out_dim"],
        graph_out_dim=dummy_config["graph_out_dim"],
    )
    return DataLoader(dataset, batch_size=2, collate_fn=mock_collate_fn)


@pytest.fixture
def mock_val_loader(dummy_config: Dict[str, Any]) -> DataLoader:
    """
    Provides a mock validation DataLoader.
    """
    dataset = MockPyGDataset(
        num_samples=2,
        in_features=dummy_config["in_dim"],
        node_out_dim=dummy_config["node_out_dim"],
        graph_out_dim=dummy_config["graph_out_dim"],
    )
    return DataLoader(dataset, batch_size=2, collate_fn=mock_collate_fn)


@pytest.fixture(scope="module")
def temp_trainer_files_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for trainer-related files.
    """
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


class TestMGNNTrainerInit:
    """
    Test suite for MGNNTrainer initialization.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available or PyTorch CUDA setup issue")
    def test_init_all_provided(
        self,
        mock_model: MockSimpleModel,
        dummy_config: Dict[str, Any],
        mock_train_loader: DataLoader,
        mock_val_loader: DataLoader,
    ) -> None:
        """
        Test initialization when all parameters are explicitly provided.
        """
        optimizer = optim.Adam(mock_model.parameters(), lr=0.01)
        loss_fn = nn.L1Loss()
        callback = MagicMock(spec=Callback)

        trainer = MGNNTrainer(
            model=mock_model,
            config=dummy_config,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            callbacks=[callback],
        )
        assert trainer.model == mock_model
        assert trainer.optimizer == optimizer
        assert trainer.loss_fn == loss_fn
        assert trainer.callbacks == [callback]

    def test_init_default_optimizer_adam(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test default Adam optimizer initialization.
        """
        trainer = MGNNTrainer(model=mock_model, config=dummy_config)
        assert isinstance(trainer.optimizer, optim.Adam)
        assert trainer.optimizer.defaults["lr"] == dummy_config["learning_rate"]

    def test_init_default_optimizer_sgd(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test default SGD optimizer initialization.
        """
        config_sgd = dummy_config.copy()
        config_sgd["optimizer"] = "sgd"
        trainer = MGNNTrainer(model=mock_model, config=config_sgd)
        assert isinstance(trainer.optimizer, optim.SGD)

    def test_init_default_optimizer_adamw(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test default AdamW optimizer initialization.
        """
        config_adamw = dummy_config.copy()
        config_adamw["optimizer"] = "adamw"
        trainer = MGNNTrainer(model=mock_model, config=config_adamw)
        assert isinstance(trainer.optimizer, optim.AdamW)

    def test_init_default_loss_regression(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test default MSELoss for regression task.
        """
        trainer = MGNNTrainer(model=mock_model, config=dummy_config)
        assert isinstance(trainer.loss_fn, nn.MSELoss)

    def test_init_default_loss_classification(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test default BCEWithLogitsLoss for classification task.
        """
        config_clf = dummy_config.copy()
        config_clf["task_type"] = "classification"
        trainer = MGNNTrainer(model=mock_model, config=config_clf)
        assert isinstance(trainer.loss_fn, nn.BCEWithLogitsLoss)

    def test_init_unsupported_optimizer(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test unsupported optimizer raises ValueError.
        """
        config_err = dummy_config.copy()
        config_err["optimizer"] = "unknown_opt"
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            MGNNTrainer(model=mock_model, config=config_err)

    def test_init_unsupported_task_type_for_loss(self, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test unsupported task type for loss function raises ValueError.
        """
        config_err = dummy_config.copy()
        config_err["task_type"] = "unknown_task"
        with pytest.raises(ValueError, match="Unsupported task type"):
            MGNNTrainer(model=mock_model, config=config_err)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for this specific test")
    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_auto_cuda(self, mock_cuda_available: MagicMock, mock_model: MockSimpleModel, dummy_config: Dict[str, Any]) -> None:
        """
        Test auto-detection of CUDA device.
        """
        config_no_device = dummy_config.copy()
        if "device" in config_no_device:
            del config_no_device["device"]

        trainer = MGNNTrainer(model=mock_model, config=config_no_device)
        assert trainer.device == "cuda"


class TestMGNNTrainerExecution:
    """
    Test suite for MGNNTrainer execution methods.
    """

    @pytest.fixture
    def trainer(
        self,
        mock_model: MockSimpleModel,
        dummy_config: Dict[str, Any],
        mock_train_loader: DataLoader,
        mock_val_loader: DataLoader,
    ) -> MGNNTrainer:
        """
        Provides a configured MGNNTrainer instance.
        """
        return MGNNTrainer(
            model=mock_model, config=dummy_config, train_loader=mock_train_loader, val_loader=mock_val_loader
        )

    def test_train_epoch(self, trainer: MGNNTrainer) -> None:
        """
        Test a single training epoch.
        """
        trainer.callbacks = [MagicMock(spec=Callback)]
        trainer.model.train = MagicMock(wraps=trainer.model.train)
        trainer.optimizer.zero_grad = MagicMock(wraps=trainer.optimizer.zero_grad)
        trainer.optimizer.step = MagicMock(wraps=trainer.optimizer.step)

        avg_loss = trainer.train_epoch()

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        trainer.model.train.assert_called_once()
        assert trainer.optimizer.zero_grad.call_count == len(trainer.train_loader)
        assert trainer.optimizer.step.call_count == len(trainer.train_loader)
        trainer.callbacks[0].on_batch_begin.assert_called()
        trainer.callbacks[0].on_batch_end.assert_called()

    def test_train_epoch_no_batch_attr(self, trainer: MGNNTrainer, dummy_config: Dict[str, Any]) -> None:
        """
        Test a single training epoch with graphs that don't have a batch attribute.
        """
        single_graph_dataset = MockPyGDataset(num_samples=2, in_features=dummy_config["in_dim"], num_nodes=1)
        single_graph_loader = DataLoader(single_graph_dataset, batch_size=1, collate_fn=lambda x: x[0])
        trainer.train_loader = single_graph_loader

        avg_loss = trainer.train_epoch()
        assert isinstance(avg_loss, float)

    def test_validate(self, trainer: MGNNTrainer) -> None:
        """
        Test the validation step.
        """
        trainer.model.eval = MagicMock(wraps=trainer.model.eval)
        avg_loss = trainer.validate()
        assert isinstance(avg_loss, float)
        trainer.model.eval.assert_called_once()

    def test_validate_no_loader(self, trainer: MGNNTrainer) -> None:
        """
        Test validation when no validation loader is provided.
        """
        trainer.val_loader = None
        avg_loss = trainer.validate()
        assert avg_loss == 0.0

    def test_validate_empty_loader(self, trainer: MGNNTrainer, dummy_config: Dict[str, Any]) -> None:
        """
        Test validation with an empty validation loader.
        """
        empty_dataset = MockPyGDataset(num_samples=0, is_empty=True)
        empty_loader = DataLoader(empty_dataset, batch_size=2, collate_fn=mock_collate_fn)
        trainer.val_loader = empty_loader
        avg_loss = trainer.validate()
        assert avg_loss == 0.0

    @patch.object(MGNNTrainer, "train_epoch", return_value=0.5)
    @patch.object(MGNNTrainer, "validate", return_value=0.4)
    def test_train_loop(self, mock_validate: MagicMock, mock_train_epoch: MagicMock, trainer: MGNNTrainer, capsys: Any) -> None:
        """
        Test the main training loop.
        """
        trainer.callbacks = [MagicMock(spec=Callback)]
        epochs = 2
        trainer.config["epochs"] = epochs

        history = trainer.train(epochs=epochs, log_interval=1)

        assert mock_train_epoch.call_count == epochs
        assert mock_validate.call_count == epochs
        assert len(history["train_loss"]) == epochs
        assert len(history["val_loss"]) == epochs
        assert history["train_loss"] == [0.5] * epochs
        assert history["val_loss"] == [0.4] * epochs

        trainer.callbacks[0].on_train_begin.assert_called_once()
        assert trainer.callbacks[0].on_epoch_begin.call_count == epochs
        assert trainer.callbacks[0].on_epoch_end.call_count == epochs
        trainer.callbacks[0].on_train_end.assert_called_once()

        captured = capsys.readouterr()
        assert f"Epoch 1/{epochs}" in captured.out
        assert f"Epoch {epochs}/{epochs}" in captured.out

    def test_train_loop_stop_training(self, trainer: MGNNTrainer) -> None:
        """
        Test that the training loop stops when `stop_training` flag is set.
        """

        class StopperCallback(Callback):
            def on_epoch_end(self, trainer_instance: MGNNTrainer, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
                if epoch == 0:
                    trainer_instance.stop_training = True

        trainer.callbacks = [StopperCallback()]
        trainer.config["epochs"] = 5
        history = trainer.train()

        assert len(history["train_loss"]) == 1

    @patch("torch.save")
    def test_save_model(self, mock_torch_save: MagicMock, trainer: MGNNTrainer, temp_trainer_files_dir: str) -> None:
        """
        Test saving the model state dictionary.
        """
        filepath = os.path.join(temp_trainer_files_dir, "model.pt")
        trainer.save_model(filepath)

        mock_torch_save.assert_called_once()
        args, _ = mock_torch_save.call_args
        saved_state_dict = args[0]
        saved_filepath = args[1]

        assert saved_filepath == filepath
        assert saved_state_dict.keys() == trainer.model.state_dict().keys()
        for key in saved_state_dict:
            assert torch.equal(saved_state_dict[key], trainer.model.state_dict()[key])

    @patch("torch.save")
    def test_save_checkpoint(self, mock_torch_save: MagicMock, trainer: MGNNTrainer, temp_trainer_files_dir: str) -> None:
        """
        Test saving a full training checkpoint.
        """
        filepath = os.path.join(temp_trainer_files_dir, "checkpoint.pt")
        trainer.history = {"train_loss": [0.1, 0.05]}
        trainer.best_val_loss = 0.04

        trainer.save_checkpoint(filepath)
        mock_torch_save.assert_called_once()
        saved_data = mock_torch_save.call_args[0][0]
        assert "model_state_dict" in saved_data
        assert "optimizer_state_dict" in saved_data
        assert saved_data["history"] == trainer.history
        assert saved_data["config"] == trainer.config
        assert saved_data["best_val_loss"] == trainer.best_val_loss

    @patch("torch.load")
    def test_load_checkpoint(self, mock_torch_load: MagicMock, trainer: MGNNTrainer) -> None:
        """
        Test loading a training checkpoint.
        """
        dummy_checkpoint = {
            "model_state_dict": {"linear.weight": torch.tensor([1.0]), "linear.bias": torch.tensor([0.5])},
            "optimizer_state_dict": {"dummy_opt_key": torch.tensor([2.0])},
            "history": {"train_loss": [0.2]},
            "config": {"loaded_config": True},
            "best_val_loss": 0.15,
        }
        mock_torch_load.return_value = dummy_checkpoint
        trainer.model.load_state_dict = MagicMock()
        trainer.optimizer.load_state_dict = MagicMock()

        trainer.load_checkpoint("dummy_path.pt")

        trainer.model.load_state_dict.assert_called_once_with(dummy_checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict.assert_called_once_with(dummy_checkpoint["optimizer_state_dict"])
        assert trainer.history == dummy_checkpoint["history"]
        assert trainer.best_val_loss == dummy_checkpoint["best_val_loss"]

    @patch("torch.load")
    def test_load_checkpoint_no_optimizer_state(self, mock_torch_load: MagicMock, trainer: MGNNTrainer) -> None:
        """
        Test loading checkpoint without optimizer state.
        """
        dummy_checkpoint = {"model_state_dict": {"linear.weight": torch.tensor([1.0]), "linear.bias": torch.tensor([0.5])}}
        mock_torch_load.return_value = dummy_checkpoint
        trainer.model.load_state_dict = MagicMock()
        trainer.optimizer.load_state_dict = MagicMock()
        trainer.load_checkpoint("dummy.pt")
        trainer.optimizer.load_state_dict.assert_not_called()

    @patch("torch.load")
    def test_load_checkpoint_no_history_or_best_loss(self, mock_torch_load: MagicMock, trainer: MGNNTrainer) -> None:
        """
        Test loading checkpoint without history or best loss.
        """
        model_state = trainer.model.state_dict()
        dummy_checkpoint = {"model_state_dict": model_state.copy()}

        mock_torch_load.return_value = dummy_checkpoint
        original_history = trainer.history.copy()
        original_best_loss = trainer.best_val_loss
        trainer.load_checkpoint("dummy.pt")
        assert trainer.history == original_history
        assert trainer.best_val_loss == original_best_loss

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_training_curves(self, mock_savefig: MagicMock, mock_show: MagicMock, trainer: MGNNTrainer, temp_trainer_files_dir: str) -> None:
        """
        Test plotting training curves.
        """
        trainer.history = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.45]}

        filepath = os.path.join(temp_trainer_files_dir, "curves.png")
        trainer.plot_training_curves(filepath=filepath)
        mock_savefig.assert_called_once_with(filepath)
        mock_show.assert_not_called()
        plt.close("all")

        mock_savefig.reset_mock()
        trainer.plot_training_curves()
        mock_savefig.assert_not_called()
        mock_show.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_training_curves_no_val_loss(self, mock_savefig: MagicMock, mock_show: MagicMock, trainer: MGNNTrainer, temp_trainer_files_dir: str) -> None:
        """
        Test plotting training curves when no validation loss is available.
        """
        trainer.history = {"train_loss": [0.5, 0.4], "val_loss": []}
        filepath = os.path.join(temp_trainer_files_dir, "curves_no_val.png")
        trainer.plot_training_curves(filepath=filepath)
        mock_savefig.assert_called_once_with(filepath)
        plt.close("all")

    def test_get_predictor(self, trainer: MGNNTrainer) -> None:
        """
        Test getting a predictor instance from the trainer.
        """
        predictor = trainer.get_predictor()
        assert isinstance(predictor, MGNNPredictor)
        assert predictor.model == trainer.model
        assert predictor.config == trainer.config
        assert predictor.device == trainer.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available or PyTorch CUDA setup issue")
class TestStandaloneTrainEpoch:
    """
    Test suite for the standalone train_epoch function.
    """

    def test_standalone_function(
        self, mock_model: MockSimpleModel, mock_train_loader: DataLoader, dummy_config: Dict[str, Any]
    ) -> None:
        """
        Test the standalone train_epoch function.
        """
        optimizer = optim.Adam(mock_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        mock_model.train = MagicMock(wraps=mock_model.train)
        optimizer.zero_grad = MagicMock(wraps=optimizer.zero_grad)
        optimizer.step = MagicMock(wraps=optimizer.step)

        avg_loss = standalone_train_epoch(mock_model, optimizer, mock_train_loader, loss_fn, dummy_config["device"])
        assert isinstance(avg_loss, float)
        mock_model.train.assert_called_once()
        assert optimizer.zero_grad.call_count == len(mock_train_loader)
        assert optimizer.step.call_count == len(mock_train_loader)


@patch("moml.models.mgnn.training.trainer.DJMGNN")
@patch("moml.models.mgnn.training.trainer.MGNNTrainer")
class TestCreateTrainerFactory:
    """
    Test suite for the create_trainer factory function.
    """

    def test_create_trainer_new_model(
        self, MockMGNNTrainer: MagicMock, MockDJMGNN: MagicMock, dummy_config: Dict[str, Any], mock_train_loader: DataLoader
    ) -> None:
        """
        Test creating a new trainer with a new model.
        """
        MockDJMGNN.return_value.parameters.return_value = [nn.Parameter(torch.randn(1))]

        create_trainer(config=dummy_config, train_loader=mock_train_loader)

        MockDJMGNN.assert_called_once_with(
            hidden_dim=dummy_config["hidden_dim"],
            n_blocks=dummy_config.get("n_blocks", 3),
            layers_per_block=dummy_config.get("layers_per_block", 2),
            jk_mode=dummy_config.get("jk_mode", "cat"),
            node_out_dim=dummy_config["node_out_dim"],
            graph_out_dim=dummy_config["graph_out_dim"],
            dropout=dummy_config.get("dropout", 0.2),
            in_dim=dummy_config["in_dim"],
            edge_attr_dim=dummy_config["edge_attr_dim"],
        )
        MockMGNNTrainer.assert_called_once()
        args, kwargs = MockMGNNTrainer.call_args
        assert kwargs["model"] == MockDJMGNN.return_value
        assert kwargs["config"] == dummy_config
        assert kwargs["train_loader"] == mock_train_loader

    def test_create_trainer_with_provided_model(
        self, MockMGNNTrainer: MagicMock, MockDJMGNN: MagicMock, dummy_config: Dict[str, Any]
    ) -> None:
        """
        Test creating a trainer with a pre-provided model instance.
        """
        my_model = MockSimpleModel()
        MockDJMGNN.return_value = my_model

        MGNNTrainer(model=my_model, config=dummy_config)
        MockDJMGNN.assert_not_called()

        create_trainer(config=dummy_config)

    def test_create_trainer_missing_model_dims_in_config(
        self, MockMGNNTrainer: MagicMock, MockDJMGNN: MagicMock, dummy_config: Dict[str, Any]
    ) -> None:
        """
        Test creating a trainer when model dimensions are missing from config,
        and inferred from data.
        """
        config_no_dims = dummy_config.copy()
        del config_no_dims["in_dim"]

        MockDJMGNN.return_value.parameters.return_value = [nn.Parameter(torch.randn(1))]

        mock_train_loader = MagicMock()
        mock_dataset = MagicMock()
        mock_sample = MagicMock()
        mock_sample.x = torch.randn(10, 16)
        mock_dataset.__getitem__.return_value = mock_sample
        mock_dataset.__len__.return_value = 1
        mock_train_loader.dataset = mock_dataset

        create_trainer(config=config_no_dims, train_loader=mock_train_loader)

        MockDJMGNN.assert_called_once()
        args, kwargs = MockDJMGNN.call_args
        assert kwargs["in_dim"] == 16
        assert kwargs["hidden_dim"] == config_no_dims["hidden_dim"]
        assert kwargs["n_blocks"] == config_no_dims.get("n_blocks", 3)
        assert kwargs["layers_per_block"] == config_no_dims.get("layers_per_block", 2)
        assert kwargs["edge_attr_dim"] == config_no_dims["edge_attr_dim"]
        assert kwargs["jk_mode"] == config_no_dims.get("jk_mode", "cat")
        assert kwargs["node_out_dim"] == config_no_dims["node_out_dim"]
        assert kwargs["graph_out_dim"] == config_no_dims["graph_out_dim"]
        assert kwargs["dropout"] == config_no_dims.get("dropout", 0.2)

        MockMGNNTrainer.assert_called_once()
        args, kwargs = MockMGNNTrainer.call_args
        assert kwargs["model"] == MockDJMGNN.return_value
        assert kwargs["config"] == config_no_dims
