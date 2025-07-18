"""
tests/test_mgnn_predictor.py

Unit tests for the MGNNPredictor class and related functions
in moml.models.mgnn.evaluation.predictor.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from moml.core.molecular_graph_processor import MolecularGraphProcessor  
from moml.models.mgnn.djmgnn import DJMGNN  
from moml.models.mgnn.evaluation.predictor import MGNNPredictor, batch_predict_from_files, create_predictor


class DummyDJMGNN(DJMGNN):
    """
    A dummy DJMGNN model for testing purposes.
    """

    def __init__(
        self,
        in_node_dim: int = 10,
        hidden_dim: int = 16,
        in_edge_dim: int = 3,
        node_output_dims: int = 1,
        graph_output_dims: int = 1,
        return_single_tensor: bool = False,
    ) -> None:
        """
        Initializes the DummyDJMGNN.

        Args:
            in_node_dim (int, optional): Input node feature dimension. Defaults to 10.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 16.
            in_edge_dim (int, optional): Input edge feature dimension. Defaults to 3.
            node_output_dims (int, optional): Output dimension for node predictions. Defaults to 1.
            graph_output_dims (int, optional): Output dimension for graph predictions. Defaults to 1.
            return_single_tensor (bool, optional): If True, forward returns only graph_pred as a single tensor.
                                                   Defaults to False.
        """
        super().__init__(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            n_blocks=1,
            layers_per_block=1,
            in_edge_dim=in_edge_dim,
            jk_mode="concat",
            node_output_dims=node_output_dims,
            graph_output_dims=graph_output_dims,
            dropout=0.0,
        )
        self.node_out_dim = node_output_dims
        self.graph_out_dim = graph_output_dims
        self.return_single_tensor = return_single_tensor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        dist: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the dummy model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge connectivity.
            edge_attr (Optional[torch.Tensor], optional): Edge features. Defaults to None.
            batch (Optional[torch.Tensor], optional): Batch assignment vector. Defaults to None.
            dist (Optional[Any], optional): Dummy distance parameter. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing node and/or graph predictions.
        """
        num_nodes = x.shape[0]
        if batch is None:
            num_graphs = 1
        else:
            num_graphs = batch.max().item() + 1

        graph_prediction = torch.randn(num_graphs, int(self.graph_out_dim), device=x.device)
        if self.return_single_tensor:
            return {"graph_pred": graph_prediction}

        return {"node_pred": torch.randn(num_nodes, int(self.node_out_dim), device=x.device), "graph_pred": graph_prediction}


@pytest.fixture(scope="module")
def temp_model_files_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for model files.

    Yields:
        str: Path to the temporary directory.
    """
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def dummy_model_instance_and_config() -> Tuple[DummyDJMGNN, Dict[str, Any]]:
    """
    Provides a dummy model instance and its configuration.

    Returns:
        Tuple[DummyDJMGNN, Dict[str, Any]]: A tuple containing the model and its config.
    """
    config = {
        "in_node_dim": 10,
        "hidden_dim": 16,
        "in_edge_dim": 3,
        "edge_attr_dim": 3,
        "n_blocks": 1,
        "layers_per_block": 1,
        "jk_mode": "concat",
        "node_output_dims": 2,
        "graph_output_dims": 1,
        "dropout": 0.0,
        "device": "cpu",
    }
    model = DummyDJMGNN(
        in_node_dim=config["in_node_dim"],
        hidden_dim=config["hidden_dim"],
        in_edge_dim=config["in_edge_dim"],
        node_output_dims=config["node_output_dims"],
        graph_output_dims=config["graph_output_dims"],
    )
    return model, config


@pytest.fixture
def dummy_model_path(temp_model_files_dir: str, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> str:
    """
    Creates a dummy model checkpoint file.

    Args:
        temp_model_files_dir (str): Temporary directory for model files.
        dummy_model_instance_and_config (Tuple[DummyDJMGNN, Dict[str, Any]]): Model and config fixture.

    Returns:
        str: Path to the dummy model checkpoint file.
    """
    model, config = dummy_model_instance_and_config
    model_path = os.path.join(temp_model_files_dir, "dummy_model.pt")
    checkpoint = {"model_state_dict": model.state_dict(), "config": config}
    torch.save(checkpoint, model_path)
    return model_path


@pytest.fixture
def dummy_graph_data() -> Data:
    """
    Provides dummy graph data for a single graph.

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 3)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def dummy_graph_data_list(dummy_graph_data: Data) -> List[Data]:
    """
    Provides a list of dummy graph data objects.

    Args:
        dummy_graph_data (Data): A dummy graph data object.

    Returns:
        List[Data]: A list of PyTorch Geometric Data objects.
    """
    data1 = dummy_graph_data.clone()
    data2_x = torch.randn(3, 10)
    data2_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data2_edge_attr = torch.randn(2, 3)
    data2 = Data(x=data2_x, edge_index=data2_edge_index, edge_attr=data2_edge_attr)
    return [data1, data2]


class TestMGNNPredictorInit:
    """
    Test suite for MGNNPredictor initialization.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available or PyTorch CUDA setup issue")
    def test_init_with_model_path(self, dummy_model_path: str, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test initialization of MGNNPredictor with a model path.
        """
        _, config = dummy_model_instance_and_config
        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor") as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            dummy_x = torch.randn(1, config["in_node_dim"])
            dummy_edge_attr = torch.randn(1, config["in_edge_dim"])
            mock_processor.mol_to_graph.return_value = Data(
                x=dummy_x, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=dummy_edge_attr
            )
            mock_create_proc.return_value = mock_processor

            predictor = MGNNPredictor(model_path=dummy_model_path, config=config)
            assert isinstance(predictor.model, DJMGNN)
            assert predictor.device == "cpu"

    def test_init_with_model_instance(self, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test initialization of MGNNPredictor with a model instance.
        """
        model, config = dummy_model_instance_and_config
        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            predictor = MGNNPredictor(model=model, config=config)
            assert predictor.model == model
            assert predictor.device == "cpu"

    def test_init_no_model_or_path(self) -> None:
        """
        Test initialization without model or path raises ValueError.
        """
        with pytest.raises(ValueError, match="Either model_path or model must be provided"):
            MGNNPredictor()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_cuda(self, mock_cuda_available: MagicMock, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test initialization with CUDA device.
        """
        model, config = dummy_model_instance_and_config
        config_cuda = config.copy()
        config_cuda["device"] = "cuda"
        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            predictor = MGNNPredictor(model=model, config=config_cuda)
            assert predictor.device == "cuda"

            config_no_device = config.copy()
            del config_no_device["device"]
            predictor_auto_cuda = MGNNPredictor(model=model, config=config_no_device)
            assert predictor_auto_cuda.device == "cuda"

    def test_load_model_infer_dims(self, temp_model_files_dir: str, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test model loading infers dimensions from original config.
        """
        model_orig, config_orig = dummy_model_instance_and_config
        config_no_dims = config_orig.copy()
        if "in_node_dim" in config_no_dims:
            del config_no_dims["in_node_dim"]
        if "in_edge_dim" in config_no_dims:
            del config_no_dims["in_edge_dim"]

        model_path_no_dims = os.path.join(temp_model_files_dir, "model_no_dims.pt")
        checkpoint = {"model_state_dict": model_orig.state_dict(), "config": config_no_dims}
        torch.save(checkpoint, model_path_no_dims)

        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor") as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            dummy_x = torch.randn(1, config_orig["in_node_dim"])
            dummy_edge_attr = torch.randn(1, config_orig["in_edge_dim"])
            mock_processor.mol_to_graph.return_value = Data(
                x=dummy_x, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=dummy_edge_attr
            )
            # Set the dimensions on the mock processor to match the original config
            mock_processor.atom_feature_dim = config_orig["in_node_dim"]
            mock_processor.bond_feature_dim = config_orig["in_edge_dim"]
            mock_create_proc.return_value = mock_processor

            predictor = MGNNPredictor(model_path=model_path_no_dims, config=config_no_dims.copy())
            assert predictor.model.in_node_dim == config_orig["in_node_dim"]
            assert predictor.model.input_edge_attr_dim == config_orig["in_edge_dim"]

    def test_load_model_file_not_found(self, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test loading non-existent model file raises ValueError.
        """
        _, config = dummy_model_instance_and_config
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            MGNNPredictor(model_path="non_existent_model.pt", config=config)

    def test_load_model_corrupted_file(self, temp_model_files_dir: str, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test loading corrupted model file raises ValueError.
        """
        _, config = dummy_model_instance_and_config
        corrupted_file_path = os.path.join(temp_model_files_dir, "corrupted_model.pt")
        with open(corrupted_file_path, "w") as f:
            f.write("This is not a torch model")

        with pytest.raises(ValueError, match="Failed to load model"):
            MGNNPredictor(model_path=corrupted_file_path, config=config)


class TestMGNNPredictorMethods:
    """
    Test suite for MGNNPredictor methods.
    """

    @pytest.fixture
    def predictor_fixture(self, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> MGNNPredictor:
        """
        Provides a MGNNPredictor instance for testing methods.
        """
        model, config = dummy_model_instance_and_config
        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor") as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            mock_create_proc.return_value = mock_processor
            return MGNNPredictor(model=model, config=config)

    def test_predict_from_graph(self, predictor_fixture: MGNNPredictor, dummy_graph_data: Data) -> None:
        """
        Test prediction from a single graph.
        """
        predictor = predictor_fixture
        model_config = predictor.config
        results = predictor.predict_from_graph(dummy_graph_data)
        assert "node_pred" in results
        assert "graph_pred" in results
        assert results["node_pred"].shape == (dummy_graph_data.x.shape[0], model_config["node_output_dims"])
        assert results["graph_pred"].shape == (1, model_config["graph_output_dims"])

    def test_predict_from_graph_model_returns_tensor(self, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]], dummy_graph_data: Data) -> None:
        """
        Test prediction when model's forward method returns a single tensor.
        """
        model, config = dummy_model_instance_and_config
        model_single_tensor = DummyDJMGNN(
            in_node_dim=config["in_node_dim"],
            hidden_dim=config["hidden_dim"],
            in_edge_dim=config["in_edge_dim"],
            node_output_dims=config["node_output_dims"],
            graph_output_dims=config["graph_output_dims"],
            return_single_tensor=True,
        )
        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            predictor = MGNNPredictor(model=model_single_tensor, config=config)
            results = predictor.predict_from_graph(dummy_graph_data)
            assert "graph_pred" in results
            assert "node_pred" not in results
            assert results["graph_pred"].shape == (1, config["graph_output_dims"])

    def test_predict_from_graph_no_edges(self, predictor_fixture: MGNNPredictor, dummy_graph_data: Data) -> None:
        """
        Test prediction from a graph with no edges.
        """
        predictor = predictor_fixture
        graph_no_edges = dummy_graph_data.clone()
        graph_no_edges.edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_no_edges.edge_attr = torch.empty((0, dummy_graph_data.edge_attr.shape[1]))

        results = predictor.predict_from_graph(graph_no_edges)
        assert "node_pred" in results
        assert "graph_pred" in results
        assert results["node_pred"].shape[0] == graph_no_edges.x.shape[0]
        assert results["graph_pred"].shape[0] == 1

    def test_predict_from_file(self, predictor_fixture: MGNNPredictor, dummy_graph_data: Data, temp_model_files_dir: str) -> None:
        """
        Test prediction from a graph file.
        """
        predictor = predictor_fixture
        file_path = os.path.join(temp_model_files_dir, "test_graph.pt")
        torch.save(dummy_graph_data, file_path)

        with patch.object(predictor.graph_processor, "file_to_graph", return_value=dummy_graph_data) as mock_file_to_graph:
            results = predictor.predict_from_file(file_path)
            mock_file_to_graph.assert_called_once_with(file_path)
            assert "graph_pred" in results

    def test_predict_from_smiles(self, predictor_fixture: MGNNPredictor, dummy_graph_data: Data) -> None:
        """
        Test prediction from a SMILES string.
        """
        predictor = predictor_fixture
        smiles = "CCO"
        with patch.object(
            predictor.graph_processor, "smiles_to_graph", return_value=dummy_graph_data
        ) as mock_smiles_to_graph:
            results = predictor.predict_from_smiles(smiles)
            mock_smiles_to_graph.assert_called_once_with(smiles)
            assert "graph_pred" in results

    def test_batch_predict(self, predictor_fixture: MGNNPredictor, dummy_graph_data_list: List[Data]) -> None:
        """
        Test batch prediction from a list of graphs.
        """
        predictor = predictor_fixture
        results = predictor.batch_predict(dummy_graph_data_list)
        assert len(results) == len(dummy_graph_data_list)
        assert "graph_pred" in results[0]

    def test_batch_predict_with_node_predictions(self, predictor_fixture: MGNNPredictor, dummy_graph_data_list: List[Data]) -> None:
        """
        Test that node predictions are properly split in batch processing.
        """
        predictor = predictor_fixture
        results = predictor.batch_predict(dummy_graph_data_list)

        assert len(results) == len(dummy_graph_data_list)

        for i, result in enumerate(results):
            assert "graph_pred" in result
            assert "node_pred" in result

            original_graph = dummy_graph_data_list[i]
            expected_nodes = original_graph.x.shape[0]
            actual_nodes = result["node_pred"].shape[0]

            assert actual_nodes == expected_nodes, f"Graph {i}: expected {expected_nodes} nodes, got {actual_nodes}"

    def test_split_batch_predictions_node_splitting(self) -> None:
        """
        Test the _split_batch_predictions method directly for node prediction splitting.
        """
        from moml.models.mgnn.evaluation.predictor import MGNNPredictor

        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            dummy_model = MagicMock()
            predictor = MGNNPredictor(model=dummy_model, config={})

        node_counts = [3, 2]
        total_nodes = sum(node_counts)

        batch_predictions = {
            "graph_pred": torch.randn(2, 1),
            "node_pred": torch.randn(total_nodes, 2),
            "node_features": torch.randn(total_nodes, 4),
        }

        individual_results = predictor._split_batch_predictions(
            batch_predictions, num_graphs=2, node_counts=node_counts
        )

        assert len(individual_results) == 2

        assert individual_results[0]["graph_pred"].shape == (1, 1)
        assert individual_results[0]["node_pred"].shape == (3, 2)
        assert individual_results[0]["node_features"].shape == (3, 4)

        assert individual_results[1]["graph_pred"].shape == (1, 1)
        assert individual_results[1]["node_pred"].shape == (2, 2)
        assert individual_results[1]["node_features"].shape == (2, 4)

        torch.testing.assert_close(individual_results[0]["node_pred"], batch_predictions["node_pred"][:3])
        torch.testing.assert_close(individual_results[1]["node_pred"], batch_predictions["node_pred"][3:])

    def test_split_batch_predictions_empty_graphs(self) -> None:
        """
        Test handling of empty graphs in node prediction splitting.
        """
        from moml.models.mgnn.evaluation.predictor import MGNNPredictor

        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            dummy_model = MagicMock()
            predictor = MGNNPredictor(model=dummy_model, config={})

        node_counts = [2, 0, 3]
        total_nodes = sum(node_counts)

        batch_predictions = {
            "graph_pred": torch.randn(3, 1),
            "node_pred": torch.randn(total_nodes, 2),
        }

        individual_results = predictor._split_batch_predictions(
            batch_predictions, num_graphs=3, node_counts=node_counts
        )

        assert len(individual_results) == 3

        assert individual_results[0]["node_pred"].shape == (2, 2)

        assert individual_results[1]["node_pred"].shape == (0, 2)

        assert individual_results[2]["node_pred"].shape == (3, 2)

    def test_split_batch_predictions_no_node_counts(self) -> None:
        """
        Test behavior when node counts are not provided.
        """
        from moml.models.mgnn.evaluation.predictor import MGNNPredictor

        with patch("moml.models.mgnn.evaluation.predictor.create_graph_processor"):
            dummy_model = MagicMock()
            predictor = MGNNPredictor(model=dummy_model, config={})

        batch_predictions = {
            "graph_pred": torch.randn(2, 1),
            "node_pred": torch.randn(5, 2),
        }

        individual_results = predictor._split_batch_predictions(
            batch_predictions, num_graphs=2, node_counts=None
        )

        assert len(individual_results) == 2
        assert "graph_pred" in individual_results[0]
        assert "graph_pred" in individual_results[1]

        assert "node_pred" not in individual_results[0]
        assert "node_pred" not in individual_results[1]

    def test_predict_from_dataloader(self, predictor_fixture: MGNNPredictor, dummy_graph_data_list: List[Data]) -> None:
        """
        Test prediction from a PyTorch Geometric DataLoader.
        """
        from torch_geometric.loader import DataLoader

        predictor = predictor_fixture
        dataloader = DataLoader(dummy_graph_data_list, batch_size=2)
        results = predictor.predict_from_dataloader(dataloader)

        total_graphs = len(dummy_graph_data_list)
        assert "graph_pred" in results
        assert results["graph_pred"].shape[0] == total_graphs

    def test_save_predictions(self, predictor_fixture: MGNNPredictor, temp_model_files_dir: str) -> None:
        """
        Test saving predictions to a JSON file.
        """
        predictor = predictor_fixture
        results = {"graph_pred": torch.randn(2, 1), "node_pred": torch.randn(5, 2)}
        output_path = os.path.join(temp_model_files_dir, "predictions.json")
        predictor.save_predictions(results, output_path)

        with open(output_path, "r") as f:
            saved_data = json.load(f)
        assert "graph_pred" in saved_data
        assert len(saved_data["graph_pred"]) == 2


class TestCreatePredictorFactory:
    """
    Test suite for the create_predictor factory function.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available or PyTorch CUDA setup issue")
    def test_create_with_model_path(self, dummy_model_path: str) -> None:
        """
        Test creating predictor with a model path.
        """
        predictor = create_predictor(model_path=dummy_model_path)
        assert isinstance(predictor, MGNNPredictor)

    def test_create_with_model_instance(self, dummy_model_instance_and_config: Tuple[DummyDJMGNN, Dict[str, Any]]) -> None:
        """
        Test creating predictor with a model instance.
        """
        model, config = dummy_model_instance_and_config
        predictor = create_predictor(model=model, config=config)
        assert isinstance(predictor, MGNNPredictor)

    def test_create_no_args(self) -> None:
        """
        Test creating predictor without arguments raises ValueError.
        """
        with pytest.raises(ValueError, match="Either model_path or model must be provided"):
            create_predictor(config={})  # type: ignore


@patch("moml.models.mgnn.evaluation.predictor.MGNNPredictor")
class TestBatchPredictFromFiles:
    """
    Test suite for batch_predict_from_files function.
    """

    @pytest.fixture
    def temp_input_dir_for_batch(self, temp_model_files_dir: str) -> str:
        """
        Creates a temporary input directory with dummy graph files for batch prediction.
        """
        input_dir = os.path.join(temp_model_files_dir, "batch_input")
        os.makedirs(input_dir, exist_ok=True)
        torch.save(Data(x=torch.randn(2, 10)), os.path.join(input_dir, "graph1.pt"))
        torch.save(Data(x=torch.randn(3, 10)), os.path.join(input_dir, "graph2.pt"))
        with open(os.path.join(input_dir, "info.txt"), "w") as f:
            f.write("some text")
        return input_dir

    def test_batch_predict_files_success(
        self, mock_mgnn_predictor: MagicMock, temp_input_dir_for_batch: str, dummy_model_path: str, temp_model_files_dir: str
    ) -> None:
        """
        Test successful batch prediction from files.
        """
        mock_predictor_instance = MagicMock(spec=MGNNPredictor)
        mock_mgnn_predictor.return_value = mock_predictor_instance

        mock_predictor_instance.batch_predict.return_value = [
            {"graph_pred": torch.tensor([[0.5]])},
            {"graph_pred": torch.tensor([[-0.2]])},
        ]
        mock_predictor_instance.graph_processor = MagicMock(spec=MolecularGraphProcessor)
        mock_predictor_instance.config = {"device": "cpu"}

        def file_to_graph_side_effect(file_path: str) -> Optional[Data]:
            if "graph1" in file_path:
                return Data(x=torch.randn(2, 10), edge_index=torch.empty((2, 0), dtype=torch.long))
            if "graph2" in file_path:
                return Data(x=torch.randn(3, 10), edge_index=torch.empty((2, 0), dtype=torch.long))
            return None

        mock_predictor_instance.graph_processor.file_to_graph = MagicMock(side_effect=file_to_graph_side_effect)

        output_dir = os.path.join(temp_model_files_dir, "batch_output")
        batch_predict_from_files(
            model_path=dummy_model_path,
            input_dir=temp_input_dir_for_batch,
            output_dir=output_dir,
            file_pattern="*.pt",
        )

        assert mock_mgnn_predictor.call_count == 1
        assert mock_predictor_instance.batch_predict.call_count == 1
        assert os.path.exists(os.path.join(output_dir, "predictions.json"))

    def test_batch_predict_no_files_found(self, mock_mgnn_predictor: MagicMock, temp_input_dir_for_batch: str, dummy_model_path: str) -> None:
        """
        Test batch prediction when no files are found.
        """
        output_dir = os.path.join(temp_input_dir_for_batch, "no_files_output")

        with pytest.raises(ValueError, match="No files found"):
            batch_predict_from_files(
                model_path=dummy_model_path,
                input_dir=temp_input_dir_for_batch,
                output_dir=output_dir,
                file_pattern="*.nonexistent",
            )
        assert not os.path.exists(output_dir)
