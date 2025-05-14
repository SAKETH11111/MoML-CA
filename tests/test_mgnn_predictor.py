"""
Unit tests for the MGNNPredictor class and related functions
in moml.models.mgnn.evaluation.predictor.
"""
import pytest
import torch
import torch.nn as nn
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock, mock_open
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem

from moml.models.mgnn.evaluation.predictor import (
    MGNNPredictor,
    create_predictor,
    batch_predict_from_files
)
from moml.models.mgnn.djmgnn import DJMGNN # Used by predictor
from moml.core.molecular_graph_processor import MolecularGraphProcessor # Used by predictor


# Dummy Model for testing
class DummyDJMGNN(DJMGNN):
    def __init__(self, in_dim=10, hidden_dim=16, edge_attr_dim=3, node_out_dim=1, graph_out_dim=1, return_single_tensor=False):
        super().__init__(
            in_dim=in_dim, hidden_dim=hidden_dim, n_blocks=1, layers_per_block=1,
            edge_attr_dim=edge_attr_dim, jk_mode='concat',
            node_out_dim=node_out_dim, graph_out_dim=graph_out_dim, dropout=0.0
        )
        self.node_out_dim = node_out_dim
        self.graph_out_dim = graph_out_dim
        self.return_single_tensor = return_single_tensor

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        num_nodes = x.shape[0]
        if batch is None:
            num_graphs = 1
            # Ensure batch is created if None for internal model logic if it relies on it
            # For this dummy, it's fine, but real models might need it.
            # batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        else:
            num_graphs = batch.max().item() + 1
        
        graph_prediction = torch.randn(num_graphs, self.graph_out_dim, device=x.device)
        if self.return_single_tensor:
            return graph_prediction # Only graph_pred as a single tensor

        return {
            'node_pred': torch.randn(num_nodes, self.node_out_dim, device=x.device),
            'graph_pred': graph_prediction
        }

@pytest.fixture(scope="module")
def temp_model_files_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

@pytest.fixture
def dummy_model_instance_and_config():
    config = {
        'in_dim': 10, 'hidden_dim': 16, 'edge_attr_dim': 3,
        'n_blocks': 1, 'layers_per_block': 1, 'jk_mode': 'concat',
        'node_out_dim': 2, 'graph_out_dim': 1, 'dropout': 0.0,
        'device': 'cpu'
    }
    model = DummyDJMGNN(
        in_dim=config['in_dim'], hidden_dim=config['hidden_dim'],
        edge_attr_dim=config['edge_attr_dim'],
        node_out_dim=config['node_out_dim'], graph_out_dim=config['graph_out_dim']
    )
    return model, config

@pytest.fixture
def dummy_model_path(temp_model_files_dir, dummy_model_instance_and_config):
    model, config = dummy_model_instance_and_config
    model_path = os.path.join(temp_model_files_dir, "dummy_model.pt")
    checkpoint = {'model_state_dict': model.state_dict(), 'config': config}
    torch.save(checkpoint, model_path)
    return model_path

@pytest.fixture
def dummy_graph_data():
    x = torch.randn(5, 10) # 5 nodes, 10 features (matches dummy_model_instance_and_config in_dim)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 3) # 4 edges, 3 edge features
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

@pytest.fixture
def dummy_graph_data_list(dummy_graph_data):
    data1 = dummy_graph_data.clone()
    data2_x = torch.randn(3, 10)
    data2_edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    data2_edge_attr = torch.randn(2,3)
    data2 = Data(x=data2_x, edge_index=data2_edge_index, edge_attr=data2_edge_attr)
    return [data1, data2]


class TestMGNNPredictorInit:
    def test_init_with_model_path(self, dummy_model_path, dummy_model_instance_and_config):
        _, config = dummy_model_instance_and_config
        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor') as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            dummy_x = torch.randn(1, config['in_dim'])
            dummy_edge_attr = torch.randn(1, config['edge_attr_dim'])
            mock_processor.mol_to_graph.return_value = Data(x=dummy_x, edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=dummy_edge_attr)
            mock_create_proc.return_value = mock_processor
            
            predictor = MGNNPredictor(model_path=dummy_model_path, config=config)
            assert isinstance(predictor.model, DJMGNN) 
            assert predictor.device == 'cpu'

    def test_init_with_model_instance(self, dummy_model_instance_and_config):
        model, config = dummy_model_instance_and_config
        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor'):
            predictor = MGNNPredictor(model=model, config=config)
            assert predictor.model == model
            assert predictor.device == 'cpu'

    def test_init_no_model_or_path(self):
        with pytest.raises(ValueError, match="Either model_path or model must be provided"):
            MGNNPredictor()

    @patch('torch.cuda.is_available', return_value=True)
    def test_init_device_cuda(self, mock_cuda_available, dummy_model_instance_and_config):
        model, config = dummy_model_instance_and_config
        config_cuda = config.copy()
        config_cuda['device'] = 'cuda' # Explicitly set cuda
        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor'):
            predictor = MGNNPredictor(model=model, config=config_cuda) # Pass config with cuda
            assert predictor.device == 'cuda'
            # Test auto-detection if config['device'] is not set
            config_no_device = config.copy()
            del config_no_device['device']
            predictor_auto_cuda = MGNNPredictor(model=model, config=config_no_device)
            assert predictor_auto_cuda.device == 'cuda'


    def test_load_model_infer_dims(self, temp_model_files_dir, dummy_model_instance_and_config):
        model_orig, config_orig = dummy_model_instance_and_config
        config_no_dims = config_orig.copy()
        if 'in_dim' in config_no_dims: del config_no_dims['in_dim']
        if 'edge_attr_dim' in config_no_dims: del config_no_dims['edge_attr_dim']
        
        model_path_no_dims = os.path.join(temp_model_files_dir, "model_no_dims.pt")
        checkpoint = {'model_state_dict': model_orig.state_dict(), 'config': config_no_dims}
        torch.save(checkpoint, model_path_no_dims)

        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor') as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            dummy_x = torch.randn(1, config_orig['in_dim']) 
            dummy_edge_attr = torch.randn(1, config_orig['edge_attr_dim'])
            mock_processor.mol_to_graph.return_value = Data(x=dummy_x, edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=dummy_edge_attr)
            mock_create_proc.return_value = mock_processor

            predictor = MGNNPredictor(model_path=model_path_no_dims, config=config_no_dims.copy())
            assert predictor.model.in_dim == config_orig['in_dim']
            assert predictor.model.edge_attr_dim == config_orig['edge_attr_dim']

    def test_load_model_file_not_found(self, dummy_model_instance_and_config):
        _, config = dummy_model_instance_and_config
        with pytest.raises(ValueError, match="Failed to load model"):
            MGNNPredictor(model_path="non_existent_model.pt", config=config)

    def test_load_model_corrupted_file(self, temp_model_files_dir, dummy_model_instance_and_config):
        _, config = dummy_model_instance_and_config
        corrupted_file_path = os.path.join(temp_model_files_dir, "corrupted_model.pt")
        with open(corrupted_file_path, 'w') as f:
            f.write("This is not a torch model")
        
        with pytest.raises(ValueError, match="Failed to load model"):
            MGNNPredictor(model_path=corrupted_file_path, config=config)


class TestMGNNPredictorMethods:
    @pytest.fixture
    def predictor_fixture(self, dummy_model_instance_and_config): # Renamed to avoid conflict
        model, config = dummy_model_instance_and_config
        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor') as mock_create_proc:
            mock_processor = MagicMock(spec=MolecularGraphProcessor)
            mock_create_proc.return_value = mock_processor
            return MGNNPredictor(model=model, config=config)

    def test_predict_from_graph(self, predictor_fixture, dummy_graph_data):
        predictor = predictor_fixture
        model_config = predictor.config
        results = predictor.predict_from_graph(dummy_graph_data)
        assert 'node_pred' in results
        assert 'graph_pred' in results
        assert results['node_pred'].shape == (dummy_graph_data.x.shape[0], model_config['node_out_dim'])
        assert results['graph_pred'].shape == (1, model_config['graph_out_dim'])

    def test_predict_from_graph_model_returns_tensor(self, dummy_model_instance_and_config, dummy_graph_data):
        model, config = dummy_model_instance_and_config
        # Re-init model to return single tensor
        model_single_tensor = DummyDJMGNN(
            in_dim=config['in_dim'], hidden_dim=config['hidden_dim'],
            edge_attr_dim=config['edge_attr_dim'],
            node_out_dim=config['node_out_dim'], graph_out_dim=config['graph_out_dim'],
            return_single_tensor=True
        )
        with patch('moml.models.mgnn.evaluation.predictor.create_graph_processor'):
            predictor = MGNNPredictor(model=model_single_tensor, config=config)
            results = predictor.predict_from_graph(dummy_graph_data)
            assert 'graph_pred' in results
            assert 'node_pred' not in results # Model only returned graph_pred
            assert results['graph_pred'].shape == (1, config['graph_out_dim'])


    def test_predict_from_graph_no_edges(self, predictor_fixture, dummy_graph_data):
        predictor = predictor_fixture
        graph_no_edges = dummy_graph_data.clone()
        graph_no_edges.edge_index = torch.empty((2,0), dtype=torch.long)
        graph_no_edges.edge_attr = torch.empty((0, dummy_graph_data.edge_attr.shape[1]))
        
        results = predictor.predict_from_graph(graph_no_edges)
        assert 'node_pred' in results
        assert 'graph_pred' in results
        # Check shapes based on the model's output dims
        assert results['node_pred'].shape[0] == graph_no_edges.x.shape[0]
        assert results['graph_pred'].shape[0] == 1


    def test_predict_from_file(self, predictor_fixture, dummy_graph_data, temp_model_files_dir):
        predictor = predictor_fixture
        dummy_file_path = os.path.join(temp_model_files_dir, "test_mol.sdf")
        mol = Chem.MolFromSmiles("CCO")
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            writer = Chem.SDWriter(dummy_file_path)
            writer.write(mol)
            writer.close()
        else:
            pytest.fail("Failed to create molecule from SMILES for test_predict_from_file")


        predictor.processor.file_to_graph.return_value = dummy_graph_data 
        with patch.object(predictor, 'predict_from_graph', wraps=predictor.predict_from_graph) as mock_predict_graph:
            predictor.predict_from_file(dummy_file_path)
            mock_predict_graph.assert_called_once_with(dummy_graph_data)
        if os.path.exists(dummy_file_path): os.remove(dummy_file_path)


    def test_predict_from_smiles(self, predictor_fixture, dummy_graph_data):
        predictor = predictor_fixture
        smiles = "CCO"
        predictor.processor.smiles_to_graph.return_value = dummy_graph_data
        with patch.object(predictor, 'predict_from_graph', wraps=predictor.predict_from_graph) as mock_predict_graph:
            predictor.predict_from_smiles(smiles)
            mock_predict_graph.assert_called_once_with(dummy_graph_data)
            predictor.processor.smiles_to_graph.assert_called_once_with(smiles)


    def test_batch_predict(self, predictor_fixture, dummy_graph_data_list):
        predictor = predictor_fixture
        with patch.object(predictor, 'predict_from_dataloader') as mock_predict_loader:
            mock_predict_loader.return_value = {"graph_pred": torch.randn(len(dummy_graph_data_list),1)}
            results = predictor.batch_predict(dummy_graph_data_list, batch_size=2)
            mock_predict_loader.assert_called_once()
            assert 'graph_pred' in results
            assert results['graph_pred'].shape[0] == len(dummy_graph_data_list)


    def test_predict_from_dataloader(self, predictor_fixture, dummy_graph_data_list):
        predictor = predictor_fixture
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        loader = PyGDataLoader(dummy_graph_data_list, batch_size=1, shuffle=False)
        
        results = predictor.predict_from_dataloader(loader)
        
        total_nodes = sum(g.x.shape[0] for g in dummy_graph_data_list)
        num_graphs = len(dummy_graph_data_list)
        model_config = predictor.config

        assert 'node_pred' in results
        assert 'graph_pred' in results
        assert results['node_pred'].shape == (total_nodes, model_config['node_out_dim'])
        assert results['graph_pred'].shape == (num_graphs, model_config['graph_out_dim'])


    def test_save_predictions(self, predictor_fixture, temp_model_files_dir):
        predictor = predictor_fixture
        output_file = os.path.join(temp_model_files_dir, "preds.json")
        config_file = os.path.join(temp_model_files_dir, "preds_config.json")
        predictions = {
            'node_pred': torch.randn(5, 2),
            'graph_pred': torch.randn(1, 1)
        }
        predictor.save_predictions(predictions, output_file, save_config=True)
        
        assert os.path.exists(output_file)
        assert os.path.exists(config_file)

        with open(output_file, 'r') as f:
            loaded_preds = json.load(f)
            assert len(loaded_preds['node_pred']) == 5
            assert len(loaded_preds['graph_pred']) == 1
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            assert loaded_config == predictor.config


class TestCreatePredictorFactory:
    def test_create_with_model_path(self, dummy_model_path):
        with patch('moml.models.mgnn.evaluation.predictor.MGNNPredictor') as MockPredictor:
            create_predictor(model_path=dummy_model_path) # config and device are optional
            MockPredictor.assert_called_once_with(model_path=dummy_model_path, model=None, config={}, device=None)

    def test_create_with_model_instance(self, dummy_model_instance_and_config):
        model, _ = dummy_model_instance_and_config
        with patch('moml.models.mgnn.evaluation.predictor.MGNNPredictor') as MockPredictor:
            create_predictor(model=model)
            MockPredictor.assert_called_once_with(model_path=None, model=model, config={}, device=None)

    def test_create_no_args(self):
        with pytest.raises(ValueError, match="Either model_path or model must be provided"):
            create_predictor()


@patch('moml.models.mgnn.evaluation.predictor.create_predictor')
class TestBatchPredictFromFiles:
    @pytest.fixture
    def temp_input_dir_for_batch(self, temp_model_files_dir):
        input_dir = os.path.join(temp_model_files_dir, "batch_input_files")
        os.makedirs(input_dir, exist_ok=True)
        for i in range(3):
            mol = Chem.MolFromSmiles("C" * (i + 1))
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                with open(os.path.join(input_dir, f"mol_{i}.mol"), 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
            else:
                # Create an empty file if mol creation fails, to test robustness
                with open(os.path.join(input_dir, f"mol_{i}_bad.mol"), 'w') as f_bad:
                    pass # Just create the file

        # Add one intentionally bad file
        with open(os.path.join(input_dir, "corrupted.mol"), "w") as f:
            f.write("This is not a mol file")
        yield input_dir

    def test_batch_predict_files_success(self, mock_create_pred, temp_input_dir_for_batch, dummy_model_path, temp_model_files_dir):
        mock_predictor_instance = MagicMock(spec=MGNNPredictor)
        
        # Simulate 3 good graphs and one error for the corrupted file
        good_graphs = [Data(x=torch.randn(i+1,10)) for i in range(3)]
        
        # Make side_effect a list: 3 good graphs, then an exception for "corrupted.mol"
        # The order depends on glob.glob, so we make it more robust by checking calls
        def file_to_graph_side_effect(file_path):
            if "corrupted.mol" in file_path:
                raise ValueError("Corrupted file")
            elif "mol_0.mol" in file_path: return good_graphs[0]
            elif "mol_1.mol" in file_path: return good_graphs[1]
            elif "mol_2.mol" in file_path: return good_graphs[2]
            else: # for mol_i_bad.mol if Chem.MolFromSmiles failed
                raise ValueError("Bad SMILES in test setup")

        mock_predictor_instance.processor.file_to_graph.side_effect = file_to_graph_side_effect
        
        mock_predictor_instance.batch_predict.return_value = {
            'graph_pred': torch.randn(3,1) # Only 3 good files
        }
        mock_create_pred.return_value = mock_predictor_instance

        output_dir = os.path.join(temp_model_files_dir, "batch_output")
        
        with patch('builtins.print') as mock_print: # To check error logging
            results = batch_predict_from_files(
                model_path=dummy_model_path,
                input_dir=temp_input_dir_for_batch,
                output_dir=output_dir,
                file_pattern="*.mol" # This will pick up good and bad .mol files
            )
        
        assert len(results) == 3 # Only 3 files successfully processed and have predictions
        assert "mol_0.mol" in results 
        assert "corrupted.mol" not in results # Should have been skipped
        assert os.path.exists(os.path.join(output_dir, "combined_predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "mol_0_pred.json"))
        
        mock_predictor_instance.processor.file_to_graph.assert_called()
        # Check that print was called for the error
        error_printed = any("Error processing" in call.args[0] and "corrupted.mol" in call.args[0] for call in mock_print.call_args_list)
        assert error_printed
        mock_predictor_instance.batch_predict.assert_called_once()
        # Ensure batch_predict was called with the correct number of good graphs
        assert len(mock_predictor_instance.batch_predict.call_args[0][0]) == 3


    def test_batch_predict_no_files_found(self, mock_create_pred, temp_input_dir_for_batch, dummy_model_path):
        empty_subdir = os.path.join(temp_input_dir_for_batch, "empty_for_real") # New empty dir
        os.makedirs(empty_subdir, exist_ok=True)
        with pytest.raises(ValueError, match="No files found"):
            batch_predict_from_files(
                model_path=dummy_model_path,
                input_dir=empty_subdir, # Use the truly empty subdir
                file_pattern="*.nonexistent" # Or a pattern that matches nothing
            )
