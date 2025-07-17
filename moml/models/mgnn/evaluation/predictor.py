"""
moml/models/mgnn/evaluation/predictor.py

MGNN Model Predictor Module

This module provides comprehensive functionality for making predictions with trained
Molecular Graph Neural Network (MGNN) models. It includes classes and functions for 
loading trained models, processing molecular data, and generating predictions for 
individual molecules or batches.

The module supports multiple input formats including SMILES strings, molecular files,
and pre-computed graph objects. It also provides utilities for batch processing and
saving prediction results.

Classes:
    MGNNPredictor: Main predictor class for loading and using trained models

Functions:
    create_predictor: Factory function to create predictor instances
    batch_predict_from_files: Batch prediction on directories of molecule files
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_BLOCKS = 3
DEFAULT_LAYERS_PER_BLOCK = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_FILE_PATTERN = "*.mol"

# Setup logging
logger = logging.getLogger(__name__)

try:
    from moml.core import create_graph_processor  # type: ignore
    HAS_CORE = True
except ImportError:
    logger.warning("moml.core not available - limited functionality")
    HAS_CORE = False
    
    def create_graph_processor(config: Dict[str, Any]) -> None:  # type: ignore
        """Dummy function when core is not available."""
        raise ImportError("moml.core required for graph processing")

try:
    from moml.models.mgnn import DJMGNN  # type: ignore
    HAS_MGNN = True
except ImportError:
    logger.warning("moml.models.mgnn not available - model loading disabled")
    HAS_MGNN = False
    
    class DJMGNN(nn.Module):  # type: ignore
        """Dummy class when MGNN is not available."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("moml.models.mgnn required for model creation")


class MGNNPredictor:
    """
    Predictor for Molecular Graph Neural Networks.

    This class handles loading trained MGNN models and making predictions on 
    molecular data. It supports various input formats and provides both single
    molecule and batch prediction capabilities.

    Attributes:
        model: The loaded neural network model
        config: Configuration dictionary for model and processing
        device: PyTorch device for computations
        graph_processor: Processor for converting molecules to graphs

    Note:
        Either model_path or model must be provided during initialization.
        The predictor automatically detects available devices and sets up
        the model for evaluation mode.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the MGNN predictor.

        Args:
            model_path: Path to saved model checkpoint file
            model: Pre-trained model instance (alternative to model_path)
            config: Configuration dictionary for model and data processing
            device: Device to use for inference ('cpu', 'cuda', etc.)

        Raises:
            ValueError: If neither model_path nor model is provided
            ImportError: If required dependencies are not available
        """
        if model_path is None and model is None:
            raise ValueError("Either model_path or model must be provided")

        if not HAS_CORE:
            raise ImportError("moml.core package required for predictor functionality")

        # Initialize configuration
        self.config = config or {}
        logger.debug(f"Initialized predictor with config keys: {list(self.config.keys())}")

        # Set computation device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")

        # Create graph processor for molecular data
        try:
            self.graph_processor = create_graph_processor(self.config)
            logger.debug("Graph processor created successfully")
        except Exception as e:
            logger.error(f"Failed to create graph processor: {e}")
            raise

        # Load or set the model
        if model is not None:
            self.model = self._setup_provided_model(model)
            logger.info("Using provided model instance")
        else:
            assert model_path is not None 
            self.model = self._load_model_from_path(model_path)
            logger.info(f"Loaded model from: {model_path}")

    def _setup_device(self, device: Optional[str]) -> str:
        """
        Set up computation device with fallback logic.

        Args:
            device: Requested device string

        Returns:
            Device string for PyTorch operations
        """
        if device is not None:
            return device
        
        config_device = self.config.get("device")
        if config_device is not None:
            return config_device
            
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_provided_model(self, model: nn.Module) -> nn.Module:
        """
        Set up a provided model instance for prediction.

        Args:
            model: Pre-trained model instance

        Returns:
            Model ready for inference
        """
        model = model.to(self.device)
        if model.training:
            model.eval()
            logger.debug("Set model to evaluation mode")
        return model

    def _load_model_from_path(self, model_path: str) -> nn.Module:
        """
        Load trained model from checkpoint file.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded and initialized model

        Raises:
            ValueError: If model loading fails
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            logger.debug(f"Loaded checkpoint with keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'tensor'}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

        # Extract model state and configuration
        model_state, config_update = self._extract_checkpoint_data(checkpoint)
        
        # Update configuration with checkpoint data
        if config_update:
            self.config.update(config_update)
            logger.debug("Updated config from checkpoint")

        # Create and load model
        model = self._create_model_from_config(model_state)
        model.load_state_dict(model_state)
        model.eval()

        return model

    def _extract_checkpoint_data(
        self, checkpoint: Union[Dict[str, Any], torch.Tensor]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Extract model state and configuration from checkpoint.

        Args:
            checkpoint: Loaded checkpoint data

        Returns:
            Tuple of (model_state_dict, config_dict)
        """
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                config_update = checkpoint.get("config")
                return model_state, config_update
            else:
                return checkpoint, None
        else:
            return checkpoint, None  # type: ignore

    def _create_model_from_config(self, model_state: Dict[str, Any]) -> nn.Module:
        """
        Create model instance from configuration and state dict.

        Args:
            model_state: Model state dictionary

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model creation fails
            ImportError: If MGNN module not available
        """
        if not HAS_MGNN:
            raise ImportError("moml.models.mgnn required for model creation")

        # Get model dimensions from config or infer from state
        in_dim = self._get_input_dimension(model_state)
        edge_attr_dim = self.config.get("edge_attr_dim", 0)

        model_params = {
            "in_node_dim": in_dim,
            "hidden_dim": self.config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            "n_blocks": self.config.get("n_blocks", DEFAULT_N_BLOCKS),
            "layers_per_block": self.config.get("layers_per_block", DEFAULT_LAYERS_PER_BLOCK),
            "in_edge_dim": edge_attr_dim,
            "jk_mode": self.config.get("jk_mode", "cat"),
            "node_output_dims": self.config.get("node_output_dims", self.config.get("node_out_dim", 1)),
            "graph_output_dims": self.config.get("graph_output_dims", self.config.get("graph_out_dim", 1)),
            "env_dim": self.config.get("env_dim", 0),
            "dropout": self.config.get("dropout", DEFAULT_DROPOUT),
        }

        try:
            model = DJMGNN(**model_params).to(self.device)  # type: ignore
            logger.debug(f"Created DJMGNN model with params: {model_params}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model: {e}")

    def _get_input_dimension(self, model_state: Dict[str, Any]) -> int:
        """
        Infer input dimension from model state or configuration.

        Args:
            model_state: Model state dictionary

        Returns:
            Input dimension for the model

        Raises:
            ValueError: If dimension cannot be determined
        """
        in_dim = self.config.get("in_dim", 0)
        if in_dim > 0:
            return in_dim

        for key, value in model_state.items():
            if "weight" in key and value.dim() == 2:
                inferred_dim = value.shape[1]
                logger.debug(f"Inferred input dimension {inferred_dim} from {key}")
                return inferred_dim

        raise ValueError(
            "Could not determine input dimension. Please specify 'in_dim' in config."
        )

    def predict_from_graph(self, graph) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a molecular graph object.

        Args:
            graph: PyTorch Geometric Data object representing molecular graph

        Returns:
            Dictionary containing prediction results with keys like 'graph_pred', 'node_pred'
        """
        # Move graph to computation device
        graph = graph.to(self.device)

        # Ensure batch dimension exists
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(
                graph.x.size(0), dtype=torch.long, device=self.device
            )
            logger.debug("Added batch dimension to graph")

        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=getattr(graph, "edge_attr", None),
                batch=graph.batch,
            )

        # Process outputs and move to CPU
        return self._process_model_outputs(outputs)

    def _process_model_outputs(self, outputs) -> Dict[str, torch.Tensor]:
        """
        Process model outputs and ensure they are on CPU.

        Args:
            outputs: Raw model outputs (dict or tensor)

        Returns:
            Dictionary with processed outputs on CPU
        """
        result = {}
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    result[key] = value.cpu()
                else:
                    result[key] = value
        elif torch.is_tensor(outputs):
            # Assume single tensor is graph-level prediction
            result["graph_pred"] = outputs.cpu()
        else:
            logger.warning(f"Unexpected output type: {type(outputs)}")
            result["unknown_output"] = outputs

        return result

    def predict_from_file(
        self, file_path: str, charges_file: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a molecule from file.

        Args:
            file_path: Path to molecular structure file (e.g., .mol, .sdf)
            charges_file: Optional path to file with partial charges

        Returns:
            Dictionary containing prediction results

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file processing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Molecule file not found: {file_path}")

        try:
            # Convert file to graph representation
            graph = self.graph_processor.file_to_graph(file_path)  # type: ignore
            logger.debug(f"Processed file: {file_path}")
            
            # Make prediction
            return self.predict_from_graph(graph)
        except Exception as e:
            raise ValueError(f"Failed to process file {file_path}: {e}")

    def predict_from_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a molecule from SMILES string.

        Args:
            smiles: SMILES string representation of molecule

        Returns:
            Dictionary containing prediction results

        Raises:
            ValueError: If SMILES processing fails

        Example:
            >>> predictions = predictor.predict_from_smiles("CCO")
            >>> print(f"Energy: {predictions['graph_pred'].item():.4f}")
        """
        if not smiles or not smiles.strip():
            raise ValueError("SMILES string cannot be empty")

        try:
            # Convert SMILES to graph representation
            graph = self.graph_processor.smiles_to_graph(smiles)  # type: ignore
            logger.debug(f"Processed SMILES: {smiles}")
            
            # Make prediction
            return self.predict_from_graph(graph)
        except Exception as e:
            raise ValueError(f"Failed to process SMILES '{smiles}': {e}")

    def batch_predict(
        self, graphs: List, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions on a batch of molecular graphs.

        Args:
            graphs: List of graph objects to process
            batch_size: Number of graphs to process simultaneously

        Returns:
            List of prediction dictionaries, one per input graph
        """
        if not graphs:
            logger.warning("Empty graph list provided")
            return []

        try:
            from torch_geometric.data import Batch
        except ImportError:
            raise ImportError("torch_geometric required for batch processing")

        # Create data loader for batching
        dataloader = DataLoader(
            graphs,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Batch.from_data_list,
        )

        logger.info(f"Processing {len(graphs)} graphs in batches of {batch_size}")

        # Extract node counts from original graphs for later splitting
        node_counts = [graph.x.shape[0] for graph in graphs]
        logger.debug(f"Node counts per graph: {node_counts}")

        # Get batched predictions
        batch_predictions = self.predict_from_dataloader(dataloader)

        # Split batched results back to individual graphs
        return self._split_batch_predictions(batch_predictions, len(graphs), node_counts)

    def _split_batch_predictions(
        self, batch_predictions: Dict[str, torch.Tensor], num_graphs: int, node_counts: Optional[List[int]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split batched predictions back to individual graph predictions.

        Args:
            batch_predictions: Dictionary with batched predictions
            num_graphs: Number of individual graphs
            node_counts: Optional list of node counts per graph for splitting node predictions

        Returns:
            List of individual prediction dictionaries
        """
        individual_predictions = [{} for _ in range(num_graphs)]

        for pred_type, preds in batch_predictions.items():
            if "graph" in pred_type and preds.shape[0] == num_graphs:
                # Graph-level predictions: one per graph
                for i in range(num_graphs):
                    individual_predictions[i][pred_type] = preds[i].unsqueeze(0)
            elif "node" in pred_type:
                # Node-level predictions: split using node counts
                if node_counts is not None and len(node_counts) == num_graphs:
                    # Split the concatenated node predictions back to individual graphs
                    if preds.shape[0] != sum(node_counts):
                        logger.warning(
                            f"Node prediction tensor size ({preds.shape[0]}) doesn't match "
                            f"total node count ({sum(node_counts)}). Skipping {pred_type}."
                        )
                        continue
                    
                    start_idx = 0
                    for i, count in enumerate(node_counts):
                        if count > 0:
                            end_idx = start_idx + count
                            individual_predictions[i][pred_type] = preds[start_idx:end_idx]
                            start_idx = end_idx
                        else:
                            # Handle empty graphs
                            individual_predictions[i][pred_type] = torch.empty(
                                (0, preds.shape[1]), dtype=preds.dtype, device=preds.device
                            )
                            logger.debug(f"Graph {i} has 0 nodes, assigned empty tensor for {pred_type}")
                else:
                    logger.warning(
                        f"Node counts not provided or mismatched ({node_counts}). "
                        f"Cannot split {pred_type} predictions properly."
                    )
            else:
                # Handle other prediction types (e.g., edge-level, if any)
                logger.debug(f"Unknown prediction type: {pred_type}, copying to all graphs")
                for i in range(num_graphs):
                    individual_predictions[i][pred_type] = preds

        return individual_predictions

    def predict_from_dataloader(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a PyTorch DataLoader.

        Args:
            dataloader: DataLoader containing batched graph data

        Returns:
            Dictionary with concatenated predictions from all batches
        """
        self.model.eval()

        # Storage for different prediction types
        prediction_storage = {
            "node_pred": [],
            "graph_pred": [],
            "node_features": [],
            "graph_features": [],
        }

        logger.info(f"Processing {len(dataloader)} batches")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, "edge_attr", None),
                    batch=batch.batch,
                )

                # Collect different types of outputs
                self._collect_batch_outputs(outputs, prediction_storage)

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

        # Concatenate all predictions
        return self._concatenate_predictions(prediction_storage)

    def _collect_batch_outputs(
        self, outputs, storage: Dict[str, List[torch.Tensor]]
    ) -> None:
        """
        Collect outputs from a single batch into storage lists.

        Args:
            outputs: Model outputs for current batch
            storage: Dictionary to store collected outputs
        """
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key in storage and torch.is_tensor(value):
                    storage[key].append(value.cpu())
        elif torch.is_tensor(outputs):
            # Single tensor output assumed to be graph predictions
            storage["graph_pred"].append(outputs.cpu())

    def _concatenate_predictions(
        self, storage: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate prediction lists into final tensors.

        Args:
            storage: Dictionary with lists of prediction tensors

        Returns:
            Dictionary with concatenated prediction tensors
        """
        results = {}
        
        for key, tensor_list in storage.items():
            if tensor_list:  # Only process non-empty lists
                try:
                    results[key] = torch.cat(tensor_list, dim=0)
                    logger.debug(f"Concatenated {key}: {results[key].shape}")
                except Exception as e:
                    logger.warning(f"Failed to concatenate {key}: {e}")

        return results

    def save_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        output_file: str,
        save_config: bool = True,
    ) -> None:
        """
        Save predictions to JSON file.

        Args:
            predictions: Dictionary with prediction tensors
            output_file: Path for output JSON file
            save_config: Whether to save configuration alongside predictions
        """
        # Create output directory
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Convert tensors to serializable format
        serializable_predictions = self._make_serializable(predictions)

        # Save predictions
        try:
            with open(output_file, "w") as f:
                json.dump(serializable_predictions, f, indent=2)
            logger.info(f"Saved predictions to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise

        # Save configuration if requested
        if save_config:
            config_file = os.path.splitext(output_file)[0] + "_config.json"
            try:
                with open(config_file, "w") as f:
                    json.dump(self.config, f, indent=2)
                logger.info(f"Saved configuration to: {config_file}")
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")

    def _make_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert data to JSON-serializable format.

        Args:
            data: Dictionary potentially containing tensors

        Returns:
            Dictionary with tensors converted to lists
        """
        serializable = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                serializable[key] = value.tolist()
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable[key] = value
            else:
                logger.warning(f"Cannot serialize {key} of type {type(value)}")
                serializable[key] = str(value)

        return serializable


def create_predictor(
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
) -> MGNNPredictor:
    """
    Factory function to create an MGNN predictor instance.

    Args:
        config: Configuration dictionary for model and processing
        model_path: Optional path to saved model checkpoint
        model: Optional pre-created model instance

    Returns:
        Configured MGNNPredictor instance

    Raises:
        ValueError: If neither model_path nor model is provided
    """
    if model is None and model_path is None:
        raise ValueError("Either model_path or model must be provided")

    if model is None and model_path is not None:
        # Load model from checkpoint with enhanced error handling
        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            
            if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
                model = DJMGNN(**model_config)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model from checkpoint with embedded config")
            else:
                # Fallback: create model from provided config
                if not config:
                    raise ValueError("Config required when checkpoint lacks model_config")
                model = DJMGNN(
                    in_node_dim=config.get("in_node_dim", config.get("in_dim", 1)),
                    hidden_dim=config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
                    n_blocks=config.get("n_blocks", DEFAULT_N_BLOCKS),
                    layers_per_block=config.get("layers_per_block", DEFAULT_LAYERS_PER_BLOCK),
                    in_edge_dim=config.get("in_edge_dim", 0),
                    jk_mode=config.get("jk_mode", "cat"),
                    node_output_dims=config.get("node_output_dims", config.get("node_out_dim", 1)),
                    graph_output_dims=config.get("graph_output_dims", config.get("graph_out_dim", 1)),
                    env_dim=config.get("env_dim", 0),
                    dropout=config.get("dropout", DEFAULT_DROPOUT),
                )
                
                # Load state dict handling both direct and nested formats
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
                logger.info("Created model from config and loaded weights")
                
        except Exception as e:
            logger.error(f"Failed to create predictor from checkpoint: {e}")
            raise ValueError(f"Could not load model from {model_path}: {e}")

    return MGNNPredictor(model=model, config=config)


def batch_predict_from_files(
    model_path: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Make predictions on a directory of molecular structure files.

    Args:
        model_path: Path to trained model checkpoint
        input_dir: Directory containing molecular structure files
        output_dir: Optional directory to save prediction results
        config: Configuration dictionary for model and processing
        file_pattern: Glob pattern to match files (e.g., "*.mol,*.sdf")
        batch_size: Number of molecules to process simultaneously
        device: Device for computations ('cpu', 'cuda', etc.)

    Returns:
        Dictionary mapping filenames to their prediction results
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create predictor instance
    logger.info(f"Creating predictor from model: {model_path}")
    predictor = MGNNPredictor(model_path=model_path, config=config, device=device)

    # Find molecular structure files
    molecule_files = _find_molecule_files(input_dir, file_pattern)
    logger.info(f"Found {len(molecule_files)} molecular files")

    if not molecule_files:
        raise ValueError(
            f"No files found in {input_dir} matching pattern '{file_pattern}'"
        )

    # Process files and create graphs
    graphs, filenames = _process_molecule_files(molecule_files, predictor)
    
    if not graphs:
        raise ValueError("No valid molecular graphs could be created")

    logger.info(f"Successfully processed {len(graphs)} molecules")

    # Generate predictions
    logger.info("Generating predictions...")
    try:
        predictions = predictor.batch_predict(graphs, batch_size)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise

    # Organize results by filename
    results = _organize_prediction_results(predictions, filenames)

    # Save results if output directory specified
    if output_dir is not None:
        _save_batch_results(results, output_dir)

    logger.info(f"Completed predictions for {len(results)} molecules")
    return results


def _find_molecule_files(input_dir: str, file_pattern: str) -> List[str]:
    """
    Find molecular structure files matching the specified pattern.

    Args:
        input_dir: Directory to search
        file_pattern: Comma-separated glob patterns

    Returns:
        List of matching file paths
    """
    molecule_files = []
    
    # Split pattern by commas and process each
    patterns = [pattern.strip() for pattern in file_pattern.split(",")]
    
    for pattern in patterns:
        search_pattern = os.path.join(input_dir, pattern)
        found_files = glob.glob(search_pattern)
        molecule_files.extend(found_files)
        logger.debug(f"Pattern '{pattern}' found {len(found_files)} files")

    # Remove duplicates and sort
    molecule_files = sorted(set(molecule_files))
    return molecule_files


def _process_molecule_files(
    molecule_files: List[str], predictor: MGNNPredictor
) -> tuple[List, List[str]]:
    """
    Process molecular files into graph representations.

    Args:
        molecule_files: List of file paths to process
        predictor: Predictor instance with graph processor

    Returns:
        Tuple of (graphs_list, successful_filenames)
    """
    graphs = []
    filenames = []

    for file_path in tqdm(molecule_files, desc="Processing molecular files"):
        try:
            # Convert file to graph
            graph = predictor.graph_processor.file_to_graph(file_path)  # type: ignore
            graphs.append(graph)
            filenames.append(os.path.basename(file_path))
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue

    logger.info(f"Successfully processed {len(graphs)}/{len(molecule_files)} files")
    return graphs, filenames


def _organize_prediction_results(
    predictions: List[Dict[str, torch.Tensor]], filenames: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Organize prediction results by filename with serializable format.

    Args:
        predictions: List of prediction dictionaries
        filenames: List of corresponding filenames

    Returns:
        Dictionary mapping filenames to serializable predictions
    """
    results = {}
    
    for i, filename in enumerate(filenames):
        if i < len(predictions):
            # Convert tensors to lists for JSON serialization
            serializable_predictions = {}
            for key, value in predictions[i].items():
                if torch.is_tensor(value):
                    serializable_predictions[key] = value.tolist()
                else:
                    serializable_predictions[key] = value
                    
            results[filename] = serializable_predictions
        else:
            logger.warning(f"No prediction available for {filename}")

    return results


def _save_batch_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Save batch prediction results to JSON file.

    Args:
        results: Dictionary of prediction results
        output_dir: Directory to save results

    Raises:
        OSError: If directory creation or file writing fails
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined predictions
        output_file = os.path.join(output_dir, "predictions.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved {len(results)} predictions to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save batch results: {e}")
        raise
