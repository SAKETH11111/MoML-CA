"""
Predictor module for DJMGNN.

This module provides functionality for making predictions with trained models.
"""

import os
import torch
import json
import glob
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch.nn as nn

from moml.core import create_graph_processor
from moml.models.mgnn import DJMGNN


class MGNNPredictor:
    """
    Predictor for molecular graph neural networks.

    This class handles loading a trained model and making predictions on new data.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the saved model checkpoint (optional if model is provided)
            model: Pre-trained model instance (optional if model_path is provided)
            config: Configuration dictionary for the model and data processing
            device: Device to use for inference
        """
        if model_path is None and model is None:
            raise ValueError("Either model_path or model must be provided")

        # Initialize config
        self.config = config or {}

        # Set device
        self.device = (
            device
            if device is not None
            else self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create graph processor
        self.graph_processor = create_graph_processor(self.config)

        # Load or set the model
        if model is not None:
            # Use the provided model directly
            self.model = model.to(self.device)
            if not self.model.training:
                self.model.eval()
        else:
            # Load the model from a file
            self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the trained model from a file.

        Args:
            model_path: Path to the model checkpoint

        Returns:
            Loaded model
        """
        # Check if the file is a checkpoint or just model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # It's a checkpoint with metadata
                model_state = checkpoint["model_state_dict"]

                # Check if config is included in the checkpoint
                if "config" in checkpoint:
                    self.config.update(checkpoint["config"])
            else:
                # It's just the model weights
                model_state = checkpoint
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}")

        # Get model dimensions from config
        in_dim = self.config.get("in_dim", 0)
        edge_attr_dim = self.config.get("edge_attr_dim", 0)

        # If dimensions are not in config, try to infer from the processor
        if in_dim == 0:
            # Infer from the model's first linear layer weight
            # Infer from the model's first linear layer weight by searching for a key containing 'weight'
            for key, value in model_state.items():
                if "weight" in key and value.dim() == 2: # Assuming linear layer weights are 2D
                    in_dim = value.shape[1]
                    break
            if in_dim == 0: # Fallback if no suitable weight found
                raise ValueError("Could not infer input dimension from model state. Please specify 'in_dim' in config.")
        if edge_attr_dim == 0:
            # This is trickier; may need to be stored in config or checkpoint
            # For now, assume it's 0 if not specified
            pass

        # Initialize model
        model = DJMGNN(
            in_node_dim=in_dim,
            hidden_dim=self.config.get("hidden_dim", 64),
            n_blocks=self.config.get("n_blocks", 3),
            layers_per_block=self.config.get("layers_per_block", 2),
            in_edge_dim=edge_attr_dim,
            jk_mode=self.config.get("jk_mode", "cat"),
            node_out_dim=self.config.get("node_out_dim", 1),
            graph_out_dim=self.config.get("graph_out_dim", 1),
            env_dim=self.config.get("env_dim", 0), # Assuming env_dim can be 0 if not specified
            dropout=self.config.get("dropout", 0.2),
        ).to(self.device)

        # Load weights
        model.load_state_dict(model_state)
        model.eval()

        return model

    def predict_from_graph(self, graph) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a molecular graph.

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Dictionary with predictions
        """
        # Move graph to device
        graph = graph.to(self.device)

        # Ensure batch dimension is set
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                x=graph.x, edge_index=graph.edge_index, edge_attr=getattr(graph, "edge_attr", None), batch=graph.batch
            )

        # Ensure all outputs are on CPU
        result = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                result[key] = value.cpu()
        elif torch.is_tensor(outputs):
            # If not a dict, assume it's graph-level predictions
            result["graph_pred"] = outputs.cpu()

        return result

    def predict_from_file(self, file_path: str, charges_file: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a molecule file.

        Args:
            file_path: Path to the molecule file
            charges_file: Optional path to a file with partial charges

        Returns:
            Dictionary with predictions
        """
        # Use the processor to create a graph from the file
        graph = self.graph_processor.file_to_graph(file_path)

        # Make prediction
        return self.predict_from_graph(graph)

    def predict_from_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with predictions
        """
        # Use the processor to create a graph from the SMILES
        graph = self.graph_processor.smiles_to_graph(smiles)

        # Make prediction
        return self.predict_from_graph(graph)

    def batch_predict(self, graphs: List, batch_size: int = 32) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions on a batch of graphs.

        Args:
            graphs: List of graph objects
            batch_size: Batch size for inference

        Returns:
            List of dictionaries with predictions
        """
        from torch_geometric.data import Batch

        dataloader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Batch.from_data_list,
        )
        
        predictions = self.predict_from_dataloader(dataloader)
        
        num_graphs = len(graphs)
        output_list = [{} for _ in range(num_graphs)]
        
        for pred_type, preds in predictions.items():
            if 'graph' in pred_type:
                for i in range(num_graphs):
                    output_list[i][pred_type] = preds[i].unsqueeze(0)
            elif 'node' in pred_type:
                # This part is more complex and depends on batch info
                # For simplicity, let's assume predict_from_dataloader can return a list of dicts
                pass # Needs more detailed implementation based on how node preds are batched

        if not any('graph' in k for k in predictions.keys()):
             # Fallback for models that return a single tensor for the batch
            if 'graph_pred' in predictions and predictions['graph_pred'].shape[0] == num_graphs:
                for i in range(num_graphs):
                    output_list[i]['graph_pred'] = predictions['graph_pred'][i].unsqueeze(0)
        return output_list

    def predict_from_dataloader(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a DataLoader.

        Args:
            dataloader: DataLoader with graphs

        Returns:
            Dictionary with predictions
        """
        self.model.eval()

        node_preds = []
        graph_preds = []
        node_features = []
        graph_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move to device
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, "edge_attr", None),
                    batch=batch.batch,
                )

                # Collect predictions
                if isinstance(outputs, dict):
                    if "node_pred" in outputs:
                        node_preds.append(outputs["node_pred"].cpu())
                    if "graph_pred" in outputs:
                        graph_preds.append(outputs["graph_pred"].cpu())
                    if "node_features" in outputs:
                        node_features.append(outputs["node_features"].cpu())
                    if "graph_features" in outputs:
                        graph_features.append(outputs["graph_features"].cpu())
                else:
                    pass
                if torch.is_tensor(outputs):
                    # If not a dict, assume it's graph-level predictions
                    graph_preds.append(outputs.cpu())

        # Concatenate predictions
        results = {}

        if node_preds:
            results["node_pred"] = torch.cat(node_preds, dim=0)

        if graph_preds:
            results["graph_pred"] = torch.cat(graph_preds, dim=0)

        if node_features:
            results["node_features"] = torch.cat(node_features, dim=0)

        if graph_features:
            results["graph_features"] = torch.cat(graph_features, dim=0)

        return results

    def save_predictions(
        self, predictions: Dict[str, torch.Tensor], output_file: str, save_config: bool = True
    ) -> None:
        """
        Save predictions to a file.

        Args:
            predictions: Dictionary with predictions
            output_file: Path to save the predictions
            save_config: Whether to also save the configuration
        """
        # Create output directory if it doesn't exist
        dirpath = os.path.dirname(output_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # Convert tensors to lists
        serializable_preds = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                serializable_preds[key] = value.tolist()
            else:
                serializable_preds[key] = value

        # Save predictions
        with open(output_file, "w") as f:
            json.dump(serializable_preds, f, indent=2)

        # Save configuration if requested
        if save_config:
            config_file = os.path.splitext(output_file)[0] + "_config.json"
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)


def create_predictor(config: Dict, model_path: Optional[str] = None, model: Optional[nn.Module] = None) -> MGNNPredictor:
    """Create a predictor instance with the given configuration.
    
    Args:
        config: Configuration dictionary
        model_path: Optional path to saved model checkpoint
        model: Optional pre-created model instance. If provided, this model will be used instead of loading from model_path.
    
    Returns:
        MGNNPredictor instance
    """
    if model is None and model_path is None:
        raise ValueError("Either model_path or model must be provided")
    
    if model is None:
        if model_path is None:
            raise ValueError("model_path cannot be None when model is not provided.")
        # Load model from checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = DJMGNN(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return MGNNPredictor(model=model, config=config)


def batch_predict_from_files(
    model_path: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    file_pattern: str = "*.mol",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Make predictions on a directory of molecule files.

    Args:
        model_path: Path to the model checkpoint
        input_dir: Directory with molecule files
        output_dir: Optional directory to save predictions
        config: Configuration for the model and data processing
        file_pattern: Pattern to match molecule files
        batch_size: Batch size for inference
        device: Device to use for inference

    Returns:
        Dictionary with predictions
    """
    # Create predictor
    predictor = MGNNPredictor(model_path=model_path, config=config, device=device)

    # Find all molecule files
    molecule_files = []
    for pattern in file_pattern.split(","):
        molecule_files.extend(glob.glob(os.path.join(input_dir, pattern.strip())))

    if not molecule_files:
        raise ValueError(f"No files found in {input_dir} matching pattern {file_pattern}")

    # Process files in batches
    graphs = []
    filenames = []

    for file_path in tqdm(molecule_files, desc="Processing files"):
        try:
            # Create graph from file
            graph = predictor.graph_processor.file_to_graph(file_path)
            graphs.append(graph)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error processing {file_path}")

    # Make predictions
    print(f"Making predictions on {len(graphs)} molecules...")
    predictions = predictor.batch_predict(graphs, batch_size)

    # Organize predictions by filename
    results = {}
    for i, filename in enumerate(filenames):
        if i < len(predictions):
            # Convert tensors to lists for JSON serialization
            serializable_preds = {}
            for key, value in predictions[i].items():
                if isinstance(value, torch.Tensor):
                    serializable_preds[key] = value.tolist()
                else:
                    serializable_preds[key] = value
            results[filename] = serializable_preds

    # Save predictions if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save combined predictions
        combined_file = os.path.join(output_dir, "predictions.json")
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)

    return results
