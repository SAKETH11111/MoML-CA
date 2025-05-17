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
        self.processor = create_graph_processor(self.config)

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
            raise ValueError(f"Failed to load model from {model_path}: {e}")

        # Get model dimensions from config
        in_dim = self.config.get("in_dim", 0)
        edge_attr_dim = self.config.get("edge_attr_dim", 0)

        # If dimensions are not in config, try to infer from the processor
        if in_dim == 0 or edge_attr_dim == 0:
            # Create a dummy molecule to determine dimensions
            import rdkit.Chem as Chem

            mol = Chem.MolFromSmiles("C")
            graph = self.processor.mol_to_graph(mol)

            if in_dim == 0:
                in_dim = graph.x.shape[1] if hasattr(graph, "x") else 0

            if edge_attr_dim == 0:
                edge_attr_dim = graph.edge_attr.shape[1] if hasattr(graph, "edge_attr") else 0

        # Initialize model
        model = DJMGNN(
            in_dim=in_dim,
            hidden_dim=self.config.get("hidden_dim", 64),
            n_blocks=self.config.get("n_blocks", 3),
            layers_per_block=self.config.get("layers_per_block", 2),
            edge_attr_dim=edge_attr_dim,
            jk_mode=self.config.get("jk_mode", "cat"),
            node_out_dim=self.config.get("node_out_dim", 1),
            graph_out_dim=self.config.get("graph_out_dim", 1),
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
        else:
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
        graph = self.processor.file_to_graph(file_path)

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
        graph = self.processor.smiles_to_graph(smiles)

        # Make prediction
        return self.predict_from_graph(graph)

    def batch_predict(self, graphs: List, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a batch of graphs.

        Args:
            graphs: List of graph objects
            batch_size: Batch size for inference

        Returns:
            Dictionary with predictions
        """
        from torch_geometric.data import Batch  # Changed import

        # Create a custom dataset for compatibility with standard DataLoader
        class GraphDataset:
            def __init__(self, graphs):
                self.graphs = graphs

            def __len__(self):
                return len(self.graphs)

            def __getitem__(self, idx):
                return self.graphs[idx]

        # Create a DataLoader with appropriate collate function
        dataset = GraphDataset(graphs)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Batch.from_data_list,  # Use PyG Batch for collation
        )

        # Make predictions
        return self.predict_from_dataloader(dataloader)

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


def create_predictor(
    model_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> MGNNPredictor:
    """
    Create a predictor for a trained model.

    Args:
        model_path: Path to the model checkpoint (optional if model is provided)
        model: The trained model instance (optional if model_path is provided)
        config: Configuration for the model and data processing
        device: Device to use for inference

    Returns:
        Configured predictor
    """
    if model_path is None and model is None:
        raise ValueError("Either model_path or model must be provided")

    # Create default config if not provided
    if config is None:
        config = {}

    # Create predictor with model path or direct model
    if model is not None:
        # Initialize predictor with model directly
        predictor = MGNNPredictor(model=model, config=config, device=device)
    else:
        # Initialize predictor with model path
        predictor = MGNNPredictor(model_path=model_path, config=config, device=device)

    return predictor


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
    predictor = create_predictor(model_path=model_path, config=config, device=device)

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
            graph = predictor.processor.file_to_graph(file_path)
            graphs.append(graph)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Make predictions
    print(f"Making predictions on {len(graphs)} molecules...")
    predictions = predictor.batch_predict(graphs, batch_size)

    # Organize predictions by filename
    results = {}

    # For graph-level predictions
    if "graph_pred" in predictions:
        graph_preds = predictions["graph_pred"]
        for i, filename in enumerate(filenames):
            if i < len(graph_preds):
                results[filename] = {"graph_pred": graph_preds[i].tolist()}

    # Save predictions if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save combined predictions
        combined_file = os.path.join(output_dir, "combined_predictions.json")
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save individual predictions
        for filename, preds in results.items():
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.json")
            with open(output_file, "w") as f:
                json.dump(preds, f, indent=2)

    return results
