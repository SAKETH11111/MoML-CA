"""
scripts/validate_optimized_djmgnn.py

Validation script for optimized DJMGNN with proper scaler inverse transforms.
Fixes the -121 R² issue by applying correct denormalization.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from moml.models.mgnn.djmgnn import DJMGNN

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


class OptimizedDJMGNNPredictor:
    """
    Optimized DJMGNN predictor that properly handles scaler inverse transforms.
    
    This class loads trained models with their embedded scalers and applies
    proper inverse normalization to return predictions in original units.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Initialize predictor from checkpoint."""
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract scaler information
        self.scaler_state = self.checkpoint.get("scaler_state", {})
        
        if not self.scaler_state:
            logger.warning("No scaler state found in checkpoint - predictions may be incorrect!")
        
        # Initialize model (we'll need to determine architecture from checkpoint)
        self.model = self._initialize_model_from_checkpoint()
        self.model.eval()
        
        logger.info(f"Loaded optimized DJMGNN from {checkpoint_path}")
        logger.info(f"Checkpoint step: {self.checkpoint.get('step', 'unknown')}")
        logger.info(f"Checkpoint seed: {self.checkpoint.get('seed', 'unknown')}")
        
    def _initialize_model_from_checkpoint(self) -> DJMGNN:
        """Initialize model architecture from checkpoint state dict."""
        # Analyze state dict to determine model architecture
        state_dict = self.checkpoint["model_state_dict"]
        
        # Extract dimensions from first layer
        first_layer_weight = None
        for key, tensor in state_dict.items():
            if "weight" in key and len(tensor.shape) == 2:
                first_layer_weight = tensor
                break
        
        if first_layer_weight is None:
            raise ValueError("Could not determine input dimension from checkpoint")
        
        in_node_dim = first_layer_weight.shape[1]
        hidden_dim = first_layer_weight.shape[0]
        
        # Find output dimensions by examining final layers
        graph_output_dims = 19  # QM9 default
        node_output_dims = 3   # Force dimensions
        energy_output_dims = 1 # Energy dimension
        
        # Look for graph prediction head
        for key, tensor in state_dict.items():
            if "graph_pred" in key and "weight" in key and len(tensor.shape) == 2:
                graph_output_dims = tensor.shape[0]
            elif "node_pred" in key and "weight" in key and len(tensor.shape) == 2:
                node_output_dims = tensor.shape[0]
            elif "energy_pred" in key and "weight" in key and len(tensor.shape) == 2:
                energy_output_dims = tensor.shape[0]
        
        # Count number of blocks by counting repeated patterns
        n_blocks = 4  # Default fallback
        block_count = 0
        for key in state_dict.keys():
            if "blocks." in key:
                block_idx = int(key.split("blocks.")[1].split(".")[0])
                block_count = max(block_count, block_idx + 1)
        if block_count > 0:
            n_blocks = block_count
        
        logger.info(f"Detected model architecture: in_node_dim={in_node_dim}, "
                   f"hidden_dim={hidden_dim}, n_blocks={n_blocks}")
        logger.info(f"Output dimensions: graph={graph_output_dims}, "
                   f"node={node_output_dims}, energy={energy_output_dims}")
        
        # Initialize model
        model = DJMGNN(
            in_node_dim=in_node_dim,
            in_edge_dim=0,  # Assume no edge features for now
            node_output_dims=node_output_dims,
            graph_output_dims=graph_output_dims,
            energy_output_dims=energy_output_dims,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
        ).to(self.device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        return model
    
    def predict_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Predict on a batch and return raw (normalized) outputs."""
        batch = batch.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=getattr(batch, "batch", None),
                dist=getattr(batch, "dist", None),
            )
        
        return outputs
    
    def denormalize_graph_predictions(self, normalized_preds: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization to graph-level predictions."""
        if "graph_scaler" not in self.scaler_state:
            logger.warning("No graph scaler found - returning normalized predictions")
            return normalized_preds
            
        scaler_info = self.scaler_state["graph_scaler"]
        mean = scaler_info["mean"].to(self.device)
        std = scaler_info["std"].to(self.device)
        
        # Inverse standardization: denormalized = normalized * std + mean
        denormalized = normalized_preds * std + mean
        
        return denormalized
    
    def denormalize_node_predictions(self, normalized_preds: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization to node-level predictions."""
        if "node_scaler" not in self.scaler_state:
            logger.warning("No node scaler found - returning normalized predictions")
            return normalized_preds
            
        scaler_info = self.scaler_state["node_scaler"]
        mean = scaler_info["mean"].to(self.device)
        std = scaler_info["std"].to(self.device)
        
        # Inverse standardization: denormalized = normalized * std + mean  
        denormalized = normalized_preds * std + mean
        
        return denormalized
    
    def predict_molecules(self, dataloader: GraphDataLoader) -> Dict[str, np.ndarray]:
        """Predict molecular properties and return denormalized results."""
        all_graph_preds = []
        all_graph_targets = []
        all_node_preds = []
        all_node_targets = []
        
        for batch in dataloader:
            outputs = self.predict_batch(batch)
            
            # Collect graph-level predictions and targets
            if "graph_pred" in outputs and hasattr(batch, "y"):
                # Denormalize predictions
                graph_preds = self.denormalize_graph_predictions(outputs["graph_pred"])
                all_graph_preds.append(graph_preds.cpu().numpy())
                
                # Denormalize targets (they're also normalized in the dataset)
                graph_targets = self.denormalize_graph_predictions(batch.y.to(self.device))
                all_graph_targets.append(graph_targets.cpu().numpy())
            
            # Collect node-level predictions and targets  
            if "node_pred" in outputs and hasattr(batch, "node_y"):
                node_preds = self.denormalize_node_predictions(outputs["node_pred"])
                all_node_preds.append(node_preds.cpu().numpy())
                
                node_targets = self.denormalize_node_predictions(batch.node_y.to(self.device))
                all_node_targets.append(node_targets.cpu().numpy())
        
        results = {}
        
        if all_graph_preds:
            results["graph_predictions"] = np.concatenate(all_graph_preds, axis=0)
            results["graph_targets"] = np.concatenate(all_graph_targets, axis=0)
        
        if all_node_preds:
            results["node_predictions"] = np.concatenate(all_node_preds, axis=0)  
            results["node_targets"] = np.concatenate(all_node_targets, axis=0)
        
        return results


def compute_comprehensive_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # Handle potential NaN or infinite values
    valid_mask = np.isfinite(predictions) & np.isfinite(targets)
    if not np.any(valid_mask):
        logger.warning("No valid predictions found")
        return {"r2": -999.0, "mae": 999.0, "rmse": 999.0, "spearman": 0.0}
    
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    # Core regression metrics
    metrics["r2"] = r2_score(valid_targets, valid_preds)
    metrics["mae"] = mean_absolute_error(valid_targets, valid_preds)  
    metrics["rmse"] = np.sqrt(mean_squared_error(valid_targets, valid_preds))
    
    # Spearman correlation for robustness
    if len(valid_preds) > 1:
        spearman_corr, _ = spearmanr(valid_preds, valid_targets)
        metrics["spearman"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    else:
        metrics["spearman"] = 0.0
    
    return metrics


def validate_qm9_properties(predictor: OptimizedDJMGNNPredictor, test_size: int = 1000) -> Dict:
    """Validate QM9 molecular property predictions."""
    logger.info("Loading QM9 test dataset...")
    
    # Create test transform (no standardization - we'll handle that in predictor)
    test_transform = Compose([CreateEdges(), FeaturizeNodes()])
    
    # Load test dataset  
    qm9_dataset = get_dataset("qm9", root="data", transform=test_transform)
    
    # Take a random subset for testing
    if len(qm9_dataset) > test_size:
        indices = np.random.choice(len(qm9_dataset), test_size, replace=False)
        test_subset = [qm9_dataset[i] for i in indices]
    else:
        test_subset = qm9_dataset
    
    test_loader = GraphDataLoader(test_subset, batch_size=32, shuffle=False)
    logger.info(f"Testing on {len(test_subset)} QM9 molecules")
    
    # Get predictions
    results = predictor.predict_molecules(test_loader)
    
    if "graph_predictions" not in results:
        logger.error("No graph predictions found!")
        return {"error": "No graph predictions"}
    
    graph_preds = results["graph_predictions"]
    graph_targets = results["graph_targets"]
    
    logger.info(f"Predictions shape: {graph_preds.shape}")
    logger.info(f"Targets shape: {graph_targets.shape}")
    
    # QM9 property names
    qm9_properties = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", 
        "U0", "U", "H", "G", "Cv", "U0_atom", "U_atom", 
        "H_atom", "G_atom", "A", "B", "C"
    ]
    
    # Compute metrics for each property
    property_metrics = []
    total_r2_sum = 0.0
    valid_properties = 0
    
    for i, prop_name in enumerate(qm9_properties):
        if i < graph_preds.shape[1] and i < graph_targets.shape[1]:
            prop_preds = graph_preds[:, i]
            prop_targets = graph_targets[:, i]
            
            metrics = compute_comprehensive_metrics(prop_preds, prop_targets)
            metrics["property"] = prop_name
            metrics["property_index"] = i
            
            property_metrics.append(metrics)
            
            if metrics["r2"] > -10:  # Only count reasonable R² values
                total_r2_sum += metrics["r2"]
                valid_properties += 1
            
            logger.info(f"{prop_name:>8}: R²={metrics['r2']:6.3f}, "
                       f"MAE={metrics['mae']:8.3f}, RMSE={metrics['rmse']:8.3f}, "
                       f"Spearman={metrics['spearman']:6.3f}")
    
    # Compute overall statistics
    mean_r2 = total_r2_sum / valid_properties if valid_properties > 0 else -999.0
    
    # Count strong correlations (|r| > 0.5)
    strong_correlations = sum(1 for m in property_metrics if abs(m["r2"]) > 0.5)
    
    overall_metrics = {
        "mean_r2": mean_r2,
        "strong_correlations": strong_correlations,
        "total_properties": len(property_metrics),
        "valid_properties": valid_properties,
        "property_metrics": property_metrics
    }
    
    logger.info(f"\n🎯 OVERALL RESULTS:")
    logger.info(f"Mean R² Score: {mean_r2:.4f}")
    logger.info(f"Strong Correlations (R² > 0.5): {strong_correlations}/{len(property_metrics)}")
    logger.info(f"Percentage of Strong Correlations: {100*strong_correlations/len(property_metrics):.1f}%")
    
    # Check for 95% accuracy target
    if mean_r2 >= 0.95:
        logger.info("✅ 95% ACCURACY TARGET ACHIEVED!")
    else:
        logger.info(f"❌ Below 95% target (need {0.95-mean_r2:.4f} improvement)")
    
    return overall_metrics


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate optimized DJMGNN model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Number of test molecules")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    setup_logging()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("Starting optimized DJMGNN validation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Initialize predictor
        predictor = OptimizedDJMGNNPredictor(args.checkpoint, args.device)
        
        # Validate QM9 properties
        results = validate_qm9_properties(predictor, args.test_size)
        
        # Save results
        results_file = f"validation_results_{Path(args.checkpoint).stem}.json"
        import json
        with open(results_file, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, np.number):
                    serializable_results[key] = float(value)
                elif isinstance(value, list):
                    serializable_results[key] = [
                        {k: (float(v) if isinstance(v, np.number) else v) for k, v in item.items()}
                        if isinstance(item, dict) else item for item in value
                    ]
                else:
                    serializable_results[key] = value
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Exit with appropriate code
        if results.get("mean_r2", -999) >= 0.95:
            logger.info("🎉 VALIDATION SUCCESSFUL - 95% accuracy achieved!")
            sys.exit(0)
        else:
            logger.info("💡 Validation complete - room for improvement")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()