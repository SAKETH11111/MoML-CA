"""
examples/joint_training_example.py

Example script demonstrating joint DJMGNN and HMGNN training.

This script provides a comprehensive example of how to use the joint training
framework for molecular property prediction using both dense and hierarchical
graph neural networks.

Usage:
    python examples/joint_training_example.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.core.hierarchical_processor import create_hierarchical_processor
from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from moml.models.mgnn import (
    create_joint_mgnn, 
    create_joint_trainer,
    EarlyStopping,
    ModelCheckpoint
)


def create_example_config():
    """Create example configuration for joint training."""
    config = {
        # Model configurations
        "djmgnn": {
            "in_node_dim": 29,
            "hidden_dim": 64,
            "n_blocks": 2,
            "layers_per_block": 2,
            "in_edge_dim": 0,
            "jk_mode": "attention",
            "node_output_dims": 1,
            "graph_output_dims": 19,
            "energy_output_dims": 1,
            "dropout": 0.1,
        },
        
        "hmgnn": {
            "scale_dims": [29, 29, 29],
            "hidden_dim": 64,
            "n_blocks": 2,
            "layers_per_block": 2,
            "jk_mode": "attention",
            "node_out_dim": 1,
            "graph_out_dim": 19,
            "cross_scale_exchange": True,
            "dropout": 0.2,
            "n_heads_cs": 4,
        },
        
        "joint": {
            "fusion_dim": 128,
            "n_fusion_heads": 4,
            "alpha": 0.5,
            "cross_model_weight": 0.1,
        },
        
        # Training configuration
        "training_strategy": "joint",  # or "alternating"
        "epochs": 5,  # Small number for example
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        
        # Task configurations
        "task_configs": {
            "qm9_properties": {"output_dim": 19, "hidden_dims": [64]},
            "force_field": {"output_dim": 1, "hidden_dims": [64]},
        },
        
        "task_weights": {
            "qm9_properties": 1.0,
            "force_field": 1.0,
        },
        
        # Hierarchical processing
        "hierarchical": {
            "coarsener": {
                "n_levels": 3,
                "clustering_method": "graclus",  # Use graclus for simplicity
                "preserve_connectivity": True,
            },
            "processor": {
                "include_cross_scale_edges": True,
                "cache_hierarchical": False,  # Disable caching for example
            }
        },
        
        # Alternating training (if using alternating strategy)
        "alternating_config": {
            "strategy": "fixed_alternating",
            "switch_frequency": 5,
        }
    }
    
    return config


class SimpleHierarchicalDataset:
    """Simple dataset wrapper for hierarchical processing."""
    
    def __init__(self, base_dataset, hierarchical_processor):
        self.base_dataset = base_dataset
        self.hierarchical_processor = hierarchical_processor
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        base_data = self.base_dataset[idx]
        
        # Create hierarchical representation
        hierarchical_data = self.hierarchical_processor.process_molecule(
            base_data, mol=None, molecule_id=f"example_mol_{idx}"
        )
        
        # Add original data
        hierarchical_data['original_data'] = base_data
        if hasattr(base_data, 'y'):
            hierarchical_data['targets'] = base_data.y
        
        return hierarchical_data


def main():
    """Run joint training example."""
    print("Joint DJMGNN and HMGNN Training Example")
    print("=" * 50)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create configuration
    config = create_example_config()
    print("Created training configuration")
    
    # Load a small subset of QM9 dataset for example
    print("Loading QM9 dataset...")
    try:
        transform = Compose([
            CreateEdges(),
            FeaturizeNodes(),
            StandardizeTargets(dataset_name="qm9")
        ])
        
        dataset = get_dataset("qm9", root="data", transform=transform)
        print(f"Loaded dataset with {len(dataset)} molecules")
        
        # Use only a small subset for this example
        subset_size = min(100, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(subset_size))
        print(f"Using subset of {subset_size} molecules for example")
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using synthetic data for example...")
        
        # Create synthetic data for demonstration
        from torch_geometric.data import Data
        
        synthetic_data = []
        for i in range(20):
            # Create small random molecular graphs
            num_nodes = torch.randint(5, 15, (1,)).item()
            x = torch.randn(num_nodes, 29)  # Node features
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            y = torch.randn(19)  # Graph-level targets
            
            data = Data(x=x, edge_index=edge_index, y=y)
            synthetic_data.append(data)
        
        # Split synthetic data
        train_dataset = synthetic_data[:16]
        val_dataset = synthetic_data[16:]
        dataset = synthetic_data
    
    # Create hierarchical processor
    print("Creating hierarchical processor...")
    hierarchical_processor = create_hierarchical_processor(config["hierarchical"])
    
    # Create hierarchical datasets
    hierarchical_train_dataset = SimpleHierarchicalDataset(train_dataset, hierarchical_processor)
    hierarchical_val_dataset = SimpleHierarchicalDataset(val_dataset, hierarchical_processor)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = GraphDataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0  # Use 0 for simplicity
    )
    
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    hierarchical_train_loader = GraphDataLoader(
        hierarchical_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    hierarchical_val_loader = GraphDataLoader(
        hierarchical_val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Create joint model
    print("Creating joint model...")
    joint_model = create_joint_mgnn(
        djmgnn_config=config["djmgnn"],
        hmgnn_config=config["hmgnn"],
        joint_config=config["joint"]
    )
    
    print(f"Joint model created with {sum(p.numel() for p in joint_model.parameters())} parameters")
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=3, min_delta=1e-4),
        # ModelCheckpoint(filepath="example_checkpoint.pt", save_best_only=True)
    ]
    
    # Create joint trainer
    print("Creating joint trainer...")
    try:
        joint_trainer = create_joint_trainer(
            djmgnn_config=config["djmgnn"],
            hmgnn_config=config["hmgnn"],
            joint_config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            hierarchical_train_loader=hierarchical_train_loader,
            hierarchical_val_loader=hierarchical_val_loader,
            device=device
        )
        
        print("Joint trainer created successfully")
        
    except Exception as e:
        print(f"Failed to create joint trainer: {e}")
        print("This is expected if some dependencies are missing")
        return
    
    # Train the model
    print("Starting joint training...")
    try:
        history = joint_trainer.train(epochs=config["epochs"], log_interval=1)
        
        print("\nTraining completed successfully!")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("This might be due to missing dependencies or data issues")
        return
    
    # Demonstrate inference
    print("\nDemonstrating inference...")
    try:
        joint_model.eval()
        with torch.no_grad():
            # Get a sample from the validation set
            sample_batch = next(iter(val_loader))
            sample_hierarchical = next(iter(hierarchical_val_loader))
            
            # Move to device
            sample_batch = sample_batch.to(device)
            
            # Prepare inputs for joint model
            inputs = {
                'x': sample_batch.x,
                'edge_index': sample_batch.edge_index,
                'edge_attr': getattr(sample_batch, 'edge_attr', None),
                'batch': getattr(sample_batch, 'batch', None),
                'scale_data': sample_hierarchical['scale_data'] if isinstance(sample_hierarchical, dict) else None,
                'use_fusion': True,
                'return_individual': True
            }
            
            # Forward pass
            outputs = joint_model(**inputs)
            
            print("Inference successful!")
            print(f"Output keys: {list(outputs.keys())}")
            
            # Print output shapes
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
    
    print("\nExample completed!")
    print("This demonstrates the basic usage of the joint training framework.")
    print("For production use, ensure all dependencies are installed and use larger datasets.")


if __name__ == "__main__":
    main()