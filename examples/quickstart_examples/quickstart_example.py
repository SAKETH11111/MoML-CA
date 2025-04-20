#!/usr/bin/env python
"""
Quickstart example for MoML.

This script demonstrates the basic usage of the MoML package for
molecular graph creation, model training, and prediction.
"""

import os
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch.utils.data import DataLoader

# Import MoML components
from moml.core import (
    create_graph_processor,
    GraphCoarsener,
    initialize_model
)
    
from moml.data import (
    prepare_dataloaders,
    split_dataset
)

from moml.models.mgnn.evaluation import (
    visualize_predictions,
    create_predictor,
    calculate_metrics
)

from moml.models.mgnn.training import (
    create_trainer
)


def simple_example():
    """
    Demonstrate simple molecular graph creation and visualization.
    """
    print("\n--- Simple Graph Creation Example ---\n")
    
    # Create graph processor with standard configuration
    processor = create_graph_processor({
        'use_partial_charges': True,
        'use_3d_coords': True,
        'use_pfas_specific_features': True
    })
    
    # Create a simple molecule from SMILES
    smiles = "C(C(F)(F)F)(C(F)(F)F)(F)F"  # Perfluorobutane
    print(f"Creating graph for: {smiles}")
    
    # Convert to graph
    graph = processor.smiles_to_graph(smiles)
    
    # Print graph information
    print(f"Graph created with {graph.num_nodes} nodes and {graph.edge_index.shape[1]//2} edges")
    print(f"Node features: {graph.x.shape[1]} dimensions")
    print(f"Edge features: {graph.edge_attr.shape[1]} dimensions")
    
    return graph


def hierarchical_graph_example(graph):
    """
    Demonstrate hierarchical graph creation.
    """
    print("\n--- Hierarchical Graph Example ---\n")
    
    # Create a coarsener with standard configuration
    coarsener = GraphCoarsener(use_pfas_features=True)
    
    # Convert RDKit molecule to graph
    mol = Chem.MolFromSmiles("C(C(F)(F)F)(C(F)(F)F)(F)F")
    
    # Create hierarchical graphs
    print("Creating hierarchical graphs...")
    graphs = coarsener.create_hierarchical_graphs(graph, mol)
    
    # Print information for each level
    for level, g in graphs.items():
        print(f"{level.upper()} level graph: {g.num_nodes} nodes, {g.edge_index.shape[1]//2} edges")
    
    return graphs


def model_training_example():
    """
    Demonstrate model training with synthetic data.
    """
    print("\n--- Model Training Example ---\n")
    
    # Create synthetic dataset
    num_graphs = 50
    num_nodes = 10
    num_node_features = 16
    num_edge_features = 4
    
    # Create random graphs
    graphs = []
    for i in range(num_graphs):
        # Random node features
        x = torch.randn(num_nodes, num_node_features)
        
        # Random edges (fully connected)
        edge_index = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst:
                    edge_index.append([src, dst])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Random edge features
        edge_attr = torch.randn(edge_index.shape[1], num_edge_features)
        
        # Random target (graph-level)
        y = torch.randn(1, 1)
        
        # Create graph object
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        graphs.append(graph)
    
    # Split dataset using standard utility
    train_graphs, val_graphs = split_dataset(graphs, train_ratio=0.8)
    
    # Create dataloaders using standard utility
    train_loader, val_loader = prepare_dataloaders(
        train_graphs,
        val_graphs,
        batch_size=10
    )
    
    # Create configuration dictionary with standard settings
    config = {
        'model_type': 'multi_task_djmgnn',
        'hidden_dim': 32,
        'n_blocks': 2,
        'layers_per_block': 1,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 10
    }
    
    # Initialize model
    model = initialize_model(config, num_node_features, num_edge_features)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train model
    print("Training model...")
    history = trainer.train(epochs=10)
    
    # Print training results
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    if 'val_loss' in history and history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Save model
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "example_model.pt")
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model, model_path, val_loader


def prediction_example(model_path, val_loader):
    """
    Demonstrate prediction with a trained model.
    """
    print("\n--- Prediction Example ---\n")
    
    # Create predictor
    predictor = create_predictor(model_path=model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict_from_dataloader(val_loader)
    
    # Extract true values and predictions
    true_values = torch.cat([graph.y for graph in val_loader.dataset])
    predicted_values = predictions['graph_pred']
    
    # Calculate metrics using standard utility
    metrics = calculate_metrics(true_values, predicted_values)
    
    # Print metrics
    print("Prediction metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Create visualization using standard utility
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, "predictions.png")
    
    fig = visualize_predictions(
        true_values,
        predicted_values,
        title="Example Predictions",
        save_path=vis_path
    )
    
    print(f"Prediction visualization saved to {vis_path}")


def main():
    """
    Run complete example workflow.
    """
    print("MoML Quickstart Example")
    print("=========================")
    
    # Simple graph creation
    graph = simple_example()
    
    # Hierarchical graph creation
    hierarchical_graphs = hierarchical_graph_example(graph)
    
    # Model training
    model, model_path, val_loader = model_training_example()
    
    # Prediction
    prediction_example(model_path, val_loader)
    
    print("\nQuickstart example completed successfully!")


if __name__ == "__main__":
    main() 