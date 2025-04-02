#!/usr/bin/env python
"""
Prediction Example for MoML-CA.

This script demonstrates how to use the MGNNPredictor class for
making predictions with a trained model on both single molecules
and batches of molecules.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
from rdkit import Chem

from moml import create_graph_processor
from moml.models.mgnn.evaluation.predictor import (
    MGNNPredictor,
    create_predictor,
    batch_predict_from_files
)
from moml.models.mgnn.evaluation.metrics import calculate_metrics
from moml.models.mgnn.training import initialize_model, MGNNConfig
from moml.utils.visualization.visualization import visualize_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prediction example")
    
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to saved model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save prediction results')
    parser.add_argument('--create_model', action='store_true',
                      help='Create a new model instead of loading from path')
    
    return parser.parse_args()


def create_sample_model():
    """
    Create a sample model for demonstrating prediction.
    """
    print("Creating a sample model...")
    
    # Create a simple molecule to determine feature dimensions
    smiles = "C(C(F)(F)F)(C(F)(F)F)(F)F"  # Perfluorobutane
    processor = create_graph_processor()
    graph = processor.smiles_to_graph(smiles)
    
    # Get dimensions
    node_features = graph.x.shape[1]
    edge_features = graph.edge_attr.shape[1]
    
    # Create configuration
    config = MGNNConfig({
        'model_type': 'multi_task_djmgnn',
        'hidden_dim': 32,
        'n_blocks': 2,
        'layers_per_block': 1,
        'in_dim': node_features,
        'edge_attr_dim': edge_features,
        'dropout': 0.2
    })
    
    # Initialize model
    model = initialize_model(config, node_features, edge_features)
    
    # Set to eval mode for prediction
    model.eval()
    
    print(f"Created model with {node_features} node features and {edge_features} edge features")
    return model, config


def single_molecule_example(model=None, model_path=None, config=None, output_dir='output'):
    """
    Demonstrate prediction on a single molecule.
    """
    print("\n--- Single Molecule Prediction Example ---\n")
    
    # Create predictor from model or model path
    if model is not None:
        print("Creating predictor from in-memory model...")
        predictor = create_predictor(model=model, config=config)
    else:
        print(f"Creating predictor from saved model: {model_path}")
        predictor = create_predictor(model_path=model_path)
    
    # Create a simple molecule for testing
    smiles = "C(C(F)(F)F)(C(F)(F)F)(F)F"  # Perfluorobutane
    print(f"Making predictions for: {smiles}")
    
    # Create RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Make prediction from SMILES directly
    print("Predicting from SMILES...")
    predictions_smiles = predictor.predict_from_smiles(smiles)
    
    # Display predictions
    print("\nPredictions from SMILES:")
    for key, value in predictions_smiles.items():
        if key == 'graph_pred':
            print(f"  Graph-level prediction: {value.squeeze().item():.4f}")
        elif key == 'node_pred':
            print(f"  Node-level predictions (first 3 nodes): {value[:3].squeeze().tolist()}")
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "single_molecule_prediction.json")
    predictor.save_predictions(predictions_smiles, output_file)
    print(f"Predictions saved to {output_file}")
    
    return predictions_smiles


def batch_prediction_example(model=None, model_path=None, config=None, output_dir='output'):
    """
    Demonstrate batch prediction on multiple molecules.
    """
    print("\n--- Batch Prediction Example ---\n")
    
    # Create simple molecules for batch prediction
    smiles_list = [
        "CC(F)(F)F",            # Trifluoromethylbenzene
        "C(C(F)(F)F)(F)(F)F",   # Hexafluoroethane
        "C(C(F)(F)F)(C(F)(F)F)(F)F",  # Perfluorobutane
        "CC(=O)O",              # Acetic acid
        "c1ccccc1"              # Benzene
    ]
    
    # Create predictor from model or model path
    print("Creating predictor...")
    if model is not None:
        predictor = create_predictor(model=model, config=config)
    else:
        predictor = create_predictor(model_path=model_path)
    
    # Process molecules to graphs
    print(f"Processing {len(smiles_list)} molecules...")
    graphs = []
    for i, smiles in enumerate(smiles_list):
        try:
            graph = predictor.processor.smiles_to_graph(smiles)
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
    
    # Batch predict
    print("Making batch predictions...")
    batch_predictions = predictor.batch_predict(graphs, batch_size=2)
    
    # Display results
    print("\nBatch prediction results:")
    print(f"Predicted {len(batch_predictions.get('graph_pred', []))} molecules")
    
    # Combine SMILES with predictions for display
    graph_preds = batch_predictions.get('graph_pred', torch.tensor([]))
    results = {}
    
    for i, (smiles, pred) in enumerate(zip(smiles_list, graph_preds)):
        pred_value = pred.item() if pred.numel() == 1 else pred.squeeze().tolist()
        results[smiles] = pred_value
        print(f"  Molecule {i+1}: {smiles} -> {pred_value:.4f}")
    
    # Save results
    output_file = os.path.join(output_dir, "batch_predictions.json")
    with open(output_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"Batch predictions saved to {output_file}")
    
    return batch_predictions, smiles_list


def directory_prediction_example(model_path=None, output_dir='output'):
    """
    Simulate prediction on a directory of molecule files.
    """
    print("\n--- Directory Prediction Example ---\n")
    
    # Directory prediction requires a saved model path
    if model_path is None:
        print("Directory prediction requires a saved model path. Skipping this example.")
        return None
    
    # For this example, we'll create a temporary directory with some sample files
    # In a real scenario, you would point to an existing directory with .mol files
    
    # Create temporary directory
    temp_dir = os.path.join(output_dir, "temp_molecules")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create some sample molecules and save them
    smiles_list = [
        "CC(F)(F)F",            # Trifluoromethylbenzene
        "C(C(F)(F)F)(F)(F)F",   # Hexafluoroethane
        "C(C(F)(F)F)(C(F)(F)F)(F)F",  # Perfluorobutane
    ]
    
    mol_files = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol_file = os.path.join(temp_dir, f"molecule_{i+1}.mol")
        Chem.MolToMolFile(mol, mol_file)
        mol_files.append(mol_file)
    
    print(f"Created {len(mol_files)} molecule files in {temp_dir}")
    
    # Create output directory
    pred_dir = os.path.join(output_dir, "directory_predictions")
    
    # Run batch prediction on the directory
    print(f"Running batch prediction on directory...")
    
    results = batch_predict_from_files(
        model_path=model_path,
        input_dir=temp_dir,
        output_dir=pred_dir,
        file_pattern="*.mol",
        batch_size=2
    )
    
    print(f"Processed {len(results)} files")
    print(f"Results saved to {pred_dir}")
    
    return results


def trainer_prediction_example(output_dir='output'):
    """
    Demonstrate getting a predictor directly from a trainer.
    """
    print("\n--- Trainer-Based Prediction Example ---\n")
    
    # This example shows how a predictor can be obtained from a trainer
    # First, we create a sample model and a simplified trainer
    model, config = create_sample_model()
    
    # In a real scenario, you would have a proper trainer with a trained model
    # For this example, we'll create a minimal trainer-like object
    class SimplifiedTrainer:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.device = 'cpu'
        
        def get_predictor(self):
            """Simplified version of the trainer's get_predictor method"""
            return create_predictor(model=self.model, config=self.config, device=self.device)
    
    # Create our simplified trainer
    trainer = SimplifiedTrainer(model, config)
    
    # Get predictor directly from the trainer
    print("Getting predictor from trainer...")
    predictor = trainer.get_predictor()
    
    # Make a prediction
    smiles = "CC(F)(F)F"  # Trifluoromethylbenzene
    print(f"Making prediction for: {smiles}")
    
    prediction = predictor.predict_from_smiles(smiles)
    
    # Display result
    print("\nPrediction result:")
    for key, value in prediction.items():
        if key == 'graph_pred':
            print(f"  Graph-level prediction: {value.squeeze().item():.4f}")
    
    return prediction


def main():
    """Run the prediction examples."""
    # Parse arguments
    args = parse_args()
    
    print("MoML-CA Prediction Example")
    print("=========================")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create or load model
    model = None
    config = None
    
    if args.create_model:
        model, config = create_sample_model()
        
        # Save the model for directory example
        model_path = os.path.join(args.output_dir, "sample_model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        args.model_path = model_path
        print(f"Saved sample model to {model_path}")
    elif args.model_path is None:
        print("No model path provided and create_model is False. Creating a sample model.")
        model, config = create_sample_model()
        
        # Save the model for directory example
        model_path = os.path.join(args.output_dir, "sample_model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        args.model_path = model_path
        print(f"Saved sample model to {model_path}")
    
    # Run the single molecule prediction example
    single_result = single_molecule_example(model, args.model_path, config, args.output_dir)
    
    # Run the batch prediction example
    batch_results, smiles_list = batch_prediction_example(model, args.model_path, config, args.output_dir)
    
    # Run the directory prediction example
    dir_results = directory_prediction_example(args.model_path, args.output_dir)
    
    # Run the trainer prediction example
    trainer_result = trainer_prediction_example(args.output_dir)
    
    print("\nPrediction examples completed successfully!")


if __name__ == "__main__":
    main() 