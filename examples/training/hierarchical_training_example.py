#!/usr/bin/env python3
"""
Hierarchical Molecular Graph Neural Network Training Example

This example demonstrates how to train a Hierarchical MGNN model for predicting
partial charges and molecular properties using quantum chemistry data.
The model processes molecular data at multiple scales (atom, functional group, motif)
with cross-scale information exchange.

Features:
- Multi-scale graph generation from SMILES
- QM data integration
- Hierarchical graph processing
- Force field parameter prediction (partial charges)
- Multi-task learning

Usage:
    python hierarchical_training_example.py [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Import MoML modules
from moml import (
    MolecularGraphProcessor,
    GraphCoarsener,
    HMGNN,
    MGNNTrainer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ForceFieldMapper,
    HierarchicalGraphDataset,
    scaffold_split,
    prepare_dataloaders,
    split_dataset
)
from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hierarchical_training_example")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Hierarchical MGNN on PFAS data")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="moml/data/datasets/processed/chemical_list/pfas20_standardized.csv",
                        help="Path to input CSV with PFAS data")
    parser.add_argument("--qm_data_path", type=str, default="moml/data/datasets/qm_data/ml_training_data.json",
                        help="Path to QM data JSON")
    parser.add_argument("--output_dir", type=str, default="results/hierarchical_training",
                        help="Directory for output files")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="Number of GNN blocks per scale")
    parser.add_argument("--layers_per_block", type=int, default=3,
                        help="Number of layers per block")
    parser.add_argument("--cross_scale", action="store_true", default=True,
                        help="Enable cross-scale information exchange")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def prepare_qm_data(data_path: str, qm_data_path: str) -> pd.DataFrame:
    """
    Load and merge molecular data with QM computation results.
    
    Args:
        data_path: Path to CSV with PFAS data
        qm_data_path: Path to QM data JSON
        
    Returns:
        DataFrame with merged data
    """
    # Load molecular data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    mol_data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(mol_data)} molecules from {data_path}")
    
    # Load QM data
    if not os.path.exists(qm_data_path):
        logger.warning(f"QM data file not found: {qm_data_path}")
        logger.info("Creating a mock QM data file for demonstration")
        
        # Create mock QM data
        qm_data = []
        for _, row in mol_data.iterrows():
            smiles = row['smiles']
            compound_id = row['common_name']
            
            # Create a molecule to get the number of atoms
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            num_atoms = mol.GetNumAtoms()
            
            # Generate mock partial charges
            mock_charges = np.random.normal(0, 0.3, num_atoms)
            # Ensure charge neutrality
            mock_charges = mock_charges - mock_charges.mean()
            
            qm_entry = {
                "compound_id": compound_id,
                "smiles": smiles,
                "energy": -1000.0 - np.random.random() * 500,
                "dipole_moment": np.random.random() * 5,
                "homo_energy": -8 - np.random.random() * 2,
                "lumo_energy": -1 - np.random.random() * 3,
                "homo_lumo_gap": 4 + np.random.random() * 3,
                "mulliken_charges": mock_charges.tolist()
            }
            
            qm_data.append(qm_entry)
        
        # Save mock data
        os.makedirs(os.path.dirname(qm_data_path), exist_ok=True)
        with open(qm_data_path, 'w') as f:
            json.dump(qm_data, f, indent=2)
        
        logger.info(f"Created mock QM data for {len(qm_data)} molecules")
    else:
        # Load actual QM data
        with open(qm_data_path, 'r') as f:
            qm_data = json.load(f)
        logger.info(f"Loaded QM data for {len(qm_data)} molecules from {qm_data_path}")
    
    # Convert QM data to DataFrame
    qm_df = pd.DataFrame(qm_data)
    
    # Merge data
    merged_data = pd.merge(mol_data, qm_df, left_on='common_name', right_on='compound_id', how='inner')
    logger.info(f"Merged data contains {len(merged_data)} molecules")
    
    return merged_data


def prepare_dataset(data: pd.DataFrame, seed: int = 42) -> Dict[str, HierarchicalGraphDataset]:
    """
    Prepare hierarchical graph datasets for training.
    
    Args:
        data: DataFrame with molecular and QM data
        seed: Random seed for splitting
        
    Returns:
        Dictionary with train, val, test datasets
    """
    # Initialize processors
    graph_processor = MolecularGraphProcessor({
        'use_pfas_specific_features': True,
        'use_3d_coords': True
    })
    
    graph_coarsener = GraphCoarsener(
        use_3d_coords=True,
        use_pfas_features=True
    )
    
    # Prepare molecules and hierarchical graphs
    hierarchical_graphs = []
    
    for _, row in data.iterrows():
        smiles = row['smiles']
        
        try:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Add 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=seed)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Set name
            mol.SetProp('_Name', row['common_name'])
            
            # Create atom-level graph with partial charges
            charges = row.get('mulliken_charges', [])
            if len(charges) == mol.GetNumAtoms():
                additional_features = {'partial_charges': charges}
            else:
                additional_features = None
            
            # Generate atom-level graph
            atom_graph = graph_processor.mol_to_graph(mol, additional_features)
            
            # Generate hierarchical graphs
            hierarchical_graph_dict = graph_coarsener.create_hierarchical_graphs(atom_graph, mol)
            
            # Extract graphs
            atom_graph = hierarchical_graph_dict['atom']
            func_group_graph = hierarchical_graph_dict['functional_group']
            motif_graph = hierarchical_graph_dict['structural_motif']
            
            # Add hierarchical information
            atom_graph.hierarchical_data = {
                'atom': atom_graph,
                'functional_group': func_group_graph,
                'structural_motif': motif_graph
            }
            
            # Add target properties
            # 1. Global properties
            atom_graph.target_property = torch.tensor([
                row.get('energy', 0.0),
                row.get('dipole_moment', 0.0),
                row.get('homo_lumo_gap', 0.0)
            ], dtype=torch.float)
            
            # 2. Node properties (partial charges)
            if len(charges) == atom_graph.num_nodes:
                atom_graph.node_target = torch.tensor(charges, dtype=torch.float).view(-1, 1)
            
            # Add mappings between scales
            atom_graph.cluster_mapping = func_group_graph.cluster_mapping
            atom_graph.structural_mapping = motif_graph.structural_mapping
            
            # Add to list
            hierarchical_graphs.append(atom_graph)
        
        except Exception as e:
            logger.warning(f"Failed to process molecule {row['common_name']}: {str(e)}")
            continue
    
    # Split dataset using scaffold split
    train_data, val_data, test_data = scaffold_split(hierarchical_graphs, seed=seed)
    
    # Create datasets
    datasets = {
        'train': HierarchicalGraphDataset(train_data),
        'val': HierarchicalGraphDataset(val_data),
        'test': HierarchicalGraphDataset(test_data)
    }
    
    return datasets


def create_hierarchical_dataloader(
    datasets: Dict[str, HierarchicalGraphDataset],
    batch_size: int
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for hierarchical graph datasets.
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size for training
        
    Returns:
        Dictionary of dataloaders
    """
    return prepare_dataloaders(
        datasets['train'],
        datasets['val'],
        batch_size=batch_size,
        test_dataset=datasets['test']
    )


def create_hierarchical_model(
    scale_dims: List[int],
    edge_attr_dims: List[int],
    hidden_dim: int,
    n_blocks: int,
    layers_per_block: int,
    cross_scale_exchange: bool
) -> HMGNN:
    """
    Create a hierarchical MGNN model.
    
    Args:
        scale_dims: List of input dimensions for each scale
        edge_attr_dims: List of edge attribute dimensions for each scale
        hidden_dim: Hidden dimension size
        n_blocks: Number of GNN blocks per scale
        layers_per_block: Number of layers per block
        cross_scale_exchange: Whether to enable cross-scale information exchange
        
    Returns:
        Initialized HMGNN model
    """
    # Create model configuration
    config = {
        'model_type': 'hierarchical_mgnn',
        'hidden_dim': hidden_dim,
        'n_blocks': n_blocks,
        'layers_per_block': layers_per_block,
        'cross_scale_exchange': cross_scale_exchange,
        'scale_dims': scale_dims,
        'edge_attr_dims': edge_attr_dims
    }
    
    # Initialize model
    model = HMGNN(config)
    
    return model


def train_hierarchical_model(
    model: HMGNN,
    dataloaders: Dict[str, DataLoader],
    config: Dict
) -> Tuple[HMGNN, Dict]:
    """
    Train the hierarchical MGNN model.
    
    Args:
        model: HMGNN model to train
        dataloaders: Dictionary of dataloaders
        config: Training configuration
        
    Returns:
        Trained model and training history
    """
    # Create trainer
    trainer = MGNNTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        LearningRateScheduler(
            mode='min',
            factor=0.5,
            patience=5
        )
    ]
    
    # Train model
    history = trainer.train(
        epochs=config['epochs'],
        callbacks=callbacks
    )
    
    return model, history


def evaluate_model(
    model: HMGNN,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained HMGNN model
        test_loader: Test dataloader
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate metrics for each scale
            for scale, pred in outputs.items():
                if scale not in metrics:
                    metrics[scale] = {
                        'mae': [],
                        'mse': [],
                        'r2': []
                    }
                
                # Calculate metrics
                if hasattr(batch, f'{scale}_target'):
                    target = getattr(batch, f'{scale}_target')
                    mae = torch.mean(torch.abs(pred - target))
                    mse = torch.mean((pred - target) ** 2)
                    r2 = 1 - mse / torch.var(target)
                    
                    metrics[scale]['mae'].append(mae.item())
                    metrics[scale]['mse'].append(mse.item())
                    metrics[scale]['r2'].append(r2.item())
    
    # Average metrics
    for scale in metrics:
        for metric in metrics[scale]:
            metrics[scale][metric] = np.mean(metrics[scale][metric])
    
    return metrics


def export_force_field_parameters(
    model: HMGNN,
    test_dataset: HierarchicalGraphDataset,
    output_dir: str,
    force_field_type: str = 'amber',
    engine: str = 'gromacs'
) -> None:
    """
    Export force field parameters for the test molecules.
    
    Args:
        model: Trained HMGNN model
        test_dataset: Test dataset
        output_dir: Output directory
        force_field_type: Type of force field to use
        engine: Molecular dynamics engine
    """
    # Create force field mapper
    mapper = ForceFieldMapper(
        force_field_type=force_field_type,
        engine=engine
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each molecule
    for i, graph in enumerate(test_dataset):
        try:
            # Get molecule name
            mol_name = graph.mol_name if hasattr(graph, 'mol_name') else f'mol_{i}'
            
            # Get predicted charges
            with torch.no_grad():
                charges = model.predict_charges(graph)
            
            # Export parameters
            output_path = os.path.join(output_dir, f'{mol_name}_{force_field_type}.top')
            mapper.export_parameters(
                graph,
                charges,
                output_path
            )
            
            logger.info(f"Exported parameters for {mol_name} to {output_path}")
        
        except Exception as e:
            logger.warning(f"Failed to export parameters for molecule {i}: {str(e)}")
            continue


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_qm_data(args.data_path, args.qm_data_path)
    datasets = prepare_dataset(data, seed=args.seed)
    dataloaders = create_hierarchical_dataloader(datasets, args.batch_size)
    
    # Get input dimensions
    sample_graph = datasets['train'][0]
    scale_dims = [
        sample_graph.x.shape[1],
        sample_graph.hierarchical_data['functional_group'].x.shape[1],
        sample_graph.hierarchical_data['structural_motif'].x.shape[1]
    ]
    edge_attr_dims = [
        sample_graph.edge_attr.shape[1],
        sample_graph.hierarchical_data['functional_group'].edge_attr.shape[1],
        sample_graph.hierarchical_data['structural_motif'].edge_attr.shape[1]
    ]
    
    # Create model
    model = create_hierarchical_model(
        scale_dims=scale_dims,
        edge_attr_dims=edge_attr_dims,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        layers_per_block=args.layers_per_block,
        cross_scale_exchange=args.cross_scale
    )
    
    # Create training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Train model
    model, history = train_hierarchical_model(model, dataloaders, config)
    
    # Evaluate model
    device = torch.device(config['device'])
    metrics = evaluate_model(model, dataloaders['test'], device)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Export force field parameters
    export_force_field_parameters(
        model,
        datasets['test'],
        args.output_dir
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()