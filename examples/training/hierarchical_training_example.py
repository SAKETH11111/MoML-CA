#!/usr/bin/env python3
"""
Hierarchical Molecular Graph Neural Network Training Example

This example demonstrates how to train a Hierarchical MGNN model for predicting
partial charges and molecular properties of PFAS compounds using quantum chemistry data.
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
from moml.core.molecular_graph import MolecularGraphProcessor
from moml.core.graph_coarsening import GraphCoarsener
from moml.models.mgnn.hierarchical_mgnn import HMGNN
from moml.models.mgnn.training.trainer import MGNNTrainer
from moml.models.mgnn.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from moml.simulation.md.force_field_mapper import ForceFieldMapper
from moml.data.dataset import HierarchicalGraphDataset
from moml.data.splitting import scaffold_split
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
            logger.warning(f"Error processing molecule {row['common_name']}: {str(e)}")
    
    logger.info(f"Created {len(hierarchical_graphs)} hierarchical graph representations")
    
    # Split dataset
    train_idx, val_idx, test_idx = scaffold_split(
        hierarchical_graphs, 
        sizes=[0.8, 0.1, 0.1],
        seed=seed
    )
    
    # Create datasets
    train_dataset = HierarchicalGraphDataset([hierarchical_graphs[i] for i in train_idx])
    val_dataset = HierarchicalGraphDataset([hierarchical_graphs[i] for i in val_idx])
    test_dataset = HierarchicalGraphDataset([hierarchical_graphs[i] for i in test_idx])
    
    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'all': HierarchicalGraphDataset(hierarchical_graphs)
    }


def create_hierarchical_dataloader(
    datasets: Dict[str, HierarchicalGraphDataset],
    batch_size: int
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for hierarchical datasets.
    
    Args:
        datasets: Dictionary with datasets
        batch_size: Batch size
        
    Returns:
        Dictionary with DataLoaders
    """
    def collate_hierarchical_graphs(batch):
        """Custom collate function for hierarchical graphs."""
        # Extract atom-level graphs
        atom_graphs = [item for item in batch]
        
        # Extract hierarchical data
        hierarchical_data = []
        cluster_mappings = []
        
        for graph in atom_graphs:
            if hasattr(graph, 'hierarchical_data'):
                # Collect scale data
                scale_data = graph.hierarchical_data
                hierarchical_data.append(scale_data)
                
                # Collect mappings
                if hasattr(graph, 'cluster_mapping'):
                    cluster_mappings.append({
                        'atom_to_funcgroup': graph.cluster_mapping,
                        'atom_to_motif': graph.structural_mapping
                    })
        
        # Create a default collate function for PyG data
        from torch_geometric.data import Batch
        collated_batch = Batch.from_data_list(atom_graphs)
        
        # Add hierarchical data
        if hierarchical_data:
            collated_batch.hierarchical_data = hierarchical_data
        if cluster_mappings:
            collated_batch.cluster_mappings = cluster_mappings
        
        return collated_batch
    
    loaders = {}
    
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate_hierarchical_graphs
        )
    
    return loaders


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
        scale_dims: List of node feature dimensions for each scale
        edge_attr_dims: List of edge feature dimensions for each scale
        hidden_dim: Hidden dimension
        n_blocks: Number of GNN blocks per scale
        layers_per_block: Number of layers per block
        cross_scale_exchange: Whether to use cross-scale information exchange
        
    Returns:
        HMGNN model
    """
    model = HMGNN(
        scale_dims=scale_dims,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        layers_per_block=layers_per_block,
        edge_attr_dims=edge_attr_dims,
        jk_mode='attention',
        node_out_dim=1,  # For partial charges
        graph_out_dim=3,  # For energy, dipole, homo-lumo gap
        cross_scale_exchange=cross_scale_exchange,
        dropout=0.2
    )
    
    return model


def train_hierarchical_model(
    model: HMGNN,
    dataloaders: Dict[str, DataLoader],
    config: Dict
) -> Tuple[HMGNN, Dict]:
    """
    Train a hierarchical MGNN model.
    
    Args:
        model: HMGNN model
        dataloaders: Dictionary with DataLoaders
        config: Training configuration
        
    Returns:
        Tuple of (trained model, training history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Define loss function for multi-task learning
    def multi_task_loss(outputs, targets):
        """Custom loss function for node and graph tasks."""
        losses = {}
        
        # Node-level loss (partial charges)
        if 'node_pred' in outputs and hasattr(targets, 'node_target'):
            node_pred = outputs['node_pred']
            node_target = targets.node_target
            node_loss = torch.nn.functional.mse_loss(node_pred, node_target)
            losses['node_loss'] = node_loss
        
        # Graph-level loss (molecular properties)
        if 'graph_pred' in outputs and hasattr(targets, 'target_property'):
            graph_pred = outputs['graph_pred']
            graph_target = targets.target_property
            graph_loss = torch.nn.functional.mse_loss(graph_pred, graph_target)
            losses['graph_loss'] = graph_loss
        
        # Compute total loss
        if losses:
            # Weight the losses (can be adjusted)
            node_weight = config.get('node_loss_weight', 0.7)
            graph_weight = config.get('graph_loss_weight', 0.3)
            
            total_loss = 0
            if 'node_loss' in losses:
                total_loss += node_weight * losses['node_loss']
            if 'graph_loss' in losses:
                total_loss += graph_weight * losses['graph_loss']
            
            return total_loss
        else:
            # Fallback to dummy loss if no targets available
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            min_delta=0.001,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=os.path.join(config['output_dir'], 'checkpoints', 'model_{epoch:03d}_{val_loss:.4f}.pt'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        LearningRateScheduler(
            schedule='reduce_on_plateau',
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]
    
    # Create trainer
    trainer = MGNNTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=multi_task_loss,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        callbacks=callbacks,
        device=device
    )
    
    # Train model
    logger.info(f"Training model on {device} for {config['epochs']} epochs")
    history = trainer.train(epochs=config['epochs'])
    
    # Plot training curves
    plot_path = os.path.join(config['output_dir'], 'training_curves.png')
    trainer.plot_training_curves(plot_path)
    logger.info(f"Training curves saved to {plot_path}")
    
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
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Track metrics
    node_mse = 0.0
    graph_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Prepare hierarchical data
            scale_data = []
            
            # Add atom-level data
            scale_data.append({
                'x': batch.x,
                'edge_index': batch.edge_index,
                'edge_attr': batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                'batch': batch.batch
            })
            
            # Check if we have hierarchical data
            if hasattr(batch, 'hierarchical_data'):
                # Extract functional group level data from first graph (they're batched together)
                func_group_data = batch.hierarchical_data[0]['functional_group']
                scale_data.append({
                    'x': func_group_data.x,
                    'edge_index': func_group_data.edge_index,
                    'edge_attr': func_group_data.edge_attr if hasattr(func_group_data, 'edge_attr') else None,
                    'batch': None  # No batch for higher scales in this simple example
                })
                
                # Extract structural motif level data
                motif_data = batch.hierarchical_data[0]['structural_motif']
                scale_data.append({
                    'x': motif_data.x,
                    'edge_index': motif_data.edge_index,
                    'edge_attr': motif_data.edge_attr if hasattr(motif_data, 'edge_attr') else None,
                    'batch': None  # No batch for higher scales in this simple example
                })
            
            # Get cluster mappings
            cluster_mappings = None
            if hasattr(batch, 'cluster_mappings'):
                cluster_mappings = [
                    batch.cluster_mappings[0]['atom_to_funcgroup'],
                    batch.cluster_mappings[0]['atom_to_motif']
                ]
            
            # Forward pass
            outputs = model(scale_data, cluster_mappings)
            
            # Calculate node-level MSE (partial charges)
            if 'node_pred' in outputs and hasattr(batch, 'node_target'):
                node_pred = outputs['node_pred']
                node_target = batch.node_target
                node_mse += torch.nn.functional.mse_loss(node_pred, node_target).item()
            
            # Calculate graph-level MSE (molecular properties)
            if 'graph_pred' in outputs and hasattr(batch, 'target_property'):
                graph_pred = outputs['graph_pred']
                graph_target = batch.target_property
                graph_mse += torch.nn.functional.mse_loss(graph_pred, graph_target).item()
            
            num_batches += 1
    
    # Calculate average metrics
    metrics = {
        'node_mse': node_mse / num_batches if num_batches > 0 else float('nan'),
        'node_rmse': (node_mse / num_batches) ** 0.5 if num_batches > 0 else float('nan'),
        'graph_mse': graph_mse / num_batches if num_batches > 0 else float('nan'),
        'graph_rmse': (graph_mse / num_batches) ** 0.5 if num_batches > 0 else float('nan')
    }
    
    return metrics


def export_force_field_parameters(
    model: HMGNN,
    test_dataset: HierarchicalGraphDataset,
    output_dir: str,
    force_field_type: str = 'amber',
    engine: str = 'gromacs'
) -> None:
    """
    Export force field parameters from model predictions.
    
    Args:
        model: Trained HMGNN model
        test_dataset: Test dataset
        output_dir: Output directory
        force_field_type: Force field type
        engine: Simulation engine
    """
    # Create force field mapper
    mapper = ForceFieldMapper(
        force_field_type=force_field_type,
        simulation_engine=engine
    )
    
    # Create output directory
    ff_output_dir = os.path.join(output_dir, 'force_field_params')
    os.makedirs(ff_output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Process a few examples
    num_examples = min(5, len(test_dataset))
    
    for i in range(num_examples):
        data = test_dataset[i]
        
        # Prepare hierarchical data
        scale_data = []
        
        # Add atom-level data
        scale_data.append({
            'x': data.x.unsqueeze(0).to(device),
            'edge_index': data.edge_index.to(device),
            'edge_attr': data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None,
            'batch': None
        })
        
        # Add functional group level data
        if hasattr(data, 'hierarchical_data'):
            func_group_data = data.hierarchical_data['functional_group']
            scale_data.append({
                'x': func_group_data.x.unsqueeze(0).to(device),
                'edge_index': func_group_data.edge_index.to(device),
                'edge_attr': func_group_data.edge_attr.to(device) if hasattr(func_group_data, 'edge_attr') else None,
                'batch': None
            })
            
            # Add structural motif level data
            motif_data = data.hierarchical_data['structural_motif']
            scale_data.append({
                'x': motif_data.x.unsqueeze(0).to(device),
                'edge_index': motif_data.edge_index.to(device),
                'edge_attr': motif_data.edge_attr.to(device) if hasattr(motif_data, 'edge_attr') else None,
                'batch': None
            })
        
        # Get cluster mappings
        cluster_mappings = None
        if hasattr(data, 'cluster_mapping') and hasattr(data, 'structural_mapping'):
            cluster_mappings = [
                data.cluster_mapping,
                data.structural_mapping
            ]
        
        # Get molecule name
        mol_name = f"molecule_{i}"
        if hasattr(data, '_name'):
            mol_name = data._name
        
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(scale_data, cluster_mappings)
            
            # Get partial charge predictions
            if 'node_pred' in outputs:
                partial_charges = outputs['node_pred'].cpu().numpy().flatten()
                
                # Create RDKit molecule from SMILES
                try:
                    mol = Chem.MolFromSmiles(data.smiles if hasattr(data, 'smiles') else "")
                    if mol is None:
                        logger.warning(f"Could not create molecule for {mol_name}")
                        continue
                    
                    # Set name
                    mol.SetProp('_Name', mol_name)
                    
                    # Add 3D coordinates
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.UFFOptimizeMolecule(mol)
                    
                    # Check if number of atoms matches
                    if mol.GetNumAtoms() != len(partial_charges):
                        logger.warning(
                            f"Number of atoms in molecule ({mol.GetNumAtoms()}) doesn't match "
                            f"number of partial charges ({len(partial_charges)})"
                        )
                        continue
                    
                    # Export to force field files
                    mol_output_dir = os.path.join(ff_output_dir, mol_name)
                    os.makedirs(mol_output_dir, exist_ok=True)
                    
                    # Convert predictions to force field parameters
                    success, results = mapper.convert_mgnn_predictions_to_force_field(
                        mol=mol,
                        node_predictions=partial_charges,
                        output_dir=mol_output_dir,
                        engine=engine
                    )
                    
                    if success:
                        logger.info(f"Exported force field parameters for {mol_name}")
                    else:
                        logger.warning(f"Failed to export force field parameters for {mol_name}")
                
                except Exception as e:
                    logger.error(f"Error exporting force field parameters for {mol_name}: {str(e)}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Prepare data
    data = prepare_qm_data(args.data_path, args.qm_data_path)
    
    # Prepare datasets
    datasets = prepare_dataset(data, args.seed)
    
    # Create dataloaders
    dataloaders = create_hierarchical_dataloader(datasets, args.batch_size)
    
    # Create model
    # Get node feature dimensions for each scale from the first batch
    example_batch = next(iter(dataloaders['train']))
    
    scale_dims = []
    edge_attr_dims = []
    
    # Atom level
    scale_dims.append(example_batch.x.shape[1])
    if hasattr(example_batch, 'edge_attr'):
        edge_attr_dims.append(example_batch.edge_attr.shape[1])
    else:
        edge_attr_dims.append(0)
    
    # Higher levels (if available)
    if hasattr(example_batch, 'hierarchical_data'):
        # Functional group level
        func_group_data = example_batch.hierarchical_data[0]['functional_group']
        scale_dims.append(func_group_data.x.shape[1])
        if hasattr(func_group_data, 'edge_attr'):
            edge_attr_dims.append(func_group_data.edge_attr.shape[1])
        else:
            edge_attr_dims.append(0)
        
        # Structural motif level
        motif_data = example_batch.hierarchical_data[0]['structural_motif']
        scale_dims.append(motif_data.x.shape[1])
        if hasattr(motif_data, 'edge_attr'):
            edge_attr_dims.append(motif_data.edge_attr.shape[1])
        else:
            edge_attr_dims.append(0)
    
    # Create model
    model = create_hierarchical_model(
        scale_dims=scale_dims,
        edge_attr_dims=edge_attr_dims,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        layers_per_block=args.layers_per_block,
        cross_scale_exchange=args.cross_scale
    )
    
    # Log model architecture
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Model architecture: {type(model).__name__}")
    logger.info(f"Number of scales: {len(scale_dims)}")
    logger.info(f"Feature dimensions: {scale_dims}")
    
    # Train model
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'node_loss_weight': 0.7,  # Weight for partial charge prediction
        'graph_loss_weight': 0.3,  # Weight for property prediction
        'output_dir': args.output_dir
    }
    
    model, history = train_hierarchical_model(model, dataloaders, config)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, dataloaders['test'], device)
    
    logger.info("Test metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")
    
    # Export force field parameters
    export_force_field_parameters(
        model=model,
        test_dataset=datasets['test'],
        output_dir=args.output_dir,
        force_field_type='amber',
        engine='gromacs'
    )
    
    logger.info("Training example completed successfully")


if __name__ == "__main__":
    main()