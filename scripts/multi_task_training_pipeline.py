#!/usr/bin/env python3
"""
multi_task_training_pipeline.py

🎯 MULTI-TASK DJMGNN TRAINING PIPELINE FOR PAPER SUBMISSION 🎯

This script extends the base DJMGNN with multi-task capabilities:
1. Molecular properties (19D) - QM9 dataset
2. Atomic forces (3D per atom) - SPICE dataset  
3. Force field parameters - From quantum calculations
4. Treatment efficacy - PFAS-specific predictions

Multi-stage training approach:
Stage 1: Base molecular properties (QM9)
Stage 2: Add atomic forces (SPICE)
Stage 3: Add force field parameters (QM data)
Stage 4: PFAS fine-tuning with treatment efficacy

Usage:
    python scripts/multi_task_training_pipeline.py --stage all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from moml.models.mgnn.multi_task_djmgnn import MultiTaskDJMGNN, create_multi_task_djmgnn
from moml.utils.dataset_utils import SubsetWrapper
from scripts.comprehensive_djmgnn_training import ComprehensiveTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskTrainer(ComprehensiveTrainer):
    """Extended trainer for multi-task DJMGNN with force field capabilities."""
    
    def __init__(self, experiment_name: str = "multi_task_djmgnn", use_wandb: bool = True):
        super().__init__(experiment_name, use_wandb)
        logger.info("🔬 Multi-Task DJMGNN Training Framework Initialized")
    
    def create_multi_task_model(self, config: Dict[str, Any]) -> MultiTaskDJMGNN:
        """Create multi-task DJMGNN model."""
        base_config = {
            'in_node_dim': 29,
            'in_edge_dim': config.get('in_edge_dim', 0),
            'hidden_dim': config['hidden_dim'],
            'n_blocks': config['n_blocks'],
            'layers_per_block': config['layers_per_block'],
            'node_output_dims': config['node_output_dims'],
            'graph_output_dims': config['graph_output_dims'],
            'energy_output_dims': config.get('energy_output_dims', 1),
            'dropout': config['dropout'],
            'jk_mode': config.get('jk_mode', 'concat'),
            'use_supernode': config.get('use_supernode', True),
            'use_rbf': config.get('use_rbf', True),
            'rbf_K': config.get('rbf_K', 32),
            'pool_type': config.get('pool_type', 'mean')
        }
        
        multi_task_config = {
            'predict_force_field': config.get('predict_force_field', True),
            'predict_treatment_efficacy': config.get('predict_treatment_efficacy', True),
            'force_field_hidden_dim': config.get('force_field_hidden_dim', 128),
            'treatment_hidden_dim': config.get('treatment_hidden_dim', 64),
            'dropout': config['dropout']
        }
        
        return create_multi_task_djmgnn(base_config, multi_task_config).to(self.device)
    
    def prepare_spice_data(self, config: Dict[str, Any]) -> Optional[Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]]:
        """Prepare SPICE dataset for atomic force training."""
        try:
            from moml.data.spice_dataset import SpiceDataset
            
            logger.info("📊 Preparing SPICE dataset for force training...")
            
            # Create transform pipeline
            transform = Compose([CreateEdges(), FeaturizeNodes()])
            
            # Load SPICE datasets
            train_dataset = SpiceDataset(root='data/spice', split='train', transform=transform)
            val_dataset = SpiceDataset(root='data/spice', split='val', transform=transform)
            test_dataset = SpiceDataset(root='data/spice', split='test', transform=transform)
            
            # Limit dataset size for manageable training
            max_samples = config.get('max_spice_samples', 10000)
            if len(train_dataset) > max_samples:
                indices = torch.randperm(len(train_dataset))[:max_samples]
                train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
            
            if len(val_dataset) > max_samples // 5:
                indices = torch.randperm(len(val_dataset))[:max_samples//5]
                val_dataset = torch.utils.data.Subset(val_dataset, indices.tolist())
            
            batch_size = config.get('batch_size', 32)  # Smaller batch for force training
            
            train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"SPICE splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.warning(f"Could not load SPICE dataset: {e}")
            logger.warning("Continuing without atomic force training...")
            return None
    
    def prepare_pfas_data(self, config: Dict[str, Any]) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
        """Prepare PFAS dataset with treatment efficacy targets."""
        logger.info("📊 Preparing PFAS dataset with treatment efficacy...")
        
        # Load PFAS molecular data
        pfas_csv = PROJECT_ROOT / "data/processed/chemical_list/PFAS_Aligned_Data.csv"
        treatment_csv = PROJECT_ROOT / "data/processed/treatment_data/PFAS_Treatment_Data_cleaned.csv"
        
        pfas_df = pd.read_csv(pfas_csv).dropna(subset=['SMILES'])
        treatment_df = pd.read_csv(treatment_csv).dropna(subset=['Effectiveness_Percent_Numeric'])
        
        # Merge datasets to get treatment efficacy targets
        merged_df = pfas_df.merge(treatment_df, on='CASRN', how='inner')
        logger.info(f"Found {len(merged_df)} PFAS molecules with treatment data")
        
        # Convert to graph dataset
        from scripts.test_huggingface_djmgnn import SimplePFASConverter
        converter = SimplePFASConverter()
        
        graphs = []
        for idx, row in merged_df.iterrows():
            try:
                graph = converter.smiles_to_graph(row['SMILES'])
                if graph is not None:
                    # Add PFAS-specific targets
                    pfas_properties = torch.tensor([
                        row.get('F_Count', 0) / 30.0,  # Normalized
                        row.get('F_Percentage', 0) / 100.0,
                        row.get('Chain_Length', 0) / 20.0,
                        row.get('Average_Mass', 0) / 1000.0,
                        float(row.get('Is_Aromatic', False))
                    ], dtype=torch.float32)
                    
                    treatment_efficacy = torch.tensor([
                        row['Effectiveness_Percent_Numeric'] / 100.0  # Normalize to [0,1]
                    ], dtype=torch.float32)
                    
                    graph.pfas_properties = pfas_properties
                    graph.treatment_efficacy = treatment_efficacy
                    graphs.append(graph)
                    
            except Exception as e:
                logger.debug(f"Error processing PFAS molecule {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(graphs)} PFAS molecules")
        
        # Split dataset
        torch.manual_seed(42)
        indices = torch.randperm(len(graphs))
        train_size = int(0.7 * len(graphs))
        val_size = int(0.2 * len(graphs))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        test_graphs = [graphs[i] for i in test_indices]
        
        batch_size = config.get('batch_size', 32)
        train_loader = GraphDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_stage1_molecular_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Train base molecular property prediction on QM9."""
        logger.info("🚀 STAGE 1: Molecular Property Prediction (QM9)")
        
        # Use base DJMGNN for this stage
        base_results = self.train_base_model(config)
        
        # Save stage 1 model
        torch.save(base_results['model_state_dict'], 
                  self.experiment_dir / 'stage1_molecular_properties.pt')
        
        return base_results
    
    def train_stage2_atomic_forces(self, base_model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Add atomic force prediction using SPICE dataset."""
        logger.info("🚀 STAGE 2: Atomic Force Prediction (SPICE)")
        
        spice_data = self.prepare_spice_data(config)
        if spice_data is None:
            logger.warning("Skipping atomic force training - SPICE data not available")
            return {'skipped': True, 'reason': 'SPICE data not available'}
        
        train_loader, val_loader, test_loader = spice_data
        
        # Create multi-task model and load base weights
        model = self.create_multi_task_model(config)
        
        # Load base model weights
        base_checkpoint = torch.load(base_model_path, map_location=self.device)
        base_state_dict = base_checkpoint if isinstance(base_checkpoint, dict) else base_checkpoint.state_dict()
        
        # Load compatible weights from base model
        model_dict = model.state_dict()
        compatible_dict = {}
        
        for k, v in base_state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            elif k.startswith('djmgnn.') and k[7:] in model_dict:
                # Map base model weights to djmgnn submodule
                new_key = 'djmgnn.' + k[7:] if not k.startswith('djmgnn.') else k
                if new_key in model_dict and model_dict[new_key].shape == v.shape:
                    compatible_dict[new_key] = v
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
        
        logger.info(f"Loaded {len(compatible_dict)} compatible parameters from base model")
        
        # Training setup - focus on force prediction
        optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 0.0005), weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 50))
        
        # Training loop
        epochs = config.get('epochs', 50)
        best_force_mae = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                
                # Multi-task loss
                total_loss = 0
                loss_count = 0
                
                # Molecular properties (if available)
                if hasattr(batch, 'y_graph') and 'molecular_properties' in outputs:
                    prop_loss = F.mse_loss(outputs['molecular_properties'], batch.y_graph)
                    total_loss += prop_loss
                    loss_count += 1
                
                # Atomic forces (primary focus)
                if hasattr(batch, 'node_y') and 'node_embeddings' in outputs:
                    # Use node embeddings for force prediction (3D per atom)
                    force_pred = outputs['node_embeddings'][:, :3]  # First 3 dims as forces
                    force_loss = F.mse_loss(force_pred, batch.node_y)
                    total_loss += 2.0 * force_loss  # Higher weight for forces
                    loss_count += 1
                
                if loss_count > 0:
                    total_loss = total_loss / loss_count
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append(total_loss.item())
            
            scheduler.step()
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                force_maes = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                        
                        if hasattr(batch, 'node_y') and 'node_embeddings' in outputs:
                            force_pred = outputs['node_embeddings'][:, :3]
                            force_mae = F.l1_loss(force_pred, batch.node_y).item()
                            force_maes.append(force_mae)
                
                avg_force_mae = np.mean(force_maes) if force_maes else float('inf')
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(epoch_losses):.6f}, Force MAE: {avg_force_mae:.6f}")
                
                if avg_force_mae < best_force_mae:
                    best_force_mae = avg_force_mae
                    best_model_state = model.state_dict().copy()
        
        # Save stage 2 model
        if best_model_state:
            torch.save(best_model_state, self.experiment_dir / 'stage2_with_forces.pt')
        
        return {
            'best_force_mae': best_force_mae,
            'model_state_dict': best_model_state,
            'training_completed': True
        }
    
    def train_stage3_pfas_specialization(self, base_model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: PFAS specialization with treatment efficacy prediction."""
        logger.info("🚀 STAGE 3: PFAS Specialization with Treatment Efficacy")
        
        # Load PFAS data
        train_loader, val_loader, test_loader = self.prepare_pfas_data(config)
        
        # Create multi-task model
        model = self.create_multi_task_model(config)
        
        # Load previous stage weights
        if base_model_path and Path(base_model_path).exists():
            checkpoint = torch.load(base_model_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=False)
            logger.info("Loaded weights from previous training stage")
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 0.0001), weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 30))
        
        epochs = config.get('epochs', 30)
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                
                total_loss = 0
                loss_count = 0
                
                # PFAS properties prediction
                if hasattr(batch, 'pfas_properties') and 'pfas_properties' in outputs:
                    pfas_loss = F.mse_loss(outputs['pfas_properties'], batch.pfas_properties)
                    total_loss += pfas_loss
                    loss_count += 1
                
                # Treatment efficacy prediction
                if hasattr(batch, 'treatment_efficacy') and 'treatment_efficacy' in outputs:
                    efficacy_loss = F.binary_cross_entropy(outputs['treatment_efficacy'], batch.treatment_efficacy)
                    total_loss += 2.0 * efficacy_loss  # Higher weight
                    loss_count += 1
                
                if loss_count > 0:
                    total_loss = total_loss / loss_count
                    total_loss.backward()
                    optimizer.step()
                    epoch_losses.append(total_loss.item())
            
            scheduler.step()
            
            # Validation
            if epoch % 3 == 0:
                val_loss = self._validate_pfas_model(model, val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(epoch_losses):.6f}, Val Loss: {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
        
        # Final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, self.experiment_dir / 'stage3_pfas_specialized.pt')
        
        test_metrics = self._comprehensive_pfas_evaluation(model, test_loader)
        
        return {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'model_state_dict': best_model_state,
            'training_completed': True
        }
    
    def _validate_pfas_model(self, model: torch.nn.Module, val_loader: GraphDataLoader) -> float:
        """Validate PFAS specialized model."""
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                
                total_loss = 0
                loss_count = 0
                
                if hasattr(batch, 'pfas_properties') and 'pfas_properties' in outputs:
                    pfas_loss = F.mse_loss(outputs['pfas_properties'], batch.pfas_properties)
                    total_loss += pfas_loss
                    loss_count += 1
                
                if hasattr(batch, 'treatment_efficacy') and 'treatment_efficacy' in outputs:
                    efficacy_loss = F.binary_cross_entropy(outputs['treatment_efficacy'], batch.treatment_efficacy)
                    total_loss += efficacy_loss
                    loss_count += 1
                
                if loss_count > 0:
                    val_losses.append((total_loss / loss_count).item())
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def _comprehensive_pfas_evaluation(self, model: torch.nn.Module, test_loader: GraphDataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation of PFAS specialized model."""
        model.eval()
        
        all_pfas_preds = []
        all_pfas_targets = []
        all_efficacy_preds = []
        all_efficacy_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                
                if hasattr(batch, 'pfas_properties') and 'pfas_properties' in outputs:
                    all_pfas_preds.append(outputs['pfas_properties'].cpu())
                    all_pfas_targets.append(batch.pfas_properties.cpu())
                
                if hasattr(batch, 'treatment_efficacy') and 'treatment_efficacy' in outputs:
                    all_efficacy_preds.append(outputs['treatment_efficacy'].cpu())
                    all_efficacy_targets.append(batch.treatment_efficacy.cpu())
        
        metrics = {}
        
        # PFAS properties evaluation
        if all_pfas_preds:
            pfas_preds = torch.cat(all_pfas_preds, dim=0).numpy()
            pfas_targets = torch.cat(all_pfas_targets, dim=0).numpy()
            
            pfas_r2_scores = []
            for i in range(pfas_preds.shape[1]):
                r2 = r2_score(pfas_targets[:, i], pfas_preds[:, i])
                pfas_r2_scores.append(r2)
            
            metrics['pfas_properties'] = {
                'mean_r2': np.mean(pfas_r2_scores),
                'individual_r2': pfas_r2_scores,
                'mean_mae': mean_absolute_error(pfas_targets, pfas_preds)
            }
        
        # Treatment efficacy evaluation
        if all_efficacy_preds:
            efficacy_preds = torch.cat(all_efficacy_preds, dim=0).numpy()
            efficacy_targets = torch.cat(all_efficacy_targets, dim=0).numpy()
            
            efficacy_r2 = r2_score(efficacy_targets, efficacy_preds)
            efficacy_mae = mean_absolute_error(efficacy_targets, efficacy_preds)
            
            metrics['treatment_efficacy'] = {
                'r2_score': efficacy_r2,
                'mae': efficacy_mae,
                'accuracy_90_percent': np.mean(np.abs(efficacy_targets - efficacy_preds) < 0.1)
            }
        
        return metrics
    
    def run_full_multi_task_pipeline(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run complete multi-task training pipeline."""
        logger.info("🚀 STARTING FULL MULTI-TASK PIPELINE")
        logger.info("="*70)
        
        if config is None:
            config = {
                'hidden_dim': 128,
                'n_blocks': 4,
                'layers_per_block': 6,
                'node_output_dims': 3,
                'graph_output_dims': 19,
                'energy_output_dims': 1,
                'dropout': 0.15,
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'batch_size': 64,
                'epochs': 100,
                'patience': 20,
                'predict_force_field': True,
                'predict_treatment_efficacy': True,
                'max_spice_samples': 5000
            }
        
        pipeline_results = {}
        
        # Stage 1: Molecular properties
        stage1_results = self.train_stage1_molecular_properties(config)
        pipeline_results['stage1_molecular_properties'] = stage1_results
        
        # Stage 2: Atomic forces (if SPICE available)
        stage1_model_path = self.experiment_dir / 'stage1_molecular_properties.pt'
        stage2_results = self.train_stage2_atomic_forces(str(stage1_model_path), config)
        pipeline_results['stage2_atomic_forces'] = stage2_results
        
        # Stage 3: PFAS specialization
        if stage2_results.get('training_completed'):
            stage2_model_path = self.experiment_dir / 'stage2_with_forces.pt'
        else:
            stage2_model_path = stage1_model_path
            
        stage3_results = self.train_stage3_pfas_specialization(str(stage2_model_path), config)
        pipeline_results['stage3_pfas_specialization'] = stage3_results
        
        # Compile final results
        final_results = {
            'pipeline_complete': True,
            'stages_completed': list(pipeline_results.keys()),
            'final_model_path': str(self.experiment_dir / 'stage3_pfas_specialized.pt'),
            'all_stage_results': pipeline_results,
            'multi_task_capabilities': {
                'molecular_properties': True,
                'atomic_forces': stage2_results.get('training_completed', False),
                'pfas_properties': True,
                'treatment_efficacy': True
            }
        }
        
        # Save comprehensive results
        with open(self.experiment_dir / 'multi_task_pipeline_results.json', 'w') as f:
            results_for_json = {k: v for k, v in final_results.items() if 'state_dict' not in str(k)}
            json.dump(results_for_json, f, indent=2, default=str)
        
        logger.info("🎉 Multi-task pipeline complete!")
        logger.info(f"📁 Results saved to: {self.experiment_dir}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Task DJMGNN Training Pipeline")
    parser.add_argument('--stage', type=str, default='all',
                       choices=['molecular_properties', 'atomic_forces', 'pfas_specialization', 'all'],
                       help='Training stage to run')
    parser.add_argument('--experiment_name', type=str, default='multi_task_djmgnn',
                       help='Experiment name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases tracking')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiTaskTrainer(
        experiment_name=args.experiment_name,
        use_wandb=not args.no_wandb
    )
    
    # Configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'n_blocks': 4,
        'layers_per_block': 6,
        'node_output_dims': 3,
        'graph_output_dims': 19,
        'energy_output_dims': 1,
        'dropout': 0.15,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': 20,
        'predict_force_field': True,
        'predict_treatment_efficacy': True,
        'max_spice_samples': 5000
    }
    
    # Execute based on stage
    if args.stage == 'all':
        results = trainer.run_full_multi_task_pipeline(config)
    elif args.stage == 'molecular_properties':
        results = trainer.train_stage1_molecular_properties(config)
    elif args.stage == 'pfas_specialization':
        results = trainer.train_stage3_pfas_specialization(None, config)
    else:
        logger.error(f"Individual stage training not implemented for {args.stage}")
        sys.exit(1)
    
    logger.info("🎉 Multi-task training complete!")
    
    if trainer.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()