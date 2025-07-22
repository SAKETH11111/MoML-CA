#!/usr/bin/env python3
"""
validate_djmgnn_95_percent.py

VALIDATE THE 95% ACCURACY CLAIM for DJMGNN model.

This script tests the DJMGNN model against its ACTUAL training targets:
1. QM9 molecular properties (19D graph-level)
2. SPICE atomic forces (3D node-level) 
3. Energy predictions (1D)

We need to verify what "95% accuracy" actually means and on which tasks.

Usage:
    python scripts/validate_djmgnn_95_percent.py --model_path /tmp/djmgnn_model/finetuned_model/pytorch_model.pt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from moml.models.mgnn.djmgnn import DJMGNN
from moml.utils.dataset_utils import SubsetWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DJMGNNValidationFramework:
    """Comprehensive validation framework for DJMGNN model performance."""
    
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self) -> DJMGNN:
        """Load the trained DJMGNN model with proper configuration."""
        # Load config
        config_path = os.path.join(os.path.dirname(self.model_path), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Model config: {config}")
        
        # Fix jk_mode compatibility
        jk_mode = config['jk_mode']
        if jk_mode == 'cat':
            jk_mode = 'concat'
        
        # Create model with correct input dimensions
        model = DJMGNN(
            in_node_dim=29,  # Actual trained input dimension
            in_edge_dim=config.get('in_edge_dim', 0),
            hidden_dim=config['hidden_dim'],
            n_blocks=config['n_blocks'],
            layers_per_block=config['layers_per_block'],
            node_output_dims=config['node_output_dims'],  # Should be 3 for forces
            graph_output_dims=config['graph_output_dims'],  # Should be 19 for properties
            dropout=config['dropout'],
            jk_mode=jk_mode,
            use_supernode=config['use_supernode'],
            use_rbf=config['use_rbf'],
            rbf_K=config['rbf_K'],
            pool_type='mean'
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        model = model.to(self.device)
        
        logger.info("✅ Model loaded and ready for validation")
        return model
    
    def test_qm9_properties(self, max_samples: int = 1000) -> Dict[str, Any]:
        """Test DJMGNN on QM9 molecular properties (graph-level predictions)."""
        logger.info("🧪 TESTING QM9 MOLECULAR PROPERTIES (Graph-level)")
        
        # Load QM9 test dataset with same transforms used in training
        transform = Compose([
            CreateEdges(), 
            FeaturizeNodes(), 
            StandardizeTargets(dataset_name="qm9")
        ])
        
        full_dataset = get_dataset("qm9", root="data", transform=transform)
        
        # Use test split (last 10% of dataset)
        torch.manual_seed(42)
        shuffled_indices = torch.randperm(len(full_dataset))
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_indices = shuffled_indices[train_size + val_size:]
        
        # Limit test size for faster validation
        test_indices = test_indices[:max_samples]
        test_subset = torch.utils.data.Subset(full_dataset, test_indices.tolist())
        test_dataset = SubsetWrapper(test_subset)
        test_loader = GraphDataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        logger.info(f"Testing on {len(test_dataset)} QM9 molecules...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, 'edge_attr', None),
                    batch=batch.batch
                )
                
                # Extract graph predictions (molecular properties)
                graph_pred = outputs['graph_pred']  # Should be [batch_size, 19]
                target = batch.y  # QM9 targets [batch_size, 19]
                
                all_predictions.append(graph_pred.cpu())
                all_targets.append(target.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
        
        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0).numpy()  # [N, 19]
        targets = torch.cat(all_targets, dim=0).numpy()  # [N, 19]
        
        results = {
            'dataset': 'QM9',
            'task': 'molecular_properties', 
            'predictions_shape': predictions.shape,
            'targets_shape': targets.shape,
            'property_metrics': []
        }
        
        # Calculate per-property metrics
        total_r2_scores = []
        for prop_idx in range(min(predictions.shape[1], targets.shape[1])):
            pred_prop = predictions[:, prop_idx]
            target_prop = targets[:, prop_idx]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(pred_prop) & np.isfinite(target_prop)
            if valid_mask.sum() < 10:
                continue
                
            pred_prop = pred_prop[valid_mask]
            target_prop = target_prop[valid_mask]
            
            r2 = r2_score(target_prop, pred_prop)
            mae = mean_absolute_error(target_prop, pred_prop)
            mse = mean_squared_error(target_prop, pred_prop)
            
            results['property_metrics'].append({
                'property_index': prop_idx,
                'r2_score': r2,
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'valid_samples': valid_mask.sum()
            })
            
            total_r2_scores.append(r2)
        
        # Overall metrics
        results['overall'] = {
            'mean_r2': np.mean(total_r2_scores),
            'median_r2': np.median(total_r2_scores), 
            'min_r2': np.min(total_r2_scores),
            'max_r2': np.max(total_r2_scores),
            'properties_above_90_percent': sum(r2 > 0.9 for r2 in total_r2_scores),
            'properties_above_95_percent': sum(r2 > 0.95 for r2 in total_r2_scores),
            'total_properties': len(total_r2_scores)
        }
        
        return results
    
    def test_spice_forces(self, max_samples: int = 500) -> Dict[str, Any]:
        """Test DJMGNN on SPICE atomic forces (node-level predictions)."""
        logger.info("🧪 TESTING SPICE ATOMIC FORCES (Node-level)")
        
        try:
            # Try to load SPICE dataset
            from moml.data.spice_dataset import SpiceDataset
            
            dataset = SpiceDataset(
                root='data/spice', 
                split='test',
                transform=Compose([CreateEdges(), FeaturizeNodes()])
            )
            
            # Limit samples for validation
            if len(dataset) > max_samples:
                indices = torch.randperm(len(dataset))[:max_samples]
                dataset = torch.utils.data.Subset(dataset, indices.tolist())
            
            test_loader = GraphDataLoader(dataset, batch_size=32, shuffle=False)
            
            logger.info(f"Testing on {len(dataset)} SPICE molecules...")
            
            all_force_predictions = []
            all_force_targets = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=getattr(batch, 'edge_attr', None),
                        batch=batch.batch
                    )
                    
                    # Extract node predictions (forces)
                    node_pred = outputs['node_pred']  # Should be [num_nodes, 3] for forces
                    force_target = batch.node_y if hasattr(batch, 'node_y') else None
                    
                    if force_target is not None:
                        all_force_predictions.append(node_pred.cpu())
                        all_force_targets.append(force_target.cpu())
                    
                    if (batch_idx + 1) % 5 == 0:
                        logger.info(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
            
            if all_force_predictions:
                # Compute force metrics
                predictions = torch.cat(all_force_predictions, dim=0).numpy()
                targets = torch.cat(all_force_targets, dim=0).numpy()
                
                # Calculate R² for each force component
                force_r2_scores = []
                for dim in range(min(predictions.shape[1], targets.shape[1])):
                    pred_dim = predictions[:, dim].flatten()
                    target_dim = targets[:, dim].flatten()
                    
                    valid_mask = np.isfinite(pred_dim) & np.isfinite(target_dim)
                    if valid_mask.sum() > 10:
                        r2 = r2_score(target_dim[valid_mask], pred_dim[valid_mask])
                        force_r2_scores.append(r2)
                
                return {
                    'dataset': 'SPICE',
                    'task': 'atomic_forces',
                    'predictions_shape': predictions.shape,
                    'targets_shape': targets.shape,
                    'force_r2_scores': force_r2_scores,
                    'mean_force_r2': np.mean(force_r2_scores) if force_r2_scores else 0.0,
                    'valid': True
                }
            else:
                return {'dataset': 'SPICE', 'task': 'atomic_forces', 'valid': False, 'error': 'No force targets found'}
                
        except Exception as e:
            logger.warning(f"Could not test SPICE forces: {e}")
            return {'dataset': 'SPICE', 'task': 'atomic_forces', 'valid': False, 'error': str(e)}
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation to verify the 95% accuracy claim."""
        logger.info("🚀 COMPREHENSIVE DJMGNN VALIDATION")
        logger.info("=" * 70)
        logger.info("Testing model against its ACTUAL training targets...")
        
        start_time = time.time()
        
        # Test QM9 properties
        qm9_results = self.test_qm9_properties()
        
        # Test SPICE forces
        spice_results = self.test_spice_forces()
        
        # Overall assessment
        overall_results = {
            'validation_time': time.time() - start_time,
            'qm9_molecular_properties': qm9_results,
            'spice_atomic_forces': spice_results,
            'assessment': self._assess_95_percent_claim(qm9_results, spice_results)
        }
        
        self._print_results(overall_results)
        
        return overall_results
    
    def _assess_95_percent_claim(self, qm9_results: Dict, spice_results: Dict) -> Dict[str, Any]:
        """Assess whether the 95% accuracy claim is validated."""
        assessment = {
            'claim_validated': False,
            'evidence': [],
            'interpretation': ''
        }
        
        # Check QM9 molecular properties
        if 'overall' in qm9_results:
            mean_r2 = qm9_results['overall']['mean_r2']
            props_above_95 = qm9_results['overall']['properties_above_95_percent']
            total_props = qm9_results['overall']['total_properties']
            
            if mean_r2 > 0.95:
                assessment['evidence'].append(f"✅ Mean R² across molecular properties: {mean_r2:.3f} > 0.95")
                assessment['claim_validated'] = True
            else:
                assessment['evidence'].append(f"⚠️ Mean R² across molecular properties: {mean_r2:.3f} < 0.95")
            
            if props_above_95 / total_props > 0.5:
                assessment['evidence'].append(f"✅ {props_above_95}/{total_props} properties have R² > 0.95")
            else:
                assessment['evidence'].append(f"⚠️ Only {props_above_95}/{total_props} properties have R² > 0.95")
        
        # Check SPICE forces
        if spice_results.get('valid', False):
            force_r2 = spice_results.get('mean_force_r2', 0.0)
            if force_r2 > 0.90:
                assessment['evidence'].append(f"✅ Mean force prediction R²: {force_r2:.3f}")
            else:
                assessment['evidence'].append(f"⚠️ Mean force prediction R²: {force_r2:.3f}")
        else:
            assessment['evidence'].append("⚠️ Could not validate force predictions")
        
        # Final interpretation
        if assessment['claim_validated']:
            assessment['interpretation'] = "✅ 95% ACCURACY CLAIM VALIDATED on molecular properties"
        else:
            assessment['interpretation'] = "⚠️ 95% accuracy claim needs clarification - specify metric and task"
        
        return assessment
    
    def _print_results(self, results: Dict[str, Any]):
        """Print comprehensive validation results."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 DJMGNN 95% ACCURACY VALIDATION RESULTS")
        logger.info("=" * 70)
        
        # QM9 Results
        qm9 = results['qm9_molecular_properties']
        if 'overall' in qm9:
            overall = qm9['overall']
            logger.info(f"\n🧬 QM9 MOLECULAR PROPERTIES:")
            logger.info(f"   Mean R²: {overall['mean_r2']:.3f}")
            logger.info(f"   Properties R² > 0.95: {overall['properties_above_95_percent']}/{overall['total_properties']}")
            logger.info(f"   Properties R² > 0.90: {overall['properties_above_90_percent']}/{overall['total_properties']}")
            logger.info(f"   Best property R²: {overall['max_r2']:.3f}")
        
        # SPICE Results
        spice = results['spice_atomic_forces']
        if spice.get('valid', False):
            logger.info(f"\n⚛️ SPICE ATOMIC FORCES:")
            logger.info(f"   Mean Force R²: {spice['mean_force_r2']:.3f}")
            logger.info(f"   Force components: {len(spice['force_r2_scores'])} tested")
        else:
            logger.info(f"\n⚛️ SPICE ATOMIC FORCES: ❌ Could not validate")
        
        # Assessment
        assessment = results['assessment']
        logger.info(f"\n🎯 95% ACCURACY CLAIM ASSESSMENT:")
        for evidence in assessment['evidence']:
            logger.info(f"   {evidence}")
        
        logger.info(f"\n{assessment['interpretation']}")
        
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate DJMGNN 95% accuracy claim")
    parser.add_argument('--model_path', type=str, 
                       default='/tmp/djmgnn_model/finetuned_model/pytorch_model.pt',
                       help='Path to DJMGNN model')
    parser.add_argument('--output_file', type=str, default='djmgnn_95_validation.json',
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Run validation
    validator = DJMGNNValidationFramework(args.model_path)
    results = validator.run_comprehensive_validation()
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📁 Detailed results saved to: {args.output_file}")
    
    # Return appropriate exit code
    if results['assessment']['claim_validated']:
        logger.info("🎉 VALIDATION PASSED!")
        sys.exit(0)
    else:
        logger.info("⚠️ VALIDATION INCONCLUSIVE")
        sys.exit(1)


if __name__ == '__main__':
    main()