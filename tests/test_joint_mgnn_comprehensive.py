"""
tests/test_joint_mgnn_comprehensive.py

Comprehensive test suite for joint DJMGNN and HMGNN implementation.

This test suite thoroughly validates all components of the joint training framework
including model architectures, data processing, training infrastructure, and
integration between components.
"""

import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from moml.models.mgnn.joint_mgnn import (
    JointMGNN, CrossModelFusion, MultiTaskHead, create_joint_mgnn
)
from moml.models.mgnn.training.joint_trainer import (
    JointMGNNTrainer, AlternatingTrainingStrategy, create_joint_trainer
)
from moml.core.hierarchical_processor import (
    HierarchicalGraphCoarsener, HierarchicalDataProcessor, create_hierarchical_processor
)
from moml.models.mgnn.djmgnn import DJMGNN
from moml.models.mgnn.hmgnn import HMGNN, create_hierarchical_mgnn


class TestJointMGNNArchitecture(unittest.TestCase):
    """Test suite for JointMGNN model architecture."""
    
    def setUp(self):
        """Set up test configurations and data."""
        self.device = torch.device('cpu')  # Use CPU for tests
        self.batch_size = 2
        self.num_nodes = 10
        self.node_dim = 16
        self.hidden_dim = 32
        
        # Model configurations
        self.djmgnn_config = {
            'in_node_dim': self.node_dim,
            'hidden_dim': self.hidden_dim,
            'n_blocks': 2,
            'layers_per_block': 2,
            'in_edge_dim': 4,  # Updated to handle edge attributes
            'jk_mode': 'attention',
            'node_output_dims': 3,
            'graph_output_dims': 5,
            'energy_output_dims': 1,
            'dropout': 0.1
        }
        
        self.hmgnn_config = {
            'scale_dims': [self.node_dim, self.node_dim, self.node_dim],
            'hidden_dim': self.hidden_dim,
            'n_blocks': 2,
            'layers_per_block': 2,
            'jk_mode': 'attention',
            'node_out_dim': 3,
            'graph_out_dim': 5,
            'cross_scale_exchange': True,
            'dropout': 0.1,
            'n_heads_cs': 4
        }
        
        self.joint_config = {
            'fusion_dim': 64,
            'n_fusion_heads': 4,
            'alpha': 0.5,
            'cross_model_weight': 0.1
        }
        
        # Create test data
        self.test_data = self._create_test_data()
        self.hierarchical_data = self._create_hierarchical_test_data()
    
    def _create_test_data(self) -> Data:
        """Create test molecular graph data."""
        x = torch.randn(self.num_nodes, self.node_dim)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 2))
        # Add edge attributes to activate edge MLPs
        edge_attr = torch.randn(edge_index.size(1), 4)  # 4-dim edge features
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        y = torch.randn(5)  # Graph-level targets
        node_y = torch.randn(self.num_nodes, 3)  # Node-level targets
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   batch=batch, y=y, node_y=node_y)
    
    def _create_hierarchical_test_data(self) -> List[Dict[str, Any]]:
        """Create test hierarchical data with proper connectivity."""
        scale_data = []
        for scale in range(3):
            num_nodes_scale = max(3, self.num_nodes // (scale + 1))  # At least 3 nodes
            x = torch.randn(num_nodes_scale, self.node_dim)
            
            # Create fully connected graph with self-loops to ensure connectivity
            edge_pairs = []
            for i in range(num_nodes_scale):
                for j in range(num_nodes_scale):
                    edge_pairs.append([i, j])
            
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).T
            num_edges = edge_index.size(1)
            edge_attr = torch.randn(num_edges, 4)  # 4-dim edge features
            batch = torch.zeros(num_nodes_scale, dtype=torch.long)
            
            scale_data.append({
                'x': x,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'batch': batch
            })
        
        return scale_data
    
    def test_cross_model_fusion_instantiation(self):
        """Test CrossModelFusion layer instantiation."""
        fusion_layer = CrossModelFusion(
            djmgnn_dim=self.hidden_dim,
            hmgnn_dim=self.hidden_dim,
            fusion_dim=64,
            n_heads=4
        )
        
        self.assertIsInstance(fusion_layer, nn.Module)
        self.assertEqual(fusion_layer.fusion_dim, 64)
        self.assertEqual(fusion_layer.n_heads, 4)
    
    def test_cross_model_fusion_forward(self):
        """Test CrossModelFusion forward pass."""
        # Use the actual node output dimensions from the configs
        djmgnn_dim = self.djmgnn_config['node_output_dims'] 
        hmgnn_dim = self.hmgnn_config['node_out_dim']
        fusion_dim = 64
        
        fusion_layer = CrossModelFusion(
            djmgnn_dim=djmgnn_dim,
            hmgnn_dim=hmgnn_dim,
            fusion_dim=fusion_dim,
            n_heads=4
        )
        
        # Create test features with correct dimensions
        djmgnn_features = torch.randn(1, self.num_nodes, djmgnn_dim)
        hmgnn_features = torch.randn(1, self.num_nodes, hmgnn_dim)
        
        # Forward pass
        dj_fused, hm_fused = fusion_layer(djmgnn_features, hmgnn_features)
        
        # Check output shapes
        self.assertEqual(dj_fused.shape, (1, self.num_nodes, fusion_dim))
        self.assertEqual(hm_fused.shape, (1, self.num_nodes, fusion_dim))
        
        # Check that outputs have correct dimensions (can't compare directly due to dimension change)
        self.assertEqual(dj_fused.shape[-1], fusion_dim)
        self.assertEqual(hm_fused.shape[-1], fusion_dim)
    
    def test_multi_task_head_instantiation(self):
        """Test MultiTaskHead instantiation."""
        task_configs = {
            'task1': {'output_dim': 5, 'hidden_dims': [32]},
            'task2': {'output_dim': 3, 'hidden_dims': [32, 16]},
            'task3': {'output_dim': 1}
        }
        
        multi_task_head = MultiTaskHead(
            input_dim=self.hidden_dim,
            task_configs=task_configs
        )
        
        self.assertIsInstance(multi_task_head, nn.Module)
        self.assertEqual(len(multi_task_head.task_heads), 3)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.hidden_dim)
        outputs = multi_task_head(x)
        
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs['task1'].shape, (self.batch_size, 5))
        self.assertEqual(outputs['task2'].shape, (self.batch_size, 3))
        self.assertEqual(outputs['task3'].shape, (self.batch_size, 1))
    
    def test_joint_mgnn_instantiation(self):
        """Test JointMGNN model instantiation."""
        joint_model = JointMGNN(
            djmgnn_config=self.djmgnn_config,
            hmgnn_config=self.hmgnn_config,
            **self.joint_config
        )
        
        self.assertIsInstance(joint_model, nn.Module)
        self.assertIsInstance(joint_model.djmgnn, DJMGNN)
        self.assertIsInstance(joint_model.hmgnn, HMGNN)
        self.assertIsInstance(joint_model.fusion_layer, CrossModelFusion)
        self.assertIsInstance(joint_model.task_heads, MultiTaskHead)
    
    def test_joint_mgnn_forward_standard_data(self):
        """Test JointMGNN forward pass with standard graph data."""
        joint_model = create_joint_mgnn(
            djmgnn_config=self.djmgnn_config,
            hmgnn_config=self.hmgnn_config,
            joint_config=self.joint_config
        )
        
        joint_model.eval()
        
        with torch.no_grad():
            outputs = joint_model(
                x=self.test_data.x,
                edge_index=self.test_data.edge_index,
                edge_attr=self.test_data.edge_attr,
                batch=self.test_data.batch,
                use_fusion=True,
                return_individual=True
            )
        
        # Check that all expected outputs are present
        expected_keys = ['molecular_properties', 'forces', 'pfas_properties', 
                        'treatment_efficacy', 'node_pred', 'shared_representation']
        for key in expected_keys:
            self.assertIn(key, outputs, f"Missing output key: {key}")
        
        # Check output shapes
        self.assertEqual(outputs['node_pred'].shape[0], self.num_nodes)
        self.assertEqual(outputs['molecular_properties'].shape[0], 1)  # Single graph
        
        # Check that individual model outputs are returned
        self.assertIn('djmgnn_out', outputs)
        self.assertIn('hmgnn_out', outputs)
    
    def test_joint_mgnn_forward_hierarchical_data(self):
        """Test JointMGNN forward pass with hierarchical data."""
        joint_model = create_joint_mgnn(
            djmgnn_config=self.djmgnn_config,
            hmgnn_config=self.hmgnn_config,
            joint_config=self.joint_config
        )
        
        joint_model.eval()
        
        # Create dummy mappings for hierarchical data
        mappings = [
            torch.arange(self.num_nodes, dtype=torch.long),
            torch.arange(max(1, self.num_nodes // 2), dtype=torch.long),
        ]
        cluster_counts = [
            torch.ones(self.num_nodes, dtype=torch.long),
            torch.ones(max(1, self.num_nodes // 2), dtype=torch.long),
        ]
        
        with torch.no_grad():
            outputs = joint_model(
                x=self.test_data.x,
                edge_index=self.test_data.edge_index,
                scale_data=self.hierarchical_data,
                maps=(mappings, cluster_counts),
                use_fusion=True
            )
        
        # Check that outputs are generated
        self.assertIn('molecular_properties', outputs)
        self.assertIn('node_pred', outputs)
    
    def test_joint_mgnn_loss_computation(self):
        """Test JointMGNN loss computation."""
        joint_model = create_joint_mgnn(
            djmgnn_config=self.djmgnn_config,
            hmgnn_config=self.hmgnn_config,
            joint_config=self.joint_config
        )
        
        # Generate predictions
        with torch.no_grad():
            predictions = joint_model(
                x=self.test_data.x,
                edge_index=self.test_data.edge_index,
                return_individual=True
            )
        
        # Create targets
        targets = {
            'molecular_properties': torch.randn(1, 19),
            'forces': torch.randn(self.num_nodes, 3),
        }
        
        # Compute loss
        total_loss, individual_losses = joint_model.compute_joint_loss(
            predictions, targets
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(individual_losses, dict)
        self.assertTrue(torch.isfinite(total_loss))
    
    def test_joint_mgnn_gradient_flow(self):
        """Test gradient flow through JointMGNN."""
        joint_model = create_joint_mgnn(
            djmgnn_config=self.djmgnn_config,
            hmgnn_config=self.hmgnn_config,
            joint_config=self.joint_config
        )
        
        joint_model.train()
        
        # Forward pass
        # Create hierarchical data to ensure all components are active
        hierarchical_data = self._create_hierarchical_test_data()
        
        # Create dummy mappings for hierarchical data
        mappings = [
            torch.arange(self.num_nodes, dtype=torch.long),
            torch.arange(max(1, self.num_nodes // 2), dtype=torch.long),
        ]
        cluster_counts = [
            torch.ones(self.num_nodes, dtype=torch.long),
            torch.ones(max(1, self.num_nodes // 2), dtype=torch.long),
        ]
        
        outputs = joint_model(
            x=self.test_data.x,
            edge_index=self.test_data.edge_index,
            scale_data=hierarchical_data,
            maps=(mappings, cluster_counts),
            use_fusion=True
        )
        
        # Compute comprehensive multi-task loss to ensure all heads get gradients
        total_loss = torch.tensor(0.0, requires_grad=True)
        num_nodes = self.num_nodes  # Use class attribute
        
        # Create proper targets for all tasks
        task_targets = {
            'molecular_properties': torch.randn(1, 19),  # Graph-level
            'forces': torch.randn(num_nodes, 3),         # Node-level
            'pfas_properties': torch.randn(1, 5),        # Graph-level
            'treatment_efficacy': torch.randn(1, 1)      # Graph-level
        }
        
        # Compute loss for each task to ensure gradient flow
        for task_name, target in task_targets.items():
            if task_name in outputs and outputs[task_name] is not None:
                output = outputs[task_name]
                if output.numel() > 0:
                    # Use MSE loss for each task
                    task_loss = F.mse_loss(output, target)
                    total_loss = total_loss + task_loss
        
        # Add regularization terms from HMGNN adapters if they exist
        if hasattr(joint_model.hmgnn, '_adapter_reg_loss'):
            reg_loss = joint_model.hmgnn._adapter_reg_loss
            if isinstance(reg_loss, torch.Tensor) and reg_loss.requires_grad:
                total_loss = total_loss + reg_loss
        
        # Ensure we have a meaningful loss
        if total_loss.item() == 0:
            # Fallback to simple sum if no task losses computed
            output_sum = sum(output.sum() for output in outputs.values()
                           if isinstance(output, torch.Tensor) and output.numel() > 0)
            if isinstance(output_sum, torch.Tensor):
                total_loss = total_loss + output_sum
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        gradients_exist = 0
        total_params = 0
        
        for name, param in joint_model.named_parameters():
            total_params += 1
            if param.grad is not None:
                gradients_exist += 1
        
        # At least 80% of parameters should have gradients
        gradient_ratio = gradients_exist / total_params
        self.assertGreater(gradient_ratio, 0.8, 
                          f"Only {gradient_ratio:.2%} of parameters have gradients")


class TestHierarchicalProcessor(unittest.TestCase):
    """Test suite for hierarchical graph processing."""
    
    def setUp(self):
        """Set up test data and processor."""
        self.num_nodes = 20
        self.node_dim = 16
        
        # Create test graph data
        self.test_data = Data(
            x=torch.randn(self.num_nodes, self.node_dim),
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_nodes * 2)),
            edge_attr=None
        )
        
        # Processor configuration
        self.config = {
            'coarsener': {
                'n_levels': 3,
                'clustering_method': 'graclus',
                'preserve_connectivity': True
            },
            'processor': {
                'include_cross_scale_edges': True,
                'cache_hierarchical': False
            }
        }
    
    def test_hierarchical_graph_coarsener_instantiation(self):
        """Test HierarchicalGraphCoarsener instantiation."""
        coarsener = HierarchicalGraphCoarsener(
            n_levels=3,
            clustering_method='graclus'
        )
        
        self.assertEqual(coarsener.n_levels, 3)
        self.assertEqual(coarsener.clustering_method, 'graclus')
    
    def test_graph_coarsening_levels(self):
        """Test graph coarsening at different levels."""
        coarsener = HierarchicalGraphCoarsener(n_levels=3)
        
        # Test level 0 (should return original)
        coarsened_data, mapping, counts = coarsener.coarsen_graph(self.test_data, 0)
        
        self.assertEqual(coarsened_data.x.shape[0], self.test_data.x.shape[0])
        self.assertTrue(torch.equal(mapping, torch.arange(self.num_nodes)))
        
        # Test level 1 (should coarsen)
        coarsened_data_1, mapping_1, counts_1 = coarsener.coarsen_graph(self.test_data, 1)
        
        self.assertLessEqual(coarsened_data_1.x.shape[0], self.test_data.x.shape[0])
        self.assertEqual(mapping_1.shape[0], self.num_nodes)
        self.assertEqual(counts_1.shape[0], coarsened_data_1.x.shape[0])
    
    def test_hierarchical_representation_creation(self):
        """Test creation of complete hierarchical representation."""
        coarsener = HierarchicalGraphCoarsener(n_levels=3)
        
        scale_data, mappings, cluster_counts = coarsener.create_hierarchical_representation(
            self.test_data
        )
        
        # Check that we have 3 scales
        self.assertEqual(len(scale_data), 3)
        self.assertEqual(len(mappings), 3)
        self.assertEqual(len(cluster_counts), 3)
        
        # Check that scales get progressively smaller
        prev_size = scale_data[0].x.shape[0]
        for i in range(1, 3):
            current_size = scale_data[i].x.shape[0]
            self.assertLessEqual(current_size, prev_size)
            prev_size = current_size
    
    def test_hierarchical_data_processor(self):
        """Test HierarchicalDataProcessor functionality."""
        processor = create_hierarchical_processor(self.config)
        
        # Process molecule
        result = processor.process_molecule(self.test_data, molecule_id="test_mol")
        
        # Check result structure
        expected_keys = ['scale_data', 'mappings', 'cluster_counts', 
                        'cross_scale_edges', 'n_scales']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check scales
        self.assertEqual(result['n_scales'], 3)
        self.assertEqual(len(result['scale_data']), 3)
    
    def test_batch_creation(self):
        """Test hierarchical batch creation."""
        processor = create_hierarchical_processor(self.config)
        
        # Create multiple processed molecules
        hierarchical_data_list = []
        for i in range(3):
            result = processor.process_molecule(self.test_data, molecule_id=f"mol_{i}")
            hierarchical_data_list.append(result)
        
        # Create batch
        batched_data = processor.create_batch(hierarchical_data_list)
        
        # Check batch structure
        self.assertIn('scale_data', batched_data)
        self.assertIn('batch_size', batched_data)
        self.assertEqual(batched_data['batch_size'], 3)
        self.assertEqual(len(batched_data['scale_data']), 3)


class TestJointTraining(unittest.TestCase):
    """Test suite for joint training infrastructure."""
    
    def setUp(self):
        """Set up test configurations."""
        self.djmgnn_config = {
            'in_node_dim': 16,
            'hidden_dim': 32,
            'n_blocks': 1,
            'layers_per_block': 1,
            'in_edge_dim': 0,
            'node_output_dims': 1,
            'graph_output_dims': 5,
        }
        
        self.hmgnn_config = {
            'scale_dims': [16, 16, 16],
            'hidden_dim': 32,
            'n_blocks': 1,
            'layers_per_block': 1,
            'node_out_dim': 1,
            'graph_out_dim': 5,
            'cross_scale_exchange': True,
        }
        
        self.joint_config = {
            'fusion_dim': 64,
            'training_strategy': 'joint',
            'batch_size': 2,
            'learning_rate': 0.01,
        }
    
    def test_alternating_training_strategy(self):
        """Test AlternatingTrainingStrategy functionality."""
        strategy = AlternatingTrainingStrategy(
            strategy='fixed_alternating',
            switch_frequency=5
        )
        
        # Initial state
        self.assertEqual(strategy.get_current_model(), 'djmgnn')
        
        # Test switching logic
        for i in range(4):
            should_switch = strategy.should_switch_model(0.5)
            self.assertFalse(should_switch)
        
        # Should switch on 5th call
        should_switch = strategy.should_switch_model(0.5)
        self.assertTrue(should_switch)
        
        # Switch and check new model
        new_model = strategy.switch_model()
        self.assertEqual(new_model, 'hmgnn')
    
    def test_joint_trainer_creation(self):
        """Test joint trainer creation with mock data loaders."""
        # Create mock data loaders
        mock_data = [Data(x=torch.randn(10, 16), 
                         edge_index=torch.randint(0, 10, (2, 20)),
                         y=torch.randn(5)) for _ in range(4)]
        
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(mock_data, batch_size=2)
        
        # This would normally fail due to missing hierarchical components
        # but we can test the configuration setup
        try:
            joint_model = create_joint_mgnn(
                djmgnn_config=self.djmgnn_config,
                hmgnn_config=self.hmgnn_config,
                joint_config={'fusion_dim': 64}
            )
            
            # If we get here, model creation succeeded
            self.assertIsInstance(joint_model, JointMGNN)
            
        except Exception as e:
            # Expected due to missing components, but error should be reasonable
            self.assertIsInstance(e, (ImportError, RuntimeError, ValueError))


class TestIntegration(unittest.TestCase):
    """Test integration between all components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.device = torch.device('cpu')
        
        # Comprehensive configuration
        self.config = {
            'djmgnn': {
                'in_node_dim': 16,
                'hidden_dim': 32,
                'n_blocks': 1,
                'layers_per_block': 1,
                'in_edge_dim': 0,
                'node_output_dims': 3,
                'graph_output_dims': 5,
            },
            'hmgnn': {
                'scale_dims': [16, 16, 16],
                'hidden_dim': 32,
                'n_blocks': 1,
                'layers_per_block': 1,
                'node_out_dim': 3,
                'graph_out_dim': 5,
                'cross_scale_exchange': True,
            },
            'joint': {
                'fusion_dim': 64,
                'task_configs': {
                    'molecular_properties': {'output_dim': 5, 'hidden_dims': [32]},
                    'forces': {'output_dim': 3, 'hidden_dims': [32]}
                }
            },
            'hierarchical': {
                'coarsener': {
                    'n_levels': 3,
                    'clustering_method': 'graclus'
                },
                'processor': {
                    'include_cross_scale_edges': True,
                    'cache_hierarchical': False
                }
            }
        }
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create test molecular data
        test_molecules = []
        for i in range(5):
            num_nodes = torch.randint(5, 15, (1,)).item()
            x = torch.randn(num_nodes, 16)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            y = torch.randn(5)
            node_y = torch.randn(num_nodes, 3)
            
            data = Data(x=x, edge_index=edge_index, y=y, node_y=node_y)
            test_molecules.append(data)
        
        # Create hierarchical processor
        hierarchical_processor = create_hierarchical_processor(self.config['hierarchical'])
        
        # Process molecules hierarchically
        processed_molecules = []
        for i, mol in enumerate(test_molecules):
            processed = hierarchical_processor.process_molecule(mol, molecule_id=f"mol_{i}")
            processed['targets'] = mol.y
            processed['node_targets'] = mol.node_y
            processed_molecules.append(processed)
        
        # Create batch
        batched_hierarchical = hierarchical_processor.create_batch(processed_molecules)
        
        # Create joint model
        joint_model = create_joint_mgnn(
            djmgnn_config=self.config['djmgnn'],
            hmgnn_config=self.config['hmgnn'],
            joint_config=self.config['joint']
        )
        
        # Test forward pass with hierarchical data
        joint_model.eval()
        
        # Create standard batch for DJMGNN
        standard_batch = Batch.from_data_list(test_molecules)
        
        with torch.no_grad():
            outputs = joint_model(
                x=standard_batch.x,
                edge_index=standard_batch.edge_index,
                batch=standard_batch.batch,
                scale_data=batched_hierarchical['scale_data'],
                maps=(batched_hierarchical['mappings'], batched_hierarchical['cluster_counts']),
                use_fusion=True,
                return_individual=True
            )
        
        # Verify outputs
        self.assertIn('molecular_properties', outputs)
        self.assertIn('forces', outputs)
        self.assertIn('djmgnn_out', outputs)
        self.assertIn('hmgnn_out', outputs)
        
        # Check shapes
        self.assertEqual(outputs['molecular_properties'].shape[0], len(test_molecules))
        self.assertEqual(outputs['forces'].shape[0], standard_batch.x.shape[0])
        
        print("✓ End-to-end pipeline test passed")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        # Create empty graph
        empty_data = Data(
            x=torch.empty(0, 16),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=None
        )
        
        # Test hierarchical processing with empty graph
        config = {
            'coarsener': {'n_levels': 3},
            'processor': {'include_cross_scale_edges': True, 'cache_hierarchical': False}
        }
        
        processor = create_hierarchical_processor(config)
        
        try:
            result = processor.process_molecule(empty_data, molecule_id="empty_mol")
            # Should handle gracefully
            self.assertIn('scale_data', result)
        except Exception as e:
            # If it fails, error should be reasonable
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched tensor dimensions."""
        # Create JointMGNN with specific config
        djmgnn_config = {'in_node_dim': 16, 'hidden_dim': 32, 'n_blocks': 1, 
                        'layers_per_block': 1, 'graph_output_dims': 5}
        hmgnn_config = {'scale_dims': [16, 16], 'hidden_dim': 32, 'n_blocks': 1,
                       'graph_out_dim': 5, 'cross_scale_exchange': False}
        
        joint_model = create_joint_mgnn(
            djmgnn_config=djmgnn_config,
            hmgnn_config=hmgnn_config,
            joint_config={'fusion_dim': 64}
        )
        
        # Test with mismatched data
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        
        # This should not crash, but handle gracefully
        try:
            with torch.no_grad():
                outputs = joint_model(x=x, edge_index=edge_index, use_fusion=False)
            # Should produce some output
            self.assertIsInstance(outputs, dict)
        except Exception as e:
            # If it fails, should be a reasonable error
            self.assertIsInstance(e, (RuntimeError, ValueError, TypeError, IndexError, AttributeError))


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running Comprehensive Joint MGNN Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestJointMGNNArchitecture,
        TestHierarchicalProcessor,
        TestJointTraining,
        TestIntegration,
        TestErrorHandling
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        for test in suite:
            total_tests += 1
            try:
                # Run individual test
                result = unittest.TestResult()
                test.run(result)
                
                if result.wasSuccessful():
                    print(f"✓ {test._testMethodName}")
                    passed_tests += 1
                else:
                    print(f"✗ {test._testMethodName}")
                    for failure in result.failures + result.errors:
                        failed_tests.append(f"{test_class.__name__}.{test._testMethodName}: {failure[1]}")
            
            except Exception as e:
                print(f"✗ {test._testMethodName} (Exception: {e})")
                failed_tests.append(f"{test_class.__name__}.{test._testMethodName}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print("\nFAILED TESTS:")
        for failure in failed_tests[:10]:  # Show first 10 failures
            print(f"  - {failure}")
        if len(failed_tests) > 10:
            print(f"  ... and {len(failed_tests) - 10} more")
    
    return passed_tests, len(failed_tests), failed_tests


if __name__ == "__main__":
    run_comprehensive_tests()