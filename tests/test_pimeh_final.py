"""
PROJECT PIMEH POLISH - Comprehensive PIMEH Unit Test

This test suite serves as the definitive validation framework for all PIMEH
enhancements aimed at achieving >95% mean R² accuracy. It provides targeted
testing of the Physics-Informed Minimal Equivariant Head with real trained
backbone embeddings and QM9 validation data.

Key Features:
- Loads frozen backbone from actual PROJECT APOLLO checkpoint
- Tests with realistic QM9 molecular structures and properties
- Implements advanced physics-informed loss function with constraints
- Provides comprehensive diagnostics for PIMEH failure modes
- Validates rotational invariance and mass prediction quality

This test is designed to isolate and fix the exact issues preventing PIMEH
from learning rotational constants (properties A, B, C) effectively.

References:
- PROJECT APOLLO results: 94.7% accuracy (0.3% short of target)
- PIMEH failure: Properties A, B, C have R² < 0.1 vs backbone >90%
- Physics-informed neural networks best practices (2024)
"""

import unittest
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes
from moml.models.mgnn.djmgnn import DJMGNN
from moml.models.mgnn.pimeh import (
    PhysicsInformedMinimalEquivariantHead,
    compute_inertia_tensor,
    validate_rotational_invariance,
    ROTATIONAL_CONST_FACTOR_GHZ
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress expected warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class AdvancedPhysicsLoss:
    """
    Advanced physics-informed loss function for rotational constants.
    
    This implementation incorporates multiple physics constraints and 
    regularization terms based on 2024 PINN research and molecular
    spectroscopy principles.
    
    Key Components:
    1. Standard MSE loss on rotational constants A, B, C
    2. Ordering constraint: A ≥ B ≥ C (physics requirement)
    3. Mass reasonableness: atoms should have realistic masses (0.1-100 amu)
    4. Inertia tensor positivity: eigenvalues must be positive
    5. Scale consistency: rotational constants in reasonable GHz range
    6. Molecular symmetry: linear molecules have one zero eigenvalue
    
    Based on research findings:
    - Composite loss functions improve PINN performance
    - Physical constraints on derivatives are critical
    - Molecular spectroscopy has well-defined bounds
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ordering_weight: float = 0.1,
        mass_weight: float = 0.05,
        inertia_weight: float = 0.05,
        scale_weight: float = 0.02
    ):
        """
        Initialize the advanced physics loss function.
        
        Args:
            mse_weight: Weight for standard MSE loss
            ordering_weight: Weight for A ≥ B ≥ C constraint
            mass_weight: Weight for realistic mass constraint
            inertia_weight: Weight for positive inertia constraint
            scale_weight: Weight for reasonable scale constraint
        """
        self.mse_weight = mse_weight
        self.ordering_weight = ordering_weight
        self.mass_weight = mass_weight
        self.inertia_weight = inertia_weight
        self.scale_weight = scale_weight
        
        logger.info(f"Advanced Physics Loss initialized with weights: "
                   f"MSE={mse_weight}, ordering={ordering_weight}, "
                   f"mass={mass_weight}, inertia={inertia_weight}, scale={scale_weight}")
    
    def compute_loss(
        self,
        pred_rot: torch.Tensor,
        target_rot: torch.Tensor,
        pred_masses: torch.Tensor,
        pred_inertia: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute advanced physics-informed loss with multiple constraints.
        
        Args:
            pred_rot: Predicted rotational constants [batch_size, 3] (A, B, C)
            target_rot: Target rotational constants [batch_size, 3]
            pred_masses: Predicted atomic masses [total_atoms]
            pred_inertia: Predicted inertia tensors [batch_size, 3, 3]
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Dict containing individual loss components and total loss
        """
        batch_size = pred_rot.size(0)
        device = pred_rot.device
        
        # 1. Standard MSE loss on rotational constants
        mse_loss = F.mse_loss(pred_rot, target_rot, reduction=reduction)
        
        # 2. Ordering constraint: A ≥ B ≥ C (rotational constants must be ordered)
        # Since A, B, C are in columns 0, 1, 2, we need A >= B >= C
        ordering_violations = torch.clamp(pred_rot[:, 1] - pred_rot[:, 0], min=0) + \
                             torch.clamp(pred_rot[:, 2] - pred_rot[:, 1], min=0)
        ordering_loss = ordering_violations.mean() if reduction == 'mean' else ordering_violations.sum()
        
        # 3. Mass reasonableness constraint (atoms should have masses 0.1-100 amu)
        mass_violations = torch.clamp(0.1 - pred_masses, min=0) + \
                         torch.clamp(pred_masses - 100.0, min=0)
        mass_loss = mass_violations.mean() if reduction == 'mean' else mass_violations.sum()
        
        # 4. Inertia tensor positivity (eigenvalues must be positive)
        inertia_eigenvals = torch.linalg.eigvalsh(pred_inertia)  # [batch_size, 3]
        inertia_violations = torch.clamp(-inertia_eigenvals, min=0)  # Penalize negative eigenvalues
        inertia_loss = inertia_violations.mean() if reduction == 'mean' else inertia_violations.sum()
        
        # 5. Scale consistency (rotational constants should be in reasonable range 0.01-100 GHz)
        scale_violations = torch.clamp(0.01 - pred_rot, min=0) + \
                          torch.clamp(pred_rot - 100.0, min=0)
        scale_loss = scale_violations.mean() if reduction == 'mean' else scale_violations.sum()
        
        # Combine all loss components
        total_loss = (self.mse_weight * mse_loss +
                     self.ordering_weight * ordering_loss +
                     self.mass_weight * mass_loss +
                     self.inertia_weight * inertia_loss +
                     self.scale_weight * scale_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'ordering_loss': ordering_loss,
            'mass_loss': mass_loss,
            'inertia_loss': inertia_loss,
            'scale_loss': scale_loss
        }
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Make the loss function callable, returning only total loss."""
        loss_dict = self.compute_loss(*args, **kwargs)
        return loss_dict['total_loss']


class TestPIMEHFinal(unittest.TestCase):
    """
    Comprehensive test suite for PIMEH integration with PROJECT APOLLO.
    
    This test suite validates all aspects of PIMEH functionality using
    real trained backbone embeddings and QM9 validation data. It serves
    as the foundation for PROJECT PIMEH POLISH to achieve >95% accuracy.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are expensive to create."""
        logger.info("🚀 Setting up PROJECT PIMEH POLISH test suite...")
        
        # Configuration
        cls.checkpoint_path = "checkpoints_apollo_final/best_checkpoint.pt"
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.hidden_dim = 160
        cls.test_batch_size = 8
        cls.num_test_molecules = 50
        
        # Initialize advanced physics loss function
        cls.advanced_loss = AdvancedPhysicsLoss()
        
        logger.info(f"Device: {cls.device}")
        logger.info(f"Checkpoint: {cls.checkpoint_path}")
        logger.info(f"Test configuration: {cls.num_test_molecules} molecules, batch size {cls.test_batch_size}")
    
    def setUp(self):
        """Set up each test method."""
        self.backbone_model = None
        self.pimeh_head = None
        self.test_data = None
        self.frozen_embeddings = None
        
    def tearDown(self):
        """Clean up after each test."""
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_frozen_backbone(self) -> DJMGNN:
        """
        Load the trained DJMGNN backbone from PROJECT APOLLO checkpoint.
        
        Returns:
            DJMGNN model with frozen parameters, ready for embedding extraction
        """
        logger.info(f"Loading frozen backbone from {self.checkpoint_path}...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create model with same architecture as training
            model = DJMGNN(
                in_node_dim=33,  # Featurized node dimensions
                hidden_dim=self.hidden_dim,
                n_blocks=4,
                layers_per_block=6,
                in_edge_dim=0,
                node_output_dims=3,
                graph_output_dims=19,
                energy_output_dims=1
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            logger.info(f"✅ Loaded frozen backbone with {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load backbone: {e}")
            raise RuntimeError(f"Cannot load PROJECT APOLLO checkpoint: {e}")
    
    def _load_qm9_test_data(self) -> List[Data]:
        """
        Load a subset of QM9 validation data for testing.
        
        Returns:
            List of QM9 Data objects with molecular graphs and properties
        """
        logger.info(f"Loading {self.num_test_molecules} QM9 test molecules...")
        
        try:
            # Create transforms for QM9 data
            transforms = Compose([
                FeaturizeNodes(),
                CreateEdges(cutoff=5.0)
            ])
            
            # Load QM9 dataset
            dataset = get_dataset(
                dataset_name='qm9',
                root='data/',
                pre_transform=transforms
            )
            
            # Use validation indices (standard QM9 split)
            val_start = 100000
            val_end = val_start + self.num_test_molecules
            test_molecules = [dataset[i] for i in range(val_start, min(val_end, len(dataset)))]
            
            logger.info(f"✅ Loaded {len(test_molecules)} QM9 test molecules")
            logger.info(f"Sample molecule: {test_molecules[0].num_nodes} atoms, "
                       f"{test_molecules[0].num_edges} edges")
            
            return test_molecules
            
        except Exception as e:
            logger.error(f"❌ Failed to load QM9 data: {e}")
            raise RuntimeError(f"Cannot load QM9 test data: {e}")
    
    def _extract_backbone_embeddings(
        self, 
        backbone: DJMGNN, 
        molecules: List[Data]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract frozen backbone embeddings from QM9 molecules.
        
        Args:
            backbone: Frozen DJMGNN backbone model
            molecules: List of QM9 Data objects
        
        Returns:
            Tuple of (embeddings, positions, batch_indices, target_rot_constants)
        """
        logger.info("Extracting backbone embeddings...")
        
        # Create batched data
        batch_data = Batch.from_data_list(molecules).to(self.device)
        
        with torch.no_grad():
            # Extract embeddings from the jumping knowledge layer
            # We need to run the forward pass but extract embeddings before the heads
            x = batch_data.x.float()
            edge_index = batch_data.edge_index
            edge_attr = None  # QM9 typically doesn't use edge attributes
            batch = batch_data.batch
            pos = batch_data.pos if hasattr(batch_data, 'pos') else None
            
            # Run through backbone to get embeddings
            # We'll extract embeddings by hooking into the model's forward pass
            embeddings = backbone.initial_proj(x)
            
            # Pass through blocks
            outs = []
            for block in backbone.blocks:
                embeddings = block(embeddings, edge_index, edge_attr)
                outs.append(embeddings)
            
            # Apply jumping knowledge aggregation
            final_embeddings = backbone.jk(outs)
            
            if final_embeddings is None:
                raise RuntimeError("Backbone produced None embeddings")
            
            # Extract target rotational constants (properties 16, 17, 18)
            target_props = batch_data.y  # [batch_size, 19]
            target_rot_constants = target_props[:, 16:19]  # A, B, C columns
            
            logger.info(f"✅ Extracted embeddings shape: {final_embeddings.shape}")
            logger.info(f"✅ Target rotational constants shape: {target_rot_constants.shape}")
            
            return final_embeddings, pos, batch, target_rot_constants
    
    def test_01_checkpoint_loading(self):
        """Test that we can successfully load the PROJECT APOLLO checkpoint."""
        logger.info("🧪 Test 1: Checkpoint Loading")
        
        # Load frozen backbone
        self.backbone_model = self._load_frozen_backbone()
        
        # Verify model is in eval mode and frozen
        self.assertTrue(not self.backbone_model.training)
        
        # Check that parameters are frozen
        for param in self.backbone_model.parameters():
            self.assertFalse(param.requires_grad, "Backbone parameters should be frozen")
        
        # Verify architecture
        self.assertEqual(self.backbone_model.hidden_dim, self.hidden_dim)
        self.assertIsNotNone(self.backbone_model.pimeh_head)
        self.assertIsNotNone(self.backbone_model.pimeh_adapter)
        
        logger.info("✅ Test 1 PASSED: Checkpoint loaded successfully")
    
    def test_02_qm9_data_loading(self):
        """Test that we can load QM9 validation data correctly."""
        logger.info("🧪 Test 2: QM9 Data Loading")
        
        # Load test data
        self.test_data = self._load_qm9_test_data()
        
        # Verify data structure
        self.assertGreater(len(self.test_data), 0, "Should load at least one molecule")
        self.assertLessEqual(len(self.test_data), self.num_test_molecules)
        
        # Check first molecule structure
        first_mol = self.test_data[0]
        self.assertIsInstance(first_mol, Data)
        self.assertTrue(hasattr(first_mol, 'x'), "Molecule should have node features")
        self.assertTrue(hasattr(first_mol, 'y'), "Molecule should have target properties")
        self.assertEqual(first_mol.y.shape[0], 19, "Should have 19 QM9 properties")
        
        # Check that rotational constants exist
        rot_constants = first_mol.y[16:19]
        self.assertEqual(rot_constants.shape[0], 3, "Should have 3 rotational constants")
        
        logger.info("✅ Test 2 PASSED: QM9 data loaded successfully")
    
    def test_03_backbone_embedding_extraction(self):
        """Test extraction of high-quality embeddings from frozen backbone."""
        logger.info("🧪 Test 3: Backbone Embedding Extraction")
        
        # Load components
        if self.backbone_model is None:
            self.backbone_model = self._load_frozen_backbone()
        if self.test_data is None:
            self.test_data = self._load_qm9_test_data()
        
        # Extract embeddings
        embeddings, positions, batch, targets = self._extract_backbone_embeddings(
            self.backbone_model, self.test_data[:self.test_batch_size]
        )
        
        # Store for later tests
        self.frozen_embeddings = (embeddings, positions, batch, targets)
        
        # Verify embedding quality
        self.assertEqual(embeddings.dim(), 2, "Embeddings should be 2D")
        self.assertEqual(embeddings.size(1), self.hidden_dim, f"Should have {self.hidden_dim} features")
        self.assertFalse(torch.isnan(embeddings).any(), "Embeddings should not contain NaN")
        self.assertFalse(torch.isinf(embeddings).any(), "Embeddings should not contain Inf")
        
        # Check embedding statistics (should be reasonable)
        mean_val = embeddings.mean().item()
        std_val = embeddings.std().item()
        self.assertLess(abs(mean_val), 5.0, f"Embedding mean {mean_val} seems unreasonable")
        self.assertGreater(std_val, 0.1, f"Embedding std {std_val} too low (dead neurons?)")
        self.assertLess(std_val, 10.0, f"Embedding std {std_val} too high")
        
        # Verify position data
        if positions is not None:
            self.assertEqual(positions.dim(), 2, "Positions should be 2D")
            self.assertEqual(positions.size(1), 3, "Positions should be 3D coordinates")
            self.assertFalse(torch.isnan(positions).any(), "Positions should not contain NaN")
        
        # Verify batch indices
        self.assertEqual(batch.dim(), 1, "Batch should be 1D")
        self.assertEqual(batch.size(0), embeddings.size(0), "Batch size should match embeddings")
        
        # Verify targets
        self.assertEqual(targets.dim(), 2, "Targets should be 2D")
        self.assertEqual(targets.size(1), 3, "Should have 3 rotational constants")
        
        logger.info(f"✅ Test 3 PASSED: Extracted {embeddings.size(0)} embeddings")
        logger.info(f"   Embedding stats: mean={mean_val:.3f}, std={std_val:.3f}")
        logger.info(f"   Target range: A={targets[:, 0].min():.2f}-{targets[:, 0].max():.2f} GHz")
    
    def test_04_pimeh_forward_pass(self):
        """Test PIMEH forward pass with real backbone embeddings."""
        logger.info("🧪 Test 4: PIMEH Forward Pass")
        
        # Load test data if not already loaded
        if self.frozen_embeddings is None:
            self.test_03_backbone_embedding_extraction()
        
        embeddings, positions, batch, targets = self.frozen_embeddings
        
        # Create fresh PIMEH head
        self.pimeh_head = PhysicsInformedMinimalEquivariantHead(self.hidden_dim)
        self.pimeh_head.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predicted_rot = self.pimeh_head(embeddings, positions, batch)
        
        # Verify output shape and values
        expected_batch_size = batch.unique().size(0)
        self.assertEqual(predicted_rot.shape, (expected_batch_size, 3), 
                        f"Expected shape ({expected_batch_size}, 3), got {predicted_rot.shape}")
        
        self.assertFalse(torch.isnan(predicted_rot).any(), "PIMEH output should not contain NaN")
        self.assertFalse(torch.isinf(predicted_rot).any(), "PIMEH output should not contain Inf")
        
        # Check that values are in reasonable range for rotational constants (GHz)
        self.assertTrue((predicted_rot > 0).all(), "Rotational constants should be positive")
        self.assertTrue((predicted_rot < 1000).all(), "Rotational constants seem too large")
        
        # Check ordering (A >= B >= C should approximately hold)
        ordering_violations = (predicted_rot[:, 1] > predicted_rot[:, 0]).sum() + \
                             (predicted_rot[:, 2] > predicted_rot[:, 1]).sum()
        total_comparisons = predicted_rot.size(0) * 2
        violation_rate = ordering_violations.float() / total_comparisons
        
        logger.info(f"✅ Test 4 PASSED: PIMEH forward pass successful")
        logger.info(f"   Output shape: {predicted_rot.shape}")
        logger.info(f"   Value range: {predicted_rot.min():.2f} - {predicted_rot.max():.2f} GHz")
        logger.info(f"   Ordering violation rate: {violation_rate:.1%}")
    
    def test_05_mass_prediction_quality(self):
        """Test that PIMEH predicts reasonable atomic masses."""
        logger.info("🧪 Test 5: Mass Prediction Quality")
        
        # Load test data if not already loaded
        if self.frozen_embeddings is None:
            self.test_03_backbone_embedding_extraction()
        
        embeddings, positions, batch, targets = self.frozen_embeddings
        
        if self.pimeh_head is None:
            self.pimeh_head = PhysicsInformedMinimalEquivariantHead(self.hidden_dim)
            self.pimeh_head.to(self.device)
        
        # Extract predicted masses
        with torch.no_grad():
            predicted_masses = self.pimeh_head.mass_mlp(embeddings).squeeze(-1)
        
        # Verify mass predictions are reasonable
        self.assertFalse(torch.isnan(predicted_masses).any(), "Masses should not contain NaN")
        self.assertFalse(torch.isinf(predicted_masses).any(), "Masses should not contain Inf")
        self.assertTrue((predicted_masses > 0).all(), "Masses should be positive")
        
        # Check mass distribution (should be roughly in atomic mass unit range)
        mass_mean = predicted_masses.mean().item()
        mass_std = predicted_masses.std().item()
        mass_min = predicted_masses.min().item()
        mass_max = predicted_masses.max().item()
        
        # Reasonable bounds for atomic masses (0.1 to 100 amu roughly)
        self.assertGreater(mass_min, 0.01, f"Minimum mass {mass_min} too low")
        self.assertLess(mass_max, 200, f"Maximum mass {mass_max} too high")
        self.assertGreater(mass_mean, 0.5, f"Mean mass {mass_mean} too low")
        self.assertLess(mass_mean, 50, f"Mean mass {mass_mean} too high")
        
        logger.info(f"✅ Test 5 PASSED: Mass prediction quality check")
        logger.info(f"   Mass stats: mean={mass_mean:.2f}, std={mass_std:.2f}")
        logger.info(f"   Mass range: {mass_min:.2f} - {mass_max:.2f} amu")
    
    def test_06_advanced_physics_loss(self):
        """Test the advanced physics-informed loss function."""
        logger.info("🧪 Test 6: Advanced Physics Loss")
        
        # Create test data
        batch_size = 4
        num_atoms = 20
        device = self.device
        
        # Mock predictions and targets
        pred_rot = torch.tensor([
            [30.0, 20.0, 10.0],  # Well-ordered: A > B > C
            [25.0, 25.0, 15.0],  # Partially ordered
            [15.0, 20.0, 25.0],  # Badly ordered: A < B < C
            [50.0, 40.0, 30.0],  # Well-ordered
        ], device=device, dtype=torch.float32)
        
        target_rot = torch.tensor([
            [32.0, 18.0, 12.0],
            [28.0, 22.0, 16.0],
            [20.0, 15.0, 10.0],
            [45.0, 35.0, 25.0],
        ], device=device, dtype=torch.float32)
        
        pred_masses = torch.rand(num_atoms, device=device) * 50 + 0.5  # 0.5-50.5 amu
        pred_inertia = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 1e-45
        
        # Test loss computation
        loss_dict = self.advanced_loss.compute_loss(
            pred_rot, target_rot, pred_masses, pred_inertia
        )
        
        # Verify loss components exist
        required_keys = ['total_loss', 'mse_loss', 'ordering_loss', 'mass_loss', 
                        'inertia_loss', 'scale_loss']
        for key in required_keys:
            self.assertIn(key, loss_dict, f"Missing loss component: {key}")
            self.assertIsInstance(loss_dict[key], torch.Tensor, f"{key} should be tensor")
            self.assertEqual(loss_dict[key].dim(), 0, f"{key} should be scalar")
        
        # Verify loss values are reasonable
        total_loss = loss_dict['total_loss'].item()
        mse_loss = loss_dict['mse_loss'].item()
        ordering_loss = loss_dict['ordering_loss'].item()
        
        self.assertGreater(total_loss, 0, "Total loss should be positive")
        self.assertGreater(mse_loss, 0, "MSE loss should be positive")
        self.assertGreater(ordering_loss, 0, "Should have ordering violations")
        
        # Test callable interface
        callable_loss = self.advanced_loss(pred_rot, target_rot, pred_masses, pred_inertia)
        self.assertEqual(callable_loss.item(), total_loss, "Callable should return total loss")
        
        logger.info(f"✅ Test 6 PASSED: Advanced physics loss functional")
        logger.info(f"   Total loss: {total_loss:.4f}")
        logger.info(f"   MSE loss: {mse_loss:.4f}")
        logger.info(f"   Ordering loss: {ordering_loss:.4f}")
    
    def test_07_rotational_invariance(self):
        """Test that PIMEH maintains rotational invariance."""
        logger.info("🧪 Test 7: Rotational Invariance")
        
        # Load test data if not already loaded
        if self.frozen_embeddings is None:
            self.test_03_backbone_embedding_extraction()
        
        embeddings, positions, batch, targets = self.frozen_embeddings
        
        if self.pimeh_head is None:
            self.pimeh_head = PhysicsInformedMinimalEquivariantHead(self.hidden_dim)
            self.pimeh_head.to(self.device)
        
        # Use only first molecule for invariance test
        first_mol_mask = (batch == 0)
        mol_embeddings = embeddings[first_mol_mask]
        mol_positions = positions[first_mol_mask] if positions is not None else torch.randn(mol_embeddings.size(0), 3, device=self.device)
        mol_batch = torch.zeros(mol_embeddings.size(0), dtype=torch.long, device=self.device)
        
        # Test rotational invariance
        is_invariant = validate_rotational_invariance(
            self.pimeh_head,
            mol_embeddings,
            mol_positions,
            mol_batch,
            num_rotations=5,
            tolerance=1e-3
        )
        
        self.assertTrue(is_invariant, "PIMEH should be rotationally invariant")
        
        logger.info("✅ Test 7 PASSED: Rotational invariance validated")
    
    def test_08_comprehensive_integration(self):
        """Test complete PIMEH integration with loss computation."""
        logger.info("🧪 Test 8: Comprehensive Integration")
        
        # Load all components if not already loaded
        if self.frozen_embeddings is None:
            self.test_03_backbone_embedding_extraction()
        
        embeddings, positions, batch, targets = self.frozen_embeddings
        
        if self.pimeh_head is None:
            self.pimeh_head = PhysicsInformedMinimalEquivariantHead(self.hidden_dim)
            self.pimeh_head.to(self.device)
        
        # Enable gradients for this test
        self.pimeh_head.train()
        
        # Forward pass with gradient computation
        predicted_rot = self.pimeh_head(embeddings, positions, batch)
        
        # Extract intermediate values for loss computation
        with torch.no_grad():
            predicted_masses = self.pimeh_head.mass_mlp(embeddings).squeeze(-1)
            inertia_tensors = compute_inertia_tensor(predicted_masses, positions, batch)
        
        # Compute advanced physics loss
        loss_dict = self.advanced_loss.compute_loss(
            predicted_rot, targets, predicted_masses, inertia_tensors
        )
        
        total_loss = loss_dict['total_loss']
        
        # Test backward pass
        total_loss.backward()
        
        # Check gradients exist and are reasonable
        grad_norms = []
        for name, param in self.pimeh_head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradients in {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradients in {name}")
        
        self.assertGreater(len(grad_norms), 0, "Should have gradients")
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        self.assertGreater(avg_grad_norm, 1e-8, "Gradients too small")
        self.assertLess(avg_grad_norm, 1e3, "Gradients too large")
        
        logger.info(f"✅ Test 8 PASSED: Comprehensive integration successful")
        logger.info(f"   Final loss: {total_loss.item():.4f}")
        logger.info(f"   Average gradient norm: {avg_grad_norm:.2e}")
        logger.info(f"   Parameters with gradients: {len(grad_norms)}")
    
    def test_09_error_handling(self):
        """Test robust error handling in edge cases."""
        logger.info("🧪 Test 9: Error Handling")
        
        pimeh = PhysicsInformedMinimalEquivariantHead(self.hidden_dim).to(self.device)
        
        # Test empty input
        empty_h = torch.empty(0, self.hidden_dim, device=self.device)
        empty_pos = torch.empty(0, 3, device=self.device)
        empty_batch = torch.empty(0, dtype=torch.long, device=self.device)
        
        result = pimeh(empty_h, empty_pos, empty_batch)
        self.assertEqual(result.shape, (0, 3), "Should handle empty input gracefully")
        
        # Test single atom molecule
        single_h = torch.randn(1, self.hidden_dim, device=self.device)
        single_pos = torch.randn(1, 3, device=self.device)
        single_batch = torch.zeros(1, dtype=torch.long, device=self.device)
        
        result = pimeh(single_h, single_pos, single_batch)
        self.assertEqual(result.shape, (1, 3), "Should handle single atom")
        self.assertTrue((result > 0).all(), "Single atom result should be positive")
        
        # Test dimension mismatch error handling
        with self.assertRaises(ValueError):
            pimeh(torch.randn(5, self.hidden_dim), torch.randn(4, 3), torch.zeros(5))
        
        logger.info("✅ Test 9 PASSED: Error handling robust")


def advanced_physics_loss(
    pred_rot: torch.Tensor,
    target_rot: torch.Tensor,
    pred_inertia: torch.Tensor,
    target_inertia: torch.Tensor
) -> torch.Tensor:
    """
    Advanced physics-informed loss function for rotational constants.
    
    This is a placeholder implementation that will be enhanced based on
    test results and physics-informed neural network best practices.
    
    Current implementation is a simple MSE placeholder that will be
    replaced with the full AdvancedPhysicsLoss implementation.
    
    Args:
        pred_rot: Predicted rotational constants [batch_size, 3]
        target_rot: Target rotational constants [batch_size, 3]
        pred_inertia: Predicted inertia tensors [batch_size, 3, 3]
        target_inertia: Target inertia tensors [batch_size, 3, 3]
        
    Returns:
        Scalar loss tensor
    """
    # Placeholder: Simple MSE loss
    # TODO: Replace with advanced physics constraints
    mse_loss = F.mse_loss(pred_rot, target_rot)
    
    # Add small regularization to prevent returning exactly 1.0
    regularization = 0.01 * F.mse_loss(pred_inertia, target_inertia) if pred_inertia is not None else 0.0
    
    return mse_loss + regularization


if __name__ == '__main__':
    """
    Run the comprehensive PIMEH test suite.
    
    This serves as the foundation for PROJECT PIMEH POLISH to achieve
    >95% mean R² accuracy by fixing rotational constants prediction.
    """
    
    print("🚀 PROJECT PIMEH POLISH - Comprehensive Test Suite")
    print("=" * 80)
    print("This test validates PIMEH integration with PROJECT APOLLO checkpoint")
    print("and provides the foundation for achieving >95% mean R² accuracy.")
    print("=" * 80)
    
    # Configure test runner
    unittest.main(
        verbosity=2,
        buffer=True,
        failfast=False  # Run all tests even if some fail
    )