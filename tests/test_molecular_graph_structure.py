"""
Tests for the MolecularGraphProcessor class.

This file originally tested the deprecated MolecularGraphBuilder class
and has been updated to use the modern MolecularGraphProcessor.
"""

import os
import sys
import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from moml.core import MolecularGraphProcessor, create_graph_processor


def create_test_molecule():
    """Create a simple test molecule (PFOA - Perfluorooctanoic acid)."""
    mol = Chem.MolFromSmiles("C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F")
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol


class TestMolecularGraphProcessor:
    """Test the MolecularGraphProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization with different options."""
        # Default initialization
        processor = create_graph_processor()
        assert processor.use_partial_charges is True
        assert processor.use_3d_coords is True
        assert processor.use_pfas_specific_features is True
        
        # Custom initialization
        processor = create_graph_processor({
            'use_partial_charges': False,
            'use_3d_coords': False,
            'use_pfas_specific_features': False
        })
        assert processor.use_partial_charges is False
        assert processor.use_3d_coords is False
        assert processor.use_pfas_specific_features is False
    
    def test_one_hot_encoding(self):
        """Test the one_hot_encoding function."""
        processor = create_graph_processor()
        
        # Test with value in choices
        encoding = processor._one_hot_encoding(1, [0, 1, 2])
        assert encoding == [0, 1, 0]
        
        # Test with value not in choices
        encoding = processor._one_hot_encoding(3, [0, 1, 2])
        assert encoding == [0, 0, 0]
    
    def test_functional_group_detection(self):
        """Test functional group detection methods."""
        processor = create_graph_processor()
        mol = create_test_molecule()
        
        # Find COOH group in PFOA
        found_carboxylic = False
        for atom in mol.GetAtoms():
            if processor._is_in_carboxylic_group(atom):
                found_carboxylic = True
                break
        
        assert found_carboxylic, "Carboxylic group not detected in PFOA"
        
        # Find CF3 groups
        cf3_groups = processor._find_cf3_groups(mol)
        assert len(cf3_groups) > 0, "CF3 groups not detected in PFOA"
    
    def test_mol_to_graph_basic(self):
        """Test basic graph conversion without partial charges."""
        processor = create_graph_processor({'use_partial_charges': False})
        mol = create_test_molecule()
        
        graph = processor.mol_to_graph(mol)
        
        # Check that the graph has the right structure
        assert isinstance(graph.x, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_attr, torch.Tensor)
        assert graph.num_nodes == mol.GetNumAtoms()
    
    def test_mol_to_graph_with_pfas_features(self):
        """Test graph conversion with PFAS-specific features."""
        processor = create_graph_processor({
            'use_partial_charges': False,
            'use_pfas_specific_features': True
        })
        mol = create_test_molecule()
        
        graph = processor.mol_to_graph(mol)
        
        # Check that the feature dimension is larger with PFAS features
        # Basic features + PFAS-specific features
        assert graph.x.shape[1] > 10
    
    def test_mol_to_graph_with_partial_charges(self):
        """Test graph conversion with partial charges."""
        processor = create_graph_processor({'use_partial_charges': True})
        mol = create_test_molecule()
        
        # Create mock partial charges
        num_atoms = mol.GetNumAtoms()
        partial_charges = np.random.uniform(-1, 1, num_atoms).tolist()
        
        graph = processor.mol_to_graph(mol, {'partial_charges': partial_charges})
        
        # Check that the feature dimension includes partial charges
        assert graph.x.shape[1] > 10
    
    def test_mol_to_graph_with_homo_lumo(self):
        """Test graph conversion with HOMO/LUMO contributions."""
        processor = create_graph_processor({'use_partial_charges': True})
        mol = create_test_molecule()
        
        # Create mock partial charges and HOMO/LUMO contributions
        num_atoms = mol.GetNumAtoms()
        partial_charges = np.random.uniform(-1, 1, num_atoms).tolist()
        
        # Mock HOMO/LUMO contributions
        homo_contributions = np.random.uniform(0, 0.2, num_atoms).tolist()
        lumo_contributions = np.random.uniform(0, 0.2, num_atoms).tolist()
        
        additional_features = {
            'partial_charges': partial_charges,
            'homo_contributions': homo_contributions,
            'lumo_contributions': lumo_contributions
        }
        
        graph = processor.mol_to_graph(mol, additional_features)
        
        # Check that the feature dimension includes HOMO/LUMO contributions
        # Basic features + PFAS features + partial charge + homo + lumo
        expected_dim = 15 + 2  # Base features + HOMO/LUMO
        assert graph.x.shape[1] >= expected_dim
    
    def test_global_features(self):
        """Test that global features are correctly calculated."""
        processor = create_graph_processor()
        mol = create_test_molecule()
        
        graph = processor.mol_to_graph(mol)
        
        # Check that global features exist and have the right shape
        assert hasattr(graph, 'y')
        assert graph.y is not None
        assert graph.y.shape[0] >= 7  # At least the basic global features
    
    def test_smiles_to_graph(self):
        """Test creating a graph from a SMILES string."""
        processor = create_graph_processor()
        
        # PFOA SMILES
        smiles = "C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F"
        
        graph = processor.smiles_to_graph(smiles)
        
        # Basic checks
        assert graph is not None
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[0] == 2  # [2, num_edges]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 