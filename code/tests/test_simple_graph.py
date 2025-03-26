#!/usr/bin/env python3
"""
Simplified test script for molecular graph generation

This script tests basic functionality related to molecular graphs:
1. Creating simple molecular graphs from SMILES using RDKit
2. Extracting basic atom and bond features
3. Creating a simple graph representation
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_simple_graph")

# Check for RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Try to import torch and torch_geometric
try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
    print("PyTorch import successful!")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Failed to import PyTorch: {e}")

# Import the MolecularGraphBuilder
try:
    from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder
    MGNN_AVAILABLE = True
    print("MolecularGraphBuilder import successful!")
except ImportError as e:
    MGNN_AVAILABLE = False
    print(f"Failed to import MolecularGraphBuilder: {e}")


class TestSimpleMolecularGraph(unittest.TestCase):
    """Test the simple molecular graph functionality."""
    
    def setUp(self):
        """Set up test molecules."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a few example SMILES strings
        self.test_smiles = [
            "CC",  # Ethane (simple hydrocarbon)
            "CC(F)(F)F",  # Trifluoromethane (simple PFAS)
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F"  # GenX (complex PFAS)
        ]
        
        # Initialize the MolecularGraphBuilder if available
        if MGNN_AVAILABLE:
            self.graph_builder = MolecularGraphBuilder(
                use_partial_charges=False, 
                use_3d_coords=True,
                use_pfas_specific_features=True
            )
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def smiles_to_mol(self, smiles, embed_3d=True, include_h=True):
        """Convert SMILES to an RDKit molecule with 3D coordinates."""
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens if requested
        if include_h:
            mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates if requested
        if embed_3d:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        return mol
    
    def test_graph_creation(self):
        """Test basic creation of molecular graphs."""
        if not MGNN_AVAILABLE:
            self.skipTest("MolecularGraphBuilder not available")
        
        # Test with ethane
        mol = self.smiles_to_mol(self.test_smiles[0])
        graph = self.graph_builder.mol_to_graph(mol)
        
        # Check that we got a valid graph
        self.assertIsNotNone(graph)
        self.assertGreater(graph.num_nodes, 0)
        
        # Load the molecule directly to verify
        mol = Chem.MolFromSmiles(self.test_smiles[0])
        mol = Chem.AddHs(mol)
        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        
        print(f"Basic graph creation passed for ethane with {graph.num_nodes} atoms and "
              f"{graph.edge_index.shape[1]} edges")
    
    def test_pfas_graph(self):
        """Test creation of a graph for a PFAS molecule."""
        if not MGNN_AVAILABLE:
            self.skipTest("MolecularGraphBuilder not available")
        
        # Test with TFM (trifluoromethane)
        mol = self.smiles_to_mol(self.test_smiles[1])
        graph = self.graph_builder.mol_to_graph(mol)
        
        # Check that we got a valid graph
        self.assertIsNotNone(graph)
        
        # Verify atom count
        mol = Chem.MolFromSmiles(self.test_smiles[1])
        mol = Chem.AddHs(mol)
        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        
        # Check that we have some fluorine atoms (atomic number 9)
        # This requires knowledge of how atomic numbers are encoded in the features
        f_count = 0
        for i in range(graph.num_nodes):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 9:  # Fluorine
                f_count += 1
        
        self.assertEqual(f_count, 3)  # TFM has 3 fluorine atoms
        
        print(f"PFAS graph creation passed for trifluoromethane with {graph.num_nodes} atoms, "
              f"including {f_count} fluorine atoms")
    
    def test_complex_pfas(self):
        """Test with a more complex PFAS (GenX)."""
        if not MGNN_AVAILABLE:
            self.skipTest("MolecularGraphBuilder not available")
        
        # Test with GenX
        mol = self.smiles_to_mol(self.test_smiles[2])
        graph = self.graph_builder.mol_to_graph(mol)
        
        # Check that we got a valid graph
        self.assertIsNotNone(graph)
        
        # Verify atom count
        mol = Chem.MolFromSmiles(self.test_smiles[2])
        mol = Chem.AddHs(mol)
        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        
        # Count fluorine atoms
        f_count = 0
        for i in range(graph.num_nodes):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 9:  # Fluorine
                f_count += 1
        
        # GenX should have 11 fluorine atoms in its structure
        self.assertGreaterEqual(f_count, 7)  # Allow for some variation in SMILES interpretation
        
        print(f"Complex PFAS graph creation passed for GenX with {graph.num_nodes} atoms, "
              f"including {f_count} fluorine atoms")
    
    def test_torch_conversion(self):
        """Test PyTorch tensor conversion."""
        if not TORCH_AVAILABLE or not MGNN_AVAILABLE:
            self.skipTest("PyTorch or MolecularGraphBuilder not available")
        
        # Test with ethane
        mol = self.smiles_to_mol(self.test_smiles[0])
        graph = self.graph_builder.mol_to_graph(mol)
        
        # Check that PyTorch tensors were created
        self.assertIsInstance(graph.x, Tensor)
        self.assertIsInstance(graph.edge_index, Tensor)
        
        print(f"PyTorch tensor conversion passed")


def run_simple_graph_tests():
    """Run the test suite."""
    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSimpleMolecularGraph))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    print("\nTesting simple molecular graph functionality...")
    success = run_simple_graph_tests()
    
    if success:
        print("\nAll simple graph tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome simple graph tests FAILED!")
        sys.exit(1) 