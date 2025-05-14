#!python
"""
Test script for the Unified Graph Generator

This script tests the functionality of the molecular graph generation utilities:
1. Creating molecular graphs from SMILES strings
2. Creating graphs from molecules with 3D coordinates
3. Testing batch processing capabilities
"""

import os
import sys
import unittest
import tempfile
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_graph_generation")

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

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
    import numpy as np
    import pandas as pd
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
    print("PyTorch and PyTorch Geometric import successful!")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Failed to import PyTorch or PyTorch Geometric: {e}")

# Import from consolidated moml modules
try:
    from moml.utils import validate_smiles
    from moml.data.processors.process_chemical_data import create_rdkit_mols
    from moml.core import MolecularGraphProcessor
    
    IMPORTS_SUCCESSFUL = True
    print("Successfully imported moml modules!")
except ImportError as e:
    print(f"Failed to import required moml modules: {e}")
    IMPORTS_SUCCESSFUL = False


class TestMolecularGraphGenerator(unittest.TestCase):
    """Test the molecular graph generation functionality."""
    
    def setUp(self):
        """Set up test molecules."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("Required dependencies not available")
            
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a few example SMILES strings
        self.test_smiles = [
            "CC",  # Ethane (simple hydrocarbon)
            "CC(F)(F)F",  # Trifluoromethane (simple PFAS)
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F"  # GenX (complex PFAS)
        ]
        
        # Convert to RDKit molecules and generate 3D coordinates
        self.mols = []
        self.mol_files = []
        
        for i, smiles in enumerate(self.test_smiles):
            is_valid, canonical_smi, error_msg = validate_smiles(smiles) # Corrected unpacking
            if is_valid:
                mol = Chem.MolFromSmiles(canonical_smi) # Create mol from canonical SMILES
                if mol is not None: # Further check if mol creation was successful
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    
                    self.mols.append(mol)
                    
                    # Save as MOL file
                    mol_file = os.path.join(self.temp_dir.name, f"test_mol_{i}.mol")
                    Chem.MolToMolFile(mol, mol_file)
                    self.mol_files.append(mol_file)
                else:
                    logger.warning(f"Could not create RDKit mol from canonical SMILES '{canonical_smi}' for original '{smiles}'")
            else:
                logger.warning(f"SMILES validation failed for '{smiles}': {error_msg}")
        
        # Initialize MolecularGraphProcessor
        self.graph_processor = MolecularGraphProcessor()
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def test_mol_to_graph_basic(self):
        """Test basic conversion of molecule to graph."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Test with ethane (simplest case)
        mol = self.mols[0]
        
        # Get atom features and adjacency matrix
        atom_features = self.graph_processor.get_atom_features(mol)
        adjacency_matrix = self.graph_processor.get_adjacency_matrix(mol)
        
        # Check basic properties
        self.assertEqual(atom_features.shape[0], mol.GetNumAtoms())
        self.assertEqual(adjacency_matrix.shape, (mol.GetNumAtoms(), mol.GetNumAtoms()))
        
        # Check feature dimensions
        expected_feature_dim = self.graph_processor.atom_feature_dim
        self.assertEqual(atom_features.shape[1], expected_feature_dim)
        
        print(f"Basic graph conversion test passed for ethane")
    
    def test_mol_to_graph_with_trifluoromethane(self):
        """Test conversion with trifluoromethane molecule."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Test with TFM (trifluoromethane)
        mol = self.mols[1]
        num_atoms = mol.GetNumAtoms()
        
        # Get atom features and adjacency matrix
        atom_features = self.graph_processor.get_atom_features(mol)
        adjacency_matrix = self.graph_processor.get_adjacency_matrix(mol)
        
        # Check basic properties
        self.assertEqual(atom_features.shape[0], num_atoms)
        self.assertEqual(adjacency_matrix.shape, (num_atoms, num_atoms))
        
        # Check that adjacency matrix is symmetric
        self.assertTrue(np.array_equal(adjacency_matrix, adjacency_matrix.T))
        
        print(f"Graph conversion passed for trifluoromethane")
    
    def test_mol_from_file_to_graph(self):
        """Test loading molecule from file and converting to graph."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Use the first mol file (ethane)
        mol_file = self.mol_files[0]
        self.assertTrue(os.path.exists(mol_file), f"Mol file not found: {mol_file}")
        
        # Load the molecule from file
        mol = Chem.MolFromMolFile(mol_file)
        self.assertIsNotNone(mol, "Failed to load molecule from MOL file")
        
        # Get atom features and adjacency matrix
        atom_features = self.graph_processor.get_atom_features(mol)
        adjacency_matrix = self.graph_processor.get_adjacency_matrix(mol)
        
        # Check basic properties
        self.assertEqual(atom_features.shape[0], mol.GetNumAtoms())
        self.assertEqual(adjacency_matrix.shape, (mol.GetNumAtoms(), mol.GetNumAtoms()))
        
        print(f"Successfully created graph from mol file")
    
    def test_process_dataframe(self):
        """Test processing a dataframe of molecules."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Create a dataframe with SMILES and convert to RDKit molecules
        df = pd.DataFrame({
            'smiles': self.test_smiles,
            'name': ['Ethane', 'Trifluoromethane', 'GenX'],
            'id': ['TEST-001', 'TEST-002', 'TEST-003']
        })
        
        # Convert SMILES to RDKit molecules
        df = create_rdkit_mols(df, smiles_col='smiles', mol_col='rdkit_mol')
        
        # Process dataframe with the graph processor
        processed_df = self.graph_processor.process_dataframe(df, mol_column='rdkit_mol')
        
        # Check that the necessary columns exist
        required_cols = ["atom_features", "adjacency_matrix", "num_atoms"]
        has_cols = all(col in processed_df.columns for col in required_cols)
        self.assertTrue(has_cols, f"Missing columns in processed dataframe")
        
        # Check that all rows have atom features and adjacency matrices
        for idx, row in processed_df.iterrows():
            self.assertIsInstance(row["atom_features"], np.ndarray)
            self.assertIsInstance(row["adjacency_matrix"], np.ndarray)
            self.assertEqual(row["atom_features"].shape[0], row["num_atoms"])
            self.assertEqual(row["adjacency_matrix"].shape, (row["num_atoms"], row["num_atoms"]))
        
        print(f"Successfully processed dataframe with {len(processed_df)} molecules")
        
    def test_save_processed_data(self):
        """Test saving processed molecular graph data."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Create a dataframe with SMILES and convert to RDKit molecules
        df = pd.DataFrame({
            'smiles': self.test_smiles,
            'name': ['Ethane', 'Trifluoromethane', 'GenX'],
            'id': ['TEST-001', 'TEST-002', 'TEST-003']
        })
        
        # Convert SMILES to RDKit molecules
        df = create_rdkit_mols(df, smiles_col='smiles', mol_col='rdkit_mol')
        
        # Process dataframe with the graph processor
        processed_df = self.graph_processor.process_dataframe(df, mol_column='rdkit_mol')
        
        # Save processed data
        output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the data (without atom_features and adjacency_matrix which don't serialize well)
        save_df = processed_df.copy()
        save_df.drop(["atom_features", "adjacency_matrix"], axis=1, inplace=True)
        output_file = os.path.join(output_dir, "molecular_graphs.csv")
        save_df.to_csv(output_file, index=False)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_file), f"Output file not created: {output_file}")
        
        print(f"Successfully saved processed data to {output_file}")


def run_graph_generation_tests():
    """Run all graph generation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMolecularGraphGenerator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("RDKit import successful!")
    
    if TORCH_AVAILABLE:
        print("PyTorch and PyTorch Geometric import successful!")
    else:
        print("PyTorch or PyTorch Geometric not available")
    
    if IMPORTS_SUCCESSFUL:
        print("Successfully imported moml modules!")
    else:
        print("Failed to import required modules")
    
    # Run the tests
    success = run_graph_generation_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 
