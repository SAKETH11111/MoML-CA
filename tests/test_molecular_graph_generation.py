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
        
        # Convert molecule to graph
        graph = self.graph_processor.mol_to_graph(mol)
        self.assertIsInstance(graph, PyGData)
        
        # Check basic properties
        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        self.assertTrue(hasattr(graph, 'x')) # Node features
        self.assertTrue(hasattr(graph, 'edge_index'))
        
        # Check feature dimensions
        self.assertEqual(graph.x.shape[0], mol.GetNumAtoms())
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)
        
        # Check edge_index (basic check, assumes undirected edges are added by mol_to_graph)
        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertGreaterEqual(graph.edge_index.shape[1], mol.GetNumBonds()) # Can be 2*num_bonds for undirected
        
        if mol.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, 'edge_attr'))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])

        print(f"Basic graph conversion test passed for ethane")
    
    def test_mol_to_graph_with_trifluoromethane(self):
        """Test conversion with trifluoromethane molecule."""
        if not TORCH_AVAILABLE or not IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")
        
        # Test with TFM (trifluoromethane)
        mol = self.mols[1]
        num_atoms = mol.GetNumAtoms()
        
        # Convert molecule to graph
        graph = self.graph_processor.mol_to_graph(mol)
        self.assertIsInstance(graph, PyGData)
        
        # Check basic properties
        self.assertEqual(graph.num_nodes, num_atoms)
        self.assertTrue(hasattr(graph, 'x'))
        self.assertEqual(graph.x.shape[0], num_atoms)
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)
        
        self.assertTrue(hasattr(graph, 'edge_index'))
        self.assertEqual(graph.edge_index.shape[0], 2)
        
        if mol.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, 'edge_attr'))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])
            
        # Check for 3D coordinates if expected by processor config
        if self.graph_processor.use_3d_coords:
            self.assertTrue(hasattr(graph, 'pos'))
            self.assertEqual(graph.pos.shape, (num_atoms, 3))
            
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
        
        # Convert molecule to graph using the processor's file_to_graph method
        graph = self.graph_processor.file_to_graph(mol_file)
        self.assertIsInstance(graph, PyGData)
        
        # Check basic properties
        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        self.assertTrue(hasattr(graph, 'x'))
        self.assertEqual(graph.x.shape[0], mol.GetNumAtoms())
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)
        
        self.assertTrue(hasattr(graph, 'edge_index'))
        if mol.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, 'edge_attr'))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])

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
        
        # Simulate processing dataframe row by row
        processed_graphs = []
        num_atoms_list = []
        for idx, row in df.iterrows():
            mol = row['rdkit_mol']
            if mol:
                graph = self.graph_processor.mol_to_graph(mol)
                if graph:
                    processed_graphs.append(graph)
                    num_atoms_list.append(mol.GetNumAtoms())
            else: # Handle cases where mol might be None if SMILES was invalid
                processed_graphs.append(None)
                num_atoms_list.append(0)

        self.assertEqual(len(processed_graphs), len(df))
        
        # Check that all valid rows have graph objects
        for i, graph_obj in enumerate(processed_graphs):
            original_mol = df.iloc[i]['rdkit_mol']
            if original_mol: # Only check if original mol was valid
                self.assertIsInstance(graph_obj, PyGData)
                self.assertEqual(graph_obj.num_nodes, num_atoms_list[i])
                self.assertTrue(hasattr(graph_obj, 'x'))
                self.assertEqual(graph_obj.x.shape[0], num_atoms_list[i])
                self.assertEqual(graph_obj.x.shape[1], self.graph_processor.atom_feature_dim)
            else:
                self.assertIsNone(graph_obj) # Expect None if original mol was None
        
        print(f"Successfully processed dataframe with {len(df[df['rdkit_mol'].notna()])} valid molecules into graphs")
        
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
        
        # Simulate processing dataframe row by row and collecting graph properties
        processed_graphs_data = []
        for idx, row in df.iterrows():
            mol = row['rdkit_mol']
            if mol:
                graph = self.graph_processor.mol_to_graph(mol)
                if graph:
                    processed_graphs_data.append({
                        'id': row['id'],
                        'name': row['name'],
                        'smiles': row['smiles'],
                        'num_nodes': graph.num_nodes,
                        'num_edges': graph.num_edges,
                        # Storing paths to individual .pt files would be another option
                    })
        
        # Create a DataFrame from the collected graph properties
        save_df = pd.DataFrame(processed_graphs_data)

        # Save processed data (now a DataFrame of graph properties)
        output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "molecular_graph_properties.csv")
        save_df.to_csv(output_file, index=False)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_file), f"Output file not created: {output_file}")
        
        # Verify content (optional, basic check)
        loaded_df = pd.read_csv(output_file)
        self.assertEqual(len(loaded_df), len(df[df['rdkit_mol'].notna()]))
        self.assertIn('num_nodes', loaded_df.columns)
        self.assertIn('id', loaded_df.columns)

        print(f"Successfully saved processed graph properties to {output_file}")


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
