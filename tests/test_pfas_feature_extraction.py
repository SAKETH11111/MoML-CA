#!/usr/bin/env python3
"""
Test script for PFAS-specific features and visualization

This script tests:
1. Creating graphs with PFAS-specific features
2. Visualizing molecular graphs with different highlighting options
3. Calculating and verifying PFAS-specific properties
"""

import os
import sys
import unittest
import tempfile
import logging
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_pfas_features")

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

# Try to import torch and matplotlib (for visualization)
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch import successful!")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Check installation.")
    sys.exit(1)

# Try to import matplotlib for visualization testing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib import successful!")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not found. Check installation.")
    sys.exit(1)

# Import from consolidated moml modules
try:
    from moml.core import calculate_molecular_descriptors
    from moml.utils import validate_smiles
    from moml.utils import (
        create_rdkit_mols,
        categorize_molecular_features as categorize_pfas_types, # Alias to match test usage
        # The following might need to be sourced from moml.utils as well, or are part of categorize_molecular_features
        # calculate_pfas_statistics,
        # identify_fluorinated_groups
    )
    # Attempting to import these directly to see if they exist in moml.utils or if tests need update
    from moml.utils import calculate_molecular_complexity as calculate_pfas_statistics # Assuming calculate_molecular_complexity is the new name
    from moml.utils import extract_fluorine_count # identify_fluorinated_groups might be related to this or part of categorize_molecular_features
    from moml.core import MolecularGraphProcessor
    from moml.utils import visualize_molecular_graph
    
    IMPORTS_SUCCESSFUL = True
    print("Successfully imported moml modules!")
except ImportError as e:
    print(f"Failed to import required moml modules: {e}")
    IMPORTS_SUCCESSFUL = False


class TestPFASFeatures(unittest.TestCase):
    """Test PFAS-specific features and visualization."""
    
    def setUp(self):
        """Set up test molecules."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Required MOML modules not available")
            
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a few example PFAS SMILES strings
        self.test_smiles = [
            "CC(F)(F)F",                              # Trifluoromethane (simple PFAS)
            "C(C(F)(F)F)C(F)(F)F",                    # Hexafluoroethane
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F",   # GenX (complex PFAS)
            "CC",                                     # Ethane (non-PFAS control)
        ]
        
        # Create a dataframe with the test SMILES
        self.test_df = pd.DataFrame({
            'smiles': self.test_smiles,
            'name': ['Trifluoromethane', 'Hexafluoroethane', 'GenX', 'Ethane'],
            'id': ['TEST-001', 'TEST-002', 'TEST-003', 'TEST-004']
        })
        
        # Convert SMILES to RDKit molecules
        self.test_df = create_rdkit_mols(self.test_df, smiles_col='smiles', mol_col='rdkit_mol')
        
        # Initialize graph processor
        self.graph_processor = MolecularGraphProcessor()
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
        
        # Close any open matplotlib figures
        if 'plt' in globals() and plt:
            plt.close('all')
    
    def test_pfas_detection(self):
        """Test detection of PFAS compounds based on fluorine count."""
        # Apply PFAS categorization
        test_df = categorize_pfas_types(self.test_df, mol_column='rdkit_mol')
        
        # The first three should be flagged as PFAS
        pfas_compounds = test_df[test_df['is_pfas'] == True]
        non_pfas_compounds = test_df[test_df['is_pfas'] == False]
        
        self.assertEqual(len(pfas_compounds), 3, "Expected 3 PFAS compounds")
        self.assertEqual(len(non_pfas_compounds), 1, "Expected 1 non-PFAS compound")
        
        # Check specific PFAS types
        self.assertEqual(test_df.iloc[0]['pfas_type'], 'CF3')  # Trifluoromethane has a CF3 group
        self.assertEqual(test_df.iloc[1]['pfas_type'], 'Multi-CF3')  # Hexafluoroethane has 2 CF3 groups
        
        # Expected fluorine counts
        expected_f_counts = [3, 6, 9, 0]
        for i, expected in enumerate(expected_f_counts):
            self.assertEqual(test_df.iloc[i]['num_fluorine'], expected, 
                            f"Expected {expected} fluorine atoms for {self.test_df.iloc[i]['name']}")
        
        print(f"Successfully identified {len(pfas_compounds)} PFAS compounds out of {len(test_df)}")
    
    def test_pfas_statistics(self):
        """Test calculation of PFAS statistics."""
        # Apply PFAS statistics
        test_df = categorize_pfas_types(self.test_df, mol_column='rdkit_mol')
        test_df = calculate_pfas_statistics(test_df, mol_column='rdkit_mol')
        
        # Check PFAS statistics columns
        required_cols = ['num_fluorine', 'num_carbon', 'f_to_c_ratio', 'avg_f_per_c']
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")
        
        # Check specific statistics
        # Trifluoromethane: 3F, 1C -> F/C ratio = 3.0
        self.assertEqual(test_df.iloc[0]['num_fluorine'], 3)
        self.assertEqual(test_df.iloc[0]['num_carbon'], 1)
        self.assertEqual(test_df.iloc[0]['f_to_c_ratio'], 3.0)
        
        # Hexafluoroethane: 6F, 2C -> F/C ratio = 3.0
        self.assertEqual(test_df.iloc[1]['num_fluorine'], 6)
        self.assertEqual(test_df.iloc[1]['num_carbon'], 2)
        self.assertEqual(test_df.iloc[1]['f_to_c_ratio'], 3.0)
        
        # Ethane: 0F, 2C -> F/C ratio = 0.0
        self.assertEqual(test_df.iloc[3]['num_fluorine'], 0)
        self.assertEqual(test_df.iloc[3]['num_carbon'], 2)
        self.assertEqual(test_df.iloc[3]['f_to_c_ratio'], 0.0)
        
        print("Successfully calculated PFAS statistics")
    
    def test_fluorinated_groups(self):
        """Test identification of fluorinated groups."""
        # Apply fluorinated group identification
        test_df = categorize_pfas_types(self.test_df, mol_column='rdkit_mol')
        test_df = identify_fluorinated_groups(test_df, mol_column='rdkit_mol')
        
        # Check group identification columns
        required_cols = ['fluorinated_groups', 'num_cf3_groups', 'num_cf2_groups', 'num_cf_groups']
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")
        
        # Trifluoromethane has 1 CF3 group
        self.assertEqual(test_df.iloc[0]['num_cf3_groups'], 1)
        self.assertEqual(test_df.iloc[0]['num_cf2_groups'], 0)
        self.assertEqual(test_df.iloc[0]['num_cf_groups'], 0)
        
        # Hexafluoroethane has 2 CF3 groups
        self.assertEqual(test_df.iloc[1]['num_cf3_groups'], 2)
        self.assertEqual(test_df.iloc[1]['num_cf2_groups'], 0)
        self.assertEqual(test_df.iloc[1]['num_cf_groups'], 0)
        
        # Ethane has no fluorinated groups
        self.assertEqual(test_df.iloc[3]['num_cf3_groups'], 0)
        self.assertEqual(test_df.iloc[3]['num_cf2_groups'], 0)
        self.assertEqual(test_df.iloc[3]['num_cf_groups'], 0)
        
        print("Successfully identified fluorinated groups")
    
    def test_graph_processing(self):
        """Test creating molecular graphs with PFAS-specific features."""
        # Process the molecules into graphs
        for i, row in self.test_df.iterrows():
            mol = row['rdkit_mol']
            if mol is not None:
                # Generate atom features
                atom_features = self.graph_processor.get_atom_features(mol)
                self.assertIsNotNone(atom_features)
                self.assertEqual(atom_features.shape[0], mol.GetNumAtoms())
                
                # Generate adjacency matrix
                adjacency_matrix = self.graph_processor.get_adjacency_matrix(mol)
                self.assertIsNotNone(adjacency_matrix)
                self.assertEqual(adjacency_matrix.shape, (mol.GetNumAtoms(), mol.GetNumAtoms()))
        
        # Process the full dataframe
        processed_df = self.graph_processor.process_dataframe(self.test_df, mol_column='rdkit_mol')
        
        # Check that we have the necessary columns
        required_cols = ["atom_features", "adjacency_matrix", "num_atoms"]
        has_cols = all(col in processed_df.columns for col in required_cols)
        self.assertTrue(has_cols, f"Missing required columns in processed dataframe")
        
        print(f"Successfully processed {len(processed_df)} molecules into graphs")
    
    @unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib not available")
    def test_visualization(self):
        """Test visualization of molecules with PFAS highlighting."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("Matplotlib not available")
        
        # Test visualization of each molecule
        for i, row in self.test_df.iterrows():
            mol = row['rdkit_mol']
            if mol is not None:
                # Create a visualization
                fig = visualize_molecular_graph(mol, highlight_feature='fluorine')
                self.assertIsNotNone(fig)
                
                # Save the visualization to a file
                output_file = os.path.join(self.temp_dir.name, f"mol_{row['id']}.png")
                plt.savefig(output_file)
                plt.close(fig)
                
                # Check that the file was created
                self.assertTrue(os.path.exists(output_file))
        
        print(f"Successfully created visualizations for {len(self.test_df)} molecules")


def run_pfas_feature_tests():
    """Run the PFAS feature tests."""
    print("\nTesting PFAS-specific features...")
    
    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPFASFeatures))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_pfas_feature_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 