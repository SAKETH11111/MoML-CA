#!python
"""
Test script for PFAS-specific features and visualization

This script tests:
1. Creating graphs with PFAS-specific features
2. Visualizing molecular graphs with different highlighting options
3. Calculating and verifying PFAS-specific properties
"""

import os
import sys
import pytest
import unittest
import tempfile
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_pfas_features")

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Check for RDKit
try:
    from rdkit import Chem
    print("RDKit import successful!")
except ImportError:
    pytest.skip("RDKit not installed, skipping PFAS feature extraction tests", allow_module_level=True)

# Try to import torch and matplotlib (for visualization)
try:
    TORCH_AVAILABLE = True
    print("PyTorch import successful!")
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not found, skipping PFAS feature extraction tests", allow_module_level=True)

# Try to import matplotlib for visualization testing
try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib import successful!")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    pytest.skip("Matplotlib not found, skipping visualization tests", allow_module_level=True)

# Import from consolidated moml modules
try:
    from moml.core import FunctionalGroupDetector
    from moml.utils import (
        create_rdkit_mols,
        categorize_molecular_features as categorize_pfas_types,
    )

    from moml.utils import (
        calculate_molecular_complexity as calculate_pfas_statistics,
    )
    from moml.utils import add_fluorinated_group_counts
    from moml.core import MolecularGraphProcessor

    IMPORTS_SUCCESSFUL = True
    print("Successfully imported moml modules!")
except ImportError:
    print(f"Failed to import required moml modules")
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
            "CC(F)(F)F",  # Trifluoromethane (simple PFAS)
            "C(C(F)(F)F)C(F)(F)F",  # Hexafluoroethane
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F",  # GenX (complex PFAS)
            "CC",  # Ethane (non-PFAS control)
        ]

        # Create a dataframe with the test SMILES
        self.test_df = pd.DataFrame(
            {
                "smiles": self.test_smiles,
                "name": ["Trifluoromethane", "Hexafluoroethane", "GenX", "Ethane"],
                "id": ["TEST-001", "TEST-002", "TEST-003", "TEST-004"],
            }
        )

        # Convert SMILES to RDKit molecules
        self.test_df = create_rdkit_mols(self.test_df, smiles_col="smiles", mol_col="rdkit_mol")

        # Initialize graph processor
        self.graph_processor = MolecularGraphProcessor()

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

        # Close any open matplotlib figures
        if "plt" in globals() and plt:
            plt.close("all")

    def test_pfas_detection(self):
        """Test detection of PFAS compounds based on fluorine count."""
        # Apply PFAS statistics (which includes Chain_Length) first
        test_df = calculate_pfas_statistics(
            self.test_df.copy(), mol_col="rdkit_mol"
        )
        # Then apply PFAS categorization
        test_df = categorize_pfas_types(test_df, mol_col="rdkit_mol")
        # Also ensure fluorinated group counts are added if pfas_type depends on them
        test_df = add_fluorinated_group_counts(test_df, mol_col="rdkit_mol")

        # The first three should be flagged as PFAS (using Has_Fluorine as proxy)
        pfas_compounds = test_df[test_df["Has_Fluorine"]]
        non_pfas_compounds = test_df[~test_df["Has_Fluorine"]]

        self.assertEqual(len(pfas_compounds), 3, "Expected 3 PFAS compounds based on Has_Fluorine")
        self.assertEqual(len(non_pfas_compounds), 1, "Expected 1 non-PFAS compound based on Has_Fluorine")

        # Check specific PFAS types (simplified based on num_cf3_groups)
        # SMILES "CC(F)(F)F" (1,1,1-Trifluoroethane) has one CF3 group on the second carbon.
        # SMILES "C(C(F)(F)F)C(F)(F)F" (Perfluoropropane) has two terminal CF3 groups.
        if "num_cf3_groups" in test_df.columns:  # Check if column exists
            self.assertEqual(test_df.iloc[0]["num_cf3_groups"], 1, "1,1,1-Trifluoroethane should have 1 CF3 group")
            self.assertEqual(test_df.iloc[1]["num_cf3_groups"], 2, "Perfluoropropane should have 2 CF3 groups")

        # Expected fluorine counts (using F_Count column from calculate_pfas_statistics)
        # Based on SMILES:
        # "CC(F)(F)F" (1,1,1-Trifluoroethane, C2H3F3) -> 3F
        # "C(C(F)(F)F)C(F)(F)F" (1,1,1,3,3,3-Hexafluoropropane, C3H2F6) -> 6F
        # "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F" (GenX, C9HF17O3) -> 9F (Corrected from 17 based on manual count)
        # "CC" (Ethane, C2H6) -> 0F
        expected_f_counts = [3, 6, 9, 0]  # Corrected F_Count for GenX
        for i, expected in enumerate(expected_f_counts):
            self.assertEqual(
                test_df.iloc[i]["F_Count"],
                expected,
                f"Expected {expected} fluorine atoms for {self.test_df.iloc[i]['name']} (SMILES: {self.test_df.iloc[i]['smiles']})",
            )

        print(
            f"Successfully identified {len(pfas_compounds)} PFAS compounds (based on Has_Fluorine) out of {len(test_df)}"
        )

    def test_pfas_statistics(self):
        """Test calculation of PFAS statistics."""
        # Apply PFAS statistics (which includes Chain_Length) first
        test_df = calculate_pfas_statistics(self.test_df.copy(), mol_col="rdkit_mol")        # Then apply PFAS categorization (if its outputs are also checked, otherwise this might not be needed here)
        test_df = categorize_pfas_types(test_df, mol_col="rdkit_mol")

        # Check PFAS statistics columns (updated column names)
        required_cols = ["F_Count", "C_Count", "f_to_c_ratio", "avg_f_per_c"]
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")

        # Check specific statistics based on actual SMILES processing:
        # SMILES "CC(F)(F)F" (1,1,1-Trifluoroethane, C2H3F3): F=3, C=2
        self.assertEqual(test_df.iloc[0]["F_Count"], 3)
        self.assertEqual(test_df.iloc[0]["C_Count"], 2)
        self.assertAlmostEqual(test_df.iloc[0]["f_to_c_ratio"], 3.0 / 2.0)

        # SMILES "C(C(F)(F)F)C(F)(F)F" (1,1,1,3,3,3-Hexafluoropropane, C3H2F6): F=6, C=3
        self.assertEqual(test_df.iloc[1]["F_Count"], 6)  # Corrected F_Count
        self.assertEqual(test_df.iloc[1]["C_Count"], 3)  # Corrected C_Count
        self.assertAlmostEqual(test_df.iloc[1]["f_to_c_ratio"], 6.0 / 3.0)  # Corrected ratio

        # SMILES "CC" (Ethane, C2H6): F=0, C=2
        self.assertEqual(test_df.iloc[3]["F_Count"], 0)
        self.assertEqual(test_df.iloc[3]["C_Count"], 2)
        self.assertAlmostEqual(test_df.iloc[3]["f_to_c_ratio"], 0.0)

        print("Successfully calculated PFAS statistics")

    def test_fluorinated_groups(self):
        """Test identification of fluorinated groups."""
        # Apply PFAS statistics (which includes Chain_Length) first
        test_df = calculate_pfas_statistics(self.test_df.copy(), mol_col="rdkit_mol")        # Then apply PFAS categorization
        test_df = categorize_pfas_types(test_df, mol_col="rdkit_mol")
        test_df = add_fluorinated_group_counts(test_df, mol_col="rdkit_mol")

        # Check group identification columns
        required_cols = ["num_cf3_groups", "num_cf2_groups", "num_cf_groups"]
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")

        # Trifluoromethane has 1 CF3 group
        self.assertEqual(test_df.iloc[0]["num_cf3_groups"], 1)
        self.assertEqual(test_df.iloc[0]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[0]["num_cf_groups"], 0)

        # Hexafluoroethane has 2 CF3 groups
        self.assertEqual(test_df.iloc[1]["num_cf3_groups"], 2)
        self.assertEqual(test_df.iloc[1]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[1]["num_cf_groups"], 0)

        # Ethane has no fluorinated groups
        self.assertEqual(test_df.iloc[3]["num_cf3_groups"], 0)
        self.assertEqual(test_df.iloc[3]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[3]["num_cf_groups"], 0)

        print("Successfully identified fluorinated groups")

    def test_graph_processing(self):
        """Test creating molecular graphs with PFAS-specific features."""
        # Process the molecules into graphs
        for i, row in self.test_df.iterrows():
            mol = row["rdkit_mol"]
            if mol is not None:
                graph_data = self.graph_processor.mol_to_graph(mol)
                self.assertIsNotNone(graph_data, f"mol_to_graph returned None for {row['name']}")
                self.assertTrue(hasattr(graph_data, "x"), "Graph data missing 'x' (atom features)")
                self.assertTrue(hasattr(graph_data, "edge_index"), "Graph data missing 'edge_index'")

                self.assertEqual(graph_data.x.shape[0], mol.GetNumAtoms())
                self.assertEqual(graph_data.x.shape[1], self.graph_processor.atom_feature_dim)

                # Check edge_index basic properties
                self.assertEqual(graph_data.edge_index.shape[0], 2)
                self.assertTrue(graph_data.edge_index.shape[1] >= 0)  # Can be 0 for single atom mol

        # Refactor dataframe processing part:
        # Create a new column 'graph_data' by applying mol_to_graph
        # This simulates what a user might do.
        processed_graphs = []
        for i, row in self.test_df.iterrows():
            mol = row["rdkit_mol"]
            if mol:
                graph = self.graph_processor.mol_to_graph(mol)
                processed_graphs.append(graph)
            else:
                processed_graphs.append(None)

        # Add the list of graphs as a new column (or handle as a list of results)
        # For this test, we'll just check the list of graphs.
        # processed_df = self.test_df.copy()
        # processed_df['graph_data_obj'] = processed_graphs

        self.assertEqual(len(processed_graphs), len(self.test_df))
        # Check first graph as an example if it exists
        if processed_graphs and processed_graphs[0] is not None:
            self.assertIsNotNone(processed_graphs[0].x)
            self.assertIsNotNone(processed_graphs[0].edge_index)
            self.assertTrue(hasattr(processed_graphs[0], "num_nodes"))

        # Corrected print statement to use a defined variable
        print(f"Successfully processed {len(self.test_df[self.test_df['rdkit_mol'].notna()])} molecules into graphs")


class TestHydroxylGroupDetection(unittest.TestCase):
    """Test hydroxyl group detection functionality."""

    def setUp(self):
        """Set up test molecules with various hydroxyl groups."""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Required MOML modules not available")

        # Test SMILES with hydroxyl groups
        self.test_molecules = {
            # Simple alcohols
            "methanol": "CO",  # Has 1 OH group
            "ethanol": "CCO",  # Has 1 OH group
            "propanol": "CCCO",  # Has 1 OH group
            "glycol": "OCCO",  # Has 2 OH groups
            "glycerol": "OCC(O)CO",  # Has 3 OH groups
            
            # Carboxylic acids (have OH but in COOH context)
            "formic_acid": "C(=O)O",  # Has 1 OH group in COOH
            "acetic_acid": "CC(=O)O",  # Has 1 OH group in COOH
            
            # Phenols
            "phenol": "c1ccc(O)cc1",  # Has 1 OH group (aromatic)
            
            # Molecules without OH groups
            "methane": "C",  # No OH groups
            "ethene": "C=C",  # No OH groups
            "benzene": "c1ccccc1",  # No OH groups
            
            # PFAS with OH groups
            "pfoa": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # Has 1 OH in COOH
        }

        # Convert to RDKit molecules
        self.rdkit_mols = {}
        for name, smiles in self.test_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)  # Add explicit hydrogens for accurate OH detection
                self.rdkit_mols[name] = mol
            else:
                self.rdkit_mols[name] = None

    def test_find_hydroxyl_groups_basic(self):
        """Test basic hydroxyl group detection."""
        # Test molecules with known OH group counts
        expected_counts = {
            "methanol": 1,
            "ethanol": 1,
            "propanol": 1,
            "glycol": 2,
            "glycerol": 3,
            "formic_acid": 1,
            "acetic_acid": 1,
            "phenol": 1,
            "methane": 0,
            "ethene": 0,
            "benzene": 0,
            "pfoa": 1,
        }

        for mol_name, expected_count in expected_counts.items():
            mol = self.rdkit_mols.get(mol_name)
            if mol is not None:
                hydroxyl_groups = FunctionalGroupDetector.find_hydroxyl_groups(mol)
                self.assertEqual(
                    len(hydroxyl_groups), 
                    expected_count,
                    f"{mol_name} should have {expected_count} hydroxyl group(s), but found {len(hydroxyl_groups)}"
                )

    def test_find_hydroxyl_groups_edge_cases(self):
        """Test edge cases for hydroxyl group detection."""
        # Test with None molecule
        result = FunctionalGroupDetector.find_hydroxyl_groups(None)
        self.assertEqual(result, [])

        # Test with empty molecule (if possible to create)
        try:
            empty_mol = Chem.MolFromSmiles("")
            if empty_mol is not None:
                result = FunctionalGroupDetector.find_hydroxyl_groups(empty_mol)
                self.assertEqual(result, [])
        except Exception:
            pass  # Skip if cannot create empty molecule

    def test_hydroxyl_oxygen_indices(self):
        """Test that returned indices correspond to oxygen atoms."""
        test_mol = self.rdkit_mols.get("glycerol")  # Should have 3 OH groups
        if test_mol is not None:
            hydroxyl_groups = FunctionalGroupDetector.find_hydroxyl_groups(test_mol)
            
            # Check that all returned indices are oxygen atoms
            for oxygen_idx in hydroxyl_groups:
                atom = test_mol.GetAtomWithIdx(oxygen_idx)
                self.assertEqual(
                    atom.GetAtomicNum(), 
                    8, 
                    f"Index {oxygen_idx} should correspond to oxygen atom, but got atomic number {atom.GetAtomicNum()}"
                )

    def test_hydroxyl_groups_in_functional_groups_dict(self):
        """Test that get_all_functional_groups includes hydroxyl groups."""
        test_mol = self.rdkit_mols.get("glycerol")  # Should have 3 OH groups
        if test_mol is not None:
            all_groups = FunctionalGroupDetector.get_all_functional_groups(test_mol)
            
            # Check that hydroxyl_groups key exists
            self.assertIn("hydroxyl_groups", all_groups)
            
            # Check that we get the expected number of hydroxyl groups
            self.assertEqual(len(all_groups["hydroxyl_groups"]), 3)

    def test_hydroxyl_groups_with_different_molecules(self):
        """Test hydroxyl group detection with various molecular structures."""
        test_cases = [
            ("methanol", 1),
            ("glycol", 2), 
            ("glycerol", 3),
            ("methane", 0),
        ]

        for mol_name, expected_count in test_cases:
            mol = self.rdkit_mols.get(mol_name)
            if mol is not None:
                with self.subTest(molecule=mol_name):
                    hydroxyl_groups = FunctionalGroupDetector.find_hydroxyl_groups(mol)
                    self.assertEqual(
                        len(hydroxyl_groups),
                        expected_count,
                        f"Failed for {mol_name}: expected {expected_count}, got {len(hydroxyl_groups)}"
                    )

    def test_hydroxyl_groups_error_handling(self):
        """Test error handling in hydroxyl group detection."""
        # Test with malformed molecule (if we can create one)
        try:
            # Create a molecule and then corrupt it somehow
            mol = Chem.MolFromSmiles("CO")
            if mol is not None:
                # The function should handle errors gracefully
                result = FunctionalGroupDetector.find_hydroxyl_groups(mol)
                self.assertIsInstance(result, list)
        except Exception:
            pass  # Expected if molecule creation fails


def run_pfas_feature_tests():
    """Run the PFAS feature tests."""
    print("\nTesting PFAS-specific features...")

    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPFASFeatures))
    suite.addTest(unittest.makeSuite(TestHydroxylGroupDetection))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_pfas_feature_tests()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
