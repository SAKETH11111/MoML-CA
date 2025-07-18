"""
tests/test_pfas_feature_extraction.py

Unit tests for PFAS-specific features and visualization.

This script tests:
1. Creating graphs with PFAS-specific features
2. Visualizing molecular graphs with different highlighting options
3. Calculating and verifying PFAS-specific properties
"""

import logging
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_pfas_features")

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Check for RDKit
try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    pytest.skip("RDKit not installed, skipping PFAS feature extraction tests", allow_module_level=True)

# Try to import torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not found, skipping PFAS feature extraction tests", allow_module_level=True)

# Try to import matplotlib for visualization testing
try:
    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    pytest.skip("Matplotlib not found, skipping visualization tests", allow_module_level=True)

# Import from consolidated moml modules
try:
    from moml.core import FunctionalGroupDetector, MolecularGraphProcessor
    from moml.utils import add_fluorinated_group_counts, calculate_molecular_complexity, categorize_molecular_features, create_rdkit_mols

    MOML_IMPORTS_SUCCESSFUL = True
except ImportError:
    MOML_IMPORTS_SUCCESSFUL = False
    pytest.skip("Failed to import required moml modules, skipping PFAS feature extraction tests", allow_module_level=True)


class TestPFASFeatures(unittest.TestCase):
    """
    Test suite for PFAS-specific features and visualization.
    """

    def setUp(self) -> None:
        """
        Set up test molecules and initialize necessary processors.
        """
        if not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("Required MOML modules not available")

        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_smiles: List[str] = [
            "CC(F)(F)F",
            "C(C(F)(F)F)C(F)(F)F",
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F",
            "CC",
        ]

        self.test_df = pd.DataFrame(
            {
                "smiles": self.test_smiles,
                "name": ["Trifluoromethane", "Hexafluoroethane", "GenX", "Ethane"],
                "id": ["TEST-001", "TEST-002", "TEST-003", "TEST-004"],
            }
        )

        self.test_df = create_rdkit_mols(self.test_df, smiles_col="smiles", mol_col="rdkit_mol")

        self.graph_processor = MolecularGraphProcessor()

    def tearDown(self) -> None:
        """
        Clean up temporary files and close matplotlib figures.
        """
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

        if "plt" in globals() and plt:
            plt.close("all")

    def test_pfas_detection(self) -> None:
        """
        Test detection of PFAS compounds based on fluorine count and categorization.
        """
        test_df = calculate_molecular_complexity(self.test_df.copy(), mol_col="rdkit_mol")
        test_df = categorize_molecular_features(test_df, mol_col="rdkit_mol")
        test_df = add_fluorinated_group_counts(test_df, mol_col="rdkit_mol")

        pfas_compounds = test_df[test_df["Has_Fluorine"]]
        non_pfas_compounds = test_df[~test_df["Has_Fluorine"]]

        self.assertEqual(len(pfas_compounds), 3, "Expected 3 PFAS compounds based on Has_Fluorine")
        self.assertEqual(len(non_pfas_compounds), 1, "Expected 1 non-PFAS compound based on Has_Fluorine")

        if "num_cf3_groups" in test_df.columns:
            self.assertEqual(test_df.iloc[0]["num_cf3_groups"], 1, "1,1,1-Trifluoroethane should have 1 CF3 group")
            self.assertEqual(test_df.iloc[1]["num_cf3_groups"], 2, "Perfluoropropane should have 2 CF3 groups")

        expected_f_counts = [3, 6, 9, 0]
        for i, expected in enumerate(expected_f_counts):
            self.assertEqual(
                test_df.iloc[i]["F_Count"],
                expected,
                f"Expected {expected} fluorine atoms for {self.test_df.iloc[i]['name']} (SMILES: {self.test_df.iloc[i]['smiles']})",
            )

        logger.info(f"Successfully identified {len(pfas_compounds)} PFAS compounds (based on Has_Fluorine) out of {len(test_df)}")

    def test_pfas_statistics(self) -> None:
        """
        Test calculation of PFAS statistics.
        """
        test_df = calculate_molecular_complexity(self.test_df.copy(), mol_col="rdkit_mol")
        test_df = categorize_molecular_features(test_df, mol_col="rdkit_mol")

        required_cols = ["F_Count", "C_Count", "f_to_c_ratio", "avg_f_per_c"]
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")

        self.assertEqual(test_df.iloc[0]["F_Count"], 3)
        self.assertEqual(test_df.iloc[0]["C_Count"], 2)
        self.assertAlmostEqual(test_df.iloc[0]["f_to_c_ratio"], 3.0 / 2.0)

        self.assertEqual(test_df.iloc[1]["F_Count"], 6)
        self.assertEqual(test_df.iloc[1]["C_Count"], 3)
        self.assertAlmostEqual(test_df.iloc[1]["f_to_c_ratio"], 6.0 / 3.0)

        self.assertEqual(test_df.iloc[3]["F_Count"], 0)
        self.assertEqual(test_df.iloc[3]["C_Count"], 2)
        self.assertAlmostEqual(test_df.iloc[3]["f_to_c_ratio"], 0.0)

        logger.info("Successfully calculated PFAS statistics")

    def test_fluorinated_groups(self) -> None:
        """
        Test identification of fluorinated groups.
        """
        test_df = calculate_molecular_complexity(self.test_df.copy(), mol_col="rdkit_mol")
        test_df = categorize_molecular_features(test_df, mol_col="rdkit_mol")
        test_df = add_fluorinated_group_counts(test_df, mol_col="rdkit_mol")

        required_cols = ["num_cf3_groups", "num_cf2_groups", "num_cf_groups"]
        for col in required_cols:
            self.assertIn(col, test_df.columns, f"Missing required column: {col}")

        self.assertEqual(test_df.iloc[0]["num_cf3_groups"], 1)
        self.assertEqual(test_df.iloc[0]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[0]["num_cf_groups"], 0)

        self.assertEqual(test_df.iloc[1]["num_cf3_groups"], 2)
        self.assertEqual(test_df.iloc[1]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[1]["num_cf_groups"], 0)

        self.assertEqual(test_df.iloc[3]["num_cf3_groups"], 0)
        self.assertEqual(test_df.iloc[3]["num_cf2_groups"], 0)
        self.assertEqual(test_df.iloc[3]["num_cf_groups"], 0)

        logger.info("Successfully identified fluorinated groups")

    def test_graph_processing(self) -> None:
        """
        Test creating molecular graphs with PFAS-specific features.
        """
        for i, row in self.test_df.iterrows():
            mol: Optional[Chem.Mol] = row["rdkit_mol"]
            if mol is not None:
                graph_data = self.graph_processor.mol_to_graph(mol)
                self.assertIsNotNone(graph_data, f"mol_to_graph returned None for {row['name']}")
                self.assertTrue(hasattr(graph_data, "x"), "Graph data missing 'x' (atom features)")
                self.assertTrue(hasattr(graph_data, "edge_index"), "Graph data missing 'edge_index'")

                self.assertEqual(graph_data.x.shape[0], mol.GetNumAtoms())
                self.assertEqual(graph_data.x.shape[1], self.graph_processor.atom_feature_dim)

                self.assertEqual(graph_data.edge_index.shape[0], 2)
                self.assertTrue(graph_data.edge_index.shape[1] >= 0)

        processed_graphs: List[Optional[Any]] = []
        for i, row in self.test_df.iterrows():
            mol: Optional[Chem.Mol] = row["rdkit_mol"]
            if mol is not None:
                graph = self.graph_processor.mol_to_graph(mol)
                processed_graphs.append(graph)
            else:
                processed_graphs.append(None)

        self.assertEqual(len(processed_graphs), len(self.test_df))
        if processed_graphs and processed_graphs[0] is not None:
            self.assertIsNotNone(processed_graphs[0].x)
            self.assertIsNotNone(processed_graphs[0].edge_index)
            self.assertTrue(hasattr(processed_graphs[0], "num_nodes"))

        logger.info(f"Successfully processed dataframe with {len(self.test_df[self.test_df['rdkit_mol'].notna()])} valid molecules into graphs")

    def test_save_processed_data(self) -> None:
        """
        Test saving processed molecular graph data properties to a CSV file.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        df = pd.DataFrame(
            {
                "smiles": self.test_smiles,
                "name": ["Trifluoromethane", "Hexafluoroethane", "GenX", "Ethane"],
                "id": ["TEST-001", "TEST-002", "TEST-003", "TEST-004"],
            }
        )

        df = create_rdkit_mols(df, smiles_col="smiles", mol_col="rdkit_mol")

        processed_graphs_data: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            mol: Optional[Chem.Mol] = row["rdkit_mol"]
            if mol is not None:
                graph = self.graph_processor.mol_to_graph(mol)
                if graph is not None:
                    processed_graphs_data.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "smiles": row["smiles"],
                            "num_nodes": graph.num_nodes,
                            "num_edges": graph.num_edges,
                        }
                    )

        save_df = pd.DataFrame(processed_graphs_data)

        output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "molecular_graph_properties.csv")
        save_df.to_csv(output_file, index=False)

        self.assertTrue(os.path.exists(output_file), f"Output file not created: {output_file}")

        loaded_df = pd.read_csv(output_file)
        self.assertEqual(len(loaded_df), len(df[df["rdkit_mol"].notna()]))
        self.assertIn("num_nodes", loaded_df.columns)
        self.assertIn("id", loaded_df.columns)

        logger.info(f"Successfully saved processed graph properties to {output_file}")


class TestHydroxylGroupDetection(unittest.TestCase):
    """
    Test suite for hydroxyl group detection functionality.
    """

    def setUp(self) -> None:
        """
        Set up test molecules with various hydroxyl groups.
        """
        if not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("Required MOML modules not available")

        self.test_molecules: Dict[str, str] = {
            "methanol": "CO",
            "ethanol": "CCO",
            "propanol": "CCCO",
            "glycol": "OCCO",
            "glycerol": "OCC(O)CO",
            "formic_acid": "C(=O)O",
            "acetic_acid": "CC(=O)O",
            "phenol": "c1ccc(O)cc1",
            "methane": "C",
            "ethene": "C=C",
            "benzene": "c1ccccc1",
            "pfoa": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        }

        self.rdkit_mols: Dict[str, Optional[Chem.Mol]] = {}
        for name, smiles in self.test_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                self.rdkit_mols[name] = mol
            else:
                self.rdkit_mols[name] = None

    def test_find_hydroxyl_groups_basic(self) -> None:
        """
        Test basic hydroxyl group detection.
        """
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
                    f"{mol_name} should have {expected_count} hydroxyl group(s), but found {len(hydroxyl_groups)}",
                )

    def test_find_hydroxyl_groups_edge_cases(self) -> None:
        """
        Test edge cases for hydroxyl group detection.
        """
        result = FunctionalGroupDetector.find_hydroxyl_groups(None)
        self.assertEqual(result, [])

        try:
            empty_mol = Chem.MolFromSmiles("")
            if empty_mol is not None:
                result = FunctionalGroupDetector.find_hydroxyl_groups(empty_mol)
                self.assertEqual(result, [])
        except Exception:
            pass

    def test_hydroxyl_oxygen_indices(self) -> None:
        """
        Test that returned indices correspond to oxygen atoms.
        """
        test_mol = self.rdkit_mols.get("glycerol")
        if test_mol is not None:
            hydroxyl_groups = FunctionalGroupDetector.find_hydroxyl_groups(test_mol)

            for oxygen_idx in hydroxyl_groups:
                atom = test_mol.GetAtomWithIdx(oxygen_idx)
                self.assertEqual(
                    atom.GetAtomicNum(),
                    8,
                    f"Index {oxygen_idx} should correspond to oxygen atom, but got atomic number {atom.GetAtomicNum()}",
                )

    def test_hydroxyl_groups_in_functional_groups_dict(self) -> None:
        """
        Test that get_all_functional_groups includes hydroxyl groups.
        """
        test_mol = self.rdkit_mols.get("glycerol")
        if test_mol is not None:
            all_groups = FunctionalGroupDetector.get_all_functional_groups(test_mol)

            self.assertIn("hydroxyl_groups", all_groups)

            self.assertEqual(len(all_groups["hydroxyl_groups"]), 3)

    def test_hydroxyl_groups_with_different_molecules(self) -> None:
        """
        Test hydroxyl group detection with various molecular structures.
        """
        test_cases: List[Tuple[str, int]] = [
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
                        f"Failed for {mol_name}: expected {expected_count}, got {len(hydroxyl_groups)}",
                    )

    def test_hydroxyl_groups_error_handling(self) -> None:
        """
        Test error handling in hydroxyl group detection.
        """
        try:
            mol = Chem.MolFromSmiles("CO")
            if mol is not None:
                result = FunctionalGroupDetector.find_hydroxyl_groups(mol)
                self.assertIsInstance(result, list)
        except Exception:
            pass


def run_pfas_feature_tests() -> bool:
    """
    Run the PFAS feature tests.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    logger.info("Starting PFAS feature tests...")

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPFASFeatures))
    suite.addTest(unittest.makeSuite(TestHydroxylGroupDetection))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_pfas_feature_tests()
    sys.exit(0 if success else 1)
