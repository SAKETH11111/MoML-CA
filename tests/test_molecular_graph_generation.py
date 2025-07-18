"""
tests/test_molecular_graph_generation.py

Unit tests for the molecular graph generation utilities.

This script tests the functionality of the molecular graph generation utilities:
1. Creating molecular graphs from SMILES strings
2. Creating graphs from molecules with 3D coordinates
3. Testing batch processing capabilities
"""

import logging
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_graph_generation")

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Check for RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    pytest.skip("RDKit not installed, skipping molecular graph generation tests", allow_module_level=True)

# Try to import torch and torch_geometric
try:
    from torch_geometric.data import Data as PyGData

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch or PyTorch Geometric not available, skipping molecular graph generation tests", allow_module_level=True)

# Import from consolidated moml modules
try:
    from moml.core import MolecularGraphProcessor
    from moml.data.processors.process_chemical_data import create_rdkit_mols
    from moml.utils import validate_smiles

    MOML_IMPORTS_SUCCESSFUL = True
except ImportError:
    MOML_IMPORTS_SUCCESSFUL = False
    pytest.skip("Failed to import required moml modules, skipping molecular graph generation tests", allow_module_level=True)


class TestMolecularGraphGenerator(unittest.TestCase):
    """
    Test suite for the molecular graph generation functionality.
    """

    def setUp(self) -> None:
        """
        Set up test molecules and initialize MolecularGraphProcessor.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("Required dependencies not available")

        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_smiles: List[str] = [
            "CC",
            "CC(F)(F)F",
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F",
        ]

        self.mols: List[Chem.Mol] = []
        self.mol_files: List[str] = []

        for i, smiles in enumerate(self.test_smiles):
            is_valid, canonical_smi, mol_obj, error_msg = validate_smiles(smiles)
            if is_valid and mol_obj is not None:
                mol = mol_obj
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore
                AllChem.MMFFOptimizeMolecule(mol)  # type: ignore

                self.mols.append(mol)

                mol_file = os.path.join(self.temp_dir.name, f"test_mol_{i}.mol")
                Chem.MolToMolFile(mol, mol_file)
                self.mol_files.append(mol_file)
            else:
                logger.warning(f"SMILES validation failed for '{smiles}': {error_msg}")

        self.graph_processor = MolecularGraphProcessor()

    def tearDown(self) -> None:
        """
        Clean up temporary files after tests.
        """
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def test_mol_to_graph_basic(self) -> None:
        """
        Test basic conversion of a simple molecule (ethane) to a graph.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        mol = self.mols[0]

        graph = self.graph_processor.mol_to_graph(mol)
        self.assertIsInstance(graph, PyGData)

        self.assertEqual(graph.num_nodes, mol.GetNumAtoms())
        self.assertTrue(hasattr(graph, "x"))
        self.assertTrue(hasattr(graph, "edge_index"))

        self.assertEqual(graph.x.shape[0], mol.GetNumAtoms())
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)

        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertGreaterEqual(graph.edge_index.shape[1], mol.GetNumBonds())

        if mol.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, "edge_attr"))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])

    def test_mol_to_graph_with_trifluoromethane(self) -> None:
        """
        Test conversion of trifluoromethane molecule to a graph.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        mol = self.mols[1]
        num_atoms = mol.GetNumAtoms()

        graph = self.graph_processor.mol_to_graph(mol)
        self.assertIsInstance(graph, PyGData)

        self.assertEqual(graph.num_nodes, num_atoms)
        self.assertTrue(hasattr(graph, "x"))
        self.assertEqual(graph.x.shape[0], num_atoms)
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)

        self.assertTrue(hasattr(graph, "edge_index"))
        self.assertEqual(graph.edge_index.shape[0], 2)

        if mol.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, "edge_attr"))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])

        if self.graph_processor.use_3d_coords:
            self.assertTrue(hasattr(graph, "pos"))
            self.assertEqual(graph.pos.shape, (num_atoms, 3))

    def test_mol_from_file_to_graph(self) -> None:
        """
        Test loading a molecule from a file and converting it to a graph.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        mol_file = self.mol_files[0]
        self.assertTrue(os.path.exists(mol_file), f"Mol file not found: {mol_file}")

        mol_from_file_raw = Chem.MolFromMolFile(mol_file, removeHs=False)
        self.assertIsNotNone(mol_from_file_raw, "Failed to load molecule from MOL file")

        mol_for_comparison = Chem.AddHs(mol_from_file_raw)

        graph = self.graph_processor.file_to_graph(mol_file)
        self.assertIsInstance(graph, PyGData)

        self.assertEqual(graph.num_nodes, mol_for_comparison.GetNumAtoms())
        self.assertTrue(hasattr(graph, "x"))
        self.assertEqual(graph.x.shape[0], mol_for_comparison.GetNumAtoms())
        self.assertEqual(graph.x.shape[1], self.graph_processor.atom_feature_dim)

        self.assertTrue(hasattr(graph, "edge_index"))
        if mol_for_comparison.GetNumBonds() > 0:
            self.assertTrue(hasattr(graph, "edge_attr"))
            self.assertEqual(graph.edge_attr.shape[1], self.graph_processor.bond_feature_dim)
            self.assertEqual(graph.edge_attr.shape[0], graph.edge_index.shape[1])

    def test_process_dataframe(self) -> None:
        """
        Test processing a pandas DataFrame of molecules into graphs.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        df = pd.DataFrame(
            {
                "smiles": self.test_smiles,
                "name": ["Ethane", "Trifluoromethane", "GenX"],
                "id": ["TEST-001", "TEST-002", "TEST-003"],
            }
        )

        df = create_rdkit_mols(df, smiles_col="smiles", mol_col="rdkit_mol")

        processed_graphs: List[Optional[PyGData]] = []
        num_atoms_list: List[int] = []
        for idx, row in df.iterrows():
            mol: Optional[Chem.Mol] = row["rdkit_mol"] # Explicitly type as Optional[Chem.Mol]
            if mol is not None: # Check if mol is not None
                graph = self.graph_processor.mol_to_graph(mol)
                if graph is not None: # Check if graph is not None
                    processed_graphs.append(graph)
                    num_atoms_list.append(mol.GetNumAtoms())
            else:
                processed_graphs.append(None)
                num_atoms_list.append(0)

        self.assertEqual(len(processed_graphs), len(df))

        for i, graph_obj in enumerate(processed_graphs):
            original_mol: Optional[Chem.Mol] = df.iloc[i]["rdkit_mol"] # Explicitly type as Optional[Chem.Mol]
            if original_mol is not None: # Check if original_mol is not None
                self.assertIsInstance(graph_obj, PyGData)
                if graph_obj is not None: # Check if graph_obj is not None
                    self.assertEqual(graph_obj.num_nodes, num_atoms_list[i])
                    self.assertTrue(hasattr(graph_obj, "x"))
                    self.assertEqual(graph_obj.x.shape[0], num_atoms_list[i])
                    self.assertEqual(graph_obj.x.shape[1], self.graph_processor.atom_feature_dim)
            else:
                self.assertIsNone(graph_obj)

        logger.info(f"Successfully processed dataframe with {len(df[df['rdkit_mol'].notna()])} valid molecules into graphs")

    def test_save_processed_data(self) -> None:
        """
        Test saving processed molecular graph data properties to a CSV file.
        """
        if not TORCH_AVAILABLE or not MOML_IMPORTS_SUCCESSFUL:
            self.skipTest("PyTorch, PyTorch Geometric, or MOML modules not available")

        df = pd.DataFrame(
            {
                "smiles": self.test_smiles,
                "name": ["Ethane", "Trifluoromethane", "GenX"],
                "id": ["TEST-001", "TEST-002", "TEST-003"],
            }
        )

        df = create_rdkit_mols(df, smiles_col="smiles", mol_col="rdkit_mol")

        processed_graphs_data: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            mol: Optional[Chem.Mol] = row["rdkit_mol"] # Explicitly type as Optional[Chem.Mol]
            if mol is not None: # Check if mol is not None
                graph = self.graph_processor.mol_to_graph(mol)
                if graph is not None: # Check if graph is not None
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


def run_graph_generation_tests() -> bool:
    """
    Run all molecular graph generation tests.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMolecularGraphGenerator)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    logger.info("Starting molecular graph generation tests...")
    success = run_graph_generation_tests()
    sys.exit(0 if success else 1)
