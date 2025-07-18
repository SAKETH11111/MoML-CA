"""
tests/test_molecular_graph_structure.py

Unit tests for the MolecularGraphProcessor class, focusing on graph structure
and feature generation.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from moml.core import MolecularGraphProcessor, create_graph_processor


def create_test_molecule() -> Chem.Mol:
    """
    Create a simple test molecule (PFOA - Perfluorooctanoic acid) with 3D coordinates.

    Returns:
        Chem.Mol: The RDKit molecule object.
    """
    mol = Chem.MolFromSmiles("C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore
    AllChem.UFFOptimizeMolecule(mol)  # type: ignore

    return mol


class TestMolecularGraphProcessor:
    """
    Test suite for the MolecularGraphProcessor class.
    """

    def test_initialization(self) -> None:
        """
        Test processor initialization with different options.
        """
        processor = create_graph_processor()
        assert processor.use_partial_charges is True
        assert processor.use_3d_coords is True
        assert processor.use_pfas_specific_features is True

        processor = create_graph_processor(
            {"use_partial_charges": False, "use_3d_coords": False, "use_pfas_specific_features": False}
        )
        assert processor.use_partial_charges is False
        assert processor.use_3d_coords is False
        assert processor.use_pfas_specific_features is False

    def test_one_hot_encoding(self) -> None:
        """
        Test the one_hot_encoding function.
        """
        processor = create_graph_processor()

        encoding = processor._one_hot_encoding(1, [0, 1, 2])
        assert encoding == [0, 1, 0]

        encoding = processor._one_hot_encoding(3, [0, 1, 2])
        assert encoding == [0, 0, 0]

    def test_functional_group_detection(self) -> None:
        """
        Test functional group detection methods.
        """
        processor = create_graph_processor()
        mol = create_test_molecule()

        found_carboxylic = False
        for atom in mol.GetAtoms():
            if processor._is_in_carboxylic_group(atom):
                found_carboxylic = True
                break

        assert found_carboxylic, "Carboxylic group not detected in PFOA"

        cf3_groups = processor.functional_group_detector.find_cf3_groups(mol)
        assert len(cf3_groups) > 0, "CF3 groups not detected in PFOA"

    def test_mol_to_graph_basic(self) -> None:
        """
        Test basic graph conversion without partial charges.
        """
        processor = create_graph_processor({"use_partial_charges": False})
        mol = create_test_molecule()

        graph = processor.mol_to_graph(mol)

        assert isinstance(graph.x, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_attr, torch.Tensor)
        assert graph.num_nodes == mol.GetNumAtoms()

    def test_mol_to_graph_with_pfas_features(self) -> None:
        """
        Test graph conversion with PFAS-specific features.
        """
        processor = create_graph_processor({"use_partial_charges": False, "use_pfas_specific_features": True})
        mol = create_test_molecule()

        graph = processor.mol_to_graph(mol)

        assert graph.x.shape[1] > 10

    def test_mol_to_graph_with_partial_charges(self) -> None:
        """
        Test graph conversion with partial charges.
        """
        processor = create_graph_processor({"use_partial_charges": True})
        mol = create_test_molecule()

        num_atoms = mol.GetNumAtoms()
        partial_charges = np.random.uniform(-1, 1, num_atoms).tolist()

        graph = processor.mol_to_graph(mol, {"partial_charges": partial_charges})

        assert graph.x.shape[1] > 10

    def test_mol_to_graph_with_homo_lumo(self) -> None:
        """
        Test graph conversion with HOMO/LUMO contributions.
        """
        processor = create_graph_processor({"use_partial_charges": True})
        mol = create_test_molecule()

        num_atoms = mol.GetNumAtoms()
        partial_charges = np.random.uniform(-1, 1, num_atoms).tolist()

        homo_contributions = np.random.uniform(0, 0.2, num_atoms).tolist()
        lumo_contributions = np.random.uniform(0, 0.2, num_atoms).tolist()

        additional_features = {
            "partial_charges": partial_charges,
            "homo_contributions": homo_contributions,
            "lumo_contributions": lumo_contributions,
        }

        graph = processor.mol_to_graph(mol, additional_features)

        expected_dim = 15 + 2
        assert graph.x.shape[1] >= expected_dim

    def test_global_features(self) -> None:
        """
        Test that global features are correctly calculated.
        """
        processor = create_graph_processor()
        mol = create_test_molecule()

        graph = processor.mol_to_graph(mol)

        assert hasattr(graph, "y")
        assert graph.y is not None
        assert graph.y.shape[0] >= 7

    def test_smiles_to_graph(self) -> None:
        """
        Test creating a graph from a SMILES string.
        """
        processor = create_graph_processor()

        smiles = "C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F"

        graph = processor.smiles_to_graph(smiles)

        assert graph is not None
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[0] == 2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
