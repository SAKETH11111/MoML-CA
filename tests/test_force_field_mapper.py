"""
tests/test_force_field_mapper.py

Unit tests for the ForceFieldMapper class in
moml.simulation.molecular_dynamics.force_field.mapper.
"""

import os
import shutil
import tempfile
from typing import Any, Generator

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from moml.simulation.molecular_dynamics.force_field.mapper import ForceFieldMapper


def create_test_mol(smiles: str, add_3d: bool = True) -> Chem.Mol:
    """
    Create an RDKit molecule from SMILES, add Hs, and optionally 3D coords.

    Args:
        smiles (str): SMILES string.
        add_3d (bool): Whether to add 3D coordinates.

    Returns:
        Chem.Mol: The created molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if add_3d:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # type: ignore[attr-defined]
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)  # type: ignore[attr-defined]
        if mol.GetNumConformers() == 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
            mol.AddConformer(conf, assignId=True)
    return mol

@pytest.fixture
def ethanol_mol_3d() -> Chem.Mol:
    """
    Returns a 3D RDKit molecule for ethanol (CCO).
    """
    return create_test_mol("CCO", add_3d=True)

@pytest.fixture
def ethanol_mol_2d() -> Chem.Mol:
    """
    Returns a 2D RDKit molecule for ethanol (CCO).
    """
    return create_test_mol("CCO", add_3d=False)

@pytest.fixture
def pfoa_frag_mol_3d() -> Chem.Mol:
    """
    Returns a 3D RDKit molecule for a PFOA fragment (CF3COOH).
    """
    return create_test_mol("C(F)(F)(F)C(=O)O", add_3d=True)

@pytest.fixture(scope="module")
def temp_output_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for output files.

    Yields:
        str: Path to the temporary directory.
    """
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

class TestForceFieldMapper:
    """
    Tests for the ForceFieldMapper class.
    """

    def test_init_default(self) -> None:
        """
        Test ForceFieldMapper initialization with default parameters.
        """
        mapper = ForceFieldMapper()
        assert mapper.config == {}

    def test_init_custom(self) -> None:
        """
        Test ForceFieldMapper initialization with custom config.
        """
        config = {"test_param": "test_value"}
        mapper = ForceFieldMapper(config=config)
        assert mapper.config == config

    def test_map_partial_charges_no_normalize(self, ethanol_mol_3d: Chem.Mol) -> None:
        """
        Test map_partial_charges without normalization.
        """
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_in = [0.1 * i for i in range(num_atoms)]
        charge_map = mapper.map_partial_charges(ethanol_mol_3d, charges_in, normalize=False)
        assert len(charge_map) == num_atoms
        for i in range(num_atoms):
            assert charge_map[i] == charges_in[i]

    def test_map_partial_charges_normalize(self, ethanol_mol_3d: Chem.Mol) -> None:
        """
        Test map_partial_charges with normalization.
        """
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_in = [0.1 + (0.05 * i) for i in range(num_atoms)]
        charge_map = mapper.map_partial_charges(ethanol_mol_3d, charges_in, normalize=True)
        assert len(charge_map) == num_atoms
        formal_charge = Chem.GetFormalCharge(ethanol_mol_3d)
        assert abs(sum(charge_map.values()) - formal_charge) < 1e-6

    def test_map_partial_charges_mismatch(self, ethanol_mol_3d: Chem.Mol) -> None:
        """
        Test map_partial_charges with mismatched number of charges.
        """
        mapper = ForceFieldMapper()
        charges_in = [0.1]
        with pytest.raises(ValueError, match="doesn't match number of charges"):
            mapper.map_partial_charges(ethanol_mol_3d, charges_in)
