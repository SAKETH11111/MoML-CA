"""
Unit tests for the ForceFieldMapper class in
moml.simulation.molecular_dynamics.force_field_mapper.
"""

import pytest
import os
import logging
import tempfile
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from moml.simulation.molecular_dynamics.force_field.mapper import ForceFieldMapper


# Helper function to create RDKit molecules
def create_test_mol(smiles: str, add_3d: bool = True) -> Chem.Mol:
    """Creates an RDKit molecule from SMILES, adds Hs, and optionally 3D coords."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if add_3d:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if mol.GetNumConformers() == 0:  # Fallback
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)
        if mol.GetNumConformers() == 0:  # Final fallback for very simple mols
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
            mol.AddConformer(conf, assignId=True)
    return mol


# Test Fixtures
@pytest.fixture
def ethanol_mol_3d() -> Chem.Mol:
    """Returns a 3D RDKit molecule for ethanol (CCO)."""
    return create_test_mol("CCO", add_3d=True)


@pytest.fixture
def ethanol_mol_2d() -> Chem.Mol:
    """Returns a 2D RDKit molecule for ethanol (CCO)."""
    return create_test_mol("CCO", add_3d=False)


@pytest.fixture
def pfoa_frag_mol_3d() -> Chem.Mol:
    """Returns a 3D RDKit molecule for a PFOA fragment (CF3COOH)."""
    return create_test_mol("C(F)(F)(F)C(=O)O", add_3d=True)


@pytest.fixture(scope="module")
def temp_output_dir():
    """Creates a temporary directory for output files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


class TestForceFieldMapper:
    """Tests for the ForceFieldMapper class."""

    def test_init_default(self):
        """Test ForceFieldMapper initialization with default parameters."""
        mapper = ForceFieldMapper()
        assert mapper.config == {}

    def test_init_custom(self):
        """Test ForceFieldMapper initialization with custom config."""
        config = {"test_param": "test_value"}
        mapper = ForceFieldMapper(config=config)
        assert mapper.config == config

    def test_map_partial_charges_no_normalize(self, ethanol_mol_3d: Chem.Mol):
        """Test map_partial_charges without normalization."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_in = [0.1 * i for i in range(num_atoms)]

        charge_map = mapper.map_partial_charges(ethanol_mol_3d, charges_in, normalize=False)
        assert len(charge_map) == num_atoms
        for i in range(num_atoms):
            assert charge_map[i] == charges_in[i]

    def test_map_partial_charges_normalize(self, ethanol_mol_3d: Chem.Mol):
        """Test map_partial_charges with normalization."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        # Make charges that don't sum to formal charge (0 for ethanol)
        charges_in = [0.1 + (0.05 * i) for i in range(num_atoms)]

        charge_map = mapper.map_partial_charges(ethanol_mol_3d, charges_in, normalize=True)
        assert len(charge_map) == num_atoms

        formal_charge = Chem.GetFormalCharge(ethanol_mol_3d)
        assert abs(sum(charge_map.values()) - formal_charge) < 1e-6  # Check sum after normalization

    def test_map_partial_charges_mismatch(self, ethanol_mol_3d: Chem.Mol):
        """Test map_partial_charges with mismatched number of charges."""
        mapper = ForceFieldMapper()
        charges_in = [0.1]  # Not enough charges
        with pytest.raises(ValueError, match="doesn't match number of charges"):
            mapper.map_partial_charges(ethanol_mol_3d, charges_in)

    def test_assign_atom_types_ethanol(self, ethanol_mol_3d: Chem.Mol):
        """Test assign_atom_types for ethanol."""
        mapper = ForceFieldMapper()
        atom_types = mapper.assign_atom_types(ethanol_mol_3d)
        # CCOH: C(H3)-C(H2)-O-H
        assert len(atom_types) == ethanol_mol_3d.GetNumAtoms()
        # Example check for one atom (e.g. Oxygen)
        oxygen_idx = -1
        for atom in ethanol_mol_3d.GetAtoms():
            if atom.GetAtomicNum() == 8:  # Oxygen
                oxygen_idx = atom.GetIdx()
                break
        assert oxygen_idx != -1
        assert atom_types[oxygen_idx] == "oh"  # Hydroxyl oxygen

    def test_assign_atom_types_pfoa_frag(self, pfoa_frag_mol_3d: Chem.Mol):
        """Test assign_atom_types for PFOA fragment."""
        mapper = ForceFieldMapper()
        atom_types = mapper.assign_atom_types(pfoa_frag_mol_3d)
        # CF3COOH
        # Check a fluorine and the carbonyl carbon
        fluorine_idx = -1
        cooh_carbon_idx = -1
        for atom in pfoa_frag_mol_3d.GetAtoms():
            if atom.GetAtomicNum() == 9:
                fluorine_idx = atom.GetIdx()
            elif atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                # Assuming COOH carbon is the only SP2 carbon here
                is_cooh_c = False
                for n in atom.GetNeighbors():
                    if n.GetAtomicNum() == 8 and n.GetTotalNumHs() > 0:  # Bonded to OH
                        is_cooh_c = True
                    if n.GetAtomicNum() == 8 and n.GetExplicitValence() > 1:  # Bonded to =O
                        is_cooh_c = True
                if is_cooh_c:
                    cooh_carbon_idx = atom.GetIdx()

        assert fluorine_idx != -1
        assert atom_types[fluorine_idx] == "f"

        # The COOH carbon typing is 'c2' in the current simplified logic
        if cooh_carbon_idx != -1:  # May not always find it with simple logic
            assert atom_types[cooh_carbon_idx] == "c2"

    def test_predict_bond_parameters_3d(self, ethanol_mol_3d: Chem.Mol):
        """Test predict_bond_parameters with 3D coordinates."""
        mapper = ForceFieldMapper()
        atom_types = mapper.assign_atom_types(ethanol_mol_3d)
        bond_params = mapper.predict_bond_parameters(ethanol_mol_3d, atom_types)

        assert len(bond_params) == ethanol_mol_3d.GetNumBonds() * 2  # Stored for both directions
        first_bond = ethanol_mol_3d.GetBondWithIdx(0)
        idx1, idx2 = first_bond.GetBeginAtomIdx(), first_bond.GetEndAtomIdx()

        assert (idx1, idx2) in bond_params
        param_set = bond_params[(idx1, idx2)]
        assert "k" in param_set and isinstance(param_set["k"], float)
        assert "r_eq" in param_set and isinstance(param_set["r_eq"], float)
        assert param_set["r_eq"] > 0.5  # Sanity check for bond length

    def test_predict_bond_parameters_2d(self, ethanol_mol_2d: Chem.Mol):
        """Test predict_bond_parameters without 3D (estimation)."""
        mapper = ForceFieldMapper()
        atom_types = mapper.assign_atom_types(ethanol_mol_2d)
        bond_params = mapper.predict_bond_parameters(ethanol_mol_2d, atom_types)
        assert len(bond_params) == ethanol_mol_2d.GetNumBonds() * 2
        # Check that r_eq is estimated (not None, and reasonable)
        first_bond = ethanol_mol_2d.GetBondWithIdx(0)
        idx1, idx2 = first_bond.GetBeginAtomIdx(), first_bond.GetEndAtomIdx()
        assert bond_params[(idx1, idx2)]["r_eq"] is not None
        assert bond_params[(idx1, idx2)]["r_eq"] > 0.5

    def test_predict_angle_parameters_3d(self, ethanol_mol_3d: Chem.Mol):
        """Test predict_angle_parameters with 3D coordinates."""
        mapper = ForceFieldMapper()
        atom_types = mapper.assign_atom_types(ethanol_mol_3d)
        angle_params = mapper.predict_angle_parameters(ethanol_mol_3d, atom_types)
        assert len(angle_params) > 0  # Ethanol should have angles
        # Find an angle, e.g., C-C-O
        cco_angle_key = None
        for atom_j_idx in range(ethanol_mol_3d.GetNumAtoms()):
            atom_j = ethanol_mol_3d.GetAtomWithIdx(atom_j_idx)
            if atom_j.GetAtomicNum() == 6:  # Central C
                neighbors = atom_j.GetNeighbors()
                c_neighbor_idx = -1
                o_neighbor_idx = -1
                for n in neighbors:
                    if n.GetAtomicNum() == 6:
                        c_neighbor_idx = n.GetIdx()
                    if n.GetAtomicNum() == 8:
                        o_neighbor_idx = n.GetIdx()
                if c_neighbor_idx != -1 and o_neighbor_idx != -1:
                    cco_angle_key = (c_neighbor_idx, atom_j_idx, o_neighbor_idx)
                    break

        assert cco_angle_key is not None and cco_angle_key in angle_params
        param_set = angle_params[cco_angle_key]
        assert "k" in param_set
        assert "theta_eq" in param_set and isinstance(param_set["theta_eq"], float)
        assert 0 < param_set["theta_eq"] < 180  # Sanity check for angle

    def test_generate_force_field_parameters(self, ethanol_mol_3d: Chem.Mol):
        """Test the main generate_force_field_parameters method."""
        mapper = ForceFieldMapper()
        # Dummy MGNN predictions (only charges for this test)
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        # The method generate_force_field_parameters expects a List[float] for partial_charges
        mgnn_charges_list = [0.01 * i for i in range(num_atoms)]

        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=mgnn_charges_list)

        assert "atom_types" in ff_params
        assert "partial_charges" in ff_params
        assert "bonds" in ff_params
        assert "angles" in ff_params
        assert "dihedrals" in ff_params
        assert len(ff_params["partial_charges"]) == num_atoms
        assert len(ff_params["bonds"]) > 0
        assert len(ff_params["angles"]) > 0

    def test_export_to_openmm(self, ethanol_mol_3d: Chem.Mol, temp_output_dir: str):
        """Test OpenMM XML export functionality."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [0.0 for _ in range(num_atoms)]  # Neutral for simplicity
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)

        base_filename = "ethanol_test"
        success, output_paths = mapper.export_to_openmm(
            parameters=ff_params, mol=ethanol_mol_3d, output_dir=temp_output_dir, base_filename=base_filename
        )

        assert success is True, "export_to_openmm success flag should be True"
        assert "xml" in output_paths
        assert os.path.exists(output_paths["xml"])

        # Basic content check for XML
        with open(output_paths["xml"], "r") as f:
            content = f.read()
            assert '<?xml version="1.0" encoding="UTF-8"?>' in content
            assert "<ForceField>" in content
            assert "<AtomTypes>" in content
            assert "<Residues>" in content
            assert "<HarmonicBondForce>" in content
            assert "<HarmonicAngleForce>" in content
            assert "<PeriodicTorsionForce>" in content
            assert "<NonbondedForce" in content  # Changed from <NonbondedForce> to <NonbondedForce
            assert "coulomb14scale" in content  # Check for NonbondedForce attributes
            assert "lj14scale" in content  # Check for NonbondedForce attributes
            assert "<UseAttributeFromResidue" in content  # Check for charge attribute
            assert "<Atom type=" in content  # Check for atom type definitions

    def test_validate_parameters_valid(self, ethanol_mol_3d: Chem.Mol):
        """Test validate_parameters with good, default parameters."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [0.0] * num_atoms  # Neutral
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)

        validation_results = mapper.validate_parameters(parameters=ff_params, mol=ethanol_mol_3d)

        assert validation_results["charge_balance_ok"]
        assert validation_results["bonds_ok"]
        assert validation_results["angles_ok"]
        assert validation_results["dihedrals_ok"]

    def test_validate_parameters_invalid_charge(self, ethanol_mol_3d: Chem.Mol):
        """Test validate_parameters with imbalanced charges."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [1.0] * num_atoms  # Highly imbalanced
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)
        ff_params["partial_charges"] = {i: 1.0 for i in range(num_atoms)}

        validation_results = mapper.validate_parameters(parameters=ff_params, mol=ethanol_mol_3d)
        assert not validation_results["charge_balance_ok"]

    def test_validate_parameters_invalid_bond(self, ethanol_mol_3d: Chem.Mol):
        """Test validate_parameters with an invalid bond length."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [0.0] * num_atoms
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)

        # Find a bond and make its r_eq invalid
        if ff_params["bonds"]:
            first_bond_key = list(ff_params["bonds"].keys())[0]
            ff_params["bonds"][first_bond_key]["r_eq"] = 0.01  # Too short

        validation_results = mapper.validate_parameters(parameters=ff_params, mol=ethanol_mol_3d)
        assert not validation_results["bonds_ok"]

    def test_validate_parameters_invalid_angle(self, ethanol_mol_3d: Chem.Mol):
        """Test validate_parameters with an invalid angle."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [0.0] * num_atoms
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)

        if ff_params["angles"]:
            first_angle_key = list(ff_params["angles"].keys())[0]
            ff_params["angles"][first_angle_key]["theta_eq"] = 300.0  # Impossible angle

        validation_results = mapper.validate_parameters(parameters=ff_params, mol=ethanol_mol_3d)
        assert not validation_results["angles_ok"]

    def test_validate_parameters_invalid_dihedral(self, ethanol_mol_3d: Chem.Mol):
        """Test validate_parameters with an invalid dihedral energy."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        charges_for_ff_params = [0.0] * num_atoms
        ff_params = mapper.generate_force_field_parameters(ethanol_mol_3d, partial_charges=charges_for_ff_params)

        if ff_params["dihedrals"]:
            first_dihedral_key = list(ff_params["dihedrals"].keys())[0]
            # Make a dihedral term have very high energy
            if ff_params["dihedrals"][first_dihedral_key]:
                ff_params["dihedrals"][first_dihedral_key][0]["k"] = 100.0  # kcal/mol, very high

        validation_results = mapper.validate_parameters(parameters=ff_params, mol=ethanol_mol_3d)
        assert not validation_results["dihedrals_ok"]

    def test_convert_mgnn_predictions_to_force_field(self, ethanol_mol_3d: Chem.Mol, temp_output_dir: str):
        """Test the main conversion entry point."""
        mapper = ForceFieldMapper()
        num_atoms = ethanol_mol_3d.GetNumAtoms()
        # Example MGNN output structure (assuming node_pred are charges)
        mgnn_predictions = {
            "node_pred": torch.tensor([0.01 * i for i in range(num_atoms)], dtype=torch.float)  # Changed to 1D tensor
        }

        success, results = mapper.convert_mgnn_predictions_to_force_field(
            mol=ethanol_mol_3d,
            node_predictions=mgnn_predictions,
            output_dir=temp_output_dir,
            base_filename="ethanol"
        )

        assert success is True
        assert "parameters" in results
        assert "validation" in results
        assert "file_paths" in results
        assert "xml" in results["file_paths"]
        assert os.path.exists(results["file_paths"]["xml"])
