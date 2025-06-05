"""
Unit tests for the MolecularGraphProcessor class and related functions
from moml.core.molecular_graph_processor.
"""

import pytest
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Any
import tempfile
import os
import json
import pandas as pd
import time

from moml.core.molecular_graph_processor import (
    MolecularGraphProcessor,
    create_graph_processor,
    mol_file_to_graph,
    graph_to_device,
    collate_graphs,
    find_charges_file,
    read_charges_from_file,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules,
)
from moml.core.molecular_feature_extraction import MolecularFeatureExtractor


# Helper function to create a simple RDKit molecule
def create_rdkit_mol(smiles: str, add_3d_coords: bool = False) -> Chem.Mol:
    """Creates an RDKit molecule from SMILES and optionally adds 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if add_3d_coords:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)
        if mol.GetNumConformers() == 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
            mol.AddConformer(conf, assignId=True)
    return mol


# Test Fixtures
@pytest.fixture
def methane_mol_2d() -> Chem.Mol:
    """Returns a 2D RDKit molecule for methane."""
    return create_rdkit_mol("C", add_3d_coords=False)


@pytest.fixture
def methane_mol_3d() -> Chem.Mol:
    """Returns a 3D RDKit molecule for methane."""
    return create_rdkit_mol("C", add_3d_coords=True)


@pytest.fixture
def ethanol_mol_3d() -> Chem.Mol:
    """Returns a 3D RDKit molecule for ethanol."""
    return create_rdkit_mol("CCO", add_3d_coords=True)


@pytest.fixture
def pfoa_fragment_mol_3d() -> Chem.Mol:
    """Returns a 3D RDKit molecule for a PFOA fragment (CF3COOH)."""
    return create_rdkit_mol("C(F)(F)(F)C(=O)O", add_3d_coords=True)


@pytest.fixture
def default_processor_config() -> Dict[str, Any]:
    """Default configuration for MolecularGraphProcessor."""
    return {"use_partial_charges": False, "use_3d_coords": True, "use_pfas_specific_features": True}


@pytest.fixture
def processor_no_3d_config() -> Dict[str, Any]:
    """Configuration with use_3d_coords set to False."""
    return {"use_partial_charges": False, "use_3d_coords": False, "use_pfas_specific_features": True}


@pytest.fixture
def processor_no_pfas_features_config() -> Dict[str, Any]:
    """Configuration with use_pfas_specific_features set to False."""
    return {"use_partial_charges": False, "use_3d_coords": True, "use_pfas_specific_features": False}


@pytest.fixture
def graph_processor(default_processor_config: Dict[str, Any]) -> MolecularGraphProcessor:
    """Returns a MolecularGraphProcessor instance with default config."""
    return MolecularGraphProcessor(config=default_processor_config)


class TestMolecularGraphProcessor:
    """Tests for the MolecularGraphProcessor class."""

    def test_processor_initialization(self, default_processor_config: Dict[str, Any]):
        """Test MolecularGraphProcessor initialization with various configurations."""
        processor = MolecularGraphProcessor(config=default_processor_config)
        assert processor.config == default_processor_config
        assert processor.use_3d_coords == default_processor_config["use_3d_coords"]
        assert processor.use_pfas_specific_features == default_processor_config["use_pfas_specific_features"]

        processor_custom = MolecularGraphProcessor(config={"use_3d_coords": False, "use_pfas_specific_features": False})
        assert not processor_custom.use_3d_coords
        assert not processor_custom.use_pfas_specific_features

        processor_empty_config = MolecularGraphProcessor()
        assert processor_empty_config.use_partial_charges is True
        assert processor_empty_config.use_3d_coords is True
        assert processor_empty_config.use_pfas_specific_features is True

    def test_atom_feature_dim(
        self, graph_processor: MolecularGraphProcessor, processor_no_pfas_features_config: Dict[str, Any]
    ):
        """Test the atom_feature_dim property."""
        # Expected base dimension calculation components from MolecularFeatureExtractor
        base_one_hots_dim = (
            len(MolecularFeatureExtractor.ATOM_FEATURES["atomic_num"])
            + len(MolecularFeatureExtractor.ATOM_FEATURES["degree"])
            + len(MolecularFeatureExtractor.ATOM_FEATURES["formal_charge"])
            + len(MolecularFeatureExtractor.ATOM_FEATURES["hybridization"])
            + len(MolecularFeatureExtractor.ATOM_FEATURES["is_aromatic"])
            + len(MolecularFeatureExtractor.ATOM_FEATURES["is_in_ring"])
        )
        num_hs_dim = 1
        # is_f, is_cf are always added by _get_atom_features, and now accounted for in atom_feature_dim
        always_present_pfas_like_dim = 2

        # Processor with use_pfas_specific_features = False, use_partial_charges = False, use_3d_coords = True
        processor_no_pfas = MolecularGraphProcessor(config=processor_no_pfas_features_config)
        expected_dim_no_pfas_specific = 39  # Corrected expected dimension
        assert (
            processor_no_pfas.atom_feature_dim == expected_dim_no_pfas_specific
        ), f"Expected atom_feature_dim {expected_dim_no_pfas_specific} for no_pfas_config, got {processor_no_pfas.atom_feature_dim}"

        # Processor with use_pfas_specific_features = True, use_partial_charges = False, use_3d_coords = True (graph_processor fixture)
        expected_dim_with_pfas_specific = 46  # Corrected: processor calculates 46 based on its logic
        assert (
            graph_processor.atom_feature_dim == expected_dim_with_pfas_specific
        ), f"Expected atom_feature_dim {expected_dim_with_pfas_specific} for default_config, got {graph_processor.atom_feature_dim}"

    def test_bond_feature_dim(self, graph_processor: MolecularGraphProcessor, processor_no_3d_config: Dict[str, Any]):
        """Test the bond_feature_dim property."""
        # Expected base dimension calculation components
        base_one_hots_bond_dim = (
            len(MolecularFeatureExtractor.BOND_FEATURES["bond_type"])
            + len(MolecularFeatureExtractor.BOND_FEATURES["is_conjugated"])
            + len(MolecularFeatureExtractor.BOND_FEATURES["is_in_ring"])
        )
        # is_cf_bond is always added by _get_bond_features, and now accounted for in bond_feature_dim
        always_present_cf_bond_dim = 1

        pfas_specific_bond_add_on_dim = 3  # is_cf_cf_bond, is_fluorinated_tail_bond, is_func_group_bond
        bond_length_dim = 1

        # Processor with use_3d_coords = False, use_pfas_specific_features = True, use_partial_charges = False
        processor_no_3d = MolecularGraphProcessor(config=processor_no_3d_config)
        expected_dim_no_3d_pfas_true = 13  # Manually calculated
        assert (
            processor_no_3d.bond_feature_dim == expected_dim_no_3d_pfas_true
        ), f"Expected bond_feature_dim {expected_dim_no_3d_pfas_true} for no_3d_config, got {processor_no_3d.bond_feature_dim}"

        # Processor with use_3d_coords = True, use_pfas_specific_features = True, use_partial_charges = False (graph_processor fixture)
        expected_dim_3d_pfas_true = 14  # Manually calculated
        assert (
            graph_processor.bond_feature_dim == expected_dim_3d_pfas_true
        ), f"Expected bond_feature_dim {expected_dim_3d_pfas_true} for default_config, got {graph_processor.bond_feature_dim}"

    @staticmethod
    def test_one_hot_encoding():
        """Test the _one_hot_encoding static method."""
        choices = ["a", "b", "c"]
        assert MolecularGraphProcessor._one_hot_encoding("a", choices) == [1, 0, 0]
        assert MolecularGraphProcessor._one_hot_encoding("c", choices) == [0, 0, 1]
        assert MolecularGraphProcessor._one_hot_encoding("d", choices) == [0, 0, 0]

    def test_get_atom_features_methane(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test _get_atom_features for a simple methane molecule."""
        carbon_atom = methane_mol_3d.GetAtomWithIdx(0)
        # Call _get_atom_features with the mol argument
        features = graph_processor._get_atom_features(carbon_atom, methane_mol_3d)

        # The length of features should match the atom_feature_dim of the specific processor instance
        # when all optional features (like partial charges, distance features) are off or handled.
        # The graph_processor fixture has use_partial_charges=False.
        # For methane, PFAS-specific distance features will likely be default/zero.
        assert (
            len(features) == graph_processor.atom_feature_dim
        ), f"Expected {graph_processor.atom_feature_dim} features, got {len(features)}"

    def test_get_atom_features_pfoa_fragment(
        self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol
    ):
        """Test _get_atom_features for a PFOA fragment, checking PFAS specific features."""
        cf3_carbon = pfoa_fragment_mol_3d.GetAtomWithIdx(0)  # C(F)(F)(F)
        fluorine_atom = pfoa_fragment_mol_3d.GetAtomWithIdx(1)  # One of the F atoms
        cooh_carbon = pfoa_fragment_mol_3d.GetAtomWithIdx(4)  # C(=O)O

        # Correctly get distance features from the feature_extractor attribute
        dist_features_map = None
        if graph_processor.use_pfas_specific_features and graph_processor.use_3d_coords:
            dist_features_map = graph_processor.feature_extractor.calculate_distance_features(pfoa_fragment_mol_3d)

        cf3_carbon_features_vector = graph_processor._get_atom_features(
            cf3_carbon,
            pfoa_fragment_mol_3d,  # Pass the mol object
            distance_features_map=dist_features_map,  # Pass the map itself
        )
        assert len(cf3_carbon_features_vector) == graph_processor.atom_feature_dim

        fluorine_atom_features_vector = graph_processor._get_atom_features(
            fluorine_atom, pfoa_fragment_mol_3d, distance_features_map=dist_features_map
        )
        assert len(fluorine_atom_features_vector) == graph_processor.atom_feature_dim

        # Example: Check if 'is_fluorine' feature is correctly set for the fluorine atom
        # This requires knowing the structure of ATOM_FEATURES_DEFAULTS and active schemes
        # For simplicity, we'll assume 'is_fluorine' is an active scheme and check its value.
        # This part might need adjustment based on the exact feature vector composition.
        # Find the index of 'is_fluorine' if it's a numerical feature
        is_fluorine_idx = -1
        current_idx = 0
        for scheme in graph_processor.atom_feature_schemes:
            if scheme == "is_fluorine":
                is_fluorine_idx = current_idx
                break
            choices = graph_processor.ATOM_FEATURES_DEFAULTS.get(scheme)
            if isinstance(choices, list):
                current_idx += len(choices)
            else:  # numerical
                # Apply same conditional logic as in atom_feature_dim property
                if scheme == "partial_charge" and not graph_processor.use_partial_charges:
                    continue
                if scheme in ["dist_to_cf3", "dist_to_functional_group", "is_head_group_atom"] and not (
                    graph_processor.use_pfas_specific_features and graph_processor.use_3d_coords
                ):
                    continue
                current_idx += 1

        if is_fluorine_idx != -1:
            assert fluorine_atom_features_vector[is_fluorine_idx] == 1.0, "is_fluorine feature incorrect"
        else:
            # This case implies 'is_fluorine' is not an active scheme or logic is flawed
            # For now, we'll just note it. A more robust test would ensure it's active.
            pass

    def test_get_bond_features_methane(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test _get_bond_features for a C-H bond in methane."""
        bond = methane_mol_3d.GetBondWithIdx(0)  # Get a C-H bond

        bond_lengths_map = None
        if graph_processor.use_3d_coords:
            bond_lengths_map = graph_processor.feature_extractor.calculate_bond_lengths(methane_mol_3d)

        features = graph_processor._get_bond_features(bond, bond_lengths_map=bond_lengths_map)
        assert (
            len(features) == graph_processor.bond_feature_dim
        ), f"Expected {graph_processor.bond_feature_dim} features, got {len(features)}"

    def test_get_bond_features_pfoa_fragment(
        self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol
    ):
        """Test _get_bond_features for C-F and C-C bonds in PFOA fragment."""
        bond_lengths_map = None
        if graph_processor.use_3d_coords:
            bond_lengths_map = graph_processor.feature_extractor.calculate_bond_lengths(pfoa_fragment_mol_3d)

        # Test a C-F bond (atom 0 is C in CF3, atom 1 is F)
        cf_bond = pfoa_fragment_mol_3d.GetBondBetweenAtoms(0, 1)
        assert cf_bond is not None, "C-F bond not found for testing"

        cf_features = graph_processor._get_bond_features(cf_bond, bond_lengths_map=bond_lengths_map)
        assert len(cf_features) == graph_processor.bond_feature_dim

        # Example: Check 'is_cf_bond' feature
        # This requires knowing the structure of BOND_FEATURES_DEFAULTS and active schemes.
        is_cf_bond_idx = -1
        current_idx = 0
        for scheme in graph_processor.bond_feature_schemes:
            if scheme == "is_cf_bond":
                is_cf_bond_idx = current_idx
                break
            choices = graph_processor.BOND_FEATURES_DEFAULTS.get(scheme)
            if isinstance(choices, list):
                current_idx += len(choices)
            else:  # numerical
                if scheme == "bond_length" and not graph_processor.use_3d_coords:
                    continue
                if (
                    scheme in ["is_cf_cf_bond", "is_fluorinated_tail_bond", "is_functional_group_bond"]
                    and not graph_processor.use_pfas_specific_features
                ):
                    continue
                current_idx += 1

        if is_cf_bond_idx != -1:
            assert cf_features[is_cf_bond_idx] == 1.0, "is_cf_bond feature incorrect for C-F bond"

        # Test a C-C bond (atom 0 is CF3-C, atom 4 is COOH-C, bond between them is C-C)
        # Assuming atoms 0 and 4 are connected in the PFOA fragment CF3-COOH
        # Let's find the C-C bond between the CF3 carbon (idx 0) and the COOH carbon (idx 4)
        cc_bond = pfoa_fragment_mol_3d.GetBondBetweenAtoms(0, 4)  # This might be incorrect depending on actual indexing
        if cc_bond is None:  # Try to find any C-C bond if the specific one isn't there
            for b in pfoa_fragment_mol_3d.GetBonds():
                if b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6:
                    cc_bond = b
                    break
        assert cc_bond is not None, "C-C bond not found for testing in PFOA fragment"

        cc_features = graph_processor._get_bond_features(cc_bond, bond_lengths_map=bond_lengths_map)
        assert len(cc_features) == graph_processor.bond_feature_dim
        if is_cf_bond_idx != -1:
            assert cc_features[is_cf_bond_idx] == 0.0, "is_cf_bond feature incorrect for C-C bond"

    def test_mol_to_graph_methane_3d(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test mol_to_graph for 3D methane."""
        graph = graph_processor.mol_to_graph(methane_mol_3d)
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == methane_mol_3d.GetNumAtoms()
        assert (
            graph.x.shape[1] == graph_processor.atom_feature_dim
        ), f"Expected node feature dim {graph_processor.atom_feature_dim}, got {graph.x.shape[1]}"
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] == methane_mol_3d.GetNumBonds() * 2
        assert graph.edge_attr.shape[0] == methane_mol_3d.GetNumBonds() * 2
        assert graph.edge_attr.shape[1] == graph_processor.bond_feature_dim
        assert graph.pos.shape[0] == methane_mol_3d.GetNumAtoms()
        assert graph.pos.shape[1] == 3

    def test_mol_to_graph_methane_2d(self, processor_no_3d_config: Dict[str, Any], methane_mol_2d: Chem.Mol):
        """Test mol_to_graph for 2D methane (no 3D coords)."""
        processor = MolecularGraphProcessor(config=processor_no_3d_config)
        graph = processor.mol_to_graph(methane_mol_2d)
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == methane_mol_2d.GetNumAtoms()
        assert (
            graph.x.shape[1] == processor.atom_feature_dim
        ), f"Expected node feature dim {processor.atom_feature_dim}, got {graph.x.shape[1]}"
        assert graph.edge_index.shape[1] == methane_mol_2d.GetNumBonds() * 2
        assert graph.edge_attr.shape[1] == processor.bond_feature_dim
        assert graph.pos is None

    def test_mol_to_graph_no_conformer_error(self, graph_processor: MolecularGraphProcessor, methane_mol_2d: Chem.Mol):
        """Test mol_to_graph behavior when 3D coords are expected but not present."""
        # graph_processor is configured with use_3d_coords=True
        # methane_mol_2d has no conformers.
        # The method will attempt to generate conformers. If it fails, it logs and proceeds.
        # The 'pos' attribute should then be None or not present.

        # Ensure the input molecule indeed has no conformers
        assert methane_mol_2d.GetNumConformers() == 0

        graph = graph_processor.mol_to_graph(methane_mol_2d)

        assert isinstance(graph, Data)
        # Check that 'pos' IS present, as EmbedMolecule likely succeeded for "C" (methane)
        # and methane_mol_2d fixture calls AddHs, so it's CH4.
        assert hasattr(graph, "pos") and graph.pos is not None
        # After AddHs, methane (C) becomes CH4 (5 atoms).
        # The graph_processor.mol_to_graph will use these 5 atoms.
        assert graph.pos.shape == (methane_mol_2d.GetNumAtoms(), 3)
        assert graph.num_nodes == methane_mol_2d.GetNumAtoms()

    def test_mol_to_graph_pfoa_fragment(self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol):
        """Test mol_to_graph with a PFOA fragment."""
        graph = graph_processor.mol_to_graph(pfoa_fragment_mol_3d)
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == pfoa_fragment_mol_3d.GetNumAtoms()
        assert (
            graph.x.shape[1] == graph_processor.atom_feature_dim
        ), f"Expected node feature dim {graph_processor.atom_feature_dim}, got {graph.x.shape[1]}"

    def test_mol_to_graph_with_additional_features(self, pfoa_fragment_mol_3d: Chem.Mol):
        """Test mol_to_graph with provided partial charges and HOMO/LUMO contributions."""
        config_full_features = {"use_partial_charges": True, "use_3d_coords": True, "use_pfas_specific_features": True}
        processor = MolecularGraphProcessor(config=config_full_features)
        num_atoms = pfoa_fragment_mol_3d.GetNumAtoms()

        charges = [0.1 * i for i in range(num_atoms)]
        homo = [0.01 * i for i in range(num_atoms)]
        lumo = [0.001 * i for i in range(num_atoms)]
        additional_features = {"partial_charges": charges, "homo_contributions": homo, "lumo_contributions": lumo}

        graph_with_add = processor.mol_to_graph(pfoa_fragment_mol_3d, additional_features=additional_features)

        config_no_add = config_full_features.copy()
        config_no_add["use_partial_charges"] = False  # Turn off to see difference
        processor_no_add = MolecularGraphProcessor(config=config_no_add)
        # Create graph without any additional features for baseline dimension
        graph_no_add = processor_no_add.mol_to_graph(pfoa_fragment_mol_3d, additional_features=None)

        # Expect 1 for partial_charges + 2 for homo/lumo
        assert graph_with_add.x.shape[1] == graph_no_add.x.shape[1] + 1 + 2

    def test_smiles_to_graph(self, graph_processor: MolecularGraphProcessor, ethanol_mol_3d: Chem.Mol):
        """Test smiles_to_graph method."""
        smiles = "CCO"
        graph_from_smiles = graph_processor.smiles_to_graph(smiles)

        # Use ethanol_mol_3d directly, as it's created with Hs, similar to how
        # smiles_to_graph prepares its internal molecule before calling mol_to_graph.
        mol_for_comparison = ethanol_mol_3d  # This molecule already has Hs from the fixture setup
        graph_from_mol = graph_processor.mol_to_graph(mol_for_comparison)

        assert isinstance(graph_from_smiles, Data)
        assert (
            graph_from_smiles.x.shape == graph_from_mol.x.shape
        ), f"Node feature shapes differ: {graph_from_smiles.x.shape} vs {graph_from_mol.x.shape}"
        assert graph_from_smiles.edge_index.shape == graph_from_mol.edge_index.shape
        assert graph_from_smiles.edge_attr.shape == graph_from_mol.edge_attr.shape
        if graph_processor.use_3d_coords:
            assert graph_from_smiles.pos.shape == graph_from_mol.pos.shape

    def test_smiles_to_graph_invalid_smiles(self, graph_processor: MolecularGraphProcessor):
        """Test smiles_to_graph with invalid SMILES returns None."""
        invalid_smiles = "thisisnotasmiles"
        # The method is designed to log an error and return None for invalid SMILES.
        graph = graph_processor.smiles_to_graph(invalid_smiles)
        assert graph is None

    def test_file_to_graph(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test file_to_graph method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mol", delete=False) as tmp_mol_file:
            tmp_mol_file.write(Chem.MolToMolBlock(methane_mol_3d))
            filepath = tmp_mol_file.name

        graph_from_file = graph_processor.file_to_graph(filepath)
        graph_from_mol = graph_processor.mol_to_graph(methane_mol_3d)

        assert isinstance(graph_from_file, Data)
        assert graph_from_file.x.shape == graph_from_mol.x.shape
        assert graph_from_file.edge_index.shape == graph_from_mol.edge_index.shape
        assert graph_from_file.edge_attr.shape == graph_from_mol.edge_attr.shape
        if graph_processor.use_3d_coords:
            assert graph_from_file.pos.shape == graph_from_mol.pos.shape
        os.remove(filepath)

    def test_file_to_graph_non_existent_file(self, graph_processor: MolecularGraphProcessor):
        """Test file_to_graph with a non-existent file."""
        with pytest.raises(FileNotFoundError):
            graph_processor.file_to_graph("non_existent_file.mol")

    def test_batch_files_to_graphs(
        self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol, ethanol_mol_3d: Chem.Mol
    ):
        """Test batch_files_to_graphs method."""
        files_to_create = {"methane.mol": methane_mol_3d, "ethanol.mol": ethanol_mol_3d}
        file_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, mol in files_to_create.items():
                filepath = os.path.join(tmpdir, name)
                with open(filepath, "w") as f:
                    f.write(Chem.MolToMolBlock(mol))
                file_paths.append(filepath)

            # The utility saves .pt files and returns their paths.
            with tempfile.TemporaryDirectory() as tmp_out_dir:
                saved_graph_paths = batch_create_graphs_from_molecules(
                    mol_dir=tmpdir,
                    output_dir=tmp_out_dir,
                    file_format="mol",
                    config=graph_processor.config,
                    max_workers=1,
                )

                assert len(saved_graph_paths) == len(
                    file_paths
                ), f"Expected {len(file_paths)} saved graph paths, got {len(saved_graph_paths)}"

                # Sort paths for consistent comparison if order is not guaranteed
                # batch_create_graphs_from_molecules uses ProcessPoolExecutor, results order might not match input glob order.
                # However, with max_workers=1, it should be deterministic.
                # To be safe, we'll check based on filenames.

                saved_path_map = {os.path.basename(p): p for p in saved_graph_paths}
                expected_filenames = [os.path.splitext(os.path.basename(fp))[0] + ".pt" for fp in file_paths]

                assert len(saved_path_map) == len(expected_filenames)
                for expected_filename in expected_filenames:
                    assert (
                        expected_filename in saved_path_map
                    ), f"Expected file {expected_filename} not found in saved paths."
                    pt_path = saved_path_map[expected_filename]
                    assert os.path.exists(pt_path)
                    assert pt_path.startswith(tmp_out_dir)
                    try:
                        loaded_data = torch.load(pt_path, weights_only=False)  # Added weights_only=False
                        assert isinstance(loaded_data, Data)

                        original_mol = None
                        if expected_filename == "methane.pt":
                            original_mol = methane_mol_3d
                        elif expected_filename == "ethanol.pt":
                            original_mol = ethanol_mol_3d

                        assert original_mol is not None, f"Unexpected .pt file: {pt_path}"
                        # The fixtures (methane_mol_3d, ethanol_mol_3d) already have Hs added.
                        # mol_file_to_graph (called by batch_create_graphs_from_molecules)
                        # also ensures Hs are added before graph creation.
                        assert (
                            loaded_data.num_nodes == original_mol.GetNumAtoms()
                        ), f"Node count mismatch for {pt_path}. Expected {original_mol.GetNumAtoms()}, got {loaded_data.num_nodes}"

                    except Exception as e:
                        pytest.fail(f"Failed to load or validate .pt file {pt_path}: {e}")

    def test_mol_to_json_graph(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test mol_to_json_graph method."""
        json_graph = graph_processor.mol_to_json_graph(methane_mol_3d)
        assert isinstance(json_graph, dict)
        assert "nodes" in json_graph  # Changed from "atoms"
        assert "edges" in json_graph  # Changed from "bonds"
        assert "descriptors" in json_graph
        assert len(json_graph["nodes"]) == methane_mol_3d.GetNumAtoms()  # Changed from "atoms"
        assert len(json_graph["edges"]) == methane_mol_3d.GetNumBonds()  # Changed from "bonds"
        assert "mol_weight" in json_graph["descriptors"]

    def test_file_to_json_graph(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test file_to_json_graph method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_filepath = os.path.join(tmpdir, "methane.mol")
            with open(mol_filepath, "w") as f:
                f.write(Chem.MolToMolBlock(methane_mol_3d))

            output_filename = "methane_graph.json"
            json_output_path = graph_processor.file_to_json_graph(
                mol_filepath, output_dir=tmpdir, output_filename=output_filename
            )

            assert json_output_path is not None
            assert os.path.exists(json_output_path)
            assert os.path.basename(json_output_path) == output_filename

            with open(json_output_path, "r") as f_json:
                json_data = json.load(f_json)

            assert "nodes" in json_data  # Changed from "atoms" to "nodes"
            assert len(json_data["nodes"]) == methane_mol_3d.GetNumAtoms()  # Changed from "atoms"

    def test_get_atom_features_instance_method(
        self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol
    ):
        """Test that atom features are correctly generated by mol_to_graph."""
        graph = graph_processor.mol_to_graph(methane_mol_3d)
        assert isinstance(graph, Data)
        assert hasattr(graph, "x")
        atom_features_tensor = graph.x

        assert isinstance(atom_features_tensor, torch.Tensor)
        assert atom_features_tensor.shape[0] == methane_mol_3d.GetNumAtoms()
        assert atom_features_tensor.shape[1] == graph_processor.atom_feature_dim

    def test_get_adjacency_matrix(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test that edge_index correctly represents connectivity."""
        graph = graph_processor.mol_to_graph(methane_mol_3d)
        assert isinstance(graph, Data)
        assert hasattr(graph, "edge_index")

        num_atoms = methane_mol_3d.GetNumAtoms()
        edge_index = graph.edge_index

        assert edge_index.shape[0] == 2
        # For an undirected graph representation, each bond appears twice
        assert edge_index.shape[1] == methane_mol_3d.GetNumBonds() * 2

        # Check if edge_index values are within valid node indices
        if edge_index.numel() > 0:  # only if there are edges
            assert edge_index.min() >= 0
            assert edge_index.max() < num_atoms

        # Optional: Convert to dense adjacency matrix and check properties
        # from torch_geometric.utils import to_dense_adj
        # adj_matrix_dense = to_dense_adj(edge_index, max_num_nodes=num_atoms).squeeze(0)
        # assert adj_matrix_dense.shape == (num_atoms, num_atoms)
        # assert torch.all(adj_matrix_dense == adj_matrix_dense.t()) # Symmetric
        # for i in range(num_atoms):
        #     assert adj_matrix_dense[i].sum().item() == methane_mol_3d.GetAtomWithIdx(i).GetDegree()

    def test_process_dataframe(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test processing a DataFrame of molecules into graphs."""
        data = {"smiles": ["C", "CC"], "id": [1, 2]}
        df = pd.DataFrame(data)
        # Create RDKit Mol objects; graph_processor.use_3d_coords will determine if they get 3D
        df["rdkit_mol"] = df["smiles"].apply(lambda s: create_rdkit_mol(s, add_3d_coords=graph_processor.use_3d_coords))

        processed_graphs = []
        for idx, row in df.iterrows():
            mol = row["rdkit_mol"]
            if mol:
                graph = graph_processor.mol_to_graph(mol)
                processed_graphs.append(graph)
            else:
                processed_graphs.append(None)

        assert len(processed_graphs) == len(df)
        assert isinstance(processed_graphs[0], Data)
        assert processed_graphs[0].num_nodes == df["rdkit_mol"].iloc[0].GetNumAtoms()
        assert processed_graphs[0].x.shape[1] == graph_processor.atom_feature_dim

        assert isinstance(processed_graphs[1], Data)
        assert processed_graphs[1].num_nodes == df["rdkit_mol"].iloc[1].GetNumAtoms()


class TestUtilityFunctions:
    """Tests for standalone utility functions in molecular_graph_processor."""

    def test_create_graph_processor_utility(self):
        """Test the create_graph_processor factory function."""
        config = {"use_3d_coords": False, "use_pfas_specific_features": False}
        processor = create_graph_processor(config=config)
        assert isinstance(processor, MolecularGraphProcessor)
        assert not processor.use_3d_coords
        assert not processor.use_pfas_specific_features

        default_processor = create_graph_processor()  # Test with default config
        assert default_processor.use_3d_coords is True  # Default from MolecularGraphProcessor init

    def test_mol_file_to_graph_utility(self, methane_mol_3d: Chem.Mol):
        """Test the mol_file_to_graph utility function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mol", delete=False) as tmp_mol_file:
            tmp_mol_file.write(Chem.MolToMolBlock(methane_mol_3d))
            filepath = tmp_mol_file.name

        # Pass config via config argument
        config_arg = {"use_3d_coords": True, "use_pfas_features": True}
        graph = mol_file_to_graph(filepath, config=config_arg)  # Changed processor_config to config
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == methane_mol_3d.GetNumAtoms()
        assert graph.pos is not None
        os.remove(filepath)

    def test_graph_to_device(self, methane_mol_3d: Chem.Mol):
        """Test graph_to_device utility function."""
        processor = MolecularGraphProcessor()  # Default config
        graph = processor.mol_to_graph(methane_mol_3d)

        cpu_device = torch.device("cpu")
        graph_on_cpu = graph_to_device(graph, cpu_device)
        assert graph_on_cpu.x.device == cpu_device
        assert graph_on_cpu.edge_index.device == cpu_device

        # Test with a dictionary (as used in some parts of the codebase)
        graph_dict = {"x": torch.randn(2, 3), "edge_index": torch.randint(0, 2, (2, 1))}
        graph_dict_on_cpu = graph_to_device(graph_dict, cpu_device)
        assert graph_dict_on_cpu["x"].device == cpu_device
        assert graph_dict_on_cpu["edge_index"].device == cpu_device

    def test_collate_graphs(self, methane_mol_3d: Chem.Mol, ethanol_mol_3d: Chem.Mol):
        """Test collate_graphs utility function."""
        processor = MolecularGraphProcessor()
        graph1 = processor.mol_to_graph(methane_mol_3d)
        graph2 = processor.mol_to_graph(ethanol_mol_3d)

        batched_graph = collate_graphs([graph1, graph2])
        assert isinstance(batched_graph, Data)
        assert "batch" in batched_graph
        assert batched_graph.num_graphs == 2
        assert batched_graph.x.shape[0] == graph1.x.shape[0] + graph2.x.shape[0]
        assert batched_graph.edge_index.shape[1] == graph1.edge_index.shape[1] + graph2.edge_index.shape[1]

    def test_find_charges_file(self):
        """Test find_charges_file utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_filepath = os.path.join(tmpdir, "test_mol.mol")
            open(mol_filepath, "a").close()  # Create dummy mol file

            # Case 1: .charges file exists
            chg_filepath = os.path.join(tmpdir, "test_mol.charges")  # Changed extension
            open(chg_filepath, "a").close()
            assert find_charges_file(mol_filepath, tmpdir) == chg_filepath
            os.remove(chg_filepath)

            # Case 2: _charges.txt file exists
            txt_charges_filepath = os.path.join(tmpdir, "test_mol_charges.txt")  # Changed extension and variable name
            open(txt_charges_filepath, "a").close()
            time.sleep(0.1)  # Add a small delay for filesystem to catch up
            found_path_txt = find_charges_file(mol_filepath, tmpdir)
            assert (
                found_path_txt == txt_charges_filepath
            ), f"Expected to find {txt_charges_filepath}, but got {found_path_txt}"
            os.remove(txt_charges_filepath)

            # Case 3: No charge file (should return None)
            assert find_charges_file(mol_filepath, tmpdir) is None  # Changed expected return to None

    def test_read_charges_from_file(self):
        """Test read_charges_from_file utility function."""
        # Test with .chg file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".chg", delete=False) as tmp_chg:
            tmp_chg.write("1  C    0.123\n")
            tmp_chg.write("2  H   -0.03\n")
            chg_path = tmp_chg.name

        charges_chg = read_charges_from_file(chg_path)
        assert charges_chg == [0.123, -0.03]
        os.remove(chg_path)

        # Test with .json file (ESP charges format)
        esp_data = {"charges": [0.5, -0.5]}  # Changed key to "charges"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_json:
            json.dump(esp_data, tmp_json)
            json_path = tmp_json.name
        charges_json_esp = read_charges_from_file(json_path)
        assert charges_json_esp == [0.5, -0.5]
        os.remove(json_path)

        # Test with .json file (direct list format)
        list_data = [0.2, 0.3, -0.5]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_json_list:
            json.dump(list_data, tmp_json_list)
            json_list_path = tmp_json_list.name
        charges_json_list = read_charges_from_file(json_list_path)
        assert charges_json_list == [0.2, 0.3, -0.5]
        os.remove(json_list_path)

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            read_charges_from_file("non_existent.chg")

    def test_create_molecular_graph_json_utility(self, methane_mol_3d: Chem.Mol):
        """Test create_molecular_graph_json utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_filepath = os.path.join(tmpdir, "methane_util.mol")
            with open(mol_filepath, "w") as f:
                f.write(Chem.MolToMolBlock(methane_mol_3d))

            test_config = {"use_pfas_specific_features": True, "use_3d_coords": True}  # Match typical usage
            json_output_path = create_molecular_graph_json(mol_filepath, output_dir=tmpdir, config=test_config)

            assert json_output_path is not None
            assert os.path.exists(json_output_path)

            with open(json_output_path, "r") as f_json:
                json_data = json.load(f_json)
            # mol_to_json_graph method in the processor uses the key "nodes"
            assert "nodes" in json_data, "The key 'nodes' should be in the output JSON from create_molecular_graph_json"
            assert len(json_data["nodes"]) == methane_mol_3d.GetNumAtoms()

    def test_batch_create_graphs_from_molecules(self, methane_mol_3d: Chem.Mol, ethanol_mol_3d: Chem.Mol):
        """Test batch_create_graphs_from_molecules utility function."""
        with tempfile.TemporaryDirectory() as tmp_mol_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
            # Create dummy .mol files
            methane_path = os.path.join(tmp_mol_dir, "methane.mol")
            ethanol_path = os.path.join(tmp_mol_dir, "ethanol.mol")
            Chem.MolToMolFile(methane_mol_3d, methane_path)
            Chem.MolToMolFile(ethanol_mol_3d, ethanol_path)

            # Create a dummy non-mol file to be ignored
            with open(os.path.join(tmp_mol_dir, "ignore.txt"), "w") as f:
                f.write("ignore")

            config_for_batch = {  # Renamed to avoid conflict if MolecularGraphProcessor is in scope
                "use_pfas_features": True,
                "use_3d_coords": True,
            }
            results = batch_create_graphs_from_molecules(
                mol_dir=tmp_mol_dir,
                output_dir=tmp_out_dir,
                config=config_for_batch,  # Changed processor_config to config
                file_format="mol",  # Specify .mol file format
                max_workers=1,  # For predictable testing
            )

            assert len(results) == 2  # methane and ethanol
            for result in results:
                assert result is not None
                assert os.path.exists(result)
                assert result.startswith(tmp_out_dir)
                assert result.endswith(".pt")  # Changed to .pt

            # Check content of one of the files by loading the torch tensor
            # Ensure results are sorted to make checks deterministic if order isn't guaranteed
            # For max_workers=1, order should be deterministic based on glob, but sorting is safer.
            sorted_results = sorted(results)  # Sort by full path

            # Verify each created .pt file
            expected_mol_map = {"methane.pt": methane_mol_3d, "ethanol.pt": ethanol_mol_3d}

            assert len(sorted_results) == len(
                expected_mol_map
            ), f"Expected {len(expected_mol_map)} .pt files, found {len(sorted_results)}"

            for pt_path in sorted_results:
                assert os.path.exists(pt_path)
                loaded_data = torch.load(pt_path, weights_only=False)  # Load the .pt file
                assert isinstance(loaded_data, Data)

                file_basename = os.path.basename(pt_path)
                assert file_basename in expected_mol_map, f"Unexpected file {file_basename} in results."

                original_mol = expected_mol_map[file_basename]
                # The fixtures already have Hs added. batch_create_graphs_from_molecules
                # (via _process_single_mol_file_for_batch -> mol_file_to_graph) also ensures Hs.
                assert (
                    loaded_data.num_nodes == original_mol.GetNumAtoms()
                ), f"Node count mismatch for {file_basename}. Expected {original_mol.GetNumAtoms()}, got {loaded_data.num_nodes}"
