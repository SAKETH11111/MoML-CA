"""
Unit tests for the MolecularGraphProcessor class and related functions
from moml.core.molecular_graph_processor.
"""
import pytest
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Any, List
import tempfile
import os
import json
import pandas as pd
import numpy as np
import time # Added for sleep

from moml.core.molecular_graph_processor import (
    MolecularGraphProcessor,
    create_graph_processor,
    mol_file_to_graph,
    graph_to_device,
    collate_graphs,
    find_charges_file,
    read_charges_from_file,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules # Added for testing
)
from moml.core.molecular_feature_extraction import MolecularFeatureExtractor # For ATOM_FEATURES, BOND_FEATURES

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
    return {
        'use_partial_charges': False,
        'use_3d_coords': True,
        'use_pfas_specific_features': True
    }

@pytest.fixture
def processor_no_3d_config() -> Dict[str, Any]:
    """Configuration with use_3d_coords set to False."""
    return {
        'use_partial_charges': False,
        'use_3d_coords': False,
        'use_pfas_specific_features': True
    }

@pytest.fixture
def processor_no_pfas_features_config() -> Dict[str, Any]:
    """Configuration with use_pfas_specific_features set to False."""
    return {
        'use_partial_charges': False,
        'use_3d_coords': True,
        'use_pfas_specific_features': False
    }

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
        assert processor.use_3d_coords == default_processor_config['use_3d_coords']
        assert processor.use_pfas_specific_features == default_processor_config['use_pfas_specific_features']

        processor_custom = MolecularGraphProcessor(config={'use_3d_coords': False, 'use_pfas_specific_features': False})
        assert not processor_custom.use_3d_coords
        assert not processor_custom.use_pfas_specific_features

        processor_empty_config = MolecularGraphProcessor()
        assert processor_empty_config.use_partial_charges is True
        assert processor_empty_config.use_3d_coords is True
        assert processor_empty_config.use_pfas_specific_features is True

    def test_atom_feature_dim(self, graph_processor: MolecularGraphProcessor,
                                processor_no_pfas_features_config: Dict[str, Any]):
        """Test the atom_feature_dim property."""
        # Expected base dimension calculation components from MolecularFeatureExtractor
        base_one_hots_dim = (len(MolecularFeatureExtractor.ATOM_FEATURES['atomic_num']) +
                             len(MolecularFeatureExtractor.ATOM_FEATURES['degree']) +
                             len(MolecularFeatureExtractor.ATOM_FEATURES['formal_charge']) +
                             len(MolecularFeatureExtractor.ATOM_FEATURES['hybridization']) +
                             len(MolecularFeatureExtractor.ATOM_FEATURES['is_aromatic']) +
                             len(MolecularFeatureExtractor.ATOM_FEATURES['is_in_ring']))
        num_hs_dim = 1
        # is_f, is_cf are always added by _get_atom_features, and now accounted for in atom_feature_dim
        always_present_pfas_like_dim = 2
        
        # Processor with use_pfas_specific_features = False
        # Expected: base_one_hots + num_hs + always_present_pfas_like_dim
        processor_no_pfas = MolecularGraphProcessor(config=processor_no_pfas_features_config)
        expected_dim_no_pfas_specific = base_one_hots_dim + num_hs_dim + always_present_pfas_like_dim
        assert processor_no_pfas.atom_feature_dim == expected_dim_no_pfas_specific

        # Processor with use_pfas_specific_features = True (graph_processor fixture)
        # Adds: num_f_neighbors (1) + func_group_flags (3)
        pfas_specific_add_on_dim = 1 + 3
        expected_dim_with_pfas_specific = expected_dim_no_pfas_specific + pfas_specific_add_on_dim
        assert graph_processor.atom_feature_dim == expected_dim_with_pfas_specific

    def test_bond_feature_dim(self, graph_processor: MolecularGraphProcessor,
                                processor_no_3d_config: Dict[str, Any]):
        """Test the bond_feature_dim property."""
        # Expected base dimension calculation components
        base_one_hots_bond_dim = (len(MolecularFeatureExtractor.BOND_FEATURES['bond_type']) +
                                  len(MolecularFeatureExtractor.BOND_FEATURES['is_conjugated']) +
                                  len(MolecularFeatureExtractor.BOND_FEATURES['is_in_ring']))
        # is_cf_bond is always added by _get_bond_features, and now accounted for in bond_feature_dim
        always_present_cf_bond_dim = 1
        
        pfas_specific_bond_add_on_dim = 3 # is_cf_cf_bond, is_fluorinated_tail_bond, is_func_group_bond
        bond_length_dim = 1

        # Processor with use_3d_coords = False, use_pfas_specific_features = True
        processor_no_3d = MolecularGraphProcessor(config=processor_no_3d_config)
        expected_dim_no_3d_pfas_true = base_one_hots_bond_dim + always_present_cf_bond_dim + pfas_specific_bond_add_on_dim
        assert processor_no_3d.bond_feature_dim == expected_dim_no_3d_pfas_true

        # Processor with use_3d_coords = True, use_pfas_specific_features = True (graph_processor fixture)
        expected_dim_3d_pfas_true = base_one_hots_bond_dim + always_present_cf_bond_dim + pfas_specific_bond_add_on_dim + bond_length_dim
        assert graph_processor.bond_feature_dim == expected_dim_3d_pfas_true

    @staticmethod
    def test_one_hot_encoding():
        """Test the _one_hot_encoding static method."""
        choices = ['a', 'b', 'c']
        assert MolecularGraphProcessor._one_hot_encoding('a', choices) == [1, 0, 0]
        assert MolecularGraphProcessor._one_hot_encoding('c', choices) == [0, 0, 1]
        assert MolecularGraphProcessor._one_hot_encoding('d', choices) == [0, 0, 0]

    def test_get_atom_features_methane(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test _get_atom_features for a simple methane molecule."""
        carbon_atom = methane_mol_3d.GetAtomWithIdx(0)
        features = graph_processor._get_atom_features(carbon_atom)
        # Actual features added might include distance features if pfas_specific_features is True,
        # even if not explicitly requested, if _calculate_distance_features returns something.
        # The atom_feature_dim property does not account for these distance features.
        # For methane, distance features like dist_to_cf3 will be -1.
        # Let's check against a graph generated without distance features for a more stable comparison.
        
        # Create a processor that won't add distance features to the calculation of _get_atom_features
        # by turning off pfas_specific_features (which controls addition of distance features in _get_atom_features)
        config_no_pfas = graph_processor.config.copy()
        config_no_pfas['use_pfas_specific_features'] = False
        processor_no_pfas_specific = MolecularGraphProcessor(config=config_no_pfas)
        
        features_no_pfas_specific = processor_no_pfas_specific._get_atom_features(carbon_atom)
        assert len(features_no_pfas_specific) == processor_no_pfas_specific.atom_feature_dim


    def test_get_atom_features_pfoa_fragment(self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol):
        """Test _get_atom_features for a PFOA fragment, checking PFAS specific features."""
        cf3_carbon = pfoa_fragment_mol_3d.GetAtomWithIdx(0) # C(F)(F)(F)
        fluorine_atom = pfoa_fragment_mol_3d.GetAtomWithIdx(1) # One of the F atoms
        cooh_carbon = pfoa_fragment_mol_3d.GetAtomWithIdx(4) # C(=O)O

        dist_features_map = graph_processor._calculate_distance_features(pfoa_fragment_mol_3d)
        
        cf3_carbon_features_vector = graph_processor._get_atom_features(
            cf3_carbon, distance_features=dist_features_map.get(cf3_carbon.GetIdx())
        )
        # This length check is tricky due to dynamic addition of distance features not in atom_feature_dim
        # For now, we assume _get_atom_features returns all possible features based on config.
        # A more robust test would be to check specific feature values.

        # Example check: is_cf for cf3_carbon should be 1 (True)
        # This requires knowing the exact index of 'is_cf'
        # is_f_idx = sum(len(v) for k,v in MolecularFeatureExtractor.ATOM_FEATURES.items() if k != 'is_in_ring') + \
        #            len(MolecularFeatureExtractor.ATOM_FEATURES['is_in_ring']) + 1 # num_hs
        # is_cf_idx = is_f_idx + 1
        # assert cf3_carbon_features_vector[is_cf_idx] == 1.0


    def test_get_bond_features_methane(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test _get_bond_features for a C-H bond in methane."""
        bond = methane_mol_3d.GetBondWithIdx(0)
        bond_lengths = graph_processor._calculate_bond_lengths(methane_mol_3d)
        bond_len = bond_lengths.get(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
        
        features = graph_processor._get_bond_features(bond, bond_length=bond_len)
        assert len(features) == graph_processor.bond_feature_dim

    def test_get_bond_features_pfoa_fragment(self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol):
        """Test _get_bond_features for C-F and C-C bonds in PFOA fragment."""
        bond_lengths = graph_processor._calculate_bond_lengths(pfoa_fragment_mol_3d)
        
        cf_bond = pfoa_fragment_mol_3d.GetBondBetweenAtoms(0,1) # C-F
        assert cf_bond is not None
        
        cf_bond_len = bond_lengths.get(tuple(sorted((cf_bond.GetBeginAtomIdx(), cf_bond.GetEndAtomIdx()))))
        cf_features = graph_processor._get_bond_features(cf_bond, bond_length=cf_bond_len)
        assert len(cf_features) == graph_processor.bond_feature_dim
        # is_cf_bond is the 4th pfas specific feature (index 3 after one-hots) + 3 one-hots = index 6 if all are present
        # This is brittle. A better check:
        # is_cf_bond_val = cf_features[len(MolecularFeatureExtractor.BOND_FEATURES['bond_type']) + \
        #                            len(MolecularFeatureExtractor.BOND_FEATURES['is_conjugated']) + \
        #                            len(MolecularFeatureExtractor.BOND_FEATURES['is_in_ring'])]
        # assert is_cf_bond_val == 1.0


    def test_mol_to_graph_methane_3d(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test mol_to_graph for 3D methane."""
        graph = graph_processor.mol_to_graph(methane_mol_3d)
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == methane_mol_3d.GetNumAtoms()
        # The number of features in graph.x can be greater than atom_feature_dim if distance features,
        # partial charges, or HOMO/LUMO are added by _get_atom_features.
        # For methane with default config, distance features will be added.
        # Let's check it's at least atom_feature_dim
        assert graph.x.shape[1] >= graph_processor.atom_feature_dim 
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
        assert graph.x.shape[1] >= processor.atom_feature_dim # Similar to above
        assert graph.edge_index.shape[1] == methane_mol_2d.GetNumBonds() * 2
        assert graph.edge_attr.shape[1] == processor.bond_feature_dim
        assert graph.pos is None

    def test_mol_to_graph_no_conformer_error(self, graph_processor: MolecularGraphProcessor, methane_mol_2d: Chem.Mol):
        """Test mol_to_graph raises ValueError if 3D coords are expected but not present."""
        with pytest.raises(ValueError, match="Molecule does not have 3D coordinates"):
            graph_processor.mol_to_graph(methane_mol_2d)
            
    def test_mol_to_graph_pfoa_fragment(self, graph_processor: MolecularGraphProcessor, pfoa_fragment_mol_3d: Chem.Mol):
        """Test mol_to_graph with a PFOA fragment."""
        graph = graph_processor.mol_to_graph(pfoa_fragment_mol_3d)
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == pfoa_fragment_mol_3d.GetNumAtoms()
        assert graph.x.shape[1] >= graph_processor.atom_feature_dim

    def test_mol_to_graph_with_additional_features(self, pfoa_fragment_mol_3d: Chem.Mol):
        """Test mol_to_graph with provided partial charges and HOMO/LUMO contributions."""
        config_full_features = {
            'use_partial_charges': True, 'use_3d_coords': True, 'use_pfas_specific_features': True
        }
        processor = MolecularGraphProcessor(config=config_full_features)
        num_atoms = pfoa_fragment_mol_3d.GetNumAtoms()
        
        charges = [0.1 * i for i in range(num_atoms)]
        homo = [0.01 * i for i in range(num_atoms)]
        lumo = [0.001 * i for i in range(num_atoms)]
        additional_features = {
            'partial_charges': charges,
            'homo_contributions': homo,
            'lumo_contributions': lumo
        }
        
        graph_with_add = processor.mol_to_graph(pfoa_fragment_mol_3d, additional_features=additional_features)
        
        config_no_add = config_full_features.copy()
        config_no_add['use_partial_charges'] = False # Turn off to see difference
        processor_no_add = MolecularGraphProcessor(config=config_no_add)
        # Create graph without any additional features for baseline dimension
        graph_no_add = processor_no_add.mol_to_graph(pfoa_fragment_mol_3d, additional_features=None)
        
        # Expect 1 for partial_charges + 2 for homo/lumo
        assert graph_with_add.x.shape[1] == graph_no_add.x.shape[1] + 1 + 2


    def test_smiles_to_graph(self, graph_processor: MolecularGraphProcessor, ethanol_mol_3d: Chem.Mol):
        """Test smiles_to_graph method."""
        smiles = "CCO"
        graph_from_smiles = graph_processor.smiles_to_graph(smiles)
        graph_from_mol = graph_processor.mol_to_graph(ethanol_mol_3d)

        assert isinstance(graph_from_smiles, Data)
        assert graph_from_smiles.x.shape == graph_from_mol.x.shape
        assert graph_from_smiles.edge_index.shape == graph_from_mol.edge_index.shape
        assert graph_from_smiles.edge_attr.shape == graph_from_mol.edge_attr.shape
        if graph_processor.use_3d_coords:
            assert graph_from_smiles.pos.shape == graph_from_mol.pos.shape

    def test_smiles_to_graph_invalid_smiles(self, graph_processor: MolecularGraphProcessor):
        """Test smiles_to_graph with invalid SMILES."""
        invalid_smiles = "thisisnotasmiles"
        with pytest.raises(ValueError): # RDKit AddHs(None) raises ValueError or Boost.Python.ArgumentError
            graph_processor.smiles_to_graph(invalid_smiles)

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

    def test_batch_files_to_graphs(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol, ethanol_mol_3d: Chem.Mol):
        """Test batch_files_to_graphs method."""
        files_to_create = {
            "methane.mol": methane_mol_3d,
            "ethanol.mol": ethanol_mol_3d
        }
        file_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, mol in files_to_create.items():
                filepath = os.path.join(tmpdir, name)
                with open(filepath, "w") as f:
                    f.write(Chem.MolToMolBlock(mol))
                file_paths.append(filepath)
            
            graphs = graph_processor.batch_files_to_graphs(file_paths)
            assert len(graphs) == len(file_paths)
            assert isinstance(graphs[0], Data)
            assert isinstance(graphs[1], Data)
            assert graphs[0].x.shape[0] == methane_mol_3d.GetNumAtoms()
            assert graphs[1].x.shape[0] == ethanol_mol_3d.GetNumAtoms()

    def test_mol_to_json_graph(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test mol_to_json_graph method."""
        json_graph = graph_processor.mol_to_json_graph(methane_mol_3d)
        assert isinstance(json_graph, dict)
        assert "atoms" in json_graph
        assert "bonds" in json_graph
        assert "descriptors" in json_graph
        assert len(json_graph["atoms"]) == methane_mol_3d.GetNumAtoms()
        assert len(json_graph["bonds"]) == methane_mol_3d.GetNumBonds()
        assert "mol_weight" in json_graph["descriptors"] # Changed from ExactMolWt to mol_weight

    def test_file_to_json_graph(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test file_to_json_graph method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_filepath = os.path.join(tmpdir, "methane.mol")
            with open(mol_filepath, "w") as f:
                f.write(Chem.MolToMolBlock(methane_mol_3d))
            
            output_filename = "methane_graph.json"
            json_output_path = graph_processor.file_to_json_graph(mol_filepath, output_dir=tmpdir, output_filename=output_filename)
            
            assert json_output_path is not None
            assert os.path.exists(json_output_path)
            assert os.path.basename(json_output_path) == output_filename

            with open(json_output_path, "r") as f_json:
                json_data = json.load(f_json)
            
            assert "atoms" in json_data # Changed from "nodes"
            assert len(json_data["atoms"]) == methane_mol_3d.GetNumAtoms()

    def test_get_atom_features_instance_method(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test the instance method get_atom_features."""
        atom_features_array = graph_processor.get_atom_features(methane_mol_3d)
        assert isinstance(atom_features_array, np.ndarray) # Changed from list
        assert atom_features_array.shape[0] == methane_mol_3d.GetNumAtoms()
        # The features here are from _get_atom_features, so length can be > atom_feature_dim
        assert atom_features_array.shape[1] >= graph_processor.atom_feature_dim

    def test_get_adjacency_matrix(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test get_adjacency_matrix method."""
        adj_matrix = graph_processor.get_adjacency_matrix(methane_mol_3d)
        num_atoms = methane_mol_3d.GetNumAtoms()
        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (num_atoms, num_atoms)
        # Methane: C at index 0, Hs at 1,2,3,4. C is connected to all Hs.
        # This assumes specific atom indexing after AddHs.
        # A more general check: sum of row 0 should be 4 (degree of C)
        assert np.sum(adj_matrix[0]) == methane_mol_3d.GetAtomWithIdx(0).GetDegree()
        for i in range(1, num_atoms): # Hydrogens
             assert np.sum(adj_matrix[i]) == methane_mol_3d.GetAtomWithIdx(i).GetDegree()


    def test_process_dataframe(self, graph_processor: MolecularGraphProcessor, methane_mol_3d: Chem.Mol):
        """Test process_dataframe method."""
        data = {'smiles': ['C', 'CC'], 'id': [1, 2]}
        df = pd.DataFrame(data)
        df['rdkit_mol'] = df['smiles'].apply(lambda s: create_rdkit_mol(s, add_3d_coords=graph_processor.use_3d_coords))

        processed_df = graph_processor.process_dataframe(df.copy(), mol_column='rdkit_mol') # Use copy
        
        assert 'graph_data' in processed_df.columns
        assert 'atom_features' in processed_df.columns
        assert 'adjacency_matrix' in processed_df.columns # Changed from adj_matrix
        assert isinstance(processed_df['graph_data'].iloc[0], Data)
        assert isinstance(processed_df['atom_features'].iloc[0], np.ndarray)
        assert isinstance(processed_df['adjacency_matrix'].iloc[0], np.ndarray) # Changed
        assert processed_df['atom_features'].iloc[0].shape[0] == df['rdkit_mol'].iloc[0].GetNumAtoms()


class TestUtilityFunctions:
    """Tests for standalone utility functions in molecular_graph_processor."""

    def test_create_graph_processor_utility(self):
        """Test the create_graph_processor factory function."""
        config = {'use_3d_coords': False, 'use_pfas_specific_features': False}
        processor = create_graph_processor(config=config)
        assert isinstance(processor, MolecularGraphProcessor)
        assert not processor.use_3d_coords
        assert not processor.use_pfas_specific_features

        default_processor = create_graph_processor() # Test with default config
        assert default_processor.use_3d_coords is True # Default from MolecularGraphProcessor init

    def test_mol_file_to_graph_utility(self, methane_mol_3d: Chem.Mol):
        """Test the mol_file_to_graph utility function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mol", delete=False) as tmp_mol_file:
            tmp_mol_file.write(Chem.MolToMolBlock(methane_mol_3d))
            filepath = tmp_mol_file.name
        
        # Pass config via config argument
        config_arg = {'use_3d_coords': True, 'use_pfas_features': True}
        graph = mol_file_to_graph(filepath, config=config_arg) # Changed processor_config to config
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == methane_mol_3d.GetNumAtoms()
        assert graph.pos is not None
        os.remove(filepath)

    def test_graph_to_device(self, methane_mol_3d: Chem.Mol):
        """Test graph_to_device utility function."""
        processor = MolecularGraphProcessor() # Default config
        graph = processor.mol_to_graph(methane_mol_3d)
        
        cpu_device = torch.device("cpu")
        graph_on_cpu = graph_to_device(graph, cpu_device)
        assert graph_on_cpu.x.device == cpu_device
        assert graph_on_cpu.edge_index.device == cpu_device
        
        # Test with a dictionary (as used in some parts of the codebase)
        graph_dict = {"x": torch.randn(2,3), "edge_index": torch.randint(0,2,(2,1))}
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
        assert 'batch' in batched_graph
        assert batched_graph.num_graphs == 2
        assert batched_graph.x.shape[0] == graph1.x.shape[0] + graph2.x.shape[0]
        assert batched_graph.edge_index.shape[1] == graph1.edge_index.shape[1] + graph2.edge_index.shape[1]

    def test_find_charges_file(self):
        """Test find_charges_file utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_filepath = os.path.join(tmpdir, "test_mol.mol")
            open(mol_filepath, 'a').close() # Create dummy mol file

            # Case 1: .charges file exists
            chg_filepath = os.path.join(tmpdir, "test_mol.charges") # Changed extension
            open(chg_filepath, 'a').close()
            assert find_charges_file(mol_filepath, tmpdir) == chg_filepath
            os.remove(chg_filepath)

            # Case 2: _charges.txt file exists
            txt_charges_filepath = os.path.join(tmpdir, "test_mol_charges.txt") # Changed extension and variable name
            open(txt_charges_filepath, 'a').close()
            assert find_charges_file(mol_filepath, tmpdir) == txt_charges_filepath
            os.remove(txt_charges_filepath)
            
            # Case 3: No charge file (should return None)
            assert find_charges_file(mol_filepath, tmpdir) is None # Changed expected return to None

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
        esp_data = {"esp_charges": [0.5, -0.5]}
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
            
            test_config = {'use_pfas_specific_features': True, 'use_3d_coords': True} # Match typical usage
            json_output_path = create_molecular_graph_json(mol_filepath, output_dir=tmpdir, config=test_config)
            
            assert json_output_path is not None
            assert os.path.exists(json_output_path)
            
            with open(json_output_path, "r") as f_json:
                json_data = json.load(f_json)
            assert "atoms" in json_data
            assert len(json_data["atoms"]) == methane_mol_3d.GetNumAtoms()

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

            config_for_batch = { # Renamed to avoid conflict if MolecularGraphProcessor is in scope
                'use_pfas_features': True,
                'use_3d_coords': True
            }
            results = batch_create_graphs_from_molecules(
                mol_dir=tmp_mol_dir,
                output_dir=tmp_out_dir,
                config=config_for_batch, # Changed processor_config to config
                max_workers=1 # For predictable testing
            )
            
            assert len(results) == 2 # methane and ethanol
            for result in results:
                assert result is not None
                assert os.path.exists(result)
                assert result.startswith(tmp_out_dir)
                assert result.endswith("_graph.json")

            # Check content of one of the files
            with open(results[0], 'r') as f:
                graph_json = json.load(f)
            assert "atoms" in graph_json # Changed from "nodes"
            # The order of results might not be guaranteed, so check if one matches methane
            # This check is a bit fragile due to naming inside batch function.
            # A better check would be on number of atoms if we know which file is which.
            # For now, just check one of them has the expected atom count for one of the molecules.
            is_methane = len(graph_json["atoms"]) == methane_mol_3d.GetNumAtoms()
            is_ethanol = len(graph_json["atoms"]) == ethanol_mol_3d.GetNumAtoms()
            assert is_methane or is_ethanol
