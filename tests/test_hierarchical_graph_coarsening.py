"""Tests for the graph coarsening functionality.

This file serves as the primary test suite for graph coarsening, covering:
1. Functional group identification
2. Graph coarsening at multiple levels
3. Hierarchical graph creation
"""

from typing import Dict, Any
from unittest.mock import patch

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from moml.core import GraphCoarsener
from moml.core.molecular_feature_extraction import FunctionalGroupDetector


class CustomGraphData(Data):
    """Custom PyTorch Geometric Data class for testing with a proper keys property."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Store all original attributes
        self._stored_keys = list(kwargs.keys())

    @property
    def keys(self) -> list:
        """Return list of attribute keys, compatible with GraphCoarsener."""
        return self._stored_keys

    def __setattr__(self, key: str, value: Any) -> None:
        """Override to track keys when attributes are set."""
        super().__setattr__(key, value)
        if not key.startswith("_") and key != "keys" and hasattr(self, "_stored_keys"):
            if key not in self._stored_keys:
                self._stored_keys.append(key)


@pytest.fixture
def test_molecule():
    """Create a simple test molecule (PFOA - Perfluorooctanoic acid)."""
    mol = Chem.MolFromSmiles("C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F")
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    return mol


@pytest.fixture
def mock_atom_graph(test_molecule):
    """Create a mock atom-level graph for testing."""
    mol = test_molecule
    num_atoms = mol.GetNumAtoms()

    # Get 3D positions
    conf = mol.GetConformer()
    positions = []
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    # Pre-calculate max possible edges
    max_edges = min(num_atoms * 6, num_atoms * (num_atoms - 1))

    # Create a CustomGraphData object with necessary attributes
    data = CustomGraphData(
        x=torch.randn(num_atoms, 16),  # Node features
        edge_index=torch.zeros(2, max_edges, dtype=torch.long),
        edge_attr=torch.randn(max_edges, 8),
        pos=torch.tensor(positions, dtype=torch.float),
        num_nodes=num_atoms,
        y=torch.randn(5),  # Global features
    )

    # Create edges (connect each atom to nearby atoms)
    edge_count = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j and edge_count < max_edges - 1:
                data.edge_index[0, edge_count] = i
                data.edge_index[1, edge_count] = j
                edge_count += 1

                if edge_count >= max_edges - 1:
                    break
        if edge_count >= max_edges - 1:
            break

    # Trim edges to actual count
    data.edge_index = data.edge_index[:, :edge_count]
    data.edge_attr = data.edge_attr[:edge_count]

    return data


@pytest.fixture
def mock_torch_geometric_data(test_molecule):
    """Create a mock PyTorch Geometric Data object for testing."""
    mol = test_molecule
    num_atoms = mol.GetNumAtoms()

    # Pre-calculate max possible edges
    max_edges = min(num_atoms * 2, num_atoms * (num_atoms - 1))

    # Node features
    x = torch.randn(num_atoms, 10)

    # Edge index and attributes
    edge_index = torch.zeros(2, max_edges, dtype=torch.long)
    edge_attr = torch.randn(max_edges, 4)

    # Create edges (connect each atom to the next)
    edge_count = 0
    for i in range(num_atoms):
        for j in range(i + 1, min(i + 2, num_atoms)):
            if i != j and edge_count < max_edges - 1:
                edge_index[0, edge_count] = i
                edge_index[1, edge_count] = j
                edge_count += 1
                if edge_count < max_edges:
                    edge_index[0, edge_count] = j
                    edge_index[1, edge_count] = i
                    edge_count += 1

    # Get 3D positions
    conf = mol.GetConformer()
    positions = []
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    # Create CustomGraphData object
    data = CustomGraphData(
        x=x,
        edge_index=edge_index[:, :edge_count],
        edge_attr=edge_attr[:edge_count],
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.randn(5),
        num_nodes=num_atoms,
    )

    return data


@pytest.fixture
def mock_functional_group_graph():
    """Create a mock functional group level graph for testing."""
    # Create a smaller graph to represent functional groups
    num_groups = 5

    # Calculate max edges for a fully connected graph
    max_edges = num_groups * (num_groups - 1)

    # Create a Data object for the functional group graph
    data = CustomGraphData(
        x=torch.randn(num_groups, 16),  # Node features
        edge_index=torch.zeros(2, max_edges, dtype=torch.long),
        edge_attr=torch.randn(max_edges, 8),
        pos=torch.randn(num_groups, 3),
        num_nodes=num_groups,
        y=torch.randn(5),  # Global features
        cluster_mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    )

    # Create a simple fully connected graph
    edge_count = 0
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                data.edge_index[0, edge_count] = i
                data.edge_index[1, edge_count] = j
                edge_count += 1
                if edge_count >= max_edges:
                    break
        if edge_count >= max_edges:
            break

    # Trim edges to actual count
    data.edge_index = data.edge_index[:, :edge_count]
    data.edge_attr = data.edge_attr[:edge_count]

    return data


class TestFunctionalGroupDetector:
    """Test the FunctionalGroupDetector class."""

    def test_identify_cf_groups(self, test_molecule) -> None:
        """Test identification of CF groups."""
        detector = FunctionalGroupDetector()

        cf_groups = detector.find_cf_groups(test_molecule)

        # Check that CF groups are identified
        assert len(cf_groups) > 0, "No CF groups identified in PFOA"

        # Check that we have different types of CF groups
        group_types = set(cf_groups.values())
        assert "CF3" in group_types, "CF3 group not identified in PFOA"

    def test_identify_carboxylic_groups(self, test_molecule) -> None:
        """Test identification of carboxylic groups."""
        detector = FunctionalGroupDetector()

        carboxylic_groups = detector.identify_carboxylic_groups(test_molecule)

        # PFOA has one carboxylic group
        assert len(carboxylic_groups) == 1, "Expected one carboxylic group in PFOA"

        # The carboxylic group should include multiple atoms
        assert len(carboxylic_groups[0]) >= 3, "Carboxylic group should include at least 3 atoms"

    def test_identify_all_functional_groups(self, test_molecule) -> None:
        """Test identification of all functional groups."""
        detector = FunctionalGroupDetector()

        cf_groups, functional_groups = detector.identify_all_functional_groups(test_molecule)

        # Check CF groups
        assert len(cf_groups) > 0, "No CF groups identified in PFOA"

        # Check other functional groups (at least the carboxylic group)
        assert len(functional_groups) >= 1, "Expected at least one functional group in PFOA"


class TestGraphCoarsener:
    """Test the GraphCoarsener class."""

    def test_initialization(self) -> None:
        """Test initialization of the GraphCoarsener."""
        coarsener = GraphCoarsener()
        assert coarsener.use_3d_coords is True

        coarsener = GraphCoarsener(use_3d_coords=False)
        assert coarsener.use_3d_coords is False

    def test_create_functional_group_graph(self, mock_atom_graph, test_molecule) -> None:
        """Test creation of functional group level graph."""
        # Create functional group level graph
        coarsener = GraphCoarsener()
        functional_group_graph = coarsener.create_functional_group_graph(mock_atom_graph, test_molecule)

        # Check that the coarsened graph has the right structure
        assert hasattr(functional_group_graph, "x")
        assert hasattr(functional_group_graph, "edge_index")
        assert hasattr(functional_group_graph, "edge_attr")
        assert hasattr(functional_group_graph, "cluster_mapping")

        # The number of nodes should be less than or equal to the atom graph
        assert functional_group_graph.num_nodes <= mock_atom_graph.num_nodes

    @patch("moml.core.molecular_feature_extraction.MolecularFeatureExtractor.calculate_distance_features")
    def test_create_structural_motif_graph(self, mock_calc_dist, mock_functional_group_graph, test_molecule) -> None:
        """Test creation of structural motif level graph.

        This test uses mocking to avoid RDKit issues with GetShortestPath.
        """
        # Mock the calculate_distance_features method to avoid RDKit errors
        mock_calc_dist.return_value = {
            i: {"dist_to_cf3": 0.5, "dist_to_functional": 0.5, "is_head_group": 0.5}
            for i in range(test_molecule.GetNumAtoms())
        }

        # Create structural motif level graph
        coarsener = GraphCoarsener()

        # Patch the _create_structural_mapping method to return a simple mapping
        with patch.object(coarsener, "_create_structural_mapping") as mock_create_mapping:
            # Get an actual atom-level graph for test_molecule
            temp_fg_graph = coarsener.create_functional_group_graph(
                coarsener.graph_processor.mol_to_graph(test_molecule), test_molecule
            )
            actual_atom_to_fg_mapping = temp_fg_graph.cluster_mapping
            unique_fg_cluster_ids = sorted(list(set(actual_atom_to_fg_mapping.values())))

            # Create a mock return value for _create_structural_mapping
            mock_map_val = {}
            if unique_fg_cluster_ids:
                split_point = len(unique_fg_cluster_ids) // 2
                if len(unique_fg_cluster_ids) == 1:
                    mock_map_val = {uid: 0 for uid in unique_fg_cluster_ids}
                else:
                    for i, fg_id in enumerate(unique_fg_cluster_ids):
                        mock_map_val[fg_id] = 0 if i < split_point or split_point == 0 else 1
                    # Ensure at least one '1' if there are multiple unique_fg_cluster_ids
                    if len(unique_fg_cluster_ids) > 1 and all(v == 0 for v in mock_map_val.values()):
                        mock_map_val[unique_fg_cluster_ids[-1]] = 1

            mock_create_mapping.return_value = mock_map_val

            # First, get an actual atom-level graph for test_molecule
            atom_level_graph_for_test_molecule = coarsener.graph_processor.mol_to_graph(test_molecule)
            assert (
                atom_level_graph_for_test_molecule is not None
            ), f"Failed to create atom-level graph for test_molecule: {Chem.MolToSmiles(test_molecule)}"

            # Now call create_structural_motif_graph with the correct atom-level graph
            structural_motif_graph = coarsener.create_structural_motif_graph(
                atom_level_graph_for_test_molecule, test_molecule
            )

        # Check that the coarsened graph has the right structure
        assert hasattr(structural_motif_graph, "x")
        assert hasattr(structural_motif_graph, "edge_index")
        assert hasattr(structural_motif_graph, "edge_attr")

        # Should have exactly 2 nodes (head and tail)
        assert structural_motif_graph.num_nodes == 2

    @patch("moml.core.molecular_feature_extraction.MolecularFeatureExtractor.calculate_distance_features")
    @patch("moml.core.hierarchical_graph_coarsener.GraphCoarsener.create_functional_group_graph")
    @patch("moml.core.hierarchical_graph_coarsener.GraphCoarsener.create_structural_motif_graph")
    def test_create_hierarchical_graphs(
        self,
        mock_motif_graph,
        mock_fg_graph,
        mock_calc_dist,
        mock_atom_graph,
        mock_functional_group_graph,
        test_molecule,
    ) -> None:
        """Test creation of hierarchical graphs with comprehensive mocking."""
        # Create mock functional group graph
        fg_graph = mock_functional_group_graph

        # Create mock structural motif graph
        motif_graph = CustomGraphData(
            x=torch.randn(2, 16),  # Node features for head and tail
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 8),
            pos=torch.randn(2, 3),
            num_nodes=2,
            y=torch.randn(5),
        )

        # Set up the mocks to return our pre-built graphs
        mock_fg_graph.return_value = fg_graph
        mock_motif_graph.return_value = motif_graph

        # Create hierarchical graphs
        coarsener = GraphCoarsener()
        hierarchical_graphs = coarsener.create_hierarchical_graphs(mock_atom_graph, test_molecule)

        # Check that we have the expected levels
        assert "atom" in hierarchical_graphs
        assert "functional_group" in hierarchical_graphs
        assert "structural_motif" in hierarchical_graphs

        # Check the graphs are as expected
        assert hierarchical_graphs["atom"] == mock_atom_graph
        assert hierarchical_graphs["functional_group"] == fg_graph
        assert hierarchical_graphs["structural_motif"] == motif_graph

        # Verify that the mocks were called
        mock_fg_graph.assert_called_once_with(mock_atom_graph, test_molecule)
        mock_motif_graph.assert_called_once_with(fg_graph, test_molecule)

    def test_with_mock_data(self, mock_torch_geometric_data, test_molecule) -> None:
        """Test graph coarsening with mock data."""
        coarsener = GraphCoarsener(use_3d_coords=True)

        # Create functional group level graph
        fg_graph = coarsener.create_functional_group_graph(mock_torch_geometric_data, test_molecule)
        assert fg_graph.num_nodes < mock_torch_geometric_data.num_nodes


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
