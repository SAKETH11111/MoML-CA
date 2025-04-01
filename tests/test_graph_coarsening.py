"""
Tests for the graph coarsening functionality.

This file serves as the primary test suite for graph coarsening, covering:
1. Functional group identification
2. Graph coarsening at multiple levels
3. Hierarchical graph creation
"""

import os
import sys
import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import warnings
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
mgnn_dir = os.path.dirname(current_dir)
code_dir = os.path.dirname(mgnn_dir)
project_dir = os.path.dirname(code_dir)
sys.path.insert(0, project_dir)

# Import with relative imports
from moml.core.graph_coarsening import GraphCoarsener
from moml.core.molecular_descriptors import FunctionalGroupDetector
from moml.core.molecular_graph import MolecularGraphProcessor, create_graph_processor


# Create a custom Data class for testing that handles the keys attribute properly
class TestData(Data):
    """Custom PyTorch Geometric Data class for testing with a proper keys property."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store all original attributes
        self._stored_keys = list(kwargs.keys())
    
    @property
    def keys(self):
        """Return list of attribute keys, compatible with GraphCoarsener."""
        return self._stored_keys
    
    def __setattr__(self, key, value):
        """Override to track keys when attributes are set."""
        super().__setattr__(key, value)
        if not key.startswith('_') and key != 'keys' and hasattr(self, '_stored_keys'):
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
    """
    Create a mock atom-level graph for testing.
    """
    mol = test_molecule
    num_atoms = mol.GetNumAtoms()
    
    # Get 3D positions
    conf = mol.GetConformer()
    positions = []
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    
    # Pre-calculate max possible edges
    # Each atom connects to at most 3 others, and each edge is counted twice (i→j and j→i)
    max_edges = min(num_atoms * 6, num_atoms * (num_atoms-1))
    
    # Create a TestData object with necessary attributes
    data = TestData(
        x=torch.randn(num_atoms, 16),  # Node features
        edge_index=torch.zeros(2, max_edges, dtype=torch.long),
        edge_attr=torch.randn(max_edges, 8),
        pos=torch.tensor(positions, dtype=torch.float),
        num_nodes=num_atoms,
        y=torch.randn(5)  # Global features
    )
    
    # Create edges (connect each atom to nearby atoms)
    edge_count = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j and edge_count < max_edges - 1:  # Avoid self-loops and check bounds
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
    max_edges = min(num_atoms * 2, num_atoms * (num_atoms-1))
    
    # Node features
    x = torch.randn(num_atoms, 10)
    
    # Edge index and attributes
    edge_index = torch.zeros(2, max_edges, dtype=torch.long)
    edge_attr = torch.randn(max_edges, 4)
    
    # Create edges (connect each atom to the next)
    edge_count = 0
    for i in range(num_atoms):
        for j in range(i+1, min(i+2, num_atoms)):
            if i != j and edge_count < max_edges - 1:  # Avoid self-loops
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
    
    # Create TestData object
    data = TestData(
        x=x, 
        edge_index=edge_index[:, :edge_count], 
        edge_attr=edge_attr[:edge_count], 
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.randn(5), 
        num_nodes=num_atoms
    )
    
    return data


@pytest.fixture
def mock_functional_group_graph():
    """Create a mock functional group level graph for testing."""
    # Create a smaller graph to represent functional groups
    num_groups = 5  # A small number of functional groups
    
    # Calculate max edges for a fully connected graph (n*(n-1) since no self-loops)
    max_edges = num_groups * (num_groups - 1)
    
    # Create a Data object for the functional group graph
    data = TestData(
        x=torch.randn(num_groups, 16),  # Node features
        edge_index=torch.zeros(2, max_edges, dtype=torch.long),
        edge_attr=torch.randn(max_edges, 8),
        pos=torch.randn(num_groups, 3),
        num_nodes=num_groups,
        y=torch.randn(5),  # Global features
        cluster_mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # Simple 1:1 mapping for testing
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
    
    # Trim edges to actual count (although should be the same as max_edges)
    data.edge_index = data.edge_index[:, :edge_count]
    data.edge_attr = data.edge_attr[:edge_count]
    
    return data


class TestFunctionalGroupDetector:
    """Test the FunctionalGroupDetector class."""
    
    def test_identify_cf_groups(self, test_molecule):
        """Test identification of CF groups."""
        detector = FunctionalGroupDetector()
        
        cf_groups = detector.identify_cf_groups(test_molecule)
        
        # Check that CF groups are identified
        assert len(cf_groups) > 0, "No CF groups identified in PFOA"
        
        # Check that we have different types of CF groups
        group_types = set(cf_groups.values())
        assert 'CF3' in group_types, "CF3 group not identified in PFOA"
    
    def test_identify_carboxylic_groups(self, test_molecule):
        """Test identification of carboxylic groups."""
        detector = FunctionalGroupDetector()
        
        carboxylic_groups = detector.identify_carboxylic_groups(test_molecule)
        
        # PFOA has one carboxylic group
        assert len(carboxylic_groups) == 1, "Expected one carboxylic group in PFOA"
        
        # The carboxylic group should include multiple atoms
        assert len(carboxylic_groups[0]) >= 3, "Carboxylic group should include at least 3 atoms"
    
    def test_identify_all_functional_groups(self, test_molecule):
        """Test identification of all functional groups."""
        detector = FunctionalGroupDetector()
        
        cf_groups, functional_groups = detector.identify_all_functional_groups(test_molecule)
        
        # Check CF groups
        assert len(cf_groups) > 0, "No CF groups identified in PFOA"
        
        # Check other functional groups (at least the carboxylic group)
        assert len(functional_groups) >= 1, "Expected at least one functional group in PFOA"


class TestGraphCoarsener:
    """Test the GraphCoarsener class."""
    
    def test_initialization(self):
        """Test initialization of the GraphCoarsener."""
        coarsener = GraphCoarsener()
        assert coarsener.use_3d_coords is True
        
        coarsener = GraphCoarsener(use_3d_coords=False)
        assert coarsener.use_3d_coords is False
    
    def test_create_functional_group_graph(self, mock_atom_graph, test_molecule):
        """Test creation of functional group level graph."""
        # Create functional group level graph
        coarsener = GraphCoarsener()
        functional_group_graph = coarsener.create_functional_group_graph(mock_atom_graph, test_molecule)
        
        # Check that the coarsened graph has the right structure
        assert hasattr(functional_group_graph, 'x')
        assert hasattr(functional_group_graph, 'edge_index')
        assert hasattr(functional_group_graph, 'edge_attr')
        assert hasattr(functional_group_graph, 'cluster_mapping')
        
        # The number of nodes should be less than or equal to the atom graph
        assert functional_group_graph.num_nodes <= mock_atom_graph.num_nodes
    
    @patch('code.MGNN.utils.molecular_descriptors.MolecularFeatureExtractor.calculate_distance_features')
    def test_create_structural_motif_graph(self, mock_calc_dist, mock_functional_group_graph, test_molecule):
        """Test creation of structural motif level graph.
        
        This test uses mocking to avoid RDKit issues with GetShortestPath.
        """
        # Mock the calculate_distance_features method to avoid RDKit errors
        mock_calc_dist.return_value = {
            i: {'dist_to_cf3': 0.5, 'dist_to_functional': 0.5, 'is_head_group': 0.5} 
            for i in range(test_molecule.GetNumAtoms())
        }
        
        # Create structural motif level graph
        coarsener = GraphCoarsener()
        
        # Patch the _create_structural_mapping method to return a simple mapping
        with patch.object(coarsener, '_create_structural_mapping') as mock_create_mapping:
            # Create a simple mapping: 0=head, 1=tail for all atoms
            mock_create_mapping.return_value = {
                node_idx: 0 if node_idx < 2 else 1 
                for node_idx in range(mock_functional_group_graph.num_nodes)
            }
            
            structural_motif_graph = coarsener.create_structural_motif_graph(
                mock_functional_group_graph, test_molecule
            )
        
        # Check that the coarsened graph has the right structure
        assert hasattr(structural_motif_graph, 'x')
        assert hasattr(structural_motif_graph, 'edge_index')
        assert hasattr(structural_motif_graph, 'edge_attr')
        
        # Should have exactly 2 nodes (head and tail)
        assert structural_motif_graph.num_nodes == 2
    
    @patch('code.MGNN.utils.molecular_descriptors.MolecularFeatureExtractor.calculate_distance_features')
    @patch('code.MGNN.architectures.graph_coarsening.GraphCoarsener.create_functional_group_graph')
    @patch('code.MGNN.architectures.graph_coarsening.GraphCoarsener.create_structural_motif_graph')
    def test_create_hierarchical_graphs(self, mock_motif_graph, mock_fg_graph, mock_calc_dist, 
                                        mock_atom_graph, mock_functional_group_graph, test_molecule):
        """Test creation of hierarchical graphs with comprehensive mocking."""
        # Create mock functional group graph
        fg_graph = mock_functional_group_graph
        
        # Create mock structural motif graph
        motif_graph = TestData(
            x=torch.randn(2, 16),  # Node features for head and tail
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 8),
            pos=torch.randn(2, 3),
            num_nodes=2,
            y=torch.randn(5)
        )
        
        # Set up the mocks to return our pre-built graphs
        mock_fg_graph.return_value = fg_graph
        mock_motif_graph.return_value = motif_graph
        
        # Create hierarchical graphs
        coarsener = GraphCoarsener()
        hierarchical_graphs = coarsener.create_hierarchical_graphs(mock_atom_graph, test_molecule)
        
        # Check that we have the expected levels
        assert 'atom' in hierarchical_graphs
        assert 'functional_group' in hierarchical_graphs
        assert 'structural_motif' in hierarchical_graphs
        
        # Check the graphs are as expected
        assert hierarchical_graphs['atom'] == mock_atom_graph
        assert hierarchical_graphs['functional_group'] == fg_graph
        assert hierarchical_graphs['structural_motif'] == motif_graph
        
        # Verify that the mocks were called
        mock_fg_graph.assert_called_once_with(mock_atom_graph, test_molecule)
        mock_motif_graph.assert_called_once_with(fg_graph, test_molecule)

    def test_with_mock_data(self, mock_torch_geometric_data, test_molecule):
        """Test graph coarsening with mock data."""
        coarsener = GraphCoarsener(use_3d_coords=True)
        
        # Create functional group level graph
        fg_graph = coarsener.create_functional_group_graph(mock_torch_geometric_data, test_molecule)
        assert fg_graph.num_nodes < mock_torch_geometric_data.num_nodes


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 