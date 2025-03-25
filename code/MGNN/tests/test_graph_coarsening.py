"""
Tests for the graph coarsening functionality.
"""

import os
import sys
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder
from code.MGNN.architectures.graph_coarsening import GraphCoarsener, FunctionalGroupIdentifier


def create_test_molecule():
    """Create a simple test molecule (PFOA - Perfluorooctanoic acid)."""
    mol = Chem.MolFromSmiles("C(C(F)(F)F)(C(C(C(C(=O)O)(F)F)(F)F)(F)F)(F)F")
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol


class TestFunctionalGroupIdentifier:
    """Test the FunctionalGroupIdentifier class."""
    
    def test_identify_cf_groups(self):
        """Test identification of CF groups."""
        identifier = FunctionalGroupIdentifier()
        mol = create_test_molecule()
        
        cf_groups = identifier.identify_cf_groups(mol)
        
        # Check that CF groups are identified
        assert len(cf_groups) > 0, "No CF groups identified in PFOA"
        
        # Check that we have different types of CF groups
        group_types = set(cf_groups.values())
        assert 'CF3' in group_types, "CF3 group not identified in PFOA"
    
    def test_identify_carboxylic_groups(self):
        """Test identification of carboxylic groups."""
        identifier = FunctionalGroupIdentifier()
        mol = create_test_molecule()
        
        carboxylic_groups = identifier.identify_carboxylic_groups(mol)
        
        # PFOA has one carboxylic group
        assert len(carboxylic_groups) == 1, "Expected one carboxylic group in PFOA"
        
        # The carboxylic group should include multiple atoms
        assert len(carboxylic_groups[0]) >= 3, "Carboxylic group should include at least 3 atoms"
    
    def test_identify_all_functional_groups(self):
        """Test identification of all functional groups."""
        identifier = FunctionalGroupIdentifier()
        mol = create_test_molecule()
        
        cf_groups, functional_groups = identifier.identify_all_functional_groups(mol)
        
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
    
    def test_create_functional_group_graph(self):
        """Test creation of functional group level graph."""
        # Create atom-level graph first
        builder = MolecularGraphBuilder(use_pfas_specific_features=True)
        mol = create_test_molecule()
        atom_graph = builder.mol_to_graph(mol)
        
        # Create functional group level graph
        coarsener = GraphCoarsener()
        functional_group_graph = coarsener.create_functional_group_graph(atom_graph, mol)
        
        # Check that the coarsened graph has the right structure
        assert hasattr(functional_group_graph, 'x')
        assert hasattr(functional_group_graph, 'edge_index')
        assert hasattr(functional_group_graph, 'edge_attr')
        assert hasattr(functional_group_graph, 'cluster_mapping')
        
        # The number of nodes should be less than or equal to the atom graph
        assert functional_group_graph.num_nodes <= atom_graph.num_nodes
    
    def test_create_structural_motif_graph(self):
        """Test creation of structural motif level graph."""
        # Create atom-level graph first
        builder = MolecularGraphBuilder(use_pfas_specific_features=True)
        mol = create_test_molecule()
        atom_graph = builder.mol_to_graph(mol)
        
        # Create structural motif level graph
        coarsener = GraphCoarsener()
        structural_motif_graph = coarsener.create_structural_motif_graph(atom_graph, mol)
        
        # Check that the coarsened graph has the right structure
        assert hasattr(structural_motif_graph, 'x')
        assert hasattr(structural_motif_graph, 'edge_index')
        assert hasattr(structural_motif_graph, 'edge_attr')
        
        # The structural motif graph should have even fewer nodes than the functional group graph
        assert structural_motif_graph.num_nodes <= atom_graph.num_nodes
        assert structural_motif_graph.num_nodes <= 2  # Just head and tail for simple PFAS
    
    def test_create_hierarchical_graphs(self):
        """Test creation of hierarchical graphs."""
        # Create atom-level graph first
        builder = MolecularGraphBuilder(use_pfas_specific_features=True)
        mol = create_test_molecule()
        atom_graph = builder.mol_to_graph(mol)
        
        # Create hierarchical graphs
        coarsener = GraphCoarsener()
        hierarchical_graphs = coarsener.create_hierarchical_graphs(atom_graph, mol)
        
        # Check that we have the expected levels
        assert 'atom' in hierarchical_graphs
        assert 'functional_group' in hierarchical_graphs
        assert 'structural_motif' in hierarchical_graphs
        
        # Check the progressive coarsening
        assert hierarchical_graphs['atom'].num_nodes >= hierarchical_graphs['functional_group'].num_nodes
        assert hierarchical_graphs['functional_group'].num_nodes >= hierarchical_graphs['structural_motif'].num_nodes


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 