"""
Graph Coarsening for PFAS Molecular Structures

This module provides functionality to create hierarchical graph representations 
of PFAS molecules at different levels of coarseness:
1. Atom level (original graph)
2. Functional group level (intermediate coarseness)
3. Structural motif level (highest coarseness)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from typing import Dict, List, Tuple, Optional, Set, Union

from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder


class FunctionalGroupIdentifier:
    """
    Identifies common functional groups in PFAS molecules.
    """
    
    # Define functional group types
    FUNCTIONAL_GROUPS = {
        'CF': 1,         # Carbon with one fluorine
        'CF2': 2,        # Carbon with two fluorines 
        'CF3': 3,        # Trifluoromethyl group
        'COOH': 4,       # Carboxylic acid group
        'SO3H': 5,       # Sulfonic acid group
        'PO3H2': 6,      # Phosphonic acid group
        'OTHER': 0       # Other atoms/groups
    }
    
    @staticmethod
    def identify_cf_groups(mol: Chem.Mol) -> Dict[int, str]:
        """
        Identify CF, CF2, and CF3 groups in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping atom indices to group types
        """
        group_assignments = {}
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                f_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9)
                
                if f_neighbors == 1:
                    group_assignments[atom.GetIdx()] = 'CF'
                elif f_neighbors == 2:
                    group_assignments[atom.GetIdx()] = 'CF2'
                elif f_neighbors == 3:
                    group_assignments[atom.GetIdx()] = 'CF3'
        
        return group_assignments
    
    @staticmethod
    def identify_carboxylic_groups(mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify COOH groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a COOH group
        """
        builder = MolecularGraphBuilder()
        carboxylic_groups = []
        
        # Find central carbon atoms of carboxylic groups
        for atom in mol.GetAtoms():
            if builder._is_in_carboxylic_group(atom):
                group_atoms = {atom.GetIdx()}
                
                # Add connected oxygen atoms and hydrogens
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetAtomicNum() == 8:  # Oxygen
                        group_atoms.add(other_atom.GetIdx())
                        
                        # If this is OH, add the hydrogen too
                        for o_bond in other_atom.GetBonds():
                            h_atom = o_bond.GetOtherAtom(other_atom)
                            if h_atom.GetAtomicNum() == 1:  # Hydrogen
                                group_atoms.add(h_atom.GetIdx())
                
                carboxylic_groups.append(group_atoms)
        
        return carboxylic_groups
    
    @staticmethod
    def identify_sulfonic_groups(mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify SO3H groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a SO3H group
        """
        builder = MolecularGraphBuilder()
        sulfonic_groups = []
        
        # Find central sulfur atoms of sulfonic groups
        for atom in mol.GetAtoms():
            if builder._is_in_sulfonic_group(atom):
                group_atoms = {atom.GetIdx()}
                
                # Add connected oxygen atoms and hydrogens
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetAtomicNum() == 8:  # Oxygen
                        group_atoms.add(other_atom.GetIdx())
                        
                        # If this is OH, add the hydrogen too
                        for o_bond in other_atom.GetBonds():
                            h_atom = o_bond.GetOtherAtom(other_atom)
                            if h_atom.GetAtomicNum() == 1:  # Hydrogen
                                group_atoms.add(h_atom.GetIdx())
                
                sulfonic_groups.append(group_atoms)
        
        return sulfonic_groups
    
    @staticmethod
    def identify_phosphonic_groups(mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify PO3H2 groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a PO3H2 group
        """
        builder = MolecularGraphBuilder()
        phosphonic_groups = []
        
        # Find central phosphorus atoms of phosphonic groups
        for atom in mol.GetAtoms():
            if builder._is_in_phosphonic_group(atom):
                group_atoms = {atom.GetIdx()}
                
                # Add connected oxygen atoms and hydrogens
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetAtomicNum() == 8:  # Oxygen
                        group_atoms.add(other_atom.GetIdx())
                        
                        # If this is OH, add the hydrogen too
                        for o_bond in other_atom.GetBonds():
                            h_atom = o_bond.GetOtherAtom(other_atom)
                            if h_atom.GetAtomicNum() == 1:  # Hydrogen
                                group_atoms.add(h_atom.GetIdx())
                
                phosphonic_groups.append(group_atoms)
        
        return phosphonic_groups
    
    @classmethod
    def identify_all_functional_groups(cls, mol: Chem.Mol) -> Tuple[Dict[int, str], List[Set[int]]]:
        """
        Identify all functional groups in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Tuple containing:
            - Dictionary mapping atom indices to CF group types
            - List of sets, where each set contains atom indices belonging to a functional group
        """
        # Identify CF groups
        cf_groups = cls.identify_cf_groups(mol)
        
        # Identify other functional groups
        carboxylic_groups = cls.identify_carboxylic_groups(mol)
        sulfonic_groups = cls.identify_sulfonic_groups(mol)
        phosphonic_groups = cls.identify_phosphonic_groups(mol)
        
        # Combine all non-CF functional groups
        all_functional_groups = carboxylic_groups + sulfonic_groups + phosphonic_groups
        
        return cf_groups, all_functional_groups


class GraphCoarsener:
    """
    Creates hierarchical graph representations of PFAS molecules.
    """
    
    def __init__(self, use_3d_coords: bool = True):
        """
        Initialize the GraphCoarsener.
        
        Args:
            use_3d_coords: Whether to include 3D coordinates in coarsened graphs
        """
        self.use_3d_coords = use_3d_coords
        self.functional_group_identifier = FunctionalGroupIdentifier()
    
    def _create_cluster_mapping(self, mol: Chem.Mol) -> Dict[int, int]:
        """
        Create a mapping from atom indices to cluster indices for functional group level.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping atom indices to cluster indices
        """
        # Identify functional groups
        cf_groups, functional_groups = self.functional_group_identifier.identify_all_functional_groups(mol)
        
        # Create initial mapping with each atom as its own cluster
        cluster_mapping = {i: i for i in range(mol.GetNumAtoms())}
        next_cluster_id = mol.GetNumAtoms()
        
        # Assign cluster IDs to functional groups (excluding CF groups)
        for group in functional_groups:
            for atom_idx in group:
                cluster_mapping[atom_idx] = next_cluster_id
            next_cluster_id += 1
        
        # Handle CF groups - they're already identified at the atom level
        
        return cluster_mapping
    
    def _create_structural_mapping(self, 
                                  mol: Chem.Mol, 
                                  cluster_mapping: Dict[int, int]) -> Dict[int, int]:
        """
        Create a mapping from cluster indices to structural motif indices.
        
        Args:
            mol: RDKit molecule
            cluster_mapping: Mapping from atom indices to functional group cluster indices
            
        Returns:
            Dictionary mapping cluster indices to structural motif indices
        """
        builder = MolecularGraphBuilder()
        distance_features = builder._calculate_distance_features(mol)
        
        # Get unique cluster IDs
        unique_clusters = set(cluster_mapping.values())
        
        # Initialize structural mapping
        structural_mapping = {}
        
        # Identify fluorinated tail (0) and head groups (1)
        head_group_cluster = 0
        tail_group_cluster = 1
        
        for atom_idx, cluster_id in cluster_mapping.items():
            # Skip if this cluster is already mapped
            if cluster_id in structural_mapping:
                continue
            
            # Check if atom is in head group based on distance features
            is_head = False
            if atom_idx in distance_features:
                is_head = distance_features[atom_idx]['is_head_group'] > 0.5
            
            # Assign cluster to head or tail structural motif
            if is_head:
                structural_mapping[cluster_id] = head_group_cluster
            else:
                structural_mapping[cluster_id] = tail_group_cluster
        
        return structural_mapping
    
    def _compute_coarsened_features(self, 
                                   data: Data, 
                                   cluster_mapping: Dict[int, int]) -> torch.Tensor:
        """
        Compute node features for the coarsened graph by aggregating original features.
        
        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices
            
        Returns:
            Tensor of node features for the coarsened graph
        """
        # Get original node features
        x = data.x
        
        # Create a reverse mapping from cluster IDs to atom indices
        clusters = {}
        for atom_idx, cluster_id in cluster_mapping.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(atom_idx)
        
        # Calculate features for each cluster by averaging the features of its atoms
        coarsened_features = []
        for cluster_id in sorted(clusters.keys()):
            atom_indices = clusters[cluster_id]
            # Average the features of all atoms in this cluster
            cluster_features = torch.mean(x[atom_indices], dim=0)
            coarsened_features.append(cluster_features)
        
        return torch.stack(coarsened_features)
    
    def _compute_coarsened_edges(self, 
                                data: Data, 
                                cluster_mapping: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edges for the coarsened graph.
        
        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices
            
        Returns:
            Tuple of (edge_index, edge_attr) for the coarsened graph
        """
        # Get original edges
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Create a set of edges between clusters
        cluster_edges = set()
        cluster_edge_attrs = {}
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            # Map to cluster IDs
            src_cluster = cluster_mapping[src]
            dst_cluster = cluster_mapping[dst]
            
            # Skip self-loops within the same cluster
            if src_cluster == dst_cluster:
                continue
            
            # Add edge between clusters
            edge = (src_cluster, dst_cluster)
            cluster_edges.add(edge)
            
            # Aggregate edge attributes
            if edge not in cluster_edge_attrs:
                cluster_edge_attrs[edge] = []
            cluster_edge_attrs[edge].append(edge_attr[i])
        
        # Create edge index and attribute tensors
        coarsened_edge_index = []
        coarsened_edge_attr = []
        
        for edge in cluster_edges:
            src_cluster, dst_cluster = edge
            coarsened_edge_index.append([src_cluster, dst_cluster])
            
            # Average edge attributes
            avg_attr = torch.mean(torch.stack(cluster_edge_attrs[edge]), dim=0)
            coarsened_edge_attr.append(avg_attr)
        
        if not coarsened_edge_index:
            # No edges in the coarsened graph
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, edge_attr.shape[1]), dtype=torch.float)
        
        return (torch.tensor(coarsened_edge_index, dtype=torch.long).t(), 
                torch.stack(coarsened_edge_attr))
    
    def _compute_coarsened_positions(self, 
                                    data: Data, 
                                    cluster_mapping: Dict[int, int]) -> Optional[torch.Tensor]:
        """
        Compute 3D positions for the coarsened graph by averaging positions of atoms in each cluster.
        
        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices
            
        Returns:
            Tensor of 3D positions for the coarsened graph, or None if no positions are available
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return None
        
        # Create a reverse mapping from cluster IDs to atom indices
        clusters = {}
        for atom_idx, cluster_id in cluster_mapping.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(atom_idx)
        
        # Calculate positions for each cluster by averaging the positions of its atoms
        coarsened_positions = []
        for cluster_id in sorted(clusters.keys()):
            atom_indices = clusters[cluster_id]
            # Average the positions of all atoms in this cluster
            cluster_pos = torch.mean(data.pos[atom_indices], dim=0)
            coarsened_positions.append(cluster_pos)
        
        return torch.stack(coarsened_positions)
    
    def create_functional_group_graph(self, data: Data, mol: Chem.Mol) -> Data:
        """
        Create a coarsened graph at the functional group level.
        
        Args:
            data: Original PyTorch Geometric Data object
            mol: RDKit molecule
            
        Returns:
            PyTorch Geometric Data object for the coarsened graph
        """
        # Create mapping from atoms to functional group clusters
        cluster_mapping = self._create_cluster_mapping(mol)
        
        # Compute features, edges, and positions for the coarsened graph
        coarsened_x = self._compute_coarsened_features(data, cluster_mapping)
        coarsened_edge_index, coarsened_edge_attr = self._compute_coarsened_edges(data, cluster_mapping)
        coarsened_pos = self._compute_coarsened_positions(data, cluster_mapping) if self.use_3d_coords else None
        
        # Create the coarsened graph
        coarsened_data = Data(
            x=coarsened_x,
            edge_index=coarsened_edge_index,
            edge_attr=coarsened_edge_attr,
            pos=coarsened_pos,
            y=data.y,  # Keep the same global features
            num_nodes=coarsened_x.shape[0],
            # Store the cluster mapping for reference
            cluster_mapping=cluster_mapping
        )
        
        # Transfer any custom attributes from the original graph
        for key in data.keys:
            if key not in ['x', 'edge_index', 'edge_attr', 'pos', 'y', 'num_nodes']:
                coarsened_data[key] = data[key]
        
        return coarsened_data
    
    def create_structural_motif_graph(self, data: Data, mol: Chem.Mol) -> Data:
        """
        Create a coarsened graph at the structural motif level.
        
        Args:
            data: Original PyTorch Geometric Data object (can be atom-level or functional group level)
            mol: RDKit molecule
            
        Returns:
            PyTorch Geometric Data object for the coarsened graph
        """
        # If input is atom-level, first create functional group level
        if not hasattr(data, 'cluster_mapping'):
            data = self.create_functional_group_graph(data, mol)
        
        # Get the cluster mapping from the functional group level
        cluster_mapping = data.cluster_mapping
        
        # Create mapping from functional groups to structural motifs
        structural_mapping = self._create_structural_mapping(mol, cluster_mapping)
        
        # Create a combined mapping from atoms to structural motifs
        combined_mapping = {atom_idx: structural_mapping[cluster_id] 
                           for atom_idx, cluster_id in cluster_mapping.items()}
        
        # Compute features, edges, and positions for the structural motif graph
        motif_x = self._compute_coarsened_features(data, combined_mapping)
        motif_edge_index, motif_edge_attr = self._compute_coarsened_edges(data, combined_mapping)
        motif_pos = self._compute_coarsened_positions(data, combined_mapping) if self.use_3d_coords else None
        
        # Create the structural motif graph
        motif_data = Data(
            x=motif_x,
            edge_index=motif_edge_index,
            edge_attr=motif_edge_attr,
            pos=motif_pos,
            y=data.y,  # Keep the same global features
            num_nodes=motif_x.shape[0],
            # Store the mappings for reference
            cluster_mapping=cluster_mapping,
            structural_mapping=structural_mapping,
            combined_mapping=combined_mapping
        )
        
        # Transfer any custom attributes from the original graph
        for key in data.keys:
            if key not in ['x', 'edge_index', 'edge_attr', 'pos', 'y', 'num_nodes', 
                          'cluster_mapping', 'structural_mapping', 'combined_mapping']:
                motif_data[key] = data[key]
        
        return motif_data
    
    def create_hierarchical_graphs(self, data: Data, mol: Chem.Mol) -> Dict[str, Data]:
        """
        Create hierarchical graph representations at multiple levels of coarseness.
        
        Args:
            data: Original PyTorch Geometric Data object (atom-level)
            mol: RDKit molecule
            
        Returns:
            Dictionary of graphs at different levels:
            - 'atom': Original atom-level graph
            - 'functional_group': Functional group level graph
            - 'structural_motif': Structural motif level graph
        """
        # Create functional group level graph
        functional_group_graph = self.create_functional_group_graph(data, mol)
        
        # Create structural motif level graph
        structural_motif_graph = self.create_structural_motif_graph(functional_group_graph, mol)
        
        return {
            'atom': data,
            'functional_group': functional_group_graph,
            'structural_motif': structural_motif_graph
        } 