"""
Molecular Graph Representation for PFAS Molecules

This module provides functionality to convert PFAS molecules into graph 
representations suitable for graph neural networks using PyTorch Geometric.
It also supports JSON-based graph representations for interoperability.
"""

import os
import json
import torch
import logging
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

from moml.core.molecular_descriptors import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor
)

# Configure logging
logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    'MolecularGraphProcessor',
    
    # Factory functions
    'create_graph_processor',
    
    # Utility functions
    'mol_file_to_graph',
    'graph_to_device',
    'collate_graphs',
    'find_charges_file',
    'read_charges_from_file',
    'create_molecular_graph_json',
    'batch_create_graphs_from_molecules'
]

class MolecularGraphProcessor:
    """
    A comprehensive class for processing molecular graphs.
    
    This class handles the creation and processing of molecular graphs from different
    input formats, providing a consistent interface for graph operations used in MGNN.
    """
    
    # Common atom and bond features
    ATOM_FEATURES = MolecularFeatureExtractor.ATOM_FEATURES
    
    BOND_FEATURES = MolecularFeatureExtractor.BOND_FEATURES
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MolecularGraphProcessor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.use_partial_charges = self.config.get('use_partial_charges', True)
        self.use_3d_coords = self.config.get('use_3d_coords', True)
        self.use_pfas_specific_features = self.config.get('use_pfas_specific_features', True)
        
        # Create instances of helper classes
        self.feature_extractor = MolecularFeatureExtractor()
        self.functional_group_detector = FunctionalGroupDetector()
    
    @property
    def atom_feature_dim(self):
        """
        Get the dimension of atom feature vectors.
        
        Returns:
            int: Dimensionality of atom feature vectors
        """
        # Calculate the feature dimension based on one-hot encodings and additional features
        dim = 0
        
        # Atomic number (one-hot)
        dim += len(self.ATOM_FEATURES['atomic_num'])
        
        # Degree (one-hot)
        dim += len(self.ATOM_FEATURES['degree'])
        
        # Formal charge (one-hot)
        dim += len(self.ATOM_FEATURES['formal_charge'])
        
        # Hybridization (one-hot)
        dim += len(self.ATOM_FEATURES['hybridization'])
        
        # Aromaticity (one-hot)
        dim += len(self.ATOM_FEATURES['is_aromatic'])
        
        # Ring membership (one-hot)
        dim += len(self.ATOM_FEATURES['is_in_ring'])
        
        # Number of hydrogens (integer value)
        dim += 1
        
        # PFAS-specific features (if enabled)
        if self.use_pfas_specific_features:
            # Fluorine flag, CF flag, CF2 flag, CF3 flag
            dim += 4
            
            # Functional group flags
            dim += 3  # COOH, SO3H, PO3H2
            
        # Other atom properties
        dim += 1  # Atomic mass
        
        return dim
    
    @property
    def bond_feature_dim(self):
        """
        Get the dimension of bond feature vectors.
        
        Returns:
            int: Dimensionality of bond feature vectors
        """
        # Calculate the feature dimension based on one-hot encodings and additional features
        dim = 0
        
        # Bond type (one-hot)
        dim += len(self.BOND_FEATURES['bond_type'])
        
        # Conjugated (one-hot)
        dim += len(self.BOND_FEATURES['is_conjugated'])
        
        # In ring (one-hot)
        dim += len(self.BOND_FEATURES['is_in_ring'])
        
        # 3D features (if enabled)
        if self.use_3d_coords:
            # Bond length
            dim += 1
        
        return dim
    
    @staticmethod
    def _one_hot_encoding(value: any, choices: list) -> List[int]:
        """
        Create a one-hot encoding of a value from a list of choices.
        
        Args:
            value: The value to encode
            choices: List of possible values
            
        Returns:
            One-hot encoded list
        """
        encoding = [0] * len(choices)
        try:
            idx = choices.index(value)
            encoding[idx] = 1
        except ValueError:
            # If value not in choices, leave encoding as all zeros
            pass
        return encoding
    
    def _is_in_carboxylic_group(self, atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a carboxylic acid group (COOH).
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            True if atom is part of a carboxylic acid group, False otherwise
        """
        return self.functional_group_detector.is_in_carboxylic_group(atom)
    
    def _is_in_sulfonic_group(self, atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a sulfonic acid group (SO3H).
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            True if atom is part of a sulfonic acid group, False otherwise
        """
        return self.functional_group_detector.is_in_sulfonic_group(atom)
    
    def _is_in_phosphonic_group(self, atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a phosphonic acid group (PO3H2).
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            True if atom is part of a phosphonic acid group, False otherwise
        """
        return self.functional_group_detector.is_in_phosphonic_group(atom)
    
    def _find_cf3_groups(self, mol: Chem.Mol) -> List[int]:
        """
        Find all CF3 (trifluoromethyl) groups in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of atom indices corresponding to carbon atoms in CF3 groups
        """
        return self.functional_group_detector.find_cf3_groups(mol)
    
    def _find_functional_groups(self, mol: Chem.Mol) -> List[int]:
        """
        Find all functional groups (COOH, SO3H, PO3H2) in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of atom indices corresponding to the central atoms of functional groups
        """
        return self.functional_group_detector.find_functional_groups(mol)
    
    def _calculate_distance_features(self, mol: Chem.Mol) -> Dict[int, Dict[str, float]]:
        """
        Calculate distance-based features for PFAS structure analysis.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping atom indices to distance-based features
        """
        return self.feature_extractor.calculate_distance_features(mol)
    
    def _calculate_bond_lengths(self, mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
        """
        Calculate bond lengths from 3D coordinates.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            
        Returns:
            Dictionary mapping bond indices (atom_idx1, atom_idx2) to bond lengths
        """
        return self.feature_extractor.calculate_bond_lengths(mol)
    
    def _get_atom_features(self, atom: Chem.Atom, partial_charge: Optional[float] = None, 
                         distance_features: Optional[Dict[str, float]] = None,
                         homo_lumo_contrib: Optional[List[float]] = None) -> List[float]:
        """
        Generate a feature vector for an atom with enhanced PFAS-specific features.
        
        Args:
            atom: RDKit Atom object
            partial_charge: Optional partial charge from QM calculations
            distance_features: Optional dictionary of distance-based features
            homo_lumo_contrib: Optional HOMO/LUMO contributions
            
        Returns:
            Feature vector for the atom
        """
        features = []
        
        # Atomic number (one-hot)
        features.extend(self._one_hot_encoding(atom.GetAtomicNum(), 
                                             self.ATOM_FEATURES['atomic_num']))
        
        # Degree (one-hot)
        features.extend(self._one_hot_encoding(atom.GetDegree(), 
                                             self.ATOM_FEATURES['degree']))
        
        # Formal charge (one-hot)
        features.extend(self._one_hot_encoding(atom.GetFormalCharge(), 
                                             self.ATOM_FEATURES['formal_charge']))
        
        # Hybridization (one-hot)
        features.extend(self._one_hot_encoding(atom.GetHybridization(), 
                                             self.ATOM_FEATURES['hybridization']))
        
        # Aromaticity (one-hot)
        features.extend(self._one_hot_encoding(1 if atom.GetIsAromatic() else 0, 
                                             self.ATOM_FEATURES['is_aromatic']))
        
        # Ring membership (one-hot)
        features.extend(self._one_hot_encoding(1 if atom.IsInRing() else 0, 
                                             self.ATOM_FEATURES['is_in_ring']))
        
        # Number of hydrogens (integer value)
        features.append(atom.GetTotalNumHs())
        
        # Basic PFAS features (already in original implementation)
        # Check if atom is fluorine or carbon connected to fluorine
        is_f = atom.GetAtomicNum() == 9
        is_cf = False
        num_f_neighbors = 0
        
        if atom.GetAtomicNum() == 6:  # Carbon
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 9:  # Fluorine
                    is_cf = True
                    num_f_neighbors += 1
        
        features.append(float(is_f))
        features.append(float(is_cf))
        
        # Enhanced PFAS-specific features
        if self.use_pfas_specific_features:
            # Number of fluorine neighbors
            features.append(float(num_f_neighbors))
            
            # Functional group membership
            is_carboxylic = float(self._is_in_carboxylic_group(atom))
            is_sulfonic = float(self._is_in_sulfonic_group(atom))
            is_phosphonic = float(self._is_in_phosphonic_group(atom))
            features.append(is_carboxylic)
            features.append(is_sulfonic)
            features.append(is_phosphonic)
            
            # Distance-based features
            if distance_features is not None:
                features.append(float(distance_features.get('dist_to_cf3', -1.0)) if distance_features.get('dist_to_cf3', -1) != -1 else -1.0)
                features.append(float(distance_features.get('dist_to_functional', -1.0)) if distance_features.get('dist_to_functional', -1) != -1 else -1.0)
                features.append(distance_features.get('is_head_group', 0.0))
        
        # Add partial charge as feature if available
        if self.use_partial_charges and partial_charge is not None:
            features.append(partial_charge)
        
        # Add HOMO/LUMO contributions if available
        if homo_lumo_contrib is not None:
            features.extend(homo_lumo_contrib)
            
        return features
    
    def _get_bond_features(self, bond: Chem.Bond, bond_length: Optional[float] = None) -> List[float]:
        """
        Generate a feature vector for a bond with enhanced PFAS-specific features.
        
        Args:
            bond: RDKit Bond object
            bond_length: Optional bond length from 3D coordinates
            
        Returns:
            Feature vector for the bond
        """
        features = []
        
        # Bond type (one-hot)
        features.extend(self._one_hot_encoding(bond.GetBondType(), 
                                             self.BOND_FEATURES['bond_type']))
        
        # Conjugation (one-hot)
        features.extend(self._one_hot_encoding(1 if bond.GetIsConjugated() else 0, 
                                             self.BOND_FEATURES['is_conjugated']))
        
        # Ring membership (one-hot)
        features.extend(self._one_hot_encoding(1 if bond.IsInRing() else 0, 
                                             self.BOND_FEATURES['is_in_ring']))
        
        # Special C-F bond feature for PFAS
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        is_cf_bond = (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 9) or \
                     (begin_atom.GetAtomicNum() == 9 and end_atom.GetAtomicNum() == 6)
        features.append(float(is_cf_bond))
        
        # Enhanced PFAS-specific bond features
        if self.use_pfas_specific_features:
            # Check if it's a bond between two carbons with fluorine atoms attached
            is_cf_cf_bond = False
            if begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 6:
                begin_f_count = sum(1 for n in begin_atom.GetNeighbors() if n.GetAtomicNum() == 9)
                end_f_count = sum(1 for n in end_atom.GetNeighbors() if n.GetAtomicNum() == 9)
                is_cf_cf_bond = begin_f_count > 0 and end_f_count > 0
            features.append(float(is_cf_cf_bond))
            
            # Check if it's a bond in the fluorinated tail
            is_fluorinated_tail_bond = is_cf_bond or is_cf_cf_bond
            features.append(float(is_fluorinated_tail_bond))
            
            # Check if it's a bond in a functional group
            is_func_group_bond = (self._is_in_carboxylic_group(begin_atom) or 
                                  self._is_in_carboxylic_group(end_atom) or
                                  self._is_in_sulfonic_group(begin_atom) or
                                  self._is_in_sulfonic_group(end_atom) or
                                  self._is_in_phosphonic_group(begin_atom) or
                                  self._is_in_phosphonic_group(end_atom))
            features.append(float(is_func_group_bond))
        
        # Add bond length if available
        if bond_length is not None:
            features.append(bond_length)
        
        return features
    
    def mol_to_graph(self, 
                    mol: Chem.Mol, 
                    additional_features: Optional[Dict[str, List[float]]] = None) -> Data:
        """
        Convert an RDKit molecule to a PyTorch Geometric graph with enhanced PFAS features.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            additional_features: Optional dictionary of additional features
                - 'partial_charges': List of partial charges from QM calculations
                - 'homo_contributions': List of HOMO contributions per atom
                - 'lumo_contributions': List of LUMO contributions per atom
                
        Returns:
            PyTorch Geometric Data object representing the molecular graph
        """
        # Verify the molecule has 3D coordinates if needed
        if self.use_3d_coords and mol.GetNumConformers() == 0:
            raise ValueError("Molecule does not have 3D coordinates, but use_3d_coords is True")
        
        # Extract additional features
        partial_charges = None
        homo_contributions = None
        lumo_contributions = None
        
        if additional_features:
            partial_charges = additional_features.get('partial_charges', None)
            homo_contributions = additional_features.get('homo_contributions', None)
            lumo_contributions = additional_features.get('lumo_contributions', None)
        
        # Calculate bond lengths if using 3D coordinates
        bond_lengths = self._calculate_bond_lengths(mol) if self.use_3d_coords else None
        
        # Calculate PFAS-specific distance features if requested
        distance_features = None
        if self.use_pfas_specific_features:
            distance_features = self._calculate_distance_features(mol)
        
        # Create node features
        num_atoms = mol.GetNumAtoms()
        x = []
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            charge = partial_charges[atom_idx] if partial_charges is not None else None
            atom_distance_features = distance_features[atom_idx] if distance_features is not None else None
            
            # Create homo/lumo feature for this atom if available
            homo_lumo = None
            if homo_contributions is not None and lumo_contributions is not None:
                homo_lumo = [homo_contributions[atom_idx], lumo_contributions[atom_idx]]
                
            atom_features = self._get_atom_features(atom, charge, atom_distance_features, homo_lumo)
            x.append(atom_features)
        
        # Convert node features to tensor
        x = torch.tensor(x, dtype=torch.float)
        
        # Create edges and edge attributes
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Get bond length if available
            bond_length = bond_lengths.get((i, j)) if bond_lengths else None
            
            # Get bond features
            bond_feats = self._get_bond_features(bond, bond_length)
            
            # Add edges in both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Add same edge features for both directions
            edge_attrs.append(bond_feats)
            edge_attrs.append(bond_feats)
        
        # If no bonds, create empty tensors
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Get 3D node positions if available and needed
        pos = None
        if self.use_3d_coords:
            conformer = mol.GetConformer()
            positions = []
            for atom_idx in range(num_atoms):
                position = conformer.GetAtomPosition(atom_idx)
                positions.append([position.x, position.y, position.z])
            pos = torch.tensor(positions, dtype=torch.float)
        
        # Create additional global features for the molecule
        global_features = [
            Descriptors.ExactMolWt(mol),              # Molecular weight
            Descriptors.TPSA(mol),                    # Topological polar surface area
            Lipinski.NumHDonors(mol),                 # Number of H-bond donors
            Lipinski.NumHAcceptors(mol),              # Number of H-bond acceptors
            Descriptors.MolLogP(mol),                 # Octanol-water partition coefficient
            mol.GetNumAtoms(),                        # Total number of atoms
            len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 9])  # Count of fluorine atoms (PFAS specific)
        ]
        
        # Add PFAS-specific global features
        if self.use_pfas_specific_features:
            # Count CF3 groups
            cf3_count = len(self._find_cf3_groups(mol))
            global_features.append(cf3_count)
            
            # Count functional groups by type
            carboxylic_count = sum(1 for a in mol.GetAtoms() if self._is_in_carboxylic_group(a))
            sulfonic_count = sum(1 for a in mol.GetAtoms() if self._is_in_sulfonic_group(a))
            phosphonic_count = sum(1 for a in mol.GetAtoms() if self._is_in_phosphonic_group(a))
            global_features.extend([carboxylic_count, sulfonic_count, phosphonic_count])
            
            # Count carbon atoms with different fluorination levels
            c_with_1f = 0
            c_with_2f = 0
            c_with_3f = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon
                    f_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9)
                    if f_count == 1:
                        c_with_1f += 1
                    elif f_count == 2:
                        c_with_2f += 1
                    elif f_count == 3:
                        c_with_3f += 1
            global_features.extend([c_with_1f, c_with_2f, c_with_3f])
        
        # Create PyTorch Geometric data object
        data = Data(
            x=x,                                      # Node features
            edge_index=edge_index,                    # Edge indices
            edge_attr=edge_attr,                      # Edge features
            pos=pos,                                  # 3D coordinates (optional)
            y=torch.tensor(global_features, dtype=torch.float),  # Global molecule features
            num_nodes=num_atoms                       # Number of nodes
        )
        
        return data
    
    def file_to_graph(self, file_path: str, additional_features: Optional[Dict[str, List[float]]] = None) -> Data:
        """
        Create a graph representation from a molecule file.
        
        Args:
            file_path: Path to molecule file (MOL/SDF format)
            additional_features: Optional dictionary of additional features
                - 'partial_charges': List of partial charges from QM calculations
                - 'homo_contributions': List of HOMO contributions per atom
                - 'lumo_contributions': List of LUMO contributions per atom
                
        Returns:
            PyTorch Geometric Data object
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Molecule file not found: {file_path}")
        
        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() in ['.mol', '.sdf']:
            mol = Chem.MolFromMolFile(file_path, removeHs=False)
        elif ext.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(file_path, removeHs=False)
        elif ext.lower() in ['.smiles', '.smi']:
            with open(file_path, 'r') as f:
                smiles = f.read().strip()
            mol = Chem.MolFromSmiles(smiles)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        if mol is None:
            raise ValueError(f"Failed to read molecule from {file_path}")
        
        # Generate graph
        graph = self.mol_to_graph(mol, additional_features)
        
        return graph
    
    def smiles_to_graph(self, smiles: str, additional_features: Optional[Dict[str, List[float]]] = None) -> Data:
        """
        Create a graph representation from a SMILES string.
        
        Args:
            smiles: SMILES string
            additional_features: Optional dictionary of additional features
                
        Returns:
            PyTorch Geometric Data object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse molecule from SMILES: {smiles}")
        
        # Generate 3D coordinates if needed and not present
        if self.use_3d_coords and mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
        
        # Generate graph
        graph = self.mol_to_graph(mol, additional_features)
        
        return graph
    
    def batch_files_to_graphs(self, file_paths: List[str]) -> List[Data]:
        """
        Process multiple molecular files into graph representations.
        
        Args:
            file_paths: List of paths to molecular files
            
        Returns:
            List of graph data objects
        """
        graphs = []
        for file_path in file_paths:
            try:
                graph = self.file_to_graph(file_path)
                graphs.append(graph)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return graphs

    def mol_to_json_graph(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Convert an RDKit molecule to a JSON-serializable graph representation.
        
        Args:
            mol: RDKit molecule with 3D coordinates
                
        Returns:
            Dictionary representing the molecular graph
        """
        # Verify the molecule has 3D coordinates
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule does not have 3D coordinates")
        
        # Get atom features
        atoms = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_features = self._get_atom_features_dict(atom)
            atom_coords = mol.GetConformer().GetAtomPosition(atom_idx)
            
            atoms.append({
                "idx": atom_idx,
                "features": atom_features,
                "coords": {
                    "x": atom_coords.x,
                    "y": atom_coords.y,
                    "z": atom_coords.z
                }
            })
        
        # Get bond features
        bonds = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # Get bond length if using 3D coordinates
            bond_length = None
            if self.use_3d_coords:
                bond_lengths = self._calculate_bond_lengths(mol)
                bond_length = bond_lengths.get((begin_idx, end_idx))
            
            # Convert bond features to dictionary
            bond_features = self._get_bond_features_dict(bond, bond_length)
            
            bonds.append({
                "begin_atom_idx": begin_idx,
                "end_atom_idx": end_idx,
                "features": bond_features
            })
        
        # Calculate molecule descriptors
        descriptors = self._get_molecule_descriptors(mol)
        
        # Create graph structure
        graph = {
            "atoms": atoms,
            "bonds": bonds,
            "descriptors": descriptors,
            "quantum_properties": {}  # Empty by default
        }
        
        return graph
    
    def _get_atom_features_dict(self, atom: Chem.Atom) -> Dict[str, Any]:
        """
        Generate a dictionary of features for an atom.
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            Dictionary of atom features
        """
        # Basic atom features
        features = {
            "atomic_num": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "hybridization": int(atom.GetHybridization()),
            "num_hydrogens": atom.GetTotalNumHs(),
            "is_aromatic": int(atom.GetIsAromatic()),
            "is_in_ring": int(atom.IsInRing()),
            "degree": atom.GetDegree(),
            "implicit_valence": atom.GetImplicitValence(),
            "explicit_valence": atom.GetExplicitValence(),
        }
        
        # Add PFAS-specific features if configured
        if self.use_pfas_specific_features:
            features.update({
                "is_halogen": int(atom.GetAtomicNum() in [9, 17, 35, 53]),  # F, Cl, Br, I
                "is_fluorine": int(atom.GetAtomicNum() == 9),
                "is_carbon": int(atom.GetAtomicNum() == 6),
                "is_oxygen": int(atom.GetAtomicNum() == 8),
                "is_sulfur": int(atom.GetAtomicNum() == 16),
                "is_nitrogen": int(atom.GetAtomicNum() == 7),
                "is_phosphorus": int(atom.GetAtomicNum() == 15),
                "is_carboxylic": int(self._is_in_carboxylic_group(atom)),
                "is_sulfonic": int(self._is_in_sulfonic_group(atom)),
                "is_phosphonic": int(self._is_in_phosphonic_group(atom))
            })
        
        return features
    
    def _get_bond_features_dict(self, bond: Chem.Bond, bond_length: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a dictionary of features for a bond.
        
        Args:
            bond: RDKit Bond object
            bond_length: Optional bond length from 3D coordinates
            
        Returns:
            Dictionary of bond features
        """
        bond_type = bond.GetBondType()
        
        # Convert bond type to integer
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type_int = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type_int = 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type_int = 3
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_type_int = 4
        else:
            bond_type_int = 0
        
        # Basic bond features
        features = {
            "bond_type": bond_type_int,
            "is_conjugated": int(bond.GetIsConjugated()),
            "is_in_ring": int(bond.IsInRing()),
            "is_aromatic": int(bond.GetIsAromatic()),
        }
        
        # Add bond length if available
        if bond_length is not None:
            features["bond_length"] = bond_length
        
        return features
    
    def _get_molecule_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate basic molecular descriptors for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of molecular descriptors
        """
        descriptors = {
            "mol_weight": Descriptors.MolWt(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "qed": Descriptors.qed(mol),
            "fraction_sp3": Descriptors.FractionCSP3(mol),
        }
        
        # Add PFAS-specific descriptors if configured
        if self.use_pfas_specific_features:
            cf_groups = self.functional_group_detector.find_cf_groups(mol)
            
            descriptors.update({
                "num_fluorine_atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9),
                "num_cf_groups": sum(1 for t in cf_groups.values() if t == 'CF'),
                "num_cf2_groups": sum(1 for t in cf_groups.values() if t == 'CF2'),
                "num_cf3_groups": sum(1 for t in cf_groups.values() if t == 'CF3'),
            })
        
        return descriptors
    
    def file_to_json_graph(self, file_path: str, output_dir: Optional[str] = None, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Create a JSON graph representation from a molecule file and optionally save it.
        
        Args:
            file_path: Path to molecule file (MOL/SDF format)
            output_dir: Optional directory to save the graph JSON file
            output_filename: Optional filename for the graph JSON file
                
        Returns:
            Path to the saved graph file if output_dir is provided, None otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Get molecule ID from filename
            mol_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load molecule from file
            mol = Chem.MolFromMolFile(file_path, removeHs=False)
            if mol is None:
                logger.error(f"Failed to load molecule from {file_path}")
                return None
            
            # Convert to JSON graph
            graph = self.mol_to_json_graph(mol)
            graph["mol_id"] = mol_id
            
            # Save graph to file if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                if output_filename is None:
                    output_filename = f"{mol_id}_graph.json"
                
                output_file = os.path.join(output_dir, output_filename)
                with open(output_file, 'w') as f:
                    json.dump(graph, f, indent=2)
                
                logger.info(f"Saved graph to {output_file}")
                return output_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating graph for {file_path}: {str(e)}")
            return None

    def get_atom_features(self, mol):
        """
        Generate feature vectors for all atoms in a molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            numpy.ndarray: Array of atom features with shape [num_atoms, num_features]
        """
        if mol is None or not hasattr(mol, 'GetAtoms'):
            # Return a minimal default array with 0 atoms and the expected feature dimension
            return np.zeros((0, self.atom_feature_dim))
            
        features = []
        for atom in mol.GetAtoms():
            features.append(self._get_atom_features(atom))
            
        if not features:
            return np.zeros((0, self.atom_feature_dim))
            
        # Convert to numpy array for consistent return type
        return np.array(features, dtype=np.float32)
        
    def get_adjacency_matrix(self, mol):
        """
        Create an adjacency matrix from a molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            numpy.ndarray: Adjacency matrix as a numpy array
        """
        if mol is None or not hasattr(mol, 'GetNumAtoms') or not hasattr(mol, 'GetBonds'):
            # Return a minimal default matrix with 0x0 dimensions
            return np.zeros((0, 0), dtype=np.float32)
            
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return np.zeros((0, 0), dtype=np.float32)
            
        adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjacency[i, j] = 1
            adjacency[j, i] = 1  # Undirected graph
            
        return adjacency
    
    def process_dataframe(self, df, mol_column='rdkit_mol'):
        """
        Process a dataframe of molecules to extract graph features.
        
        Args:
            df: Pandas DataFrame containing RDKit molecules
            mol_column: Name of column containing RDKit molecules
            
        Returns:
            DataFrame with added graph features
        """
        if mol_column not in df.columns:
            raise ValueError(f"Molecule column '{mol_column}' not found in dataframe")
            
        # Initialize new columns
        df['atom_features'] = None
        df['adjacency_matrix'] = None
        df['num_atoms'] = 0
        df['graph_data'] = None
        
        # Process each molecule
        for idx, row in df.iterrows():
            mol = row[mol_column]
            # Verify we have a valid RDKit molecule object
            if mol is not None and hasattr(mol, 'GetNumAtoms'):
                try:
                    # Get atom features
                    atom_features = self.get_atom_features(mol)
                    df.at[idx, 'atom_features'] = atom_features
                    
                    # Get adjacency matrix
                    adjacency = self.get_adjacency_matrix(mol)
                    df.at[idx, 'adjacency_matrix'] = adjacency
                    
                    # Get number of atoms
                    df.at[idx, 'num_atoms'] = mol.GetNumAtoms()
                    
                    # Create a PyTorch Geometric Data object
                    try:
                        graph = self.mol_to_graph(mol)
                        df.at[idx, 'graph_data'] = graph
                    except Exception as e:
                        logger.warning(f"Error creating graph for molecule at index {idx}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing molecule at index {idx}: {e}")
        
        return df


def create_graph_processor(config: Dict[str, Any] = None) -> MolecularGraphProcessor:
    """
    Create a MolecularGraphProcessor with the specified configuration.
    
    Args:
        config: Configuration dictionary with processing parameters
        
    Returns:
        Configured MolecularGraphProcessor instance
    """
    return MolecularGraphProcessor(config)


def mol_file_to_graph(mol_file_path: str, 
                     charges_file_path: Optional[str] = None, 
                     use_3d: bool = True) -> Data:
    """
    Create a graph representation from a molecule file and optional charges file.
    
    Args:
        mol_file_path: Path to molecule file (MOL/SDF format)
        charges_file_path: Optional path to file with partial charges
        use_3d: Whether to use 3D coordinates
        
    Returns:
        PyTorch Geometric Data object
    """
    # Check if file exists
    if not os.path.exists(mol_file_path):
        raise FileNotFoundError(f"Molecule file not found: {mol_file_path}")
    
    # Read molecule from file
    mol = Chem.MolFromMolFile(mol_file_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {mol_file_path}")
    
    # Read charges if provided
    partial_charges = None
    if charges_file_path and os.path.exists(charges_file_path):
        with open(charges_file_path, 'r') as f:
            partial_charges = [float(line.strip()) for line in f if line.strip()]
    
    # Create processor
    processor = MolecularGraphProcessor({
        'use_partial_charges': (partial_charges is not None),
        'use_3d_coords': use_3d
    })
    
    # Generate graph
    additional_features = {'partial_charges': partial_charges} if partial_charges else None
    return processor.mol_to_graph(mol, additional_features)


# Utility functions for working with graphs

def graph_to_device(graph: Union[Dict[str, torch.Tensor], Any], device: torch.device) -> Union[Dict[str, torch.Tensor], Any]:
    """
    Move a graph or batch of graphs to the specified device.
    
    Args:
        graph: A graph object or dictionary containing graph tensors
        device: PyTorch device to move tensors to
        
    Returns:
        Graph with tensors moved to the specified device
    """
    if isinstance(graph, dict):
        for key, value in graph.items():
            if isinstance(value, torch.Tensor):
                graph[key] = value.to(device)
    else:
        # Assume it's a PyTorch Geometric Data object
        graph = graph.to(device)
    
    return graph


def collate_graphs(graphs: List[Data]) -> Data:
    """
    Collate a list of PyTorch Geometric Data objects into a batch.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        
    Returns:
        Batched Data object
    """
    from torch_geometric.data import Batch
    
    # Use PyTorch Geometric's built-in batching
    batch = Batch.from_data_list(graphs)
    
    return batch 


def find_charges_file(mol_file: str, charges_dir: str) -> str:
    """
    Find the corresponding charges file for a molecule file.
    
    Args:
        mol_file: Path to the molecule file
        charges_dir: Directory containing charge files
        
    Returns:
        Path to the charges file if found, otherwise None
    """
    import os
    
    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    
    # Try different possible extensions for charges file
    for ext in ['.charges', '.q', '_charges.txt', '_q.txt']:
        charge_file = os.path.join(charges_dir, f"{base_name}{ext}")
        if os.path.exists(charge_file):
            return charge_file
    
    return None


def read_charges_from_file(charge_file: str) -> list:
    """
    Read partial charges from a file.
    
    Args:
        charge_file: Path to the file containing partial charges
        
    Returns:
        List of partial charges
    """
    import numpy as np
    
    # Different charge file formats
    charges = None
    
    try:
        # Try to read as simple text file with one charge per line
        with open(charge_file, 'r') as f:
            lines = f.readlines()
        
        # Try to parse charges
        charges = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    charges.append(float(line))
                except ValueError:
                    # If line contains multiple values, take the first one
                    parts = line.split()
                    if parts:
                        try:
                            charges.append(float(parts[0]))
                        except ValueError:
                            continue
    
    except Exception as e:
        print(f"Error reading charges from {charge_file}: {e}")
        return None
    
    return charges if charges else None

def create_molecular_graph_json(mol_file: str, output_dir: str, use_pfas_features: bool = True) -> Optional[str]:
    """
    Create a molecular graph from a mol file and save as JSON.
    
    This is a utility function that uses MolecularGraphProcessor internally.
    
    Args:
        mol_file: Path to the mol file
        output_dir: Directory to save the graph
        use_pfas_features: Whether to include PFAS-specific features
        
    Returns:
        Path to the generated graph file if successful, None otherwise
    """
    # Create a processor with appropriate configuration
    config = {'use_pfas_specific_features': use_pfas_features}
    processor = create_graph_processor(config)
    
    # Generate and save the graph
    return processor.file_to_json_graph(mol_file, output_dir)

def batch_create_graphs_from_molecules(mol_dir: str, output_dir: str, 
                                      use_pfas_features: bool = True, 
                                      max_workers: int = 4) -> List[str]:
    """
    Create molecular graphs for all molecules in a directory.
    
    Args:
        mol_dir: Directory containing mol files
        output_dir: Directory to save the graphs
        use_pfas_features: Whether to include PFAS-specific features
        max_workers: Maximum number of worker processes to use
        
    Returns:
        List of paths to the generated graph files
    """
    import concurrent.futures
    
    # Check if directories exist
    if not os.path.exists(mol_dir):
        logger.error(f"Molecule directory does not exist: {mol_dir}")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of mol files
    mol_files = [os.path.join(mol_dir, f) for f in os.listdir(mol_dir) if f.endswith('.mol')]
    if not mol_files:
        logger.warning(f"No mol files found in {mol_dir}")
        return []
    
    logger.info(f"Processing {len(mol_files)} molecules in parallel with {max_workers} workers")
    
    # Process molecules in parallel
    graph_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                create_molecular_graph_json, 
                mol_file, 
                output_dir, 
                use_pfas_features
            ): mol_file for mol_file in mol_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            mol_file = futures[future]
            mol_id = os.path.splitext(os.path.basename(mol_file))[0]
            
            try:
                graph_file = future.result()
                if graph_file:
                    graph_files.append(graph_file)
                else:
                    logger.warning(f"Failed to create graph for {mol_id}")
            except Exception as e:
                logger.error(f"Error processing {mol_id}: {str(e)}")
    
    logger.info(f"Created {len(graph_files)} molecular graphs")
    return graph_files 
