"""
Molecular Graph Representation for PFAS Molecules

This module provides functionality to convert PFAS molecules into graph 
representations suitable for graph neural networks using PyTorch Geometric.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from typing import Dict, List, Tuple, Optional, Union, Set


class MolecularGraphBuilder:
    """
    Creates graph representations of molecules optimized with quantum chemistry software.
    
    This class converts PFAS molecules into graph representations, where:
    - Nodes (atoms) have features including atomic properties and quantum properties
    - Edges (bonds) include bond properties and spatial information
    """
    
    # Atomic number to one-hot encoding mapping for common PFAS elements
    ATOM_FEATURES = {
        'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17],  # H, C, N, O, F, P, S, Cl
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'formal_charge': [-2, -1, 0, 1, 2],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP, 
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
        'is_aromatic': [0, 1],
        'is_in_ring': [0, 1],
    }
    
    # Bond features
    BOND_FEATURES = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'is_conjugated': [0, 1],
        'is_in_ring': [0, 1],
    }
    
    def __init__(self, use_partial_charges: bool = True, use_3d_coords: bool = True,
                 use_pfas_specific_features: bool = True):
        """
        Initialize the MolecularGraphBuilder.
        
        Args:
            use_partial_charges: Whether to include partial charges from QM calculations
            use_3d_coords: Whether to include 3D coordinates in the graph
            use_pfas_specific_features: Whether to include PFAS-specific features
        """
        self.use_partial_charges = use_partial_charges
        self.use_3d_coords = use_3d_coords
        self.use_pfas_specific_features = use_pfas_specific_features
    
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
        if atom.GetAtomicNum() != 6:  # Must be carbon
            return False
        
        # Check for C=O and C-O pattern
        o_double_bond = False
        o_single_bond = False
        
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetAtomicNum() == 8:  # Oxygen
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    o_double_bond = True
                elif bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    # Check if this O is bonded to H
                    for o_bond in other_atom.GetBonds():
                        if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                            o_single_bond = True
                            break
        
        return o_double_bond and o_single_bond
    
    def _is_in_sulfonic_group(self, atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a sulfonic acid group (SO3H).
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            True if atom is part of a sulfonic acid group, False otherwise
        """
        if atom.GetAtomicNum() != 16:  # Must be sulfur
            return False
        
        # For sulfonic acid, we need S bonded to 3 O atoms, at least one with OH
        o_count = 0
        oh_count = 0
        
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetAtomicNum() == 8:  # Oxygen
                o_count += 1
                # Check if this O is bonded to H
                for o_bond in other_atom.GetBonds():
                    if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                        oh_count += 1
                        break
        
        return o_count >= 3 and oh_count >= 1
    
    def _is_in_phosphonic_group(self, atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a phosphonic acid group (PO3H2).
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            True if atom is part of a phosphonic acid group, False otherwise
        """
        if atom.GetAtomicNum() != 15:  # Must be phosphorus
            return False
        
        # Similar to sulfonic acid check
        o_count = 0
        oh_count = 0
        
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetAtomicNum() == 8:  # Oxygen
                o_count += 1
                # Check if this O is bonded to H
                for o_bond in other_atom.GetBonds():
                    if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                        oh_count += 1
                        break
        
        return o_count >= 3 and oh_count >= 1

    def _find_cf3_groups(self, mol: Chem.Mol) -> List[int]:
        """
        Find all CF3 (trifluoromethyl) groups in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of atom indices corresponding to carbon atoms in CF3 groups
        """
        cf3_groups = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                f_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9)
                if f_neighbors == 3:
                    cf3_groups.append(atom.GetIdx())
        return cf3_groups
    
    def _find_functional_groups(self, mol: Chem.Mol) -> List[int]:
        """
        Find all functional groups (COOH, SO3H, PO3H2) in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of atom indices corresponding to the central atoms of functional groups
        """
        functional_groups = []
        for atom in mol.GetAtoms():
            if (self._is_in_carboxylic_group(atom) or 
                self._is_in_sulfonic_group(atom) or 
                self._is_in_phosphonic_group(atom)):
                functional_groups.append(atom.GetIdx())
        return functional_groups
    
    def _calculate_distance_features(self, mol: Chem.Mol) -> Dict[int, Dict[str, float]]:
        """
        Calculate distance-based features for PFAS structure analysis.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping atom indices to distance-based features
        """
        # Find CF3 groups and functional groups
        cf3_groups = self._find_cf3_groups(mol)
        functional_groups = self._find_functional_groups(mol)
        
        # Calculate distance features for each atom
        distances = {}
        for atom_idx in range(mol.GetNumAtoms()):
            # Distance to nearest CF3 group
            min_dist_cf3 = float('inf')
            for cf3_idx in cf3_groups:
                # Use RDKit's built-in shortest path method
                path = Chem.GetShortestPath(mol, atom_idx, cf3_idx)
                if path and len(path) - 1 < min_dist_cf3:
                    min_dist_cf3 = len(path) - 1
            
            if min_dist_cf3 == float('inf'):
                min_dist_cf3 = -1  # No CF3 group found
            
            # Distance to nearest functional group
            min_dist_func = float('inf')
            for func_idx in functional_groups:
                path = Chem.GetShortestPath(mol, atom_idx, func_idx)
                if path and len(path) - 1 < min_dist_func:
                    min_dist_func = len(path) - 1
            
            if min_dist_func == float('inf'):
                min_dist_func = -1  # No functional group found
            
            # Determine if atom is in head group or fluorinated tail
            is_head_group = False
            if min_dist_func != -1 and min_dist_cf3 != -1:
                is_head_group = min_dist_func < min_dist_cf3
            
            # Store distance features
            distances[atom_idx] = {
                'dist_to_cf3': min_dist_cf3,
                'dist_to_func': min_dist_func,
                'is_head_group': float(is_head_group),
            }
        
        return distances
    
    def _get_atom_features(self, atom: Chem.Atom, partial_charge: Optional[float] = None, 
                         distance_features: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Generate a feature vector for an atom with enhanced PFAS-specific features.
        
        Args:
            atom: RDKit Atom object
            partial_charge: Optional partial charge from QM calculations
            distance_features: Optional dictionary of distance-based features
            
        Returns:
            Feature vector for the atom
        """
        features = []
        
        # Atomic number (one-hot)
        features.extend(self._one_hot_encoding(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']))
        
        # Degree (one-hot)
        features.extend(self._one_hot_encoding(atom.GetDegree(), self.ATOM_FEATURES['degree']))
        
        # Formal charge (one-hot)
        features.extend(self._one_hot_encoding(atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge']))
        
        # Hybridization (one-hot)
        features.extend(self._one_hot_encoding(atom.GetHybridization(), self.ATOM_FEATURES['hybridization']))
        
        # Aromaticity (one-hot)
        features.extend(self._one_hot_encoding(1 if atom.GetIsAromatic() else 0, self.ATOM_FEATURES['is_aromatic']))
        
        # Ring membership (one-hot)
        features.extend(self._one_hot_encoding(1 if atom.IsInRing() else 0, self.ATOM_FEATURES['is_in_ring']))
        
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
                features.append(float(distance_features['dist_to_cf3']) if distance_features['dist_to_cf3'] != -1 else -1.0)
                features.append(float(distance_features['dist_to_func']) if distance_features['dist_to_func'] != -1 else -1.0)
                features.append(distance_features['is_head_group'])
        
        # Add partial charge as feature if available
        if self.use_partial_charges and partial_charge is not None:
            features.append(partial_charge)
        
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
        features.extend(self._one_hot_encoding(bond.GetBondType(), self.BOND_FEATURES['bond_type']))
        
        # Conjugation (one-hot)
        features.extend(self._one_hot_encoding(1 if bond.GetIsConjugated() else 0, self.BOND_FEATURES['is_conjugated']))
        
        # Ring membership (one-hot)
        features.extend(self._one_hot_encoding(1 if bond.IsInRing() else 0, self.BOND_FEATURES['is_in_ring']))
        
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
    
    def _calculate_bond_lengths(self, mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
        """
        Calculate bond lengths from 3D coordinates.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            
        Returns:
            Dictionary mapping bond indices (atom_idx1, atom_idx2) to bond lengths
        """
        bond_lengths = {}
        conf = mol.GetConformer()
        
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            pos1 = conf.GetAtomPosition(idx1)
            pos2 = conf.GetAtomPosition(idx2)
            
            # Calculate Euclidean distance
            length = np.sqrt((pos1.x - pos2.x)**2 + 
                             (pos1.y - pos2.y)**2 + 
                             (pos1.z - pos2.z)**2)
            
            # Store bond length (both directions)
            bond_lengths[(idx1, idx2)] = length
            bond_lengths[(idx2, idx1)] = length
            
        return bond_lengths
    
    def mol_to_graph(self, 
                     mol: Chem.Mol, 
                     partial_charges: Optional[List[float]] = None,
                     homo_lumo_contributions: Optional[List[List[float]]] = None) -> Data:
        """
        Convert an RDKit molecule to a PyTorch Geometric graph with enhanced PFAS features.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            partial_charges: Optional list of partial charges from QM calculations
            homo_lumo_contributions: Optional list of HOMO/LUMO contributions per atom
            
        Returns:
            PyTorch Geometric Data object representing the molecular graph
        """
        # Verify the molecule has 3D coordinates if needed
        if self.use_3d_coords and mol.GetNumConformers() == 0:
            raise ValueError("Molecule does not have 3D coordinates, but use_3d_coords is True")
        
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
            homo_lumo = homo_lumo_contributions[atom_idx] if homo_lumo_contributions is not None else None
            
            atom_features = self._get_atom_features(atom, charge, atom_distance_features)
            
            # Add HOMO/LUMO contributions if available
            if homo_lumo is not None:
                atom_features.extend(homo_lumo)
                
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
    
    # Create builder
    builder = MolecularGraphBuilder(
        use_partial_charges=(partial_charges is not None),
        use_3d_coords=use_3d
    )
    
    # Generate graph
    return builder.mol_to_graph(mol, partial_charges)


def batch_create_graphs(molecule_dir: str, output_dir: str) -> None:
    """
    Batch process molecule files to create graph representations.
    
    Args:
        molecule_dir: Directory containing molecule files
        output_dir: Directory to save the graph files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all molecule files
    for filename in os.listdir(molecule_dir):
        if filename.endswith('.mol') or filename.endswith('.sdf'):
            mol_path = os.path.join(molecule_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # Look for charges file with the same base name
            charges_path = None
            for ext in ['.charges', '.chg', '_charges.txt']:
                possible_path = os.path.join(molecule_dir, f"{base_name}{ext}")
                if os.path.exists(possible_path):
                    charges_path = possible_path
                    break
            
            # Create graph
            output_path = os.path.join(output_dir, f"{base_name}_graph.pt")
            try:
                graph = mol_file_to_graph(mol_path, charges_path)
                torch.save(graph, output_path)
                print(f"Created graph for {base_name}")
            except Exception as e:
                print(f"Error creating graph for {base_name}: {e}") 