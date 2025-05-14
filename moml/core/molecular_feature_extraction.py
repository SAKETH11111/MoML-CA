"""
Molecular Descriptors for PFAS Analysis

This module provides shared functionality for analyzing PFAS molecular structures,
including functional group detection, feature extraction, and chemical property
calculations used across the MGNN package.
"""

from rdkit import Chem
from typing import Dict, List, Tuple, Set
import numpy as np

class FunctionalGroupDetector:
    """
    Class for detecting functional groups in molecules.
    
    This is the single source of truth for functional group detection in the MoML library.
    All functional group detection should use this class to maintain consistency.
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
    def is_in_carboxylic_group(atom: Chem.Atom) -> bool:
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
    
    @staticmethod
    def is_in_sulfonic_group(atom: Chem.Atom) -> bool:
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
    
    @staticmethod
    def is_in_phosphonic_group(atom: Chem.Atom) -> bool:
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

    @staticmethod
    def find_cf_groups(mol: Chem.Mol) -> Dict[int, str]:
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
    def find_cf3_groups(mol: Chem.Mol) -> List[int]:
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
    
    @classmethod
    def identify_carboxylic_groups(cls, mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify COOH groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a COOH group
        """
        carboxylic_groups = []
        
        # Find central carbon atoms of carboxylic groups
        for atom in mol.GetAtoms():
            if cls.is_in_carboxylic_group(atom):
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
    
    @classmethod
    def identify_sulfonic_groups(cls, mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify SO3H groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a SO3H group
        """
        sulfonic_groups = []
        
        # Find central sulfur atoms of sulfonic groups
        for atom in mol.GetAtoms():
            if cls.is_in_sulfonic_group(atom):
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
    
    @classmethod
    def identify_phosphonic_groups(cls, mol: Chem.Mol) -> List[Set[int]]:
        """
        Identify PO3H2 groups and return sets of atom indices for each group.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of sets, where each set contains atom indices belonging to a PO3H2 group
        """
        phosphonic_groups = []
        
        # Find central phosphorus atoms of phosphonic groups
        for atom in mol.GetAtoms():
            if cls.is_in_phosphonic_group(atom):
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
    def find_functional_groups(cls, mol: Chem.Mol) -> List[int]:
        """
        Find all functional groups (COOH, SO3H, PO3H2) in the molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of atom indices corresponding to the central atoms of functional groups
        """
        functional_groups = []
        for atom in mol.GetAtoms():
            if (cls.is_in_carboxylic_group(atom) or 
                cls.is_in_sulfonic_group(atom) or 
                cls.is_in_phosphonic_group(atom)):
                functional_groups.append(atom.GetIdx())
        return functional_groups
    
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
        cf_groups = cls.find_cf_groups(mol)
        
        # Identify other functional groups
        carboxylic_groups = cls.identify_carboxylic_groups(mol)
        sulfonic_groups = cls.identify_sulfonic_groups(mol)
        phosphonic_groups = cls.identify_phosphonic_groups(mol)
        
        # Combine all non-CF functional groups
        all_functional_groups = carboxylic_groups + sulfonic_groups + phosphonic_groups
        
        return cf_groups, all_functional_groups

    def get_all_functional_groups(self, mol) -> dict:
        """
        Comprehensive function to detect all functional groups in one pass.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping functional group names to atom indices
        """
        groups = {
            'cf3_groups': self.find_cf3_groups(mol),
            'cf2_groups': self.find_cf2_groups(mol),
            'cf_groups': self.find_cf_groups(mol),
            'carboxylic_groups': self.find_carboxylic_groups(mol),
            'sulfonic_groups': self.find_sulfonic_groups(mol),
            'phosphonic_groups': self.find_phosphonic_groups(mol),
            'amino_groups': self.find_amino_groups(mol),
            'hydroxyl_groups': self.find_hydroxyl_groups(mol)
        }
        return groups

class MolecularFeatureExtractor:
    """
    Extracts features from molecular structures for graph representation.
    
    Provides common feature extraction methods used across the MGNN package.
    """
    
    # Common atom and bond features mapping
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
    
    @staticmethod
    def one_hot_encoding(value: any, choices: list) -> List[int]:
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
    
    @classmethod
    def calculate_distance_features(cls, mol: Chem.Mol) -> Dict[int, Dict[str, float]]:
        """
        Calculate distance-based features for PFAS structure analysis.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary mapping atom indices to distance-based features
        """
        # Find CF3 groups and functional groups
        detector = FunctionalGroupDetector()
        cf3_groups = detector.find_cf3_groups(mol)
        functional_groups = detector.find_functional_groups(mol)
        
        # Calculate distance features for each atom
        distances = {}
        for atom_idx in range(mol.GetNumAtoms()):
            # Distance to nearest CF3 group
            min_dist_cf3 = float('inf')
            if not cf3_groups:
                min_dist_cf3 = -1
            else:
                for cf3_idx in cf3_groups:
                    if atom_idx == cf3_idx:
                        min_dist_cf3 = 0
                        break
                    # Use RDKit's built-in shortest path method
                    path = Chem.GetShortestPath(mol, atom_idx, cf3_idx)
                    if path: # path can be empty if no path exists
                        dist = len(path) - 1
                        if dist < min_dist_cf3:
                            min_dist_cf3 = dist
                if min_dist_cf3 == float('inf'): # If still inf, means no path found or cf3_groups was empty initially
                    min_dist_cf3 = -1
            
            # Distance to nearest functional group
            min_dist_func = float('inf')
            if not functional_groups:
                min_dist_func = -1
            else:
                for func_idx in functional_groups:
                    if atom_idx == func_idx:
                        min_dist_func = 0
                        break
                    path = Chem.GetShortestPath(mol, atom_idx, func_idx)
                    if path: # path can be empty if no path exists
                        dist = len(path) - 1
                        if dist < min_dist_func:
                            min_dist_func = dist
                if min_dist_func == float('inf'): # If still inf, means no path found
                    min_dist_func = -1
            
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
    
    @classmethod
    def calculate_bond_lengths(cls, mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
        """
        Calculate bond lengths from 3D coordinates.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            
        Returns:
            Dictionary mapping bond indices (atom_idx1, atom_idx2) to bond lengths
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule does not have 3D coordinates")
            
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

def calculate_molecular_descriptors(mol) -> dict:
    """
    Calculate molecular descriptors for a molecule.
    
    This is the single source of truth for molecular descriptor calculation.
    All descriptor calculations should use this function to maintain consistency.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Dictionary of molecular descriptors
    """
    from rdkit.Chem import Descriptors, Lipinski
    
    if mol is None:
        return {}
    
    descriptors = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'h_bond_donors': Lipinski.NumHDonors(mol),
        'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
        'topological_polar_surface_area': Descriptors.TPSA(mol),
        'fraction_sp3': Descriptors.FractionCSP3(mol)
    }
    
    return descriptors

def extract_fingerprints(mol, fingerprint_type='morgan', radius=2, nBits=2048):
    """
    Extract molecular fingerprints for a molecule.
    
    Args:
        mol: RDKit molecule object
        fingerprint_type: Type of fingerprint to generate (morgan, maccs, etc.)
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprints
        
    Returns:
        Fingerprint as a bit vector or array
    """
    from rdkit.Chem import AllChem
    from rdkit.Chem import MACCSkeys
    import numpy as np
    
    if mol is None:
        return None
    
    if fingerprint_type.lower() == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    elif fingerprint_type.lower() == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    elif fingerprint_type.lower() == 'rdkit':
        fp = Chem.RDKFingerprint(mol, fpSize=nBits)
        return np.array(fp)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")