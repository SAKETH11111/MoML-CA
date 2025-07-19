"""
moml/core/molecular_feature_extraction.py

Molecular descriptors and feature extraction for PFAS analysis and machine learning.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
# RDKit imports
from rdkit import Chem

# Suppress stub attribute errors for RDKit modules
from rdkit.Chem import Descriptors, Lipinski, QED  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


class FunctionalGroupDetector:
    """
    Class for detecting functional groups in molecules.

    This is the single source of truth for functional group detection in the MoML library.
    All functional group detection should use this class to maintain consistency across
    the codebase and ensure reproducible results.
    
    The class provides methods for identifying common PFAS functional groups including:
    - CF, CF2, CF3 groups (fluorinated carbons)
    - Carboxylic acid groups (COOH)
    - Sulfonic acid groups (SO3H) 
    - Phosphonic acid groups (PO3H2)
    """

    # Define functional group types with consistent identifiers
    FUNCTIONAL_GROUPS = {
        "CF": 1,     # Carbon with one fluorine
        "CF2": 2,    # Carbon with two fluorines
        "CF3": 3,    # Trifluoromethyl group
        "COOH": 4,   # Carboxylic acid group
        "SO3H": 5,   # Sulfonic acid group
        "PO3H2": 6,  # Phosphonic acid group
        "OTHER": 0,  # Other atoms/groups
    }

    @staticmethod
    def is_in_carboxylic_group(atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a carboxylic acid group (COOH).

        This method identifies carbon atoms that are part of the characteristic
        C=O and C-OH pattern of carboxylic acids.

        Args:
            atom: RDKit Atom object to analyze

        Returns:
            True if atom is part of a carboxylic acid group, False otherwise
        """
        if atom.GetAtomicNum() != 6:  # Must be carbon
            return False

        # Check for C=O and C-O pattern characteristic of carboxylic acids
        o_double_bond = False
        o_single_bond = False

        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetAtomicNum() == 8:  # Oxygen
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    o_double_bond = True
                elif bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    # Check if this O is bonded to H (making it OH)
                    for o_bond in other_atom.GetBonds():
                        if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                            o_single_bond = True
                            break

        return o_double_bond and o_single_bond

    @staticmethod
    def is_in_sulfonic_group(atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a sulfonic acid group (SO3H).

        This method identifies sulfur atoms that are part of the characteristic
        SO3H pattern with the sulfur bonded to three oxygens, at least one with OH.

        Args:
            atom: RDKit Atom object to analyze

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
                # Check if this O is bonded to H (making it OH)
                for o_bond in other_atom.GetBonds():
                    if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                        oh_count += 1
                        break

        return o_count >= 3 and oh_count >= 1

    @staticmethod
    def is_in_phosphonic_group(atom: Chem.Atom) -> bool:
        """
        Check if atom is part of a phosphonic acid group (PO3H2).

        This method identifies phosphorus atoms that are part of the characteristic
        PO3H2 pattern with the phosphorus bonded to oxygens with OH groups.

        Args:
            atom: RDKit Atom object to analyze

        Returns:
            True if atom is part of a phosphonic acid group, False otherwise
        """
        if atom.GetAtomicNum() != 15:  # Must be phosphorus
            return False

        # Similar pattern to sulfonic acid - P bonded to oxygens with OH groups
        o_count = 0
        oh_count = 0

        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            if other_atom.GetAtomicNum() == 8:  # Oxygen
                o_count += 1
                # Check if this O is bonded to H (making it OH)
                for o_bond in other_atom.GetBonds():
                    if o_bond.GetOtherAtom(other_atom).GetAtomicNum() == 1:  # Hydrogen
                        oh_count += 1
                        break

        return o_count >= 3 and oh_count >= 1

    @staticmethod
    def find_cf_groups(mol: Chem.Mol) -> Dict[int, str]:
        """
        Identify CF, CF2, and CF3 groups in the molecule.

        This method analyzes carbon atoms and counts their fluorine neighbors
        to classify them into CF, CF2, or CF3 groups.

        Args:
            mol: RDKit molecule to analyze

        Returns:
            Dictionary mapping carbon atom indices to group types ('CF', 'CF2', 'CF3')
        """
        group_assignments = {}

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                f_neighbors = sum(
                    1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9
                )

                if f_neighbors == 1:
                    group_assignments[atom.GetIdx()] = "CF"
                elif f_neighbors == 2:
                    group_assignments[atom.GetIdx()] = "CF2"
                elif f_neighbors == 3:
                    group_assignments[atom.GetIdx()] = "CF3"

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
                f_neighbors = sum(
                    1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9
                )
                if f_neighbors == 3:
                    cf3_groups.append(atom.GetIdx())
        return cf3_groups

    @staticmethod
    def find_cf2_groups(mol: Chem.Mol) -> List[int]:
        """
        Find all CF2 groups in the molecule.

        Args:
            mol: RDKit molecule

        Returns:
            List of atom indices corresponding to carbon atoms in CF2 groups
        """
        cf2_groups = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                f_neighbors = sum(
                    1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9
                )
                if f_neighbors == 2:
                    cf2_groups.append(atom.GetIdx())
        return cf2_groups

    @staticmethod
    def find_cf1_groups(mol: Chem.Mol) -> List[int]:
        """
        Find all CF (monofluoro) groups on a carbon atom.

        Args:
            mol: RDKit molecule

        Returns:
            List of atom indices corresponding to carbon atoms in CF groups
        """
        cf1_groups = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                f_neighbors = sum(
                    1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9
                )
                if f_neighbors == 1:
                    cf1_groups.append(atom.GetIdx())
        return cf1_groups

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
            if (
                cls.is_in_carboxylic_group(atom)
                or cls.is_in_sulfonic_group(atom)
                or cls.is_in_phosphonic_group(atom)
            ):
                functional_groups.append(atom.GetIdx())
        return functional_groups

    @classmethod
    def identify_all_functional_groups(
        cls, mol: Chem.Mol
    ) -> Tuple[Dict[int, str], List[Set[int]]]:
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

    @staticmethod
    def find_hydroxyl_groups(mol: Chem.Mol) -> List[int]:
        """
        Find all hydroxyl groups in the molecule.

        Args:
            mol: RDKit molecule

        Returns:
            List of atom indices corresponding to oxygen atoms in hydroxyl groups
        """
        if mol is None:
            return []

        try:
            hydroxyl_pattern = Chem.MolFromSmarts("[OH]")
            if hydroxyl_pattern is None:
                logger.warning("Failed to create SMARTS pattern for hydroxyl groups")
                return []

            matches = mol.GetSubstructMatches(hydroxyl_pattern)
            return [match[0] for match in matches]  # Return oxygen atom indices
        except Exception as e:
            logger.error(f"Error finding hydroxyl groups: {e}")
            return []

    @classmethod
    def get_all_functional_groups(cls, mol: Chem.Mol) -> dict:
        """
        Comprehensive function to detect all functional groups in one pass.

        Args:
            mol: RDKit molecule

        Returns:
            Dictionary mapping functional group names to atom indices or sets of atom indices
        """
        groups = {
            "cf3_groups": cls.find_cf3_groups(mol),
            "cf2_groups": cls.find_cf2_groups(mol),
            "cf_groups": cls.find_cf1_groups(mol),  # For C-F (single F)
            "carboxylic_groups": cls.identify_carboxylic_groups(mol),
            "sulfonic_groups": cls.identify_sulfonic_groups(mol),
            "phosphonic_groups": cls.identify_phosphonic_groups(mol),
            "hydroxyl_groups": cls.find_hydroxyl_groups(mol),
        }
        return groups


class MolecularFeatureExtractor:
    """
    Extracts comprehensive molecular features for graph representation and machine learning.

    This class provides a comprehensive suite of feature extraction methods for
    molecular structures, with special emphasis on PFAS molecules. It includes:
    
    - Distance-based features for structural analysis
    - Bond length calculations from 3D coordinates  
    - Common molecular descriptors (MW, LogP, TPSA, etc.)
    - Fingerprint generation for similarity analysis
    - PFAS-specific structural features
    
    Methods are designed to work seamlessly with graph neural networks and other
    machine learning models in the MoML-CA package.
    """

    # Common atom and bond features mapping
    ATOM_FEATURES = {
        "atomic_num": [1, 6, 7, 8, 9, 15, 16, 17],  # H, C, N, O, F, P, S, Cl
        "degree": [0, 1, 2, 3, 4, 5, 6],
        "formal_charge": [-2, -1, 0, 1, 2],
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
        "is_aromatic": [0, 1],
        "is_in_ring": [0, 1],
    }

    BOND_FEATURES = {
        "bond_type": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        "is_conjugated": [0, 1],
        "is_in_ring": [0, 1],
    }

    @staticmethod
    def one_hot_encoding(value: Any, choices: list) -> List[int]:
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
        
        This method computes structural features based on distances within the molecule:
        - Distance to nearest CF3 group (fluorinated tail analysis)
        - Distance to nearest functional group (head group analysis)  
        - Head group classification (closer to functional group than CF3)

        Args:
            mol: RDKit molecule with valid structure

        Returns:
            Dictionary mapping atom indices to distance-based features:
            - 'dist_to_cf3': Distance to nearest CF3 group
            - 'dist_to_functional': Distance to nearest functional group
            - 'is_head_group': Boolean indicator for head group membership
        """
        # Find CF3 groups and functional groups
        detector = FunctionalGroupDetector()
        cf3_groups = detector.find_cf3_groups(mol)
        functional_groups = detector.find_functional_groups(mol)

        # Calculate distance features for each atom
        distances = {}
        for atom_idx in range(mol.GetNumAtoms()):
            # Distance to nearest CF3 group
            min_dist_cf3 = float("inf")
            if not cf3_groups:
                min_dist_cf3 = -1.0  # No groups exist in the molecule
            else:
                for cf3_idx in cf3_groups:
                    if atom_idx == cf3_idx:
                        min_dist_cf3 = 0.0  # Atom itself is the target group
                        break
                    # Use RDKit's built-in shortest path method
                    path = Chem.GetShortestPath(mol, atom_idx, cf3_idx)
                    if path:  # path can be empty if no path exists
                        dist = len(path) - 1
                        if dist < min_dist_cf3:
                            min_dist_cf3 = dist
                if min_dist_cf3 == float("inf"):  # If still inf, means no path found
                    min_dist_cf3 = float("inf")  # Groups exist but no path found

            # Distance to nearest functional group
            min_dist_func = float("inf")
            if not functional_groups:
                min_dist_func = -1.0  # No groups exist in the molecule
            else:
                for func_idx in functional_groups:
                    if atom_idx == func_idx:
                        min_dist_func = 0.0  # Atom itself is the target group
                        break
                    path = Chem.GetShortestPath(mol, atom_idx, func_idx)
                    if path:  # path can be empty if no path exists
                        dist = len(path) - 1
                        if dist < min_dist_func:
                            min_dist_func = dist
                if min_dist_func == float("inf"):  # If still inf, means no path found
                    min_dist_func = float("inf")  # Groups exist but no path found

            # Determine if atom is in head group or fluorinated tail
            is_head_group = False
            if min_dist_func != -1 and min_dist_cf3 != -1:
                is_head_group = min_dist_func < min_dist_cf3

            distances[atom_idx] = {
                "dist_to_cf3": min_dist_cf3,
                "dist_to_functional": min_dist_func,
                "is_head_group": float(is_head_group),
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
            length = np.sqrt(
                (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2
            )

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
    # Use module-level Descriptors, Lipinski, QED
    if mol is None:
        return {  # Return a dictionary with NaN or default values
            "molecular_weight": np.nan,
            "logp": np.nan,
            "num_heavy_atoms": 0,
            "num_rotatable_bonds": 0,
            "h_bond_donors": 0,
            "h_bond_acceptors": 0,
            "topological_polar_surface_area": np.nan,
            "fraction_sp3": np.nan,
            "qed": np.nan,
            "num_atoms": 0,
            "num_bonds": 0,
            "num_rings": 0,
        }

    # Ensure hydrogens are added for accurate descriptor calculation, especially num_atoms
    mol_with_hs = Chem.AddHs(mol)

    descriptors = {
        "molecular_weight": Descriptors.MolWt(mol_with_hs),
        "logp": Descriptors.MolLogP(mol_with_hs),
        "num_heavy_atoms": mol_with_hs.GetNumHeavyAtoms(),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol_with_hs),
        "h_bond_donors": Lipinski.NumHDonors(mol_with_hs),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol_with_hs),
        "topological_polar_surface_area": Descriptors.TPSA(mol_with_hs),
        "fraction_sp3": Descriptors.FractionCSP3(mol_with_hs),
        "qed": QED.qed(mol_with_hs),
        "num_atoms": mol_with_hs.GetNumAtoms(),
        "num_bonds": mol_with_hs.GetNumBonds(),
        "num_rings": Lipinski.RingCount(mol_with_hs),
    }

    return descriptors


def extract_fingerprints(mol, fingerprint_type="morgan", radius=2, nBits=2048):
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

    if mol is None:
        return None

    if fingerprint_type.lower() == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)  # type: ignore
        return np.array(fp)
    elif fingerprint_type.lower() == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)  # type: ignore
        return np.array(fp)
    elif fingerprint_type.lower() == "rdkit":
        fp = Chem.RDKFingerprint(mol, fpSize=nBits)
        return np.array(fp)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
