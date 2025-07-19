"""
moml/simulation/molecular_dynamics/force_field/mapper.py

Force Field Parameter Mapper Module for Molecular Dynamics Simulations

This module provides functionality to convert machine learning predictions
(from MGNN models) into force field parameters for OpenMM molecular dynamics simulations.
It handles the transformation of node-level predictions (such as partial charges)
into complete force field parameter sets that can be used for MD simulations.

Key features:
- Conversion of ML model predictions to atom-specific parameters
- Generation of bond, angle, and dihedral parameters
- Parameter validation for physical reasonableness
- Export to OpenMM-compatible XML format

The central class is ForceFieldMapper which orchestrates the conversion process
from ML predictions to simulation-ready force field files.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

# Configure logging via application entry point
# Get module logger
logger = logging.getLogger("force_field_mapper")


class ForceFieldMapper:
    """
    Converts ML model predictions to force field parameters for OpenMM MD simulations.

    This class provides methods to map machine learning model predictions (particularly
    from molecular graph neural networks) to molecular mechanics force field parameters
    suitable for OpenMM simulations. It handles the complete workflow from prediction
    to parameter generation to validation and export.

    Key capabilities:
    1. Map node-level predictions (e.g., partial charges) to atoms
    2. Generate bond, angle, and dihedral parameters based on molecular topology
    3. Validate parameter quality and physical reasonableness
    4. Export parameters to OpenMM XML force field format

    The typical workflow is:
    1. Create a ForceFieldMapper instance
    2. Call convert_mgnn_predictions_to_force_field() with ML model predictions
    3. Use the resulting force field files in OpenMM simulations
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ForceFieldMapper.

        Args:
            config: Additional configuration parameters for customizing the mapping process.
                   Can include validation thresholds and parameter generation options.
        """
        # Store configuration
        self.config = config or {}

        # Used for parameter validation
        self.validation_cutoffs = {
            "charge_balance": 0.01,  # Maximum allowed deviation from neutrality
            "bond_length_deviation": 0.1,  # Angstroms
            "angle_deviation": 5.0,  # Degrees
            "dihedral_energy_max": 20.0,  # kcal/mol
        }

    def map_partial_charges(self, mol: Chem.Mol, charges: List[float], normalize: bool = True) -> Dict[int, float]:
        """
        Map predicted partial charges to atoms in a molecule.

        This method assigns partial charges from machine learning predictions to each atom
        in the molecule. It can optionally normalize the charges to ensure the total molecular
        charge matches the formal charge.

        Args:
            mol: RDKit molecule object with atom information
            charges: List of partial charges corresponding to atoms in the molecule
            normalize: Whether to normalize charges to ensure neutrality (or match formal charge)

        Returns:
            Dict[int, float]: Dictionary mapping atom indices to partial charges

        Raises:
            ValueError: If the number of atoms doesn't match the number of charges
        """
        if mol.GetNumAtoms() != len(charges):
            raise ValueError(f"Number of atoms ({mol.GetNumAtoms()}) doesn't match number of charges ({len(charges)})")

        # Get formal molecular charge
        formal_charge = Chem.GetFormalCharge(mol)

        # Map charges to atoms
        charge_map = {i: charges[i] for i in range(mol.GetNumAtoms())}

        # Normalize charges if requested
        if normalize:
            total_charge = sum(charges)
            charge_correction = (formal_charge - total_charge) / len(charges)

            # Apply correction to make total charge match formal charge
            for i in range(mol.GetNumAtoms()):
                charge_map[i] += charge_correction

        return charge_map

    def assign_atom_types(self, mol: Chem.Mol) -> Dict[int, str]:
        """
        Assign atom types based on element and hybridization.

        This method creates atom type identifiers for each atom in the molecule based on
        its chemical element, hybridization state, and bonding environment. These atom types
        are used to parameterize the force field and are compatible with common force field
        naming conventions (similar to AMBER/GAFF).

        Args:
            mol: RDKit molecule object with atom information

        Returns:
            Dict[int, str]: Dictionary mapping atom indices to atom type strings
        """
        atom_types = {}

        for i, atom in enumerate(mol.GetAtoms()):
            element = atom.GetSymbol()
            hyb = atom.GetHybridization()
            is_aromatic = atom.GetIsAromatic()

            # Generate atom type based on element and hybridization
            if element == "C":
                if is_aromatic:
                    atype = "ca"  # Aromatic carbon
                elif hyb == Chem.rdchem.HybridizationType.SP3:
                    atype = "c3"  # SP3 carbon
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    atype = "c2"  # SP2 carbon
                elif hyb == Chem.rdchem.HybridizationType.SP:
                    atype = "c1"  # SP carbon
                else:
                    atype = "c3"  # Default
            elif element == "N":
                if is_aromatic:
                    atype = "na"  # Aromatic nitrogen
                elif hyb == Chem.rdchem.HybridizationType.SP3:
                    atype = "n3"  # SP3 nitrogen
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    atype = "n2"  # SP2 nitrogen
                elif hyb == Chem.rdchem.HybridizationType.SP:
                    atype = "n1"  # SP nitrogen
                else:
                    atype = "n3"  # Default
            elif element == "O":
                if hyb == Chem.rdchem.HybridizationType.SP3:
                    atype = "oh"  # Hydroxyl oxygen
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    atype = "o"  # Carbonyl oxygen
                else:
                    atype = "o"  # Default
            elif element == "F":
                atype = "f"  # Fluorine
            elif element == "H":
                # Check what the hydrogen is bonded to
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if "C" in neighbors:
                    atype = "hc"  # H attached to C
                elif "N" in neighbors:
                    atype = "hn"  # H attached to N
                elif "O" in neighbors:
                    atype = "ho"  # H attached to O
                else:
                    atype = "h1"  # Default hydrogen
            else:
                # For other elements, use lowercase symbol as type
                atype = element.lower()

            atom_types[i] = atype

        return atom_types

    def predict_bond_parameters(
        self, mol: Chem.Mol, atom_types: Dict[int, str]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Predict bond parameters based on atom types and molecular geometry.

        This method generates force field parameters for all bonds in the molecule,
        including equilibrium bond lengths and force constants. It uses 3D coordinates
        if available, or estimates parameters based on atom types and bond orders.

        Args:
            mol: RDKit molecule object with bond information
            atom_types: Dictionary mapping atom indices to atom type strings

        Returns:
            Dict[Tuple[int, int], Dict[str, float]]: Dictionary mapping bond indices
                (atom_i, atom_j) to parameter dictionaries containing:
                - type_i: Atom type for first atom
                - type_j: Atom type for second atom
                - k: Force constant (kcal/mol/A^2)
                - r_eq: Equilibrium bond length (Angstroms)
                - bond_type: Bond type as string
        """
        # Calculate equilibrium bond lengths from 3D geometry if available
        has_3d = mol.GetNumConformers() > 0

        bond_params = {}

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Get atom types
            type_i = atom_types[i]
            type_j = atom_types[j]

            # Default force constants based on bond type
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                k = 300.0  # kcal/mol/A^2
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                k = 500.0  # kcal/mol/A^2
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                k = 700.0  # kcal/mol/A^2
            elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                k = 400.0  # kcal/mol/A^2
            else:
                k = 300.0  # Default

            # Get equilibrium bond length from conformer if available
            r_eq = None
            if has_3d:
                conf = mol.GetConformer()
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                r_eq = ((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2 + (pos_i.z - pos_j.z) ** 2) ** 0.5
            else:
                # Estimate based on atom types and bond type
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)

                # Simple bond length estimate based on covalent radii
                radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05, "P": 1.07}

                r_i = radii.get(atom_i.GetSymbol(), 0.75)  # Default radius if element not found
                r_j = radii.get(atom_j.GetSymbol(), 0.75)

                # Adjust for bond type
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    factor = 1.0
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    factor = 0.9
                elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    factor = 0.8
                elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    factor = 0.95
                else:
                    factor = 1.0

                r_eq = (r_i + r_j) * factor

            # Store bond parameters
            bond_params[(i, j)] = {
                "type_i": type_i,
                "type_j": type_j,
                "k": k,
                "r_eq": r_eq,
                "bond_type": str(bond.GetBondType()),
            }

            # Also store for reverse direction (i,j) -> (j,i)
            bond_params[(j, i)] = bond_params[(i, j)]

        return bond_params

    def predict_angle_parameters(
        self, mol: Chem.Mol, atom_types: Dict[int, str]
    ) -> Dict[Tuple[int, int, int], Dict[str, float]]:
        """
        Predict angle parameters based on atom types and molecular geometry.

        This method generates force field parameters for all bond angles in the molecule,
        including equilibrium angles and force constants. It uses 3D coordinates
        if available, or estimates parameters based on central atom hybridization.

        Args:
            mol: RDKit molecule object with bond information
            atom_types: Dictionary mapping atom indices to atom type strings

        Returns:
            Dict[Tuple[int, int, int], Dict[str, float]]: Dictionary mapping angle indices
                (atom_i, atom_j, atom_k) to parameter dictionaries containing:
                - type_i: Atom type for first atom
                - type_j: Atom type for central atom
                - type_k: Atom type for third atom
                - k: Force constant (kcal/mol/rad^2)
                - theta_eq: Equilibrium angle (degrees)
        """
        # Check if 3D coordinates are available
        has_3d = mol.GetNumConformers() > 0

        # Find all angles in the molecule
        angle_params = {}

        # Iterate through central atoms
        for j in range(mol.GetNumAtoms()):
            atom_j = mol.GetAtomWithIdx(j)

            # Get neighbors of central atom
            neighbors = [n.GetIdx() for n in atom_j.GetNeighbors()]

            # If atom has at least 2 neighbors, it's part of an angle
            if len(neighbors) >= 2:
                # Generate all angle combinations
                for idx1 in range(len(neighbors)):
                    for idx2 in range(idx1 + 1, len(neighbors)):
                        i = neighbors[idx1]
                        k = neighbors[idx2]

                        # Get atom types
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        type_k = atom_types[k]

                        # Default force constant based on atom types
                        ktheta = 50.0  # kcal/mol/rad^2

                        # Get equilibrium angle from conformer if available
                        theta_eq = None
                        if has_3d:
                            conf = mol.GetConformer()
                            pos_i = conf.GetAtomPosition(i)
                            pos_j = conf.GetAtomPosition(j)
                            pos_k = conf.GetAtomPosition(k)

                            # Calculate vectors
                            v1 = np.array([pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z])
                            v2 = np.array([pos_k.x - pos_j.x, pos_k.y - pos_j.y, pos_k.z - pos_j.z])

                            # Normalize vectors
                            v1 = v1 / np.linalg.norm(v1)
                            v2 = v2 / np.linalg.norm(v2)

                            # Calculate angle in degrees
                            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                            theta_eq = np.arccos(cos_angle) * 180.0 / np.pi
                        else:
                            # Estimate based on hybridization
                            atom_j = mol.GetAtomWithIdx(j)
                            hyb = atom_j.GetHybridization()

                            if hyb == Chem.rdchem.HybridizationType.SP3:
                                theta_eq = 109.5  # Tetrahedral
                            elif hyb == Chem.rdchem.HybridizationType.SP2:
                                theta_eq = 120.0  # Trigonal planar
                            elif hyb == Chem.rdchem.HybridizationType.SP:
                                theta_eq = 180.0  # Linear
                            else:
                                theta_eq = 109.5  # Default

                        # Store angle parameters
                        angle_params[(i, j, k)] = {
                            "type_i": type_i,
                            "type_j": type_j,
                            "type_k": type_k,
                            "k": ktheta,
                            "theta_eq": theta_eq,
                        }

                        # Also store for reverse direction (i,j,k) -> (k,j,i)
                        angle_params[(k, j, i)] = angle_params[(i, j, k)]

        return angle_params

    def predict_dihedral_parameters(
        self, mol: Chem.Mol, atom_types: Dict[int, str]
    ) -> Dict[Tuple[int, int, int, int], List[Dict[str, float]]]:
        """
        Predict dihedral parameters based on atom types and molecular geometry.

        This method generates force field parameters for all dihedral angles in the molecule,
        including proper torsions with appropriate periodicity and energy barriers.
        It uses bond order and hybridization information to determine appropriate
        torsional potentials.

        Args:
            mol: RDKit molecule object with bond information
            atom_types: Dictionary mapping atom indices to atom type strings

        Returns:
            Dict[Tuple[int, int, int, int], List[Dict[str, float]]]: Dictionary mapping dihedral indices
                (atom_i, atom_j, atom_k, atom_l) to lists of parameter dictionaries.
                Each parameter dictionary contains:
                - type: Type of dihedral (e.g., "proper")
                - k: Force constant (kcal/mol)
                - n: Periodicity
                - phase: Phase offset (degrees)
                - type_i, type_j, type_k, type_l: Atom types for the four atoms
                - observed_phi: Observed dihedral angle if 3D coordinates are available
        """
        # Find all dihedral angles in the molecule
        dihedral_params = {}

        # Check if 3D coordinates are available
        has_3d = mol.GetNumConformers() > 0

        # Iterate through all bonds
        for bond in mol.GetBonds():
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()

            # Get neighbors of j excluding k
            neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]

            # Get neighbors of k excluding j
            neighbors_k = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

            # If both atoms have other neighbors, we have dihedrals
            if neighbors_j and neighbors_k:
                for i in neighbors_j:
                    for l_neighbor in neighbors_k:
                        # Get atom types
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        type_k = atom_types[k]
                        type_l = atom_types[l_neighbor]

                        # Determine if this is a special dihedral
                        bond_type = bond.GetBondType()
                        is_sp2_sp2 = (
                            mol.GetAtomWithIdx(j).GetHybridization() == Chem.rdchem.HybridizationType.SP2
                            and mol.GetAtomWithIdx(k).GetHybridization() == Chem.rdchem.HybridizationType.SP2
                        )

                        # Parameters for proper dihedrals
                        if bond_type == Chem.rdchem.BondType.SINGLE and not is_sp2_sp2:
                            # Rotatable single bond - use 3-fold potential
                            params = [
                                {"type": "proper", "k": 0.5, "n": 3, "phase": 0.0},
                                {"type": "proper", "k": 0.0, "n": 2, "phase": 180.0},
                                {"type": "proper", "k": 0.0, "n": 1, "phase": 0.0},
                            ]
                        elif bond_type == Chem.rdchem.BondType.SINGLE and is_sp2_sp2:
                            # sp2-sp2 single bond - use 2-fold potential
                            params = [
                                {"type": "proper", "k": 0.0, "n": 3, "phase": 0.0},
                                {"type": "proper", "k": 2.0, "n": 2, "phase": 180.0},
                                {"type": "proper", "k": 0.0, "n": 1, "phase": 0.0},
                            ]
                        elif bond_type == Chem.rdchem.BondType.DOUBLE:
                            # Double bond - use stiff 2-fold potential
                            params = [
                                {"type": "proper", "k": 0.0, "n": 3, "phase": 0.0},
                                {"type": "proper", "k": 10.0, "n": 2, "phase": 180.0},
                                {"type": "proper", "k": 0.0, "n": 1, "phase": 0.0},
                            ]
                        elif bond_type == Chem.rdchem.BondType.AROMATIC:
                            # Aromatic bond - use AMBER-like parameters
                            params = [
                                {"type": "proper", "k": 0.0, "n": 3, "phase": 0.0},
                                {"type": "proper", "k": 7.0, "n": 2, "phase": 180.0},
                                {"type": "proper", "k": 0.0, "n": 1, "phase": 0.0},
                            ]
                        else:
                            # Default
                            params = [{"type": "proper", "k": 1.0, "n": 2, "phase": 180.0}]

                        # If 3D coordinates are available, get the current dihedral angle
                        if has_3d:
                            conf = mol.GetConformer()
                            pos_i = conf.GetAtomPosition(i)
                            pos_j = conf.GetAtomPosition(j)
                            pos_k = conf.GetAtomPosition(k)
                            pos_l = conf.GetAtomPosition(l_neighbor)

                            # Calculate dihedral angle
                            p0 = np.array([pos_i.x, pos_i.y, pos_i.z])
                            p1 = np.array([pos_j.x, pos_j.y, pos_j.z])
                            p2 = np.array([pos_k.x, pos_k.y, pos_k.z])
                            p3 = np.array([pos_l.x, pos_l.y, pos_l.z])

                            v1 = p1 - p0
                            v2 = p2 - p1
                            v3 = p3 - p2

                            n1 = np.cross(v1, v2)
                            n2 = np.cross(v2, v3)

                            # Normalize normal vectors
                            n1 = n1 / np.linalg.norm(n1)
                            n2 = n2 / np.linalg.norm(n2)

                            # Calculate dihedral angle
                            cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)

                            # Determine sign
                            if np.dot(np.cross(n1, n2), v2) < 0:
                                phi = -np.arccos(cos_phi) * 180.0 / np.pi
                            else:
                                phi = np.arccos(cos_phi) * 180.0 / np.pi

                            # Store the observed dihedral angle
                            for p in params:
                                p["observed_phi"] = phi

                        # Add atom types to parameters
                        for p in params:
                            p["type_i"] = type_i
                            p["type_j"] = type_j
                            p["type_k"] = type_k
                            p["type_l"] = type_l

                        # Store dihedral parameters
                        dihedral_params[(i, j, k, l_neighbor)] = params

                        # Also store for reverse direction (i,j,k,l) -> (l,k,j,i)
                        dihedral_params[(l_neighbor, k, j, i)] = params

        return dihedral_params

    def generate_force_field_parameters(
        self, mol: Chem.Mol, partial_charges: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete force field parameters for a molecule.

        This method orchestrates the generation of all force field parameters needed
        for molecular dynamics simulations, including partial charges, atom types,
        and bonded parameters (bonds, angles, dihedrals).

        Args:
            mol: RDKit molecule object with 3D coordinates if available
            partial_charges: Optional list of partial charges from ML predictions.
                            If None, Gasteiger charges will be computed.

        Returns:
            Dict[str, Any]: Dictionary containing all force field parameters:
                - mol_name: Molecule name
                - atom_types: Atom type assignments
                - partial_charges: Partial charge assignments
                - bonds: Bond parameters
                - angles: Angle parameters
                - dihedrals: Dihedral parameters
        """
        # Calculate Gasteiger charges if none provided
        logger.debug(f"generate_force_field_parameters: initial partial_charges type: {type(partial_charges)}")
        if isinstance(partial_charges, list) and partial_charges:
            logger.debug(
                f"generate_force_field_parameters: initial partial_charges first element type: {type(partial_charges[0])}, length: {len(partial_charges)}"
            )
        elif partial_charges is not None:
            logger.debug(f"generate_force_field_parameters: initial partial_charges content: {partial_charges}")

        if partial_charges is None:
            logger.debug("generate_force_field_parameters: partial_charges is None, computing Gasteiger.")
            mol.UpdatePropertyCache(strict=False)
            ComputeGasteigerCharges(mol)
            partial_charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
            logger.debug(f"generate_force_field_parameters: Gasteiger charges computed: {partial_charges}")

        # Map charges to atoms
        charge_map = self.map_partial_charges(mol, partial_charges)
        logger.debug(f"generate_force_field_parameters: charge_map created: {charge_map}")

        # Assign atom types
        atom_types = self.assign_atom_types(mol)

        # Predict bond parameters
        bond_params = self.predict_bond_parameters(mol, atom_types)

        # Predict angle parameters
        angle_params = self.predict_angle_parameters(mol, atom_types)

        # Predict dihedral parameters
        dihedral_params = self.predict_dihedral_parameters(mol, atom_types)

        # Collect all parameters
        parameters = {
            "mol_name": mol.GetProp("_Name") if mol.HasProp("_Name") else "MOL",
            "atom_types": atom_types,
            "partial_charges": charge_map,
            "bonds": bond_params,
            "angles": angle_params,
            "dihedrals": dihedral_params,
        }
        logger.debug(
            f"generate_force_field_parameters: Returning parameters dict with keys: {list(parameters.keys())}, type: {type(parameters)}"
        )
        if "partial_charges" in parameters:
            logger.debug(
                f"generate_force_field_parameters: 'partial_charges' key exists. Type: {type(parameters['partial_charges'])}, Content: {parameters['partial_charges']}"
            )
        else:
            logger.error("generate_force_field_parameters: CRITICAL - 'partial_charges' key is MISSING before return.")

        return parameters

    def validate_parameters(self, parameters: Dict[str, Any], mol: Chem.Mol) -> Dict[str, Any]:
        """
        Validate force field parameters for physical reasonableness.

        This method performs a series of checks on the generated force field parameters
        to ensure they are physically reasonable and suitable for molecular dynamics
        simulations. It checks charge balance, bond lengths, angles, and dihedral energies.

        Args:
            parameters: Dictionary with force field parameters from generate_force_field_parameters
            mol: RDKit molecule object with 3D coordinates if available

        Returns:
            Dict[str, Any]: Validation results dictionary containing:
                - passed: Boolean indicating if all validation checks passed
                - issues: List of dictionaries describing validation issues
                - charge_balance_ok: Boolean for charge balance check
                - bonds_ok: Boolean for bond parameter check
                - angles_ok: Boolean for angle parameter check
                - dihedrals_ok: Boolean for dihedral parameter check
        """
        validation = {
            "passed": True,
            "issues": [],
            "charge_balance_ok": True,
            "bonds_ok": True,
            "angles_ok": True,
            "dihedrals_ok": True,
        }

        # 1. Check charge balance
        total_charge = sum(parameters["partial_charges"].values())
        formal_charge = Chem.GetFormalCharge(mol)

        charge_diff = abs(total_charge - formal_charge)
        if charge_diff > self.validation_cutoffs["charge_balance"]:
            validation["passed"] = False
            validation["charge_balance_ok"] = False
            validation["issues"].append(
                {
                    "type": "charge_balance",
                    "message": f"Total charge ({total_charge:.4f}) deviates from formal charge ({formal_charge}) by {charge_diff:.4f}, exceeding threshold of {self.validation_cutoffs['charge_balance']}",
                }
            )

        # 2. Check bond parameters
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Get actual bond length
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                actual_length = ((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2 + (pos_i.z - pos_j.z) ** 2) ** 0.5

                # Get predicted bond length
                if (i, j) in parameters["bonds"]:
                    predicted_length = parameters["bonds"][(i, j)]["r_eq"]

                    # Check if within tolerance
                    diff = abs(actual_length - predicted_length)
                    if diff > self.validation_cutoffs["bond_length_deviation"]:
                        validation["passed"] = False
                        validation["bonds_ok"] = False
                        validation["issues"].append(
                            {
                                "type": "bond_length",
                                "atoms": (i, j),
                                "message": f"Bond length for atoms {i}-{j} deviates by {diff:.4f} Å (actual: {actual_length:.4f}, predicted: {predicted_length:.4f})",
                            }
                        )

        # 3. Check angle parameters
        if "angles" in parameters:
            for angle_key, angle_param in parameters["angles"].items():
                if not (0 < angle_param.get("theta_eq", 109.5) < 180.0):  # Basic sanity check
                    validation["passed"] = False
                    validation["angles_ok"] = False
                    validation["issues"].append(
                        {
                            "type": "angle_value",
                            "angle": angle_key,
                            "message": f"Angle {angle_key} has unrealistic theta_eq: {angle_param.get('theta_eq', 'N/A'):.2f}°",
                        }
                    )
                    break
        else:
            validation["angles_ok"] = False
            validation["issues"].append({"type": "missing_angles", "message": "Angle parameters missing."})

        # 4. Check dihedral parameters
        if "dihedrals" in parameters:
            for dihedral_key, dihedral_terms in parameters["dihedrals"].items():
                for term_params in dihedral_terms:
                    if term_params.get("k", 0.0) > self.validation_cutoffs["dihedral_energy_max"]:
                        validation["passed"] = False
                        validation["dihedrals_ok"] = False
                        validation["issues"].append(
                            {
                                "type": "dihedral_energy",
                                "dihedral": dihedral_key,
                                "message": f"Dihedral {dihedral_key} term has very high k: {term_params.get('k', 0.0):.2f} kcal/mol",
                            }
                        )
                        break
                if not validation["dihedrals_ok"]:
                    break
        else:
            validation["dihedrals_ok"] = False
            validation["issues"].append({"type": "missing_dihedrals", "message": "Dihedral parameters missing."})

        return validation

    def export_to_openmm(
        self, parameters: Dict[str, Any], mol: Chem.Mol, output_dir: str, base_filename: str
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Export force field parameters to OpenMM XML format.

        This method writes the force field parameters to an OpenMM-compatible XML file
        that can be used directly in molecular dynamics simulations. It organizes the
        parameters into standard OpenMM force field components including atom types,
        bonds, angles, dihedrals, and nonbonded interactions.

        Args:
            parameters: Dictionary with force field parameters from generate_force_field_parameters
            mol: RDKit molecule object for which parameters were generated
            output_dir: Directory to save output files
            base_filename: Base name for output files (without extension)

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple containing:
                - Boolean indicating success or failure
                - Dictionary mapping file types to file paths (e.g., {"xml": "/path/to/file.xml"})

        Raises:
            OSError: If directory creation or file writing fails
            Exception: For other errors during XML generation
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get molecule name
        mol_name = parameters["mol_name"]

        # File path for XML
        xml_file = os.path.join(output_dir, f"{mol_name}.xml")

        try:
            with open(xml_file, "w") as f:
                # Write XML header
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<ForceField>\n\n')

                # Write AtomTypes section
                f.write('  <AtomTypes>\n')
                f.write('    <!-- name = unique string; class = mixing group; element = IUPAC; mass = amu -->\n')
                
                # Track written atom types to avoid duplicates
                written_types = set()
                
                for i, atype in parameters["atom_types"].items():
                    atom = mol.GetAtomWithIdx(i)
                    element = atom.GetSymbol()
                    mass = atom.GetMass()
                    
                    # Skip if already written
                    if atype in written_types:
                        continue
                        
                    f.write(f'    <Type name="{atype}" class="{atype}" element="{element}" mass="{mass:.6f}"/>\n')
                    written_types.add(atype)
                
                f.write('  </AtomTypes>\n\n')

                # Write Residues section
                f.write('  <Residues>\n')
                f.write(f'    <Residue name="{mol_name}">\n')
                
                # Write atoms
                for i in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(i)
                    atype = parameters["atom_types"][i]
                    charge = parameters["partial_charges"][i]
                    f.write(f'      <Atom name="{atom.GetSymbol()}{i+1}" type="{atype}" charge="{charge:.6f}"/>\n')
                
                # Write bonds
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    atom_i = mol.GetAtomWithIdx(i)
                    atom_j = mol.GetAtomWithIdx(j)
                    f.write(f'      <Bond atomName1="{atom_i.GetSymbol()}{i+1}" atomName2="{atom_j.GetSymbol()}{j+1}"/>\n')
                
                f.write('    </Residue>\n')
                f.write('  </Residues>\n\n')

                # Write HarmonicBondForce section
                f.write('  <HarmonicBondForce>\n')
                f.write('    <!-- Harmonic bonds: length in nm, k in kJ mol-1 nm-2 -->\n')
                
                # Track written bonds to avoid duplicates
                written_bonds = set()
                
                for bond_idxs, bond_param in parameters["bonds"].items():
                    i, j = bond_idxs
                    type_i = bond_param["type_i"]
                    type_j = bond_param["type_j"]
                    
                    # Skip if already written
                    bond_key = tuple(sorted([type_i, type_j]))
                    if bond_key in written_bonds:
                        continue
                    
                    r_eq = bond_param["r_eq"] / 10.0  # Convert to nm
                    k = bond_param["k"] * 2.0 * 418.4  # Convert to kJ/mol/nm^2
                    
                    f.write(f'    <Bond class1="{type_i}" class2="{type_j}" length="{r_eq:.6f}" k="{k:.1f}"/>\n')
                    written_bonds.add(bond_key)
                
                f.write('  </HarmonicBondForce>\n\n')

                # Write HarmonicAngleForce section
                f.write('  <HarmonicAngleForce>\n')
                f.write('    <!-- Harmonic angles: angle in rad, k in kJ mol-1 rad-2 -->\n')
                
                # Track written angles to avoid duplicates
                written_angles = set()
                
                for angle_idxs, angle_param in parameters["angles"].items():
                    i, j, k = angle_idxs
                    type_i = angle_param["type_i"]
                    type_j = angle_param["type_j"]
                    type_k = angle_param["type_k"]
                    
                    # Skip if already written
                    angle_key = tuple(sorted([type_i, type_j, type_k]))
                    if angle_key in written_angles:
                        continue
                    
                    theta_eq = angle_param["theta_eq"] * np.pi / 180.0  # Convert to radians
                    ktheta = angle_param["k"] * 2.0 * 4.184  # Convert to kJ/mol/rad^2
                    
                    f.write(f'    <Angle class1="{type_i}" class2="{type_j}" class3="{type_k}" angle="{theta_eq:.6f}" k="{ktheta:.1f}"/>\n')
                    written_angles.add(angle_key)
                
                f.write('  </HarmonicAngleForce>\n\n')

                # Write PeriodicTorsionForce section
                f.write('  <PeriodicTorsionForce>\n')
                f.write('    <!-- Proper torsions: Fourier series -->\n')
                
                # Track written dihedrals to avoid duplicates
                written_dihedrals = set()
                
                for dihedral_idxs, dihedral_param_list in parameters["dihedrals"].items():
                    i, j, k, l_neighbor = dihedral_idxs
                    type_i = parameters["atom_types"][i]
                    type_j = parameters["atom_types"][j]
                    type_k = parameters["atom_types"][k]
                    type_l = parameters["atom_types"][l_neighbor]
                    
                    # Skip if already written
                    dihedral_key = tuple(sorted([type_i, type_j, type_k, type_l]))
                    if dihedral_key in written_dihedrals:
                        continue
                    
                    # Write each term in the dihedral
                    terms = []
                    for params in dihedral_param_list:
                        if params["type"] == "proper":
                            k = params["k"] * 4.184  # Convert to kJ/mol
                            phase = params["phase"] * np.pi / 180.0  # Convert to radians
                            n = params["n"]
                            terms.append(f'periodicity{n}="{n}" phase{n}="{phase:.6f}" k{n}="{k:.6f}"')
                    
                    if terms:
                        f.write(f'    <Proper type1="{type_i}" type2="{type_j}" type3="{type_k}" type4="{type_l}" {" ".join(terms)}/>\n')
                    
                    written_dihedrals.add(dihedral_key)
                
                f.write('  </PeriodicTorsionForce>\n\n')

                # Write NonbondedForce section
                f.write('  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">\n')
                f.write('    <UseAttributeFromResidue name="charge"/>\n')
                
                # Track written atom types to avoid duplicates
                written_nb_types = set()
                
                for i, atype in parameters["atom_types"].items():
                    if atype in written_nb_types:
                        continue
                    
                    # Simplified LJ parameters
                    sigma = 0.3  # nm
                    epsilon = 0.5  # kJ/mol
                    
                    f.write(f'    <Atom type="{atype}" sigma="{sigma:.6f}" epsilon="{epsilon:.6f}"/>\n')
                    written_nb_types.add(atype)
                
                f.write('  </NonbondedForce>\n\n')

                # Close ForceField tag
                f.write('</ForceField>\n')

            return True, {"xml": xml_file}

        except Exception as e:
            logger.error(f"Error exporting to OpenMM: {str(e)}")
            return False, {}

    def convert_mgnn_predictions_to_force_field(
        self,
        mol: Chem.Mol,
        node_predictions: Union[Dict, List[float]],
        output_dir: str,
        base_filename: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Convert MGNN model predictions to force field files.

        This is the main entry point for converting machine learning predictions
        to molecular dynamics parameters. It handles the complete workflow from
        extracting partial charges from model predictions to generating and validating
        force field parameters to exporting them to OpenMM XML format.

        Args:
            mol: RDKit molecule object with 3D coordinates if available
            node_predictions: Node-level predictions from MGNN model (partial charges),
                             either as a list of values or a dictionary with 'node_pred' key
            output_dir: Directory to save output files
            base_filename: Base name for output files (without extension)

        Returns:
            Tuple[bool, Dict[str, Any]]: Tuple containing:
                - Boolean indicating success or failure
                - Dictionary with results including:
                  - parameters: Complete force field parameters
                  - validation: Validation results
                  - file_paths: Paths to generated files

        Raises:
            ValueError: If node predictions format is invalid or incompatible with molecule
        """
        # Extract partial charges from node predictions
        if isinstance(node_predictions, dict) and "node_pred" in node_predictions:
            tensor_data = node_predictions["node_pred"]
            if hasattr(tensor_data, 'tolist'):
                partial_charges = tensor_data.squeeze().tolist()
            else:
                partial_charges = tensor_data
        elif isinstance(node_predictions, list):
            partial_charges = node_predictions
        else:
            logger.error("Invalid node predictions format")
            return False, {}

        # Check if number of charges matches number of atoms
        if len(partial_charges) != mol.GetNumAtoms():
            logger.error(
                f"Number of partial charges ({len(partial_charges)}) doesn't match number of atoms ({mol.GetNumAtoms()})"
            )
            return False, {}

        # Generate force field parameters
        parameters = self.generate_force_field_parameters(mol, partial_charges)

        # Validate parameters
        validation = self.validate_parameters(parameters, mol)

        # Export parameters to files
        success, file_paths = self.export_to_openmm(parameters, mol, output_dir, base_filename)

        if not success:
            logger.error("Failed to export force field parameters")
            return False, {}

        # Prepare results
        results = {"parameters": parameters, "validation": validation, "file_paths": file_paths}

        return True, results


def create_force_field_mapper(config: Optional[Dict[str, Any]] = None) -> ForceFieldMapper:
    """
    Create a ForceFieldMapper instance.

    This is a factory function that creates and returns a ForceFieldMapper instance
    with the specified configuration. It provides a convenient way to instantiate
    the mapper without directly calling the constructor.

    Args:
        config: Optional dictionary containing configuration parameters for the
                ForceFieldMapper. Can include validation thresholds and parameter
                generation options.

    Returns:
        ForceFieldMapper: A configured ForceFieldMapper instance ready to convert
                         ML model predictions to force field parameters.

    Example:
        >>> mapper = create_force_field_mapper({"charge_balance_threshold": 0.005})
        >>> parameters = mapper.generate_force_field_parameters(mol, charges)
    """
    return ForceFieldMapper(config)
