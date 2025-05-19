"""
Force Field Parameter Mapper Module

This module provides functionality to convert machine learning predictions
(from MGNN models) into force field parameters for OpenMM molecular dynamics simulations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging via application entry point
# Get module logger
logger = logging.getLogger("force_field_mapper")


class ForceFieldMapper:
    """
    Converts ML model predictions to force field parameters for OpenMM MD simulations.

    This class provides methods to:
    1. Map node-level predictions (e.g., partial charges) to atoms
    2. Convert predictions to OpenMM XML force field format
    3. Validate parameter quality
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a ForceFieldMapper instance with optional configuration.
        
        If no configuration is provided, default validation cutoffs for charge balance, bond length deviation, angle deviation, and dihedral energy are set.
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
        Assigns predicted partial charges to atoms in an RDKit molecule, with optional normalization.
        
        If normalization is enabled, adjusts the charges so their sum matches the molecule's formal charge by distributing the correction evenly across all atoms.
        
        Args:
        	mol: The RDKit molecule to which charges are assigned.
        	charges: List of predicted partial charges, one per atom.
        	normalize: If True, normalizes charges to match the molecule's formal charge.
        
        Returns:
        	A dictionary mapping atom indices to their assigned partial charges.
        
        Raises:
        	ValueError: If the number of charges does not match the number of atoms in the molecule.
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
        Assigns atom types to each atom in an RDKit molecule based on element, hybridization, and aromaticity.
        
        Returns:
            A dictionary mapping atom indices to atom type strings suitable for force field parameterization.
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
        Predicts bond force constants and equilibrium bond lengths for each bond in a molecule.
        
        Uses 3D coordinates if available to calculate bond lengths; otherwise estimates them from covalent radii and bond type. Assigns default force constants based on bond order and stores parameters for both directions of each bond.
        
        Args:
        	mol: RDKit molecule object.
        	atom_types: Mapping of atom indices to atom type strings.
        
        Returns:
        	A dictionary mapping atom index pairs to bond parameter dictionaries, including atom types, force constant, equilibrium bond length, and bond type.
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
        Predicts angle force field parameters for all angles in a molecule.
        
        For each angle defined by three connected atoms, assigns atom types, a default force constant, and an equilibrium angle based on 3D geometry if available or estimated from hybridization otherwise. Returns a dictionary mapping angle index triples to parameter dictionaries.
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
        Predict dihedral parameters based on atom types and geometry.

        Args:
            mol: RDKit molecule
            atom_types: Dictionary mapping atom indices to atom types

        Returns:
            Dictionary mapping dihedral indices to dihedral parameters
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
                    for l in neighbors_k:
                        # Get atom types
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        type_k = atom_types[k]
                        type_l = atom_types[l]

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
                            pos_l = conf.GetAtomPosition(l)

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
                        dihedral_params[(i, j, k, l)] = params

                        # Also store for reverse direction (i,j,k,l) -> (l,k,j,i)
                        dihedral_params[(l, k, j, i)] = params

        return dihedral_params

    def generate_force_field_parameters(
        self, mol: Chem.Mol, partial_charges: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generates force field parameters for a molecule, including atom types, partial charges, bonds, angles, and dihedrals.
        
        If partial charges are not provided, Gasteiger charges are computed. The resulting parameters are returned as a dictionary suitable for use in OpenMM simulations.
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
            AllChem.ComputeGasteigerCharges(mol)
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
        Validates generated force field parameters against chemical and physical criteria.
        
        Checks charge balance, bond length deviations (if 3D coordinates are available), angle equilibrium values, and dihedral force constants. Records any issues found and sets flags indicating the validity of each parameter type.
        
        Args:
            parameters: Dictionary containing force field parameters for the molecule.
            mol: RDKit molecule object to validate against.
        
        Returns:
            A dictionary summarizing validation results, including pass/fail flags and any issues detected.
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
        Exports force field parameters to an OpenMM-compatible XML file.
        
        Writes atom types, residues, bonds, angles, dihedrals, and nonbonded parameters in XML format, performing necessary unit conversions and avoiding duplicate entries. Returns a tuple indicating success and a dictionary with the XML file path.
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
                
                for dihedral_idxs, dihedral_params in parameters["dihedrals"].items():
                    i, j, k, l = dihedral_idxs
                    type_i = parameters["atom_types"][i]
                    type_j = parameters["atom_types"][j]
                    type_k = parameters["atom_types"][k]
                    type_l = parameters["atom_types"][l]
                    
                    # Skip if already written
                    dihedral_key = tuple(sorted([type_i, type_j, type_k, type_l]))
                    if dihedral_key in written_dihedrals:
                        continue
                    
                    # Write each term in the dihedral
                    terms = []
                    for term in dihedral_params:
                        if term["type"] == "proper":
                            k = term["k"] * 4.184  # Convert to kJ/mol
                            phase = term["phase"] * np.pi / 180.0  # Convert to radians
                            n = term["n"]
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
        Converts MGNN node-level predictions into OpenMM-compatible force field files.
        
        Extracts partial charges from MGNN predictions, generates and validates force field parameters, and exports them as OpenMM XML files. Returns a tuple indicating success and a results dictionary containing parameters, validation results, and file paths.
        
        Args:
            mol: RDKit molecule to parameterize.
            node_predictions: MGNN node-level predictions (partial charges) as a list or dict.
            output_dir: Directory where output files will be saved.
            base_filename: Base name for the generated files.
        
        Returns:
            A tuple (success, results), where success is True if export succeeded, and results is a dictionary with parameter data, validation results, and file paths.
        """
        # Extract partial charges from node predictions
        if isinstance(node_predictions, dict) and "node_pred" in node_predictions:
            partial_charges = node_predictions["node_pred"].tolist()
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
    Creates and returns a ForceFieldMapper instance with optional configuration.
    
    Args:
        config: Optional dictionary of configuration parameters for the mapper.
    
    Returns:
        A ForceFieldMapper object initialized with the provided configuration.
    """
    return ForceFieldMapper(config)
