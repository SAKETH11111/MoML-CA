#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecular Data Processing Utilities

This module contains general-purpose molecular data processing functions used across
different data processors in the MoML framework.
"""

import pandas as pd
from rdkit import Chem
import logging
import numpy as np
from moml.core.molecular_feature_extraction import FunctionalGroupDetector  # Added import

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("molecular_processing")


def create_rdkit_mols(df: pd.DataFrame, smiles_col: str = "SMILES", mol_col: str = "ROMol") -> pd.DataFrame:
    """Create RDKit molecule objects from SMILES strings.

    Args:
        df: DataFrame containing SMILES strings
        smiles_col: Name of column containing SMILES strings
        mol_col: Name of column to store RDKit molecules

    Returns:
        DataFrame with RDKit molecules and validity flags
    """
    logger.info("=== Creating RDKit Molecule Objects ===")

    valid_col = "Valid_SMILES"
    if smiles_col != "SMILES":
        valid_col = f"is_valid_{smiles_col}"

    # Check if the SMILES column exists
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataframe")

    # Create RDKit molecules and validity flags
    def smiles_to_mol_with_hs(s):
        if pd.notna(s):
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                return Chem.AddHs(mol)
        return None

    df[mol_col] = df[smiles_col].apply(smiles_to_mol_with_hs)
    df[valid_col] = df[mol_col].notna()

    # Log statistics
    valid_count = df[valid_col].sum()
    total_count = len(df)
    logger.info(f"Created RDKit molecules for {valid_count}/{total_count} SMILES strings")

    return df


def extract_fluorine_count(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """Extract fluorine count from RDKit molecules.

    Args:
        df: DataFrame containing RDKit molecules
        mol_col: Name of column containing RDKit molecules

    Returns:
        DataFrame with added fluorine count columns
    """
    logger.info("=== Extracting Fluorine Count ===")

    def count_fluorine(mol):
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

    def calc_f_percentage(mol, f_count):
        if mol is None or f_count == 0:
            return 0.0
        total_atoms = mol.GetNumAtoms()
        return (f_count / total_atoms) * 100

    # Calculate fluorine counts
    df["F_Count"] = df[mol_col].apply(count_fluorine)
    df["F_Percentage"] = df.apply(lambda x: calc_f_percentage(x[mol_col], x["F_Count"]), axis=1)

    logger.info("Added fluorine count and percentage columns")
    return df


def calculate_molecular_complexity(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """Calculate molecular complexity features.

    Args:
        df: DataFrame containing RDKit molecules
        mol_col: Name of column containing RDKit molecules

    Returns:
        DataFrame with added molecular complexity features
    """
    logger.info("=== Calculating Molecular Complexity ===")

    def calc_f_percentage(mol, f_count):
        if mol is None or f_count == 0:
            return 0.0
        total_atoms = mol.GetNumAtoms()
        return (f_count / total_atoms) * 100

    # Calculate basic molecular descriptors
    df["MW_RDKit"] = df[mol_col].apply(lambda x: Chem.Descriptors.ExactMolWt(x) if x is not None else None)
    df["Rotatable_Bonds"] = df[mol_col].apply(
        lambda x: Chem.Descriptors.NumRotatableBonds(x) if x is not None else None
    )
    df["H_Acceptors"] = df[mol_col].apply(lambda x: Chem.Descriptors.NumHAcceptors(x) if x is not None else None)
    df["H_Donors"] = df[mol_col].apply(lambda x: Chem.Descriptors.NumHDonors(x) if x is not None else None)

    # Calculate ring information
    df["Ring_Count"] = df[mol_col].apply(lambda x: Chem.Descriptors.RingCount(x) if x is not None else None)
    df["Aromatic_Rings"] = df[mol_col].apply(lambda x: Chem.Descriptors.NumAromaticRings(x) if x is not None else None)

    # Calculate carbon and fluorine counts
    df["C_Count"] = df[mol_col].apply(
        lambda x: sum(1 for atom in x.GetAtoms() if atom.GetSymbol() == "C") if x is not None else None
    )
    df["F_Count"] = df[mol_col].apply(
        lambda x: sum(1 for atom in x.GetAtoms() if atom.GetSymbol() == "F") if x is not None else None
    )

    # Calculate chain length (longest carbon chain)
    def get_chain_length(mol):
        # Compute distance matrix once
        dist_matrix = Chem.GetDistanceMatrix(mol)
        # Identify carbon atom indices
        carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C"]
        if len(carbon_indices) < 2:
            return 0
        # Extract submatrix for carbon atoms and find maximum distance
        sub_matrix = dist_matrix[np.ix_(carbon_indices, carbon_indices)]
        max_length = int(np.max(sub_matrix))
        return max_length

    df["Chain_Length"] = df[mol_col].apply(get_chain_length)

    # Calculate CF bonds
    def count_cf_bonds(mol):
        if mol is None:
            return 0
        return sum(
            1
            for bond in mol.GetBonds()
            if (bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "F")
            or (bond.GetBeginAtom().GetSymbol() == "F" and bond.GetEndAtom().GetSymbol() == "C")
        )

    df["CF_Bonds"] = df[mol_col].apply(count_cf_bonds)

    # Calculate F percentage
    df["F_Percentage"] = df.apply(lambda x: calc_f_percentage(x[mol_col], x["F_Count"]), axis=1)

    # Calculate F/C ratio
    # Ensure C_Count is not zero to avoid division by zero errors.
    # If C_Count is 0, F_to_C_Ratio can be 0 if F_Count is also 0, or undefined (e.g., np.nan) if F_Count > 0.
    # For simplicity, if C_Count is 0, F_to_C_Ratio will be 0.
    df["f_to_c_ratio"] = df.apply(
        lambda row: row["F_Count"] / row["C_Count"] if row["C_Count"] and row["C_Count"] > 0 else 0, axis=1
    )
    # avg_f_per_c is assumed to be the same as f_to_c_ratio for now
    df["avg_f_per_c"] = df["f_to_c_ratio"]

    logger.info("Added molecular complexity features, including F/C ratio.")
    return df


def categorize_molecular_features(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """Categorize molecules based on structural features.

    Args:
        df: DataFrame containing RDKit molecules
        mol_col: Name of column containing RDKit molecules

    Returns:
        DataFrame with added molecular feature categories
    """
    logger.info("=== Categorizing Molecular Features ===")

    def is_aromatic(mol):
        if mol is None:
            return False
        return any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    def has_rings(mol):
        if mol is None:
            return False
        return Chem.Descriptors.RingCount(mol) > 0

    def is_cyclic(mol):
        if mol is None:
            return False
        return Chem.Descriptors.RingCount(mol) > 0 and not any(atom.GetSymbol() == "F" for atom in mol.GetAtoms())

    def is_branched(mol):
        if mol is None:
            return False
        return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())

    def has_fluorine(mol):
        if mol is None:
            return False
        return any(atom.GetSymbol() == "F" for atom in mol.GetAtoms())

    # Add feature flags
    df["Is_Aromatic"] = df[mol_col].apply(is_aromatic)
    df["Has_Rings"] = df[mol_col].apply(has_rings)
    df["Is_Cyclic"] = df[mol_col].apply(is_cyclic)
    df["Is_Branched"] = df[mol_col].apply(is_branched)
    df["Has_Fluorine"] = df[mol_col].apply(has_fluorine)

    # Categorize chain length
    df["Chain_Category"] = pd.cut(
        df["Chain_Length"],
        bins=[-float("inf"), 2, 4, 6, float("inf")],
        labels=["Very Short", "Short", "Medium", "Long"],
    )

    logger.info("Added molecular feature categories")
    return df


def add_fluorinated_group_counts(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """
    Adds counts of CF3, CF2, and CF groups to the DataFrame.

    Args:
        df: DataFrame containing RDKit molecules.
        mol_col: Name of the column containing RDKit molecules.

    Returns:
        DataFrame with added columns: 'num_cf3_groups', 'num_cf2_groups', 'num_cf_groups'.
    """
    logger.info("=== Adding Fluorinated Group Counts ===")
    if mol_col not in df.columns:
        raise ValueError(f"Molecule column '{mol_col}' not found in DataFrame.")

    detector = FunctionalGroupDetector()

    num_cf3_groups = []
    num_cf2_groups = []
    num_cf_groups = []

    for mol in df[mol_col]:
        if mol:
            num_cf3_groups.append(len(detector.find_cf3_groups(mol)))
            num_cf2_groups.append(len(detector.find_cf2_groups(mol)))
            num_cf_groups.append(len(detector.find_cf1_groups(mol)))  # Assuming find_cf1_groups for single C-F
        else:
            num_cf3_groups.append(0)
            num_cf2_groups.append(0)
            num_cf_groups.append(0)

    df["num_cf3_groups"] = num_cf3_groups
    df["num_cf2_groups"] = num_cf2_groups
    df["num_cf_groups"] = num_cf_groups

    logger.info("Added num_cf3_groups, num_cf2_groups, num_cf_groups columns.")
    return df
