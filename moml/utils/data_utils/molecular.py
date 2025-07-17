"""
moml/utils/data_utils/molecular.py

This module provides utility functions for molecular data processing, primarily
focused on feature extraction and analysis from RDKit molecule objects within
pandas DataFrames. It includes functions for creating molecule objects,
calculating physicochemical properties, and identifying specific structural
features relevant to PFAS analysis.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from moml.core.molecular_feature_extraction import FunctionalGroupDetector

# Logger for this module
logger = logging.getLogger(__name__)


def create_rdkit_mols(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    mol_col: str = "ROMol",
    mol_cache_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create RDKit molecule objects from a DataFrame column of SMILES strings.

    A new column is added to indicate the validity of each SMILES string.

    Args:
        df (pd.DataFrame): The input DataFrame.
        smiles_col (str, optional): The name of the column containing SMILES
            strings. Defaults to "SMILES".
        mol_col (str, optional): The name of the new column where RDKit
            molecule objects will be stored. Defaults to "ROMol".

    Returns:
        pd.DataFrame: The DataFrame with an added column for RDKit molecules
        and a validity flag column.

    Raises:
        ValueError: If the specified SMILES column is not found.
    """
    logger.info("Creating RDKit molecule objects from SMILES...")

    # If a cache column is provided and exists, use it to avoid re-parsing.
    if mol_cache_col and mol_cache_col in df.columns:
        logger.info(f"Using pre-parsed molecules from column '{mol_cache_col}'.")
        df[mol_col] = df[mol_cache_col].apply(lambda m: Chem.AddHs(m) if m is not None else None)
    else:
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in DataFrame.")

        def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
            """Safely convert a single SMILES string to an RDKit molecule."""
            if pd.notna(smiles):
                try:
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol:
                        return Chem.AddHs(mol)
                except Exception as e:
                    logger.debug(f"Failed to parse SMILES '{smiles}': {e}")
            return None

        df[mol_col] = df[smiles_col].apply(smiles_to_mol)

    validity_col = f"is_valid_{smiles_col}"
    df[validity_col] = df[mol_col].notna()

    valid_count = df[validity_col].sum()
    logger.info(
        f"Processed {len(df)} SMILES strings. "
        f"{valid_count} were valid, {len(df) - valid_count} were invalid."
    )
    return df


def calculate_molecular_properties(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """
    Calculate a suite of molecular descriptors and properties.

    Args:
        df (pd.DataFrame): The input DataFrame containing a column of RDKit
            molecule objects.
        mol_col (str, optional): The name of the column with RDKit molecules.
            Defaults to "ROMol".

    Returns:
        pd.DataFrame: The DataFrame with added columns for each calculated property.
    """
    logger.info("Calculating molecular properties and descriptors...")

    def get_descriptor(mol: Chem.Mol, func, default=None):
        """Helper to safely apply a descriptor function."""
        return func(mol) if mol else default

    # Physicochemical properties
    df["MW_RDKit"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.ExactMolWt))  # type: ignore
    df["Rotatable_Bonds"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.NumRotatableBonds))  # type: ignore
    df["H_Acceptors"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.NumHAcceptors))  # type: ignore
    df["H_Donors"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.NumHDonors))  # type: ignore

    # Ring and aromaticity information
    df["Ring_Count"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.RingCount))  # type: ignore
    df["Aromatic_Rings"] = df[mol_col].apply(lambda m: get_descriptor(m, Descriptors.NumAromaticRings))  # type: ignore

    # Atom counts
    df["C_Count"] = df[mol_col].apply(lambda m: sum(a.GetSymbol() == 'C' for a in m.GetAtoms()) if m else 0)
    df["F_Count"] = df[mol_col].apply(lambda m: sum(a.GetSymbol() == 'F' for a in m.GetAtoms()) if m else 0)

    # Fluorine-to-Carbon ratio
    df["F_to_C_Ratio"] = df.apply(
        lambda row: row["F_Count"] / row["C_Count"] if row["C_Count"] > 0 else 0, axis=1
    )

    # Lower-case variant expected by unit tests
    df["f_to_c_ratio"] = df["F_to_C_Ratio"]

    # Average F per carbon (alias for ratio)
    df["avg_f_per_c"] = df["F_to_C_Ratio"]

    logger.info("Successfully added molecular property columns.")
    return df


def categorize_molecular_features(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """
    Categorize molecules based on key structural features.

    This function adds several boolean flags to the DataFrame, such as whether
    a molecule is aromatic, cyclic, or branched.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mol_col (str, optional): The column containing RDKit molecules.
            Defaults to "ROMol".

    Returns:
        pd.DataFrame: The DataFrame with added categorical feature columns.
    """
    logger.info("Categorizing molecules based on structural features...")

    def has_feature(mol: Chem.Mol, func) -> bool:
        """Helper to safely check for a boolean feature."""
        return func(mol) if mol else False

    df["is_aromatic"] = df[mol_col].apply(
        lambda m: has_feature(m, lambda mol: any(a.GetIsAromatic() for a in mol.GetAtoms()))
    )
    df["is_cyclic"] = df[mol_col].apply(
        lambda m: has_feature(m, lambda mol: Descriptors.RingCount(mol) > 0)  # type: ignore
    )
    df["is_branched"] = df[mol_col].apply(
        lambda m: has_feature(m, lambda mol: any(a.GetDegree() > 2 for a in mol.GetAtoms()))
    )
    df["has_fluorine"] = df[mol_col].apply(
        lambda m: has_feature(m, lambda mol: any(a.GetSymbol() == 'F' for a in mol.GetAtoms()))
    )

    # Provide uppercase-friendly alias expected by some downstream code/tests
    df["Has_Fluorine"] = df["has_fluorine"]
    logger.info("Successfully added feature categorization flags.")
    return df


def add_fluorinated_group_counts(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:
    """
    Count the occurrences of common fluorinated functional groups.

    This function leverages the `FunctionalGroupDetector` to find and count
    -CF3, -CF2, and -CF groups in each molecule.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mol_col (str, optional): The name of the column containing RDKit
            molecules. Defaults to "ROMol".

    Returns:
        pd.DataFrame: The DataFrame with added columns for each fluorinated
        group count.
    """
    logger.info("Adding fluorinated functional group counts...")
    detector = FunctionalGroupDetector()

    def get_group_count(mol: Chem.Mol, key: str) -> int:
        """Helper to get the count of a specific functional group list inside the detector output."""
        if mol is None:
            return 0
        groups = detector.get_all_functional_groups(mol)
        return len(groups.get(key, []))

    # Use canonical keys produced by FunctionalGroupDetector.get_all_functional_groups
    df["num_cf3_groups"] = df[mol_col].apply(lambda m: get_group_count(m, "cf3_groups"))
    df["num_cf2_groups"] = df[mol_col].apply(lambda m: get_group_count(m, "cf2_groups"))
    df["num_cf_groups"] = df[mol_col].apply(lambda m: get_group_count(m, "cf_groups"))

    logger.info("Successfully added fluorinated group counts.")
    return df

# Backward-compatibility wrappers
def extract_fluorine_count(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:  # type: ignore
    """Legacy wrapper that extracts fluorine counts.

    This function is maintained for backward compatibility. It now delegates to
    `calculate_molecular_properties` and simply returns the input DataFrame to
    preserve the original API behavior.
    """
    # Ensure the required columns exist by invoking the property calculator.
    df = calculate_molecular_properties(df, mol_col=mol_col)
    return df


def calculate_molecular_complexity(df: pd.DataFrame, mol_col: str = "ROMol") -> pd.DataFrame:  # type: ignore
    """Legacy alias for `calculate_molecular_properties`.

    Args:
        df (pd.DataFrame): Input DataFrame.
        mol_col (str, optional): Column containing RDKit molecules.

    Returns:
        pd.DataFrame: DataFrame with molecular property columns.
    """
    return calculate_molecular_properties(df, mol_col=mol_col)
