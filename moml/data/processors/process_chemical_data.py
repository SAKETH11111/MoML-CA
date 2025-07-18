"""
moml/data/processors/process_chemical_data.py

PFAS chemical list data processing module for cleaning and feature engineering.

This module provides functions for cleaning and preprocessing the PFAS Chemical
List dataset, including data standardization, feature extraction, and molecular
complexity calculations using RDKit.
"""

import logging
import re
from pathlib import Path
from typing import Optional, cast

import pandas as pd
from rdkit import Chem

from moml.utils import (
    calculate_molecular_complexity,
    categorize_molecular_features,
    clean_column_names,
    convert_numeric_columns,
    create_rdkit_mols,
    extract_fluorine_count,
    handle_missing_values,
    inspect_data,
    load_data,
    save_processed_data,
    standardize_text_data,
)

# Constants
ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = (
    ROOT_DIR / "moml" / "data" / "datasets" / "raw" / "PFAS_Chemical_List.csv"
)
CLEANED_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "chemical_list" /
    "PFAS_Chemical_List_cleaned.csv"
)
ENGINEERED_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "chemical_list" /
    "PFAS_Chemical_List_engineered.csv"
)
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "chemical_list"

COLUMN_MAPPING = {
    "DTXSID": "DTXSID",
    "PREFERRED NAME": "Preferred_Name",
    "CASRN": "CASRN",
    "INCHIKEY": "InChIKey",
    "IUPAC NAME": "IUPAC_Name",
    "SMILES": "SMILES",
    "INCHI STRING": "InChI_String",
    "MOLECULAR FORMULA": "Molecular_Formula",
    "AVERAGE MASS": "Average_Mass",
    "MONOISOTOPIC MASS": "Monoisotopic_Mass",
    "QC Level": "QC_Level",
    "# ToxCast Active": "ToxCast_Active_Count",
    "Total Assays": "Total_Assays",
    "% ToxCast Active": "ToxCast_Active_Percent",
}

NUMERIC_COLUMNS = [
    "Average_Mass",
    "Monoisotopic_Mass",
    "ToxCast_Active_Count",
    "Total_Assays",
    "ToxCast_Active_Percent",
]

TEXT_COLUMNS = [
    "Preferred_Name",
    "IUPAC_Name",
    "SMILES",
    "InChI_String",
    "Molecular_Formula",
]

# Configure logger
logger = logging.getLogger(__name__)


def clean_dtxsid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DTXSID column to extract ID from URLs.

    Extracts DTXSID values from EPA CompTox Dashboard URLs, returning
    just the identifier string (e.g., 'DTXSID1234567').

    Args:
        df: DataFrame containing DTXSID column to clean.

    Returns:
        DataFrame with cleaned DTXSID values.
    """
    logger.info("Cleaning DTXSID column")

    if "DTXSID" not in df.columns:
        logger.warning("DTXSID column not found, skipping cleaning")
        return df

    def extract_dtxsid(value: str) -> str:
        """Extract DTXSID from EPA CompTox URL or return original value."""
        if isinstance(value, str) and "comptox.epa.gov" in value:
            match = re.search(r"(DTXSID\d+)", value)
            return match.group(1) if match else value
        return value

    df["DTXSID"] = df["DTXSID"].apply(extract_dtxsid)
    logger.info("Successfully extracted DTXSID IDs from URLs")

    return df


def create_basic_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic derived features from existing data.

    Generates binary flags and derived features from the raw data,
    such as ToxCast activity indicators.

    Args:
        df: DataFrame to add derived features to.

    Returns:
        DataFrame with additional derived feature columns.
    """
    logger.info("Creating basic derived features")

    if "ToxCast_Active_Count" in df.columns:
        df["Is_ToxCast_Active"] = (df["ToxCast_Active_Count"] > 0).astype(int)
        logger.info("Created binary flag for ToxCast activity")
    else:
        logger.warning("ToxCast_Active_Count column not found")

    return df


def safe_parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely parse SMILES string to RDKit molecule object.

    Args:
        smiles: SMILES string to parse.

    Returns:
        RDKit molecule object or None if parsing fails.
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    
    try:
        return Chem.MolFromSmiles(str(smiles))
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to parse SMILES: {smiles}, error: {e}")
        return None


def clean_data() -> pd.DataFrame:
    """
    Clean and preprocess the raw PFAS Chemical List dataset.

    Performs comprehensive data cleaning including column standardization,
    data type conversion, missing value handling, and basic feature creation.

    Returns:
        Cleaned and preprocessed PFAS Chemical List DataFrame.

    Raises:
        FileNotFoundError: If raw data file doesn't exist.
        ValueError: If critical columns are missing or data is corrupted.
    """
    logger.info("Starting PFAS Chemical List data cleaning process")

    # Load and inspect data
    df = load_data(RAW_DATA_PATH)
    inspect_data(df)

    # Clean column names
    df = clean_column_names(df, COLUMN_MAPPING)

    # Clean DTXSID values
    df = clean_dtxsid(df)

    # Convert numeric columns
    df = convert_numeric_columns(df, NUMERIC_COLUMNS)

    # Handle missing values
    df = handle_missing_values(df)

    # Standardize text data
    df = standardize_text_data(df, TEXT_COLUMNS)

    # Create basic derived features
    df = create_basic_derived_features(df)

    # Save cleaned data
    save_processed_data(df, CLEANED_DATA_PATH)

    logger.info("PFAS Chemical List data cleaning completed successfully")
    logger.info(f"Cleaned data saved to: {CLEANED_DATA_PATH}")

    return df


def engineer_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Execute the feature engineering pipeline for PFAS data.

    Performs advanced feature extraction including molecular complexity
    calculations, fluorine content analysis, and structural categorization
    using RDKit molecular descriptors.

    Args:
        df: DataFrame to engineer features for. If None, loads cleaned data.

    Returns:
        DataFrame with engineered molecular features.

    Raises:
        KeyError: If required SMILES column is missing.
        ValueError: If no valid SMILES entries are found.
    """
    logger.info("Starting PFAS Chemical List feature engineering process")

    # Load cleaned data if not provided
    if df is None:
        df = load_data(CLEANED_DATA_PATH)
    
    # Ensure df is a DataFrame at this point
    assert isinstance(df, pd.DataFrame), "df must be a DataFrame"

    # Ensure SMILES column exists
    if "SMILES" not in df.columns:
        error_msg = "SMILES column missing in dataframe"
        logger.error(error_msg)
        raise KeyError(error_msg)

    # Create molecule cache to avoid redundant parsing
    if "rdkit_mol_cache" not in df.columns:
        logger.info("Creating RDKit molecule cache")
        df = df.copy()  # Ensure we have a DataFrame copy
        df["rdkit_mol_cache"] = df["SMILES"].apply(safe_parse_smiles)

        # Remove invalid molecules
        invalid_mask = df["rdkit_mol_cache"].isna()
        num_invalid = int(invalid_mask.sum())

        if num_invalid > 0:
            logger.warning(
                f"Found {num_invalid} invalid SMILES entries, removing them"
            )
            df = df[~invalid_mask].copy()

        if df.empty:
            error_msg = "No valid SMILES entries to engineer features"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Create RDKit molecules
    df = create_rdkit_mols(df, mol_cache_col="rdkit_mol_cache")

    # Extract fluorine counts
    df = extract_fluorine_count(df)

    # Calculate molecular complexity
    df = calculate_molecular_complexity(df)

    # Categorize molecular features
    df = categorize_molecular_features(df)

    # Save engineered data
    save_processed_data(df, ENGINEERED_DATA_PATH)

    logger.info("Feature engineering completed successfully")
    logger.info(f"Engineered data saved to: {ENGINEERED_DATA_PATH}")

    return df


def main(mode: str = "all") -> None:
    """
    Run the complete data processing pipeline.

    Args:
        mode: Processing mode - 'clean', 'engineer', or 'all'.

    Raises:
        ValueError: If invalid mode is specified.
    """
    valid_modes = {"clean", "engineer", "all"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

    logger.info(f"Running data processing pipeline in '{mode}' mode")

    if mode in ["clean", "all"]:
        df = clean_data()
    else:
        df = load_data(CLEANED_DATA_PATH)

    if mode in ["engineer", "all"]:
        engineer_features(df)

    logger.info("Data processing pipeline completed")


if __name__ == "__main__":
    main()
