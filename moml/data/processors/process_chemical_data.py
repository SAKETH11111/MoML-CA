#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PFAS Chemical List Data Processing

This script performs data cleaning and feature engineering on the PFAS Chemical List dataset:
1. Initial inspection of the dataset
2. Cleaning and standardization of column names
3. Handling missing values
4. Standardizing text data
5. Converting data types
6. Extracting fluorine counts from SMILES strings
7. Calculating molecular complexity features
8. Categorizing PFAS compounds by structural types
9. Saving the processed datasets
"""

import re
import logging
import pandas as pd
from pathlib import Path
from rdkit import Chem

# Configure logger
logger = logging.getLogger(__name__)

# Import consolidated MoML functions

# Import utility functions
from moml.utils import (
    load_data,
    inspect_data,
    clean_column_names,
    convert_numeric_columns,
    handle_missing_values,
    standardize_text_data,
    save_processed_data,
    create_rdkit_mols,
    extract_fluorine_count,
    calculate_molecular_complexity,
    categorize_molecular_features,
)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "PFAS_Chemical_List.csv"
CLEANED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
ENGINEERED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_engineered.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "chemical_list"


def clean_dtxsid(df):
    """Clean DTXSID column to extract just the ID from URLs."""
    print("\n=== Cleaning DTXSID Column ===")

    if "DTXSID" in df.columns:

        def extract_dtxsid(value):
            if isinstance(value, str) and "comptox.epa.gov" in value:
                match = re.search(r"(DTXSID\d+)", value)
                return match.group(1) if match else value
            return value

        df["DTXSID"] = df["DTXSID"].apply(extract_dtxsid)
        print("Extracted DTXSID IDs from URLs")

    return df


def create_basic_derived_features(df):
    """Create basic derived features from the data."""
    print("\n=== Creating Basic Derived Features ===")

    if "ToxCast_Active_Count" in df.columns:
        df["Is_ToxCast_Active"] = (df["ToxCast_Active_Count"] > 0).astype(int)
        print("Created binary flag for ToxCast activity")

    return df


def clean_data():
    """Main function to execute the data cleaning pipeline."""
    print("Starting PFAS Chemical List data cleaning process...")

    # Load and inspect data
    df = load_data(RAW_DATA_PATH)
    inspect_data(df)

    # Clean column names
    column_mapping = {
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
    df = clean_column_names(df, column_mapping)

    # Clean DTXSID values
    df = clean_dtxsid(df)

    # Convert numeric columns
    numeric_columns = [
        "Average_Mass",
        "Monoisotopic_Mass",
        "ToxCast_Active_Count",
        "Total_Assays",
        "ToxCast_Active_Percent",
    ]
    df = convert_numeric_columns(df, numeric_columns)

    # Handle missing values
    df = handle_missing_values(df)

    # Standardize text data
    text_columns = ["Preferred_Name", "IUPAC_Name", "SMILES", "InChI_String", "Molecular_Formula"]
    df = standardize_text_data(df, text_columns)

    # Create basic derived features
    df = create_basic_derived_features(df)

    # Save cleaned data
    save_processed_data(df, CLEANED_DATA_PATH)

    print("\nPFAS Chemical List data cleaning process completed successfully!")
    print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")

    return df


def engineer_features(df=None):
    """Main function to execute the feature engineering pipeline."""
    print("Starting PFAS Chemical List feature engineering process...")

    # Load cleaned data if not provided
    if df is None:
        df = load_data(CLEANED_DATA_PATH)

    # Ensure SMILES column exists
    if "SMILES" not in df.columns:
        logger.error("SMILES column missing in dataframe, cannot engineer features")
        raise KeyError("SMILES column missing in dataframe")

    # Add a column to cache parsed molecules to avoid redundant parsing
    if "rdkit_mol_cache" not in df.columns:
        # Safely validate SMILES entries and create molecule cache
        def safe_parse_smiles(s):
            if pd.isna(s) or not isinstance(s, str):
                return None
            try:
                return Chem.MolFromSmiles(str(s))
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Failed to parse SMILES: {s}, error: {e}")
                return None

        df["rdkit_mol_cache"] = df["SMILES"].apply(safe_parse_smiles)

        # Create a mask for valid molecules
        invalid_mask = df["rdkit_mol_cache"].isna()
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:
            logger.warning(f"Found {num_invalid} invalid SMILES entries before feature engineering")
            logger.warning(f"Dropping {num_invalid} invalid SMILES entries before feature engineering")
            df = df[~invalid_mask]

        if df.empty:
            logger.error("All SMILES entries are invalid, aborting feature engineering")
            raise ValueError("No valid SMILES entries to engineer features")

    # Create RDKit molecules, passing the cached molecules if available
    df = create_rdkit_mols(df, mol_cache_col="rdkit_mol_cache")

    # Extract fluorine counts
    df = extract_fluorine_count(df)

    # Calculate molecular complexity
    df = calculate_molecular_complexity(df)

    # Categorize molecular features
    df = categorize_molecular_features(df)

    # Save engineered data
    save_processed_data(df, ENGINEERED_DATA_PATH)

    print("\nPFAS Chemical List feature engineering process completed successfully!")
    print(f"Engineered data saved to: {ENGINEERED_DATA_PATH}")

    return df


def main(mode="all"):
    """Main function to run the data processing pipeline.

    Args:
        mode: Processing mode ('clean', 'engineer', or 'all')
    """
    if mode in ["clean", "all"]:
        df = clean_data()
    else:
        df = load_data(CLEANED_DATA_PATH)

    if mode in ["engineer", "all"]:
        engineer_features(df)


if __name__ == "__main__":
    main()
