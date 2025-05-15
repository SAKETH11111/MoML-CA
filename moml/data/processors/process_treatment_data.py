#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PFAS Treatment Data Processing

This script performs data cleaning and feature engineering on the PFAS Treatment Data dataset:
1. Initial inspection of the dataset
2. Cleaning and standardization of column names
3. Converting data types (temperature, time)
4. Handling missing values
5. Calculating treatment effectiveness where missing
6. Creating derived features (binary success outcome, temperature/time bins)
7. Standardizing identifiers for alignment with Chemical List
8. Saving the processed dataset

This script now uses the consolidated moml architecture for data processing.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

# Import from moml packages

# Import utility functions
from moml.utils import (
    load_data,
    inspect_data,
    clean_column_names,
    convert_numeric_columns,
    handle_missing_values,
    save_processed_data,
    extract_numeric_from_text,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("treatment_data_processing")

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[4]
RAW_TREATMENT_PATH = ROOT_DIR / "data" / "raw" / "PFAS_Treatment_Data.csv"
CLEANED_CHEMICAL_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
PROCESSED_TREATMENT_PATH = ROOT_DIR / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "treatment_data"


def clean_casrn(df):
    """Clean and standardize CASRN values."""
    logger.info("=== Cleaning CASRN Values ===")

    # Extract the main CASRN from text (e.g., "375-73-5 as acid" -> "375-73-5")
    def extract_casrn(value):
        if pd.isna(value):
            return value
        # Extract the pattern XX-XX-X or XXX-XX-X
        match = re.search(r"\d+-\d+-\d+", str(value))
        return match.group(0) if match else value

    # Apply the function to the CASRN column
    df["CASRN"] = df["CASRN"].apply(extract_casrn)

    logger.info("Cleaned CASRN values")
    return df


def convert_time_to_minutes(time_str):
    """Convert time string to minutes."""
    if pd.isna(time_str):
        return np.nan

    time_str = str(time_str).lower()

    # Extract numeric values
    numeric_values = re.findall(r"\d+\.?\d*", time_str)
    if not numeric_values:
        return np.nan

    value = float(numeric_values[0])

    # Convert to minutes based on units
    if "second" in time_str or "sec" in time_str or "s" in time_str:
        return value / 60.0
    elif "minute" in time_str or "min" in time_str:
        return value
    elif "hour" in time_str or "hr" in time_str or "h" in time_str:
        return value * 60.0
    elif "day" in time_str or "d" in time_str:
        return value * 24 * 60.0
    else:
        return value  # Assume minutes if no unit specified


def calculate_effectiveness(df):
    """Calculate treatment effectiveness where missing."""
    logger.info("=== Calculating Treatment Effectiveness ===")

    # Calculate effectiveness where initial and post concentration are available
    mask = (
        df["Effectiveness_Percent_Numeric"].isna()
        & df["Initial_Concentration_Numeric"].notna()
        & df["Post_Concentration_Numeric"].notna()
        & (df["Initial_Concentration_Numeric"] > 0)  # Avoid division by zero
    )

    df.loc[mask, "Effectiveness_Percent_Numeric"] = (
        (df.loc[mask, "Initial_Concentration_Numeric"] - df.loc[mask, "Post_Concentration_Numeric"])
        / df.loc[mask, "Initial_Concentration_Numeric"]
        * 100
    )

    # Clip values to 0-100% range
    df["Effectiveness_Percent_Numeric"] = df["Effectiveness_Percent_Numeric"].clip(0, 100)

    # Count how many missing values were filled
    filled_count = mask.sum()
    logger.info(f"Calculated effectiveness for {filled_count} entries")

    # Document missing effectiveness data
    missing_count = df["Effectiveness_Percent_Numeric"].isna().sum()
    logger.info(f"Still missing effectiveness data for {missing_count} entries")

    return df


def create_derived_features(df):
    """Create derived features from the data."""
    logger.info("=== Creating Derived Features ===")

    # Binary outcome for successful treatment (>80% effectiveness)
    if "Effectiveness_Percent_Numeric" in df.columns:
        df["Treatment_Success"] = df["Effectiveness_Percent_Numeric"] > 80
        logger.info("Created binary treatment success feature")

    # Bin temperature into categories
    if "Treatment_Temp_C" in df.columns:
        # Define temperature bins
        temp_bins = [-float("inf"), 25, 100, 400, float("inf")]
        temp_labels = ["Ambient", "Low", "Medium", "High"]

        df["Temperature_Category"] = pd.cut(
            df["Treatment_Temp_C"], bins=temp_bins, labels=temp_labels, include_lowest=True
        )
        logger.info("Created temperature category feature")

    # Bin treatment time into categories
    if "Treatment_Time_Minutes" in df.columns:
        # Define time bins (in minutes)
        time_bins = [-float("inf"), 30, 180, 1440, float("inf")]  # Up to 30min, 3hrs, 24hrs, >24hrs
        time_labels = ["Short", "Medium", "Long", "Extended"]

        df["Time_Category"] = pd.cut(
            df["Treatment_Time_Minutes"], bins=time_bins, labels=time_labels, include_lowest=True
        )
        logger.info("Created time category feature")

    return df


def standardize_with_chemical_list(df, chem_df):
    """Standardize identifiers with the chemical list dataset."""
    logger.info("=== Standardizing with Chemical List ===")

    # Get CASRNs from both datasets
    chem_casrns = set(chem_df["CASRN"])
    treat_casrns = set(df["CASRN"])

    # Find common CASRNs
    common_casrns = chem_casrns.intersection(treat_casrns)
    logger.info(f"Found {len(common_casrns)} common CASRNs between datasets")

    # Filter to only include treatments for chemicals in the chemical list
    df = df[df["CASRN"].isin(common_casrns)]
    logger.info(f"Filtered to {len(df)} treatment records")

    return df


def process_treatment_data():
    """Main function to execute the treatment data processing pipeline."""
    logger.info("Starting PFAS Treatment Data processing...")

    # Load data
    df = load_data(RAW_TREATMENT_PATH)
    chem_df = load_data(CLEANED_CHEMICAL_PATH)

    # Inspect data
    inspect_data(df)

    # Clean column names
    column_mapping = {
        "Analyte Abv.": "Analyte_Abbreviation",
        "Analyte Name": "Chemical_Name",
        "CASRN": "CASRN",
        "Treatment Process": "Treatment_Process",
        "Test Scale": "Test_Scale",
        "Matrix": "Matrix",
        "Matrix Detail": "Matrix_Detail",
        "Treatment Temperature (°C)": "Treatment_Temp_C",
        "Treatment Time": "Treatment_Time",
        "Condition Details": "Condition_Details",
        "Effectiveness (%)": "Effectiveness_Percent",
        "Effectiveness Type": "Effectiveness_Type",
        "Measure Details": "Measure_Details",
        "Initial Concentration": "Initial_Concentration",
        "Initial Details": "Initial_Details",
        "Post Concentration": "Post_Concentration",
        "Post Details": "Post_Details",
        "Offgas Concentration": "Offgas_Concentration",
        "Additional Details": "Additional_Details",
        "Citation": "Citation",
    }
    df = clean_column_names(df, column_mapping)

    # Clean CASRN values
    df = clean_casrn(df)

    # Convert numeric columns
    numeric_columns = ["Treatment_Temp_C"]
    df = convert_numeric_columns(df, numeric_columns)

    # Convert treatment time to minutes
    df["Treatment_Time_Minutes"] = df["Treatment_Time"].apply(convert_time_to_minutes)

    # Extract numeric values from concentration columns
    concentration_columns = ["Initial_Concentration", "Post_Concentration", "Offgas_Concentration"]
    for col in concentration_columns:
        if col in df.columns:
            df[f"{col}_Numeric"] = df[col].apply(extract_numeric_from_text)

    # Handle missing values
    df = handle_missing_values(df)

    # Calculate effectiveness where missing
    df = calculate_effectiveness(df)

    # Create derived features
    df = create_derived_features(df)

    # Standardize with chemical list
    df = standardize_with_chemical_list(df, chem_df)

    # Save processed data
    save_processed_data(df, PROCESSED_TREATMENT_PATH)

    logger.info("PFAS Treatment Data processing completed successfully!")
    logger.info(f"Processed data saved to: {PROCESSED_TREATMENT_PATH}")

    return df


if __name__ == "__main__":
    process_treatment_data()
