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
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[4]
RAW_TREATMENT_PATH = ROOT_DIR / "data" / "raw" / "PFAS_Treatment_Data.csv"
CLEANED_CHEMICAL_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
PROCESSED_TREATMENT_PATH = ROOT_DIR / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "treatment_data"

def load_data():
    """Load the raw PFAS Treatment Data dataset."""
    print(f"Loading data from: {RAW_TREATMENT_PATH}")
    return pd.read_csv(RAW_TREATMENT_PATH, encoding='latin1')

def load_chemical_data():
    """Load the cleaned PFAS Chemical List for standardization."""
    print(f"Loading chemical data from: {CLEANED_CHEMICAL_PATH}")
    return pd.read_csv(CLEANED_CHEMICAL_PATH)

def inspect_data(df):
    """Perform initial inspection of the dataset."""
    print("\n=== Initial Data Inspection ===")
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())

def clean_column_names(df):
    """Rename columns for clarity."""
    print("\n=== Cleaning Column Names ===")
    
    # Create a mapping of old column names to new column names
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
        "Citation": "Citation"
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    print("Columns renamed successfully")
    return df

def clean_casrn(df):
    """Clean and standardize CASRN values."""
    print("\n=== Cleaning CASRN Values ===")
    
    # Extract the main CASRN from text (e.g., "375-73-5 as acid" -> "375-73-5")
    def extract_casrn(value):
        if pd.isna(value):
            return value
        # Extract the pattern XX-XX-X or XXX-XX-X
        match = re.search(r'\d+-\d+-\d+', str(value))
        return match.group(0) if match else value
    
    # Apply the function to the CASRN column
    df['CASRN'] = df['CASRN'].apply(extract_casrn)
    
    print(f"Cleaned CASRN values")
    return df

def convert_numeric_columns(df):
    """Convert columns to appropriate data types."""
    print("\n=== Converting Numeric Columns ===")
    
    # Convert temperature to numeric
    if "Treatment_Temp_C" in df.columns:
        df["Treatment_Temp_C"] = pd.to_numeric(df["Treatment_Temp_C"], errors="coerce")
        print("Converted Treatment_Temp_C to numeric")
    
    # Convert treatment time to numeric (minutes)
    if "Treatment_Time" in df.columns:
        
        def convert_time_to_minutes(time_str):
            if pd.isna(time_str):
                return np.nan
            
            time_str = str(time_str).lower()
            
            # Extract numeric values
            numeric_values = re.findall(r'\d+\.?\d*', time_str)
            if not numeric_values:
                return np.nan
            
            value = float(numeric_values[0])
            
            # Convert to minutes based on units
            if 'second' in time_str or 'sec' in time_str or 's' in time_str:
                return value / 60.0
            elif 'minute' in time_str or 'min' in time_str:
                return value
            elif 'hour' in time_str or 'hr' in time_str or 'h' in time_str:
                return value * 60.0
            elif 'day' in time_str or 'd' in time_str:
                return value * 24 * 60.0
            else:
                return value  # Assume minutes if no unit specified
        
        df['Treatment_Time_Minutes'] = df['Treatment_Time'].apply(convert_time_to_minutes)
        print("Converted Treatment_Time to numeric minutes")
    
    # Try to convert effectiveness to numeric
    if "Effectiveness_Percent" in df.columns:
        df["Effectiveness_Percent_Numeric"] = pd.to_numeric(df["Effectiveness_Percent"], errors="coerce")
        print("Converted Effectiveness_Percent to numeric")
    
    # Extract numeric values from concentration columns
    concentration_columns = ['Initial_Concentration', 'Post_Concentration', 'Offgas_Concentration']
    
    for col in concentration_columns:
        if col in df.columns:
            def extract_numeric_value(value):
                if pd.isna(value):
                    return np.nan
                
                value = str(value)
                
                # Handle "ND" (Non-Detect) as 0
                if value.lower() == 'nd' or value.lower() == 'not detected':
                    return 0.0
                
                # Extract first numeric value
                numeric_values = re.findall(r'\d+\.?\d*', value)
                return float(numeric_values[0]) if numeric_values else np.nan
            
            df[f"{col}_Numeric"] = df[col].apply(extract_numeric_value)
            print(f"Extracted numeric values from {col}")
    
    return df

def calculate_effectiveness(df):
    """Calculate treatment effectiveness where missing."""
    print("\n=== Calculating Treatment Effectiveness ===")
    
    # Calculate effectiveness where initial and post concentration are available
    mask = (
        df['Effectiveness_Percent_Numeric'].isna() & 
        df['Initial_Concentration_Numeric'].notna() & 
        df['Post_Concentration_Numeric'].notna() &
        (df['Initial_Concentration_Numeric'] > 0)  # Avoid division by zero
    )
    
    df.loc[mask, 'Effectiveness_Percent_Numeric'] = (
        (df.loc[mask, 'Initial_Concentration_Numeric'] - df.loc[mask, 'Post_Concentration_Numeric']) / 
        df.loc[mask, 'Initial_Concentration_Numeric'] * 100
    )
    
    # Clip values to 0-100% range
    df['Effectiveness_Percent_Numeric'] = df['Effectiveness_Percent_Numeric'].clip(0, 100)
    
    # Count how many missing values were filled
    filled_count = mask.sum()
    print(f"Calculated effectiveness for {filled_count} entries")
    
    # Document missing effectiveness data
    missing_count = df['Effectiveness_Percent_Numeric'].isna().sum()
    print(f"Still missing effectiveness data for {missing_count} entries")
    
    return df

def create_derived_features(df):
    """Create derived features from the data."""
    print("\n=== Creating Derived Features ===")
    
    # Binary outcome for successful treatment (>80% effectiveness)
    if 'Effectiveness_Percent_Numeric' in df.columns:
        df['Treatment_Success'] = df['Effectiveness_Percent_Numeric'] > 80
        print("Created binary treatment success feature")
    
    # Bin temperature into categories
    if 'Treatment_Temp_C' in df.columns:
        # Define temperature bins
        temp_bins = [-float('inf'), 25, 100, 400, float('inf')]
        temp_labels = ['Ambient', 'Low', 'Medium', 'High']
        
        df['Temperature_Category'] = pd.cut(
            df['Treatment_Temp_C'], 
            bins=temp_bins,
            labels=temp_labels,
            include_lowest=True
        )
        print("Created temperature category feature")
    
    # Bin treatment time into categories
    if 'Treatment_Time_Minutes' in df.columns:
        # Define time bins (in minutes)
        time_bins = [-float('inf'), 30, 180, 1440, float('inf')]  # Up to 30min, 3hrs, 24hrs, >24hrs
        time_labels = ['Short', 'Medium', 'Long', 'Extended']
        
        df['Time_Category'] = pd.cut(
            df['Treatment_Time_Minutes'], 
            bins=time_bins,
            labels=time_labels,
            include_lowest=True
        )
        print("Created time category feature")
    
    return df

def standardize_with_chemical_list(df, chem_df):
    """Standardize identifiers with the PFAS Chemical List."""
    print("\n=== Standardizing Identifiers with Chemical List ===")
    
    # Create a mapping of CASRNs from the chemical list
    cas_to_name = dict(zip(chem_df['CASRN'], chem_df['Preferred_Name']))
    
    # Check for matches and mismatches
    matching_cases = df['CASRN'].isin(chem_df['CASRN']).sum()
    total_cases = len(df)
    
    print(f"Treatment data has {matching_cases} matching CASRNs out of {total_cases} ({matching_cases/total_cases*100:.2f}%)")
    
    # Add a matched flag
    df['In_Chemical_List'] = df['CASRN'].isin(chem_df['CASRN'])
    
    # Add standardized chemical name from the chemical list where available
    df['Standardized_Chemical_Name'] = df['CASRN'].map(cas_to_name)
    
    # Where not available, use the original name
    df['Standardized_Chemical_Name'] = df['Standardized_Chemical_Name'].fillna(df['Chemical_Name'])
    
    return df

def save_processed_data(df):
    """Save the processed dataset."""
    print(f"\n=== Saving Processed Data to {PROCESSED_TREATMENT_PATH} ===")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(PROCESSED_TREATMENT_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(PROCESSED_TREATMENT_PATH, index=False)
    print("Processed data saved successfully")
    
    return df

def process_treatment_data():
    """Main function to execute the data processing pipeline."""
    print("Starting PFAS Treatment Data processing...")
    
    # Load data
    df = load_data()
    
    # Inspect data
    inspect_data(df)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Clean CASRN values
    df = clean_casrn(df)
    
    # Convert numeric columns
    df = convert_numeric_columns(df)
    
    # Calculate effectiveness where missing
    df = calculate_effectiveness(df)
    
    # Create derived features
    df = create_derived_features(df)
    
    # Load chemical data for standardization
    try:
        chem_df = load_chemical_data()
        df = standardize_with_chemical_list(df, chem_df)
    except FileNotFoundError:
        print("Chemical list data not found. Skipping standardization.")
    
    # Save processed data
    df = save_processed_data(df)
    
    print("\nPFAS Treatment Data processing completed successfully!")
    print(f"Processed data saved to: {PROCESSED_TREATMENT_PATH}")
    
    return df

if __name__ == "__main__":
    process_treatment_data() 
