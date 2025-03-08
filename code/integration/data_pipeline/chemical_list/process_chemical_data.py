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

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments, rdMolDescriptors, AllChem

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[4]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "PFAS_Chemical_List.csv"
CLEANED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
ENGINEERED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_engineered.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "chemical_list"

#######################
# Data Cleaning Functions
#######################

def load_data():
    """Load the raw PFAS Chemical List dataset."""
    print(f"Loading data from: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)

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
    """Rename columns for consistency."""
    print("\n=== Cleaning Column Names ===")
    
    # Create a mapping of old column names to new column names
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
        "% ToxCast Active": "ToxCast_Active_Percent"
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    print("Columns renamed successfully")
    return df

def clean_dtxsid(df):
    """Clean DTXSID column to extract just the ID from URLs."""
    print("\n=== Cleaning DTXSID Column ===")
    
    # Extract DTXSID from URL using regex
    if "DTXSID" in df.columns:
        # Function to extract DTXSID from URL or keep as is if already clean
        def extract_dtxsid(value):
            if isinstance(value, str) and "comptox.epa.gov" in value:
                match = re.search(r'(DTXSID\d+)', value)
                return match.group(1) if match else value
            return value
        
        # Apply the function to the DTXSID column
        df["DTXSID"] = df["DTXSID"].apply(extract_dtxsid)
        print("Extracted DTXSID IDs from URLs")
    
    return df

def convert_numeric_columns(df):
    """Convert numeric columns to appropriate data types."""
    print("\n=== Converting Numeric Columns ===")
    
    # Convert Average_Mass to numeric
    if "Average_Mass" in df.columns:
        df["Average_Mass"] = pd.to_numeric(df["Average_Mass"], errors="coerce")
        print("Converted Average_Mass to numeric")
    
    # Convert Monoisotopic_Mass to numeric
    if "Monoisotopic_Mass" in df.columns:
        # First clean the column by removing any non-numeric characters
        df["Monoisotopic_Mass"] = df["Monoisotopic_Mass"].astype(str).str.replace(r'[^\d.]', '', regex=True)
        # Then convert to numeric
        df["Monoisotopic_Mass"] = pd.to_numeric(df["Monoisotopic_Mass"], errors="coerce")
        print("Converted Monoisotopic_Mass to numeric")
    
    # Convert ToxCast_Active_Count to numeric
    if "ToxCast_Active_Count" in df.columns:
        df["ToxCast_Active_Count"] = pd.to_numeric(df["ToxCast_Active_Count"], errors="coerce")
        print("Converted ToxCast_Active_Count to numeric")
    
    # Convert Total_Assays to numeric
    if "Total_Assays" in df.columns:
        df["Total_Assays"] = pd.to_numeric(df["Total_Assays"], errors="coerce")
        print("Converted Total_Assays to numeric")
    
    # Convert ToxCast_Active_Percent to numeric
    if "ToxCast_Active_Percent" in df.columns:
        df["ToxCast_Active_Percent"] = pd.to_numeric(df["ToxCast_Active_Percent"], errors="coerce")
        print("Converted ToxCast_Active_Percent to numeric")
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n=== Handling Missing Values ===")
    
    # Check missing values before handling
    print("Missing values before handling:")
    print(df.isnull().sum())
    
    # Handle missing values for each column type
    
    # For numerical columns, replace NaN with median
    numeric_cols = ["Average_Mass", "Monoisotopic_Mass", "Total_Assays", "ToxCast_Active_Percent"]
    for col in numeric_cols:
        if col in df.columns and not df[col].isna().all():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    # For categorical/text columns, replace NaN with "Unknown"
    text_cols = ["InChIKey", "IUPAC_Name", "SMILES", "InChI_String", "Molecular_Formula"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            print(f"Filled missing values in {col} with 'Unknown'")
    
    # Handle ToxCast_Active_Count specifically
    if "ToxCast_Active_Count" in df.columns:
        # If all values are NaN after conversion, set to 0
        if df["ToxCast_Active_Count"].isna().all():
            df["ToxCast_Active_Count"] = 0
            print("Set all ToxCast_Active_Count values to 0 as they were all missing")
        else:
            # Otherwise, fill NaN with 0 (assuming no activity)
            df["ToxCast_Active_Count"] = df["ToxCast_Active_Count"].fillna(0)
            print("Filled missing values in ToxCast_Active_Count with 0")
    
    # Check missing values after handling
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    
    return df

def standardize_text_data(df):
    """Standardize text data in the dataset."""
    print("\n=== Standardizing Text Data ===")
    
    # Columns to standardize
    text_cols = ["Preferred_Name", "IUPAC_Name", "SMILES", "InChI_String", "Molecular_Formula"]
    
    for col in text_cols:
        if col in df.columns:
            # Remove trailing spaces
            df[col] = df[col].str.strip()
            print(f"Removed trailing spaces from {col}")
            
            # Remove special characters from identifiers if needed
            if col in ["Preferred_Name", "IUPAC_Name"]:
                # Keep only alphanumeric characters, spaces, and common chemical notation symbols
                df[col] = df[col].str.replace(r'[^\w\s\-\(\)\[\]\{\}\.,;:+=/\\]', '', regex=True)
                print(f"Removed special characters from {col}")
    
    return df

def create_basic_derived_features(df):
    """Create basic derived features from the data."""
    print("\n=== Creating Basic Derived Features ===")
    
    # Create a binary flag for ToxCast activity
    if "ToxCast_Active_Count" in df.columns:
        df["Is_ToxCast_Active"] = (df["ToxCast_Active_Count"] > 0).astype(int)
        print("Created binary flag for ToxCast activity")
    
    return df

def save_cleaned_data(df):
    """Save the cleaned dataset."""
    print(f"\n=== Saving Cleaned Data to {CLEANED_DATA_PATH} ===")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print("Cleaned data saved successfully")
    
    return df

def clean_data():
    """Main function to execute the data cleaning pipeline."""
    print("Starting PFAS Chemical List data cleaning process...")
    
    # Load data
    df = load_data()
    
    # Inspect data
    inspect_data(df)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Clean DTXSID column
    df = clean_dtxsid(df)
    
    # Convert numeric columns
    df = convert_numeric_columns(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Standardize text data
    df = standardize_text_data(df)
    
    # Create basic derived features
    df = create_basic_derived_features(df)
    
    # Save cleaned data
    df = save_cleaned_data(df)
    
    print("\nPFAS Chemical List data cleaning process completed successfully!")
    print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")
    
    return df

#######################
# Feature Engineering Functions
#######################

def load_cleaned_data():
    """Load the cleaned PFAS Chemical List dataset."""
    print(f"Loading data from: {CLEANED_DATA_PATH}")
    return pd.read_csv(CLEANED_DATA_PATH)

def create_rdkit_mols(df):
    """Create RDKit molecule objects from SMILES strings."""
    print("\n=== Creating RDKit Molecule Objects ===")
    
    # Function to create RDKit molecule from SMILES
    def smiles_to_mol(smiles):
        if smiles == "Unknown" or pd.isna(smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Try to sanitize the molecule
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
            return mol
        except:
            return None
    
    # Apply the function to the SMILES column
    df['ROMol'] = df['SMILES'].apply(smiles_to_mol)
    
    # Count valid molecules
    valid_mols = df['ROMol'].notna().sum()
    print(f"Created {valid_mols} valid RDKit molecules out of {len(df)} compounds")
    
    return df

def extract_fluorine_count(df):
    """Extract fluorine count from SMILES strings."""
    print("\n=== Extracting Fluorine Count ===")
    
    # Function to count fluorine atoms in a molecule
    def count_fluorine(mol):
        if mol is None:
            return 0
        try:
            # Count F atoms in the molecule
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        except:
            return 0
    
    # Apply the function to the ROMol column
    df['F_Count'] = df['ROMol'].apply(count_fluorine)
    
    # Calculate percentage of fluorine atoms
    def calc_f_percentage(mol, f_count):
        if mol is None or f_count == 0:
            return 0
        try:
            total_atoms = mol.GetNumAtoms()
            return (f_count / total_atoms) * 100 if total_atoms > 0 else 0
        except:
            return 0
    
    df['F_Percentage'] = df.apply(lambda x: calc_f_percentage(x['ROMol'], x['F_Count']), axis=1)
    
    print(f"Average fluorine count: {df['F_Count'].mean():.2f}")
    print(f"Maximum fluorine count: {df['F_Count'].max()}")
    
    return df

def calculate_molecular_complexity(df):
    """Calculate molecular complexity features."""
    print("\n=== Calculating Molecular Complexity Features ===")
    
    # Function to calculate number of carbon atoms
    def count_carbon(mol):
        if mol is None:
            return 0
        try:
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        except:
            return 0
    
    # Function to calculate C-F bonds
    def count_cf_bonds(mol):
        if mol is None:
            return 0
        try:
            count = 0
            for bond in mol.GetBonds():
                atoms = [mol.GetAtomWithIdx(bond.GetBeginAtomIdx()), 
                         mol.GetAtomWithIdx(bond.GetEndAtomIdx())]
                symbols = [atom.GetSymbol() for atom in atoms]
                if 'C' in symbols and 'F' in symbols:
                    count += 1
            return count
        except:
            return 0
    
    # Function to estimate chain length using number of atoms
    def estimate_chain_length(mol):
        if mol is None:
            return 0
        try:
            # Use a simple approximation: number of carbon atoms
            # This is a rough estimate but more efficient than path calculations
            return count_carbon(mol)
        except:
            return 0
    
    # Calculate basic molecular properties
    df['C_Count'] = df['ROMol'].apply(count_carbon)
    df['CF_Bonds'] = df['ROMol'].apply(count_cf_bonds)
    df['Chain_Length'] = df['ROMol'].apply(estimate_chain_length)
    
    # Calculate more complex descriptors
    def calc_descriptors(mol):
        if mol is None:
            return pd.Series([0, 0, 0, 0, 0, 0])
        
        try:
            return pd.Series([
                Descriptors.MolWt(mol),                      # Molecular weight
                Lipinski.NumRotatableBonds(mol),             # Number of rotatable bonds
                Lipinski.NumHAcceptors(mol),                 # Number of H-bond acceptors
                Lipinski.NumHDonors(mol),                    # Number of H-bond donors
                rdMolDescriptors.CalcNumRings(mol),          # Number of rings
                rdMolDescriptors.CalcNumAromaticRings(mol)   # Number of aromatic rings
            ])
        except:
            return pd.Series([0, 0, 0, 0, 0, 0])
    
    # Apply the function to calculate descriptors
    descriptors = df['ROMol'].apply(calc_descriptors)
    descriptors.columns = ['MW_RDKit', 'Rotatable_Bonds', 'H_Acceptors', 'H_Donors', 'Ring_Count', 'Aromatic_Rings']
    
    # Add descriptors to the dataframe
    df = pd.concat([df, descriptors], axis=1)
    
    print("Calculated molecular complexity features")
    
    return df

def categorize_pfas_types(df):
    """Categorize PFAS compounds by structural types."""
    print("\n=== Categorizing PFAS Types ===")
    
    # Function to determine if a molecule is aromatic
    def is_aromatic(mol):
        if mol is None:
            return False
        try:
            return any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        except:
            return False
    
    # Function to determine if a molecule has rings
    def has_rings(mol):
        if mol is None:
            return False
        try:
            return rdMolDescriptors.CalcNumRings(mol) > 0
        except:
            return False
    
    # Function to determine if a molecule is cyclic
    def is_cyclic(mol):
        if mol is None:
            return False
        try:
            return mol.GetRingInfo().NumRings() > 0
        except:
            return False
    
    # Function to determine if a molecule is branched
    def is_branched(mol):
        if mol is None:
            return False
        try:
            # A molecule is branched if any atom has more than 2 neighbors
            return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())
        except:
            return False
    
    # Categorize by chain length
    df['Chain_Category'] = pd.cut(
        df['Chain_Length'], 
        bins=[0, 4, 7, float('inf')], 
        labels=['Short-chain', 'Medium-chain', 'Long-chain'],
        include_lowest=True
    )
    
    # Categorize by structural features
    df['Is_Aromatic'] = df['ROMol'].apply(is_aromatic)
    df['Has_Rings'] = df['ROMol'].apply(has_rings)
    df['Is_Cyclic'] = df['ROMol'].apply(is_cyclic)
    df['Is_Branched'] = df['ROMol'].apply(is_branched)
    
    # Create a combined PFAS type category
    def determine_pfas_type(row):
        if pd.isna(row['ROMol']) or row['ROMol'] is None:
            return 'Unknown'
        
        type_components = []
        
        # Chain length
        if not pd.isna(row['Chain_Category']):
            type_components.append(row['Chain_Category'])
        
        # Structural features
        if row['Is_Aromatic']:
            type_components.append('Aromatic')
        if row['Is_Cyclic'] and not row['Is_Aromatic']:
            type_components.append('Cyclic')
        if row['Is_Branched']:
            type_components.append('Branched')
        
        # Fluorination level
        if row['F_Count'] > 0:
            if row['F_Percentage'] > 50:
                type_components.append('Highly-fluorinated')
            elif row['F_Percentage'] > 25:
                type_components.append('Moderately-fluorinated')
            else:
                type_components.append('Lightly-fluorinated')
        
        return ' '.join(type_components) if type_components else 'Unknown'
    
    df['PFAS_Type'] = df.apply(determine_pfas_type, axis=1)
    
    # Print category statistics
    print("\nPFAS Chain Length Categories:")
    print(df['Chain_Category'].value_counts())
    
    print("\nPFAS Structural Features:")
    print(f"Aromatic: {df['Is_Aromatic'].sum()}")
    print(f"Has Rings: {df['Has_Rings'].sum()}")
    print(f"Cyclic: {df['Is_Cyclic'].sum()}")
    print(f"Branched: {df['Is_Branched'].sum()}")
    
    print("\nTop 10 PFAS Types:")
    print(df['PFAS_Type'].value_counts().head(10))
    
    return df

def save_engineered_data(df):
    """Save the engineered dataset."""
    print(f"\n=== Saving Engineered Data to {ENGINEERED_DATA_PATH} ===")
    
    # Remove RDKit molecule objects before saving
    df = df.drop(columns=['ROMol'])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(ENGINEERED_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print("Engineered data saved successfully")
    
    return df

def engineer_features(df=None):
    """Main function to execute the feature engineering pipeline."""
    print("Starting PFAS Chemical List feature engineering process...")
    
    # Load data if not provided
    if df is None:
        df = load_cleaned_data()
    
    # Create RDKit molecule objects
    df = create_rdkit_mols(df)
    
    # Extract fluorine count
    df = extract_fluorine_count(df)
    
    # Calculate molecular complexity
    df = calculate_molecular_complexity(df)
    
    # Categorize PFAS types
    df = categorize_pfas_types(df)
    
    # Save engineered data
    df = save_engineered_data(df)
    
    print("\nPFAS Chemical List feature engineering process completed successfully!")
    print(f"Engineered data saved to: {ENGINEERED_DATA_PATH}")
    
    # Return summary of new features
    print("\n=== Summary of New Features ===")
    new_features = ['F_Count', 'F_Percentage', 'C_Count', 'CF_Bonds', 'Chain_Length', 
                   'MW_RDKit', 'Rotatable_Bonds', 'H_Acceptors', 'H_Donors', 
                   'Ring_Count', 'Aromatic_Rings', 'Chain_Category', 'Is_Aromatic', 
                   'Has_Rings', 'Is_Cyclic', 'Is_Branched', 'PFAS_Type']
    
    print(f"Added {len(new_features)} new features:")
    for feature in new_features:
        print(f"- {feature}")
    
    return df

def main(mode='all'):
    """Main function to execute the data processing pipeline.
    
    Args:
        mode (str): Processing mode - 'clean', 'engineer', or 'all'
    """
    if mode == 'clean' or mode == 'all':
        df = clean_data()
    else:
        df = None
    
    if mode == 'engineer' or mode == 'all':
        engineer_features(df if mode == 'all' else None)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PFAS Chemical List data')
    parser.add_argument('--mode', choices=['clean', 'engineer', 'all'], default='all',
                        help='Processing mode: clean, engineer, or all (default)')
    
    args = parser.parse_args()
    main(args.mode) 