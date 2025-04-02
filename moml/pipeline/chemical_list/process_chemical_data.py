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
import re
from pathlib import Path
from rdkit import Chem

# Import consolidated MoML functions
from moml.core import (
    validate_smiles,
    calculate_molecular_descriptors,
    MolecularFeatureExtractor
)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[4]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "PFAS_Chemical_List.csv"
CLEANED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
ENGINEERED_DATA_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_engineered.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "chemical_list"

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
    
    df = df.rename(columns=column_mapping)
    print("Columns renamed successfully")
    return df

def clean_dtxsid(df):
    """Clean DTXSID column to extract just the ID from URLs."""
    print("\n=== Cleaning DTXSID Column ===")
    
    if "DTXSID" in df.columns:
        def extract_dtxsid(value):
            if isinstance(value, str) and "comptox.epa.gov" in value:
                match = re.search(r'(DTXSID\d+)', value)
                return match.group(1) if match else value
            return value
        
        df["DTXSID"] = df["DTXSID"].apply(extract_dtxsid)
        print("Extracted DTXSID IDs from URLs")
    
    return df

def convert_numeric_columns(df):
    """Convert numeric columns to appropriate data types."""
    print("\n=== Converting Numeric Columns ===")
    
    if "Average_Mass" in df.columns:
        df["Average_Mass"] = pd.to_numeric(df["Average_Mass"], errors="coerce")
        print("Converted Average_Mass to numeric")
    
    if "Monoisotopic_Mass" in df.columns:
        df["Monoisotopic_Mass"] = df["Monoisotopic_Mass"].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df["Monoisotopic_Mass"] = pd.to_numeric(df["Monoisotopic_Mass"], errors="coerce")
        print("Converted Monoisotopic_Mass to numeric")
    
    if "ToxCast_Active_Count" in df.columns:
        df["ToxCast_Active_Count"] = pd.to_numeric(df["ToxCast_Active_Count"], errors="coerce")
        print("Converted ToxCast_Active_Count to numeric")
    
    if "Total_Assays" in df.columns:
        df["Total_Assays"] = pd.to_numeric(df["Total_Assays"], errors="coerce")
        print("Converted Total_Assays to numeric")
    
    if "ToxCast_Active_Percent" in df.columns:
        df["ToxCast_Active_Percent"] = pd.to_numeric(df["ToxCast_Active_Percent"], errors="coerce")
        print("Converted ToxCast_Active_Percent to numeric")
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n=== Handling Missing Values ===")
    
    print("Missing values before handling:")
    print(df.isnull().sum())
    
    numeric_cols = ["Average_Mass", "Monoisotopic_Mass", "Total_Assays", "ToxCast_Active_Percent"]
    for col in numeric_cols:
        if col in df.columns and not df[col].isna().all():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    text_cols = ["InChIKey", "IUPAC_Name", "SMILES", "InChI_String", "Molecular_Formula"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            print(f"Filled missing values in {col} with 'Unknown'")
    
    if "ToxCast_Active_Count" in df.columns:
        if df["ToxCast_Active_Count"].isna().all():
            df["ToxCast_Active_Count"] = 0
            print("Set all ToxCast_Active_Count values to 0 as they were all missing")
        else:
            df["ToxCast_Active_Count"] = df["ToxCast_Active_Count"].fillna(0)
            print("Filled missing values in ToxCast_Active_Count with 0")
    
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    
    return df

def standardize_text_data(df):
    """Standardize text data in the dataset."""
    print("\n=== Standardizing Text Data ===")
    
    text_cols = ["Preferred_Name", "IUPAC_Name", "SMILES", "InChI_String", "Molecular_Formula"]
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
            print(f"Removed trailing spaces from {col}")
            
            if col in ["Preferred_Name", "IUPAC_Name"]:
                df[col] = df[col].str.replace(r'[^\w\s\-\(\)\[\]\{\}\.,;:+=/\\]', '', regex=True)
                print(f"Removed special characters from {col}")
    
    return df

def create_basic_derived_features(df):
    """Create basic derived features from the data."""
    print("\n=== Creating Basic Derived Features ===")
    
    if "ToxCast_Active_Count" in df.columns:
        df["Is_ToxCast_Active"] = (df["ToxCast_Active_Count"] > 0).astype(int)
        print("Created binary flag for ToxCast activity")
    
    return df

def save_cleaned_data(df):
    """Save the cleaned dataset."""
    print(f"\n=== Saving Cleaned Data to {CLEANED_DATA_PATH} ===")
    
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print("Cleaned data saved successfully")
    
    return df

def clean_data():
    """Main function to execute the data cleaning pipeline."""
    print("Starting PFAS Chemical List data cleaning process...")
    
    df = load_data()
    
    inspect_data(df)
    
    df = clean_column_names(df)
    
    df = clean_dtxsid(df)
    
    df = convert_numeric_columns(df)
    
    df = handle_missing_values(df)
    
    df = standardize_text_data(df)
    
    df = create_basic_derived_features(df)
    
    df = save_cleaned_data(df)
    
    print("\nPFAS Chemical List data cleaning process completed successfully!")
    print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")
    
    return df

def load_cleaned_data():
    """Load the cleaned PFAS Chemical List dataset."""
    print(f"Loading data from: {CLEANED_DATA_PATH}")
    return pd.read_csv(CLEANED_DATA_PATH)

def create_rdkit_mols(df, smiles_col=None, mol_col=None):
    """Create RDKit molecule objects from SMILES strings.
    
    Args:
        df (pandas.DataFrame): Dataframe containing SMILES strings
        smiles_col (str, optional): Name of column containing SMILES strings. Defaults to 'SMILES' if None.
        mol_col (str, optional): Name of column to store RDKit molecules. Defaults to 'ROMol' if None.
    
    Returns:
        pandas.DataFrame: Dataframe with RDKit molecules and validity flags
    """
    print("\n=== Creating RDKit Molecule Objects ===")
    
    # Set default column names if not provided
    smiles_col = smiles_col or 'SMILES'
    mol_col = mol_col or 'ROMol'
    
    valid_col = 'Valid_SMILES'
    if smiles_col != 'SMILES':
        valid_col = f'is_valid_{smiles_col}'
    
    # Check if the SMILES column exists
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataframe")
    
    # Use MoML's consolidated function to validate SMILES and create RDKit molecules
    smiles_validation_results = [
        validate_smiles(smiles) if smiles != "Unknown" and not pd.isna(smiles) else (None, False) 
        for smiles in df[smiles_col]
    ]
    
    # Extract molecules and validity flags
    df[mol_col] = [result[0] for result in smiles_validation_results]
    df[valid_col] = [result[1] for result in smiles_validation_results]
    
    valid_mols = df[valid_col].sum()
    print(f"Created {valid_mols} valid RDKit molecules out of {len(df)} compounds")
    
    return df

def extract_fluorine_count(df):
    """Extract fluorine count from SMILES strings."""
    print("\n=== Extracting Fluorine Count ===")
    
    def count_fluorine(mol):
        if mol is None:
            return 0
        try:
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        except:
            return 0
    
    df['F_Count'] = df['ROMol'].apply(count_fluorine)
    
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
    
    # Initialize the feature extractor from MoML
    feature_extractor = MolecularFeatureExtractor()
    
    # Calculate descriptors for valid molecules
    descriptor_columns = ['MW_RDKit', 'Rotatable_Bonds', 'H_Acceptors', 'H_Donors', 
                         'Ring_Count', 'Aromatic_Rings', 'C_Count', 'F_Count', 'CF_Bonds']
    
    for idx, row in df[df['ROMol'].notna()].iterrows():
        # Use MoML's consolidated function to calculate descriptors
        mol = row['ROMol']
        descriptors = calculate_molecular_descriptors(mol)
        
        # Map the descriptors to our dataframe
        df.at[idx, 'MW_RDKit'] = descriptors.get('molecular_weight', 0)
        df.at[idx, 'Rotatable_Bonds'] = descriptors.get('rotatable_bonds', 0)
        df.at[idx, 'H_Acceptors'] = descriptors.get('h_acceptors', 0)
        df.at[idx, 'H_Donors'] = descriptors.get('h_donors', 0)
        df.at[idx, 'Ring_Count'] = descriptors.get('ring_count', 0)
        df.at[idx, 'Aromatic_Rings'] = descriptors.get('aromatic_rings', 0)
        
        # Extract atom counts directly from the molecule
        df.at[idx, 'C_Count'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        df.at[idx, 'F_Count'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        
        # Custom calculation for CF bonds
        cf_bonds = 0
        for bond in mol.GetBonds():
            atoms = [mol.GetAtomWithIdx(bond.GetBeginAtomIdx()), 
                     mol.GetAtomWithIdx(bond.GetEndAtomIdx())]
            symbols = [atom.GetSymbol() for atom in atoms]
            if 'C' in symbols and 'F' in symbols:
                cf_bonds += 1
        df.at[idx, 'CF_Bonds'] = cf_bonds
    
    # Fill NaN values for compounds with invalid molecules
    df[descriptor_columns] = df[descriptor_columns].fillna(0)
    
    # Calculate chain length as C count for simplicity
    df['Chain_Length'] = df['C_Count']
    
    # Calculate fluorine percentage
    def calc_f_percentage(mol, f_count):
        if mol is None or f_count == 0:
            return 0
        try:
            total_atoms = mol.GetNumAtoms()
            return (f_count / total_atoms) * 100 if total_atoms > 0 else 0
        except:
            return 0
    
    df['F_Percentage'] = df.apply(lambda x: calc_f_percentage(x['ROMol'], x['F_Count']), axis=1)
    
    print("Calculated molecular complexity features")
    
    return df

def categorize_pfas_types(df, mol_column=None):
    """Categorize PFAS compounds by structural types.
    
    Args:
        df (pandas.DataFrame): Dataframe containing RDKit molecules
        mol_column (str, optional): Name of column containing RDKit molecules. Defaults to 'ROMol' if None.
    
    Returns:
        pandas.DataFrame: Dataframe with PFAS categorization information
    """
    print("\n=== Categorizing PFAS Types ===")
    
    # Set default column name if not provided
    mol_column = mol_column or 'ROMol'
    
    # Check if the molecule column exists
    if mol_column not in df.columns:
        raise ValueError(f"Molecule column '{mol_column}' not found in dataframe")
    
    # Use the feature extractor to determine molecular properties
    feature_extractor = MolecularFeatureExtractor()
    
    def is_aromatic(mol):
        if mol is None:
            return False
        try:
            return feature_extractor.has_aromatic_rings(mol)
        except:
            return False
    
    def has_rings(mol):
        if mol is None:
            return False
        try:
            return feature_extractor.count_rings(mol) > 0
        except:
            return False
    
    def is_cyclic(mol):
        if mol is None:
            return False
        try:
            return feature_extractor.count_rings(mol) > 0
        except:
            return False
    
    def is_branched(mol):
        if mol is None:
            return False
        try:
            return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())
        except:
            return False
    
    # Determine if compounds contain fluorine
    def has_fluorine(mol):
        if mol is None:
            return False
        try:
            return any(atom.GetSymbol() == 'F' for atom in mol.GetAtoms())
        except:
            return False
    
    # Count fluorine atoms
    def count_fluorine(mol):
        if mol is None:
            return 0
        try:
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        except:
            return 0
    
    # Count carbon atoms
    def count_carbon(mol):
        if mol is None:
            return 0
        try:
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        except:
            return 0
    
    # Add basic PFAS flags
    df['is_pfas'] = df[mol_column].apply(has_fluorine)
    df['num_fluorine'] = df[mol_column].apply(count_fluorine)
    df['num_carbon'] = df[mol_column].apply(count_carbon)
    
    df['Chain_Category'] = pd.cut(
        df['num_carbon'], 
        bins=[0, 4, 7, float('inf')], 
        labels=['Short-chain', 'Medium-chain', 'Long-chain'],
        include_lowest=True
    )
    
    df['Is_Aromatic'] = df[mol_column].apply(is_aromatic)
    df['Has_Rings'] = df[mol_column].apply(has_rings)
    df['Is_Cyclic'] = df[mol_column].apply(is_cyclic)
    df['Is_Branched'] = df[mol_column].apply(is_branched)
    
    def determine_pfas_type(row):
        if pd.isna(row[mol_column]) or row[mol_column] is None:
            return 'Unknown'
        
        if not row['is_pfas']:
            return 'Non-PFAS'
            
        # Detect specific fluorinated groups
        mol = row[mol_column]
        smiles = Chem.MolToSmiles(mol) if mol else ''
        
        if 'C(F)(F)F' in smiles:
            if smiles.count('C(F)(F)F') > 1:
                return 'Multi-CF3'
            return 'CF3'
        
        if row['Is_Aromatic']:
            return 'Aromatic PFAS'
            
        type_components = []
        
        if not pd.isna(row['Chain_Category']):
            type_components.append(row['Chain_Category'])
        
        if row['Is_Cyclic'] and not row['Is_Aromatic']:
            type_components.append('Cyclic')
            
        if row['Is_Branched']:
            type_components.append('Branched')
        
        if row['num_fluorine'] > 0:
            if row['num_fluorine'] > 6:
                type_components.append('Highly-fluorinated')
            elif row['num_fluorine'] > 3:
                type_components.append('Moderately-fluorinated')
            else:
                type_components.append('Lightly-fluorinated')
        
        return ' '.join(type_components) if type_components else 'Other PFAS'
    
    df['pfas_type'] = df.apply(determine_pfas_type, axis=1)
    
    print("\nPFAS Categories:")
    if 'pfas_type' in df.columns:
        print(df['pfas_type'].value_counts())
    
    print(f"\nFound {df['is_pfas'].sum()} PFAS compounds out of {len(df)} total")
    
    return df

def calculate_pfas_statistics(df, mol_column=None):
    """Calculate PFAS-specific statistics for compounds.
    
    Args:
        df (pandas.DataFrame): Dataframe containing RDKit molecules
        mol_column (str, optional): Name of column containing RDKit molecules. Defaults to 'ROMol' if None.
    
    Returns:
        pandas.DataFrame: Dataframe with PFAS statistics
    """
    print("\n=== Calculating PFAS Statistics ===")
    
    # Set default column name if not provided
    mol_column = mol_column or 'ROMol'
    
    # Check if the molecule column exists
    if mol_column not in df.columns:
        raise ValueError(f"Molecule column '{mol_column}' not found in dataframe")
    
    # Ensure we have fluorine and carbon counts
    if 'num_fluorine' not in df.columns or 'num_carbon' not in df.columns:
        # If not already calculated, run the categorization function first
        if 'is_pfas' not in df.columns:
            df = categorize_pfas_types(df, mol_column=mol_column)
    
    # Calculate F:C ratio
    df['f_to_c_ratio'] = df.apply(
        lambda row: row['num_fluorine'] / row['num_carbon'] if row['num_carbon'] > 0 else 0, 
        axis=1
    )
    
    # Calculate average F per C
    df['avg_f_per_c'] = df.apply(
        lambda row: row['num_fluorine'] / row['num_carbon'] if row['num_carbon'] > 0 else 0, 
        axis=1
    )
    
    # Calculate molecular weight using RDKit
    def calculate_mw(mol):
        if mol is None:
            return 0
        try:
            # Use RDKit's built-in molecular weight calculator
            from rdkit.Chem import Descriptors
            return Descriptors.MolWt(mol)
        except Exception as e:
            print(f"Error calculating molecular weight: {e}")
            return 0
    
    if 'molecular_weight' not in df.columns:
        df['molecular_weight'] = df[mol_column].apply(calculate_mw)
    
    # Calculate percentage of fluorine by weight
    atomic_weight_f = 18.998  # Atomic weight of fluorine
    df['f_weight_percentage'] = df.apply(
        lambda row: (row['num_fluorine'] * atomic_weight_f / row['molecular_weight']) * 100 
        if row['molecular_weight'] > 0 else 0,
        axis=1
    )
    
    print("\nPFAS Statistics Summary:")
    print(f"Average F:C ratio: {df['f_to_c_ratio'].mean():.2f}")
    print(f"Average molecular weight: {df['molecular_weight'].mean():.2f}")
    print(f"Average fluorine content (% by atoms): {(df['num_fluorine'].sum() / df[mol_column].apply(lambda m: m.GetNumAtoms() if m else 0).sum() * 100):.2f}%")
    
    return df

def identify_fluorinated_groups(df, mol_column=None):
    """Identify specific fluorinated functional groups in molecules.
    
    Args:
        df (pandas.DataFrame): Dataframe containing RDKit molecules
        mol_column (str, optional): Name of column containing RDKit molecules. Defaults to 'ROMol' if None.
    
    Returns:
        pandas.DataFrame: Dataframe with fluorinated group information
    """
    print("\n=== Identifying Fluorinated Groups ===")
    
    # Set default column name if not provided
    mol_column = mol_column or 'ROMol'
    
    # Check if the molecule column exists
    if mol_column not in df.columns:
        raise ValueError(f"Molecule column '{mol_column}' not found in dataframe")
    
    # Define SMARTS patterns for fluorinated groups
    fluorinated_groups = {
        'CF3': '[C;!$(C([F])([F])([F])C([F])([F])F)]([F])([F])[F]',  # CF3 not part of CF3CF3
        'CF2': '[C;!$(C([F])([F])C([F])([F])([F]))]([F])([F])[!F]',  # CF2 group
        'CF': '[C;!$(C([F])C([F])([F])([F]));!$(C([F])C([F])([F])[!F])]([F])([!F])[!F]',  # CF group
        'CF3CF3': 'FC(F)(F)C(F)(F)F',  # CF3CF3 group (hexafluoroethane)
        'CF3CF2': 'FC(F)(F)C(F)F',     # CF3CF2 group
    }
    
    # Initialize columns for group counts
    for group_name in fluorinated_groups:
        col_name = f'num_{group_name.lower()}_groups'
        df[col_name] = 0
    
    # Initialize a column for detailed group information
    df['fluorinated_groups'] = ''
    
    # Count occurrences of each fluorinated group
    for group_name, smarts in fluorinated_groups.items():
        # Create a pattern for the group
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            print(f"Warning: Invalid SMARTS pattern for {group_name}: {smarts}")
            continue
        
        # Count matches in each molecule
        col_name = f'num_{group_name.lower()}_groups'
        
        def count_matches(mol):
            if mol is None:
                return 0
            try:
                return len(mol.GetSubstructMatches(pattern))
            except Exception as e:
                print(f"Error matching pattern {group_name}: {e}")
                return 0
        
        df[col_name] = df[mol_column].apply(count_matches)
    
    # Create a summary of fluorinated groups for each molecule
    def summarize_groups(row):
        summary = []
        for group_name in fluorinated_groups:
            col_name = f'num_{group_name.lower()}_groups'
            count = row[col_name]
            if count > 0:
                summary.append(f"{group_name}:{count}")
        return ', '.join(summary) if summary else 'None'
    
    df['fluorinated_groups'] = df.apply(summarize_groups, axis=1)
    
    # Convenience columns for the most common groups
    if 'num_cf3_groups' not in df.columns and 'num_cf3cf3_groups' in df.columns:
        df['num_cf3_groups'] = df['num_cf3_groups'] + (2 * df['num_cf3cf3_groups'])
    
    # Combined column for CF2 groups (including those in longer chains)
    if 'num_cf2_groups' not in df.columns and 'num_cf3cf2_groups' in df.columns:
        df['num_cf2_groups'] = df['num_cf2_groups'] + df['num_cf3cf2_groups']
    
    # Simplified CF group count
    if 'num_cf_groups' not in df.columns:
        df['num_cf_groups'] = df[mol_column].apply(
            lambda mol: len([b for b in mol.GetBonds() 
                            if mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol() in ['C', 'F'] 
                            and mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol() in ['C', 'F']
                            and set([mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(), 
                                    mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()]) == set(['C', 'F'])]) if mol else 0
        )
    
    print("\nFluorinated Group Summary:")
    for group_name in fluorinated_groups:
        col_name = f'num_{group_name.lower()}_groups'
        if col_name in df.columns:
            total = df[col_name].sum()
            print(f"  {group_name}: {total} occurrences in {(df[col_name] > 0).sum()} compounds")
    
    return df

def save_engineered_data(df):
    """Save the engineered dataset."""
    print(f"\n=== Saving Engineered Data to {ENGINEERED_DATA_PATH} ===")
    
    df = df.drop(columns=['ROMol'])
    
    os.makedirs(os.path.dirname(ENGINEERED_DATA_PATH), exist_ok=True)
    
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print("Engineered data saved successfully")
    
    return df

def engineer_features(df=None):
    """Main function to execute the feature engineering pipeline."""
    print("Starting PFAS Chemical List feature engineering process...")
    
    if df is None:
        df = load_cleaned_data()
    
    df = create_rdkit_mols(df)
    
    df = extract_fluorine_count(df)
    
    df = calculate_molecular_complexity(df)
    
    df = categorize_pfas_types(df)
    
    df = calculate_pfas_statistics(df)
    
    df = identify_fluorinated_groups(df)
    
    df = save_engineered_data(df)
    
    print("\nPFAS Chemical List feature engineering process completed successfully!")
    print(f"Engineered data saved to: {ENGINEERED_DATA_PATH}")
    
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
