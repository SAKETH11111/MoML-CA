#!/usr/bin/env python3
"""
Module for processing PFAS molecular structures.
Handles standardization and validation of SMILES strings using RDKit.
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit import RDLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("molecule_processing")

# Suppress RDKit logging except for warnings and errors
RDLogger.logger().setLevel(RDLogger.WARNING)


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a SMILES string and convert to canonical form.
    
    Args:
        smiles: The SMILES string to validate
        
    Returns:
        Tuple containing:
            - Boolean indicating if SMILES is valid
            - Canonical SMILES (if valid, otherwise None)
            - Error message (if invalid, otherwise None)
    """
    if not smiles or not isinstance(smiles, str):
        return False, None, "Empty or non-string SMILES input"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, f"Invalid SMILES: {smiles}"
        
        # Generate canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return True, canonical_smiles, None
    
    except Exception as e:
        return False, None, f"Error processing SMILES: {str(e)}"


def process_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process a CSV file containing PFAS data, validating SMILES strings.
    
    Args:
        csv_path: Path to the CSV file with SMILES data
        
    Returns:
        DataFrame with added columns for RDKit molecule objects and validation results
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} compounds")
        
        # Check if SMILES column exists
        if 'SMILES' not in df.columns:
            logger.error("SMILES column not found in the dataset")
            raise ValueError("SMILES column not found in the dataset")
        
        # Initialize new columns
        df['rdkit_mol'] = None
        df['canonical_smiles'] = None
        df['is_valid_smiles'] = False
        df['smiles_error'] = None
        
        # Process each SMILES string
        valid_count = 0
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            is_valid, canonical_smiles, error = validate_smiles(smiles)
            
            df.at[idx, 'is_valid_smiles'] = is_valid
            df.at[idx, 'smiles_error'] = error
            
            if is_valid:
                df.at[idx, 'canonical_smiles'] = canonical_smiles
                df.at[idx, 'rdkit_mol'] = Chem.MolFromSmiles(canonical_smiles)
                valid_count += 1
            
        logger.info(f"Successfully processed {valid_count}/{len(df)} compounds")
        return df
    
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, output_dir: str, base_filename: str) -> Dict[str, str]:
    """
    Save processed molecular data to disk in multiple formats.
    
    Args:
        df: DataFrame with processed molecular data
        output_dir: Directory to save files
        base_filename: Base name for output files
        
    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}
    
    # Save as CSV (without RDKit mol objects)
    csv_df = df.drop(columns=['rdkit_mol'])
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    csv_df.to_csv(csv_path, index=False)
    output_files['csv'] = csv_path
    
    # Save valid molecules as pickle file
    valid_mols = {}
    for idx, row in df[df['is_valid_smiles']].iterrows():
        valid_mols[row['common_name']] = row['rdkit_mol']
    
    pkl_path = os.path.join(output_dir, f"{base_filename}_mols.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(valid_mols, f)
    output_files['pickle'] = pkl_path
    
    # Generate a report of validation issues
    if not df['is_valid_smiles'].all():
        report_df = df[~df['is_valid_smiles']][['common_name', 'SMILES', 'smiles_error']]
        report_path = os.path.join(output_dir, f"{base_filename}_validation_issues.csv")
        report_df.to_csv(report_path, index=False)
        output_files['issues_report'] = report_path
    
    logger.info(f"Saved processed data to {output_dir}")
    return output_files


def calculate_basic_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic molecular descriptors for valid molecules.
    
    Args:
        df: DataFrame with RDKit molecule objects
        
    Returns:
        DataFrame with additional descriptor columns
    """
    # Only process valid molecules
    valid_df = df[df['is_valid_smiles']].copy()
    
    # Calculate basic descriptors
    valid_df['molecular_weight'] = valid_df['rdkit_mol'].apply(
        lambda mol: Descriptors.MolWt(mol) if mol is not None else None
    )
    
    valid_df['logp'] = valid_df['rdkit_mol'].apply(
        lambda mol: Descriptors.MolLogP(mol) if mol is not None else None
    )
    
    valid_df['num_heavy_atoms'] = valid_df['rdkit_mol'].apply(
        lambda mol: mol.GetNumHeavyAtoms() if mol is not None else None
    )
    
    valid_df['num_rotatable_bonds'] = valid_df['rdkit_mol'].apply(
        lambda mol: Descriptors.NumRotatableBonds(mol) if mol is not None else None
    )
    
    # Merge back with original DataFrame
    result_df = df.copy()
    for col in ['molecular_weight', 'logp', 'num_heavy_atoms', 'num_rotatable_bonds']:
        result_df[col] = None
    
    for idx, row in valid_df.iterrows():
        for col in ['molecular_weight', 'logp', 'num_heavy_atoms', 'num_rotatable_bonds']:
            result_df.at[idx, col] = row[col]
    
    return result_df 