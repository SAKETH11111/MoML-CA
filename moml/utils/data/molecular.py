#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecular Data Processing Utilities

This module contains general-purpose molecular data processing functions used across
different data processors in the MoML framework.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Union, List, Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("molecular_processing")

def create_rdkit_mols(df: pd.DataFrame, 
                     smiles_col: str = 'SMILES',
                     mol_col: str = 'ROMol') -> pd.DataFrame:
    """Create RDKit molecule objects from SMILES strings.
    
    Args:
        df: DataFrame containing SMILES strings
        smiles_col: Name of column containing SMILES strings
        mol_col: Name of column to store RDKit molecules
    
    Returns:
        DataFrame with RDKit molecules and validity flags
    """
    logger.info("=== Creating RDKit Molecule Objects ===")
    
    valid_col = 'Valid_SMILES'
    if smiles_col != 'SMILES':
        valid_col = f'is_valid_{smiles_col}'
    
    # Check if the SMILES column exists
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataframe")
    
    # Create RDKit molecules and validity flags
    df[mol_col] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) else None)
    df[valid_col] = df[mol_col].notna()
    
    # Log statistics
    valid_count = df[valid_col].sum()
    total_count = len(df)
    logger.info(f"Created RDKit molecules for {valid_count}/{total_count} SMILES strings")
    
    return df

def extract_fluorine_count(df: pd.DataFrame, 
                         mol_col: str = 'ROMol') -> pd.DataFrame:
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
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    
    def calc_f_percentage(mol, f_count):
        if mol is None or f_count == 0:
            return 0.0
        total_atoms = mol.GetNumAtoms()
        return (f_count / total_atoms) * 100
    
    # Calculate fluorine counts
    df['F_Count'] = df[mol_col].apply(count_fluorine)
    df['F_Percentage'] = df.apply(lambda x: calc_f_percentage(x[mol_col], x['F_Count']), axis=1)
    
    logger.info("Added fluorine count and percentage columns")
    return df

def calculate_molecular_complexity(df: pd.DataFrame, 
                                mol_col: str = 'ROMol') -> pd.DataFrame:
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
    df['MW_RDKit'] = df[mol_col].apply(lambda x: Chem.Descriptors.ExactMolWt(x) if x is not None else None)
    df['Rotatable_Bonds'] = df[mol_col].apply(lambda x: Chem.Descriptors.NumRotatableBonds(x) if x is not None else None)
    df['H_Acceptors'] = df[mol_col].apply(lambda x: Chem.Descriptors.NumHAcceptors(x) if x is not None else None)
    df['H_Donors'] = df[mol_col].apply(lambda x: Chem.Descriptors.NumHDonors(x) if x is not None else None)
    
    # Calculate ring information
    df['Ring_Count'] = df[mol_col].apply(lambda x: Chem.Descriptors.RingCount(x) if x is not None else None)
    df['Aromatic_Rings'] = df[mol_col].apply(lambda x: Chem.Descriptors.NumAromaticRings(x) if x is not None else None)
    
    # Calculate carbon and fluorine counts
    df['C_Count'] = df[mol_col].apply(lambda x: sum(1 for atom in x.GetAtoms() if atom.GetSymbol() == 'C') if x is not None else None)
    df['F_Count'] = df[mol_col].apply(lambda x: sum(1 for atom in x.GetAtoms() if atom.GetSymbol() == 'F') if x is not None else None)
    
    # Calculate chain length (longest carbon chain)
    def get_chain_length(mol):
        if mol is None:
            return 0
        # Get all carbon atoms
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
        if not carbon_atoms:
            return 0
        # Find the longest path between carbon atoms
        max_length = 0
        for start in carbon_atoms:
            for end in carbon_atoms:
                if start != end:
                    path_length = len(Chem.GetShortestPath(mol, start.GetIdx(), end.GetIdx()))
                    max_length = max(max_length, path_length)
        return max_length
    
    df['Chain_Length'] = df[mol_col].apply(get_chain_length)
    
    # Calculate CF bonds
    def count_cf_bonds(mol):
        if mol is None:
            return 0
        return sum(1 for bond in mol.GetBonds() 
                  if (bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'F') or
                     (bond.GetBeginAtom().GetSymbol() == 'F' and bond.GetEndAtom().GetSymbol() == 'C'))
    
    df['CF_Bonds'] = df[mol_col].apply(count_cf_bonds)
    
    # Calculate F percentage
    df['F_Percentage'] = df.apply(lambda x: calc_f_percentage(x[mol_col], x['F_Count']), axis=1)
    
    logger.info("Added molecular complexity features")
    return df

def categorize_molecular_features(df: pd.DataFrame, 
                               mol_col: str = 'ROMol') -> pd.DataFrame:
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
        return Chem.Descriptors.RingCount(mol) > 0 and not any(atom.GetSymbol() == 'F' for atom in mol.GetAtoms())
    
    def is_branched(mol):
        if mol is None:
            return False
        return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())
    
    def has_fluorine(mol):
        if mol is None:
            return False
        return any(atom.GetSymbol() == 'F' for atom in mol.GetAtoms())
    
    # Add feature flags
    df['Is_Aromatic'] = df[mol_col].apply(is_aromatic)
    df['Has_Rings'] = df[mol_col].apply(has_rings)
    df['Is_Cyclic'] = df[mol_col].apply(is_cyclic)
    df['Is_Branched'] = df[mol_col].apply(is_branched)
    df['Has_Fluorine'] = df[mol_col].apply(has_fluorine)
    
    # Categorize chain length
    df['Chain_Category'] = pd.cut(
        df['Chain_Length'],
        bins=[-float('inf'), 2, 4, 6, float('inf')],
        labels=['Very Short', 'Short', 'Medium', 'Long']
    )
    
    logger.info("Added molecular feature categories")
    return df 