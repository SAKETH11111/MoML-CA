#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PFAS Data Analysis and Alignment

This script performs exploratory data analysis (EDA) and alignment between
the PFAS Chemical List and Treatment Data datasets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_CHEMICAL_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_engineered.csv"
PROCESSED_TREATMENT_PATH = ROOT_DIR / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
ALIGNED_DATA_PATH = ROOT_DIR / "data" / "processed" / "aligned" / "PFAS_Aligned_Data.csv"
VISUALIZATION_DIR = ROOT_DIR / "experiments" / "results" / "cross_analysis"

def load_data():
    """Load the processed PFAS datasets."""
    print(f"Loading chemical data from: {PROCESSED_CHEMICAL_PATH}")
    chemical_df = pd.read_csv(PROCESSED_CHEMICAL_PATH)
    
    print(f"Loading treatment data from: {PROCESSED_TREATMENT_PATH}")
    treatment_df = pd.read_csv(PROCESSED_TREATMENT_PATH)
    
    return chemical_df, treatment_df

def analyze_chemical_data(df):
    """Perform EDA on the PFAS Chemical List data."""
    print("\n=== PFAS Chemical List Analysis ===")
    
    # Basic statistics
    print(f"Number of chemicals: {len(df)}")
    print(f"Number of unique CASRNs: {df['CASRN'].nunique()}")
    
    # Create directory if it doesn't exist
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Plot molecular weight distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Average_Mass'], bins=30, kde=True)
    plt.title('Distribution of Molecular Weights')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / 'molecular_weight_distribution.png')
    plt.close()
    
    # Plot fluorine count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['F_Count'], bins=30, kde=True)
    plt.title('Distribution of Fluorine Counts')
    plt.xlabel('Number of Fluorine Atoms')
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / 'fluorine_count_distribution.png')
    plt.close()
    
    # Plot PFAS type distribution
    plt.figure(figsize=(12, 8))
    top_types = df['PFAS_Type'].value_counts().head(10)
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title('Top 10 PFAS Types')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'top_pfas_types.png')
    plt.close()
    
    print("Chemical data visualizations created")

def analyze_treatment_data(df):
    """Perform EDA on the PFAS Treatment Data."""
    print("\n=== PFAS Treatment Data Analysis ===")
    
    # Basic statistics
    print(f"Number of treatment records: {len(df)}")
    print(f"Number of unique chemicals treated: {df['CASRN'].nunique()}")
    print(f"Number of treatment processes: {df['Treatment_Process'].nunique()}")
    
    # Plot temperature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Treatment_Temp_C'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Treatment Temperatures')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / 'temperature_distribution.png')
    plt.close()
    
    # Plot treatment time distribution (log scale)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Treatment_Time_Minutes'].dropna(), bins=30, kde=True, log_scale=True)
    plt.title('Distribution of Treatment Times (Log Scale)')
    plt.xlabel('Time (minutes, log scale)')
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / 'treatment_time_distribution.png')
    plt.close()
    
    # Plot effectiveness distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Effectiveness_Percent_Numeric'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Treatment Effectiveness')
    plt.xlabel('Effectiveness (%)')
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / 'effectiveness_distribution.png')
    plt.close()
    
    # Plot treatment success rate
    plt.figure(figsize=(8, 6))
    success_counts = df['Treatment_Success'].value_counts()
    labels = ['Unsuccessful (<80%)', 'Successful (≥80%)']
    plt.pie(success_counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('Treatment Success Rate')
    plt.savefig(VISUALIZATION_DIR / 'treatment_success_rate.png')
    plt.close()
    
    # Plot top treatment processes
    plt.figure(figsize=(12, 8))
    top_processes = df['Treatment_Process'].value_counts().head(10)
    sns.barplot(x=top_processes.values, y=top_processes.index)
    plt.title('Top 10 Treatment Processes')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'top_treatment_processes.png')
    plt.close()
    
    # Plot treatment success by process
    plt.figure(figsize=(12, 8))
    process_success = df.groupby('Treatment_Process')['Treatment_Success'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=process_success.values * 100, y=process_success.index)
    plt.title('Treatment Success Rate by Process (Top 10)')
    plt.xlabel('Success Rate (%)')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'success_by_process.png')
    plt.close()
    
    print("Treatment data visualizations created")

def create_correlation_analysis(chem_df, treat_df):
    """Perform correlation analysis between chemical properties and treatment effectiveness."""
    print("\n=== Correlation Analysis ===")
    
    # Merge the datasets on CASRN
    merged_df = pd.merge(
        treat_df[['CASRN', 'Treatment_Process', 'Treatment_Temp_C', 'Treatment_Time_Minutes', 
                 'Effectiveness_Percent_Numeric', 'Treatment_Success']],
        chem_df[['CASRN', 'Average_Mass', 'F_Count', 'F_Percentage', 'C_Count', 'Chain_Length', 
                'Ring_Count', 'Is_Aromatic', 'Is_Cyclic', 'Is_Branched']],
        on='CASRN',
        how='inner'
    )
    
    print(f"Merged dataset has {len(merged_df)} records")
    
    # Create correlation matrix
    corr_cols = ['Effectiveness_Percent_Numeric', 'Treatment_Temp_C', 'Treatment_Time_Minutes',
                'Average_Mass', 'F_Count', 'F_Percentage', 'C_Count', 'Chain_Length', 'Ring_Count']
    
    corr_matrix = merged_df[corr_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix: Chemical Properties vs Treatment Parameters')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'chemical_treatment_correlation.png')
    plt.close()
    
    # Plot effectiveness vs fluorine content
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='F_Count', y='Effectiveness_Percent_Numeric', alpha=0.6)
    plt.title('Treatment Effectiveness vs Fluorine Count')
    plt.xlabel('Number of Fluorine Atoms')
    plt.ylabel('Effectiveness (%)')
    plt.savefig(VISUALIZATION_DIR / 'effectiveness_vs_fluorine.png')
    plt.close()
    
    # Plot effectiveness vs molecular weight
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='Average_Mass', y='Effectiveness_Percent_Numeric', alpha=0.6)
    plt.title('Treatment Effectiveness vs Molecular Weight')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Effectiveness (%)')
    plt.savefig(VISUALIZATION_DIR / 'effectiveness_vs_molecular_weight.png')
    plt.close()
    
    # Plot temperature vs molecular weight, colored by success
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='Average_Mass', y='Treatment_Temp_C', hue='Treatment_Success', alpha=0.6)
    plt.title('Treatment Temperature vs Molecular Weight')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Temperature (°C)')
    plt.savefig(VISUALIZATION_DIR / 'temperature_vs_molecular_weight.png')
    plt.close()
    
    print("Correlation analysis visualizations created")
    
    return merged_df

def align_datasets(chem_df, treat_df):
    """Align the PFAS Chemical List and Treatment Data datasets."""
    print("\n=== Aligning Datasets ===")
    
    # Check dataset alignment
    chem_casrns = set(chem_df['CASRN'])
    treat_casrns = set(treat_df['CASRN'])
    
    common_casrns = chem_casrns.intersection(treat_casrns)
    only_in_chem = chem_casrns - treat_casrns
    only_in_treat = treat_casrns - chem_casrns
    
    print(f"Chemicals in both datasets: {len(common_casrns)}")
    print(f"Chemicals only in Chemical List: {len(only_in_chem)}")
    print(f"Chemicals only in Treatment Data: {len(only_in_treat)}")
    
    # Create aligned dataset with selected columns
    aligned_df = pd.merge(
        chem_df[[
            'CASRN', 'Preferred_Name', 'SMILES', 'Molecular_Formula', 'Average_Mass', 
            'F_Count', 'F_Percentage', 'C_Count', 'CF_Bonds', 'Chain_Length', 
            'MW_RDKit', 'Rotatable_Bonds', 'H_Acceptors', 'H_Donors', 'Ring_Count', 
            'Aromatic_Rings', 'Chain_Category', 'Is_Aromatic', 'Has_Rings', 
            'Is_Cyclic', 'Is_Branched', 'PFAS_Type'
        ]],
        treat_df[[
            'CASRN', 'Treatment_Process', 'Test_Scale', 'Matrix', 'Treatment_Temp_C', 
            'Treatment_Time_Minutes', 'Effectiveness_Percent_Numeric', 'Treatment_Success',
            'Temperature_Category', 'Time_Category', 'Initial_Concentration_Numeric',
            'Post_Concentration_Numeric'
        ]],
        on='CASRN',
        how='inner'
    )
    
    print(f"Aligned dataset has {len(aligned_df)} records")
    
    # Save aligned dataset
    aligned_df.to_csv(ALIGNED_DATA_PATH, index=False)
    print(f"Aligned dataset saved to: {ALIGNED_DATA_PATH}")
    
    return aligned_df

def check_class_imbalance(df):
    """Check for class imbalance in treatment success."""
    print("\n=== Checking Class Imbalance ===")
    
    success_counts = df['Treatment_Success'].value_counts()
    print("Treatment Success Counts:")
    print(success_counts)
    
    success_ratio = success_counts[True] / len(df) if True in success_counts else 0
    print(f"Success ratio: {success_ratio:.2f}")
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Treatment_Success', data=df)
    plt.title('Treatment Success Distribution')
    plt.xlabel('Treatment Success (>80% Effectiveness)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['False', 'True'])
    plt.savefig(VISUALIZATION_DIR / 'class_imbalance.png')
    plt.close()
    
    # Create class distribution by chemical type
    if 'PFAS_Type' in df.columns:
        # Get top 5 PFAS types
        top_types = df['PFAS_Type'].value_counts().head(5).index
        
        # Filter to only include top types
        filtered_df = df[df['PFAS_Type'].isin(top_types)]
        
        # Plot success rate by PFAS type
        plt.figure(figsize=(12, 8))
        sns.countplot(x='PFAS_Type', hue='Treatment_Success', data=filtered_df)
        plt.title('Treatment Success by PFAS Type')
        plt.xlabel('PFAS Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Success')
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / 'success_by_pfas_type.png')
        plt.close()
    
    print("Class imbalance visualizations created")

def main():
    """Main function to execute the data analysis and alignment."""
    print("Starting PFAS data analysis and alignment process...")
    
    # Create visualization directory
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Load data
    try:
        chemical_df, treatment_df = load_data()
        
        # Analyze chemical data
        analyze_chemical_data(chemical_df)
        
        # Analyze treatment data
        analyze_treatment_data(treatment_df)
        
        # Align datasets
        aligned_df = align_datasets(chemical_df, treatment_df)
        
        # Create correlation analysis
        merged_df = create_correlation_analysis(chemical_df, treatment_df)
        
        # Check class imbalance
        check_class_imbalance(aligned_df)
        
        print("\nPFAS data analysis and alignment process completed successfully!")
        print(f"Visualizations saved to: {VISUALIZATION_DIR}")
        print(f"Aligned dataset saved to: {ALIGNED_DATA_PATH}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both processed datasets exist before running this script.")

if __name__ == "__main__":
    main() 