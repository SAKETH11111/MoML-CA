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
    plot_distribution(df, 'Average_Mass', 'Molecular Weight', 'molecular_weight_distribution.png')
    
    # Plot fluorine count distribution
    plot_distribution(df, 'F_Count', 'Number of Fluorine Atoms', 'fluorine_count_distribution.png')
    
    # Plot PFAS type distribution
    plot_top_types(df, 'PFAS_Type', 'Top 10 PFAS Types', 'top_pfas_types.png')
    
    print("Chemical data visualizations created")

def analyze_treatment_data(df):
    """Perform EDA on the PFAS Treatment Data."""
    print("\n=== PFAS Treatment Data Analysis ===")
    
    # Basic statistics
    print(f"Number of treatment records: {len(df)}")
    print(f"Number of unique chemicals treated: {df['CASRN'].nunique()}")
    print(f"Number of treatment processes: {df['Treatment_Process'].nunique()}")
    
    # Plot temperature distribution
    plot_distribution(df, 'Treatment_Temp_C', 'Temperature (°C)', 'temperature_distribution.png')
    
    # Plot treatment time distribution (log scale)
    plot_distribution(df, 'Treatment_Time_Minutes', 'Time (minutes, log scale)', 'treatment_time_distribution.png', log_scale=True)
    
    # Plot effectiveness distribution
    plot_distribution(df, 'Effectiveness_Percent_Numeric', 'Effectiveness (%)', 'effectiveness_distribution.png')
    
    # Plot treatment success rate
    plot_pie_chart(df, 'Treatment_Success', ['Unsuccessful (<80%)', 'Successful (≥80%)'], 'treatment_success_rate.png')
    
    # Plot top treatment processes
    plot_top_types(df, 'Treatment_Process', 'Top 10 Treatment Processes', 'top_treatment_processes.png')
    
    # Plot treatment success by process
    plot_success_by_process(df, 'Treatment_Process', 'success_by_process.png')
    
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
    plot_heatmap(corr_matrix, 'Correlation Matrix: Chemical Properties vs Treatment Parameters', 'chemical_treatment_correlation.png')
    
    # Plot effectiveness vs fluorine content
    plot_scatter(merged_df, 'F_Count', 'Effectiveness_Percent_Numeric', 'Number of Fluorine Atoms', 'Effectiveness (%)', 'effectiveness_vs_fluorine.png')
    
    # Plot effectiveness vs molecular weight
    plot_scatter(merged_df, 'Average_Mass', 'Effectiveness_Percent_Numeric', 'Molecular Weight', 'Effectiveness (%)', 'effectiveness_vs_molecular_weight.png')
    
    # Plot temperature vs molecular weight, colored by success
    plot_scatter(merged_df, 'Average_Mass', 'Treatment_Temp_C', 'Molecular Weight', 'Temperature (°C)', 'temperature_vs_molecular_weight.png', hue='Treatment_Success')
    
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
    plot_count(df, 'Treatment_Success', 'Treatment Success (>80% Effectiveness)', 'class_imbalance.png')
    
    # Create class distribution by chemical type
    if 'PFAS_Type' in df.columns:
        # Get top 5 PFAS types
        top_types = df['PFAS_Type'].value_counts().head(5).index
        
        # Filter to only include top types
        filtered_df = df[df['PFAS_Type'].isin(top_types)]
        
        # Plot success rate by PFAS type
        plot_success_by_type(filtered_df, 'PFAS_Type', 'success_by_pfas_type.png')
    
    print("Class imbalance visualizations created")

def plot_distribution(df, column, xlabel, filename, log_scale=False):
    """Plot distribution of a given column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), bins=30, kde=True, log_scale=log_scale)
    plt.title(f'Distribution of {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_top_types(df, column, title, filename):
    """Plot top types of a given column."""
    plt.figure(figsize=(12, 8))
    top_types = df[column].value_counts().head(10)
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title(title)
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_pie_chart(df, column, labels, filename):
    """Plot pie chart of a given column."""
    plt.figure(figsize=(8, 6))
    counts = df[column].value_counts()
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title(f'{column} Distribution')
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_success_by_process(df, column, filename):
    """Plot treatment success by process."""
    plt.figure(figsize=(12, 8))
    process_success = df.groupby(column)['Treatment_Success'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=process_success.values * 100, y=process_success.index)
    plt.title('Treatment Success Rate by Process (Top 10)')
    plt.xlabel('Success Rate (%)')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_heatmap(corr_matrix, title, filename):
    """Plot heatmap of correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_scatter(df, x, y, xlabel, ylabel, filename, hue=None):
    """Plot scatter plot of two columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_count(df, column, xlabel, filename):
    """Plot count of a given column."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'{xlabel} Distribution')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.xticks([0, 1], ['False', 'True'])
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

def plot_success_by_type(df, column, filename):
    """Plot success rate by type."""
    plt.figure(figsize=(12, 8))
    sns.countplot(x=column, hue='Treatment_Success', data=df)
    plt.title('Treatment Success by PFAS Type')
    plt.xlabel('PFAS Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Success')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / filename)
    plt.close()

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
