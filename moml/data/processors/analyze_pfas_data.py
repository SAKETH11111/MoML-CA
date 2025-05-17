#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PFAS Data Analysis

This script performs exploratory data analysis (EDA) and alignment between the chemical and treatment datasets:
1. Loads and analyzes the chemical dataset
2. Loads and analyzes the treatment dataset
3. Creates correlation analysis between datasets
4. Aligns datasets based on common identifiers
5. Checks for class imbalance in treatment outcomes

This script now uses the consolidated moml architecture for data analysis.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Import utility functions
from moml.utils import (
    load_data,
    inspect_data,
    handle_missing_values,
    plot_distribution,
    plot_top_types,
    plot_pie_chart,
    plot_heatmap,
    plot_scatter,
    plot_success_rate,
    plot_count,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pfas_data_analysis")

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[3]
CLEANED_CHEMICAL_PATH = ROOT_DIR / "data" / "processed" / "chemical_list" / "PFAS_Chemical_List_cleaned.csv"
CLEANED_TREATMENT_PATH = ROOT_DIR / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results" / "analysis"


def analyze_chemical_data(df):
    """Analyze the chemical dataset."""
    logger.info("=== Analyzing Chemical Dataset ===")

    # Create results directory
    os.makedirs(RESULTS_DIR / "chemical", exist_ok=True)

    # Analyze molecular weight distribution
    if "MW_RDKit" in df.columns:
        plot_distribution(
            df,
            "MW_RDKit",
            "Molecular Weight (g/mol)",
            RESULTS_DIR / "chemical" / "molecular_weight_dist.png",
            log_scale=True,
        )

    # Analyze fluorine content
    if "F_Percentage" in df.columns:
        plot_distribution(
            df, "F_Percentage", "Fluorine Content (%)", RESULTS_DIR / "chemical" / "fluorine_content_dist.png"
        )

    # Analyze chain length distribution
    if "Chain_Length" in df.columns:
        plot_count(df, "Chain_Length", "Chain Length", RESULTS_DIR / "chemical" / "chain_length_dist.png")

    # Analyze structural features
    feature_columns = ["Is_Aromatic", "Has_Rings", "Is_Cyclic", "Is_Branched", "Has_Fluorine"]

    for feature in feature_columns:
        if feature in df.columns:
            plot_pie_chart(
                df,
                feature,
                [f"Has {feature}", f"No {feature}"],
                RESULTS_DIR / "chemical" / f"{feature.lower()}_distribution.png",
            )

    logger.info("Chemical dataset analysis completed")


def analyze_treatment_data(df):
    """Analyze the treatment dataset."""
    logger.info("=== Analyzing Treatment Dataset ===")

    # Create results directory
    os.makedirs(RESULTS_DIR / "treatment", exist_ok=True)

    # Analyze effectiveness distribution
    if "Effectiveness_Percent_Numeric" in df.columns:
        plot_distribution(
            df,
            "Effectiveness_Percent_Numeric",
            "Treatment Effectiveness (%)",
            RESULTS_DIR / "treatment" / "effectiveness_dist.png",
        )

    # Analyze treatment processes
    if "Treatment_Process" in df.columns:
        plot_top_types(
            df, "Treatment_Process", "Top Treatment Processes", RESULTS_DIR / "treatment" / "treatment_processes.png"
        )

    # Analyze success rate by process
    if "Treatment_Process" in df.columns and "Treatment_Success" in df.columns:
        plot_success_rate(
            df, "Treatment_Process", "Treatment_Success", RESULTS_DIR / "treatment" / "success_by_process.png"
        )

    # Analyze temperature impact
    if "Treatment_Temp_C" in df.columns and "Effectiveness_Percent_Numeric" in df.columns:
        plot_scatter(
            df,
            "Treatment_Temp_C",
            "Effectiveness_Percent_Numeric",
            "Temperature (°C)",
            "Effectiveness (%)",
            RESULTS_DIR / "treatment" / "temp_vs_effectiveness.png",
            hue="Treatment_Process",
        )

    logger.info("Treatment dataset analysis completed")


def create_correlation_analysis(chem_df, treat_df):
    """Create correlation analysis between chemical and treatment data."""
    logger.info("=== Creating Correlation Analysis ===")

    # Create results directory
    os.makedirs(RESULTS_DIR / "correlations", exist_ok=True)

    # Merge datasets on CASRN
    merged_df = pd.merge(chem_df, treat_df, on="CASRN", how="inner", suffixes=("_chem", "_treat"))

    # Select numeric columns for correlation
    numeric_columns = [
        "MW_RDKit",
        "F_Percentage",
        "Chain_Length",
        "Treatment_Temp_C",
        "Treatment_Time_Minutes",
        "Effectiveness_Percent_Numeric",
    ]

    # Filter to columns that exist in the merged dataset
    numeric_columns = [col for col in numeric_columns if col in merged_df.columns]

    if numeric_columns:
        # Calculate correlation matrix
        corr_matrix = merged_df[numeric_columns].corr()

        # Plot correlation heatmap
        plot_heatmap(corr_matrix, "Feature Correlations", RESULTS_DIR / "correlations" / "feature_correlations.png")

    logger.info("Correlation analysis completed")


def align_datasets(chem_df, treat_df):
    """Align chemical and treatment datasets."""
    logger.info("=== Aligning Datasets ===")

    # Get CASRNs from both datasets
    chem_casrns = set(chem_df["CASRN"])
    treat_casrns = set(treat_df["CASRN"])

    # Find common CASRNs
    common_casrns = chem_casrns.intersection(treat_casrns)
    logger.info(f"Found {len(common_casrns)} common CASRNs between datasets")

    # Create aligned datasets
    aligned_chem = chem_df[chem_df["CASRN"].isin(common_casrns)]
    aligned_treat = treat_df[treat_df["CASRN"].isin(common_casrns)]

    logger.info(f"Aligned chemical dataset: {len(aligned_chem)} records")
    logger.info(f"Aligned treatment dataset: {len(aligned_treat)} records")

    return aligned_chem, aligned_treat


def check_class_imbalance(df):
    """Check for class imbalance in treatment outcomes."""
    logger.info("=== Checking Class Imbalance ===")

    if "Treatment_Success" in df.columns:
        # Calculate class distribution with raw counts (not normalized)
        class_dist = df["Treatment_Success"].value_counts(normalize=False)
        logger.info(f"Class distribution:\n{class_dist}")

        # Generate dynamic labels from unique values
        labels = class_dist.index.astype(str).tolist()
        
        # Plot class distribution with dynamic labels
        plot_pie_chart(
            df,
            "Treatment_Success",
            labels,
            RESULTS_DIR / "treatment" / "class_distribution.png",
        )

        # Define mappings for positive and negative classes
        positive_classes = [True, 'True', 'Successful', 'Yes', 1, '1', 'true', 'yes']
        negative_classes = [False, 'False', 'Unsuccessful', 'No', 0, '0', 'false', 'no']

        # Calculate class counts
        dist_dict = class_dist.to_dict()
        
        # Sum up all positive class variations
        pos_count = sum(dist_dict.get(cls, 0) for cls in positive_classes if cls in dist_dict)
        
        # Sum up all negative class variations
        neg_count = sum(dist_dict.get(cls, 0) for cls in negative_classes if cls in dist_dict)
        
        logger.info(f"Positive class count: {pos_count}")
        logger.info(f"Negative class count: {neg_count}")

        # Calculate imbalance ratio (handle zero case with float('inf'))
        if pos_count == 0 and neg_count == 0:
            logger.warning("No valid class values found for Treatment_Success, cannot compute imbalance.")
        else:
            imbalance_ratio = float('inf') if neg_count == 0 else pos_count / neg_count
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    logger.info("Class imbalance check completed")


def main():
    """Main function to execute the analysis pipeline."""
    logger.info("Starting PFAS data analysis...")

    # Load data
    chem_df = load_data(CLEANED_CHEMICAL_PATH)
    treat_df = load_data(CLEANED_TREATMENT_PATH)

    # Inspect data
    inspect_data(chem_df)
    inspect_data(treat_df)

    # Handle missing values
    chem_df = handle_missing_values(chem_df)
    treat_df = handle_missing_values(treat_df)

    # Analyze datasets
    analyze_chemical_data(chem_df)
    analyze_treatment_data(treat_df)

    # Create correlation analysis
    create_correlation_analysis(chem_df, treat_df)

    # Align datasets
    aligned_chem, aligned_treat = align_datasets(chem_df, treat_df)

    # Check class imbalance
    check_class_imbalance(aligned_treat)

    logger.info("PFAS data analysis completed successfully!")


if __name__ == "__main__":
    main()
