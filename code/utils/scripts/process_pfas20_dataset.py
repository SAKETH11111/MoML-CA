#!/usr/bin/env python3
"""
Process PFAS-20 dataset to convert SMILES to standardized molecular representations.

This script handles the data standardization for the 20 PFAS compounds
by validating SMILES strings and generating standardized molecular representations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the path to import project modules
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Updated imports to match directory structure
from code.utils.helper_functions.molecular.molecule_processing import (
    process_dataset,
    save_processed_data,
    calculate_basic_descriptors
)


def main():
    """Process PFAS-20 dataset and save standardized molecular representations."""
    parser = argparse.ArgumentParser(description="Process PFAS-20 SMILES data")
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/pfas_final_dataset.csv",
        help="Path to input CSV file with PFAS data"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/processed/chemical_list",
        help="Directory to save processed files"
    )
    parser.add_argument(
        "--base-name",
        "-b",
        default="pfas20_standardized",
        help="Base filename for output files"
    )
    parser.add_argument(
        "--calculate-descriptors",
        "-d",
        action="store_true",
        help="Calculate basic molecular descriptors"
    )
    args = parser.parse_args()
    
    # Resolve input path relative to project root if not absolute
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)
    
    # Resolve output directory relative to project root if not absolute
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    # Process the dataset
    print(f"Processing PFAS-20 dataset from: {input_path}")
    df = process_dataset(input_path)
    
    # Calculate descriptors if requested
    if args.calculate_descriptors:
        print("Calculating basic molecular descriptors...")
        df = calculate_basic_descriptors(df)
    
    # Save processed data
    output_files = save_processed_data(df, output_dir, args.base_name)
    
    # Print summary
    valid_count = df['is_valid_smiles'].sum()
    total_count = len(df)
    print(f"\nProcessing Summary:")
    print(f"- Total compounds: {total_count}")
    print(f"- Valid SMILES: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"- Invalid SMILES: {total_count - valid_count}")
    
    print("\nOutput Files:")
    for file_type, file_path in output_files.items():
        print(f"- {file_type}: {file_path}")


if __name__ == "__main__":
    main() 