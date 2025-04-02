#!/usr/bin/env python3
"""
Simple Test Script for the ORCA PFAS wrapper.

This script performs calculations using the ORCA wrapper for any input dataset.
It uses a minimal basis set (STO-3G) for rapid calculations.

Use this script when:
1. You need a quick verification that the wrapper is working
2. You want to test changes to the wrapper without long computation times
3. You need a fast test for CI/CD pipelines or debugging

For comprehensive testing with a more complete basis set, use test_orca_wrapper.py instead.
"""

import os
import sys
import logging
import subprocess
import pandas as pd
from pathlib import Path
import csv
import platform
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_orca_simple")

def main():
    """Run ORCA calculations on the specified dataset."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ORCA calculations on a dataset.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file with molecule data")
    parser.add_argument("--output_dir", type=str, help="Directory for output files")
    parser.add_argument("--smiles_col", type=str, default="canonical_smiles", 
                        help="Column name containing SMILES strings (default: canonical_smiles)")
    parser.add_argument("--id_col", type=str, default="common_name", 
                        help="Column name containing molecule identifiers (default: common_name)")
    parser.add_argument("--max_compounds", type=int, default=None,
                        help="Maximum number of compounds to process (default: all)")
    parser.add_argument("--functional", type=str, default="B3LYP",
                        help="Computational method/functional to use (default: B3LYP)")
    parser.add_argument("--basis_set", type=str, default="STO-3G",
                        help="Basis set to use (default: STO-3G)")
    parser.add_argument("--num_procs", type=int, default=1,
                        help="Number of processors to use per calculation (default: 1)")
    parser.add_argument("--memory", type=int, default=2000,
                        help="Memory allocation in MB (default: 2000)")
    parser.add_argument("--max_jobs", type=int, default=1,
                        help="Maximum number of concurrent jobs (default: 1)")
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    wrapper_path = script_dir / "orca_pfas_wrapper.py"
    
    # Use provided output directory or default
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "quick_test_output"
    
    logger.info(f"Starting ORCA wrapper test with output to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    input_csv = args.input_csv
    
    # If no input CSV provided, try to use standardized dataset or fall back to sample data
    if not input_csv:
        try:
            standardized_data_path = Path(script_dir).parent.parent.parent / "data" / "processed" / "chemical_list" / "pfas20_standardized.csv"
            
            logger.info(f"No input CSV provided. Trying standardized data from: {standardized_data_path}")
            
            if standardized_data_path.exists():
                input_csv = str(standardized_data_path)
            else:
                # Fall back to sample data
                sample_data_path = script_dir / "sample_pfas_data.csv"
                if sample_data_path.exists():
                    input_csv = str(sample_data_path)
                    logger.info(f"Using sample data: {input_csv}")
                else:
                    logger.error("No input data found. Please provide an input CSV file.")
                    sys.exit(1)
        except Exception as e:
            logger.error(f"Error finding input data: {e}")
            sys.exit(1)
    
    # If we have a maximum number of compounds to process, create a filtered CSV
    if args.max_compounds:
        try:
            logger.info(f"Loading data from: {input_csv}")
            df = pd.read_csv(input_csv)
            
            # Ensure ID and SMILES columns exist
            if args.smiles_col not in df.columns:
                logger.error(f"SMILES column '{args.smiles_col}' not found in input data")
                sys.exit(1)
            
            if args.id_col not in df.columns:
                logger.warning(f"ID column '{args.id_col}' not found in input data. Using index as ID.")
                df['compound_id'] = [f"Compound{i}" for i in range(len(df))]
                args.id_col = 'compound_id'
            
            # Limit to max_compounds
            df = df.head(args.max_compounds)
            
            # Create a filtered CSV file
            filtered_csv = output_dir / "filtered_input.csv"
            df.to_csv(filtered_csv, index=False)
            
            logger.info(f"Created filtered input with {len(df)} compounds at: {filtered_csv}")
            input_csv = str(filtered_csv)
        except Exception as e:
            logger.error(f"Error processing input data: {e}")
            sys.exit(1)
    
    # Check platform-specific settings
    orca_path = None
    openmpi_path = None
    
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        logger.info("Detected macOS ARM64 platform")
        
        # Try to find OpenMPI and ORCA
        openmpi_path = '/opt/homebrew/bin'
        if os.path.exists('/opt/homebrew/bin/mpirun'):
            logger.info(f"Found OpenMPI at: {openmpi_path}")
        else:
            logger.warning("OpenMPI not found at expected location")
            openmpi_path = None
        
        orca_path = os.path.expanduser('~/Library/orca_6_0_1/orca')
        if os.path.exists(orca_path):
            logger.info(f"Found ORCA at: {orca_path}")
        else:
            logger.warning("ORCA 6.0.1 not found at expected location")
            orca_path = None
    
    # Build command to test ORCA wrapper
    command = [
        "python", str(wrapper_path),
        "--input_csv", input_csv,
        "--output_dir", str(output_dir),
        "--functional", args.functional,
        "--basis_set", args.basis_set,
        "--num_procs", str(args.num_procs),
        "--memory", str(args.memory),
        "--max_jobs", str(args.max_jobs)
    ]
    
    # Add SMILES and ID column arguments if they're not the defaults
    if args.smiles_col != "canonical_smiles":
        command.extend(["--smiles_col", args.smiles_col])
    
    if args.id_col != "common_name":
        command.extend(["--id_col", args.id_col])
    
    if orca_path:
        command.extend(["--orca_path", orca_path])
    
    if openmpi_path:
        command.extend(["--openmpi_path", openmpi_path])
    
    # Run command
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check results
    if result.returncode == 0:
        logger.info("ORCA wrapper executed successfully")
        logger.info("Output:")
        for line in result.stdout.splitlines():
            logger.info(line)
    else:
        logger.error("ORCA wrapper execution failed")
        logger.error(f"Error output: {result.stderr}")
    
    # Check result files
    results_csv = output_dir / "orca_results_summary.csv"
    if os.path.exists(results_csv):
        logger.info(f"Results summary file created: {results_csv}")
        try:
            results_df = pd.read_csv(results_csv)
            logger.info(f"Results contain {len(results_df)} entries")
            
            # Log success rate
            success_count = results_df['calculation_success'].sum() if 'calculation_success' in results_df.columns else 0
            logger.info(f"Successful calculations: {success_count}/{len(results_df)}")
        except Exception as e:
            logger.error(f"Error reading results file: {e}")
    else:
        logger.warning(f"Results summary file not found: {results_csv}")
    
    ml_data = output_dir / "ml_training_data.json"
    if os.path.exists(ml_data):
        logger.info(f"ML training data file created: {ml_data}")
        with open(ml_data, "r") as f:
            ml_data_content = json.load(f)
            logger.info(f"ML training data contains {len(ml_data_content)} entries")
    else:
        logger.warning(f"ML training data file not found: {ml_data}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 