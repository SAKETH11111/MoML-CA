#!/usr/bin/env python3
"""
Quick Test Script for the ORCA PFAS wrapper.

This script performs a fast test of the ORCA wrapper using only the 
GenX molecule (PFAS003) from the standardized dataset. It uses a minimal
basis set (STO-3G) for rapid calculations.

Use this script when:
1. You need a quick verification that the wrapper is working
2. You want to test changes to the wrapper without long computation times
3. You need a fast test for CI/CD pipelines or debugging

For comprehensive testing with multiple compounds, use test_orca_wrapper.py instead.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_orca_simple")

def main():
    """Run a quick test with GenX molecule from the standardized data."""
    
    logger.info("Starting quick test of ORCA PFAS wrapper with GenX molecule")
    
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    wrapper_path = script_dir / "orca_pfas_wrapper.py"
    output_dir = script_dir / "quick_test_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load GenX data from the standardized dataset
    try:
        standardized_data_path = Path(script_dir).parent.parent.parent / "data" / "processed" / "chemical_list" / "pfas20_standardized.csv"
        
        logger.info(f"Loading standardized data from: {standardized_data_path}")
        
        # Check if file exists
        if not standardized_data_path.exists():
            logger.error(f"Standardized data file not found at: {standardized_data_path}")
            standardized_data_path = None
        else:
            # Read the standardized dataset
            df = pd.read_csv(standardized_data_path)
            
            # Filter for GenX
            genx_data = df[df['common_name'] == 'GenX']
            
            if genx_data.empty:
                logger.error("GenX data not found in standardized dataset")
                # Fall back to sample data
                sample_data_path = script_dir / "sample_pfas_data.csv"
                sample_df = pd.read_csv(sample_data_path)
                genx_data = sample_df[sample_df['ID'] == 'PFAS003']
                
                # Create a CSV file with only GenX data
                genx_only_path = output_dir / "genx_only.csv"
                with open(genx_only_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Name', 'SMILES'])
                    writer.writerow(['PFAS003', 'GenX', genx_data['SMILES'].values[0]])
                
                logger.info(f"Created filtered CSV with sample GenX data at: {genx_only_path}")
            else:
                # Create a CSV file with only GenX data using standardized SMILES
                genx_only_path = output_dir / "genx_only.csv"
                with open(genx_only_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Name', 'SMILES'])
                    writer.writerow(['PFAS003', 'GenX', genx_data['canonical_smiles'].values[0]])
                
                logger.info(f"Created filtered CSV with standardized GenX data at: {genx_only_path}")
    except Exception as e:
        logger.error(f"Error loading standardized data: {e}")
        # Fallback to sample data
        sample_data_path = script_dir / "sample_pfas_data.csv"
        sample_df = pd.read_csv(sample_data_path)
        genx_data = sample_df[sample_df['ID'] == 'PFAS003']
        
        # Create a CSV file with only GenX data
        genx_only_path = output_dir / "genx_only.csv"
        with open(genx_only_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'SMILES'])
            writer.writerow(['PFAS003', 'GenX', genx_data['SMILES'].values[0]])
        
        logger.info(f"Created filtered CSV with sample GenX data at: {genx_only_path}")
    
    # Check platform-specific settings
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
        "--input_csv", str(genx_only_path),
        "--output_dir", str(output_dir),
        "--functional", "B3LYP",  # Will use B3LYP-D3BJ for improved noncovalent interactions
        "--basis_set", "STO-3G",  # Use a smaller basis set for faster testing
        "--num_procs", "1",
        "--memory", "2000",
        "--max_jobs", "1"
    ]
    
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
        with open(results_csv, "r") as f:
            logger.info(f"Results content: {f.read()}")
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
    
    logger.info("Quick test completed")

if __name__ == "__main__":
    main() 