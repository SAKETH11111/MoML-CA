#!/usr/bin/env python3
"""
Full Test Script for the ORCA PFAS wrapper.

This script performs a comprehensive test of the ORCA wrapper using the 
complete sample_pfas_data.csv dataset. It uses a more complete basis set (6-31G*)
which provides more accurate results but takes longer to run.

Use this script when:
1. You need to test the wrapper with multiple PFAS compounds
2. You want to verify the wrapper with a more complete basis set
3. You want to do comprehensive testing before production runs

For quick testing with just the GenX molecule, use test_orca_simple.py instead.
"""

import os
import sys
import logging
import subprocess
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_orca_wrapper")

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA = os.path.join(CURRENT_DIR, "sample_pfas_data.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "full_test_output")

def main():
    """Run comprehensive test of ORCA wrapper on sample data."""
    
    logger.info("Starting comprehensive test of ORCA PFAS wrapper")
    
    # Check if sample data exists
    if not os.path.exists(SAMPLE_DATA):
        logger.error(f"Sample data not found: {SAMPLE_DATA}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine if we're running on macOS ARM64
    is_mac_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    if is_mac_arm64:
        logger.info("Detected macOS ARM64 platform - using optimized settings")
    
    # Build command for running the wrapper
    cmd = [
        "python", os.path.join(CURRENT_DIR, "orca_pfas_wrapper.py"),
        "--input_csv", SAMPLE_DATA,
        "--output_dir", OUTPUT_DIR,
        "--functional", "B3LYP",  # Will use B3LYP-D3BJ for improved noncovalent interactions
        "--basis_set", "6-31G*",
        "--num_procs", "1",
        "--memory", "2000",
        "--max_jobs", "1"
    ]
    
    # Add Mac-specific options if on macOS ARM64
    if is_mac_arm64:
        # Check for OpenMPI installation from Homebrew
        openmpi_path = "/opt/homebrew/bin"
        if os.path.exists(os.path.join(openmpi_path, "mpirun")):
            logger.info(f"Found OpenMPI installation at: {openmpi_path}")
            cmd.extend(["--openmpi_path", openmpi_path])
        else:
            logger.warning("OpenMPI installation not found. If parallel calculations fail, install OpenMPI as per the README instructions.")
        
        # Check for ORCA installation in default macOS location
        home_dir = os.path.expanduser("~")
        orca_path = os.path.join(home_dir, "Library", "orca_6_0_1", "orca")
        if os.path.exists(orca_path):
            logger.info(f"Found ORCA installation at: {orca_path}")
            cmd.extend(["--orca_path", orca_path])
        else:
            logger.warning("ORCA 6.0.1 installation not found in standard macOS location.")
    
    # Execute command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("ORCA wrapper executed successfully")
        logger.info("Output:")
        for line in process.stdout.splitlines():
            logger.info(f"  {line}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"ORCA wrapper execution failed with error code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        sys.exit(1)
    
    # Check results
    results_file = os.path.join(OUTPUT_DIR, "orca_results_summary.csv")
    ml_data_file = os.path.join(OUTPUT_DIR, "ml_training_data.json")
    
    if os.path.exists(results_file):
        logger.info(f"Results summary file created: {results_file}")
        # Display first few lines of results file
        try:
            with open(results_file, 'r') as f:
                header = f.readline().strip()
                first_result = f.readline().strip()
                logger.info(f"Results header: {header}")
                logger.info(f"First result: {first_result}")
        except Exception as e:
            logger.error(f"Error reading results file: {e}")
    else:
        logger.warning(f"Results summary file not found: {results_file}")
    
    if os.path.exists(ml_data_file):
        logger.info(f"ML training data file created: {ml_data_file}")
        # Count number of entries in ML data file
        try:
            import json
            with open(ml_data_file, 'r') as f:
                ml_data = json.load(f)
                logger.info(f"ML training data contains {len(ml_data)} entries")
        except Exception as e:
            logger.error(f"Error reading ML data file: {e}")
    else:
        logger.warning(f"ML training data file not found: {ml_data_file}")
    
    logger.info("Comprehensive test completed")

if __name__ == "__main__":
    main() 