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
import platform
import json
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_orca_simple")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ORCA calculations on a dataset.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file with molecule data")
    parser.add_argument("--output_dir", type=str, help="Directory for output files")
    parser.add_argument("--smiles_col", type=str, default="canonical_smiles", help="Column name containing SMILES strings")
    parser.add_argument("--id_col", type=str, default="common_name", help="Column name containing molecule identifiers")
    parser.add_argument("--max_compounds", type=int, default=None, help="Maximum number of compounds to process")
    parser.add_argument("--functional", type=str, default="B3LYP", help="Computational method/functional to use")
    parser.add_argument("--basis_set", type=str, default="STO-3G", help="Basis set to use")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of processors per calculation")
    parser.add_argument("--memory", type=int, default=2000, help="Memory in MB per processor")
    parser.add_argument("--max_jobs", type=int, default=1, help="Maximum concurrent jobs")
    return parser.parse_args()


def setup_paths(args):
    script_dir = Path(__file__).resolve().parent
    wrapper_path = script_dir / "orca_pfas_wrapper.py"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "quick_test_output"
    logger.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return script_dir, wrapper_path, output_dir


def load_input_data(args, script_dir, output_dir):
    input_csv = args.input_csv
    if not input_csv:
        # fallback logic
        try:
            std = Path(script_dir).parents[2] / "data/processed/chemical_list/pfas20_standardized.csv"
            input_csv = str(std) if std.exists() else str(script_dir / "sample_pfas_data.csv")
        except Exception as e:
            logger.error(f"Error finding input data: {e}")
            sys.exit(1)
    if args.max_compounds:
        try:
            df = pd.read_csv(input_csv)
            # validate columns
            if args.smiles_col not in df.columns:
                logger.error(f"SMILES column '{args.smiles_col}' not found")
                sys.exit(1)
            if args.id_col not in df.columns:
                df["compound_id"] = [f"Compound{i}" for i in range(len(df))]
                args.id_col = "compound_id"
            df = df.head(args.max_compounds)
            filtered = output_dir / "filtered_input.csv"
            df.to_csv(filtered, index=False)
            input_csv = str(filtered)
        except Exception as e:
            logger.error(f"Error processing input data: {e}")
            sys.exit(1)
    return input_csv


def detect_platform_specific_settings():
    orca_path = None
    openmpi_path = None
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        openmpi_path = "/opt/homebrew/bin" if Path("/opt/homebrew/bin/mpirun").exists() else None
        orca_path = os.path.expanduser("~/Library/orca_6_0_1/orca") if Path(os.path.expanduser("~/Library/orca_6_0_1/orca")).exists() else None
    return orca_path, openmpi_path


def build_command(args, wrapper_path, input_csv, output_dir, orca_path, openmpi_path):
    cmd = [sys.executable, str(wrapper_path), "--input_csv", input_csv, "--output_dir", str(output_dir), "--functional", args.functional, "--basis_set", args.basis_set, "--num_procs", str(args.num_procs), "--memory", str(args.memory), "--max_jobs", str(args.max_jobs)]
    if args.smiles_col != "canonical_smiles": cmd += ["--smiles_col", args.smiles_col]
    if args.id_col != "common_name": cmd += ["--id_col", args.id_col]
    if orca_path: cmd += ["--orca_path", orca_path]
    if openmpi_path: cmd += ["--openmpi_path", openmpi_path]
    return cmd


def run_orca_wrapper(command):
    logger.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error(f"Error executing wrapper: {e}")
        sys.exit(1)
    if result.returncode == 0:
        for line in result.stdout.splitlines(): logger.info(line)
    else:
        logger.error(f"Execution failed: {result.stderr}")


def check_result_files(output_dir):
    results_csv = output_dir / "orca_results_summary.csv"
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            logger.info(f"Results entries: {len(df)}")
        except Exception as e:
            logger.error(f"Error reading results: {e}")
    ml = output_dir / "ml_training_data.json"
    if ml.exists():
        try:
            data = json.load(open(ml))
            logger.info(f"ML data entries: {len(data)}")
        except Exception as e:
            logger.error(f"Error reading ML data: {e}")
    logger.info("Test completed")


def main():
    args = parse_arguments()
    script_dir, wrapper_path, output_dir = setup_paths(args)
    input_csv = load_input_data(args, script_dir, output_dir)
    orca_path, openmpi_path = detect_platform_specific_settings()
    command = build_command(args, wrapper_path, input_csv, output_dir, orca_path, openmpi_path)
    run_orca_wrapper(command)
    check_result_files(output_dir)


if __name__ == "__main__":
    main()
