#!/usr/bin/env python3
# Copyright 2025 MoML-CA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
ORCA PFAS Wrapper

This script provides a command-line interface for running ORCA calculations on PFAS molecules.
It handles the entire workflow from SMILES input to final quantum mechanical properties.

Usage:
    python moml/simulation/qm/orca_pfas_wrapper.py --input_csv path/to/input.csv --output_dir path/to/output
    python moml/simulation/qm/orca_pfas_wrapper.py --help

This module provides a command-line interface and utility functions to:
1.  Process PFAS (Per- and Polyfluoroalkyl Substances) compounds from input CSV files.
2.  Generate 3D molecular structures from SMILES strings.
3.  Create ORCA input files tailored for PFAS quantum chemistry calculations.
4.  Execute ORCA calculations, either serially or in parallel.
5.  Orchestrate the conversion of ORCA output files to QM9-style NPZ format
    using an external script.
6.  Generate a summary report of the calculation outcomes.

The wrapper is designed to facilitate high-throughput quantum mechanics
calculations for PFAS, which are crucial for generating data for training
machine learning models within the MoML-CA framework. Specific attention is
given to the unique electronic properties of PFAS molecules.

For detailed help:
    python moml/simulation/qm/orca_pfas_wrapper.py --help
"""

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orca_pfas_wrapper")

# Suppress RDKit informational and debug messages, showing only warnings and errors
RDLogger.logger().setLevel(RDLogger.WARNING)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the ORCA wrapper script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="ORCA PFAS Wrapper for quantum chemistry calculations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing PFAS data. Must include SMILES and an identifier column.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where ORCA calculation files and results will be saved.",
    )

    # Optional arguments for data handling
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="canonical_smiles",
        help="Name of the column in the input CSV that contains the SMILES strings for PFAS molecules.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="common_name",
        help="Name of the column in the input CSV that contains unique identifiers for the PFAS molecules.",
    )

    # Optional arguments for ORCA calculation parameters
    parser.add_argument(
        "--functional",
        type=str,
        default="wB97X-D",  # Changed default to align with MoML-CA QM protocol
        help="Density functional to be used in ORCA calculations (e.g., 'wB97X-D', 'B3LYP').",
    )
    parser.add_argument(
        "--basis_set",
        type=str,
        default="def2-TZVP",  # Changed default to align with MoML-CA QM protocol
        help="Basis set to be used in ORCA calculations (e.g., 'def2-TZVP', '6-31G*').",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=1,
        help="Number of processors (cores) to allocate for each individual ORCA calculation.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=2000,
        help="Memory allocation in MB for each individual ORCA calculation.",
    )
    parser.add_argument(
        "--orca_path",
        type=str,
        default=None,
        help="Full path to the ORCA executable. If None, the script will try common paths or use 'orca' from system PATH.",
    )
    parser.add_argument(
        "--openmpi_path",
        type=str,
        default=None,
        help="Path to OpenMPI binaries, if required for parallel ORCA execution. If None, system defaults are used.",
    )

    # Optional arguments for calculation types
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Perform geometry optimization. This is the default unless --sp_only is specified.",
    )
    parser.add_argument(
        "--charges",
        action="store_true",
        help="Request calculation of partial atomic charges (e.g., Mulliken, Loewdin) by ORCA.",
    )
    parser.add_argument(
        "--sp_only",
        action="store_true",
        help="Run a single-point energy calculation only. If set, geometry optimization is skipped unless --optimize is also explicitly set.",
    )
    parser.add_argument(
        "--solvent_model",
        type=str,
        default="CPCM(Water)",
        help="Implicit solvent model to use (e.g., 'CPCM(Water)', 'SMD(Water)'). Set to None or empty string for gas phase.",
    )

    # Optional arguments for execution control
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=1,
        help="Maximum number of ORCA calculations to run concurrently. Set to 1 for serial execution.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode. Generates input files and simulates ORCA execution without actual calculations.",
    )
    parser.add_argument(
        "--conversion_script_path",
        type=str,
        default="scripts/orca_json_to_qm9_npz.py",
        help="Path to the Python script used for converting ORCA output to QM9 NPZ format.",
    )

    return parser.parse_args()


def generate_3d_structure(smiles: str, molecule_id: str) -> Optional[Chem.Mol]:
    """
    Generates a 3D molecular structure from a SMILES string.

    This involves parsing the SMILES, adding explicit hydrogens, embedding the
    molecule to generate initial 3D coordinates, and performing a quick
    force field optimization (MMFF) to refine the geometry.

    Args:
        smiles: The SMILES string of the molecule.
        molecule_id: A unique identifier for the molecule, used for logging.

    Returns:
        An RDKit Mol object with 3D coordinates if successful, otherwise None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Failed to parse SMILES for {molecule_id}: {smiles}")
            return None

        mol = Chem.AddHs(mol)

        # Embed molecule with a fixed random seed for reproducibility
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:  # Embedding failed
            logger.warning(
                f"Initial 3D coordinate generation failed for {molecule_id}. Trying with useRandomCoords=True."
            )
            status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if status == -1:
                logger.error(f"3D coordinate generation ultimately failed for {molecule_id} after multiple attempts.")
                return None

        # Refine the structure using MMFF94 force field and check for failure
        opt_status = AllChem.MMFFOptimizeMolecule(mol)
        if opt_status != 0:
            logger.error(f"MMFF optimization failed for {molecule_id}, status {opt_status}")
            return None
        logger.info(f"Successfully generated 3D structure for {molecule_id}")
        return mol

    except Exception as e:
        logger.error(f"Exception during 3D structure generation for {molecule_id} (SMILES: {smiles}): {e}")
        return None


def generate_orca_input(
    mol: Chem.Mol,
    molecule_id: str,
    output_dir: str,
    functional: str,
    basis_set: str,
    num_procs: int,
    memory: int,
    optimize_geom: bool,
    calculate_charges: bool,
    sp_only: bool,
    solvent_model: Optional[str],
) -> Tuple[bool, str]:
    """
    Generates an ORCA input file for a given molecule and calculation parameters.

    The input file is configured for PFAS calculations, considering typical
    requirements like tight SCF convergence and appropriate keywords for
    optimization or single-point calculations.

    Args:
        mol: RDKit Mol object with 3D coordinates.
        molecule_id: Unique identifier for the molecule.
        output_dir: Directory where the ORCA input file will be saved.
        functional: Density functional to use (e.g., 'wB97X-D').
        basis_set: Basis set to use (e.g., 'def2-TZVP').
        num_procs: Number of processors for parallel execution.
        memory: Memory allocation in MB per processor.
        optimize_geom: Whether to perform geometry optimization.
        calculate_charges: Whether to request partial charge calculation.
        sp_only: If True, sets up a single-point energy calculation.
        solvent_model: Implicit solvent model (e.g., 'CPCM(Water)'). None for gas phase.


    Returns:
        A tuple (success, input_file_path):
        - success (bool): True if input file generation was successful, False otherwise.
        - input_file_path (str): The path to the generated ORCA input file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        input_file_path = os.path.join(output_dir, f"{molecule_id}.inp")

        keywords = [functional, basis_set, "TightSCF"]

        if sp_only and not optimize_geom:  # Pure single point
            pass  # No specific keyword, default is SP
        elif optimize_geom:
            keywords.append("Opt")
            keywords.append("Freq")  # Also calculate frequencies to confirm minimum

        if calculate_charges:
            keywords.append("Mulliken")  # Request Mulliken charges
            keywords.append("Loewdin")  # Request Loewdin charges
            keywords.append("CHELPG")  # Request CHELPG charges, often good for PFAS

        if solvent_model and solvent_model.strip():
            keywords.append(solvent_model)
            if (
                "SMD" in solvent_model.upper() and "CPCM" not in solvent_model.upper()
            ):  # ORCA needs smd true in cpcm block
                pass  # SMD is handled differently below for ORCA syntax

        input_block = f"! {' '.join(keywords)}\n\n"

        if num_procs > 1:
            input_block += f"%pal\n  nprocs {num_procs}\nend\n\n"

        input_block += f"%maxcore {memory}\n\n"  # Memory per core

        # SCF settings for robustness, especially with PFAS
        input_block += "%scf\n  MaxIter 300\n  Convergence Tight\nend\n\n"

        if optimize_geom and not sp_only:
            input_block += "%geom\n  MaxIter 300\n  Convergence Tight\n  Trust -0.1\nend\n\n"

        # Add CPCM/SMD block if solvent model requires it
        if solvent_model and solvent_model.strip():
            if "CPCM" in solvent_model.upper() or "SMD" in solvent_model.upper():
                input_block += "%cpcm\n"
                if "SMD" in solvent_model.upper():  # e.g. SMD(Water)
                    solvent_name = solvent_model.split("(")[-1].split(")")[0]
                    input_block += f'  smd true\n  smdsolvent "{solvent_name}"\n'
                elif "CPCM" in solvent_model.upper():  # e.g. CPCM(Water)
                    solvent_name = solvent_model.split("(")[-1].split(")")[0]
                    input_block += f'  solvent "{solvent_name}"\n'
                input_block += "end\n\n"

        charge = Chem.GetFormalCharge(mol)
        # Determine multiplicity (assuming singlet for closed-shell PFAS, can be parameterized)
        num_radical_electrons = 0
        for atom in mol.GetAtoms():
            num_radical_electrons += atom.GetNumRadicalElectrons()
        multiplicity = num_radical_electrons + 1

        input_block += f"* xyz {charge} {multiplicity}\n"
        conformer = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            input_block += f"  {atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n"
        input_block += "*\n"

        with open(input_file_path, "w") as f:
            f.write(f"# ORCA Input for {molecule_id}\n")
            f.write("# Generated by MoML-CA ORCA Wrapper for PFAS Simulation\n")
            f.write(f"# Functional: {functional}, Basis: {basis_set}, Solvent: {solvent_model}\n\n")
            f.write(input_block)

        # Save a MOL file for reference and easy visualization
        mol_file_path = os.path.join(output_dir, f"{molecule_id}.mol")
        Chem.MolToMolFile(mol, mol_file_path)

        logger.info(f"Generated ORCA input for {molecule_id} at {input_file_path}")
        return True, input_file_path

    except Exception as e:
        logger.error(f"Error generating ORCA input for {molecule_id}: {e}")
        return False, ""


def run_orca_calculation_mock(input_file_path: str) -> Tuple[bool, str]:
    """Mock function for ORCA calculation."""
    logger.info(f"Mocking ORCA calculation for {input_file_path}")
    return True, ""


def run_orca_calculation(
    input_file_path: str, orca_executable: Optional[str], openmpi_bin_path: Optional[str], mock_run: bool
) -> Tuple[bool, str]:
    """
    Runs a single ORCA calculation, either for real or in mock mode.

    Handles locating the ORCA executable and setting up the environment
    for parallel execution if OpenMPI path is provided.

    Args:
        input_file_path: Path to the ORCA input file (.inp).
        orca_executable: Full path to the ORCA executable. If None, attempts to find it.
        openmpi_bin_path: Path to OpenMPI binaries (optional, for parallel runs).
        mock_run: If True, simulates the ORCA run without actual execution.

    Returns:
        A tuple (success, output_file_path):
        - success (bool): True if ORCA ran (or was mocked) successfully, False otherwise.
        - output_file_path (str): Path to the ORCA output file (.out).
    """
    input_dir = os.path.dirname(input_file_path)
    molecule_name = os.path.basename(input_file_path).replace(".inp", "")
    output_file_path = os.path.join(input_dir, f"{molecule_name}.out")

    if mock_run:
        logger.info(f"Mock mode: Simulating ORCA calculation for {molecule_name}")
        with open(output_file_path, "w") as f:
            f.write(f"ORCA Mockup Output for {molecule_name}\n")
            f.write("---------------------------------------------\n")
            f.write("Calculation Type: Mocked for PFAS\n")
            f.write("FINAL SINGLE POINT ENERGY     -123.456789 Eh\n")  # Mock energy
            f.write("ORCA TERMINATED NORMALLY\n")
        time.sleep(0.1)  # Simulate some calculation time
        return True, output_file_path

    # Find ORCA executable if not provided
    if orca_executable is None:
        common_paths = ["orca", "/opt/orca/orca", os.path.expanduser("~/orca/orca")]  # Add more if needed
        for path_option in common_paths:
            try:
                # Check ORCA version via stdout and stderr
                result = subprocess.run([path_option, "--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = (result.stdout or "") + (result.stderr or "")
                if "Program Version ORCA" in output:
                    orca_executable = path_option
                    logger.info(f"Using ORCA executable found at: {orca_executable}")
                    break
                else:
                    logger.warning(f"Version check for {path_option} did not confirm ORCA.")
            except FileNotFoundError:
                continue
        if orca_executable is None:
            logger.warning("ORCA executable not found; switching to mock mode.")
            mock_run = True
            # Early return using the mock_run path to avoid using None as orca_executable
            return run_orca_calculation_mock(input_file_path)

    env = os.environ.copy()
    if openmpi_bin_path:
        env["PATH"] = f"{openmpi_bin_path}:{env.get('PATH', '')}"
        logger.info(f"Using OpenMPI from: {openmpi_bin_path} for parallel ORCA run.")

    command = [orca_executable, input_file_path]
    logger.info(f"Running ORCA for {molecule_name}: {' '.join(command)} in {input_dir}")

    try:
        # ORCA writes its main output to stdout, which we redirect to the .out file
        with open(output_file_path, "w") as outfile, tempfile.NamedTemporaryFile(mode='w+', delete=False) as errfile:
            process = subprocess.run(
                command, env=env, cwd=input_dir, stdout=outfile, stderr=errfile, text=True, check=False
            )

            # Read stderr from temp file while it's still open
            errfile.seek(0)
            stderr_content = errfile.read()

        # Close and remove the temporary file outside the with block
        os.unlink(errfile.name)

        if process.returncode != 0:
            logger.error(f"ORCA calculation for {molecule_name} failed with return code {process.returncode}.")
            logger.error(f"ORCA STDERR:\n{stderr_content}")
            # Even if it fails, the output file might contain useful info for debugging
            if os.path.exists(output_file_path):
                logger.warning(f"Partial output file may exist at {output_file_path}")
            return False, output_file_path  # Return path for inspection

        # Verify normal termination in the output file
        with open(output_file_path, "r") as f_out:
            content = f_out.read()
            if "ORCA TERMINATED NORMALLY" not in content:
                logger.error(
                    f"ORCA calculation for {molecule_name} did not terminate normally. Check {output_file_path}."
                )
                logger.error(f"ORCA STDERR:\n{stderr_content}")
                return False, output_file_path

        logger.info(f"ORCA calculation completed successfully for {molecule_name}. Output: {output_file_path}")
        return True, output_file_path

    except FileNotFoundError:
        logger.error(f"ORCA executable not found at {orca_executable}. Please check the path.")
        return False, ""
    except Exception as e:
        logger.error(f"An unexpected error occurred while running ORCA for {molecule_name}: {e}")
        return False, ""


def process_molecule(
    smiles: str,
    molecule_id: str,
    base_output_dir: str,
    functional: str,
    basis_set: str,
    num_procs: int,
    memory: int,
    orca_path: Optional[str],
    openmpi_path: Optional[str],
    optimize_geom: bool,
    calculate_charges: bool,
    sp_only: bool,
    mock_run: bool,
    solvent_model: Optional[str],
    conversion_script_path: str,
) -> Dict[str, Any]:
    """
    Processes a single PFAS molecule through the entire ORCA workflow.

    This includes 3D structure generation, ORCA input file creation,
    ORCA calculation execution, and orchestrating the conversion of
    results to NPZ format.

    Args:
        smiles: SMILES string of the PFAS molecule.
        molecule_id: Unique identifier for the molecule.
        base_output_dir: Base directory where results for all molecules are stored.
                         A subdirectory for this specific molecule will be created here.
        functional: Density functional for ORCA.
        basis_set: Basis set for ORCA.
        num_procs: Number of processors for ORCA.
        memory: Memory (MB) for ORCA.
        orca_path: Path to ORCA executable.
        openmpi_path: Path to OpenMPI binaries.
        optimize_geom: Flag to perform geometry optimization.
        calculate_charges: Flag to calculate partial charges.
        sp_only: Flag for single-point calculation only.
        mock_run: Flag to run in mock mode.
        solvent_model: Implicit solvent model for ORCA.
        conversion_script_path: Path to the script for ORCA output to NPZ conversion.


    Returns:
        A dictionary containing the processing status and paths to key output files.
        Keys: "molecule_id", "smiles", "success", "npz_file" (if successful),
        "error" (if failed).
    """
    # Sanitize molecule_id to be filesystem-friendly
    clean_molecule_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(molecule_id))
    molecule_specific_dir = os.path.join(base_output_dir, clean_molecule_id)
    os.makedirs(molecule_specific_dir, exist_ok=True)

    logger.info(f"Starting processing for molecule: {molecule_id} (SMILES: {smiles})")
    result_summary: Dict[str, Any] = {"molecule_id": molecule_id, "smiles": smiles, "success": False}

    # Step 1: Generate 3D structure
    mol_3d = generate_3d_structure(smiles, molecule_id)
    if mol_3d is None:
        result_summary["error"] = "Failed to generate 3D structure."
        logger.error(f"Failed 3D structure generation for {molecule_id}.")
        return result_summary

    # Step 2: Generate ORCA input file
    input_generated, orca_input_file = generate_orca_input(
        mol_3d,
        clean_molecule_id,
        molecule_specific_dir,
        functional,
        basis_set,
        num_procs,
        memory,
        optimize_geom,
        calculate_charges,
        sp_only,
        solvent_model,
    )
    if not input_generated:
        result_summary["error"] = "Failed to generate ORCA input file."
        logger.error(f"Failed ORCA input generation for {molecule_id}.")
        return result_summary

    # Step 3: Run ORCA calculation
    calc_success, orca_output_file = run_orca_calculation(orca_input_file, orca_path, openmpi_path, mock_run)
    if not calc_success:
        result_summary["error"] = "ORCA calculation failed or did not terminate normally."
        logger.error(f"ORCA calculation failed for {molecule_id}. Output at: {orca_output_file}")
        # Even if failed, provide the output file path for debugging
        result_summary["orca_output_file"] = orca_output_file
        return result_summary
    result_summary["orca_output_file"] = orca_output_file

    # Step 4: Convert ORCA output to QM9-style NPZ using the external script
    # This step assumes the orca_json_to_qm9_npz.py script handles parsing and NPZ creation.
    npz_file_path = os.path.join(molecule_specific_dir, f"{clean_molecule_id}_qm9.npz")
    conversion_command = [
        sys.executable,
        conversion_script_path,
        orca_output_file,
        "-o",
        npz_file_path,
        "--molecule_id",
        str(molecule_id),  # Pass original molecule_id
        "--smiles",
        smiles,
    ]

    logger.info(f"Converting ORCA output to NPZ for {molecule_id}: {' '.join(conversion_command)}")
    try:
        conversion_process = subprocess.run(
            conversion_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=base_output_dir,  # Run from base_output_dir if script uses relative paths
        )
        logger.info(f"Successfully converted ORCA output to NPZ for {molecule_id}: {npz_file_path}")
        logger.debug(f"Conversion script STDOUT for {molecule_id}:\n{conversion_process.stdout}")
        if conversion_process.stderr:
            logger.warning(f"Conversion script STDERR for {molecule_id}:\n{conversion_process.stderr}")
        result_summary["npz_file"] = npz_file_path
        result_summary["success"] = True
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Error converting ORCA output to NPZ for {molecule_id}.\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return Code: {e.returncode}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}"
        )
        result_summary["error"] = error_message
        logger.error(error_message)
        return result_summary
    except FileNotFoundError:
        error_message = f"Conversion script not found at {conversion_script_path} for {molecule_id}."
        result_summary["error"] = error_message
        logger.error(error_message)
        return result_summary

    logger.info(f"Successfully completed processing for molecule: {molecule_id}")
    return result_summary


def process_molecules_parallel(
    df_molecules: pd.DataFrame,
    output_dir_base: str,
    cli_args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Processes multiple PFAS molecules in parallel using a process pool.

    Each molecule is processed by the `process_molecule` function.

    Args:
        df_molecules: DataFrame containing molecule information (SMILES, ID).
        output_dir_base: Base directory for all output files.
        cli_args: Parsed command-line arguments containing all necessary parameters.

    Returns:
        A list of dictionaries, where each dictionary is the result from
        `process_molecule` for a single PFAS compound.
    """
    num_molecules = len(df_molecules)
    logger.info(f"Starting parallel processing of {num_molecules} molecules using up to {cli_args.max_jobs} worker(s).")
    all_results: List[Dict[str, Any]] = []

    # Determine if geometry optimization should be performed
    # If sp_only is true, optimize is false, unless optimize is also explicitly true.
    optimize_geometry = not cli_args.sp_only
    if cli_args.optimize:  # If --optimize is explicitly given, it takes precedence
        optimize_geometry = True

    with concurrent.futures.ProcessPoolExecutor(max_workers=cli_args.max_jobs) as executor:
        future_to_mol_id: Dict[concurrent.futures.Future, str] = {}
        for _, row_data in df_molecules.iterrows():
            smiles_str = row_data[cli_args.smiles_col]
            mol_identifier = str(row_data[cli_args.id_col])

            future = executor.submit(
                process_molecule,
                smiles_str,
                mol_identifier,
                output_dir_base,
                cli_args.functional,
                cli_args.basis_set,
                cli_args.num_procs,
                cli_args.memory,
                cli_args.orca_path,
                cli_args.openmpi_path,
                optimize_geometry,
                cli_args.charges,
                cli_args.sp_only,
                cli_args.mock,
                cli_args.solvent_model,
                cli_args.conversion_script_path,
            )
            future_to_mol_id[future] = mol_identifier

        for future_item in concurrent.futures.as_completed(future_to_mol_id):
            mol_id_completed = future_to_mol_id[future_item]
            try:
                mol_result = future_item.result()
                all_results.append(mol_result)
                logger.info(f"Finished processing for {mol_id_completed}. Success: {mol_result.get('success', False)}")
            except Exception as exc:
                logger.error(f"Molecule {mol_id_completed} generated an exception during parallel processing: {exc}")
                all_results.append(
                    {
                        "molecule_id": mol_id_completed,
                        "smiles": df_molecules[df_molecules[cli_args.id_col].astype(str) == mol_id_completed][
                            cli_args.smiles_col
                        ].iloc[0],
                        "success": False,
                        "error": f"Unhandled exception in worker: {exc}",
                    }
                )
    return all_results


def main() -> None:
    """
    Main execution function for the ORCA PFAS wrapper.

    Parses arguments, loads input data, processes molecules (potentially in
    parallel), and saves a summary of the results.
    """
    args = parse_args()

    logger.info("Initializing MoML-CA ORCA PFAS Wrapper.")
    logger.info(f"Full configuration: {vars(args)}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        logger.info(f"Loading PFAS data from input CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        logger.info(f"Successfully loaded {len(df)} compounds from CSV.")
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {args.input_csv}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input CSV data: {e}")
        sys.exit(1)

    if args.smiles_col not in df.columns:
        logger.error(
            f"SMILES column '{args.smiles_col}' not found in the input CSV. Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)
    if args.id_col not in df.columns:
        logger.warning(f"ID column '{args.id_col}' not found. Using DataFrame index as molecule identifier.")
        df[args.id_col] = df.index.astype(str)

    # Check if conversion script exists
    if not os.path.isfile(args.conversion_script_path):
        logger.error(f"ORCA to NPZ conversion script not found at: {args.conversion_script_path}")
        logger.error("Please ensure the script exists and the path is correct via --conversion_script_path.")
        sys.exit(1)

    start_time = time.time()
    processed_results = process_molecules_parallel(df, args.output_dir, args)
    end_time = time.time()

    # Prepare and save results summary
    summary_data = []
    for res in processed_results:
        summary_entry = {
            "molecule_id": res.get("molecule_id"),
            "smiles": res.get("smiles"),
            "calculation_success": res.get("success", False),
            "npz_file_path": res.get("npz_file"),
            "orca_output_file": res.get("orca_output_file"),
            "error_message": res.get("error"),
            "functional": args.functional,
            "basis_set": args.basis_set,
            "solvent_model": args.solvent_model,
        }
        summary_data.append(summary_entry)

    results_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(args.output_dir, "orca_pfas_wrapper_summary.csv")
    try:
        results_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Processing summary saved to: {summary_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")

    successful_runs = sum(1 for r in processed_results if r.get("success"))
    total_runs = len(processed_results)
    logger.info(f"Total molecules processed: {total_runs}")
    logger.info(f"Successful calculations: {successful_runs}")
    if total_runs > 0:
        success_rate = (successful_runs / total_runs) * 100
        logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds.")
    logger.info("ORCA PFAS Wrapper execution finished.")


if __name__ == "__main__":
    main()
