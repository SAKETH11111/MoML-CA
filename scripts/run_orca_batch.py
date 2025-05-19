"""
Automates ORCA quantum mechanical calculations for a list of SDF files.

This script processes a list of molecules, generates ORCA input files,
runs ORCA calculations, and stores the results.
"""
import os
import subprocess
import logging
import argparse
from typing import List, Optional, Tuple
from pathlib import Path # Added for Path operations
import glob # Added for file globbing

from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Default list of SDF files to process
DEFAULT_SDF_FILES: List[str] = [
    "data/diverse_pfas_sdf_batch/DTXSID8031865.sdf", # PFOA
    "data/diverse_pfas_sdf_batch/DTXSID3031864.sdf", # PFOS
    "data/diverse_pfas_sdf_batch/DTXSID8031863.sdf", # PFNA
    "data/diverse_pfas_sdf_batch/DTXSID7040150.sdf", # PFHxS
    "data/diverse_pfas_sdf_batch/DTXSID3031862.sdf", # PFHxA
    "data/diverse_pfas_sdf_batch/DTXSID5030030.sdf", # PFBS
    "data/diverse_pfas_sdf_batch/DTXSID4059916.sdf", # PFBA
    "data/diverse_pfas_sdf_batch/DTXSID70880215.sdf", # GenX
    "data/diverse_pfas_sdf_batch/DTXSID5044572.sdf", # 6:2 FTOH
    "data/diverse_pfas_sdf_batch/DTXSID7029904.sdf", # 8:2 FTOH
]

ORCA_OUTPUT_DIR: str = "genx_orca_results_diverse_batch" # New output directory for this batch
ORCA_KEYWORDS: str = "! HF 6-31G Opt CHELPG" # Using HF/6-31G as per original task context


def get_orca_executable(args_path: Optional[str]) -> Optional[str]:
    """
    Determines the ORCA executable path.

    Priority:
    1. Command-line argument (--orca_path)
    2. Environment variable (ORCA_PATH)

    Args:
        args_path: Path from command-line arguments.

    Returns:
        The path to the ORCA executable, or None if not found.
    """
    if args_path:
        if os.path.isfile(args_path):
            logging.info(f"Using ORCA executable from command line: {args_path}")
            return args_path
        else:
            logging.warning(
                f"ORCA path from command line not found: {args_path}. Checking environment variable."
            )

    env_path = os.environ.get("ORCA_PATH")
    if env_path:
        if os.path.isfile(env_path):
            logging.info(f"Using ORCA executable from ORCA_PATH environment variable: {env_path}")
            return env_path
        else:
            logging.warning(
                f"ORCA path from ORCA_PATH environment variable not found: {env_path}."
            )
    
    logging.error(
        "ORCA executable not found. Please specify via --orca_path or ORCA_PATH environment variable."
    )
    return None


def get_molecule_charge_multiplicity(mol: Chem.Mol) -> Tuple[int, int]:
    """
    Calculates the total formal charge of the molecule and assumes multiplicity 1.

    Args:
        mol: RDKit molecule object.

    Returns:
        A tuple (charge, multiplicity).
    """
    charge = Chem.GetFormalCharge(mol)
    multiplicity = 1  # Assuming singlet for closed-shell neutral molecules
    # For radicals or charged species, multiplicity might need adjustment.
    # For HF, multiplicity is 2S+1. For a singlet, S=0, mult=1. For a doublet, S=1/2, mult=2.
    # If the number of electrons is odd, multiplicity must be at least 2.
    num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge
    if num_electrons % 2 != 0: # Odd number of electrons
        multiplicity = 2 # Doublet
        logging.info(f"Molecule has an odd number of electrons ({num_electrons}). Setting multiplicity to 2 (doublet).")
    else:
        logging.info(f"Molecule has an even number of electrons ({num_electrons}). Setting multiplicity to 1 (singlet).")

    return charge, multiplicity


def generate_orca_input_content(
    mol: Chem.Mol, charge: int, multiplicity: int, num_cores: int
) -> Optional[str]:
    """
    Generates the content for an ORCA input file.

    Args:
        mol: RDKit molecule object with 3D coordinates.
        charge: The total charge of the molecule.
        multiplicity: The spin multiplicity of the molecule.
        num_cores: Number of processor cores to use for the calculation.

    Returns:
        A string containing the ORCA input file content, or None if error.
    """
    try:
        conformer = mol.GetConformer()
    except ValueError:
        logging.error("Molecule does not have a 3D conformer.")
        return None

    atom_lines = []
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        atom_lines.append(f"  {atom.GetSymbol():<2} {pos.x:12.8f} {pos.y:12.8f} {pos.z:12.8f}")

    coordinates_block = "\n".join(atom_lines)

    pal_block = ""
    if num_cores > 0: # Only add PAL block if num_cores is specified and > 0
        pal_block = f"""%pal
  nprocs {num_cores}
end
"""
    return f"""{ORCA_KEYWORDS}
{pal_block}
* xyz {charge} {multiplicity}
{coordinates_block}
*
"""

def cleanup_previous_orca_files(base_name: str, directory: str) -> None:
    """
    Deletes ORCA output files from previous runs for a given molecule.
    Common ORCA extensions: .out, .err, .gbw, .prop, .xyz (structure), .hess, .pc_chelpg, etc.
    Also removes .tmp files that ORCA might leave.
    """
    logging.info(f"Cleaning up previous ORCA files for {base_name} in {directory}...")
    count_deleted = 0
    # More comprehensive list of extensions ORCA might produce for a given base_name
    # This doesn't include files that ORCA puts in subdirectories (like some MD files)
    # but should cover typical single point/opt/freq outputs.
    orca_extensions_to_clean = [
        ".out", ".err", ".gbw", ".inp", ".prop", ".xyz", ".hess", ".pc_chelpg",
        ".opt", ".cis", ".mp2", ".scfp", ".scfgrad", ".densities", ".dip", ".soc",
        ".trj", ".md", ".energies", ".ges", ".vpot", ".xtbopt.log", ".omd",
        ".oof", ".ocosmo", ".smd", ".cpcm", ".engrad", ".hessian", ".freq",
        ".raman", ".uvspec", ".cdspec", ".xray", ".epr", ".nmr", ".mdci",
        ".loc", ".nbo", ".aim", ".elf", ".mep", ".dens", ".spin", ".mdrestart",
        ".chk", ".tmp", ".engrad", ".property.txt", ".citations.tmp", ".ginp.tmp",
        ".gu.tmp", ".int.tmp", ".SHARKINP.tmp", ".bas0", ".bas1", ".bas2", ".bas3",
        ".bas4", ".bas5", ".en.tmp" # Added from observed files
    ]
    for ext in orca_extensions_to_clean:
        # Using Path.glob for safer path construction and matching
        for f_path in Path(directory).glob(f"{base_name}{ext}"):
            try:
                f_path.unlink()
                logging.debug(f"Deleted old file: {f_path}")
                count_deleted +=1
            except OSError as e:
                logging.warning(f"Could not delete old file {f_path}: {e}")
    if count_deleted > 0:
        logging.info(f"Cleaned up {count_deleted} old ORCA files for {base_name}.")
    else:
        logging.info(f"No old ORCA files found for {base_name} to clean up.")


def run_orca_calculation(
    orca_executable: str, input_file_path_str: str, output_dir_str: str
) -> bool:
    """
    Runs an ORCA calculation using subprocess, redirecting stdout/stderr to files.

    Args:
        orca_executable: Path to the ORCA executable.
        input_file_path_str: Path to the ORCA input file (as string).
        output_dir_str: Directory where ORCA should run and write output files (as string).

    Returns:
        True if ORCA run completed successfully (exit code 0), False otherwise.
    """
    input_file_path = Path(input_file_path_str)
    output_dir = Path(output_dir_str)

    input_file_basename = input_file_path.name
    base_name_for_output = input_file_path.stem # e.g., "DTXSID0059794" from "DTXSID0059794.inp"
    
    output_file_path = output_dir / f"{base_name_for_output}.out"
    error_file_path = output_dir / f"{base_name_for_output}.err"

    logging.info(f"Starting ORCA calculation for {input_file_basename} in {output_dir}")
    logging.info(f"  Input file: {input_file_path}")
    logging.info(f"  Output file will be: {output_file_path}")
    logging.info(f"  Error file will be: {error_file_path}")
    
    command = [orca_executable, input_file_basename] # ORCA expects just the filename if cwd is set
    logging.info(f"Executing ORCA command: {' '.join(command)} in directory {output_dir}")

    try:
        with open(output_file_path, "w") as f_stdout, open(error_file_path, "w") as f_stderr:
            process = subprocess.run(
                command,
                cwd=output_dir, # Run ORCA from the output directory
                stdout=f_stdout,
                stderr=f_stderr,
                text=True, # Ensure text mode for stdout/stderr
                check=False, # Don't raise exception for non-zero exit codes immediately
                timeout=7200  # Timeout after 2 hours, adjust as needed
            )
        
        # Files are written directly by subprocess.
        logging.info(f"ORCA stdout saved to {output_file_path}")
        # stderr is always created, check if it's empty for success indication
        
        if process.returncode == 0:
            logging.info(f"ORCA run completed successfully for {input_file_basename} (exit code 0).")
            if error_file_path.exists() and error_file_path.stat().st_size == 0:
                logging.info(f"Empty error file {error_file_path} generated by successful run, removing it.")
                error_file_path.unlink()
            elif error_file_path.exists():
                 logging.info(f"ORCA stderr (though successful run, not empty) saved to {error_file_path}")
            return True
        else:
            logging.error(f"ORCA run failed for {input_file_basename} with exit code {process.returncode}.")
            logging.error(f"ORCA stderr for {input_file_basename} saved to {error_file_path}")
            # Optionally, log snippets if needed, but files are saved.
            try:
                with open(output_file_path, "r") as f_out_read:
                    logging.error(f"Captured STDOUT (see {output_file_path}):\n{f_out_read.read(1000)}...")
                # error_file_path is already written, so just refer to it.
                with open(error_file_path, "r") as f_err_read:
                    content = f_err_read.read(1000)
                    if content:
                        logging.error(f"Captured STDERR (see {error_file_path}):\n{content}...")
                    else:
                        logging.error(f"Captured STDERR (see {error_file_path}): [EMPTY, but exit code was non-zero]")
            except Exception as e_read:
                logging.warning(f"Could not read output/error files for logging snippet: {e_read}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"ORCA run for {input_file_basename} timed out after specified duration.")
        logging.error(f"Partial ORCA stdout (if any) saved to {output_file_path}")
        logging.error(f"Partial ORCA stderr (if any) saved to {error_file_path}")
        return False
    except FileNotFoundError:
        logging.error(
            f"ORCA executable not found at {orca_executable}. Ensure it's correctly specified and in PATH or use full path."
        )
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running ORCA for {input_file_basename}: {e}")
        logging.error(f"Check {output_file_path} and {error_file_path} for any partial output.")
        return False


def process_sdf_file(
    sdf_file_path: str, orca_executable: str, output_dir_str: str, num_cores: int
) -> bool:
    """
    Processes a single SDF file: reads molecule, generates ORCA input, runs ORCA.
    Includes cleanup of previous output files.

    Args:
        sdf_file_path: Path to the SDF file.
        orca_executable: Path to the ORCA executable.
        output_dir_str: Directory to save ORCA input and output files.
        num_cores: Number of cores for ORCA calculation.

    Returns:
        True if processing was successful, False otherwise.
    """
    logging.info(f"Processing {sdf_file_path}...")
    output_dir = Path(output_dir_str) # Ensure output_dir is a Path object
    base_name = Path(sdf_file_path).stem # e.g., "DTXSID0059794" from "data/.../DTXSID0059794.sdf"
    
    # Cleanup previous files for this molecule
    cleanup_previous_orca_files(base_name, str(output_dir))

    orca_input_file_path = output_dir / f"{base_name}.inp"

    if not Path(sdf_file_path).exists():
        logging.error(f"SDF file not found: {sdf_file_path}")
        return False

    mol = Chem.MolFromMolFile(sdf_file_path, removeHs=False)
    if mol is None:
        logging.error(f"Could not read molecule from {sdf_file_path}")
        return False
    
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        logging.info(f"No explicit hydrogens found in {base_name}. Adding them.")
        mol = Chem.AddHs(mol, addCoords=True)

    if mol.GetNumConformers() == 0:
        logging.info(f"No 3D conformer in {base_name}. Attempting to generate one.")
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1: # EmbedMolecule returns -1 on failure
            logging.error(f"Failed to embed molecule {base_name} to generate 3D conformer.")
            return False
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            logging.warning(f"UFF optimization failed for {base_name}: {e}. Using embedded conformer.")

    if mol.GetNumConformers() == 0: # Double check after embedding attempt
        logging.error(f"Still no 3D conformer for {base_name} after generation attempt.")
        return False

    charge, multiplicity = get_molecule_charge_multiplicity(mol)
    logging.info(f"Molecule {base_name}: Charge={charge}, Multiplicity={multiplicity}")

    orca_input_content = generate_orca_input_content(mol, charge, multiplicity, num_cores)
    if not orca_input_content:
        logging.error(f"Failed to generate ORCA input for {base_name}")
        return False

    try:
        with open(orca_input_file_path, "w") as f:
            f.write(orca_input_content)
        logging.info(f"Generated ORCA input file: {orca_input_file_path}")
    except IOError as e:
        logging.error(f"Failed to write ORCA input file {orca_input_file_path}: {e}")
        return False

    return run_orca_calculation(orca_executable, str(orca_input_file_path), str(output_dir))


def main():
    """
    Main function to parse arguments and run the ORCA batch process.
    """
    parser = argparse.ArgumentParser(
        description="Run ORCA calculations for a list of SDF files."
    )
    parser.add_argument(
        "--sdf_files",
        nargs="+",
        default=DEFAULT_SDF_FILES,
        help="List of SDF files to process. Defaults to a predefined list.",
    )
    parser.add_argument(
        "--orca_path",
        type=str,
        default=os.environ.get("ORCA_PATH"), # Check env var first
        help="Path to the ORCA executable. Overrides ORCA_PATH environment variable. If not set, script will try to find a default.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=ORCA_OUTPUT_DIR,
        help=f"Directory to store ORCA input and output files. Default: {ORCA_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=1, # Default to 1 core if not specified
        help="Number of processor cores for ORCA to use. Default: 1.",
    )
    args = parser.parse_args()

    # Refined orca_executable retrieval
    orca_executable = args.orca_path
    if not orca_executable: # If --orca_path not given and ORCA_PATH env var was not set or empty
        # Attempt to use a common default if not found by other means (e.g. from previous successful run)
        # This is a fallback, ideally user provides it or sets ORCA_PATH
        common_orca_path = "/Users/saketh/Library/orca_6_0_1/orca"
        if Path(common_orca_path).is_file():
            logging.info(f"Using common default ORCA executable: {common_orca_path}")
            orca_executable = common_orca_path
        else:
            logging.error(
                "ORCA executable not found. Please specify via --orca_path, ORCA_PATH environment variable, or ensure it's at a known default location."
            )
            return
    elif not Path(orca_executable).is_file():
        logging.error(
            f"Specified ORCA executable not found at {orca_executable}. "
            "Please check the path."
        )
        return
    else:
        logging.info(f"Using ORCA executable: {orca_executable}")


    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"ORCA output will be saved in: {output_dir_path.resolve()}")

    successful_runs = 0
    failed_runs = 0

    for sdf_file in args.sdf_files:
        if process_sdf_file(sdf_file, orca_executable, str(output_dir_path), args.num_cores):
            successful_runs += 1
        else:
            failed_runs += 1
            logging.info(f"Continuing to next molecule after failure with {Path(sdf_file).name}.")
        logging.info("-" * 50)

    logging.info("Batch processing finished.")
    logging.info(f"Successful ORCA runs: {successful_runs}")
    logging.info(f"Failed ORCA runs: {failed_runs}")


if __name__ == "__main__":
    main()