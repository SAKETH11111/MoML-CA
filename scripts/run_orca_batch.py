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
from pathlib import Path
import re # Added
import csv # Added

from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s", # Updated format
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__) # Use a named logger

# Updated CONFIG
ORCA_KEYWORDS = "! B3LYP STO-3G Opt CHELPG" # Added Opt and CHELPG back

ORCA_OUTPUT_BLOCK = """\
%output
   Print[P_Basis] 2
   Print[P_Mulliken] 1
   Print[P_Hirshfeld] 1
end
""" # Removed Print[P_ESP] 1

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
ORCA_OUTPUT_DIR: str = "orca_results_b3lyp_sto3g" # Updated
DATASET_CSV: str = "qm_dataset.csv" # Added


def get_orca_executable(args_path: Optional[str], default_exe_path: str = "/opt/orca/orca") -> Optional[str]: # Added default_exe_path
    """
    Determines the path to the ORCA executable based on command-line argument, environment variable, or a default location.
    
    Checks for the executable in the following order: command-line argument (`args_path`), `ORCA_PATH` environment variable, and a default path. Returns the path if found, or None if not found.
    """
    if args_path:
        if os.path.isfile(args_path):
            logger.info(f"Using ORCA executable from command line: {args_path}")
            return args_path
        else:
            logging.warning(
                f"ORCA path from command line not found: {args_path}. Checking environment variable."
            )

    env_path = os.environ.get("ORCA_PATH")
    if env_path:
        if Path(env_path).is_file(): # Use Path
            logger.info(f"Using ORCA executable from ORCA_PATH environment variable: {env_path}")
            return env_path
        else:
            logger.warning(
                f"ORCA path from ORCA_PATH environment variable not found: {env_path}."
            )
    
    # Fallback to a common default if not found by other means
    if Path(default_exe_path).is_file():
        logger.info(f"Using common default ORCA executable: {default_exe_path}")
        return default_exe_path

    logger.error(
        "ORCA executable not found. Please specify via --orca_path, ORCA_PATH environment variable, or ensure it's at a known default location."
    )
    return None


def get_molecule_charge_multiplicity(mol: Chem.Mol) -> Tuple[int, int]: # Renamed from get_molecule_charge_mult
    """
    Calculates the formal charge and spin multiplicity of a molecule.
    
    The spin multiplicity is set to 2 if the total number of electrons is odd, otherwise 1.
    
    Args:
    	mol: RDKit molecule object.
    
    Returns:
    	A tuple containing the formal charge and spin multiplicity.
    """
    q = Chem.GetFormalCharge(mol)
    electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - q
    mult = 2 if electrons % 2 else 1 # More direct from new script
    logger.info(f"Molecule: Charge={q}, Electrons={electrons}, Multiplicity={mult}")
    return q, mult


def generate_orca_input_content( # Renamed from generate_orca_input
    mol: Chem.Mol, charge: int, mult: int, ncores: int # Renamed multiplicity to mult
) -> Optional[str]:
    """
    Generates the content for an ORCA input file for a given molecule.
    
    Includes calculation keywords, optional parallelization settings, output block, and the molecule's 3D coordinates in XYZ format. Returns None if the molecule lacks a 3D conformer.
    """
    try:
        conformer = mol.GetConformer()
    except ValueError:
        logger.error("Molecule does not have a 3D conformer.")
        return None

    xyz_coords = "\n".join( # Renamed from atom_lines and coordinates_block
        f"  {a.GetSymbol():<2} {conformer.GetAtomPosition(i).x:12.8f}"
        f" {conformer.GetAtomPosition(i).y:12.8f} {conformer.GetAtomPosition(i).z:12.8f}"
        for i, a in enumerate(mol.GetAtoms())
    )

    pal_block = f"%pal nprocs {ncores}\nend\n" if ncores > 0 else "" # Ensure newline if block exists, 0 means no block
    
    # Using new ORCA_KEYWORDS and ORCA_OUTPUT_BLOCK
    return (
        f"{ORCA_KEYWORDS}\n{pal_block}{ORCA_OUTPUT_BLOCK}\n"
        f"* xyz {charge} {mult}\n{xyz_coords}\n*\n"
    )

# ────────────────────────── ORCA OUTPUT PARSING (New) ───────────────────────────
_re_energy = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_re_dipole = re.compile(r"Total Dipole Moment.*?\n\s+X\s+Y\s+Z.*?\n"
                        r".*?\n\s+Dipole Magnitude\s+\(Debye\)\s+(-?\d+\.\d+)", re.S) # Corrected to capture Debye value
_re_chelpg = re.compile(r"CHELPG Charges.*?\n(.*?)\n\n", re.S)

def parse_orca_out(out_path: Path) -> Optional[dict]:
    """
    Parses an ORCA output file to extract the final single point energy, total dipole moment magnitude, and CHELPG atomic charges.
    
    Returns:
        A dictionary containing the molecule ID, energy in Hartree, dipole moment in Debye, and per-atom CHELPG charges (as `q_atom_1`, `q_atom_2`, ...), or None if any essential data is missing or cannot be parsed.
    """
    if not out_path.exists():
        logger.error(f"ORCA output file not found for parsing: {out_path}")
        return None
    text = out_path.read_text(errors="ignore")

    mE = _re_energy.search(text)
    mD = _re_dipole.search(text)
    mQ = _re_chelpg.search(text)

    if not (mE and mD and mQ):
        logging.warning(f"Couldn't parse all required data (Energy, Dipole, CHELPG) from {out_path.name}")
        # Log which specific parts are missing for better debugging
        if not mE: logging.warning(f"  Missing: FINAL SINGLE POINT ENERGY in {out_path.name}")
        if not mD: logging.warning(f"  Missing: Total Dipole Moment (Debye) in {out_path.name}")
        if not mQ: logging.warning(f"  Missing: CHELPG Charges block in {out_path.name}")
        return None

    charges = []
    for line in mQ.group(1).strip().splitlines():
        parts = line.split()
        # Expecting lines like: "0   O :    -0.616322" or "0 C   -0.123456"
        if len(parts) >= 3 and parts[0].isdigit(): # Check if first part is a digit (atom index)
            try:
                charges.append(float(parts[-1])) # Charge is the last part
            except ValueError:
                logger.warning(f"Could not parse charge from CHELPG line: '{line}' in {out_path.name}")
                continue # Skip this line if charge parsing fails

    if not charges:
        logging.warning(f"No CHELPG charges were successfully parsed from {out_path.name}")
        # We might still want to return other data if charges are missing but energy/dipole are present
        # For now, let's make CHELPG charges essential for a "complete" parse.
        return None


    return {
        "molid": out_path.stem,
        "energy_hartree": float(mE.group(1)),
        "dipole_D": float(mD.group(1)),
        **{f"q_atom_{i+1}": q for i, q in enumerate(charges)}, # More descriptive charge keys
    }
# ───────────────────────────────────────────────────────────────────────────────

def cleanup_previous_orca_files(base_name: str, directory: str) -> None:
    """
    Removes previous ORCA output and temporary files for a given molecule in the specified directory.
    
    Deletes files matching the molecule's base name with common ORCA-related extensions to prevent conflicts with new calculations.
    """
    logger.info(f"Cleaning up previous ORCA files for {base_name} in {directory}...")
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
        logger.info(f"Cleaned up {count_deleted} old ORCA files for {base_name}.")
    else:
        logger.info(f"No old ORCA files found for {base_name} to clean up.")


def run_orca_calculation(
    orca_executable: str, input_file_path_str: str, output_dir_str: str
) -> bool:
    """
    Runs an ORCA quantum chemistry calculation using the specified input file.
    
    Executes the ORCA program as a subprocess in the given output directory, redirecting standard output and error streams to corresponding files. Returns True if the calculation completes successfully (exit code 0), otherwise returns False.
    """
    input_file_path = Path(input_file_path_str)
    output_dir = Path(output_dir_str)

    input_file_basename = input_file_path.name
    base_name_for_output = input_file_path.stem # e.g., "DTXSID0059794" from "DTXSID0059794.inp"
    
    output_file_path = output_dir / f"{base_name_for_output}.out"
    error_file_path = output_dir / f"{base_name_for_output}.err"

    logger.info(f"Starting ORCA calculation for {input_file_basename} in {output_dir}")
    logger.info(f"  Input file: {input_file_path}")
    logger.info(f"  Output file will be: {output_file_path}")
    logger.info(f"  Error file will be: {error_file_path}")
    
    command = [orca_executable, input_file_basename] # ORCA expects just the filename if cwd is set
            logger.info(f"Executing ORCA command: {' '.join(command)} in directory {output_dir}")

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
        logger.info(f"ORCA stdout saved to {output_file_path}")
        # stderr is always created, check if it's empty for success indication
        
        if process.returncode == 0:
            logger.info(f"ORCA run completed successfully for {input_file_basename} (exit code 0).")
            if error_file_path.exists() and error_file_path.stat().st_size == 0:
                logger.info(f"Empty error file {error_file_path} generated by successful run, removing it.")
                error_file_path.unlink()
            elif error_file_path.exists():
                 logger.info(f"ORCA stderr (though successful run, not empty) saved to {error_file_path}")
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


def process_sdf_file( # Renamed from process_molecule in new script, kept existing name
    sdf_file_path: Path, # Changed to Path type
    orca_executable: str,
    output_dir: Path, # Changed to Path type
    num_cores: int
) -> Optional[dict]: # Returns dict or None
    """
    Processes a single SDF file by preparing the molecule, generating an ORCA input file, running the ORCA calculation, and parsing the output for quantum mechanical properties.
    
    Attempts to generate a 3D conformer if missing, optimizes geometry, and cleans up previous ORCA files before calculation. Returns a dictionary of parsed results if successful, or None if any step fails.
    """
    logger.info(f"Processing {sdf_file_path}...")
    base_name = sdf_file_path.stem
    
    cleanup_previous_orca_files(base_name, str(output_dir))

    orca_input_file_path = output_dir / f"{base_name}.inp"

    if not sdf_file_path.exists(): # Use Path.exists()
        logger.error(f"SDF file not found: {sdf_file_path}")
        return None

    mol = Chem.MolFromMolFile(str(sdf_file_path), removeHs=False) # Ensure str for RDKit
    if mol is None:
        logger.error(f"Could not read molecule from {sdf_file_path}")
        return None
    
    # Conformer generation logic from new script (slightly different from old)
    if mol.GetNumConformers() == 0:
        logger.info(f"No 3D conformer in {base_name}. Adding Hs and attempting to generate one.")
        mol = Chem.AddHs(mol, addCoords=True) # Add Hs before embedding if no conformer
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(randomSeed=0xF00D)) == -1: # Using ETKDGv3 from generate_3d_mol
            logger.error(f"Conformer generation failed for {base_name}")
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol) # MMFF94 optimization
        except Exception as e:
            logger.warning(f"MMFF optimization failed for {base_name}: {e}. Using embedded conformer.")
    
    if mol.GetNumConformers() == 0: # Double check
        logger.error(f"Still no 3D conformer for {base_name} after generation attempt.")
        return None

    charge, multiplicity = get_molecule_charge_multiplicity(mol)
    # logging.info for charge/mult already in get_molecule_charge_multiplicity

    orca_input_content = generate_orca_input_content(mol, charge, multiplicity, num_cores)
    if not orca_input_content:
        logger.error(f"Failed to generate ORCA input for {base_name}")
        return None

    try:
        orca_input_file_path.write_text(orca_input_content) # Use Path.write_text
        logger.info(f"Generated ORCA input file: {orca_input_file_path}")
    except IOError as e:
        logger.error(f"Failed to write ORCA input file {orca_input_file_path}: {e}")
        return None

    if not run_orca_calculation(orca_executable, str(orca_input_file_path), str(output_dir)):
        logger.error(f"ORCA calculation failed for {base_name}")
        # Attempt to parse even if ORCA failed, might get partial data or specific error messages
        # However, the new parse_orca_out expects certain fields, so it might return None anyway.
        # For now, if ORCA fails, we consider the molecule processing failed for dataset.
        return None
    
    # Parse output if ORCA run was successful
    parsed_data = parse_orca_out(output_dir / f"{base_name}.out")
    if parsed_data:
        logger.info(f"Successfully parsed ORCA output for {base_name}.")
    else:
        logger.warning(f"Failed to parse all required data from ORCA output for {base_name}.")
        # Even if parsing fails, the ORCA run itself might have produced files.
        # The function should return None if parsing is incomplete for dataset purposes.
    return parsed_data


def main():
    """
    Coordinates the batch ORCA quantum mechanical calculations for a list of SDF files and compiles the results into a CSV dataset.
    
    Parses command-line arguments for SDF file paths, ORCA executable location, output directory, and number of processor cores. For each molecule, it prepares input files, runs ORCA, parses outputs, and collects results. Successfully parsed results are appended to a CSV file, with logging of progress and summary statistics.
    """
    parser = argparse.ArgumentParser(
        description="Run ORCA calculations for a list of SDF files and assemble QM dataset."
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
        default=os.environ.get("ORCA_PATH"),
        help="Path to the ORCA executable. Overrides ORCA_PATH. Defaults to /opt/orca/orca if neither is set.",
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
        default=4, # Updated default from new script
        help="Number of processor cores for ORCA to use. Default: 4.",
    )
    args = parser.parse_args()

    orca_executable = get_orca_executable(args.orca_path, default_exe_path="/opt/orca/orca") # Pass new default
    if not orca_executable:
        return

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"ORCA output will be saved in: {output_dir_path.resolve()}")

    dataset_rows = [] # For collecting parsed data
    successful_runs = 0
    failed_runs = 0

    for sdf_file_str in args.sdf_files:
        sdf_file_path = Path(sdf_file_str) # Convert to Path object
        parsed_result = process_sdf_file(sdf_file_path, orca_executable, output_dir_path, args.num_cores)
        if parsed_result:
            dataset_rows.append(parsed_result)
            successful_runs += 1
        else:
            failed_runs += 1
            logger.info(f"Continuing to next molecule after failure or incomplete parsing for {sdf_file_path.name}.")
        logger.info("-" * 50)

    logger.info("Batch processing finished.")
    logger.info(f"Successful ORCA runs with complete parsing: {successful_runs}")
    logger.info(f"Failed ORCA runs or incomplete parsing: {failed_runs}")

    # Write / append CSV dataset
    if dataset_rows:
        dataset_file_path = Path(DATASET_CSV)
        # Determine fieldnames from all collected data to handle cases where some molecules might have more/less CHELPG charges
        all_keys = set()
        for row in dataset_rows:
            all_keys.update(row.keys())
        
        # Ensure a consistent order, e.g., molid, energy, dipole, then sorted charges
        fieldnames = ['molid', 'energy_hartree', 'dipole_D']
        charge_keys = sorted([k for k in all_keys if k.startswith('q_atom_')], key=lambda x: int(x.split('_')[-1]))
        fieldnames.extend(charge_keys)
        # Add any other keys that might have been missed (though unlikely with current parsing)
        fieldnames.extend(sorted(list(all_keys - set(fieldnames))))


        csv_exists = dataset_file_path.is_file()
        try:
            with open(dataset_file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore') # ignore charges not present for a molecule
                if not csv_exists or dataset_file_path.stat().st_size == 0 : # also check if file is empty
                    writer.writeheader()
                writer.writerows(dataset_rows)
            logger.info(f"Wrote/Appended {len(dataset_rows)} records to {DATASET_CSV}")
        except IOError as e:
            logger.error(f"Could not write to CSV file {DATASET_CSV}: {e}")
    else:
        logger.warning("No successful ORCA runs with complete parsing to write to dataset CSV.")


if __name__ == "__main__":
    main()