"""
scripts/run_orca_batch.py

This script automates the execution of quantum mechanical (QM) calculations
using the ORCA software package for a batch of molecules provided as SDF files.

It performs the following steps for each molecule:
1.  Reads an SDF file to get the molecular structure.
2.  Generates a 3D conformer if one is not present.
3.  Determines the molecule's charge and spin multiplicity.
4.  Creates an ORCA input file (`.inp`) with specified QM settings.
5.  Optionally cleans up previous calculation files.
6.  Executes the ORCA calculation as a subprocess.
7.  Parses the resulting output file (`.out`) to extract key properties like
    energy, dipole moment, and partial charges.
8.  Aggregates the results from all molecules and saves them to a CSV file.
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from rdkit import Chem
from rdkit.Chem import AllChem

# --- Configuration ---
# ORCA calculation settings. "Opt" performs a geometry optimization.
# "CHELPG" requests the calculation of CHELPG atomic charges.
ORCA_KEYWORDS = "! B3LYP STO-3G Opt CHELPG"
ORCA_OUTPUT_BLOCK = """\
%output
   Print[P_Basis] 2
   Print[P_Mulliken] 1
   Print[P_Hirshfeld] 1
end
"""
DEFAULT_ORCA_EXECUTABLE = "/opt/orca/orca"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for efficiency.
RE_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(" + r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?" + r")")
RE_DIPOLE = re.compile(
    r"Total Dipole Moment.*?\n\s+X\s+Y\s+Z.*?\n"
    r".*?\n\s+Dipole Magnitude\s+\(Debye\)\s+(" + r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?" + r")",
    re.DOTALL | re.IGNORECASE,
)
RE_CHELPG_BLOCK = re.compile(r"CHELPG Charges.*?\n(.*?)\n\n", re.DOTALL)
RE_CHARGE_LINE = re.compile(r"^\s*\d+\s+[A-Za-z]+\s*:?\s*(" + r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?" + r")")


def find_orca_executable(provided_path: Optional[str]) -> Optional[str]:
    """
    Finds a valid ORCA executable path.

    Order of precedence:
    1. Path provided via command-line argument.
    2. `ORCA_PATH` environment variable.
    3. A common default path (`/opt/orca/orca`).

    Args:
        provided_path: The path from the command-line argument, if any.

    Returns:
        The validated path to the ORCA executable, or None if not found.
    """
    paths_to_check = [
        (provided_path, "command line"),
        (os.environ.get("ORCA_PATH"), "ORCA_PATH environment variable"),
        (DEFAULT_ORCA_EXECUTABLE, "default location"),
    ]
    for path, source in paths_to_check:
        if path and Path(path).is_file():
            logger.info(f"Using ORCA executable from {source}: {path}")
            return str(path)
    return None


def get_molecule_properties(mol: Chem.Mol) -> Tuple[int, int]:
    """Calculates the formal charge and spin multiplicity of an RDKit molecule."""
    charge = Chem.GetFormalCharge(mol)
    num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge
    multiplicity = 1 if num_electrons % 2 == 0 else 2
    logger.info(f"Molecule properties: Charge={charge}, Multiplicity={multiplicity}")
    return charge, multiplicity


def generate_orca_input_content(
    mol: Chem.Mol, charge: int, multiplicity: int, n_cores: int
) -> Optional[str]:
    """Generates the string content for an ORCA input file."""
    try:
        conformer = mol.GetConformer()
        xyz_coords = "\n".join(
            f"  {atom.GetSymbol():<2} {pos.x:12.8f} {pos.y:12.8f} {pos.z:12.8f}"
            for atom, pos in zip(mol.GetAtoms(), conformer.GetPositions())
        )
    except ValueError:
        logger.error("Molecule has no 3D conformer to generate coordinates from.")
        return None

    pal_block = f"%pal nprocs {n_cores}\nend\n" if n_cores > 1 else ""
    return (
        f"{ORCA_KEYWORDS}\n{pal_block}{ORCA_OUTPUT_BLOCK}\n"
        f"* xyz {charge} {multiplicity}\n{xyz_coords}\n*\n"
    )


def parse_orca_output(out_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses an ORCA output file to extract key QM properties.

    Args:
        out_path: Path to the ORCA .out file.

    Returns:
        A dictionary with extracted properties, or None if parsing fails.
    """
    if not out_path.exists():
        logger.error(f"ORCA output file not found for parsing: {out_path}")
        return None
    content = out_path.read_text(errors="ignore")

    energy_match = RE_ENERGY.search(content)
    dipole_match = RE_DIPOLE.search(content)
    chelpg_block_match = RE_CHELPG_BLOCK.search(content)

    if not all([energy_match, dipole_match, chelpg_block_match]):
        logger.warning(f"Could not parse all required data from {out_path.name}.")
        return None

    if chelpg_block_match is None: 
        logger.warning(f"CHELPG charge block regex failed in {out_path.name}.")
        return None

    charges = RE_CHARGE_LINE.findall(chelpg_block_match.group(1))
    if not charges:
        logger.warning(f"CHELPG charge block found but no charges parsed in {out_path.name}.")
        return None

    if energy_match is None or dipole_match is None: 
        logger.warning(f"Energy or Dipole match failed in {out_path.name}.")
        return None

    parsed_data: Dict[str, Any] = {
        "molid": out_path.stem,
        "energy_hartree": float(energy_match.group(1)),
        "dipole_D": float(dipole_match.group(1)),
    }
    parsed_data.update({f"q_atom_{i+1}": float(q) for i, q in enumerate(charges)})
    return parsed_data


def cleanup_orca_files(base_name: str, directory: Path):
    """Removes previous ORCA output and temporary files."""
    logger.info(f"Cleaning up previous ORCA files for {base_name} in {directory}...")
    extensions = [
        ".out", ".err", ".gbw", ".inp", ".prop", ".xyz", ".hess", ".opt",
        ".tmp", ".property.txt"
    ]
    for ext in extensions:
        for f in directory.glob(f"{base_name}{ext}"):
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Could not delete old file {f}: {e}")


def run_orca_calculation(
    orca_executable: str, input_path: Path, output_dir: Path
) -> bool:
    """
    Executes a single ORCA calculation as a subprocess.

    Args:
        orca_executable: Path to the ORCA executable.
        input_path: Path to the ORCA input file.
        output_dir: Directory where the calculation will run.

    Returns:
        True if the calculation completes successfully (exit code 0), False otherwise.
    """
    base_name = input_path.stem
    out_path = output_dir / f"{base_name}.out"
    err_path = output_dir / f"{base_name}.err"

    command = [orca_executable, str(input_path.resolve())]
    logger.info(f"Running command: {' '.join(command)}")

    try:
        with open(out_path, "w") as f_out, open(err_path, "w") as f_err:
            process = subprocess.run(
                command,
                cwd=output_dir,
                stdout=f_out,
                stderr=f_err,
                check=True,
                text=True,
            )
        logger.info(f"ORCA calculation for {base_name} completed successfully.")
        return True
    except FileNotFoundError:
        logger.error(f"ORCA executable not found at: {orca_executable}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ORCA calculation for {base_name} failed with exit code {e.returncode}.")
        logger.error(f"Check logs for details: {out_path} and {err_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during ORCA run for {base_name}: {e}")
    return False


def process_sdf_file(
    sdf_path: Path,
    orca_executable: str,
    output_dir: Path,
    n_cores: int,
    no_cleanup: bool,
) -> Optional[Dict[str, Any]]:
    """
    Full processing pipeline for a single SDF file.

    Generates input, runs calculation, and parses output.

    Returns:
        A dictionary of parsed QM properties, or None if any step fails.
    """
    logger.info(f"--- Processing SDF file: {sdf_path.name} ---")
    base_name = sdf_path.stem
    input_path = output_dir / f"{base_name}.inp"
    out_path = output_dir / f"{base_name}.out"

    suppl = Chem.SDMolSupplier(str(sdf_path))
    mol = next(suppl, None)
    if not mol:
        logger.error(f"Could not read molecule from {sdf_path}")
        return None

    charge, multiplicity = get_molecule_properties(mol)
    input_content = generate_orca_input_content(mol, charge, multiplicity, n_cores)
    if not input_content:
        return None

    output_dir.mkdir(exist_ok=True)
    if not no_cleanup:
        cleanup_orca_files(base_name, output_dir)
    input_path.write_text(input_content)

    success = run_orca_calculation(orca_executable, input_path, output_dir)
    if not success:
        return None

    return parse_orca_output(out_path)


def main():
    """Main function to run the ORCA batch processing script."""
    parser = argparse.ArgumentParser(description="Run ORCA calculations for a batch of SDF files.")
    parser.add_argument("sdf_files", nargs="+", help="Paths to one or more input SDF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store ORCA inputs and outputs.")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores for parallel ORCA jobs.")
    parser.add_argument("--orca_path", type=str, help="Path to the ORCA executable.")
    parser.add_argument("--no_cleanup", action="store_true", help="Do not delete previous ORCA output files.")
    parser.add_argument("--output_csv", type=str, default="qm_dataset.csv", help="Path for the final output CSV file.")
    args = parser.parse_args()

    orca_executable = find_orca_executable(args.orca_path)
    if not orca_executable:
        logger.error("Could not find ORCA executable. Exiting.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = []
    failed_sdfs = []

    for sdf_path_str in args.sdf_files:
        sdf_path = Path(sdf_path_str)
        if not sdf_path.exists():
            logger.warning(f"SDF file not found: {sdf_path}. Skipping.")
            failed_sdfs.append(sdf_path.name)
            continue
        
        result = process_sdf_file(
            sdf_path, orca_executable, output_dir, args.n_cores, args.no_cleanup
        )
        if result:
            all_results.append(result)
        else:
            failed_sdfs.append(sdf_path.name)
    
    if not all_results:
        logger.warning("No molecules were successfully processed.")
        return

    # Write results to CSV
    output_csv_path = output_dir / args.output_csv
    header = list(all_results[0].keys())
    try:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Successfully wrote QM data for {len(all_results)} molecules to {output_csv_path}")
    except IOError as e:
        logger.error(f"Failed to write output CSV file: {e}")

    if failed_sdfs:
        logger.warning(f"\n--- Processing Summary ---")
        logger.warning(f"Failed to process {len(failed_sdfs)} SDF files:")
        for fname in failed_sdfs:
            logger.warning(f"  - {fname}")


if __name__ == "__main__":
    main()