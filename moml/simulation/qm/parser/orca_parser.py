"""
moml/simulation/qm/parser/orca_parser.py

This module provides a comprehensive suite of tools for managing quantum
mechanical calculations using ORCA. It handles the entire workflow, from
generating 3D molecular structures from SMILES strings to creating ORCA input
files, executing calculations, and parsing the resulting output files to
extract key quantum chemical properties.

The functionality is designed to be robust and scalable, supporting both single-
molecule processing and large-scale batch operations with parallel execution.
"""

import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.WARNING)

logger = logging.getLogger(__name__)

# A robust regex for matching floating-point numbers in scientific notation.
FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

# Conversion factor from Hartree to electron-volts (eV).
HARTREE_TO_EV = 27.211

# Pre-compiled regex patterns for efficient parsing of ORCA output files.
ORCA_PATTERNS = {
    "calculation_completed": re.compile(r"ORCA TERMINATED NORMALLY"),
    "error": re.compile(r"ERROR|Error|ORCA TERMINATED ABNORMALLY"),
    "mulliken_charges_header": re.compile(r"MULLIKEN ATOMIC CHARGES"),
    "loewdin_charges_header": re.compile(r"LOEWDIN ATOMIC CHARGES"),
    "charge_line": re.compile(f"^\\s*\\d+\\s+[A-Za-z]{{1,3}}\\s*:?\\s*({FLOAT})"),
    "dipole_moment": re.compile(
        f"DIPOLE MOMENT(?:.|\\n)*?Total\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})",
        re.DOTALL,
    ),
    "homo_lumo_gap_direct": re.compile(
        f"HOMO-LUMO gap:\\s*({FLOAT})\\s*Eh\\s*=\\s*({FLOAT})\\s*eV"
    ),
    "homo_energy": re.compile(f"^\\s*\\d+\\s+({FLOAT})\\s+({FLOAT})+\\s+\\(HOMO\\)", re.MULTILINE),
    "lumo_energy": re.compile(f"^\\s*\\d+\\s+({FLOAT})\\s+({FLOAT})+\\s+\\(LUMO\\)", re.MULTILINE),
    "geometry": re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\).*?\n(.*?)\n\n", re.DOTALL),
    "atom_line": re.compile(f"(\\w+)\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})"),
    "block_end": re.compile(r"\n\s*\n|[A-Z\s]{10,}\n-{5,}"),
}

def _parse_charge_block(content: str, header_pattern: re.Pattern) -> List[float]:
    """
    Extracts a block of atomic charges (Mulliken or Loewdin) from ORCA output.

    Args:
        content (str): The full content of the ORCA output file.
        header_pattern (re.Pattern): The regex pattern for the charge block header.

    Returns:
        List[float]: A list of atomic charges, or an empty list if not found.
    """
    charges = []
    header_match = header_pattern.search(content)
    if not header_match:
        return charges

    start_index = header_match.end()
    search_block = content[start_index:]
    end_match = ORCA_PATTERNS["block_end"].search(search_block)
    block_limit = end_match.start() if end_match else len(search_block)
    relevant_block = search_block[:block_limit]

    for line in relevant_block.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        match = ORCA_PATTERNS["charge_line"].match(line_strip)
        if match:
            try:
                charges.append(float(match.group(1)))
            except ValueError:
                logger.warning(
                    f"Could not parse float from charge line: '{line_strip}'"
                )
        elif charges and not line_strip.startswith("-"):
            # Stop if a non-empty, non-matching line is found after finding charges
            break
    return charges


def _parse_optimized_geometry(content: str) -> List[Dict[str, Any]]:
    """
    Extracts the final optimized geometry from ORCA output.

    Args:
        content (str): The full content of the ORCA output file.

    Returns:
        List[Dict[str, Any]]: A list of atoms, each represented by a dictionary
        containing the atomic symbol and 3D coordinates. Returns an empty
        list if not found.
    """
    geometry = []
    geometry_match = ORCA_PATTERNS["geometry"].search(content)
    if not geometry_match:
        return geometry

    geometry_text = geometry_match.group(1)
    for line in geometry_text.splitlines():
        if not line.strip():
            continue
        atom_match = ORCA_PATTERNS["atom_line"].search(line)
        if atom_match:
            symbol = atom_match.group(1)
            x, y, z = (
                float(atom_match.group(2)),
                float(atom_match.group(3)),
                float(atom_match.group(4)),
            )
            geometry.append({"symbol": symbol, "coordinates": [x, y, z]})
    return geometry


def _calculate_homo_lumo_gap(content: str) -> Optional[float]:
    """
    Extracts or calculates the HOMO-LUMO gap in eV.

    Args:
        content (str): The full content of the ORCA output file.

    Returns:
        Optional[float]: The HOMO-LUMO gap in eV, or None if it cannot be determined.
    """
    # First, try to find the gap reported directly by ORCA.
    gap_match = ORCA_PATTERNS["homo_lumo_gap_direct"].search(content)
    if gap_match:
        return float(gap_match.group(2))

    # If not found, try to calculate it from individual orbital energies.
    homo_match = ORCA_PATTERNS["homo_energy"].search(content)
    lumo_match = ORCA_PATTERNS["lumo_energy"].search(content)

    if homo_match and lumo_match:
        homo_energy = float(homo_match.group(1))  # Energy is in Hartree
        lumo_energy = float(lumo_match.group(1))
        gap_hartree = lumo_energy - homo_energy
        return gap_hartree * HARTREE_TO_EV

    return None

def parse_orca_output(
    orca_output_path: str,
) -> Dict[str, Union[List[float], np.ndarray, Dict, None]]:
    """
    Parse an ORCA output file to extract key quantum mechanical data.

    This function performs a single pass over the file to efficiently extract
    multiple properties.

    Args:
        orca_output_path (str): The file path to the ORCA output file.

    Returns:
        Dict[str, Union[List[float], np.ndarray, Dict, None]]: A dictionary
        containing the extracted data. Keys include:
        - 'status': Calculation status ('completed', 'error', 'incomplete').
        - 'mulliken_charges': List of Mulliken partial atomic charges.
        - 'loewdin_charges': List of Loewdin partial atomic charges.
        - 'dipole_moment': Dipole moment vector [dx, dy, dz, total].
        - 'homo_lumo_gap': The HOMO-LUMO gap in eV.
        - 'optimized_geometry': Cartesian coordinates of the optimized structure.
        - 'error_message': A description of the error if one occurred.

    Raises:
        FileNotFoundError: If the specified `orca_output_path` does not exist.
    """
    if not os.path.exists(orca_output_path):
        raise FileNotFoundError(f"ORCA output file not found: {orca_output_path}")

    result: Dict[str, Any] = {
        "status": "incomplete",
        "mulliken_charges": [],
        "loewdin_charges": [],
        "dipole_moment": None,
        "homo_lumo_gap": None,
        "optimized_geometry": [],
        "error_message": None,
    }

    try:
        with open(orca_output_path, "r") as f:
            content = f.read()

        # Determine the final status of the calculation.
        if ORCA_PATTERNS["calculation_completed"].search(content):
            result["status"] = "completed"
        elif ORCA_PATTERNS["error"].search(content):
            result["status"] = "error"

        # Extract all relevant properties using helper functions.
        result["mulliken_charges"] = _parse_charge_block(
            content, ORCA_PATTERNS["mulliken_charges_header"]
        )
        result["loewdin_charges"] = _parse_charge_block(
            content, ORCA_PATTERNS["loewdin_charges_header"]
        )
        result["optimized_geometry"] = _parse_optimized_geometry(content)
        result["homo_lumo_gap"] = _calculate_homo_lumo_gap(content)

        # Extract dipole moment directly.
        dipole_match = ORCA_PATTERNS["dipole_moment"].search(content)
        if dipole_match:
            result["dipole_moment"] = [float(g) for g in dipole_match.groups()]

    except Exception as e:
        logger.error(f"Error parsing ORCA output '{orca_output_path}': {e}")
        result["status"] = "error"
        result["error_message"] = str(e)

    return result


def extract_partial_charges_from_orca(
    orca_output_path: str, charge_type: str = "mulliken"
) -> List[float]:
    """
    Extract a specific type of partial charges from an ORCA output file.

    Args:
        orca_output_path (str): Path to the ORCA output file.
        charge_type (str, optional): The type of charges to extract, either
            'mulliken' or 'loewdin'. Defaults to "mulliken".

    Returns:
        List[float]: A list of partial charges. Returns an empty list if the
        requested charge type is not found.
    """
    data = parse_orca_output(orca_output_path)
    charge_type_lower = charge_type.lower()
    charge_key = f"{charge_type_lower}_charges"

    # If the requested charge type exists, return it directly.
    if charge_key in data and isinstance(data[charge_key], list):
        return data[charge_key]  # type: ignore[return-value]

    # Fallback logic – default to Mulliken charges if an unsupported type was requested.
    if charge_type_lower not in {"mulliken", "loewdin"}:
        logger.warning(
            f"Unknown charge type '{charge_type}' requested. Falling back to Mulliken charges."
        )
        mulliken_charges = data.get("mulliken_charges", [])
        return mulliken_charges if isinstance(mulliken_charges, list) else []

    # Supported type requested but not present in the output – return empty list.
    return []


def smiles_to_3d_structure(
    smiles: str, molecule_id: str, optimize: bool = True
) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to a 3D RDKit molecule object.

    This process includes adding hydrogens, embedding the molecule in 3D space,
    and optionally performing a force field optimization.

    Args:
        smiles (str): The SMILES string of the molecule.
        molecule_id (str): A unique identifier for the molecule, used for logging.
        optimize (bool, optional): If True, perform MMFF94 force field
            optimization. Defaults to True.

    Returns:
        Optional[Chem.Mol]: An RDKit molecule object with a 3D conformer, or
        None if the conversion fails at any step.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"[{molecule_id}] Failed to parse SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)

        # Generate initial 3D coordinates.
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:  # type: ignore
            # Match test expectation wording
            logger.error(f"Coordinate generation failed for {molecule_id}")
            return None

        # Optimize the geometry using a molecular mechanics force field.
        if optimize:
            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore

        return mol

    except Exception as e:
        logger.error(f"[{molecule_id}] Error creating 3D structure: {e}")
        return None


def create_orca_input(
    mol: Chem.Mol,
    molecule_id: str,
    output_dir: str,
    functional: str = "B3LYP",
    basis_set: str = "6-31G*",
    num_procs: int = 4,
    memory_mb: int = 4000,
    **kwargs,
) -> Tuple[bool, str]:
    """
    Create an ORCA input file (.inp) for a given molecule.

    Args:
        mol (Chem.Mol): An RDKit molecule object with a 3D conformer.
        molecule_id (str): A unique identifier for the molecule.
        output_dir (str): The directory where the input file will be saved.
        functional (str, optional): The DFT functional. Defaults to "B3LYP".
        basis_set (str, optional): The basis set. Defaults to "6-31G*".
        num_procs (int, optional): The number of processors for parallel execution.
            Defaults to 4.
        memory_mb (int, optional): The maximum memory per core in MB.
            Defaults to 4000.

    Returns:
        Tuple[bool, str]: (success flag, path to created input file or empty string)
    """
    try:
        # Support legacy keyword alias 'memory'
        if "memory" in kwargs and not kwargs.get("memory_mb"):
            memory_mb = int(kwargs["memory"])

        os.makedirs(output_dir, exist_ok=True)
        input_file_path = os.path.join(output_dir, f"{molecule_id}.inp")

        # Adjust functional names for ORCA compatibility.
        if functional.upper() == "B3LYP":
            orca_functional = "B3LYP D3BJ"  # Add dispersion correction
        elif functional.upper() == "WB97X-D":
            orca_functional = "wB97X-D3"
        else:
            orca_functional = functional

        with open(input_file_path, "w") as f:
            f.write(f"# ORCA Input for {molecule_id}\n\n")
            f.write(f"! {orca_functional} {basis_set} OPT TIGHTSCF\n\n")

            if num_procs > 1:
                f.write(f"%pal nprocs {num_procs} end\n\n")

            f.write(f"%maxcore {memory_mb}\n\n")

            f.write("%geom MaxIter 250 Convergence Tight end\n\n")

            # Write atomic coordinates in XYZ format.
            f.write("* xyz 0 1\n")
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conformer.GetAtomPosition(atom.GetIdx())
                f.write(f"  {atom.GetSymbol():<2} {pos.x:12.8f} {pos.y:12.8f} {pos.z:12.8f}\n")
            f.write("*\n")

        # Save a MOL file for easy visualization and record-keeping.
        mol_file_path = os.path.join(output_dir, f"{molecule_id}.mol")
        Chem.MolToMolFile(mol, mol_file_path)

        return True, input_file_path

    except Exception as e:
        logger.error(f"Error creating ORCA input file for {molecule_id}: {e}")
        return False, ""


def run_orca_calculation(
    input_file_path: str,
    orca_executable: Optional[str] = None,
    *,
    orca_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Execute an ORCA calculation as a subprocess.

    Args:
        input_file_path (str): The path to the ORCA input file (.inp).
        orca_executable (Optional[str], optional): The path to the ORCA
            executable. If None, it will be searched for in common system
            locations. Defaults to None.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean success flag and the
        path to the generated output file (.out).

    Raises:
        FileNotFoundError: If the ORCA executable cannot be found.
    """
    input_dir = os.path.dirname(input_file_path)
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(input_dir, f"{base_name}.out")

    # Allow legacy keyword 'orca_path' used in unit tests
    if orca_path is not None:
        orca_executable = orca_path

    # Find the ORCA executable if still not provided.
    if orca_executable is None:
        possible_paths = ["/opt/orca/orca", "orca"]  # System path is last resort
        for path in possible_paths:
            # Check if it's a valid file or if it's in the system's PATH.
            if os.path.exists(path) or shutil.which(path):
                orca_executable = path  # type: ignore
                break
        if orca_executable is None:
            raise FileNotFoundError("ORCA executable not found.")

    command = [orca_executable, input_file_path]
    logger.info(f"Running ORCA: {' '.join(command)}")

    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=input_dir,
            check=False,  # Don't raise exception on non-zero exit code
        )

        if process.returncode != 0:
            logger.error(
                f"ORCA process for '{base_name}' failed with code "
                f"{process.returncode}.\nSTDERR: {process.stderr}"
            )
            return False, output_file_path

        if not os.path.exists(output_file_path):
            logger.error(f"ORCA output file was not created: {output_file_path}")
            return False, output_file_path

        logger.info(f"ORCA calculation completed for: {output_file_path}")
        return True, output_file_path

    except FileNotFoundError:
        logger.error(f"ORCA executable not found at '{orca_executable}'")
        return False, ""
    except Exception as e:
        logger.error(f"An unexpected error occurred while running ORCA: {e}")
        return False, ""


def process_molecule(
    smiles: str,
    molecule_id: str,
    output_dir: str,
    functional: str,
    basis_set: str,
    num_procs: int,
    memory_mb: int,
    orca_executable: Optional[str],
) -> Dict[str, Any]:
    """
    Execute the full QM workflow for a single molecule.

    This workflow includes: 3D structure generation, ORCA input creation,
    calculation execution, and output parsing.

    Args:
        smiles (str): The SMILES string of the molecule.
        molecule_id (str): A unique identifier for the molecule.
        output_dir (str): The main directory for all outputs.
        functional (str): The DFT functional.
        basis_set (str): The basis set.
        num_procs (int): The number of processors for the calculation.
        memory_mb (int): Memory per core in MB.
        orca_executable (Optional[str]): Path to the ORCA executable.

    Returns:
        Dict[str, Any]: A dictionary containing the results, status, and any
        errors for the processed molecule.
    """
    result: Dict[str, Any] = {"id": molecule_id, "status": "failed", "data": None}
    mol_dir = os.path.join(output_dir, str(molecule_id))
    os.makedirs(mol_dir, exist_ok=True)

    mol = smiles_to_3d_structure(smiles, molecule_id)
    if not mol:
        result["error"] = "Failed to create 3D structure"
        return result

    success_inp, input_file = create_orca_input(
        mol,
        molecule_id,
        mol_dir,
        functional,
        basis_set,
        num_procs,
        memory_mb,
    )
    if not success_inp:
        result["error"] = "Failed to create ORCA input file."
        return result

    success, output_file = run_orca_calculation(input_file, orca_executable=orca_executable)
    if not success:
        result["error"] = "ORCA calculation failed or produced no output."
        return result

    data = parse_orca_output(output_file)

    # Store parsed data under 'data' key and propagate status if available
    result["data"] = data
    if isinstance(data, dict) and "status" in data:
        result["status"] = data["status"]

    # Save final results to a JSON file for persistent storage.
    results_file = os.path.join(mol_dir, f"{molecule_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def batch_process_molecules(
    molecules_df: pd.DataFrame,
    output_dir: str,
    functional: str = "B3LYP",
    basis_set: str = "6-31G*",
    num_procs: int = 4,
    memory_mb: int = 4000,
    orca_executable: Optional[str] = None,
    max_workers: int = 1,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Process a batch of molecules in parallel using a process pool.

    Args:
        molecules_df (pd.DataFrame): DataFrame with molecule data.
        output_dir (str): The root directory for all calculation outputs.
        functional (str, optional): DFT functional. Defaults to "B3LYP".
        basis_set (str, optional): Basis set. Defaults to "6-31G*".
        num_procs (int, optional): Processors per ORCA job. Defaults to 4.
        memory_mb (int, optional): Memory per core in MB. Defaults to 4000.
        orca_executable (Optional[str], optional): Path to ORCA executable.
        max_workers (int, optional): Number of parallel Python workers.
            Defaults to 1 (sequential).
        smiles_col (str, optional): Column name for SMILES. Defaults to "SMILES".
        id_col (str, optional): Column name for molecule IDs. Defaults to "ID".

    Returns:
        pd.DataFrame: A DataFrame containing the results of all processed molecules.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    process_args = [
        (
            row[smiles_col],
            row[id_col],
                    output_dir,
                    functional,
                    basis_set,
                    num_procs,
            memory_mb,
            orca_executable,
        )
        for _, row in molecules_df.iterrows()
    ]

    # Use a ProcessPoolExecutor for true parallelism.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # starmap is used to pass multiple arguments to the worker function.
        future_results = executor.map(lambda p: process_molecule(*p), process_args)

        for result in future_results:
            all_results.append(result)
            logger.info(
                f"Completed processing for {result['id']} "
                f"with status: {result['status']}"
            )

    # Create and save a summary DataFrame.
    results_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, "batch_processing_summary.csv")
    summary_df = results_df[["id", "status", "error"]].copy()
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Batch processing summary saved to: {summary_file}")

    return results_df
