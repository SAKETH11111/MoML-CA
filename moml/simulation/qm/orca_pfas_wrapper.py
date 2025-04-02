#!/usr/bin/env python3
"""
ORCA PFAS Wrapper for Quantum Chemistry Calculations

This module provides functionality to:
1. Process PFAS compounds from input CSV files
2. Generate ORCA input files for quantum chemistry calculations
3. Run ORCA calculations in parallel or serial mode
4. Extract and process results for ML model training
5. Generate standardized output files for downstream analysis

Usage:
    python orca_pfas_wrapper.py --input_csv path/to/input.csv --output_dir path/to/output
    
For detailed help:
    python orca_pfas_wrapper.py --help
"""

import os
import sys
import argparse
import pandas as pd
import json
import logging
from pathlib import Path
import time
import concurrent.futures
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orca_pfas_wrapper")

# Suppress RDKit logging except for warnings and errors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.WARNING)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORCA PFAS Wrapper")
    
    # Required arguments
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV file with PFAS data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output files")
    
    # Optional arguments
    parser.add_argument("--smiles_col", type=str, default="canonical_smiles",
                        help="Column name containing SMILES strings")
    parser.add_argument("--id_col", type=str, default="common_name",
                        help="Column name containing molecule identifiers")
    parser.add_argument("--functional", type=str, default="B3LYP",
                        help="Computational method/functional to use")
    parser.add_argument("--basis_set", type=str, default="6-31G*",
                        help="Basis set to use")
    parser.add_argument("--num_procs", type=int, default=1,
                        help="Number of processors to use per calculation")
    parser.add_argument("--memory", type=int, default=2000,
                        help="Memory allocation in MB")
    parser.add_argument("--max_jobs", type=int, default=1,
                        help="Maximum number of concurrent jobs")
    parser.add_argument("--orca_path", type=str, default=None,
                        help="Path to ORCA executable")
    parser.add_argument("--openmpi_path", type=str, default=None,
                        help="Path to OpenMPI binaries")
    parser.add_argument("--optimize", action="store_true",
                        help="Perform geometry optimization")
    parser.add_argument("--charges", action="store_true",
                        help="Calculate partial charges")
    parser.add_argument("--sp_only", action="store_true",
                        help="Run single-point calculation only (no optimization)")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no actual ORCA calculations)")
    
    return parser.parse_args()

def generate_3d_structure(smiles: str, molecule_id: str) -> Optional[Chem.Mol]:
    """
    Generate 3D structure from SMILES string.
    
    Args:
        smiles: SMILES string
        molecule_id: Molecule identifier
        
    Returns:
        RDKit molecule with 3D coordinates, or None if failed
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Failed to parse SMILES for {molecule_id}: {smiles}")
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            logger.error(f"3D coordinate generation failed for {molecule_id}")
            return None
        
        # Clean up the structure with force field optimization
        AllChem.MMFFOptimizeMolecule(mol)
        
        return mol
    
    except Exception as e:
        logger.error(f"Error generating 3D structure for {molecule_id}: {str(e)}")
        return None

def generate_orca_input(
    mol: Chem.Mol, 
    molecule_id: str, 
    output_dir: str, 
    functional: str = "B3LYP", 
    basis_set: str = "6-31G*",
    num_procs: int = 1,
    memory: int = 2000,
    optimize: bool = True,
    charges: bool = True,
    sp_only: bool = False
) -> Tuple[bool, str]:
    """
    Generate ORCA input file for a molecule.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        molecule_id: Molecule identifier
        output_dir: Directory for output files
        functional: Computational method/functional
        basis_set: Basis set
        num_procs: Number of processors
        memory: Memory allocation in MB
        optimize: Whether to perform geometry optimization
        charges: Whether to calculate partial charges
        sp_only: Whether to run single-point calculation only
        
    Returns:
        Tuple of (success, input_file_path)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure calculation type
        calc_type = ""
        if sp_only:
            # Single-point calculation only
            calc_type = ""  # No special keywords needed
        elif optimize:
            # Geometry optimization
            calc_type = "OPT"
        
        # Configure ORCA functional
        orca_functional = functional
        # Add dispersion correction to B3LYP
        if functional == "B3LYP":
            orca_functional = "B3LYP D3BJ"
        
        # Configure charge calculation
        charge_calc = ""
        if charges:
            # Request both Mulliken and Löwdin charges
            charge_calc = " MULLIKEN LOEWDIN"
        
        # Output file path
        input_file = os.path.join(output_dir, f"{molecule_id}.inp")
        
        # Write ORCA input file
        with open(input_file, "w") as f:
            # Header
            f.write(f"# ORCA Input for {molecule_id}\n")
            f.write(f"# Generated by MoML-CA ORCA wrapper\n")
            f.write(f"# Functional: {functional}, Basis: {basis_set}\n\n")
            
            # Main calculation line
            f.write(f"! {orca_functional} {basis_set} {calc_type}{charge_calc} TIGHTSCF\n\n")
            
            # Parallel settings
            if num_procs > 1:
                f.write("%pal\n")
                f.write(f"  nprocs {num_procs}\n")
                f.write("end\n\n")
            
            # Memory settings
            f.write(f"%maxcore {memory}\n\n")
            
            # SCF settings
            f.write("%scf\n")
            f.write("  MaxIter 250\n")
            f.write("  Convergence Tight\n")
            f.write("end\n\n")
            
            # Optimization settings if applicable
            if optimize and not sp_only:
                f.write("%geom\n")
                f.write("  MaxIter 250\n")
                f.write("  Convergence Tight\n")
                f.write("end\n\n")
            
            # Molecule specification
            # Get total molecular charge (0 for neutral molecules)
            mol_charge = Chem.GetFormalCharge(mol)
            # Assuming singlet spin state (multiplicity = 1)
            multiplicity = 1
            
            # XYZ format
            f.write(f"* xyz {mol_charge} {multiplicity}\n")
            
            # Write atom coordinates
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                symbol = atom.GetSymbol()
                pos = conf.GetAtomPosition(idx)
                f.write(f"  {symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
            
            f.write("*\n")
        
        # Also save molecule in MOL format for visualization
        mol_file = os.path.join(output_dir, f"{molecule_id}.mol")
        Chem.MolToMolFile(mol, mol_file)
        
        return True, input_file
    
    except Exception as e:
        logger.error(f"Error generating ORCA input for {molecule_id}: {str(e)}")
        return False, ""

def run_orca_calculation(
    input_file: str, 
    orca_path: Optional[str] = None,
    openmpi_path: Optional[str] = None,
    mock: bool = False
) -> Tuple[bool, str]:
    """
    Run ORCA calculation.
    
    Args:
        input_file: Path to ORCA input file
        orca_path: Path to ORCA executable (or None to use system default)
        openmpi_path: Path to OpenMPI binaries (or None to use system default)
        mock: Whether to run in mock mode (no actual calculation)
        
    Returns:
        Tuple of (success, output_file_path)
    """
    # Get input directory and output file path
    input_dir = os.path.dirname(input_file)
    mol_id = os.path.basename(input_file).replace(".inp", "")
    output_file = os.path.join(input_dir, f"{mol_id}.out")
    
    # Mock mode - create a fake output file
    if mock:
        logger.info(f"Mock mode: Simulating ORCA calculation for {mol_id}")
        
        # Create minimal mock output
        with open(output_file, "w") as f:
            f.write(f"ORCA Mockup Output for {mol_id}\n")
            f.write("---------------------------------------------\n\n")
            f.write("ORCA TERMINATED NORMALLY\n")
            
            # Add fake Mulliken charges section
            f.write("\nMULLIKEN ATOMIC CHARGES\n")
            f.write("---------------------------------------------\n")
            f.write("   0 C :    0.123456\n")
            f.write("   1 F :   -0.123456\n")
            
            # Add fake LOEWDIN charges section
            f.write("\nLOEWDIN ATOMIC CHARGES\n")
            f.write("---------------------------------------------\n")
            f.write("   0 C :    0.098765\n")
            f.write("   1 F :   -0.098765\n")
            
            # Add fake HOMO-LUMO section
            f.write("\nHOMO-LUMO gap:     0.1234 Eh =     3.5678 eV\n")
            
            # Add fake dipole moment
            f.write("\nDIPOLE MOMENT\n")
            f.write("---------------------------------------------\n")
            f.write("X:     1.234 Y:     2.345 Z:     3.456 Total:     4.567\n")
            
        time.sleep(0.5)  # Simulate calculation time
        return True, output_file
    
    # Real mode - run actual ORCA calculation
    try:
        # Find ORCA executable
        if orca_path is None:
            # Try common installation paths
            possible_paths = [
                "/usr/local/bin/orca",
                "/opt/orca/orca",
                os.path.expanduser("~/Library/orca_6_0_1/orca"),
                "orca"  # System PATH
            ]
            
            for path in possible_paths:
                if path == "orca" or os.path.exists(path):
                    orca_path = path
                    logger.info(f"Using ORCA: {orca_path}")
                    break
            
            if orca_path is None:
                logger.error("ORCA executable not found")
                return False, ""
        
        # Set up environment variables for OpenMPI if specified
        env = os.environ.copy()
        if openmpi_path:
            env["PATH"] = f"{openmpi_path}:{env.get('PATH', '')}"
            logger.info(f"Using OpenMPI from: {openmpi_path}")
        
        # Run ORCA calculation
        cmd = [orca_path, input_file]
        logger.info(f"Running: {' '.join(cmd)} in {input_dir}")
        
        process = subprocess.run(
            cmd,
            env=env,
            cwd=input_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Check if output file was created
        if not os.path.exists(output_file):
            logger.error(f"ORCA calculation failed - no output file created for {mol_id}")
            logger.error(f"STDERR: {process.stderr}")
            return False, ""
        
        # Check process return code
        if process.returncode != 0:
            logger.error(f"ORCA calculation failed with code {process.returncode} for {mol_id}")
            logger.error(f"STDERR: {process.stderr}")
            
            # Check if we have a partial output file that might be usable
            if os.path.getsize(output_file) > 100:
                logger.warning(f"Using partial output file for {mol_id}")
                return True, output_file
            
            return False, output_file
        
        logger.info(f"ORCA calculation completed successfully for {mol_id}")
        return True, output_file
    
    except Exception as e:
        logger.error(f"Error running ORCA calculation for {mol_id}: {str(e)}")
        return False, ""

def parse_orca_output(output_file: str) -> Dict[str, Any]:
    """
    Parse ORCA output file to extract quantum mechanical properties.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Dictionary of extracted properties
    """
    if not os.path.exists(output_file):
        logger.error(f"ORCA output file not found: {output_file}")
        return {
            "success": False,
            "error": "Output file not found"
        }
    
    # Initialize result dictionary
    result = {
        "success": False,
        "energy": None,
        "dipole_moment": None,
        "homo_energy": None,
        "lumo_energy": None,
        "homo_lumo_gap": None,
        "mulliken_charges": [],
        "loewdin_charges": []
    }
    
    try:
        # Read output file
        with open(output_file, "r") as f:
            content = f.read()
        
        # Check if calculation completed successfully
        if "ORCA TERMINATED NORMALLY" in content:
            result["success"] = True
        else:
            result["error"] = "ORCA calculation did not terminate normally"
            return result
        
        # Extract final energy
        energy_match = re.search(r"FINAL SINGLE POINT ENERGY\s+([-]?\d+\.\d+)", content)
        if energy_match:
            result["energy"] = float(energy_match.group(1))
        
        # Extract dipole moment
        dipole_match = re.search(r"DIPOLE MOMENT\s*\n.*?X\s+([-+]?\d+\.\d+).*?Y\s+([-+]?\d+\.\d+).*?Z\s+([-+]?\d+\.\d+).*?Total\s+([-+]?\d+\.\d+)", content, re.DOTALL)
        if dipole_match:
            result["dipole_moment"] = {
                "x": float(dipole_match.group(1)),
                "y": float(dipole_match.group(2)),
                "z": float(dipole_match.group(3)),
                "total": float(dipole_match.group(4))
            }
        
        # Extract HOMO-LUMO energies
        homo_match = re.search(r"HOMO:\s*\d+\s+([-+]?\d+\.\d+)\s*Eh", content)
        lumo_match = re.search(r"LUMO:\s*\d+\s+([-+]?\d+\.\d+)\s*Eh", content)
        if homo_match and lumo_match:
            homo_energy = float(homo_match.group(1))
            lumo_energy = float(lumo_match.group(1))
            result["homo_energy"] = homo_energy
            result["lumo_energy"] = lumo_energy
            result["homo_lumo_gap"] = lumo_energy - homo_energy
        
        # Extract HOMO-LUMO gap directly if available
        gap_match = re.search(r"HOMO-LUMO gap:\s*([-+]?\d+\.\d+)\s*Eh\s*=\s*([-+]?\d+\.\d+)\s*eV", content)
        if gap_match:
            result["homo_lumo_gap_ev"] = float(gap_match.group(2))
        
        # Extract Mulliken charges
        mulliken_section = re.search(r"MULLIKEN ATOMIC CHARGES.*?\n(.*?)\n\n", content, re.DOTALL)
        if mulliken_section:
            lines = mulliken_section.group(1).strip().split("\n")
            for line in lines:
                if re.match(r"\s*\d+\s+\w+\s*:", line):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        charge = float(parts[1].strip())
                        result["mulliken_charges"].append(charge)
        
        # Extract Loewdin charges
        loewdin_section = re.search(r"LOEWDIN ATOMIC CHARGES.*?\n(.*?)\n\n", content, re.DOTALL)
        if loewdin_section:
            lines = loewdin_section.group(1).strip().split("\n")
            for line in lines:
                if re.match(r"\s*\d+\s+\w+\s*:", line):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        charge = float(parts[1].strip())
                        result["loewdin_charges"].append(charge)
        
        return result
    
    except Exception as e:
        logger.error(f"Error parsing ORCA output: {str(e)}")
        return {
            "success": False,
            "error": f"Error parsing output: {str(e)}"
        }
import re

def process_molecule(
    smiles: str,
    molecule_id: str,
    output_dir: str,
    functional: str = "B3LYP",
    basis_set: str = "6-31G*",
    num_procs: int = 1,
    memory: int = 2000,
    orca_path: Optional[str] = None,
    openmpi_path: Optional[str] = None,
    optimize: bool = True,
    charges: bool = True,
    sp_only: bool = False,
    mock: bool = False
) -> Dict[str, Any]:
    """
    Process a single molecule through the ORCA workflow.
    
    Args:
        smiles: SMILES string
        molecule_id: Molecule identifier
        output_dir: Directory for output files
        functional: Computational method/functional
        basis_set: Basis set
        num_procs: Number of processors
        memory: Memory allocation in MB
        orca_path: Path to ORCA executable
        openmpi_path: Path to OpenMPI binaries
        optimize: Whether to perform geometry optimization
        charges: Whether to calculate partial charges
        sp_only: Whether to run single-point calculation only
        mock: Whether to run in mock mode
        
    Returns:
        Dictionary with calculation results
    """
    # Create clean molecule ID (remove special characters)
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', molecule_id)
    
    # Create molecule-specific output directory
    mol_dir = os.path.join(output_dir, clean_id)
    os.makedirs(mol_dir, exist_ok=True)
    
    logger.info(f"Processing molecule: {molecule_id}")
    
    # Initialize result dictionary
    result = {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "success": False,
        "calculation_type": {
            "functional": functional,
            "basis_set": basis_set,
            "optimize": optimize and not sp_only,
            "charges": charges,
            "sp_only": sp_only
        }
    }
    
    # 1. Generate 3D structure
    logger.info(f"Generating 3D structure for {molecule_id}")
    mol = generate_3d_structure(smiles, molecule_id)
    if mol is None:
        result["error"] = "Failed to generate 3D structure"
        return result
    
    # 2. Generate ORCA input
    logger.info(f"Generating ORCA input for {molecule_id}")
    success, input_file = generate_orca_input(
        mol, clean_id, mol_dir, 
        functional, basis_set, 
        num_procs, memory,
        optimize, charges, sp_only
    )
    if not success:
        result["error"] = "Failed to generate ORCA input"
        return result
    
    # 3. Run ORCA calculation
    logger.info(f"Running ORCA calculation for {molecule_id}")
    success, output_file = run_orca_calculation(
        input_file, orca_path, openmpi_path, mock
    )
    if not success:
        result["error"] = "ORCA calculation failed"
        return result
    
    # 4. Parse ORCA output
    logger.info(f"Parsing ORCA output for {molecule_id}")
    qm_data = parse_orca_output(output_file)
    
    # 5. Update result dictionary
    result.update(qm_data)
    
    # 6. Save result to JSON file
    result_file = os.path.join(mol_dir, f"{clean_id}_results.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Completed processing {molecule_id}: {result['success']}")
    return result

def process_molecules_parallel(
    df: pd.DataFrame,
    output_dir: str,
    functional: str,
    basis_set: str,
    num_procs: int,
    memory: int,
    max_jobs: int,
    orca_path: Optional[str],
    openmpi_path: Optional[str],
    optimize: bool,
    charges: bool,
    sp_only: bool,
    mock: bool,
    smiles_col: str,
    id_col: str
) -> List[Dict[str, Any]]:
    """
    Process multiple molecules in parallel.
    
    Args:
        df: DataFrame with molecules
        output_dir: Directory for output files
        functional: Computational method/functional
        basis_set: Basis set
        num_procs: Number of processors per calculation
        memory: Memory allocation in MB per calculation
        max_jobs: Maximum number of concurrent jobs
        orca_path: Path to ORCA executable
        openmpi_path: Path to OpenMPI binaries
        optimize: Whether to perform geometry optimization
        charges: Whether to calculate partial charges
        sp_only: Whether to run single-point calculation only
        mock: Whether to run in mock mode
        smiles_col: Column name with SMILES strings
        id_col: Column name with molecule identifiers
        
    Returns:
        List of result dictionaries
    """
    logger.info(f"Processing {len(df)} molecules with {max_jobs} concurrent jobs")
    results = []
    
    # Process in parallel if max_jobs > 1
    if max_jobs > 1 and not mock:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
            futures = {}
            
            for _, row in df.iterrows():
                smiles = row[smiles_col]
                molecule_id = str(row[id_col])
                
                future = executor.submit(
                    process_molecule,
                    smiles, molecule_id, output_dir,
                    functional, basis_set,
                    num_procs, memory,
                    orca_path, openmpi_path,
                    optimize, charges, sp_only, mock
                )
                
                futures[future] = molecule_id
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                molecule_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {molecule_id}: {result['success']}")
                except Exception as e:
                    logger.error(f"Error processing {molecule_id}: {str(e)}")
                    results.append({
                        "molecule_id": molecule_id,
                        "success": False,
                        "error": str(e)
                    })
    else:
        # Process sequentially
        for _, row in df.iterrows():
            smiles = row[smiles_col]
            molecule_id = str(row[id_col])
            
            try:
                result = process_molecule(
                    smiles, molecule_id, output_dir,
                    functional, basis_set,
                    num_procs, memory,
                    orca_path, openmpi_path,
                    optimize, charges, sp_only, mock
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {molecule_id}: {str(e)}")
                results.append({
                    "molecule_id": molecule_id,
                    "success": False,
                    "error": str(e)
                })
    
    return results

def prepare_ml_data(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare ML training data from ORCA results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of ML training data dictionaries
    """
    ml_data = []
    
    for result in results:
        if not result.get("success", False):
            continue
        
        # Create ML data entry
        ml_entry = {
            "compound_id": result["molecule_id"],
            "smiles": result["smiles"],
        }
        
        # Add quantum mechanical properties
        if result.get("energy") is not None:
            ml_entry["energy"] = result["energy"]
        
        if result.get("dipole_moment") is not None:
            ml_entry["dipole_moment"] = result["dipole_moment"].get("total")
            ml_entry["dipole_vector"] = [
                result["dipole_moment"].get("x"),
                result["dipole_moment"].get("y"),
                result["dipole_moment"].get("z")
            ]
        
        if result.get("homo_energy") is not None:
            ml_entry["homo_energy"] = result["homo_energy"]
        
        if result.get("lumo_energy") is not None:
            ml_entry["lumo_energy"] = result["lumo_energy"]
        
        if result.get("homo_lumo_gap") is not None:
            ml_entry["homo_lumo_gap"] = result["homo_lumo_gap"]
            # Convert to eV if not already
            ml_entry["homo_lumo_gap_ev"] = result.get("homo_lumo_gap_ev", result["homo_lumo_gap"] * 27.211)
        
        # Add partial charges if available
        if "mulliken_charges" in result and result["mulliken_charges"]:
            ml_entry["mulliken_charges"] = result["mulliken_charges"]
        
        if "loewdin_charges" in result and result["loewdin_charges"]:
            ml_entry["loewdin_charges"] = result["loewdin_charges"]
        
        ml_data.append(ml_entry)
    
    return ml_data

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    logger.info("ORCA PFAS Wrapper Configuration:")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Functional: {args.functional}")
    logger.info(f"Basis set: {args.basis_set}")
    logger.info(f"Number of processors per job: {args.num_procs}")
    logger.info(f"Memory per job: {args.memory} MB")
    logger.info(f"Maximum concurrent jobs: {args.max_jobs}")
    logger.info(f"SMILES column: {args.smiles_col}")
    logger.info(f"ID column: {args.id_col}")
    
    if args.orca_path:
        logger.info(f"ORCA path: {args.orca_path}")
    
    if args.openmpi_path:
        logger.info(f"OpenMPI path: {args.openmpi_path}")
    
    if args.mock:
        logger.info("Running in MOCK mode (no actual ORCA calculations)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    try:
        logger.info(f"Loading input data from {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded {len(df)} compounds")
    except Exception as e:
        logger.error(f"Error loading input data: {str(e)}")
        sys.exit(1)
    
    # Verify required columns
    if args.smiles_col not in df.columns:
        logger.error(f"SMILES column '{args.smiles_col}' not found in input data")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    if args.id_col not in df.columns:
        logger.warning(f"ID column '{args.id_col}' not found in input data. Using index as ID.")
        df['compound_id'] = [f"Compound{i}" for i in range(len(df))]
        args.id_col = 'compound_id'
    
    # Process molecules
    logger.info(f"Processing {len(df)} molecules")
    
    # Determine optimization and charge settings
    optimize = not args.sp_only
    if args.optimize:
        optimize = True
    
    # Process molecules
    results = process_molecules_parallel(
        df, args.output_dir,
        args.functional, args.basis_set,
        args.num_procs, args.memory, args.max_jobs,
        args.orca_path, args.openmpi_path,
        optimize, args.charges, args.sp_only, args.mock,
        args.smiles_col, args.id_col
    )
    
    # Prepare results summary
    results_df = pd.DataFrame([
        {
            "compound_id": r["molecule_id"],
            "smiles": r["smiles"],
            "calculation_success": r.get("success", False),
            "energy": r.get("energy"),
            "dipole_moment": r.get("dipole_moment", {}).get("total") if r.get("dipole_moment") else None,
            "homo_lumo_gap_ev": r.get("homo_lumo_gap_ev"),
            "error": r.get("error"),
            "functional": args.functional,
            "basis_set": args.basis_set
        }
        for r in results
    ])
    
    # Save results summary
    summary_file = os.path.join(args.output_dir, "orca_results_summary.csv")
    results_df.to_csv(summary_file, index=False)
    logger.info(f"Saved results summary to {summary_file}")
    
    # Prepare and save ML training data
    ml_data = prepare_ml_data(results)
    ml_data_file = os.path.join(args.output_dir, "ml_training_data.json")
    with open(ml_data_file, "w") as f:
        json.dump(ml_data, f, indent=2)
    logger.info(f"Saved ML training data to {ml_data_file}")
    
    # Summary
    success_count = sum(1 for r in results if r.get("success", False))
    logger.info(f"Completed {len(results)} calculations")
    logger.info(f"Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    logger.info("ORCA wrapper completed successfully")

if __name__ == "__main__":
    main()