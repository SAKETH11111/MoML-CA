#!/usr/bin/env python3
"""
ORCA QM Wrapper for PFAS Compounds

This script handles the end-to-end process of:
1. Reading SMILES strings from dataset
2. Converting to 3D structures using RDKit
3. Creating ORCA input files
4. Managing job submissions
5. Collecting and processing results

The module is designed to generate quantum mechanical data for PFAS compounds
that can be used for training machine learning models, particularly MGNN.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import concurrent.futures
from datetime import datetime
import json

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orca_pfas_wrapper")

# Suppress RDKit logging except for warnings and errors
RDLogger.logger().setLevel(RDLogger.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORCA QM wrapper for PFAS compounds")
    
    parser.add_argument(
        "--input_csv", 
        type=str, 
        required=True,
        help="Path to CSV file with SMILES data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory for ORCA input/output files"
    )
    
    parser.add_argument(
        "--functional", 
        type=str, 
        choices=["B3LYP", "wB97X-D"], 
        default="B3LYP",
        help="DFT functional to use"
    )
    
    parser.add_argument(
        "--basis_set", 
        type=str, 
        default="6-31G*",
        help="Basis set for calculations"
    )
    
    parser.add_argument(
        "--num_procs", 
        type=int, 
        default=4,
        help="Number of processors for ORCA"
    )
    
    parser.add_argument(
        "--memory", 
        type=int, 
        default=4000,
        help="Memory (in MB) for ORCA"
    )
    
    parser.add_argument(
        "--max_jobs", 
        type=int, 
        default=1,
        help="Maximum number of concurrent ORCA jobs"
    )
    
    parser.add_argument(
        "--smiles_column", 
        type=str, 
        default="SMILES",
        help="Column name containing SMILES strings"
    )
    
    parser.add_argument(
        "--id_column", 
        type=str, 
        default="ID",
        help="Column name containing compound identifiers"
    )
    
    parser.add_argument(
        "--orca_path", 
        type=str, 
        default=None,
        help="Path to ORCA executable (if not in PATH)"
    )
    
    parser.add_argument(
        "--openmpi_path", 
        type=str, 
        default=None,
        help="Path to OpenMPI bin directory (for Mac ARM64)"
    )
    
    return parser.parse_args()


def smiles_to_3d_structure(smiles: str, molecule_id: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to 3D structure using RDKit.
    
    Args:
        smiles: SMILES string of the molecule
        molecule_id: Identifier for the molecule
        
    Returns:
        RDKit molecule with 3D coordinates, or None if conversion fails
    """
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Failed to create molecule from SMILES: {smiles}")
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Run force field optimization to get realistic coordinates
        AllChem.MMFFOptimizeMolecule(mol)
        
        logger.info(f"Successfully generated 3D structure for molecule {molecule_id}")
        return mol
    
    except Exception as e:
        logger.error(f"Error creating 3D structure for {molecule_id}: {str(e)}")
        return None


def create_orca_input(
    mol: Chem.Mol,
    molecule_id: str,
    output_dir: str,
    functional: str = "B3LYP",
    basis_set: str = "6-31G*",
    num_procs: int = 4,
    memory: int = 4000
) -> Tuple[bool, str]:
    """
    Create ORCA input file for a molecule.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        molecule_id: Identifier for the molecule
        output_dir: Directory to write input files
        functional: DFT functional (B3LYP or wB97X-D). B3LYP will use D3BJ dispersion correction
                   for improved treatment of noncovalent interactions.
        basis_set: Basis set for calculation
        num_procs: Number of processors to use
        memory: Memory in MB
        
    Returns:
        Tuple of (success_status, input_file_path)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Fix functional name for ORCA (wB97X-D -> wB97X-D3, add D3BJ for B3LYP)
        orca_functional = functional
        if functional == "wB97X-D":
            orca_functional = "wB97X-D3"
        elif functional == "B3LYP":
            orca_functional = "B3LYP D3BJ"  # Add D3BJ dispersion correction for noncovalent interactions
        
        # Create input file path
        input_file = os.path.join(output_dir, f"{molecule_id}.inp")
        
        # Write ORCA input file
        with open(input_file, "w") as f:
            # Basic job configuration
            f.write(f"# ORCA Input File for {molecule_id}\n")
            f.write("# Generated by MoML-CA ORCA wrapper\n\n")
            
            # Calculation method and resources
            f.write(f"! {orca_functional} {basis_set} OPT\n")  # Removed FREQ
            f.write("! TIGHTSCF\n\n")  # Removed RIJCOSX which requires parallel processing
            
            # Parallel settings - only add if using more than 1 proc
            if num_procs > 1:
                f.write("%pal\n")
                f.write(f"  nprocs {num_procs}\n")
                f.write("end\n\n")
            
            # Memory settings
            f.write(f"%maxcore {memory}\n\n")
            
            # Calculation setup
            f.write("%scf\n")
            f.write("  MaxIter 250\n")
            f.write("  Convergence Tight\n")
            f.write("end\n\n")
            
            # Geometry optimization settings
            f.write("%geom\n")
            f.write("  MaxIter 250\n")
            f.write("  Convergence Tight\n")
            f.write("end\n\n")
            
            # Simplified output options
            f.write("%output\n")
            f.write("  PrintLevel Normal\n")
            f.write("end\n\n")
            
            # Molecular coordinates - XYZ format
            f.write("* xyz 0 1\n")
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = conf.GetAtomPosition(idx)
                symbol = atom.GetSymbol()
                f.write(f"  {symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
            f.write("*\n")
        
        logger.info(f"Created ORCA input file: {input_file}")
        return True, input_file
    
    except Exception as e:
        logger.error(f"Error creating ORCA input file: {str(e)}")
        return False, ""


def run_orca_calculation(input_file: str, orca_path: str = None, num_procs: int = 4) -> Tuple[bool, str]:
    """
    Run an ORCA calculation.
    
    Args:
        input_file: Path to ORCA input file
        orca_path: Path to ORCA executable. If None, will try to use system PATH.
        num_procs: Number of processors to use
        
    Returns:
        Tuple of (success_status, output_file_path)
    """
    try:
        input_dir = os.path.dirname(input_file)
        base_name = os.path.basename(input_file).replace(".inp", "")
        output_file = os.path.join(input_dir, f"{base_name}.out")
        
        # Determine ORCA path - default to the installed location
        if orca_path is None:
            orca_path = "/Users/saketh/Library/orca_6_0_1/orca"
            
            if os.path.exists(orca_path):
                logger.info(f"Using installed ORCA: {orca_path}")
            else:
                orca_path = "orca"  # Fallback to system PATH
                logger.info("Using ORCA from system PATH")
        
        # For single processor mode, run ORCA directly without shell redirection
        if num_procs <= 1:
            # Build direct command to run ORCA
            cmd = [orca_path, input_file]
            
            logger.info(f"Running ORCA in single processor mode: {' '.join(cmd)}")
            
            # Run ORCA directly but capture output
            with open(output_file, "w") as out_f:
                process = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    check=False
                )
        else:
            # For parallel mode, use mpirun
            cmd = ["mpirun", "-np", str(num_procs), orca_path, input_file]
            
            logger.info(f"Running ORCA in parallel mode: {' '.join(cmd)}")
            
            # Run with mpirun
            with open(output_file, "w") as out_f:
                process = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    check=False
                )
        
        # Check if calculation was successful
        if process.returncode != 0:
            logger.error(f"ORCA calculation failed with return code {process.returncode}")
            stderr_output = process.stderr.decode('utf-8')
            if stderr_output:
                logger.error(f"stderr: {stderr_output}")
            return False, output_file
        
        logger.info(f"ORCA calculation completed successfully: {output_file}")
        return True, output_file
    
    except Exception as e:
        logger.error(f"Error running ORCA calculation: {str(e)}")
        return False, ""


def parse_orca_output(output_file: str) -> Dict[str, Any]:
    """
    Parse ORCA output file to extract relevant data.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Dictionary containing extracted properties
    """
    try:
        results = {
            "filename": output_file,
            "status": "failed",
            "energy": None,
            "optimized_geometry": [],
            "mulliken_charges": {},
            "orbital_energies": {
                "homo": None,
                "lumo": None,
                "gap": None
            },
            "frequencies": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Read output file
        with open(output_file, "r") as f:
            lines = f.readlines()
        
        # Check if calculation finished
        scf_convergence = False
        opt_convergence = False
        normal_termination = False
        
        # Extract data from output file
        in_final_geom = False
        in_mulliken = False
        in_orbital_energies = False
        in_frequencies = False
        atom_dict = {}
        
        for i, line in enumerate(lines):
            # Check for normal termination
            if "ORCA TERMINATED NORMALLY" in line:
                normal_termination = True
                results["status"] = "completed"
            
            # Check for SCF convergence
            if "SCF CONVERGED" in line:
                scf_convergence = True
            
            # Check for optimization convergence
            if "OPTIMIZATION CONVERGED" in line:
                opt_convergence = True
                results["status"] = "completed"
            
            # Get final SCF energy
            if "FINAL SINGLE POINT ENERGY" in line:
                results["energy"] = float(line.split()[-1])
            
            # Get optimized geometry
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line and i < len(lines) - 5:
                in_final_geom = True
                results["optimized_geometry"] = []  # Reset geometry list
                # Skip the header lines
                coord_start = i + 2
                continue
            
            if in_final_geom and i >= coord_start:
                if "---" in line or len(line.strip()) == 0:
                    in_final_geom = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        atom = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        
                        if atom.isalpha():  # Ensure it's an atom symbol
                            atom_idx = len(atom_dict)
                            atom_dict[atom_idx] = atom
                            results["optimized_geometry"].append({
                                "atom": atom,
                                "coordinates": [x, y, z]
                            })
                    except (ValueError, IndexError):
                        pass
            
            # Get Mulliken charges
            if "MULLIKEN ATOMIC CHARGES" in line:
                in_mulliken = True
                # Skip header line
                continue
            
            if in_mulliken:
                if "SUM OF MULLIKEN ATOMIC CHARGES" in line or len(line.strip()) == 0:
                    in_mulliken = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        atom_idx = int(parts[0])
                        charge = float(parts[2])
                        if atom_idx in atom_dict:
                            results["mulliken_charges"][atom_dict[atom_idx]] = charge
                    except (ValueError, IndexError):
                        pass
            
            # Get HOMO-LUMO gap
            if "ORBITAL ENERGIES" in line:
                in_orbital_energies = True
                homo_energy = None
                lumo_energy = None
                continue
            
            if in_orbital_energies:
                if "HOMO-LUMO GAP" in line:
                    try:
                        gap = float(line.split()[-2])
                        results["orbital_energies"]["gap"] = gap
                    except (ValueError, IndexError):
                        pass
                elif len(line.strip()) == 0 or "----" in line and homo_energy is not None:
                    in_orbital_energies = False
                    continue
                elif "OCC" in line and "E(Eh)" in line:
                    continue
                elif line.strip() and len(line.split()) >= 4:
                    parts = line.split()
                    try:
                        occ = float(parts[1])
                        energy = float(parts[2])
                        if occ == 0 and lumo_energy is None:  # First unoccupied orbital = LUMO
                            lumo_energy = energy
                            results["orbital_energies"]["lumo"] = energy
                        elif occ > 0:  # Last occupied orbital before LUMO = HOMO
                            homo_energy = energy
                            results["orbital_energies"]["homo"] = energy
                    except (ValueError, IndexError):
                        pass
            
            # Get vibrational frequencies
            if "VIBRATIONAL FREQUENCIES" in line:
                in_frequencies = True
                continue
            
            if in_frequencies:
                if "NORMAL MODES" in line or len(line.strip()) == 0:
                    in_frequencies = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        freq = float(parts[1])
                        results["frequencies"].append(freq)
                    except (ValueError, IndexError):
                        pass
        
        # Final status check - if we have energy and normal termination, mark as completed
        if results["energy"] is not None and (normal_termination or scf_convergence):
            results["status"] = "completed"
        
        return results
    
    except Exception as e:
        logger.error(f"Error parsing ORCA output {output_file}: {str(e)}")
        return {"filename": output_file, "status": "error", "error": str(e)}


def process_molecule(
    smiles: str,
    molecule_id: str,
    output_dir: str,
    functional: str,
    basis_set: str,
    num_procs: int,
    memory: int,
    orca_path: str
) -> Dict[str, Any]:
    """
    Process a single molecule through the entire workflow.
    
    Args:
        smiles: SMILES string of the molecule
        molecule_id: Identifier for the molecule
        output_dir: Directory for output files
        functional: DFT functional
        basis_set: Basis set
        num_procs: Number of processors
        memory: Memory in MB
        orca_path: Path to ORCA executable
        
    Returns:
        Dictionary with results and status
    """
    results = {
        "id": molecule_id,
        "smiles": smiles,
        "status": "failed",
        "error": None,
        "data": None
    }
    
    # Create molecule-specific directory
    mol_dir = os.path.join(output_dir, molecule_id)
    os.makedirs(mol_dir, exist_ok=True)
    
    # 1. Convert SMILES to 3D structure
    mol = smiles_to_3d_structure(smiles, molecule_id)
    if mol is None:
        results["error"] = "Failed to create 3D structure"
        return results
    
    # 2. Create ORCA input file
    success, input_file = create_orca_input(
        mol, 
        molecule_id, 
        mol_dir, 
        functional, 
        basis_set,
        num_procs,
        memory
    )
    
    if not success:
        results["error"] = "Failed to create ORCA input file"
        return results
    
    # 3. Run ORCA calculation
    success, output_file = run_orca_calculation(input_file, orca_path, num_procs)
    if not success:
        results["error"] = "ORCA calculation failed"
        return results
    
    # 4. Parse ORCA output
    data = parse_orca_output(output_file)
    results["status"] = data["status"]
    results["data"] = data
    
    # 5. If calculation was successful and we have optimized geometry, create visualization
    if results["status"] == "completed" and data["optimized_geometry"]:
        try:
            # Create a new molecule with optimized geometry
            opt_mol = Chem.RWMol()
            
            # Add atoms to the molecule
            atom_map = {}  # Maps atom indices to RDKit atom indices
            for i, atom_data in enumerate(data["optimized_geometry"]):
                atom_symbol = atom_data["atom"]
                atom = Chem.Atom(atom_symbol)
                atom_idx = opt_mol.AddAtom(atom)
                atom_map[i] = atom_idx
            
            # Add bonds (this is simplified - would need proper bond order detection)
            # For now, we just create a 3D visualization of atoms
            
            # Set 3D coordinates
            conf = Chem.Conformer(len(data["optimized_geometry"]))
            for i, atom_data in enumerate(data["optimized_geometry"]):
                x, y, z = atom_data["coordinates"]
                pos = Chem.rdGeometry.Point3D(x, y, z)
                conf.SetAtomPosition(atom_map[i], pos)
            
            opt_mol.AddConformer(conf)
            
            # Write molecule to file
            vis_file = os.path.join(mol_dir, f"{molecule_id}_optimized.mol")
            Chem.MolToMolFile(opt_mol, vis_file)
            logger.info(f"Created 3D visualization: {vis_file}")
        except Exception as e:
            logger.warning(f"Could not create visualization for {molecule_id}: {str(e)}")
    
    # Save results to JSON file
    results_file = os.path.join(mol_dir, f"{molecule_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main function to run the ORCA wrapper."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up environment for OpenMPI on Mac ARM64 if needed
    if args.openmpi_path:
        logger.info(f"Setting up OpenMPI environment using: {args.openmpi_path}")
        # Add OpenMPI to PATH
        os.environ["PATH"] = f"{args.openmpi_path}:{os.environ.get('PATH', '')}"
        
        # Set LD_LIBRARY_PATH for OpenMPI libraries
        openmpi_lib = os.path.join(os.path.dirname(args.openmpi_path), "lib")
        if os.path.exists(openmpi_lib):
            os.environ["LD_LIBRARY_PATH"] = f"{openmpi_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            logger.info(f"Added OpenMPI libraries: {openmpi_lib}")
    
    # Read input CSV
    try:
        df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded dataset with {len(df)} compounds")
        
        # Check required columns
        if args.smiles_column not in df.columns:
            logger.error(f"SMILES column '{args.smiles_column}' not found in dataset")
            sys.exit(1)
        
        if args.id_column not in df.columns:
            logger.warning(f"ID column '{args.id_column}' not found, using index as ID")
            df[args.id_column] = [f"mol_{i}" for i in range(len(df))]
            
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    # Process molecules
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = []
        
        for _, row in df.iterrows():
            smiles = row[args.smiles_column]
            molecule_id = str(row[args.id_column])
            
            # Skip if SMILES is invalid
            if pd.isna(smiles) or not isinstance(smiles, str) or not smiles:
                logger.warning(f"Skipping invalid SMILES for {molecule_id}")
                continue
            
            # Submit job to process molecule
            future = executor.submit(
                process_molecule,
                smiles,
                molecule_id,
                args.output_dir,
                args.functional,
                args.basis_set,
                args.num_procs,
                args.memory,
                args.orca_path
            )
            
            futures.append(future)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed {result['id']} with status: {result['status']}")
            except Exception as e:
                logger.error(f"Error processing molecule: {str(e)}")
    
    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_file = os.path.join(args.output_dir, "orca_results_summary.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Prepare data for MGNN training
    success_count = sum(1 for result in results if result["status"] == "completed")
    logger.info(f"Successfully completed {success_count}/{len(results)} calculations")
    
    # Extract and format data for ML training
    ml_data = []
    for result in results:
        if result["status"] == "completed" and result["data"] is not None:
            ml_data.append({
                "molecule_id": result["id"],
                "smiles": result["smiles"],
                "total_energy": result["data"]["energy"],
                "homo": result["data"]["orbital_energies"]["homo"],
                "lumo": result["data"]["orbital_energies"]["lumo"],
                "gap": result["data"]["orbital_energies"]["gap"],
                "charges": result["data"]["mulliken_charges"],
                "geometry": result["data"]["optimized_geometry"]
            })
    
    # Save ML-ready data
    if ml_data:
        ml_data_file = os.path.join(args.output_dir, "ml_training_data.json")
        with open(ml_data_file, "w") as f:
            json.dump(ml_data, f, indent=2)
        logger.info(f"ML training data saved to {ml_data_file}")
    else:
        logger.warning("No successful calculations for ML training data")


if __name__ == "__main__":
    main() 