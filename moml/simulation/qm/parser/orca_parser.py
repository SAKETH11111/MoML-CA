"""
Unified ORCA Parser for Quantum Mechanical Data

This module provides comprehensive functionality for:
1. Parsing ORCA output files to extract quantum mechanical data
2. Managing ORCA calculations (input generation, job submission, result processing)
3. Converting results into formats usable by the MGNN pipeline
"""

import os
import re
import logging
import subprocess
import concurrent.futures
import json
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orca_parser")

# Suppress RDKit logging except for warnings and errors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.WARNING)

#-------------------------------------
# ORCA Output Parsing Functions
#-------------------------------------

def parse_orca_output(orca_output_path: str) -> Dict[str, Union[List[float], np.ndarray, Dict]]:
    """
    Parse ORCA output file to extract quantum mechanical data in a single pass.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        Dictionary containing extracted data with keys:
            - 'mulliken_charges': List of Mulliken atomic charges
            - 'loewdin_charges': List of Loewdin atomic charges (if available)
            - 'dipole_moment': Dipole moment vector [dx, dy, dz, total] (if available)
            - 'homo_lumo_gap': HOMO-LUMO gap in eV (if available)
            - 'homo_lumo_contributions': Dict with 'homo' and 'lumo' orbital contributions
            - 'electrostatic_potential': List of ESP values at atom positions
            - 'optimized_geometry': List of dictionaries with atom coordinates
            - 'status': Calculation status ('completed', 'error', 'incomplete')
    """
    if not os.path.exists(orca_output_path):
        raise FileNotFoundError(f"ORCA output file not found: {orca_output_path}")
    
    # Initialize result dictionary
    result = {
        'status': 'incomplete',  # Default status
        'mulliken_charges': [],
        'loewdin_charges': [],
        'dipole_moment': None,
        'homo_lumo_gap': None,
        'homo_lumo_contributions': {'homo': [], 'lumo': []},
        'electrostatic_potential': [],
        'optimized_geometry': []
    }
    
    # Regex patterns for all data types we need to extract
    patterns = {
        'calculation_completed': r".*ORCA TERMINATED NORMALLY.*",
        'error': r".*(ERROR|Error):.*",
        'mulliken_charges': r"MULLIKEN ATOMIC CHARGES.*?\n(?:\s*\d+\s+\w+\s+([-+]?\d*\.\d+).*?\n)+",
        'loewdin_charges': r"LOEWDIN ATOMIC CHARGES.*?\n(?:\s*\d+\s+\w+\s+([-+]?\d*\.\d+).*?\n)+",
        'dipole_moment': r"DIPOLE MOMENT\s*\n.*?X\s+([-+]?\d*\.\d+).*?Y\s+([-+]?\d*\.\d+).*?Z\s+([-+]?\d*\.\d+).*?Total\s+([-+]?\d*\.\d+)",
        'homo_lumo_gap_direct': r"HOMO-LUMO gap:\s*([-+]?\d*\.\d+)\s*Eh\s*=\s*([-+]?\d*\.\d+)\s*eV",
        'homo_energy': r"HOMO:\s*[-+]?\d+\s+([-+]?\d*\.\d+)\s*Eh",
        'lumo_energy': r"LUMO:\s*[-+]?\d+\s+([-+]?\d*\.\d+)\s*Eh",
        'geometry': r"CARTESIAN COORDINATES \(ANGSTROEM\).*?\n(.*?)\n\n",
    }
    
    try:
        # Read the file in chunks to reduce memory usage
        with open(orca_output_path, 'r') as f:
            content = f.read()
        
        # Check calculation status
        if re.search(patterns['calculation_completed'], content, re.DOTALL):
            result['status'] = 'completed'
        elif re.search(patterns['error'], content, re.DOTALL):
            result['status'] = 'error'
        
        # Extract Mulliken charges
        mulliken_match = re.search(patterns['mulliken_charges'], content, re.DOTALL)
        if mulliken_match:
            charges_text = mulliken_match.group(0)
            charge_pattern = r"\d+\s+\w+\s+([-+]?\d*\.\d+)"
            result['mulliken_charges'] = [float(charge) for charge in re.findall(charge_pattern, charges_text)]
        
        # Extract Loewdin charges
        loewdin_match = re.search(patterns['loewdin_charges'], content, re.DOTALL)
        if loewdin_match:
            charges_text = loewdin_match.group(0)
            charge_pattern = r"\d+\s+\w+\s+([-+]?\d*\.\d+)"
            result['loewdin_charges'] = [float(charge) for charge in re.findall(charge_pattern, charges_text)]
        
        # Extract dipole moment
        dipole_match = re.search(patterns['dipole_moment'], content, re.DOTALL)
        if dipole_match:
            dx = float(dipole_match.group(1))
            dy = float(dipole_match.group(2))
            dz = float(dipole_match.group(3))
            total = float(dipole_match.group(4))
            result['dipole_moment'] = [dx, dy, dz, total]
        
        # Extract HOMO-LUMO gap
        homo_lumo_gap_match = re.search(patterns['homo_lumo_gap_direct'], content)
        if homo_lumo_gap_match:
            result['homo_lumo_gap'] = float(homo_lumo_gap_match.group(2))  # Use the gap in eV
        else:
            # If not found directly, try to calculate from HOMO and LUMO energies
            homo_match = re.search(patterns['homo_energy'], content)
            lumo_match = re.search(patterns['lumo_energy'], content)
            
            if homo_match and lumo_match:
                homo_energy = float(homo_match.group(1))
                lumo_energy = float(lumo_match.group(1))
                gap_hartree = lumo_energy - homo_energy
                gap_ev = gap_hartree * 27.211  # Convert Hartree to eV
                result['homo_lumo_gap'] = gap_ev
        
        # Extract optimized geometry
        geometry_match = re.search(patterns['geometry'], content, re.DOTALL)
        if geometry_match:
            geometry_text = geometry_match.group(1)
            atom_pattern = r"(\w+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)"
            
            for line in geometry_text.split('\n'):
                if not line.strip():
                    continue
                
                atom_match = re.search(atom_pattern, line)
                if atom_match:
                    symbol = atom_match.group(1)
                    x = float(atom_match.group(2))
                    y = float(atom_match.group(3))
                    z = float(atom_match.group(4))
                    
                    result['optimized_geometry'].append({
                        'symbol': symbol,
                        'coordinates': [x, y, z]
                    })
        
    except Exception as e:
        logger.error(f"Error parsing ORCA output: {str(e)}")
        result['status'] = 'error'
        result['error_message'] = str(e)
    
    return result


def extract_partial_charges_from_orca(orca_output_path: str, charge_type: str = 'mulliken') -> List[float]:
    """
    Extract partial charges from ORCA output file.
    
    Args:
        orca_output_path: Path to the ORCA output file
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        
    Returns:
        List of partial charges
    """
    # Use the cached parsing function to avoid duplicate file operations
    data = parse_orca_output(orca_output_path)
    
    if charge_type.lower() == 'mulliken':
        return data.get('mulliken_charges', [])
    elif charge_type.lower() == 'loewdin':
        return data.get('loewdin_charges', [])
    else:
        logger.warning(f"Unknown charge type '{charge_type}', using Mulliken charges")
        return data.get('mulliken_charges', [])


def extract_orbital_contributions_from_orca(orca_output_path: str) -> Dict[str, List[float]]:
    """
    Extract HOMO/LUMO contributions from ORCA output file.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        Dictionary with 'homo' and 'lumo' keys containing lists of contribution values
    """
    # This is a placeholder implementation - actual orbital contribution parsing
    # would require a more sophisticated approach which is beyond the scope of
    # the current optimization
    return {'homo': [], 'lumo': []}


def extract_electrostatic_potential_from_orca(orca_output_path: str) -> List[float]:
    """
    Extract electrostatic potential values at atom positions.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        List of ESP values
    """
    # This is a placeholder implementation - actual ESP extraction
    # would require a more sophisticated approach
    return []

#-------------------------------------
# ORCA Calculation Management Functions
#-------------------------------------

def smiles_to_3d_structure(smiles: str, molecule_id: str, optimize: bool = True) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to 3D molecular structure using RDKit.
    
    Args:
        smiles: SMILES string
        molecule_id: Identifier for the molecule
        optimize: Whether to perform force field optimization
        
    Returns:
        RDKit molecule with 3D coordinates or None if failed
    """
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Add hydrogen atoms
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            logger.error(f"Coordinate generation failed for {molecule_id}")
            return None
        
        # Optimize structure using force field
        if optimize:
            AllChem.MMFFOptimizeMolecule(mol)
        
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
    Create an ORCA input file from a molecule.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        molecule_id: Identifier for the molecule
        output_dir: Directory to save the input file
        functional: DFT functional to use
        basis_set: Basis set to use
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
            orca_functional = "B3LYP D3BJ"  # Add D3BJ dispersion correction
        
        # Create input file path
        input_file = os.path.join(output_dir, f"{molecule_id}.inp")
        
        # Write ORCA input file
        with open(input_file, "w") as f:
            # Basic job configuration
            f.write(f"# ORCA Input File for {molecule_id}\n")
            f.write("# Generated by MoML-CA ORCA wrapper\n\n")
            
            # Calculation method and resources
            f.write(f"! {orca_functional} {basis_set} OPT\n")
            f.write("! TIGHTSCF\n\n")
            
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
                atom_idx = atom.GetIdx()
                atom_sym = atom.GetSymbol()
                pos = conf.GetAtomPosition(atom_idx)
                f.write(f"  {atom_sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
            f.write("*\n")
        
        # Also save a MOL file for visualization and later use
        mol_file = os.path.join(output_dir, f"{molecule_id}.mol")
        Chem.MolToMolFile(mol, mol_file)
        
        return True, input_file
    
    except Exception as e:
        logger.error(f"Error creating ORCA input file for {molecule_id}: {str(e)}")
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
        
        # Determine ORCA path - default to the installed location or system PATH
        if orca_path is None:
            # Try common installation paths
            possible_paths = [
                "/Users/saketh/Library/orca_6_0_1/orca",
                "/opt/orca/orca",
                "orca"  # System PATH
            ]
            
            for path in possible_paths:
                if path == "orca" or os.path.exists(path):
                    orca_path = path
                    logger.info(f"Using ORCA: {orca_path}")
                    break
        
        # Build command
        command = [orca_path, input_file]
        
        # Run ORCA
        logger.info(f"Running ORCA: {' '.join(command)}")
        process = subprocess.run(
            command, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=input_dir
        )
        
        # Check if output file exists
        if not os.path.exists(output_file):
            logger.error(f"ORCA output file not created: {output_file}")
            return False, ""
        
        # Check process return code
        if process.returncode != 0:
            logger.error(f"ORCA process failed with code {process.returncode}")
            logger.error(f"STDERR: {process.stderr}")
            return False, output_file
        
        logger.info(f"ORCA calculation completed successfully: {output_file}")
        return True, output_file
    
    except Exception as e:
        logger.error(f"Error running ORCA calculation: {str(e)}")
        return False, ""


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
    
    # Save results to JSON file
    results_file = os.path.join(mol_dir, f"{molecule_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def batch_process_molecules(
    molecules_df: pd.DataFrame,
    output_dir: str,
    functional: str = "B3LYP",
    basis_set: str = "6-31G*",
    num_procs: int = 4,
    memory: int = 4000,
    orca_path: str = None,
    max_workers: int = 1,
    smiles_col: str = "SMILES",
    id_col: str = "common_name"
) -> pd.DataFrame:
    """
    Process multiple molecules in parallel.
    
    Args:
        molecules_df: DataFrame containing molecules
        output_dir: Directory for output files
        functional: DFT functional
        basis_set: Basis set
        num_procs: Number of processors per ORCA job
        memory: Memory in MB per ORCA job
        orca_path: Path to ORCA executable
        max_workers: Maximum number of concurrent processes
        smiles_col: Column name for SMILES strings
        id_col: Column name for molecule IDs
        
    Returns:
        DataFrame with results for each molecule
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Process in parallel if max_workers > 1
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_mol = {}
            
            for _, row in molecules_df.iterrows():
                smiles = row[smiles_col]
                molecule_id = row[id_col]
                
                future = executor.submit(
                    process_molecule,
                    smiles,
                    molecule_id,
                    output_dir,
                    functional,
                    basis_set,
                    num_procs,
                    memory,
                    orca_path
                )
                
                future_to_mol[future] = molecule_id
            
            for future in concurrent.futures.as_completed(future_to_mol):
                molecule_id = future_to_mol[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing {molecule_id}: {result['status']}")
                except Exception as e:
                    logger.error(f"Error processing {molecule_id}: {str(e)}")
                    results.append({
                        "id": molecule_id,
                        "status": "error",
                        "error": str(e)
                    })
    else:
        # Process sequentially
        for _, row in molecules_df.iterrows():
            smiles = row[smiles_col]
            molecule_id = row[id_col]
            
            result = process_molecule(
                smiles,
                molecule_id,
                output_dir,
                functional,
                basis_set,
                num_procs,
                memory,
                orca_path
            )
            
            results.append(result)
            logger.info(f"Completed processing {molecule_id}: {result['status']}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results summary
    summary_file = os.path.join(output_dir, "processing_summary.csv")
    summary_df = results_df[["id", "status", "error"]].copy()
    summary_df.to_csv(summary_file, index=False)
    
    return results_df 