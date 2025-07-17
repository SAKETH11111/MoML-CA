"""
scripts/generate_force_field_labels.py

This script generates force field parameter labels from ORCA QM calculations.

It performs the following steps:
1.  Parses ORCA output files (.out) to extract the final optimized geometry
    and Mulliken partial charges for a set of molecules.
2.  Loads a CSV file that maps molecule identifiers (e.g., DTXSID) to their
    corresponding SMILES strings.
3.  For each molecule, it reconstructs an RDKit molecule object from its SMILES
    string and assigns the 3D coordinates from the ORCA output.
4.  Uses a `ForceFieldMapper` to generate bond, angle, and dihedral parameters
    based on the molecule's topology and the calculated partial charges.
5.  Aggregates the generated labels for all molecules and saves them to a
    single structured JSON file.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdGeometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.simulation.molecular_dynamics.force_field.mapper import ForceFieldMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def safe_parse_orca_output(path: Path) -> Dict[str, List[Any]]:
    """
    Lightweight ORCA output parser for geometry and Mulliken charges.

    This parser is optimized to quickly extract only the final geometry and
    Mulliken charges, which is faster than using the full parser for this specific task.

    Args:
        path: Path to the ORCA output file.

    Returns:
        A dictionary containing the parsed data.
    """
    data: Dict[str, List[Any]] = {"mulliken_charges": [], "optimized_geometry": []}
    if not path.exists():
        logger.warning(f"ORCA output file not found: {path}")
        return data

    try:
        content = path.read_text()
        
        # Find last occurrence of Mulliken charges
        charge_section = content.rfind("MULLIKEN ATOMIC CHARGES")
        if charge_section != -1:
            charges_text = content[charge_section:].splitlines()[2:]
            for line in charges_text:
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    data["mulliken_charges"].append(float(parts[-1]))
                else:
                    break
        
        # Find last occurrence of Cartesian coordinates
        geom_section = content.rfind("CARTESIAN COORDINATES (ANGSTROEM)")
        if geom_section != -1:
            geom_text = content[geom_section:].splitlines()[2:]
            for line in geom_text:
                parts = line.split()
                if len(parts) >= 4:
                    data["optimized_geometry"].append({
                        "symbol": parts[0], "coordinates": list(map(float, parts[1:4]))
                    })
                elif not parts: # Stop at empty line
                    break

    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}", exc_info=True)
    
    return data


def load_smiles_map(csv_path: Path) -> Dict[str, str]:
    """Loads a mapping from molecule IDs to SMILES strings from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required = ["DTXSID", "SMILES"]
        if not all(col in df.columns for col in required):
            raise KeyError(f"Missing required columns: {', '.join(required)}")
        return dict(zip(df["DTXSID"].astype(str), df["SMILES"]))
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error loading SMILES map from {csv_path}: {e}")
        raise


def create_mol_from_geom(geom: List[Dict[str, Any]], smiles: str) -> Optional[Chem.Mol]:
    """Constructs an RDKit molecule from SMILES and assigns 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not create molecule from SMILES: {smiles}")
        return None
    mol = Chem.AddHs(mol)

    if len(geom) != mol.GetNumAtoms():
        logger.warning(f"Atom count mismatch for {smiles}: {len(geom)} vs {mol.GetNumAtoms()}")
        return None

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, atom_data in enumerate(geom):
        pos = rdGeometry.Point3D(*atom_data["coordinates"])
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf, assignId=True)
    return mol


def generate_labels(orca_dir: Union[str, Path], smiles_map: Dict[str, str]) -> Dict[str, Any]:
    """Generates force field parameter labels for all molecules.

    Args:
        orca_dir: Directory containing ORCA ``.out`` files. Can be a ``str`` or ``Path``.
        smiles_map: Mapping from molecule identifier (file stem) to SMILES string.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary of force-field parameters keyed by molecule id.
    """
    orca_dir = Path(orca_dir)

    mapper = ForceFieldMapper()
    labels: Dict[str, Any] = {}

    for out_file in orca_dir.glob("*.out"):
        mol_id = out_file.stem
        smiles = smiles_map.get(mol_id)
        if not smiles:
            logger.warning(f"No SMILES found for {mol_id}, skipping.")
            continue

        data = safe_parse_orca_output(out_file)
        if not data["optimized_geometry"]:
            logger.warning(f"No geometry found for {mol_id}, skipping.")
            continue

        mol = create_mol_from_geom(data["optimized_geometry"], smiles)
        if not mol:
            continue

        try:
            params = mapper.generate_force_field_parameters(
                mol, partial_charges=data["mulliken_charges"]
            )
            # Convert tuple keys to strings for JSON serialization
            params["bonds"] = {"-".join(map(str, k)): v for k, v in params.get("bonds", {}).items()}
            params["angles"] = {"-".join(map(str, k)): v for k, v in params.get("angles", {}).items()}
            params["dihedrals"] = {"-".join(map(str, k)): v for k, v in params.get("dihedrals", {}).items()}
            labels[mol_id] = params
            logger.info(f"Successfully generated labels for {mol_id}")
        except Exception as e:
            logger.error(f"Failed to generate parameters for {mol_id}: {e}", exc_info=True)
            
    return labels


def main():
    """Main function to run the force field label generation workflow."""
    parser = argparse.ArgumentParser(
        description="Generate force field parameter labels from ORCA output files."
    )
    parser.add_argument(
        "--orca_dir", type=str, required=True, help="Directory containing ORCA output files (.out)."
    )
    parser.add_argument(
        "--smiles_csv", type=str, required=True, help="CSV file mapping DTXSID to SMILES."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file for labels."
    )
    args = parser.parse_args()

    try:
        smiles_map = load_smiles_map(Path(args.smiles_csv))
        labels = generate_labels(Path(args.orca_dir), smiles_map)
        
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labels, f, indent=2)
        
        logger.info(f"Wrote labels for {len(labels)} molecules to {output_path}")

    except (FileNotFoundError, KeyError) as e:
        logger.error(f"A critical error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
