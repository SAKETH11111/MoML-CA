import os
import json
import pandas as pd
from typing import Dict, Any, List

from rdkit import Chem

from moml.simulation.molecular_dynamics.force_field.mapper import ForceFieldMapper


def safe_parse_orca_output(path: str) -> Dict[str, Any]:
    """Lightweight ORCA output parser extracting geometry and Mulliken charges.

    The full parser in :mod:`moml.simulation.qm.parser.orca_parser` # Updated reference
    uses a complex regular expression for the dipole moment section which can be
    very slow on large output files.  For the purpose of generating force-field
    labels we only need the final Mulliken charges and the optimised geometry.
    This helper parses those sections using a simple line based approach.
    
    Args:
        path: Path to the ORCA output file
        
    Returns:
        Dictionary containing parsed data, or empty structure if file cannot be read
    """

    data: Dict[str, Any] = {"mulliken_charges": [], "optimized_geometry": []}

    try:
        with open(path, "r") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print(f"Warning: ORCA output file not found: {path}")
        return data
    except IOError as e:
        print(f"Warning: Failed to read ORCA output file {path}")
        return data

    # Mulliken charges -- use the last occurrence in the file
    last_idx = -1
    for i, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line:
            last_idx = i
    if last_idx != -1:
        charges: List[float] = []
        i = last_idx + 2  # skip header and dashed line
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("Sum of atomic charges"):
                break
            parts = line.split()
            try:
                charges.append(float(parts[-1]))
            except (ValueError, IndexError):
                break
            i += 1
        data["mulliken_charges"] = charges

    # Optimized geometry -- take the last Cartesian coordinate block
    last_idx = -1
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            last_idx = i
    if last_idx != -1:
        atoms = []
        i = last_idx + 2
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("-") or line.startswith("CARTESIAN"):
                break
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    break
                atoms.append({"symbol": symbol, "coordinates": [x, y, z]})
            i += 1
        data["optimized_geometry"] = atoms

    return data


def load_smiles_map(csv_path: str) -> Dict[str, str]:
    """
    Loads a mapping from molecule IDs to SMILES strings from a CSV file.
    
    The CSV file must contain columns named 'DTXSID' and 'SMILES'. Returns a dictionary mapping each 'DTXSID' value (as a string) to its corresponding 'SMILES' string.
    """
    df = pd.read_csv(csv_path)
    
    required_columns = ['DTXSID', 'SMILES']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise KeyError(
            f"Missing required columns in CSV file '{csv_path}': {', '.join(missing_cols)}. "
            "Expected columns: 'DTXSID', 'SMILES'."
        )
    
    return dict(zip(df['DTXSID'].astype(str), df['SMILES']))


def create_mol_from_orca_geom(geom, smiles: str) -> Chem.Mol:
    """
    Constructs an RDKit molecule from a SMILES string and assigns 3D coordinates from an ORCA geometry.
    
    Args:
        geom: List of atom dictionaries with 'coordinates' (x, y, z) from ORCA output.
        smiles: The SMILES string representing the molecule.
    
    Returns:
        An RDKit molecule with explicit hydrogens and a conformer set to the provided geometry.
    
    Raises:
        ValueError: If the SMILES string is invalid or if the atom count does not match the geometry.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to create molecule from SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    if len(geom) != mol.GetNumAtoms():
        raise ValueError(
            f"Geometry atom count ({len(geom)}) does not match SMILES atoms ({mol.GetNumAtoms()})"
        )

    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, atom in enumerate(geom):
        x, y, z = atom["coordinates"]
        conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(x, y, z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def generate_labels(orca_dir: str, smiles_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Generates force field parameter labels for molecules using ORCA output files and a SMILES mapping.
    
    Iterates over ORCA output files in a directory, extracts Mulliken charges and optimized geometries, constructs RDKit molecules, and generates force field parameters using a force field mapper. Parameter keys are converted to string format for JSON serialization. Molecules without geometry or SMILES are skipped.
    
    Args:
        orca_dir: Path to the directory containing ORCA output files.
        smiles_map: Dictionary mapping molecule IDs to SMILES strings.
    
    Returns:
        A dictionary mapping molecule IDs to their generated force field parameters.
    """
    mapper = ForceFieldMapper()
    labels = {}
    for fname in os.listdir(orca_dir):
        if not fname.endswith('.out'):
            continue
        mol_id = os.path.splitext(fname)[0]
        out_path = os.path.join(orca_dir, fname)
        data = safe_parse_orca_output(out_path)
        charges = data.get('mulliken_charges', [])
        geom = data.get('optimized_geometry', [])
        if not geom:
            continue
        smiles = smiles_map.get(mol_id)
        if not smiles:
            print(f'SMILES not found for {mol_id}, skipping')
            continue
        mol = create_mol_from_orca_geom(geom, smiles)
        try:
            params = mapper.generate_force_field_parameters(mol, partial_charges=charges)
            # Convert tuple keys to strings for JSON serialization
            params["bonds"] = {"-".join(map(str, k)): v for k, v in params.get("bonds", {}).items()}
            params["angles"] = {"-".join(map(str, k)): v for k, v in params.get("angles", {}).items()}
            params["dihedrals"] = {"-".join(map(str, k)): v for k, v in params.get("dihedrals", {}).items()}
            labels[mol_id] = params
        except Exception as e:
            print(f'Failed for {mol_id}: {e}')
    return labels


def main():
    """
    Runs the workflow to generate force field parameter labels from ORCA output files.
    
    Loads a mapping of molecule IDs to SMILES strings from a CSV file, processes ORCA output files to extract molecular geometries and charges, generates force field parameters, and writes the results to a JSON file.
    """
    csv_path = os.path.join('data', 'processed', 'chemical_list', 'PFAS_Chemical_List_cleaned.csv')
    orca_dir = 'orca_results_b3lyp_sto3g'
    smiles_map = load_smiles_map(csv_path)
    labels = generate_labels(orca_dir, smiles_map)
    out_file = os.path.join(orca_dir, 'force_field_labels.json')
    with open(out_file, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f'Wrote labels for {len(labels)} molecules to {out_file}')


if __name__ == '__main__':
    main() 
