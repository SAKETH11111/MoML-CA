import os
import json
import pandas as pd
from typing import Dict, Any, List

from rdkit import Chem

from moml.simulation.molecular_dynamics.force_field_mapper import ForceFieldMapper


def safe_parse_orca_output(path: str) -> Dict[str, Any]:
    """Lightweight ORCA output parser extracting geometry and Mulliken charges.

    The full parser in :mod:`moml.simulation.quantum_mechanics.parser.orca_parser`
    uses a complex regular expression for the dipole moment section which can be
    very slow on large output files.  For the purpose of generating force-field
    labels we only need the final Mulliken charges and the optimised geometry.
    This helper parses those sections using a simple line based approach.
    """

    data: Dict[str, Any] = {"mulliken_charges": [], "optimized_geometry": []}

    with open(path, "r") as fh:
        lines = fh.readlines()

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
    df = pd.read_csv(csv_path)
    return dict(zip(df['DTXSID'].astype(str), df['SMILES']))


def create_mol_from_orca_geom(geom, smiles: str) -> Chem.Mol:
    """Create an RDKit molecule with geometry from ORCA output."""

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
