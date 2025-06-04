import os
import json
import pandas as pd
from typing import Dict, Any

from rdkit import Chem

from moml.simulation.quantum_mechanics.parser.orca_parser import parse_orca_output
from moml.simulation.molecular_dynamics.force_field_mapper import ForceFieldMapper


def load_smiles_map(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    return dict(zip(df['DTXSID'].astype(str), df['SMILES']))


def create_mol_from_orca_geom(geom):
    xyz_block = f"{len(geom)}\n\n"
    for atom in geom:
        symbol = atom['symbol']
        x, y, z = atom['coordinates']
        xyz_block += f"{symbol} {x} {y} {z}\n"
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError('Failed to create molecule from ORCA geometry')
    return mol


def generate_labels(orca_dir: str, smiles_map: Dict[str, str]) -> Dict[str, Any]:
    mapper = ForceFieldMapper()
    labels = {}
    for fname in os.listdir(orca_dir):
        if not fname.endswith('.out'):
            continue
        mol_id = os.path.splitext(fname)[0]
        out_path = os.path.join(orca_dir, fname)
        data = parse_orca_output(out_path)
        charges = data.get('mulliken_charges', [])
        geom = data.get('optimized_geometry', [])
        if not geom:
            continue
        mol = create_mol_from_orca_geom(geom)
        try:
            params = mapper.generate_force_field_parameters(mol, partial_charges=charges)
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