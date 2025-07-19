"""
tests/test_3d_structure_generation.py

Tests 3D structure generation from SMILES strings using RDKit, a key step in ORCA input preparation.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

try:
    from rdkit import Chem
except ImportError:
    pytest.skip("RDKit not installed, skipping 3D structure generation tests", allow_module_level=True)

from moml.simulation.qm.parser.orca_parser import smiles_to_3d_structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_3d_generation")


def run_tests() -> bool:
    """
    Run the 3D structure generation tests for a set of SMILES strings.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    test_cases: List[Tuple[str, str]] = [
        ("CC", "Ethane"),
        ("CC(F)(F)F", "Trifluoromethane"),
        ("c1ccccc1", "Benzene"),
        ("C(C(F)(F)F)C(F)(F)F", "Hexafluoroethane"),
        ("C(F)(F)(F)C(F)(F)C(F)(F)F", "Perfluoropropane"),
    ]

    success_count = 0
    failure_count = 0

    print("\n===== 3D Structure Generation Tests =====\n")

    temp_dir = tempfile.TemporaryDirectory()

    for idx, (smiles, name) in enumerate(test_cases):
        print(f"Test {idx + 1}: {name}")
        print(f"  SMILES: {smiles}")

        mol = smiles_to_3d_structure(smiles, name)

        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_conformers = mol.GetNumConformers()

            print(f"  Success: Generated 3D structure with {num_atoms} atoms, {num_bonds} bonds")
            print(f"  Number of conformers: {num_conformers}")

            if num_conformers > 0:
                conf = mol.GetConformer()
                pos = conf.GetAtomPosition(0)
                print(f"  First atom coordinates: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")

                mol_file = os.path.join(temp_dir.name, f"{name}.mol")
                Chem.MolToMolFile(mol, mol_file)
                print(f"  Saved to: {mol_file}")

                success_count += 1
            else:
                print("  Failed: No conformers generated")
                failure_count += 1
        else:
            print("  Failed: Could not generate 3D structure")
            failure_count += 1

        print()

    temp_dir.cleanup()

    print("\n===== Summary =====")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failure_count}")

    return success_count == len(test_cases)


if __name__ == "__main__":
    print("Testing 3D structure generation functionality...")
    success = run_tests()
    if success:
        print("\nAll 3D structure generation tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome 3D structure generation tests FAILED!")
        sys.exit(1)
else:
    pytest.skip("Script style 3D structure tests, skipping under pytest", allow_module_level=True)
