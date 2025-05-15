#!python
"""
Test script for 3D structure generation from SMILES

This script tests the conversion of SMILES strings to 3D molecular structures
using RDKit, which is a key component of our ORCA input preparation.
"""

import os
import sys
import logging
import tempfile

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw

    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_3d_generation")


def smiles_to_3d_structure(smiles, molecule_id="test_molecule", optimize=True):
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


def run_tests():
    """Run the 3D structure generation tests."""
    # Test cases
    test_cases = [
        # Simple molecules
        ("CC", "Ethane"),
        ("CC(F)(F)F", "Trifluoromethane"),
        ("c1ccccc1", "Benzene"),
        # PFAS compounds
        ("C(C(F)(F)F)C(F)(F)F", "Hexafluoroethane"),
        ("C(F)(F)(F)C(F)(F)C(F)(F)F", "Perfluoropropane"),
    ]

    success_count = 0
    failure_count = 0

    print("\n===== 3D Structure Generation Tests =====\n")

    temp_dir = tempfile.TemporaryDirectory()

    for idx, (smiles, name) in enumerate(test_cases):
        print(f"Test {idx+1}: {name}")
        print(f"  SMILES: {smiles}")

        # Generate 3D structure
        mol = smiles_to_3d_structure(smiles, name)

        if mol is not None:
            # Get basic information about the molecule
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_conformers = mol.GetNumConformers()

            print(f"  Success: Generated 3D structure with {num_atoms} atoms, {num_bonds} bonds")
            print(f"  Number of conformers: {num_conformers}")

            # Check that 3D coordinates exist
            if num_conformers > 0:
                conf = mol.GetConformer()
                pos = conf.GetAtomPosition(0)
                print(f"  First atom coordinates: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")

                # Save to MOL file for inspection
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

    # Clean up
    temp_dir.cleanup()

    # Print summary
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
