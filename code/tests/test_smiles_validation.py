#!/usr/bin/env python3
"""
Simple test script to verify SMILES validation functionality

This script tests the SMILES validation function from the main module.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Import the validate_smiles function from the main module
try:
    # Try different import approaches
    try:
        from code.utils.helper_functions.molecular.molecule_processing import validate_smiles
        print("Successfully imported validate_smiles function (package import)")
    except ImportError:
        # Fall back to direct import
        sys.path.append(os.path.join(project_root, 'code', 'utils', 'helper_functions', 'molecular'))
        from molecule_processing import validate_smiles
        print("Successfully imported validate_smiles function (direct import)")
except ImportError as e:
    print(f"Failed to import validate_smiles function: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_smiles_validation")


def calculate_basic_descriptors(mol):
    """
    Calculate basic molecular descriptors for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with descriptor values
    """
    if mol is None:
        return {}
    
    try:
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
        }
        return descriptors
    except Exception as e:
        logger.error(f"Error calculating descriptors: {e}")
        return {}


def run_tests():
    """Run the validation tests."""
    # Test cases
    test_cases = [
        # Valid SMILES
        ("CC(F)(F)F", True),  # Trifluoromethane
        ("c1ccccc1", True),  # Benzene
        ("C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F", True),  # PFAS compound
        
        # Invalid SMILES
        ("invalid_smiles", False),
        ("C1=CC=C", False),  # Incomplete ring
        ("", False)  # Empty string
    ]
    
    success_count = 0
    failure_count = 0
    
    print("\n===== SMILES Validation Tests =====\n")
    
    for idx, (smiles, expected_valid) in enumerate(test_cases):
        is_valid, canonical, error = validate_smiles(smiles)
        
        if is_valid == expected_valid:
            result = "PASS"
            success_count += 1
        else:
            result = "FAIL"
            failure_count += 1
        
        print(f"Test {idx+1}: {result}")
        print(f"  SMILES: {smiles}")
        print(f"  Expected valid: {expected_valid}, Actual valid: {is_valid}")
        
        if is_valid:
            print(f"  Canonical SMILES: {canonical}")
            
            # Calculate some descriptors
            mol = Chem.MolFromSmiles(canonical)
            descriptors = calculate_basic_descriptors(mol)
            print(f"  Molecular weight: {descriptors.get('molecular_weight')}")
            print(f"  LogP: {descriptors.get('logp')}")
        else:
            print(f"  Error: {error}")
        
        print()
    
    # Print summary
    print("\n===== Summary =====")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failure_count}")
    
    return success_count == len(test_cases)


if __name__ == "__main__":
    print("Testing SMILES validation functionality...")
    success = run_tests()
    if success:
        print("\nAll SMILES validation tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome SMILES validation tests FAILED!")
        sys.exit(1) 