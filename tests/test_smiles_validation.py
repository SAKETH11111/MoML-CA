#!/usr/bin/env python3
"""
Test script for SMILES validation functionality

This script verifies the SMILES validation functions in moml.core module,
ensuring they correctly handle valid, invalid, and edge case structures.
"""

import os
import sys
import logging

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Import from consolidated moml module
from moml.core import validate_smiles, calculate_molecular_descriptors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_smiles_validation")


def run_tests():
    """Run SMILES validation tests."""
    print("\n===== SMILES Validation Tests =====\n")
    
    # Test case 1: Valid SMILES
    print("Test case 1: Valid SMILES structures")
    valid_smiles = [
        "C",                      # Methane
        "CC",                     # Ethane
        "c1ccccc1",               # Benzene
        "CC(=O)O",                # Acetic acid
        "C(C(F)(F)F)C(F)(F)F",    # Hexafluoroethane
        "ClC(Cl)(Cl)Cl",          # Carbon tetrachloride
        "C1CCC(CC1)C(=O)O",       # Cyclohexanecarboxylic acid
        "CC1=C(C=C(C=C1)C(=O)O)C",# 2,5-Dimethylbenzoic acid
        "C[N+](C)(C)CC([O-])=O",  # Betaine (zwitterion)
    ]
    
    valid_count = 0
    for smiles in valid_smiles:
        is_valid, canonical, mol = validate_smiles(smiles)
        
        if is_valid:
            valid_count += 1
            print(f"  ✓ {smiles} → {canonical}")
            
            # Calculate some basic descriptors as a further check
            if mol is not None:
                descriptors = calculate_molecular_descriptors(mol)
                print(f"     MW: {descriptors['molecular_weight']:.2f}, LogP: {descriptors['logp']:.2f}")
        else:
            print(f"  ✗ {smiles} (Failed - should be valid)")
    
    print(f"\nValidated {valid_count}/{len(valid_smiles)} valid SMILES")
    valid_test_passed = valid_count == len(valid_smiles)
    
    # Test case 2: Invalid SMILES
    print("\nTest case 2: Invalid SMILES structures")
    invalid_smiles = [
        "X",                       # Invalid atom
        "C(",                      # Unclosed parenthesis
        "CC(",                     # Unclosed parenthesis
        "C1CC",                    # Unclosed ring
        "C1CC2",                   # Unclosed rings
        "C1CCCC",                  # Unclosed ring (5 should be 5-membered)
        "C12C2",                   # Invalid ring connection
        "invalidstructure",        # Random string
        "C:C",                     # Invalid bond type
        "CC#CC#CC#CC#CC",          # Cumulative triple bonds (unstable)
    ]
    
    invalid_count = 0
    for smiles in invalid_smiles:
        is_valid, canonical, mol = validate_smiles(smiles)
        
        if not is_valid:
            invalid_count += 1
            print(f"  ✓ {smiles} (Correctly identified as invalid)")
        else:
            print(f"  ✗ {smiles} → {canonical} (Failed - should be invalid)")
    
    print(f"\nValidated {invalid_count}/{len(invalid_smiles)} invalid SMILES")
    invalid_test_passed = invalid_count == len(invalid_smiles)
    
    # Test case 3: Edge cases
    print("\nTest case 3: Edge case SMILES structures")
    edge_smiles = [
        "[H]",                     # Hydrogen atom
        "c1cc[nH]c1",              # Pyrrole (aromatic nitrogen with attached H)
        "[NH4+]",                  # Ammonium ion
        "[2H]O[2H]",               # Heavy water (D2O)
        "C=1C=CC=CC=1",            # Benzene with '=' bonds
        "[13CH4]",                 # Isotope-labeled methane
        "[*]c1ccccc1",             # Unspecified atom
        "[Fe]",                    # Iron atom
        "Cc1c(C)cccc1C",           # 1,2,3-trimethylbenzene with explicit C
        "[Pt](Cl)(Cl)(Cl)(Cl)",    # PtCl4 (square planar)
    ]
    
    edge_count = 0
    for smiles in edge_smiles:
        is_valid, canonical, mol = validate_smiles(smiles)
        
        if is_valid:
            edge_count += 1
            print(f"  ✓ {smiles} → {canonical}")
        else:
            print(f"  ✗ {smiles} (Failed - should be edge case valid)")
    
    print(f"\nValidated {edge_count}/{len(edge_smiles)} edge case SMILES")
    edge_test_passed = edge_count >= len(edge_smiles) * 0.7  # Allow some failures in edge cases
    
    # Print summary
    print("\n===== Summary =====")
    test_results = {
        "Valid SMILES": valid_test_passed,
        "Invalid SMILES": invalid_test_passed,
        "Edge cases": edge_test_passed
    }
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    return all_passed


# Define pytest style tests
class TestSmilesValidation:
    """Pytest style tests for SMILES validation."""
    
    def test_valid_smiles(self):
        """Test validation of valid SMILES strings."""
        valid_smiles = [
            "C",                      # Methane
            "CC",                     # Ethane
            "c1ccccc1",               # Benzene
            "CC(=O)O",                # Acetic acid
            "C(C(F)(F)F)C(F)(F)F",    # Hexafluoroethane
        ]
        
        for smiles in valid_smiles:
            is_valid, canonical, mol = validate_smiles(smiles)
            assert is_valid, f"SMILES string {smiles} should be valid"
            assert canonical is not None, f"Canonical form should not be None for {smiles}"
            assert mol is not None, f"RDKit mol should not be None for {smiles}"
    
    def test_invalid_smiles(self):
        """Test validation of invalid SMILES strings."""
        invalid_smiles = [
            "X",                       # Invalid atom
            "C(",                      # Unclosed parenthesis
            "CC(",                     # Unclosed parenthesis
            "C1CC",                    # Unclosed ring
            "invalidstructure",        # Random string
        ]
        
        for smiles in invalid_smiles:
            is_valid, canonical, mol = validate_smiles(smiles)
            assert not is_valid, f"SMILES string {smiles} should be invalid"
    
    def test_descriptor_calculation(self):
        """Test calculation of molecular descriptors for valid SMILES."""
        test_smiles = "CC(=O)O"  # Acetic acid
        
        # Validate SMILES and get mol object
        is_valid, canonical, mol = validate_smiles(test_smiles)
        assert is_valid, f"SMILES string {test_smiles} should be valid"
        
        # Calculate descriptors
        descriptors = calculate_molecular_descriptors(mol)
        
        # Check that required descriptors exist
        assert "molecular_weight" in descriptors
        assert "logp" in descriptors
        assert "num_atoms" in descriptors
        assert "num_bonds" in descriptors
        assert "num_rings" in descriptors
        
        # Check specific values for acetic acid
        assert 58 < descriptors["molecular_weight"] < 62  # ~60.05 g/mol
        assert descriptors["num_atoms"] == 8  # 2C + 4H + 2O = 8
        assert descriptors["num_bonds"] >= 7  # 1C-C + 1C=O + 1C-O + 4C-H = 7


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll SMILES validation tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome SMILES validation tests FAILED!")
        sys.exit(1) 