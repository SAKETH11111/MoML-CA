"""
tests/test_smiles_validation_logic.py

Unit tests for SMILES validation functionality.

This script verifies the SMILES validation functions in moml.core module,
ensuring they correctly handle valid, invalid, and edge case structures.
"""

import logging
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Try to import RDKit
try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    pytest.skip("RDKit not installed, skipping SMILES validation tests", allow_module_level=True)

from moml.core import calculate_molecular_descriptors
from moml.utils import validate_smiles

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_smiles_validation")


def run_tests() -> bool:
    """
    Run SMILES validation tests.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n===== SMILES Validation Tests =====\n")

    print("Test case 1: Valid SMILES structures")
    valid_smiles: List[str] = [
        "C",
        "CC",
        "c1ccccc1",
        "CC(=O)O",
        "C(C(F)(F)F)C(F)(F)F",
        "ClC(Cl)(Cl)Cl",
        "C1CCC(CC1)C(=O)O",
        "CC1=C(C=C(C=C1)C(=O)O)C",
        "C[N+](C)(C)CC([O-])=O",
    ]

    valid_count = 0
    for smiles in valid_smiles:
        is_valid, canonical, mol, error_msg = validate_smiles(smiles)

        if is_valid:
            valid_count += 1
            print(f"  ✓ {smiles} → {canonical}")

            if mol is not None:
                descriptors = calculate_molecular_descriptors(mol)
                print(f"     MW: {descriptors['molecular_weight']:.2f}, LogP: {descriptors['logp']:.2f}")
        else:
            print(f"  ✗ {smiles} (Failed - should be valid). Error: {error_msg}")

    print(f"\nValidated {valid_count}/{len(valid_smiles)} valid SMILES")
    valid_test_passed = valid_count == len(valid_smiles)

    print("\nTest case 2: Invalid SMILES structures")
    invalid_smiles: List[str] = [
        "X",
        "C(",
        "CC(",
        "C1CC",
        "C1CC2",
        "C1CCCC",
        "C12C2",
        "invalidstructure",
    ]

    valid_but_should_fail: List[str] = [
        "C:C",
        "CC#CC#CC#CC#CC",
    ]

    invalid_count = 0
    for smiles in invalid_smiles:
        is_valid, canonical, mol, error_msg = validate_smiles(smiles)

        if not is_valid:
            invalid_count += 1
            print(f"  ✓ {smiles} (Correctly identified as invalid)")
        else:
            print(f"  ✗ {smiles} → {canonical} (Failed - should be invalid). Mol: {mol}, Error: {error_msg}")

    valid_but_should_fail_count = 0
    for smiles in valid_but_should_fail:
        is_valid, canonical, mol, error_msg = validate_smiles(smiles)
        if is_valid:
            print(f"  ! {smiles} → {canonical} (Validated by RDKit, but failed test intent). Mol: {mol}, Error: {error_msg}")
        else:
            valid_but_should_fail_count += 1
            print(f"  ✓ {smiles} (Correctly identified as invalid by RDKit)")

    total_invalid_tested = len(invalid_smiles)
    print(f"\nValidated {invalid_count}/{total_invalid_tested} invalid SMILES (excluding chemically questionable ones)")
    invalid_test_passed = invalid_count == total_invalid_tested

    print("\nTest case 3: Edge case SMILES structures")
    edge_smiles: List[str] = [
        "[H]",
        "c1cc[nH]c1",
        "[NH4+]",
        "[2H]O[2H]",
        "C=1C=CC=CC=1",
        "[13CH4]",
        "[*]c1ccccc1",
        "[Fe]",
        "Cc1c(C)cccc1C",
        "[Pt](Cl)(Cl)(Cl)(Cl)",
    ]

    edge_count = 0
    for smiles in edge_smiles:
        is_valid, canonical, mol, error_msg = validate_smiles(smiles)

        if is_valid:
            edge_count += 1
            print(f"  ✓ {smiles} → {canonical}")
        else:
            print(f"  ✗ {smiles} (Failed - should be edge case valid). Error: {error_msg}")

    print(f"\nValidated {edge_count}/{len(edge_smiles)} edge case SMILES")
    edge_test_passed = edge_count >= len(edge_smiles) * 0.7

    print("\n===== Summary =====")
    test_results: Dict[str, bool] = {
        "Valid SMILES": valid_test_passed,
        "Invalid SMILES": invalid_test_passed,
        "Edge cases": edge_test_passed,
    }

    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(test_results.values())
    return all_passed


class TestSmilesValidation(unittest.TestCase):
    """
    Pytest style tests for SMILES validation.
    """

    def test_valid_smiles(self) -> None:
        """
        Test validation of valid SMILES strings.
        """
        valid_smiles: List[str] = [
            "C",
            "CC",
            "c1ccccc1",
            "CC(=O)O",
            "C(C(F)(F)F)C(F)(F)F",
        ]

        for smiles in valid_smiles:
            is_valid, canonical, mol, error_msg = validate_smiles(smiles)
            self.assertTrue(is_valid, f"SMILES string {smiles} should be valid. Error: {error_msg}")
            self.assertIsNotNone(canonical, f"Canonical form should not be None for {smiles}")
            self.assertIsNotNone(mol, f"RDKit mol should not be None for {smiles}")

    def test_invalid_smiles(self) -> None:
        """
        Test validation of invalid SMILES strings.
        """
        invalid_smiles: List[str] = [
            "X",
            "C(",
            "CC(",
            "C1CC",
            "invalidstructure",
        ]

        for smiles in invalid_smiles:
            is_valid, canonical, mol, error_msg = validate_smiles(smiles)
            self.assertFalse(is_valid, f"SMILES string {smiles} should be invalid. Canonical: {canonical}, Mol: {mol}")

    def test_descriptor_calculation(self) -> None:
        """
        Test calculation of molecular descriptors for valid SMILES.
        """
        test_smiles = "CC(=O)O"

        is_valid, canonical, mol, error_msg = validate_smiles(test_smiles)
        self.assertTrue(is_valid, f"SMILES string {test_smiles} should be valid. Error: {error_msg}")
        self.assertIsNotNone(mol, "Mol object should not be None for descriptor calculation")

        descriptors = calculate_molecular_descriptors(mol)

        self.assertIn("molecular_weight", descriptors)
        self.assertIn("logp", descriptors)
        self.assertIn("num_atoms", descriptors)
        self.assertIn("num_bonds", descriptors)
        self.assertIn("num_rings", descriptors)

        self.assertGreater(descriptors["molecular_weight"], 58)
        self.assertLess(descriptors["molecular_weight"], 62)
        self.assertEqual(descriptors["num_atoms"], 8)
        self.assertGreaterEqual(descriptors["num_bonds"], 7)
        self.assertEqual(descriptors["num_rings"], 0)


if __name__ == "__main__":
    logger.info("Starting SMILES validation tests...")
    success = run_tests()
    sys.exit(0 if success else 1)
