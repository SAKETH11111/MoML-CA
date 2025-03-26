#!/usr/bin/env python3
"""
Test script for dataset processing functionality

This script tests the processing of a small mock PFAS dataset,
including loading data, validating SMILES, and calculating descriptors.
Uses the functions from the main molecule_processing module.
"""

import os
import sys
import pandas as pd
import logging
import tempfile

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Try to import from the main module
try:
    # Try different import approaches
    try:
        from code.utils.helper_functions.molecular.molecule_processing import (
            validate_smiles,
            process_dataset,
            calculate_basic_descriptors
        )
        print("Successfully imported from molecule_processing (package import)")
    except ImportError:
        # Fall back to direct import
        sys.path.append(os.path.join(project_root, 'code', 'utils', 'helper_functions', 'molecular'))
        from molecule_processing import (
            validate_smiles,
            process_dataset,
            calculate_basic_descriptors
        )
        print("Successfully imported from molecule_processing (direct import)")
except ImportError as e:
    print(f"Failed to import from molecule_processing: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_dataset_processing")

# Create a small mock dataset
MOCK_DATA = [
    {
        "compound_id": "PFAS-001",
        "smiles": "C(C(F)(F)F)C(F)(F)F",
        "name": "Hexafluoroethane",
        "formula": "C2F6",
        "cas_number": "76-16-4"
    },
    {
        "compound_id": "PFAS-002",
        "smiles": "C(F)(F)(F)C(F)(F)C(F)(F)F",
        "name": "Perfluoropropane",
        "formula": "C3F8",
        "cas_number": "76-19-7"
    },
    {
        "compound_id": "PFAS-003",
        "smiles": "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "name": "Perfluorobutane",
        "formula": "C4F10",
        "cas_number": "355-25-9"
    },
    {
        "compound_id": "PFAS-004", 
        "smiles": "invalidsmilesstring",
        "name": "Invalid Compound",
        "formula": "Unknown",
        "cas_number": "00-00-0"
    },
    {
        "compound_id": "PFAS-005",
        "smiles": "CC(=O)O",
        "name": "Acetic Acid",
        "formula": "C2H4O2",
        "cas_number": "64-19-7"
    }
]


def create_mock_dataset(output_file=None):
    """Create a mock dataset for testing."""
    df = pd.DataFrame(MOCK_DATA)
    
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df


def process_mock_dataset(df, smiles_col="smiles"):
    """
    Process a mock dataset using the main validate_smiles function.
    
    Args:
        df: DataFrame with SMILES strings
        smiles_col: Column name containing SMILES
        
    Returns:
        Tuple of (processed DataFrame, count of valid SMILES)
    """
    # Create new columns
    df["valid_smiles"] = False
    df["canonical_smiles"] = ""
    df["mol_weight"] = 0.0
    df["logp"] = 0.0
    
    valid_count = 0
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        
        # Use the imported validate_smiles function
        is_valid, canonical, _ = validate_smiles(smiles)
        
        if is_valid:
            # Mark as valid
            df.at[idx, "valid_smiles"] = True
            df.at[idx, "canonical_smiles"] = canonical
            
            # Create molecule and calculate descriptors
            mol = Chem.MolFromSmiles(canonical)
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            df.at[idx, "mol_weight"] = mol_weight
            df.at[idx, "logp"] = logp
            
            valid_count += 1
    
    return df, valid_count


def run_tests():
    """Run the dataset processing tests."""
    print("\n===== Dataset Processing Tests =====\n")
    
    # Create temporary directory for test files
    temp_dir = tempfile.TemporaryDirectory()
    test_csv = os.path.join(temp_dir.name, "mock_pfas_data.csv")
    processed_csv = os.path.join(temp_dir.name, "processed_pfas_data.csv")
    
    # Test 1: Create mock dataset
    print("Test 1: Creating mock dataset")
    df = create_mock_dataset(test_csv)
    if os.path.exists(test_csv):
        file_size = os.path.getsize(test_csv)
        print(f"  Success: Created mock dataset with {len(df)} compounds")
        print(f"  File saved to: {test_csv} ({file_size} bytes)")
        test1_passed = True
    else:
        print("  Failed: Could not create mock dataset file")
        test1_passed = False
    
    print()
    
    # Test 2: Validate SMILES and calculate descriptors
    print("Test 2: Validating SMILES and calculating descriptors")
    processed_df, valid_count = process_mock_dataset(df)
    
    expected_valid = 4  # All except the invalid one
    if valid_count == expected_valid:
        print(f"  Success: Validated {valid_count}/{len(df)} SMILES strings")
        test2_passed = True
    else:
        print(f"  Failed: Expected {expected_valid} valid SMILES, got {valid_count}")
        test2_passed = False
    
    # Save processed data
    processed_df.to_csv(processed_csv, index=False)
    print(f"  Processed data saved to: {processed_csv}")
    
    print()
    
    # Test 3: Check descriptor calculation
    print("Test 3: Checking descriptor calculations")
    
    # Get descriptor stats
    valid_mols = processed_df[processed_df["valid_smiles"]]
    avg_weight = valid_mols["mol_weight"].mean()
    avg_logp = valid_mols["logp"].mean()
    
    print(f"  Number of valid molecules: {len(valid_mols)}")
    print(f"  Average molecular weight: {avg_weight:.2f}")
    print(f"  Average LogP: {avg_logp:.2f}")
    
    # Quick check for molecular weight range (should be reasonable for these compounds)
    if 50 < avg_weight < 500:
        print("  Success: Molecular weight calculations look reasonable")
        weight_check_passed = True
    else:
        print("  Warning: Molecular weight values may be incorrect")
        weight_check_passed = False
    
    # Check specific values for a known compound (hexafluoroethane)
    hexafluoroethane = processed_df[processed_df["name"] == "Hexafluoroethane"]
    if not hexafluoroethane.empty:
        h_weight = hexafluoroethane.iloc[0]["mol_weight"]
        h_logp = hexafluoroethane.iloc[0]["logp"]
        
        print(f"  Hexafluoroethane:")
        print(f"    Molecular weight: {h_weight:.2f} (expected ~152)")
        print(f"    LogP: {h_logp:.2f}")
        
        # Hexafluoroethane should have a MW around 152 g/mol
        if 145 < h_weight < 160:
            print("    Success: Hexafluoroethane molecular weight is correct")
            test3_passed = weight_check_passed
        else:
            print("    Failed: Hexafluoroethane molecular weight is incorrect")
            test3_passed = False
    else:
        print("  Failed: Could not find Hexafluoroethane in the dataset")
        test3_passed = False
    
    # Clean up
    temp_dir.cleanup()
    
    # Print summary
    print("\n===== Summary =====")
    test_results = {
        "Create mock dataset": test1_passed,
        "Validate SMILES": test2_passed,
        "Calculate descriptors": test3_passed
    }
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    return all_passed


if __name__ == "__main__":
    print("Testing dataset processing functionality...")
    success = run_tests()
    if success:
        print("\nAll dataset processing tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome dataset processing tests FAILED!")
        sys.exit(1) 