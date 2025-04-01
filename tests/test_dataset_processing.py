#!/usr/bin/env python3
"""
Test script for dataset processing functionality

This script tests the processing of a small mock PFAS dataset,
including loading data, validating SMILES, and calculating descriptors.
Uses the functions from the moml.core and moml.data modules.
"""

import os
import sys
import pandas as pd
import logging
import tempfile
import pytest

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

# Import from consolidated moml modules
from moml.core import calculate_molecular_descriptors
from moml.data import process_dataset, save_processed_molecules

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
    
    # Test 2: Process dataset using the consolidated function
    print("Test 2: Processing dataset with new consolidated functions")
    processed_df = process_dataset(test_csv, smiles_col="smiles", id_col="compound_id")
    
    valid_count = processed_df['is_valid_smiles'].sum()
    expected_valid = 4  # All except the invalid one
    
    if valid_count == expected_valid:
        print(f"  Success: Validated {valid_count}/{len(processed_df)} SMILES strings")
        test2_passed = True
    else:
        print(f"  Failed: Expected {expected_valid} valid SMILES, got {valid_count}")
        test2_passed = False
    
    # Add molecular descriptors
    for idx, row in processed_df[processed_df['is_valid_smiles']].iterrows():
        descriptors = calculate_molecular_descriptors(row['rdkit_mol'])
        for name, value in descriptors.items():
            processed_df.at[idx, name] = value
    
    # Save processed data
    output_files = save_processed_molecules(
        processed_df, 
        temp_dir.name, 
        "processed_pfas_data"
    )
    
    print(f"  Processed data saved to: {output_files}")
    
    print()
    
    # Test 3: Check descriptor calculation
    print("Test 3: Checking descriptor calculations")
    
    # Get descriptor stats
    valid_mols = processed_df[processed_df["is_valid_smiles"]]
    avg_weight = valid_mols["molecular_weight"].mean()
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
        h_weight = hexafluoroethane.iloc[0]["molecular_weight"]
        h_logp = hexafluoroethane.iloc[0]["logp"]
        
        print(f"  Hexafluoroethane:")
        print(f"    Molecular weight: {h_weight:.2f} (expected ~138)")
        print(f"    LogP: {h_logp:.2f}")
        
        # Hexafluoroethane should have a MW around 138 g/mol
        if 135 < h_weight < 142:
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
        "Process dataset": test2_passed,
        "Calculate descriptors": test3_passed
    }
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    return all_passed


# Define pytest style tests as well
class TestDatasetProcessing:
    """Pytest style tests for dataset processing."""
    
    @pytest.fixture
    def mock_dataset_file(self, tmp_path):
        """Create a temporary mock dataset file."""
        test_csv = tmp_path / "mock_pfas_data.csv"
        df = create_mock_dataset(test_csv)
        return test_csv
    
    def test_process_dataset(self, mock_dataset_file):
        """Test that the dataset processing function works correctly."""
        processed_df = process_dataset(mock_dataset_file, smiles_col="smiles", id_col="compound_id")
        
        # Check basic processing results
        assert 'is_valid_smiles' in processed_df.columns
        assert 'rdkit_mol' in processed_df.columns
        
        # Should have 4 valid SMILES
        assert processed_df['is_valid_smiles'].sum() == 4
        
        # Add descriptors
        for idx, row in processed_df[processed_df['is_valid_smiles']].iterrows():
            descriptors = calculate_molecular_descriptors(row['rdkit_mol'])
            for name, value in descriptors.items():
                processed_df.at[idx, name] = value
        
        # Check hexafluoroethane weight
        hexafluoroethane = processed_df[processed_df["name"] == "Hexafluoroethane"]
        assert not hexafluoroethane.empty
        assert 135 < hexafluoroethane.iloc[0]["molecular_weight"] < 142
    
    def test_save_processed_data(self, mock_dataset_file, tmp_path):
        """Test saving processed data to various formats."""
        processed_df = process_dataset(mock_dataset_file, smiles_col="smiles", id_col="compound_id")
        
        # Add descriptors
        for idx, row in processed_df[processed_df['is_valid_smiles']].iterrows():
            descriptors = calculate_molecular_descriptors(row['rdkit_mol'])
            for name, value in descriptors.items():
                processed_df.at[idx, name] = value
        
        # Save the data
        output_files = save_processed_molecules(
            processed_df, 
            tmp_path, 
            "processed_data"
        )
        
        # Check that the CSV file was created
        assert 'csv' in output_files
        assert os.path.exists(output_files['csv'])
        
        # Check that pickle file was created if rdkit_mol exists
        if 'pickle' in output_files:
            assert os.path.exists(output_files['pickle'])


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll dataset processing tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome dataset processing tests FAILED!")
        sys.exit(1) 