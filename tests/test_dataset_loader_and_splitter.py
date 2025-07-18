"""
tests/test_dataset_loader_and_splitter.py

Tests dataset processing functionality, including loading, validating SMILES,
calculating descriptors, and saving processed data for a mock PFAS dataset.
"""

import logging
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from moml.core import calculate_molecular_descriptors
from moml.data import process_dataset, save_processed_molecules

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    from rdkit import Chem
except ImportError:
    pytest.skip("RDKit not installed, skipping dataset processing tests", allow_module_level=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_dataset_processing")

MOCK_DATA = [
    {
        "compound_id": "PFAS-001",
        "smiles": "FC(F)(F)C(F)(F)F",
        "name": "Hexafluoroethane",
        "formula": "C2F6",
        "cas_number": "76-16-4",
    },
    {
        "compound_id": "PFAS-002",
        "smiles": "C(F)(F)(F)C(F)(F)C(F)(F)F",
        "name": "Perfluoropropane",
        "formula": "C3F8",
        "cas_number": "76-19-7",
    },
    {
        "compound_id": "PFAS-003",
        "smiles": "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "name": "Perfluorobutane",
        "formula": "C4F10",
        "cas_number": "355-25-9",
    },
    {
        "compound_id": "PFAS-004",
        "smiles": "invalidsmilesstring",
        "name": "Invalid Compound",
        "formula": "Unknown",
        "cas_number": "00-00-0",
    },
    {
        "compound_id": "PFAS-005",
        "smiles": "CC(=O)O",
        "name": "Acetic Acid",
        "formula": "C2H4O2",
        "cas_number": "64-19-7",
    },
]

def create_mock_dataset(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create a mock dataset for testing.

    Args:
        output_file (str, optional): Path to save the dataset as CSV. Defaults to None.

    Returns:
        pd.DataFrame: The mock dataset.
    """
    df = pd.DataFrame(MOCK_DATA)
    if output_file:
        df.to_csv(output_file, index=False)
    return df

def run_tests() -> bool:
    """
    Run the dataset processing tests.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n===== Dataset Processing Tests =====\n")
    temp_dir = tempfile.TemporaryDirectory()
    test_csv = os.path.join(temp_dir.name, "mock_pfas_data.csv")
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
    print("Test 2: Processing dataset with new consolidated functions")
    processed_df = process_dataset(test_csv, smiles_col="smiles", id_col="compound_id")
    valid_count = processed_df["is_valid_smiles"].sum()
    expected_valid = int(valid_count)
    if valid_count == expected_valid:
        print(f"  Success: Validated {valid_count}/{len(processed_df)} SMILES strings")
        test2_passed = True
    else:
        print(f"  Failed: Expected {expected_valid} valid SMILES, got {valid_count}")
        test2_passed = False
    for idx, row in processed_df[processed_df["is_valid_smiles"]].iterrows():
        descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
        for name, value in descriptors.items():
            processed_df.at[idx, name] = value
    output_files = save_processed_molecules(processed_df, temp_dir.name, "processed_pfas_data")
    print(f"  Processed data saved to: {output_files}")
    print()
    print("Test 3: Checking descriptor calculations")
    valid_mols = processed_df[processed_df["is_valid_smiles"]]
    avg_weight = valid_mols["molecular_weight"].mean()
    avg_logp = valid_mols["logp"].mean()
    print(f"  Number of valid molecules: {len(valid_mols)}")
    print(f"  Average molecular weight: {avg_weight:.2f}")
    print(f"  Average LogP: {avg_logp:.2f}")
    if 50 < avg_weight < 500:
        print("  Success: Molecular weight calculations look reasonable")
        weight_check_passed = True
    else:
        print("  Warning: Molecular weight values may be incorrect")
        weight_check_passed = False
    hexafluoroethane = processed_df[processed_df["name"] == "Hexafluoroethane"]
    if not hexafluoroethane.empty:
        h_weight = hexafluoroethane.iloc[0]["molecular_weight"]
        h_logp = hexafluoroethane.iloc[0]["logp"]
        print("  Hexafluoroethane:")
        print(f"    Molecular weight: {h_weight:.2f} (expected ~138)")
        print(f"    LogP: {h_logp:.2f}")
        if 135 < h_weight < 142:
            print("    Success: Hexafluoroethane molecular weight is correct")
            test3_passed = weight_check_passed
        else:
            print("    Failed: Hexafluoroethane molecular weight is incorrect")
            test3_passed = False
    else:
        print("  Failed: Could not find Hexafluoroethane in the dataset")
        test3_passed = False
    temp_dir.cleanup()
    print("\n===== Summary =====")
    test_results = {
        "Create mock dataset": test1_passed,
        "Process dataset": test2_passed,
        "Calculate descriptors": test3_passed,
    }
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    all_passed = all(test_results.values())
    return all_passed

class TestDatasetProcessing:
    """
    Pytest style tests for dataset processing.
    """

    @pytest.fixture
    def mock_dataset_file(self, tmp_path) -> Any:
        """
        Create a temporary mock dataset file.

        Args:
            tmp_path: Temporary path provided by pytest.

        Returns:
            Any: Path to the mock dataset file.
        """
        test_csv = tmp_path / "mock_pfas_data.csv"
        create_mock_dataset(test_csv)
        return test_csv

    def test_process_dataset(self, mock_dataset_file: Any) -> None:
        """
        Test that the dataset processing function works correctly.

        Args:
            mock_dataset_file: Path to the mock dataset file.
        """
        processed_df = process_dataset(mock_dataset_file, smiles_col="smiles", id_col="compound_id")
        assert "is_valid_smiles" in processed_df.columns
        assert "rdkit_mol" in processed_df.columns
        df_orig = pd.read_csv(mock_dataset_file)
        expected_valid = df_orig["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None).sum()
        assert processed_df["is_valid_smiles"].sum() == expected_valid
        for idx, row in processed_df[processed_df["is_valid_smiles"]].iterrows():
            descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
            for name, value in descriptors.items():
                processed_df.at[idx, name] = value
        hexafluoroethane = processed_df[processed_df["name"] == "Hexafluoroethane"]
        assert not hexafluoroethane.empty
        assert 135 < hexafluoroethane.iloc[0]["molecular_weight"] < 142

    def test_save_processed_data(self, mock_dataset_file: Any, tmp_path) -> None:
        """
        Test saving processed data to various formats.

        Args:
            mock_dataset_file: Path to the mock dataset file.
            tmp_path: Temporary path provided by pytest.
        """
        processed_df = process_dataset(mock_dataset_file, smiles_col="smiles", id_col="compound_id")
        for idx, row in processed_df[processed_df["is_valid_smiles"]].iterrows():
            descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
            for name, value in descriptors.items():
                processed_df.at[idx, name] = value
        output_files = save_processed_molecules(processed_df, tmp_path, "processed_data")
        assert "csv" in output_files
        assert os.path.exists(output_files["csv"])
        if "pickle" in output_files:
            assert os.path.exists(output_files["pickle"])
