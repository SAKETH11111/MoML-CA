#!/usr/bin/env python3
"""
Test script for MolecularGraphProcessor class

This script tests the functionality of the MolecularGraphProcessor class
for converting molecular structures to graph representations.
"""

import os
import sys
import pandas as pd
import numpy as np
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
from moml.core import validate_smiles, create_rdkit_mols
from moml.graph import MolecularGraphProcessor


def run_tests():
    """Run the molecular graph processor tests."""
    print("\n===== Molecular Graph Processor Tests =====\n")
    
    # Test SMILES strings
    test_smiles = [
        "C(C(F)(F)F)C(F)(F)F",  # Hexafluoroethane
        "C(F)(F)(F)C(F)(F)C(F)(F)F",  # Perfluoropropane
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
    ]
    
    test_names = ["Hexafluoroethane", "Perfluoropropane", "Acetic acid", "Benzene"]
    
    # Create a test dataframe
    test_df = pd.DataFrame({
        "smiles": test_smiles,
        "name": test_names,
        "id": [f"TEST-{i+1:03d}" for i in range(len(test_smiles))]
    })
    
    # Test 1: Create RDKit molecules
    print("Test 1: Creating RDKit molecules")
    test_df = create_rdkit_mols(test_df, smiles_col="smiles", mol_col="rdkit_mol")
    
    if "rdkit_mol" in test_df.columns:
        valid_mols = test_df["rdkit_mol"].apply(lambda x: x is not None).sum()
        print(f"  Success: Created {valid_mols}/{len(test_df)} valid RDKit molecules")
        test1_passed = valid_mols == len(test_df)
    else:
        print("  Failed: Could not create RDKit molecules")
        test1_passed = False
    
    print()
    
    # Test 2: Initialize MolecularGraphProcessor
    print("Test 2: Initializing MolecularGraphProcessor")
    try:
        processor = MolecularGraphProcessor()
        print("  Success: Initialized MolecularGraphProcessor")
        test2_passed = True
    except Exception as e:
        print(f"  Failed: Could not initialize MolecularGraphProcessor - {e}")
        test2_passed = False
        return False
    
    print()
    
    # Test 3: Generate atom features
    print("Test 3: Generating atom features")
    atom_features_list = []
    
    for idx, row in test_df.iterrows():
        mol = row["rdkit_mol"]
        if mol is not None:
            atom_feats = processor.get_atom_features(mol)
            atom_features_list.append(atom_feats)
            print(f"  {row['name']}: {atom_feats.shape} atom features")
    
    if atom_features_list:
        print(f"  Success: Generated atom features for {len(atom_features_list)} molecules")
        
        # Check the first molecule's atom features
        first_features = atom_features_list[0]
        if isinstance(first_features, np.ndarray) and first_features.ndim == 2:
            print(f"  First molecule has {first_features.shape[0]} atoms with {first_features.shape[1]} features each")
            test3_passed = True
        else:
            print("  Warning: Unexpected atom features format")
            test3_passed = False
    else:
        print("  Failed: Could not generate atom features")
        test3_passed = False
    
    print()
    
    # Test 4: Generate adjacency matrices
    print("Test 4: Generating adjacency matrices")
    adj_matrices = []
    
    for idx, row in test_df.iterrows():
        mol = row["rdkit_mol"]
        if mol is not None:
            adj_matrix = processor.get_adjacency_matrix(mol)
            adj_matrices.append(adj_matrix)
            print(f"  {row['name']}: {adj_matrix.shape} adjacency matrix")
    
    if adj_matrices:
        print(f"  Success: Generated adjacency matrices for {len(adj_matrices)} molecules")
        
        # Check the first molecule's adjacency matrix
        first_adj = adj_matrices[0]
        if isinstance(first_adj, np.ndarray) and first_adj.ndim == 2:
            print(f"  First molecule has a {first_adj.shape[0]}×{first_adj.shape[1]} adjacency matrix")
            # Check that it's symmetric
            if np.array_equal(first_adj, first_adj.T):
                print("  First adjacency matrix is symmetric, as expected")
                test4_passed = True
            else:
                print("  Warning: First adjacency matrix is not symmetric")
                test4_passed = False
        else:
            print("  Warning: Unexpected adjacency matrix format")
            test4_passed = False
    else:
        print("  Failed: Could not generate adjacency matrices")
        test4_passed = False
    
    print()
    
    # Test 5: Generate molecular graphs
    print("Test 5: Generating complete molecular graphs")
    
    temp_dir = tempfile.TemporaryDirectory()
    
    try:
        # Process all molecules in the dataframe
        processed_data = processor.process_dataframe(test_df, mol_column="rdkit_mol")
        
        if not processed_data.empty:
            # Check that the necessary columns exist
            required_cols = ["atom_features", "adjacency_matrix", "num_atoms"]
            has_cols = all(col in processed_data.columns for col in required_cols)
            
            if has_cols:
                print(f"  Success: Generated complete molecular graphs for {len(processed_data)} molecules")
                
                # Save the processed data
                output_path = os.path.join(temp_dir.name, "test_molecular_graphs.csv")
                
                # Check if we can save to file
                processed_data_for_save = processed_data.copy()
                
                # Remove atom features and adjacency matrices for CSV save
                processed_data_for_save.drop(["atom_features", "adjacency_matrix"], axis=1, inplace=True)
                processed_data_for_save.to_csv(output_path, index=False)
                
                if os.path.exists(output_path):
                    print(f"  Saved processed data to: {output_path}")
                    test5_passed = True
                else:
                    print("  Warning: Could not save processed data")
                    test5_passed = False
            else:
                missing_cols = [col for col in required_cols if col not in processed_data.columns]
                print(f"  Failed: Missing required columns: {missing_cols}")
                test5_passed = False
        else:
            print("  Failed: Empty processed data")
            test5_passed = False
    except Exception as e:
        print(f"  Failed: Error processing the dataframe - {e}")
        test5_passed = False
    
    # Clean up
    temp_dir.cleanup()
    
    # Print summary
    print("\n===== Summary =====")
    test_results = {
        "Create RDKit molecules": test1_passed,
        "Initialize processor": test2_passed,
        "Generate atom features": test3_passed,
        "Generate adjacency matrices": test4_passed,
        "Generate molecular graphs": test5_passed
    }
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    return all_passed


# Define pytest style tests as well
class TestMolecularGraphProcessor:
    """Pytest style tests for the MolecularGraphProcessor."""
    
    @pytest.fixture
    def test_dataframe(self):
        """Create a test dataframe with SMILES strings."""
        test_smiles = [
            "C(C(F)(F)F)C(F)(F)F",  # Hexafluoroethane
            "C(F)(F)(F)C(F)(F)C(F)(F)F",  # Perfluoropropane
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
        ]
        
        test_names = ["Hexafluoroethane", "Perfluoropropane", "Acetic acid", "Benzene"]
        
        df = pd.DataFrame({
            "smiles": test_smiles,
            "name": test_names,
            "id": [f"TEST-{i+1:03d}" for i in range(len(test_smiles))]
        })
        
        # Create RDKit molecules
        df = create_rdkit_mols(df, smiles_col="smiles", mol_col="rdkit_mol")
        
        return df
    
    @pytest.fixture
    def processor(self):
        """Create a MolecularGraphProcessor instance."""
        return MolecularGraphProcessor()
    
    def test_processor_initialization(self, processor):
        """Test that the processor initializes correctly."""
        assert processor is not None
        assert processor.atom_feature_dim > 0
        assert processor.bond_feature_dim > 0
    
    def test_atom_features(self, processor, test_dataframe):
        """Test that atom features are generated correctly."""
        # Get the first molecule
        mol = test_dataframe.iloc[0]["rdkit_mol"]
        assert mol is not None
        
        # Generate atom features
        atom_features = processor.get_atom_features(mol)
        
        # Check that the features are correctly shaped
        assert isinstance(atom_features, np.ndarray)
        assert atom_features.ndim == 2
        assert atom_features.shape[0] == mol.GetNumAtoms()
        assert atom_features.shape[1] == processor.atom_feature_dim
    
    def test_adjacency_matrix(self, processor, test_dataframe):
        """Test that adjacency matrices are generated correctly."""
        # Get the first molecule
        mol = test_dataframe.iloc[0]["rdkit_mol"]
        assert mol is not None
        
        # Generate adjacency matrix
        adj_matrix = processor.get_adjacency_matrix(mol)
        
        # Check that the matrix is correctly shaped
        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.ndim == 2
        assert adj_matrix.shape[0] == mol.GetNumAtoms()
        assert adj_matrix.shape[1] == mol.GetNumAtoms()
        
        # Check that it's symmetric
        assert np.array_equal(adj_matrix, adj_matrix.T)
    
    def test_process_dataframe(self, processor, test_dataframe):
        """Test that the processor can process a full dataframe."""
        # Process the dataframe
        processed_data = processor.process_dataframe(test_dataframe, mol_column="rdkit_mol")
        
        # Check that the dataframe was processed correctly
        assert not processed_data.empty
        assert "atom_features" in processed_data.columns
        assert "adjacency_matrix" in processed_data.columns
        assert "num_atoms" in processed_data.columns
        
        # Check that every row has the expected data
        for idx, row in processed_data.iterrows():
            assert isinstance(row["atom_features"], np.ndarray)
            assert isinstance(row["adjacency_matrix"], np.ndarray)
            assert row["num_atoms"] > 0
            assert row["atom_features"].shape[0] == row["num_atoms"]
            assert row["adjacency_matrix"].shape == (row["num_atoms"], row["num_atoms"])


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll molecular graph processor tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome molecular graph processor tests FAILED!")
        sys.exit(1) 