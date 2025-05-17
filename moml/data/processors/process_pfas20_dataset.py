#!/usr/bin/env python3
"""
Process PFAS-20 dataset to convert SMILES to standardized molecular representations.

This script handles the data standardization for the 20 PFAS compounds
by validating SMILES strings and generating standardized molecular representations
using the MoML framework.
"""

import os
import argparse
import pandas as pd
from rdkit import Chem
import logging
import tempfile

from moml.core import GraphCoarsener, calculate_molecular_descriptors
from moml.data import process_dataset, save_processed_molecules
from moml.utils import create_rdkit_mols, calculate_molecular_complexity

# Configure logging
logger = logging.getLogger(__name__)


def process_pfas20_dataset(input_path: str) -> pd.DataFrame:
    """
    Process the PFAS-20 dataset from a CSV file.

    Args:
        input_path: Path to the input CSV file

    Returns:
        DataFrame with processed molecular data
    """
    # Use our consolidated function to process the dataset
    df = process_dataset(input_path, smiles_col="smiles", id_col="id")

    # Create RDKit molecules
    df = create_rdkit_mols(df, smiles_col="smiles", mol_col="rdkit_mol")

    # Calculate molecular complexity
    df = calculate_molecular_complexity(df, mol_col="rdkit_mol")

    return df


def calculate_descriptors_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic molecular descriptors for the dataset.

    Args:
        df: DataFrame with molecular data

    Returns:
        DataFrame with added descriptors
    """
    # Only calculate for valid molecules
    valid_mask = df["is_valid_smiles"]

    # Calculate descriptors for valid molecules using our new function
    for idx, row in df[valid_mask].iterrows():
        descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
        for name, value in descriptors.items():
            df.at[idx, name] = value

    return df


def save_and_create_graphs(df: pd.DataFrame, output_dir: str, base_name: str) -> dict:
    """
    Save processed data and create molecular graphs.

    Args:
        df: DataFrame with processed data
        output_dir: Directory to save files
        base_name: Base name for output files

    Returns:
        Dictionary mapping file types to file paths
    """
    # First save the processed data using our consolidated function
    output_files = save_processed_molecules(df, output_dir, base_name)

    # Create molecular graphs using MoML
    valid_mols = df[df["is_valid_smiles"]]["rdkit_mol"].tolist()
    valid_ids = df[df["is_valid_smiles"]]["id"].tolist()

    # Create a directory for graph files
    graphs_dir = os.path.join(output_dir, f"{base_name}_graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Generate graphs using MoML's GraphCoarsener
    coarsener = GraphCoarsener(use_3d_coords=True, use_pfas_features=True)

    # Create graphs for each molecule
    for mol, mol_id in zip(valid_mols, valid_ids):
        # Generate 3D coordinates if needed
        mol_with_coords = mol
        try:
            # We'll create a copy with hydrogens that's used only for 3D coordinates generation
            mol_with_h = Chem.AddHs(Chem.Mol(mol))
            embed_result = Chem.AllChem.EmbedMolecule(mol_with_h)  # type: ignore
            
            if embed_result == -1:
                logger.warning(f"3D coordinate generation failed for molecule {mol_id}. Proceeding with 2D structure.")
            else:
                # If successful, use the 3D structure (without explicit H)
                Chem.AllChem.MMFFOptimizeMolecule(mol_with_h)
                mol_with_coords = Chem.RemoveHs(mol_with_h)
        except Exception as e:
            logger.warning(f"Could not embed molecule {mol_id} in 3D (SMILES: {Chem.MolToSmiles(mol)}): {e}. Proceeding with 2D structure.")
            # Continue with the original molecule (2D structure)

        # Save the hierarchical graphs: write Mol to temp file for file-based API
        with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as tmp:
            tmp_path = tmp.name
            Chem.MolToMolFile(mol_with_coords, tmp_path)
        try:
            coarsener.create_from_molecule_file(
                mol_file=tmp_path,
                output_dir=graphs_dir,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Could not remove temporary file {tmp_path}")

    output_files["molecular_graphs"] = graphs_dir
    return output_files


def main():
    """Process PFAS-20 dataset and save standardized molecular representations."""
    parser = argparse.ArgumentParser(description="Process PFAS-20 SMILES data")
    parser.add_argument(
        "--input", "-i", default="data/processed/pfas_final_dataset.csv", help="Path to input CSV file with PFAS data"
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/processed/chemical_list", help="Directory to save processed files"
    )
    parser.add_argument("--base-name", "-b", default="pfas20_standardized", help="Base filename for output files")
    parser.add_argument(
        "--calculate-descriptors", "-d", action="store_true", help="Calculate basic molecular descriptors"
    )
    args = parser.parse_args()

    # Process the dataset
    print(f"Processing PFAS-20 dataset from: {args.input}")
    df = process_pfas20_dataset(args.input)

    # Calculate descriptors if requested
    if args.calculate_descriptors:
        print("Calculating basic molecular descriptors...")
        df = calculate_descriptors_for_dataset(df)

    # Save processed data and create graphs
    output_files = save_and_create_graphs(df, args.output_dir, args.base_name)

    # Print summary
    valid_count = df["is_valid_smiles"].sum()
    total_count = len(df)
    print("\nProcessing Summary:")
    print(f"- Total compounds: {total_count}")
    if total_count > 0:
        print(f"- Valid SMILES: {valid_count} ({valid_count/total_count*100:.1f}%)")
    else:
        print(f"- Valid SMILES: {valid_count} (0.0%)")
    print(f"- Invalid SMILES: {total_count - valid_count}")

    print("\nOutput Files:")
    for file_type, file_path in output_files.items():
        print(f"- {file_type}: {file_path}")


if __name__ == "__main__":
    main()
