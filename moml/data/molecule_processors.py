#!/usr/bin/env python
"""
Data Processing Utilities

This module provides functions for processing molecule files into graph representations
suitable for machine learning. It can be used as a library or run as a command line script.
"""

import os
import glob
import torch
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Any
import pandas as pd
import pickle

from moml.core import create_graph_processor, find_charges_file, read_charges_from_file
from moml.utils import validate_smiles


def process_mol_file(mol_file: str, processor=None, charges_file: Optional[str] = None) -> Any:
    """
    Process a single molecule file into a graph representation.

    Args:
        mol_file: Path to the molecule file
        processor: Optional molecular graph processor instance
        charges_file: Optional path to a file with partial charges

    Returns:
        Graph representation of the molecule
    """
    # Create processor if not provided
    if processor is None:
        processor = create_graph_processor()

    # Read charges if available
    additional_features = None
    if charges_file:
        try:
            partial_charges = read_charges_from_file(charges_file)
            if partial_charges is not None:
                additional_features = {"partial_charges": partial_charges}
        except Exception as e:
            print(f"Error reading charges from {charges_file}: {e}")

    # Process file using the graph processor
    return processor.file_to_graph(mol_file, additional_features)


def process_mol_file_to_graph(
    mol_file: str, output_file: Optional[str] = None, processor=None, charges_file: Optional[str] = None
) -> str:
    """
    Process a molecule file into a graph and save it to disk.

    Args:
        mol_file: Path to the molecule file
        output_file: Optional path to save the graph
        processor: Optional molecular graph processor instance
        charges_file: Optional path to a file with partial charges

    Returns:
        Path to the saved graph file
    """
    # Process the molecule
    graph = process_mol_file(mol_file, processor, charges_file)

    # Determine output path if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(mol_file))[0]
        output_dir = os.path.dirname(mol_file)
        output_file = os.path.join(output_dir, f"{base_name}_graph.pt")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save graph to file
    torch.save(graph, output_file)

    return output_file


def batch_process_molecules(
    input_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    charges_dir: Optional[str] = None,
    file_pattern: str = "*.mol,*.sdf",
) -> List[str]:
    """
    Process all molecule files in the input directory and save graph representations
    to the output directory.

    Args:
        input_dir: Directory containing molecule files
        output_dir: Directory to save processed graph files
        config: Optional configuration for graph processing
        charges_dir: Optional directory containing charge files
        file_pattern: Pattern(s) to match molecule files

    Returns:
        List of paths to the saved graph files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create processor with config
    processor = create_graph_processor(config)

    # Get list of molecule files
    mol_files = []
    for pattern in file_pattern.split(","):
        mol_files.extend(glob.glob(os.path.join(input_dir, pattern.strip())))

    if not mol_files:
        print(f"No files found matching pattern '{file_pattern}' in {input_dir}")
        return []

    # Process each molecule file
    processed_files = []
    for mol_file in tqdm(mol_files, desc="Processing molecules"):
        try:
            # Find corresponding charges file if charges_dir is provided
            charges_file = None
            if charges_dir:
                charges_file = find_charges_file(mol_file, charges_dir)

            # Generate output file path
            base_name = os.path.splitext(os.path.basename(mol_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_graph.pt")

            # Process and save the file
            process_mol_file_to_graph(
                mol_file=mol_file, output_file=output_file, processor=processor, charges_file=charges_file
            )

            processed_files.append(output_file)

        except Exception as e:
            print(f"Error processing {mol_file}: {e}")

    print(f"Processed {len(processed_files)} molecules successfully")

    # Save configuration for reference if provided
    if config:
        config_path = os.path.join(output_dir, "preprocessing_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved preprocessing configuration to {config_path}")

    return processed_files


def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess molecule files into graph representations")

    # Input/Output arguments
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing molecule files (.mol or .sdf)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed graph files (.pt)")
    parser.add_argument(
        "--charges_dir", type=str, default=None, help="Optional directory containing partial charge files"
    )
    parser.add_argument("--file_pattern", type=str, default="*.mol,*.sdf", help="Pattern(s) to match molecule files")

    # Graph processing configuration
    parser.add_argument("--use_qm", action="store_true", help="Include QM properties in the graph")
    parser.add_argument("--use_3d", action="store_true", help="Include 3D coordinates in the graph")
    parser.add_argument("--use_conformers", action="store_true", help="Generate and include multiple conformers")
    parser.add_argument("--add_h", action="store_true", help="Add explicit hydrogens to molecules")

    return parser.parse_args()


def main():
    """Run preprocessing from command line arguments."""
    args = parse_args()

    # Convert arguments to configuration dictionary
    config = {"use_qm": args.use_qm, "use_3d": args.use_3d, "use_conformers": args.use_conformers, "add_h": args.add_h}

    # Process the molecules
    batch_process_molecules(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        charges_dir=args.charges_dir,
        file_pattern=args.file_pattern,
    )

    print("Preprocessing completed successfully")


if __name__ == "__main__":
    main()


def process_dataset(csv_path: str, smiles_col: str = "SMILES", id_col: str = None) -> pd.DataFrame:
    """
    Process a CSV file containing molecular data, validating SMILES strings.

    Args:
        csv_path: Path to the CSV file with SMILES data
        smiles_col: Name of the column containing SMILES strings
        id_col: Name of the column containing molecule IDs (optional)

    Returns:
        DataFrame with added columns for validation results
    """
    import logging

    logger = logging.getLogger("moml.data")

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} compounds")

        # Check if SMILES column exists
        if smiles_col not in df.columns:
            logger.error(f"{smiles_col} column not found in the dataset")
            raise ValueError(f"{smiles_col} column not found in the dataset")

        # Initialize new columns
        df["rdkit_mol"] = None
        df["canonical_smiles"] = None
        df["is_valid_smiles"] = False
        df["smiles_error"] = None

        # Process each SMILES string
        valid_count = 0
        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            is_valid, canonical_smiles, mol_obj, error = validate_smiles(smiles)

            df.at[idx, "is_valid_smiles"] = is_valid
            df.at[idx, "smiles_error"] = error

            if is_valid:
                df.at[idx, "canonical_smiles"] = canonical_smiles
                df.at[idx, "rdkit_mol"] = mol_obj
                valid_count += 1

        logger.info(f"Successfully processed {valid_count}/{len(df)} compounds")
        return df

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise


def save_processed_molecules(df: pd.DataFrame, output_dir: str, base_filename: str) -> dict:
    """
    Save processed molecular data to disk in multiple formats.

    Args:
        df: DataFrame with processed molecular data
        output_dir: Directory to save files
        base_filename: Base name for output files

    Returns:
        Dictionary with paths to saved files
    """
    import os
    import logging

    logger = logging.getLogger("moml.data")

    os.makedirs(output_dir, exist_ok=True)
    output_files = {}

    # Save as CSV (without RDKit mol objects)
    csv_df = df.copy()
    if "rdkit_mol" in csv_df.columns:
        csv_df = csv_df.drop(columns=["rdkit_mol"])

    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    csv_df.to_csv(csv_path, index=False)
    output_files["csv"] = csv_path

    # Save valid molecules as pickle file if rdkit_mol column exists
    if "rdkit_mol" in df.columns and "is_valid_smiles" in df.columns:
        # Find ID column to use as keys
        id_col = None
        for col in ["id", "ID", "molecule_id", "common_name", "CASRN"]:
            if col in df.columns:
                id_col = col
                break

        if id_col:
            valid_mols = {}
            for idx, row in df[df["is_valid_smiles"]].iterrows():
                valid_mols[row[id_col]] = row["rdkit_mol"]

            pkl_path = os.path.join(output_dir, f"{base_filename}_mols.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(valid_mols, f)
            output_files["pickle"] = pkl_path

    # Generate a report of validation issues if applicable
    if "is_valid_smiles" in df.columns and not df["is_valid_smiles"].all():
        cols_to_include = ["smiles_error"]

        # Add ID column if available
        for col in ["id", "ID", "molecule_id", "common_name", "CASRN"]:
            if col in df.columns:
                cols_to_include.insert(0, col)
                break

        # Add SMILES column
        for col in df.columns:
            if "smiles" in col.lower() and col != "canonical_smiles" and col != "is_valid_smiles":
                cols_to_include.insert(1, col)
                break

        report_df = df[~df["is_valid_smiles"]][cols_to_include]
        report_path = os.path.join(output_dir, f"{base_filename}_validation_issues.csv")
        report_df.to_csv(report_path, index=False)
        output_files["issues_report"] = report_path

    logger.info(f"Saved processed data to {output_dir}")
    return output_files


def batch_process_molecules_dataset(
    input_file: str,
    output_dir: str,
    config: dict = None,
    smiles_col: str = "SMILES",
    id_col: str = None,
    calculate_descriptors: bool = True,
) -> dict:
    """
    Process a dataset of molecules from CSV, validate SMILES, and optionally calculate descriptors.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed files
        config: Configuration for processing
        smiles_col: Name of the column containing SMILES strings
        id_col: Name of the column containing molecule IDs
        calculate_descriptors: Whether to calculate molecular descriptors

    Returns:
        Dictionary with paths to saved files
    """
    import os
    from moml.core import calculate_molecular_descriptors

    # Process the dataset
    df = process_dataset(input_file, smiles_col, id_col)

    # Calculate descriptors if requested
    if calculate_descriptors:
        # Calculate descriptors for valid molecules
        for idx, row in df[df["is_valid_smiles"]].iterrows():
            descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
            for name, value in descriptors.items():
                df.at[idx, name] = value

    # Save processed data
    base_filename = os.path.splitext(os.path.basename(input_file))[0] + "_processed"
    return save_processed_molecules(df, output_dir, base_filename)
