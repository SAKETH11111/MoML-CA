#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate MOL/SDF files from a CSV containing SMILES strings.

This script reads a CSV file, extracts SMILES strings and corresponding
identifiers, converts SMILES to 3D RDKit molecule objects, and saves
them as individual MOL or SDF files in a specified output directory.
"""

import argparse
import logging
import os
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Converts a SMILES string into an RDKit molecule with 3D coordinates.
    
    Attempts to parse the SMILES string, add hydrogens, generate 3D coordinates using ETKDGv3 embedding, and optimize geometry with MMFF94. Returns the resulting molecule with hydrogens, or None if any critical step fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return None

        mol_h = Chem.AddHs(mol)

        # Embed molecule with 3D coordinates
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 0xF00D  # For reproducibility
        embed_params.useRandomCoords = True # Start with random coordinates
        embed_params.numThreads = 0 # Use all available cores
        status = AllChem.EmbedMolecule(mol_h, embed_params)

        if status == -1:
            logger.warning(
                f"Could not generate 3D coordinates for SMILES: {smiles}. Trying with useBasicKnowledge."
            )
            embed_params_bk = AllChem.ETKDGv3()
            embed_params_bk.randomSeed = 0xF00D
            embed_params_bk.useRandomCoords = True
            embed_params_bk.useBasicKnowledge = True # Use basic knowledge for problematic cases
            embed_params_bk.numThreads = 0
            status = AllChem.EmbedMolecule(mol_h, embed_params_bk)
            if status == -1:
                logger.error(
                    f"Failed to generate 3D coordinates for SMILES: {smiles} even with basic knowledge."
                )
                return None
        
        # Optimize the geometry using MMFF94
        try:
            AllChem.MMFFOptimizeMolecule(mol_h)
        except Exception as e:
            logger.warning(
                f"Could not optimize 3D structure for SMILES: {smiles}. Error: {e}. Proceeding with unoptimized structure."
            )
        
        return mol_h # Return with hydrogens, common for SDF/MOL files intended for further processing

    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return None


def save_molecule_file(
    mol: Chem.Mol, output_dir: Path, identifier: str, file_format: str
) -> bool:
    """
    Saves an RDKit molecule to a file in MOL or SDF format using a sanitized identifier.
    
    Args:
        identifier: Used to generate a safe filename for the output file.
        file_format: Output file format; must be 'mol' or 'sdf'.
    
    Returns:
        True if the molecule was saved successfully; False otherwise.
    """
    if not mol:
        return False

    # Sanitize identifier for use as a filename
    safe_identifier = "".join(
        c if c.isalnum() or c in ('.', '_') else '_' for c in str(identifier)
    )
    if not safe_identifier:
        safe_identifier = "unnamed_molecule"

    filename = f"{safe_identifier}.{file_format.lower()}"
    output_path = output_dir / filename

    try:
        if file_format.lower() == "mol":
            Chem.MolToMolFile(mol, str(output_path))
        elif file_format.lower() == "sdf":
            writer = Chem.SDWriter(str(output_path))
            writer.write(mol)
            writer.close()
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return False
        logger.info(f"Saved molecule {identifier} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Could not save molecule {identifier} to {output_path}: {e}")
        return False


def main():
    """
    Parses command-line arguments and processes SMILES strings to generate 3D molecular structure files.
    
    Reads lists of SMILES strings and corresponding identifiers, generates 3D molecular structures for each valid pair, and saves them as MOL or SDF files in the specified output directory. Logs progress, warnings for invalid inputs, and a summary of results upon completion.
    """
    parser = argparse.ArgumentParser(
        description="Generate MOL/SDF files from lists of SMILES strings and identifiers."
    )
    parser.add_argument(
        "--smiles_list",
        nargs="+",
        required=True,
        help="List of SMILES strings to process.",
    )
    parser.add_argument(
        "--id_list",
        nargs="+",
        required=True,
        help="List of corresponding molecule identifiers (for filenames).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated MOL/SDF files.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["mol", "sdf"],
        default="sdf",
        help="Output file format (mol or sdf). Default is sdf.",
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process (for testing). Processes all by default.",
    )

    args = parser.parse_args()

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if len(args.smiles_list) != len(args.id_list):
        logger.error(
            "The number of SMILES strings must match the number of identifiers."
        )
        return

    processed_count = 0
    success_count = 0

    for i, smiles in enumerate(args.smiles_list):
        if args.max_molecules is not None and processed_count >= args.max_molecules:
            logger.info(f"Reached max_molecules limit of {args.max_molecules}.")
            break

        identifier = args.id_list[i]

        if not isinstance(smiles, str) or not smiles.strip():
            logger.warning(
                f"Skipping item {i+1} due to missing or invalid SMILES for ID: {identifier}"
            )
            continue
        
        if not isinstance(identifier, str) or not identifier.strip(): # Identifiers also need to be valid strings
            logger.warning(
                f"Skipping item {i+1} due to missing or invalid identifier for SMILES: {smiles}"
            )
            continue
        
        logger.info(f"Processing molecule {identifier} (SMILES: {smiles})")
        mol_3d = generate_3d_mol(smiles)

        if mol_3d:
            if save_molecule_file(
                mol_3d, output_dir_path, identifier, args.output_format
            ):
                success_count += 1
        processed_count += 1

    logger.info(
        f"Processing complete. Successfully generated {success_count} out of {processed_count} attempted molecules."
    )
    logger.info(f"Output files are located in: {output_dir_path.resolve()}")


if __name__ == "__main__":
    main()