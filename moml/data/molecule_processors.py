"""
moml/data/molecule_processors.py

Molecular data processing utilities for graph representation conversion.

This module provides comprehensive functions for processing molecule files into
graph representations suitable for machine learning. It supports batch processing,
validation, descriptor calculation, and multiple file formats. The module can be
used as a library or run as a command-line script.

Functions:
    process_mol_file: Process single molecule file to graph
    process_mol_file_to_graph: Process and save molecule graph to disk
    batch_process_molecules: Batch process molecule directory
    graph_batch_process: Simplified batch processing for pipelines
    process_dataset: Process CSV datasets with SMILES validation
    save_processed_molecules: Save processed data in multiple formats
    batch_process_molecules_dataset: Complete dataset processing pipeline

Example:
    Basic usage for single file processing:
    
    >>> from moml.data.molecule_processors import process_mol_file
    >>> graph = process_mol_file('molecule.sdf')
    
    Command-line usage:
    
    $ python -m moml.data.molecule_processors --input_dir molecules/ \
        --output_dir graphs/ --use_3d --add_h
"""

import argparse
import glob
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Constants
DEFAULT_FILE_PATTERN = "*.mol,*.sdf"
DEFAULT_SIMPLE_PATTERN = "*.mol"
DEFAULT_SMILES_COLUMN = "SMILES"
DEFAULT_CONFIG_FILENAME = "preprocessing_config.json"
DEFAULT_GRAPH_SUFFIX = "_graph.pt"
DEFAULT_MOLS_SUFFIX = "_mols.pkl"
DEFAULT_ISSUES_SUFFIX = "_validation_issues.csv"
DEFAULT_PROCESSED_SUFFIX = "_processed"

# Common ID column names to search for
ID_COLUMN_CANDIDATES = ["id", "ID", "molecule_id", "common_name", "CASRN"]

# Configure module logger
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem  # type: ignore
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Molecular processing will be limited.")
    
    # Create dummy Chem module for fallback
    class Chem:  # type: ignore
        """Dummy Chem class when RDKit is not available."""
        
        @staticmethod
        def MolFromMolFile(mol_file):
            """Dummy method that returns None."""
            return None

try:
    from moml.core import (  # type: ignore
        calculate_molecular_descriptors,
        create_graph_processor,  # type: ignore
        find_charges_file,  # type: ignore
        read_charges_from_file,  # type: ignore
    )
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    logger.warning("moml.core not available. Core functionality will be limited.")
    
    def create_graph_processor(config=None):  # type: ignore
        """Dummy function when core is not available."""
        raise ImportError("moml.core package required for graph processing")
    
    def find_charges_file(mol_file: str, charges_dir: Optional[str] = None) -> Optional[str]:  # type: ignore
        """Dummy function when core is not available."""
        return None
    
    def read_charges_from_file(charge_file):  # type: ignore
        """Dummy function when core is not available."""
        return None
    
    def calculate_molecular_descriptors(mol):  # type: ignore
        """Dummy function when core is not available."""
        return {}

try:
    from moml.utils.data_utils.validation import validate_smiles
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    logger.warning("SMILES validation not available.")
    
    def validate_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[Any], Optional[str]]:  # type: ignore
        """Dummy validation function matching real signature."""
        return False, None, None, "Validation not available"


def process_mol_file(
    mol_file: str,
    processor=None,
    charges_file: Optional[str] = None
) -> Any:
    """
    Process a single molecule file into a graph representation.
    
    This function converts a molecular structure file (MOL, SDF) into a graph
    representation suitable for machine learning. It can optionally include
    partial charges and other quantum mechanical properties.

    Args:
        mol_file: Path to the molecule file (MOL, SDF format)
        processor: Optional molecular graph processor instance. If None,
                  a default processor will be created
        charges_file: Optional path to file containing partial charges

    Returns:
        Graph representation of the molecule (PyTorch Geometric Data object)
        Returns None if processing fails
    
    Example:
        >>> graph = process_mol_file('caffeine.sdf')
        >>> print(graph.x.shape)  # Node features
        >>> print(graph.edge_index.shape)  # Edge connectivity
    
    Raises:
        ImportError: If required dependencies are not available
        FileNotFoundError: If molecule file doesn't exist
    """
    if not HAS_CORE:
        raise ImportError("moml.core package required for molecule processing")
    
    if not os.path.exists(mol_file):
        raise FileNotFoundError(f"Molecule file not found: {mol_file}")
    
    # Create processor if not provided
    if processor is None:
        try:
            processor = create_graph_processor()
        except Exception as e:
            logger.error(f"Failed to create graph processor: {e}")
            raise

    # Read additional features if charges file provided
    additional_features = None
    if charges_file:
        additional_features = _process_charges_file(mol_file, charges_file)

    # Process file using the graph processor
    try:
        graph = processor.file_to_graph(mol_file, additional_features)
        if graph is None:
            logger.warning(f"Graph processing returned None for {mol_file}")
        return graph
    except Exception as e:
        logger.error(f"Failed to process molecule file {mol_file}: {e}")
        return None


def _process_charges_file(mol_file: str, charges_file: str) -> Optional[Dict]:
    """
    Process partial charges file for a molecule.
    
    Args:
        mol_file: Path to molecule file (for validation)
        charges_file: Path to charges file
        
    Returns:
        Dictionary with partial charges or None if processing fails
    """
    if not HAS_RDKIT:
        logger.warning("RDKit not available, cannot validate charges")
        return None
    
    try:
        partial_charges = read_charges_from_file(charges_file)
        if partial_charges is None:
            logger.warning(f"No charges read from {charges_file}")
            return None
        
        # Validate charges length matches atom count
        mol = Chem.MolFromMolFile(mol_file)
        if mol is None:
            logger.error(f"Could not read molecule from {mol_file}")
            return None
        
        expected_atoms = mol.GetNumAtoms()
        if len(partial_charges) != expected_atoms:
            logger.error(
                f"Charges length mismatch: got {len(partial_charges)}, "
                f"expected {expected_atoms} for {mol_file}"
            )
            return None
        
        return {"partial_charges": partial_charges}
        
    except Exception as e:
        logger.error(f"Error processing charges file {charges_file}: {e}")
        return None


def process_mol_file_to_graph(
    mol_file: str,
    output_file: Optional[str] = None,
    processor=None,
    charges_file: Optional[str] = None
) -> str:
    """
    Process a molecule file into a graph and save it to disk.
    
    This function processes a molecular structure file and saves the resulting
    graph representation to a PyTorch file. The output path is automatically
    generated if not specified.

    Args:
        mol_file: Path to the molecule file
        output_file: Optional path to save the graph. If None, automatically
                    generated based on input filename
        processor: Optional molecular graph processor instance
        charges_file: Optional path to file with partial charges

    Returns:
        Path to the saved graph file
    
    Example:
        >>> saved_path = process_mol_file_to_graph('molecule.sdf')
        >>> print(saved_path)  # 'molecule_graph.pt'
    
    Raises:
        ValueError: If graph processing fails
        OSError: If file operations fail
    """
    # Process the molecule
    graph = process_mol_file(mol_file, processor, charges_file)
    if graph is None:
        error_msg = f"Graph processing failed for {mol_file}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Determine output path if not provided
    if output_file is None:
        output_file = _generate_output_path(mol_file, DEFAULT_GRAPH_SUFFIX)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save graph to file
    try:
        torch.save(graph, output_file)
        logger.debug(f"Saved graph to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to save graph to {output_file}: {e}")
        raise


def _generate_output_path(input_file: str, suffix: str) -> str:
    """
    Generate output file path based on input file.
    
    Args:
        input_file: Path to input file
        suffix: Suffix to append to filename
        
    Returns:
        Generated output file path
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(input_file)
    return os.path.join(output_dir, f"{base_name}{suffix}")


def batch_process_molecules(
    input_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    charges_dir: Optional[str] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
) -> List[str]:
    """
    Process all molecule files in a directory and save graph representations.
    
    This function processes all molecule files matching the specified pattern
    in the input directory and saves the resulting graph representations to
    the output directory. It supports various molecular file formats and
    optional partial charges.

    Args:
        input_dir: Directory containing molecule files
        output_dir: Directory to save processed graph files
        config: Optional configuration dictionary for graph processing
        charges_dir: Optional directory containing partial charge files
        file_pattern: Comma-separated patterns to match molecule files

    Returns:
        List of paths to successfully processed graph files
    
    Example:
        >>> processed = batch_process_molecules(
        ...     'molecules/', 'graphs/',
        ...     config={'use_3d': True, 'add_h': True}
        ... )
        >>> print(f"Processed {len(processed)} molecules")
    
    Raises:
        OSError: If input/output directories don't exist or can't be created
        ImportError: If required dependencies are missing
    """
    if not HAS_CORE:
        raise ImportError("moml.core package required for batch processing")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create processor with config
    try:
        processor = create_graph_processor(config)
    except Exception as e:
        logger.error(f"Failed to create graph processor: {e}")
        raise

    # Get list of molecule files
    mol_files = _find_molecule_files(input_dir, file_pattern)

    if not mol_files:
        logger.warning(
            f"No files found matching pattern '{file_pattern}' in {input_dir}"
        )
        return []

    logger.info(f"Found {len(mol_files)} molecule files to process")
    
    # Process each molecule file
    processed_files = []
    failed_count = 0
    
    for mol_file in tqdm(mol_files, desc="Processing molecules"):
        try:
            # Find corresponding charges file if charges_dir provided
            charges_file = None
            if charges_dir:
                charges_file = find_charges_file(mol_file, charges_dir)

            # Generate output file path
            base_name = os.path.splitext(os.path.basename(mol_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}{DEFAULT_GRAPH_SUFFIX}")

            # Process and save the file
            result_path = process_mol_file_to_graph(
                mol_file=mol_file,
                output_file=output_file,
                processor=processor,
                charges_file=charges_file
            )
            processed_files.append(result_path)

        except Exception as e:
            logger.error(f"Error processing {mol_file}: {e}")
            failed_count += 1
    
    # Log summary
    success_count = len(processed_files)
    logger.info(
        f"Batch processing complete: {success_count} successful, "
        f"{failed_count} failed"
    )

    # Save configuration for reference if provided
    if config:
        _save_config(output_dir, config)
    
    return processed_files


def _find_molecule_files(input_dir: str, file_pattern: str) -> List[str]:
    """
    Find molecule files matching the specified patterns.
    
    Args:
        input_dir: Directory to search
        file_pattern: Comma-separated file patterns
        
    Returns:
        List of matching file paths
    """
    mol_files = []
    for pattern in file_pattern.split(","):
        pattern = pattern.strip()
        files = glob.glob(os.path.join(input_dir, pattern))
        mol_files.extend(files)
    
    # Remove duplicates and sort
    return sorted(list(set(mol_files)))


def _save_config(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Save configuration to JSON file for reference.
    
    Args:
        output_dir: Directory to save config file
        config: Configuration dictionary
    """
    config_path = os.path.join(output_dir, DEFAULT_CONFIG_FILENAME)
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved preprocessing configuration to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save config to {config_path}: {e}")


def graph_batch_process(
    input_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    file_pattern: str = DEFAULT_SIMPLE_PATTERN,
) -> List[str]:
    """
    Simplified batch processing for pipeline integration.
    
    This function provides a streamlined interface for batch processing
    molecule files, specifically designed for use in pipeline contexts.
    It has fewer parameters than batch_process_molecules and uses logging
    instead of print statements.

    Args:
        input_dir: Directory containing molecule files
        output_dir: Directory to save processed graph files
        config: Optional configuration for graph processing
        file_pattern: Single file pattern to match molecule files

    Returns:
        List of paths to successfully processed graph files
    
    Example:
        >>> from moml.data.molecule_processors import graph_batch_process
        >>> processed = graph_batch_process(
        ...     'input/', 'output/',
        ...     config={'use_qm': True}
        ... )
    
    Note:
        This function does not support charges_dir parameter. Quantum
        properties should be handled via the config parameter.
    """
    if not HAS_CORE:
        raise ImportError("moml.core package required for batch processing")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create processor with config
    try:
        processor = create_graph_processor(config)
    except Exception as e:
        logger.error(f"Failed to create graph processor: {e}")
        raise

    # Get list of molecule files
    mol_files = glob.glob(os.path.join(input_dir, file_pattern))

    if not mol_files:
        logger.warning(
            f"No files found matching pattern '{file_pattern}' in {input_dir}"
        )
        return []

    logger.info(f"Processing {len(mol_files)} molecule files")

    # Process each molecule file
    processed_files = []
    failed_count = 0
    
    for mol_file in tqdm(mol_files, desc="Processing molecules"):
        try:
            # Generate output file path
            base_name = os.path.splitext(os.path.basename(mol_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}{DEFAULT_GRAPH_SUFFIX}")

            # Process and save the file
            result_path = process_mol_file_to_graph(
                mol_file=mol_file,
                output_file=output_file,
                processor=processor
            )
            processed_files.append(result_path)

        except Exception as e:
            logger.error(f"Error processing {mol_file}: {e}")
            failed_count += 1
    
    # Log summary
    success_count = len(processed_files)
    logger.info(
        f"Processed {success_count} molecules successfully, "
        f"{failed_count} failed"
    )

    # Save configuration for reference if provided
    if config:
        _save_config(output_dir, config)

    return processed_files


def process_dataset(
    csv_path: str,
    smiles_col: str = DEFAULT_SMILES_COLUMN,
    id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Process a CSV dataset containing molecular data with SMILES validation.
    
    This function loads a CSV file containing SMILES strings, validates each
    SMILES, and creates RDKit molecule objects for valid entries. It adds
    several columns with validation results and canonical SMILES.

    Args:
        csv_path: Path to CSV file containing molecular data
        smiles_col: Name of column containing SMILES strings
        id_col: Name of column containing molecule IDs (optional)

    Returns:
        DataFrame with added validation columns:
        - rdkit_mol: RDKit molecule objects (for valid SMILES)
        - canonical_smiles: Canonical SMILES representation
        - is_valid_smiles: Boolean indicating SMILES validity
        - smiles_error: Error message for invalid SMILES
    
    Example:
        >>> df = process_dataset('molecules.csv', smiles_col='SMILES')
        >>> valid_count = df['is_valid_smiles'].sum()
        >>> print(f"Valid SMILES: {valid_count}/{len(df)}")
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
        ImportError: If validation dependencies are missing
    """
    if not HAS_VALIDATION:
        raise ImportError("SMILES validation functionality not available")
    
    # Validate file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} compounds")
    except Exception as e:
        logger.error(f"Failed to load CSV file {csv_path}: {e}")
        raise

    # Check if SMILES column exists
    if smiles_col not in df.columns:
        error_msg = f"Column '{smiles_col}' not found in dataset"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize new columns for validation results
    df["rdkit_mol"] = None
    df["canonical_smiles"] = None
    df["is_valid_smiles"] = False
    df["smiles_error"] = None

    # Process each SMILES string
    valid_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating SMILES"):
        smiles = row[smiles_col]
        
        # Skip empty/null SMILES
        if pd.isna(smiles) or str(smiles) == "":  # type: ignore
            df.at[idx, "smiles_error"] = "Empty SMILES string"
            continue
        
        # Validate SMILES
        is_valid, canonical_smiles, mol_obj, error = validate_smiles(smiles)  # type: ignore[arg-type]

        # Store validation results
        df.at[idx, "is_valid_smiles"] = is_valid
        df.at[idx, "smiles_error"] = error

        if is_valid:
            df.at[idx, "canonical_smiles"] = canonical_smiles
            df.at[idx, "rdkit_mol"] = mol_obj
            valid_count += 1

    # Log summary
    logger.info(
        f"SMILES validation complete: {valid_count}/{len(df)} "
        f"({100*valid_count/len(df):.1f}%) valid"
    )
    
    return df


def save_processed_molecules(
    df: pd.DataFrame,
    output_dir: str,
    base_filename: str
) -> Dict[str, str]:
    """
    Save processed molecular data to disk in multiple formats.
    
    This function saves the processed DataFrame in several formats:
    - CSV file (without RDKit molecule objects)
    - Pickle file with valid molecule objects (if available)
    - Validation issues report (if there are invalid SMILES)

    Args:
        df: DataFrame with processed molecular data
        output_dir: Directory to save output files
        base_filename: Base name for output files (without extension)

    Returns:
        Dictionary mapping format names to file paths:
        - 'csv': Path to CSV file
        - 'pickle': Path to pickle file (if molecule objects exist)
        - 'issues_report': Path to validation issues report (if applicable)
    
    Example:
        >>> files = save_processed_molecules(df, 'output/', 'molecules')
        >>> print(files['csv'])  # 'output/molecules.csv'
    
    Raises:
        OSError: If directory creation or file writing fails
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}

    # Save as CSV (without RDKit mol objects for compatibility)
    csv_df = df.copy()
    if "rdkit_mol" in csv_df.columns:
        csv_df = csv_df.drop(columns=["rdkit_mol"])

    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    try:
        csv_df.to_csv(csv_path, index=False)
        output_files["csv"] = csv_path
        logger.info(f"Saved CSV data to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise

    # Save valid molecules as pickle file if available
    if "rdkit_mol" in df.columns and "is_valid_smiles" in df.columns:
        pickle_path = _save_molecule_pickle(df, output_dir, base_filename)
        if pickle_path:
            output_files["pickle"] = pickle_path

    # Generate validation issues report if applicable
    if "is_valid_smiles" in df.columns and not bool(df["is_valid_smiles"].all()):
        report_path = _save_validation_report(df, output_dir, base_filename)
        if report_path:
            output_files["issues_report"] = report_path

    logger.info(f"Saved processed data to {output_dir}")
    return output_files


def _save_molecule_pickle(
    df: pd.DataFrame,
    output_dir: str,
    base_filename: str
) -> Optional[str]:
    """
    Save valid molecule objects to pickle file.
    
    Args:
        df: DataFrame with molecule data
        output_dir: Output directory
        base_filename: Base filename
        
    Returns:
        Path to pickle file or None if failed
    """
    # Find appropriate ID column
    id_col = _find_id_column(df)
    if not id_col:
        logger.warning("No ID column found, skipping pickle save")
        return None
    
    try:
        # Create dictionary of valid molecules
        valid_molecules = {}
        valid_df = df[df["is_valid_smiles"]]
        
        for _, row in valid_df.iterrows():
            mol_id = row[id_col]
            mol_obj = row["rdkit_mol"]
            if mol_obj is not None:
                valid_molecules[mol_id] = mol_obj
        
        # Save to pickle file
        pickle_path = os.path.join(output_dir, f"{base_filename}{DEFAULT_MOLS_SUFFIX}")
        with open(pickle_path, "wb") as f:
            pickle.dump(valid_molecules, f)
        
        logger.info(f"Saved {len(valid_molecules)} molecules to {pickle_path}")
        return pickle_path
        
    except Exception as e:
        logger.error(f"Failed to save molecule pickle: {e}")
        return None


def _save_validation_report(
    df: pd.DataFrame,
    output_dir: str,
    base_filename: str
) -> Optional[str]:
    """
    Save validation issues report.
    
    Args:
        df: DataFrame with validation data
        output_dir: Output directory
        base_filename: Base filename
        
    Returns:
        Path to report file or None if failed
    """
    try:
        # Determine columns to include in report
        cols_to_include = ["smiles_error"]
        
        # Add ID column if available
        id_col = _find_id_column(df)
        if id_col:
            cols_to_include.insert(0, id_col)
        
        # Add original SMILES column
        smiles_col = _find_smiles_column(df)
        if smiles_col:
            cols_to_include.insert(-1, smiles_col)
        
        # Create report for invalid SMILES
        invalid_df = df[~df["is_valid_smiles"]]
        available_cols = [col for col in cols_to_include if col in invalid_df.columns]
        report_df = invalid_df[available_cols]
        assert isinstance(report_df, pd.DataFrame)  # Type assertion for linter
        
        # Save report
        report_path = os.path.join(
            output_dir, f"{base_filename}{DEFAULT_ISSUES_SUFFIX}"
        )
        report_df.to_csv(report_path, index=False)
        
        logger.info(f"Saved validation report to {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")
        return None


def _find_id_column(df: pd.DataFrame) -> Optional[str]:
    """Find appropriate ID column in DataFrame."""
    for col in ID_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Find SMILES column in DataFrame."""
    for col in df.columns:
        if "smiles" in col.lower() and col not in [
            "canonical_smiles", "is_valid_smiles"
        ]:
            return col
    return None


def batch_process_molecules_dataset(
    input_file: str,
    output_dir: str,
    config: Optional[Dict] = None,
    smiles_col: str = DEFAULT_SMILES_COLUMN,
    id_col: Optional[str] = None,
    calculate_descriptors: bool = True,
) -> Dict[str, str]:
    """
    Complete pipeline for processing molecular datasets from CSV.
    
    This function provides a complete pipeline for processing molecular
    datasets: loading CSV, validating SMILES, calculating descriptors,
    and saving results in multiple formats.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed files
        config: Configuration for processing (currently unused)
        smiles_col: Name of column containing SMILES strings
        id_col: Name of column containing molecule IDs
        calculate_descriptors: Whether to calculate molecular descriptors

    Returns:
        Dictionary with paths to saved files
    
    Example:
        >>> results = batch_process_molecules_dataset(
        ...     'molecules.csv', 'output/',
        ...     calculate_descriptors=True
        ... )
        >>> print(results['csv'])  # Path to processed CSV
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ImportError: If required dependencies are missing
    """
    logger.info(f"Starting dataset processing for {input_file}")

    # Process the dataset (validate SMILES)
    df = process_dataset(input_file, smiles_col, id_col)

    # Calculate descriptors if requested
    if calculate_descriptors and HAS_CORE:
        df = _add_molecular_descriptors(df)
    elif calculate_descriptors and not HAS_CORE:
        logger.warning("Cannot calculate descriptors: moml.core not available")
    
    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    base_filename += DEFAULT_PROCESSED_SUFFIX

    # Save processed data
    output_files = save_processed_molecules(df, output_dir, base_filename)
    
    logger.info("Dataset processing completed successfully")
    return output_files


def _add_molecular_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add molecular descriptors to DataFrame for valid molecules.
    
    Args:
        df: DataFrame with valid molecule objects
        
    Returns:
        DataFrame with added descriptor columns
    """
    logger.info("Calculating molecular descriptors for valid molecules")
    
    valid_mask = df["is_valid_smiles"]
    descriptor_count = 0
    
    for idx, row in tqdm(
        df[valid_mask].iterrows(),
        total=valid_mask.sum(),
        desc="Calculating descriptors"
    ):
        mol_obj = row["rdkit_mol"]
        if mol_obj is not None:
            try:
                descriptors = calculate_molecular_descriptors(mol_obj)
                for name, value in descriptors.items():
                    df.at[idx, name] = value
                descriptor_count += 1
            except Exception as e:
                logger.warning(f"Failed to calculate descriptors for row {idx}: {e}")
    
    logger.info(f"Calculated descriptors for {descriptor_count} molecules")
    return df


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for molecular preprocessing.
    
    Returns:
        Parsed command line arguments
    
    Example:
        $ python molecule_processors.py --input_dir molecules/ \
            --output_dir graphs/ --use_3d --add_h
    """
    parser = argparse.ArgumentParser(
        description="Preprocess molecule files into graph representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python molecule_processors.py --input_dir molecules/ --output_dir graphs/
  
  # With 3D coordinates and explicit hydrogens
  python molecule_processors.py --input_dir molecules/ --output_dir graphs/ \
    --use_3d --add_h
  
  # With partial charges
  python molecule_processors.py --input_dir molecules/ --output_dir graphs/ \
    --charges_dir charges/
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing molecule files (.mol or .sdf)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed graph files (.pt)"
    )
    parser.add_argument(
        "--charges_dir",
        type=str,
        default=None,
        help="Optional directory containing partial charge files"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default=DEFAULT_FILE_PATTERN,
        help="Pattern(s) to match molecule files (comma-separated)"
    )
    
    # Graph processing configuration
    parser.add_argument(
        "--use_qm",
        action="store_true",
        help="Include quantum mechanical properties in the graph"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Include 3D coordinates in the graph"
    )
    parser.add_argument(
        "--use_conformers",
        action="store_true",
        help="Generate and include multiple conformers"
    )
    parser.add_argument(
        "--add_h",
        action="store_true",
        help="Add explicit hydrogens to molecules"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Run molecular preprocessing from command line arguments.
    
    This function serves as the main entry point when the module is run
    as a script. It parses command line arguments and runs batch processing.
    
    Example:
        $ python -m moml.data.molecule_processors --input_dir molecules/ \
            --output_dir graphs/ --use_3d
    """
    args = parse_args()
    
    # Convert arguments to configuration dictionary
    config = {
        "use_qm": args.use_qm,
        "use_3d": args.use_3d,
        "use_conformers": args.use_conformers,
        "add_h": args.add_h,
    }
    
    try:
        # Process the molecules
        processed_files = batch_process_molecules(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
            charges_dir=args.charges_dir,
            file_pattern=args.file_pattern,
        )
        
        print(f"Successfully processed {len(processed_files)} molecules")
        print("Preprocessing completed successfully")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()