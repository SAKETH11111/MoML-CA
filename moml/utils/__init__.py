"""
moml/utils/__init__.py

This package provides utility functions for data processing, validation,
and molecular analysis in the MoML-CA framework.
"""

# Data processing utilities
from .data_utils.data_processing import (
    load_data,
    inspect_data,
    clean_column_names,
    convert_numeric_columns,
    handle_missing_values,
    standardize_text_data,
    extract_numeric_from_text,
    save_processed_data,
)

# SMILES validation
from .data_utils.validation import validate_smiles

# Molecular utilities
from .data_utils.molecular import (
    create_rdkit_mols,
    extract_fluorine_count,
    calculate_molecular_complexity,
    calculate_molecular_properties,
    categorize_molecular_features,
    add_fluorinated_group_counts,
)

__all__ = [
    # Data processing
    "load_data",
    "inspect_data",
    "clean_column_names",
    "convert_numeric_columns",
    "handle_missing_values",
    "standardize_text_data",
    "extract_numeric_from_text",
    "save_processed_data",
    # Validation
    "validate_smiles",
    # Molecular utilities
    "create_rdkit_mols",
    "extract_fluorine_count",
    "calculate_molecular_complexity",
    "calculate_molecular_properties",
    "categorize_molecular_features",
    "add_fluorinated_group_counts",
]
