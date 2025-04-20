"""
Quantum Calculations Module

This package provides utilities for quantum mechanical calculations and parsing using ORCA.
"""

from .orca_parser import (
    parse_orca_output,
    extract_partial_charges_from_orca,
    extract_orbital_contributions_from_orca,
    process_molecule,
    batch_process_molecules
)

__all__ = [
    "parse_orca_output",
    "extract_partial_charges_from_orca",
    "extract_orbital_contributions_from_orca",
    "process_molecule",
    "batch_process_molecules"
] 