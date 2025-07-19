"""
moml/simulation/__init__.py

This package exposes the main interfaces for the simulation tools, including
utilities for both molecular dynamics (MD) and quantum mechanics (QM).
"""

# Molecular dynamics helpers
from .molecular_dynamics.force_field.mapper import (
    ForceFieldMapper,
    create_force_field_mapper,
)

# Quantum mechanics (ORCA) parsers and I/O
from .qm.parser import (
    parse_orca_output,
    extract_partial_charges_from_orca,
    smiles_to_3d_structure,
    create_orca_input,
    run_orca_calculation,
    process_molecule,
    batch_process_molecules,
)

__all__ = [
    # MD exports
    "ForceFieldMapper",
    "create_force_field_mapper",
    # QM exports
    "parse_orca_output",
    "extract_partial_charges_from_orca",
    "smiles_to_3d_structure",
    "create_orca_input",
    "run_orca_calculation",
    "process_molecule",
    "batch_process_molecules",
]
