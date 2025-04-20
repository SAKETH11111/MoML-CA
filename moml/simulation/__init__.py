"""
MoML Simulation Package

Public API:
- ForceFieldMapper, create_force_field_mapper: MD force field mapping
- parse_orca_output, extract_partial_charges_from_orca, extract_orbital_contributions_from_orca, extract_electrostatic_potential_from_orca: ORCA parser functions
- smiles_to_3d_structure, create_orca_input, run_orca_calculation: ORCA input/output utilities
- process_molecule, batch_process_molecules: high-level QM workflows
"""

# Molecular dynamics helpers
from .molecular_dynamics.force_field_mapper import (
    ForceFieldMapper,
    create_force_field_mapper,
)

# Quantum mechanics (ORCA) parsers and I/O
from .quantum_mechanics.parser.orca_parser import (
    parse_orca_output,
    extract_partial_charges_from_orca,
    extract_orbital_contributions_from_orca,
    extract_electrostatic_potential_from_orca,
    smiles_to_3d_structure,
    create_orca_input,
    run_orca_calculation,
    process_molecule,
    batch_process_molecules,
)

__all__ = [
    'ForceFieldMapper',
    'create_force_field_mapper',
    'parse_orca_output',
    'extract_partial_charges_from_orca',
    'extract_orbital_contributions_from_orca',
    'extract_electrostatic_potential_from_orca',
    'smiles_to_3d_structure',
    'create_orca_input',
    'run_orca_calculation',
    'process_molecule',
    'batch_process_molecules',
]