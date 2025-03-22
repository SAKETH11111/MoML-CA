# Molecular Helper Functions

This directory contains utility functions for processing and handling molecular data for the MoML-CA project.

## Contents

- `molecule_processing.py`: Core functionality for processing SMILES strings and molecular data
  - SMILES validation and canonicalization
  - Molecular structure processing
  - Basic descriptor calculation
  - File I/O for molecular data

## Usage

These utilities are used by various scripts and modules within the MoML-CA project for:

1. Processing chemical input data from various sources
2. Standardizing molecular representations for consistent handling
3. Calculating essential molecular properties
4. Converting between molecular formats (SMILES, RDKit objects, etc.)

For examples of how to use these functions, see the processing scripts in the `code/utils/scripts` directory.

## Dependencies

- RDKit
- Pandas
- NumPy 