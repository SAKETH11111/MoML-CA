# MoML-CA Utilities

This directory contains utility modules, functions, and scripts used throughout the MoML-CA project for Per- and Polyfluoroalkyl Substances (PFAS) analysis.

## Directory Structure

```
utils/
├── helper_functions/     # Reusable helper functions
│   ├── molecular/        # Molecular data processing utilities
│   └── ...               # Other helper function categories
├── plotting/             # Visualization utilities
│   ├── molecules/        # Molecular structure visualization
│   └── ...               # Other plotting utilities
└── scripts/              # Executable utility scripts
```

## Core Functionality

The utilities provide the following core functionality:

1. **Molecular Data Processing**: SMILES validation, standardization, and conversion to computational objects
2. **Property Calculation**: Calculation of molecular descriptors and properties
3. **Visualization**: Generation of 2D molecular structure representations and other visualizations
4. **Data Conversion**: Tools for converting between different data formats

## Key Components

### Helper Functions

- **Molecular Processing**: Functions for handling molecular data
  - SMILES validation and canonicalization
  - Molecular object generation
  - Basic descriptor calculations

### Plotting Utilities

- **Molecule Visualization**: Tools for generating molecular structure visualizations
  - 2D structure generation and rendering
  - Individual and grid visualizations
  - Image file generation

### Utility Scripts

- **Data Processing**: Scripts for processing datasets
  - `process_pfas_dataset.py`: Convert SMILES to standardized molecular representations
  - `visualize_pfas_molecules.py`: Generate visualizations of molecular structures

## Usage

Each subdirectory contains its own README with specific usage instructions for the utilities in that directory. For a general overview:

1. **Molecular Processing**: Functions for standardizing and validating SMILES strings
2. **Visualization**: Tools for generating 2D molecular structure visualizations
3. **Scripts**: Executables for specific data processing and visualization tasks

## Dependencies

These utilities rely on:
- RDKit for molecular processing and visualization
- NumPy and Pandas for data handling
- Matplotlib for visualization
- PIL (Pillow) for image processing 