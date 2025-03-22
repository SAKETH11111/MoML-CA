# Molecule Visualization Utilities

This directory contains utilities for visualizing molecular structures, particularly for PFAS compounds in the MoML-CA project.

## Contents

- `molecule_visualization.py`: Functions for generating 2D molecular structure visualizations
  - Generation of 2D molecular coordinates
  - Single molecule rendering
  - Grid-style visualizations of multiple compounds
  - Saving molecule images to files

## Features

- 2D structure generation with optimized coordinates
- Customizable rendering options (size, labels, etc.)
- Grid layouts for comparing multiple structures
- Support for both individual and batch visualization

## Usage

These visualization utilities are used to:

1. Generate visual representations of chemical compounds
2. Create publication-quality molecular structure images
3. Provide visual tools for dataset exploration
4. Support structure-based analysis with visual verification

For examples of how to use these visualization functions, see the `visualize_pfas_molecules.py` script in the `code/utils/scripts` directory.

## Dependencies

- RDKit
- Matplotlib
- NumPy
- PIL (Pillow) 