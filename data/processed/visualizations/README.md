# PFAS-20 Molecular Visualizations

This directory contains 2D visualizations of the 20 PFAS compounds in the dataset, generated using RDKit.

## Contents

- `pfas20_compounds_grid.png`: A grid visualization of all 20 PFAS compounds in the dataset, labeled with their common names.
- `individual/`: Directory containing individual PNG images for each PFAS compound.

## Purpose

These visualizations serve several purposes:

1. **Visual Inspection**: Allow researchers to visually inspect the structures of PFAS compounds
2. **Structural Comparison**: Enable visual comparison of different PFAS classes and functional groups
3. **Documentation**: Provide visual documentation of the compounds included in the study
4. **Presentations/Publications**: Ready-to-use images for presentations, reports, or publications

## Generation

These visualizations were generated using the `visualize_pfas20_mols.py` script in the `code/utils/scripts` directory. The script loads molecular objects from the processed pickle file and renders them using RDKit's drawing functionality.

## Visual Features

The molecular visualizations include:

- Atom labels (C, F, O, S, P, N, etc.)
- Bond types (single, double)
- Compound name labels
- 2D coordinates optimized for clarity

To regenerate or customize these visualizations, see the `code/utils/scripts/README.md` file for instructions on using the visualization tools. 