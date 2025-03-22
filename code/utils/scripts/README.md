# MoML-CA Utility Scripts

This directory contains executable scripts for various data processing, analysis, and visualization tasks in the MoML-CA project.

## PFAS-20 Data Standardization Scripts

- `process_pfas20_dataset.py`: Convert SMILES strings to standardized molecular representations
  - Validates SMILES strings
  - Generates canonical SMILES representations
  - Creates RDKit molecular objects
  - Calculates basic molecular descriptors
  - Saves processed data for further analysis

- `visualize_pfas20_mols.py`: Generate visualizations of processed molecular structures
  - Creates a grid visualization of all 20 compounds
  - Generates individual molecule images
  - Customizable display options (size, layout, etc.)

## Usage

### Processing PFAS-20 Dataset

```bash
python process_pfas20_dataset.py [options]
```

Options:
- `-i, --input`: Path to input CSV file (default: data/processed/pfas_final_dataset.csv)
- `-o, --output-dir`: Directory to save processed files (default: data/processed/chemical_list)
- `-b, --base-name`: Base filename for output files (default: pfas20_standardized)
- `-d, --calculate-descriptors`: Calculate basic molecular descriptors

### Visualizing PFAS-20 Molecules

```bash
python visualize_pfas20_mols.py [options]
```

Options:
- `-i, --input`: Path to pickle file with processed molecules (default: data/processed/chemical_list/pfas20_rdkit_mols.pkl)
- `-o, --output-dir`: Directory to save visualizations (default: data/processed/visualizations)
- `-g, --grid-filename`: Filename for the molecule grid image (default: pfas20_compounds_grid.png)
- `-c, --columns`: Number of columns in the molecule grid (default: 4)

## Dependencies

These scripts require the utilities provided in:
- `code.utils.helper_functions.molecular` for molecular processing
- `code.utils.plotting.molecules` for visualization

