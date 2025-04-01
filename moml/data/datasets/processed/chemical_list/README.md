# PFAS-20 Dataset

This directory contains the standardized molecular representations of the 20 PFAS compounds in the dataset. These standardized representations serve as the foundation for all subsequent computational analyses in the MoML-CA project.

## Files

- `pfas20_standardized.csv`: Contains all original data plus additional columns:
  - `canonical_smiles`: Standardized SMILES notation in canonical form
  - `is_valid_smiles`: Boolean indicating if the SMILES string is valid
  - `smiles_error`: Error message if the SMILES failed validation (empty if valid)
  - `molecular_weight`: Calculated molecular weight
  - `logp`: Calculated octanol-water partition coefficient
  - `num_heavy_atoms`: Number of non-hydrogen atoms
  - `num_rotatable_bonds`: Number of rotatable bonds

- `pfas20_rdkit_mols.pkl`: Pickle file containing a dictionary mapping compound names to RDKit molecular objects. This is the primary resource for computational modeling.

## Data Summary

The dataset contains 20 PFAS compounds with the following characteristics:

- All 20 compounds (100%) had valid SMILES strings that were successfully converted to molecular objects
- The compounds represent various PFAS classes:
  - Legacy PFAS (PFOA, PFOS, etc.)
  - Alternative/replacement PFAS (GenX, ADONA, etc.)
  - PFAS precursors
- The dataset includes various functional groups:
  - Carboxylic acids
  - Sulfonic acids
  - Phosphonic acids
  - Sulfonamides
  - Others

## Usage

This standardized dataset serves as input for:

1. Molecular descriptor calculation
2. Molecular dynamics simulations
3. Machine learning model development
4. Structure-activity relationship studies

## Visualizations

Visualizations of these molecular structures can be found in the `data/processed/visualizations` directory, including:

- A grid representation of all compounds (`pfas20_compounds_grid.png`)
- Individual molecular structure images for each compound 