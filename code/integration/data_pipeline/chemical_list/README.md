# PFAS Chemical List Data Preprocessing and Feature Engineering

This directory contains scripts for cleaning, preprocessing, and feature engineering of the PFAS Chemical List dataset.

## Overview

The PFAS Chemical List dataset contains information about per- and polyfluoroalkyl substances (PFAS), including their chemical structures, properties, and toxicity data. The preprocessing and feature engineering steps include:

1. Initial inspection of the dataset
2. Cleaning and standardization of column names
3. Handling missing values
4. Standardizing text data
5. Converting data types
6. Extracting fluorine counts and molecular complexity features
7. Categorizing PFAS compounds by structural types

## Scripts

### `clean_pfas_chemical_list.py`

This script performs the following operations:

- Loads the raw PFAS Chemical List dataset
- Inspects the dataset dimensions, data types, and missing values
- Renames columns for consistency (e.g., "IUPAC NAME" → "IUPAC_Name")
- Cleans the DTXSID column to extract IDs from URLs
- Converts numeric columns to appropriate data types
- Handles missing values:
  - For numerical columns: replaces NaN with median values
  - For categorical/text columns: replaces NaN with "Unknown"
- Standardizes text data by removing trailing spaces and special characters
- Creates derived features (e.g., binary flag for ToxCast activity)
- Saves the processed dataset to `data/processed/PFAS_Chemical_List_cleaned.csv`

### `engineer_pfas_features.py`

This script performs feature engineering on the cleaned PFAS Chemical List dataset:

- Creates RDKit molecule objects from SMILES strings
- Extracts fluorine counts and percentages
- Calculates molecular complexity features:
  - Carbon count
  - C-F bonds
  - Chain length (estimated by carbon count)
  - Molecular weight
  - Rotatable bonds
  - H-bond acceptors and donors
  - Ring count
  - Aromatic rings
- Categorizes PFAS compounds by structural types:
  - Chain length categories (short, medium, long)
  - Structural features (aromatic, cyclic, branched)
  - Fluorination level (lightly, moderately, highly)
  - Combined PFAS type
- Saves the engineered dataset to `data/processed/PFAS_Chemical_List_engineered.csv`

### `pfas_dataset_summary.py`

This script generates a summary report of the cleaned PFAS Chemical List dataset, including:

- Basic descriptive statistics
- Molecular properties analysis
- Toxicity data analysis
- Missing data handling analysis
- Visualization of key distributions

The summary plots are saved to `experiments/results/data_preprocessing/`.

### `visualize_pfas_features.py`

This script creates visualizations of the engineered PFAS features:

- Fluorine content visualizations:
  - Fluorine count distribution
  - Fluorine percentage distribution
  - Relationship between carbon count and fluorine count
  - Relationship between molecular weight and fluorine percentage
- Structural feature visualizations:
  - Chain length distribution
  - Chain category distribution
  - Structural features distribution
  - Ring count distribution
- PFAS type visualizations:
  - Top 10 PFAS types
  - Fluorination level distribution
  - Relationship between chain category and fluorination level
- Correlation visualizations:
  - Correlation matrix of numerical features

The visualizations are saved to `experiments/results/feature_engineering/`.

## Usage

To clean and preprocess the PFAS Chemical List dataset:

```bash
python code/integration/data_pipeline/clean_pfas_chemical_list.py
```

To perform feature engineering on the cleaned dataset:

```bash
python code/integration/data_pipeline/engineer_pfas_features.py
```

To generate a summary report of the cleaned dataset:

```bash
python code/integration/data_pipeline/pfas_dataset_summary.py
```

To create visualizations of the engineered features:

```bash
python code/integration/data_pipeline/visualize_pfas_features.py
```

## Data Cleaning Results

The cleaning process resulted in:

- Standardized column names for consistency
- Extracted DTXSID IDs from URLs
- Converted all numeric columns to appropriate data types
- Handled missing values in all columns
- Standardized text data by removing trailing spaces and special characters
- Created a binary flag for ToxCast activity

## Feature Engineering Results

The feature engineering process resulted in 17 new features:

- **Fluorine Content Features**:
  - `F_Count`: Number of fluorine atoms
  - `F_Percentage`: Percentage of fluorine atoms in the molecule
  
- **Molecular Complexity Features**:
  - `C_Count`: Number of carbon atoms
  - `CF_Bonds`: Number of carbon-fluorine bonds
  - `Chain_Length`: Estimated chain length (carbon count)
  - `MW_RDKit`: Molecular weight calculated by RDKit
  - `Rotatable_Bonds`: Number of rotatable bonds
  - `H_Acceptors`: Number of hydrogen bond acceptors
  - `H_Donors`: Number of hydrogen bond donors
  - `Ring_Count`: Number of rings
  - `Aromatic_Rings`: Number of aromatic rings
  
- **PFAS Type Categories**:
  - `Chain_Category`: Chain length category (Short-chain, Medium-chain, Long-chain)
  - `Is_Aromatic`: Whether the molecule is aromatic
  - `Has_Rings`: Whether the molecule has rings
  - `Is_Cyclic`: Whether the molecule is cyclic
  - `Is_Branched`: Whether the molecule is branched
  - `PFAS_Type`: Combined PFAS type category

## Key Insights

- The dataset contains 14,735 PFAS compounds
- The average fluorine count is 10.86 atoms per molecule
- The maximum fluorine count is 126 atoms
- The average fluorine percentage is 43.65%
- Most compounds (60.8%) are moderately fluorinated (25-50% fluorine)
- Most compounds (60.8%) are long-chain PFAS
- Most compounds (99.8%) are branched
- The most common PFAS type is "Long-chain Aromatic Branched Moderately-fluorinated"

## Next Steps

After feature engineering, the enriched dataset can be used for:

1. Machine learning model development for property prediction
2. Graph Neural Network (GNN) training for molecular property prediction
3. Structure-activity relationship analysis
4. Clustering and classification of PFAS compounds
5. Integration with other datasets for comprehensive PFAS analysis 