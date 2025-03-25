# MoML-CA PFAS Data Pipeline

This directory contains the data processing pipeline for the MoML-CA PFAS project. The pipeline is organized into separate modules for each dataset, with shared functionality for cross-dataset analysis.

## Directory Structure

### Code Organization

```
data_pipeline/
├── chemical_list/                      # PFAS Chemical List processing
│   ├── process_chemical_data.py        # Data cleaning and feature engineering
│
├── treatment_data/                     # PFAS Treatment Data processing
│   ├── process_treatment_data.py       # Data cleaning and feature engineering
│
├── analyze_pfas_data.py                # Cross-dataset analysis and alignment
└── README.md                           # This file
```

### Data Organization

```
data/
├── raw/                                # Raw input data
│   ├── PFAS_Chemical_List.csv          # Raw chemical list data
│   └── PFAS_Treatment_Data.csv         # Raw treatment data
│
└── processed/                          # Processed output data
    ├── chemical_list/                  # Processed chemical data
    │   ├── PFAS_Chemical_List_cleaned.csv
    │   └── PFAS_Chemical_List_engineered.csv
    │
    ├── treatment_data/                 # Processed treatment data
    │   └── PFAS_Treatment_Data_cleaned.csv
    │
    └── aligned/                        # Aligned datasets
        └── PFAS_Aligned_Data.csv
```

### Results Organization

```
experiments/results/
├── chemical_list/                      # Chemical list analysis results
├── treatment_data/                     # Treatment data analysis results
└── cross_analysis/                     # Cross-dataset analysis results
```

## Datasets

The pipeline processes two main datasets:

1. **PFAS Chemical List**: Contains information about per- and polyfluoroalkyl substances (PFAS), including their chemical structures, properties, and toxicity data.

2. **PFAS Treatment Data**: Contains information about different treatment processes applied to PFAS chemicals, along with their effectiveness.

## Workflow

The data processing workflow follows these steps:

1. **Data Cleaning**: Raw data is loaded, inspected, and cleaned to handle missing values, standardize formats, and ensure consistency.

2. **Feature Engineering**: Additional features are derived from the cleaned data to enhance analysis and modeling capabilities.

3. **Data Analysis**: Exploratory data analysis is performed to understand patterns, distributions, and relationships in the data.

4. **Data Alignment**: The datasets are aligned based on common identifiers (CASRN) to enable integrated analysis.

## Usage

### PFAS Chemical List Processing

```bash
# Process the chemical list data (cleaning and feature engineering)
python code/integration/data_pipeline/chemical_list/process_chemical_data.py

# For specific steps only:
python code/integration/data_pipeline/chemical_list/process_chemical_data.py --mode clean
python code/integration/data_pipeline/chemical_list/process_chemical_data.py --mode engineer
```

### PFAS Treatment Data Processing

```bash
# Process the treatment data
python code/integration/data_pipeline/treatment_data/process_treatment_data.py
```

### Cross-Dataset Analysis

```bash
# Perform analysis across both datasets
python code/integration/data_pipeline/analyze_pfas_data.py
```

## Output

The processed datasets are saved to the following locations:

- `data/processed/chemical_list/PFAS_Chemical_List_cleaned.csv`: Cleaned chemical list data
- `data/processed/chemical_list/PFAS_Chemical_List_engineered.csv`: Chemical list with engineered features
- `data/processed/treatment_data/PFAS_Treatment_Data_cleaned.csv`: Cleaned treatment data
- `data/processed/aligned/PFAS_Aligned_Data.csv`: Aligned dataset combining both sources

Visualizations and analysis results are saved to:

- `experiments/results/chemical_list/`: Chemical list analysis results
- `experiments/results/treatment_data/`: Treatment data analysis results
- `experiments/results/cross_analysis/`: Cross-dataset analysis results 
