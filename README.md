# MoML-CA_PFAS: Molecular Machine Learning for PFAS Characterization

This repository contains the code and data for the MoML-CA (Molecular Machine Learning - Chemical Analysis) project focused on PFAS characterization.

## Project Structure

- **data/**: Contains all data used in the project
  - **raw/**: Original QM data, environmental measurements, etc.
  - **processed/**: Preprocessed and cleaned data files
  - **QM_data/**: High-quality QM results for selected PFAS molecules
  - **MD_simulations/**: Input files, scripts, and outputs from MD runs
  - **time_series/**: Extracted time-series data from MD for LSTM training

- **code/**: All source code for the project
  - **MGNN/**: Graph Neural Network component code
  - **MD/**: Molecular Dynamics simulation code
  - **LSTM/**: LSTM-based time-series modeling code
  - **integration/**: Pipeline code linking components
  - **utils/**: Utility functions and tools

- **experiments/**: Experiment-specific files
  - **notebooks/**: Jupyter notebooks for exploration
  - **scripts/**: Experiment scripts
  - **results/**: Output from experiments
  - **logs/**: Log files
  - **configs/**: Experiment configuration files

- **models/**: Saved model checkpoints and pre-trained models

- **config/**: Global configuration files

- **deployment/**: Files for real-world deployment

## Setup and Usage

[To be added]

## License

[To be added]