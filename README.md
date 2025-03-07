# MoML-CA_PFAS

## Multi-scale Modeling of PFAS using Machine Learning and Computational Approaches

This repository contains code and data for modeling Per- and Polyfluoroalkyl Substances (PFAS) using a multi-scale approach that combines Graph Neural Networks (GNNs), Molecular Dynamics (MD) simulations, and Long Short-Term Memory (LSTM) networks.

## Project Overview

PFAS are a group of man-made chemicals that have been manufactured and used in a variety of industries around the world since the 1940s. Due to their persistence in the environment and potential adverse human health effects, there is significant interest in understanding their behavior and developing effective remediation strategies.

This project aims to:

1. Predict molecular properties of PFAS compounds using Message-Passing Graph Neural Networks (MGNNs)
2. Simulate PFAS behavior in aqueous environments using Molecular Dynamics
3. Forecast long-term environmental fate using LSTM networks trained on time-series data

## Repository Structure

```
MoML-CA_PFAS/
├── data/                      # Data storage
├── code/                      # Source code for all components
├── experiments/               # Experiment-specific files
├── models/                    # Saved model checkpoints
├── config/                    # Configuration files
└── deployment/                # Deployment-related files
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for MD simulations and deep learning)
- OpenMM 7.5+ (for MD simulations)
- PyTorch 1.9+ (for deep learning components)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/MoML-CA_PFAS.git
   cd MoML-CA_PFAS
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline

To run the complete modeling pipeline:

```
python experiments/scripts/run_pipeline.py --config config/training_config.yaml
```

### Running Individual Components

#### MGNN Training

```
python code/MGNN/training/train_mgnn.py --config config/hyperparameters/mgnn_config.yaml
```

#### MD Simulations

```
python code/MD/simulation_setup/setup_simulation.py --molecule [MOLECULE_ID] --config config/simulation_config.json
```

#### LSTM Training

```
python code/LSTM/training/train_lstm.py --data data/time_series/[DATASET].csv --config config/hyperparameters/lstm_config.yaml
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work is supported by [Funding Agency/Grant Number]
- We thank [Collaborators/Institutions] for their contributions and support. 