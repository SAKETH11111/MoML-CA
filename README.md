# MoML-CA: Molecular Machine Learning for Chemical Applications

MoML-CA is a Python package for molecular representation learning and property prediction using Graph Neural Networks. The package provides a comprehensive set of tools for converting molecular structures to graph representations, training GNN models, and predicting molecular properties.

## Features

- **Molecular Graph Creation**: Convert SMILES and RDKit molecules to graph representations with extensive feature extraction
- **Hierarchical Graph Representations**: Create multi-level graph representations for improved model performance
- **Modular Model Architecture**: Flexible and extensible GNN architectures with easy configuration
- **Training Utilities**: Comprehensive training pipelines with callbacks and monitoring
- **Evaluation Tools**: Metrics calculation and visualization of predictions
- **Example Scripts**: Ready-to-use examples for common molecular machine learning tasks
- **Command-Line Tools**: Easy-to-use CLI for model training and prediction
- **Data Processing**: Efficient batch processing of molecular datasets
- **Visualization**: Tools for visualizing molecular graphs and model predictions

## Large Files Handling

Large data files (>100MB) like training datasets and models are not stored in the Git repository. These files are ignored by Git via the `.gitignore` file and should be shared via alternative methods (cloud storage, direct transfer, etc.).

Large files in the `data/qm9/processed/` directory (particularly `*.pt` files) are automatically excluded from Git.

## Installation

```bash
# Clone the repository (choose HTTPS or SSH)
git clone https://github.com/SAKETH11111/MoML-CA.git
# or, if you have SSH keys configured:
# git clone git@github.com:SAKETH11111/MoML-CA.git
cd MoML-CA

# Create a conda environment
conda env create -f environment.yml

# Activate the environment
conda activate moml-ca

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import torch
from rdkit import Chem
from moml.core import create_graph_processor
from moml.models.mgnn.training import initialize_model, MGNNConfig, create_trainer
from moml.models.mgnn.evaluation.predictor import create_predictor

# Create molecular graph
processor = create_graph_processor({'use_partial_charges': True})
smiles = "C(C(F)(F)F)(C(F)(F)F)(F)F"  # Perfluorobutane
graph = processor.smiles_to_graph(smiles)

# Initialize model with configuration
config = MGNNConfig({
    'model_type': 'multi_task_djmgnn',
    'hidden_dim': 64,
    'n_blocks': 3
})
model = initialize_model(config, graph.x.shape[1], graph.edge_attr.shape[1])

# Train model with dataloaders
trainer = create_trainer(config=config, train_loader=train_loader, val_loader=val_loader)
# Note: train_loader and val_loader should be PyTorch DataLoader objects containing your training and validation datasets.
# See the examples directory (examples/training_examples or examples/quickstart_examples) for how to create these dataloaders.
# Example:
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
history = trainer.train(epochs=50)

# Make predictions
predictor = create_predictor(model_path="path/to/saved_model.pt")  # Or pass model directly
predictions = predictor.predict_from_dataloader(val_loader)  # Or predictor.predict([graph])
```

See the [examples directory](examples) for more comprehensive examples.

### Generating force field labels

After running ORCA calculations you can generate a JSON file containing atom
types, partial charges and other force field parameters for each PFAS molecule:

```bash
python scripts/generate_force_field_labels.py
```

The output `force_field_labels.json` will be placed in
`orca_results_b3lyp_sto3g/`.

## Project Structure

```
MoML-CA/
├── moml/                        # Main package directory
│   ├── core/                    # Core functionality
│   │   ├── graph_coarsening.py      # Graph coarsening algorithms
│   │   └── molecular_graph.py       # Molecular graph representation
│   ├── models/                  # Model implementations
│   │   ├── mgnn/                    # MGNN models
│   │   │   ├── djmgnn.py               # DJMGNN implementation
│   │   │   ├── training/               # Training utilities
│   │   │   └── evaluation/             # Evaluation utilities
│   │   └── lstm/                    # LSTM models
│   ├── data/                    # Data handling utilities
│   │   ├── dataset.py               # Dataset implementations
│   │   └── processors.py            # Data processors
│   ├── utils/                   # Utility functions
│   │   ├── visualization/           # Visualization tools
│   │   ├── molecular/               # Molecular utilities
│   │   └── graph/                   # Graph utilities
│   ├── pipeline/                # Pipeline orchestration
│   ├── simulation/              # Simulation utilities
│   └── __init__.py              # Package initialization
├── examples/                    # Example scripts
│   ├── quickstart/              # Quickstart examples
│   ├── training/                # Training examples
│   ├── prediction/              # Prediction examples
│   ├── molecular_graph/         # Molecular graph examples
│   └── preprocess/              # Preprocessing examples
└── tests/                       # Test directory
```

## Recent Improvements

- **Enhanced Model Architecture**: Improved hierarchical graph representations and attention mechanisms
- **Streamlined API**: Simplified interface with factory functions and better error handling
- **Advanced Training Features**: Added support for mixed precision training and gradient accumulation
- **Improved Data Processing**: Enhanced batch processing and memory efficiency
- **Better Visualization**: New tools for visualizing molecular graphs and model attention
- **Command-Line Interface**: Added CLI tools for common tasks
- **Documentation**: Comprehensive documentation with examples and tutorials

## Documentation

See the [docs](docs/) directory for comprehensive documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For guidelines on contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the terms of the MIT license.
