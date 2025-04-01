# MoML-CA: Molecular Machine Learning for Chemical Applications

MoML-CA is a Python package for molecular representation learning and property prediction using Graph Neural Networks. The package provides a comprehensive set of tools for converting molecular structures to graph representations, training GNN models, and predicting molecular properties.

## Features

- **Molecular Graph Creation**: Convert SMILES and RDKit molecules to graph representations with extensive feature extraction
- **Hierarchical Graph Representations**: Create multi-level graph representations for improved model performance
- **Modular Model Architecture**: Flexible and extensible GNN architectures with easy configuration
- **Training Utilities**: Comprehensive training pipelines with callbacks and monitoring
- **Evaluation Tools**: Metrics calculation and visualization of predictions
- **Example Scripts**: Ready-to-use examples for common molecular machine learning tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MoML-CA.git
cd MoML-CA

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import torch
from rdkit import Chem
from code.MGNN import (
    create_graph_processor,
    initialize_model,
    create_trainer,
    create_predictor,
    MGNNConfig
)

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
trainer = create_trainer(model, config, train_loader, val_loader)
history = trainer.train(epochs=50)

# Make predictions
predictor = create_predictor(model)
predictions = predictor.predict([graph])
```

See the [examples directory](code/MGNN/examples) for more comprehensive examples.

## Project Structure

```
MoML-CA/
├── code/
│   └── MGNN/                      # Main package directory
│       ├── architectures/         # Model architectures and components
│       │   ├── graph_coarsening.py     # Graph coarsening algorithms
│       │   └── molecular_graph.py      # Molecular graph representation
│       ├── models/                # Model implementations
│       │   ├── base.py                 # Base model class
│       │   └── multi_task_djmgnn.py    # Multi-task DJMGNN implementation
│       ├── training/              # Training utilities
│       │   ├── callbacks.py            # Training callbacks
│       │   └── trainer.py              # Model trainer
│       ├── evaluation/            # Evaluation utilities
│       │   ├── metrics.py              # Metrics calculation
│       │   └── visualization.py        # Visualization tools
│       ├── data/                  # Data handling utilities
│       │   ├── dataset.py              # Dataset implementations
│       │   └── transforms.py           # Data transforms
│       ├── utils/                 # Utility functions
│       │   ├── config.py               # Configuration handling
│       │   └── functional.py           # Functional utilities
│       ├── examples/              # Example scripts
│       │   ├── README.md               # Examples documentation
│       │   └── quickstart.py           # Quickstart example
│       └── __init__.py            # Package initialization
└── tests/                        # Test directory
```

## Recent Improvements

- **Modular API**: Simplified API with factory functions for common operations
- **Improved Model Architecture**: Enhanced model architectures with hierarchical graph representations
- **Enhanced Training Pipeline**: Flexible training utilities with callbacks and monitoring
- **Comprehensive Examples**: Ready-to-use examples for common tasks
- **Better Documentation**: Improved documentation and code comments

## Documentation

See the [docs](docs/) directory for comprehensive documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
