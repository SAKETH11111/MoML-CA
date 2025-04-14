# MoML: Molecular Machine Learning Library

MoML (Molecular Machine Learning) is a Python library for molecular graph representation learning and analysis, designed to streamline the process of preparing molecular data for machine learning applications.

## Core Features

- **Molecular Data Processing**: Validate and standardize molecular data from SMILES strings
- **Molecular Graph Generation**: Generate graph representations of molecules with configurable features
- **Graph Hierarchies**: Build hierarchical molecular graphs for coarse-grained modeling
- **Feature Engineering**: Extract meaningful chemical and physical features from molecular structures
- **Model Training**: Comprehensive training utilities with callbacks and monitoring
- **Prediction Pipeline**: End-to-end pipeline for making predictions on new molecules
- **Visualization**: Tools for visualizing molecular graphs and model attention

## Architecture

MoML is organized into several key modules:

- **`moml.core`**: Core functionality for molecular representations and graph processing

  - `molecular_graph.py`: Graph generation and processing for molecules
  - `graph_coarsening.py`: Hierarchical graph generation and coarsening
  - `molecular_descriptors.py`: Feature extraction and molecular property calculation

- **`moml.data`**: Data handling and processing utilities

  - `dataset.py`: Dataset classes for molecular data
  - `loader.py`: Functions for loading datasets from files
  - `splitting.py`: Utilities for splitting datasets (training/validation/test)
  - `processors/`: Dataset-specific processing functions

- **`moml.models`**: Machine learning models and training utilities

  - `mgnn/`: Multi-level Graph Neural Network implementations
  - `lstm/`: LSTM-based models for sequential data
  - `training/`: Training utilities and callbacks
  - `evaluation/`: Model evaluation and metrics

- **`moml.pipeline`**: End-to-end processing pipelines

  - `orchestrator.py`: Coordinates the execution of multiple pipeline stages
  - `chemical_list/`: Pipeline components for chemical listing data

- **`moml.utils`**: Utility functions and tools
  - `visualization/`: Tools for visualizing molecular graphs and results
  - `molecular/`: Molecular manipulation utilities
  - `graph/`: Graph processing utilities

## Getting Started

```python
from moml import create_graph_processor, initialize_model, create_trainer
from moml.models.mgnn import MGNNConfig
from moml.models.mgnn.evaluation import create_predictor

# Create a molecular graph processor
processor = create_graph_processor({
    'use_3d_coords': True,
    'use_partial_charges': True
})

# Process a molecule
smiles = "C(C(F)(F)F)(C(F)(F)F)(F)F"  # Perfluorobutane
graph = processor.smiles_to_graph(smiles)

# Initialize and train a model
config = MGNNConfig({
    'model_type': 'multi_task_djmgnn',
    'hidden_dim': 64,
    'n_blocks': 3
})
model = initialize_model(config, graph.x.shape[1], graph.edge_attr.shape[1])

# Train the model
trainer = create_trainer(model, config, train_loader, val_loader)
history = trainer.train(epochs=50)

# Make predictions
predictor = create_predictor(model)
predictions = predictor.predict([graph])
```

## Code Organization Principles

MoML follows these design principles:

1. **Single Source of Truth**: Core functionality is consolidated in the `moml.core` module
2. **Separation of Concerns**: Each module and file has a distinct, well-defined purpose
3. **Composability**: Components can be used independently or composed into pipelines
4. **Backward Compatibility**: Legacy code is maintained with deprecation warnings
5. **Type Safety**: Comprehensive type hints and validation
6. **Error Handling**: Clear error messages and graceful failure modes

## Development Guidelines

To maintain code quality and avoid reintroducing redundancies, please follow these guidelines:

1. **Use the main package imports**

   - Import from the top-level package (`import moml`) when possible
   - Example: `from moml import MolecularGraphProcessor, create_graph_processor`

2. **Avoid duplicating functionality**

   - Check if a function already exists before implementing a new one
   - Extend existing classes rather than creating parallel implementations
   - Add tests for new functionality

3. **Keep the codebase simple**

   - Don't create separate modules for similar functionality
   - Place related functionality in the same module
   - Follow the "There should be one-- and preferably only one --obvious way to do it" principle

4. **Improve documentation**
   - Keep docstrings up to date and comprehensive
   - Include examples in docstrings for complex functions
   - Update README files when adding significant features

## Usage Examples

See the `examples/` directory for usage examples and tutorials.

## Dependencies

- RDKit
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
