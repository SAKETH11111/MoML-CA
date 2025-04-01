# MoML: Molecular Machine Learning Library

MoML (Molecular Machine Learning) is a Python library for molecular graph representation learning and analysis, designed to streamline the process of preparing molecular data for machine learning applications.

## Core Features

- **Molecular Data Processing**: Validate and standardize molecular data from SMILES strings
- **Molecular Graph Generation**: Generate graph representations of molecules with configurable features
- **Graph Hierarchies**: Build hierarchical molecular graphs for coarse-grained modeling
- **Feature Engineering**: Extract meaningful chemical and physical features from molecular structures

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

- **`moml.pipeline`**: End-to-end processing pipelines
  - `orchestrator.py`: Coordinates the execution of multiple pipeline stages
  - `chemical_list/`: Pipeline components for chemical listing data

## Getting Started

```python
from moml.core import validate_smiles, calculate_molecular_descriptors, MolecularGraphProcessor
from moml.data import process_dataset

# Process a dataset of SMILES strings
df = process_dataset('path/to/molecules.csv', smiles_col='SMILES', id_col='ID')

# Create a molecular graph processor with QM support
processor = MolecularGraphProcessor(use_3d_coords=True, use_partial_charges=True)

# Process a molecule from the dataset
mol = df[df['is_valid_smiles']]['rdkit_mol'].iloc[0]
graph = processor.process_molecule(mol)

# Calculate molecular descriptors
descriptors = calculate_molecular_descriptors(mol)
print(f"Molecule has {descriptors['ring_count']} rings and {descriptors['h_donors']} H-donors")
```

## Code Organization Principles

MoML follows these design principles:

1. **Single Source of Truth**: Core functionality is consolidated in the `moml.core` module
2. **Separation of Concerns**: Each module and file has a distinct, well-defined purpose
3. **Composability**: Components can be used independently or composed into pipelines
4. **Backward Compatibility**: Legacy code is maintained with deprecation warnings

## Directory Structure

- **`core/`**: Core functionality for molecular graph representation and descriptors
- **`data/`**: Dataset management and processing utilities
- **`models/`**: Machine learning models for molecular property prediction
- **`pipeline/`**: Pipeline orchestration for end-to-end workflows
- **`simulation/`**: Molecular dynamics and quantum mechanical simulation interfaces
- **`utils/`**: General utilities for working with molecular data

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

This project is licensed under the MIT License - see the LICENSE file for details.
