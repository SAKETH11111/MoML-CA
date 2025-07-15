# MoML-CA Examples

This directory contains example scripts demonstrating how to use the MoML-CA (Molecular Machine Learning for Chemical Applications) library. Each subdirectory focuses on a specific aspect of the library's functionality.

## Directory Structure

- **`data_loading_example.py`**: Demonstration of loading and batching molecular graph data.
- **`graph_generation_examples/`**: Examples for creating and visualizing molecular graphs.
  - `molecular_graph_gen.py`: Unified tool for generating various molecular graph representations.
  - `hierarchical_graph_example.py`: Creating hierarchical molecular graphs.
- **`prediction_examples/`**: Examples for making predictions with trained models.
  - `run_predict.py`: Command-line interface for molecular property predictions.
- **`preprocessing_examples/`**: Examples for data preprocessing.
  - `preprocess_example.py`: Preprocessing molecular structures into graph representations.
- **`training_examples/`**: Examples for model training and evaluation.
  - `train_example.py`: Training MGNN models on molecular graph data.

## Data Loading Example

The `data_loading_example.py` script demonstrates how to use the `PFASDataLoader` to load and batch molecular graph data for MGNN training workflows.

### What You'll Learn

- Loading single molecules by ID.
- Creating batches from multiple molecule IDs.
- Configuring the data loader for specific environmental features and label types.

### Usage

```bash
# From the project root directory:
python -m examples.data_loading_example --data_dir ./data/example_dataset
```

## Graph Generation Examples

The graph generation examples demonstrate how to create various types of molecular graphs.

### `molecular_graph_gen.py`

A unified command-line tool for generating various types of molecular graph representations including atomic-level, hierarchical, and functional group analysis with optional quantum enhancement.

#### What You'll Learn

- Creating atomic-level graphs from molecule files.
- Generating hierarchical graphs (atom, functional group, structural motif).
- Analyzing functional groups in a molecule.
- Integrating quantum properties from ORCA output files.

#### Usage

```bash
# Generate an atomic-level graph with visualization
python -m examples.graph_generation_examples.molecular_graph_gen atomic --mol molecule.mol --orca output.out --visualize

# Generate hierarchical graphs
python -m examples.graph_generation_examples.molecular_graph_gen hierarchical --mol pfoa.mol --output_dir ./graphs

# Analyze functional groups
python -m examples.graph_generation_examples.molecular_graph_gen analyze --mol molecule.mol
```

### `hierarchical_graph_example.py`

A command-line tool for creating hierarchical molecular graph representations at different granularity levels (atom, functional group, structural motif) with optional quantum properties integration.

#### What You'll Learn

- Creating hierarchical graphs from a molecule file.
- Integrating quantum properties from ORCA output.
- Visualizing the generated hierarchical graphs.

#### Usage

```bash
# Create hierarchical graphs and visualize them
python -m examples.graph_generation_examples.hierarchical_graph_example --mol_file molecule.mol --visualize
```

## Prediction Examples

The prediction examples show how to use trained models for making predictions.

### `run_predict.py`

A command-line interface for making molecular property predictions using trained MGNN models with support for both single molecule and batch processing modes.

#### What You'll Learn

- Loading a trained model.
- Making predictions on a single molecule or a batch of molecules.
- Saving prediction results.

#### Usage

```bash
# Single molecule prediction
python -m examples.prediction_examples.run_predict --model_path model.pt --mol_file molecule.mol

# Batch prediction
python -m examples.prediction_examples.run_predict --model_path model.pt --mol_file ./molecules/ --batch_mode
```

## Preprocessing Examples

The preprocessing examples show how to prepare data for training.

### `preprocess_example.py`

A command-line tool for preprocessing molecular structures into graph representations suitable for machine learning with comprehensive feature statistics generation.

#### What You'll Learn

- Processing a directory of molecular files into graph representations.
- Generating feature statistics for model configuration.
- Saving processed graphs to disk for faster training.

#### Usage

```bash
# Preprocess a directory of molecules
python -m examples.preprocessing_examples.preprocess_example --input_dir ./molecules --output_dir ./processed
```

## Training Examples

The training examples demonstrate how to train models on molecular datasets.

### `train_example.py`

A command-line tool for training MGNN models on molecular graph data with comprehensive configuration options and checkpoint management.

#### What You'll Learn

- Configuring and training an MGNN model.
- Using training and validation datasets.
- Saving model checkpoints and training configuration.

#### Usage

```bash
# Train a model
python -m examples.training_examples.train_example --train_dir ./data/train --output_dir ./models
```

## Output

Running the examples will create:

- An `output` or specified output directory in your current working directory.
- Saved model files (`.pt` format).
- Visualization files (`.png` format).
- Preprocessed data files (`.pt` format).
- Log files and metrics.

## Additional Resources

- Check the [main documentation](../docs/) for detailed API reference.
- See the [tests](../tests/) directory for more usage examples.
- Visit the [project website](https://github.com/SAKETH11111/MoML-CA) for updates.
