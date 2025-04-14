# MoML-CA Examples

This directory contains example scripts demonstrating how to use the MoML-CA (Molecular Machine Learning for Chemical Applications) library. Each subdirectory focuses on a specific aspect of the library's functionality.

## Directory Structure

- **`quickstart/`**: Basic examples to get started with MoML-CA

  - `quickstart.py`: Comprehensive example of core functionality
  - `custom_model.py`: Example of implementing a custom model

- **`training/`**: Examples for model training and evaluation

  - `train_model.py`: Complete training pipeline example
  - `evaluate_model.py`: Model evaluation and metrics calculation
  - `hyperparameter_tuning.py`: Hyperparameter optimization example

- **`prediction/`**: Examples for making predictions with trained models

  - `prediction_example.py`: Basic prediction pipeline
  - `run_predict.py`: Command-line interface for predictions
  - `batch_predict.py`: Batch prediction on multiple molecules

- **`molecular_graph/`**: Examples for working with molecular graphs

  - `graph_creation.py`: Creating and manipulating molecular graphs
  - `graph_features.py`: Extracting and visualizing graph features
  - `hierarchical_graphs.py`: Working with hierarchical graph representations

- **`preprocess/`**: Examples for data preprocessing
  - `preprocessing_example.py`: Basic preprocessing pipeline
  - `batch_preprocess.py`: Batch processing of molecular datasets
  - `feature_extraction.py`: Extracting molecular features

## Quickstart Example

The `quickstart.py` script provides a comprehensive example showcasing the core functionality of MoML-CA:

```bash
# From the project root directory:
python -m examples.quickstart.quickstart
```

### What You'll Learn

- Converting molecules to graph representations
- Creating hierarchical molecular graphs
- Configuring and initializing a model
- Training a model with the provided trainer utilities
- Making predictions with a trained model
- Calculating metrics and visualizing results

## Training Examples

The training examples demonstrate how to train models on molecular datasets:

```bash
# Train a model
python -m examples.training.train_model --config config.yaml

# Evaluate a trained model
python -m examples.training.evaluate_model --model_path model.pt

# Perform hyperparameter tuning
python -m examples.training.hyperparameter_tuning --study_name "optimization_study"
```

### What You'll Learn

- Setting up training configurations
- Using callbacks and monitoring
- Implementing custom training loops
- Evaluating model performance
- Optimizing hyperparameters

## Prediction Examples

The prediction examples show how to use trained models for making predictions:

```bash
# Predict properties for a single molecule
python -m examples.prediction.run_predict --model_path model.pt --mol_file molecule.mol

# Batch prediction on multiple molecules
python -m examples.prediction.batch_predict --model_path model.pt --input_dir molecules/
```

### What You'll Learn

- Loading trained models
- Making predictions on new molecules
- Processing prediction results
- Visualizing predictions
- Batch processing for efficiency

## Molecular Graph Examples

The molecular graph examples demonstrate working with graph representations:

```bash
# Create and visualize molecular graphs
python -m examples.molecular_graph.graph_creation --smiles "CCO"

# Extract and analyze graph features
python -m examples.molecular_graph.graph_features --input_file molecules.csv

# Work with hierarchical graphs
python -m examples.molecular_graph.hierarchical_graphs --config graph_config.yaml
```

### What You'll Learn

- Creating molecular graphs from SMILES
- Extracting graph features
- Visualizing graph structures
- Working with hierarchical representations
- Analyzing graph properties

## Preprocessing Examples

The preprocessing examples show how to prepare data for training:

```bash
# Preprocess a dataset
python -m examples.preprocess.preprocessing_example --input_dir data/ --output_dir processed/

# Batch process multiple files
python -m examples.preprocess.batch_preprocess --config preprocess_config.yaml

# Extract molecular features
python -m examples.preprocess.feature_extraction --input_file molecules.csv
```

### What You'll Learn

- Preprocessing molecular datasets
- Extracting molecular features
- Creating graph representations
- Handling different file formats
- Optimizing preprocessing pipelines

## Output

Running the examples will create:

- An `output` directory in your current working directory
- Saved model files (`.pt` format)
- Visualization files (`.png` format)
- Preprocessed data files
- Log files and metrics

## Additional Resources

- Check the [main documentation](../docs/) for detailed API reference
- See the [tests](../tests/) directory for more usage examples
- Visit the [project website](https://github.com/yourusername/MoML-CA) for updates
