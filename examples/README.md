# MoML-CA Examples

This directory contains example scripts demonstrating how to use the MoML-CA (Molecular Machine Learning for Chemical Applications) library.

## Quickstart Example

The `quickstart.py` script provides a comprehensive example showcasing the core functionality of MoML-CA:

1. Creating molecular graphs from SMILES
2. Building hierarchical molecular graphs using GraphCoarsener
3. Training a model with synthetic data
4. Making predictions and evaluating results

### Running the Quickstart Example

```bash
# From the project root directory:
python -m examples.quickstart.quickstart

# Alternatively, if the script is executable:
./examples/quickstart/quickstart.py
```

### What You'll Learn

The quickstart example demonstrates:

- How to convert molecules to graph representations
- Creating hierarchical molecular graphs for improved performance
- Configuring and initializing a model
- Training a model with the provided trainer utilities
- Making predictions with a trained model
- Calculating metrics and visualizing results

## Custom Model Example

The `custom_model.py` script demonstrates how to implement and use a custom model architecture with the MoML-CA framework:

```bash
# From the project root directory:
python -m examples.custom_model
```

### What You'll Learn

The custom model example demonstrates:

- How to implement a custom GNN architecture compatible with the MoML-CA framework
- Integrating custom model components (GCN layers, residual connections, different pooling strategies)
- Creating a synthetic classification dataset
- Training a custom model using the MoML-CA trainer
- Making predictions and evaluating classification results
- Visualizing the classification performance

## Preprocessing Example

The `preprocessing_example.py` script demonstrates how to preprocess molecule files into graph representations for machine learning:

```bash
# From the project root directory:
python -m examples.preprocess.preprocessing_example --input_dir path/to/molecules --output_dir path/to/output
```

### What You'll Learn

The preprocessing example demonstrates:

- How to preprocess multiple molecule files in batch
- Configuring the preprocessing pipeline for different molecular features
- Working with 3D coordinates and partial charges
- Creating serialized graph representations for faster loading
- How preprocessed data connects with the dataset/dataloader system

## Prediction Example

The `prediction_example.py` script demonstrates how to use a trained model to make predictions on molecules:

```bash
# From the project root directory:
python -m examples.prediction.prediction_example --model_path path/to/model.pt --output_dir output
```

### What You'll Learn

The prediction example demonstrates:

- How to load a trained model and create a predictor
- Making predictions on a single molecule using SMILES
- Batch prediction on multiple molecules
- Processing a directory of molecule files for prediction
- Saving and interpreting prediction results

## Command-Line Prediction

The `run_predict.py` script provides a command-line interface for making predictions with trained models:

```bash
# Predict properties for a single molecule
python -m examples.prediction.run_predict --model_path path/to/model.pt --mol_file path/to/molecule.mol

# Predict properties for all molecules in a directory
python -m examples.prediction.run_predict --model_path path/to/model.pt --mol_file path/to/molecules/ --batch_mode
```

This script makes it easy to run predictions from the command line without writing any code, and supports both single-molecule and batch processing modes.

## Output

Running the examples will create:

- An `output` directory in your current working directory
- A saved model file (`example_model.pt` or `custom_model.pt`)
- Visualization files (`predictions.png` or `custom_model_predictions.png`)

## Additional Examples

Check out other examples in this directory for more specialized use cases:

- Feature extraction and visualization
- Working with real molecular datasets
- Transfer learning for molecular property prediction
- Custom model architectures
