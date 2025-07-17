"""
examples/training_examples/train_example.py

A command-line tool for training MGNN models on molecular graph data with comprehensive configuration options and checkpoint management.
"""

import argparse
import glob
import json
import os
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from moml.models.mgnn.training import create_trainer

# Constants
DEFAULT_OUTPUT_DIR = './model_checkpoints'
DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_BLOCKS = 2
DEFAULT_LAYERS_PER_BLOCK = 3
DEFAULT_DROPOUT = 0.2
DEFAULT_JK_MODE = 'cat'
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
SUPPORTED_JK_MODES = ['cat', 'max', 'sum']
CONFIG_FILENAME = 'training_config.json'
GRAPH_FILE_PATTERN = '*.pt'


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for MGNN model training.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - train_dir: Directory with training molecular graph data
            - val_dir: Optional validation data directory
            - output_dir: Directory for model checkpoints
            - Model configuration: hidden_dim, n_blocks, layers_per_block, etc.
            - Training configuration: epochs, batch_size, learning_rate, etc.
            - device: Computing device selection
    """
    parser = argparse.ArgumentParser(
        description='Train MGNN model on molecular data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --train_dir ./data/train --output_dir ./models
  %(prog)s --train_dir ./graphs --val_dir ./val_graphs --epochs 200
  %(prog)s --train_dir ./data --hidden_dim 128 --n_blocks 3 --device cuda
        """
    )

    # I/O arguments
    parser.add_argument(
        '--train_dir',
        type=str,
        required=True,
        help='Directory containing training molecular graph data'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default=None,
        help='Directory containing validation molecular graph data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to save model checkpoints'
    )

    # Model configuration arguments
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--n_blocks',
        type=int,
        default=DEFAULT_N_BLOCKS,
        help='Number of GNN blocks'
    )
    parser.add_argument(
        '--layers_per_block',
        type=int,
        default=DEFAULT_LAYERS_PER_BLOCK,
        help='Number of layers per block'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DEFAULT_DROPOUT,
        help='Dropout rate'
    )
    parser.add_argument(
        '--jk_mode',
        type=str,
        default=DEFAULT_JK_MODE,
        choices=SUPPORTED_JK_MODES,
        help='Jumping knowledge mode'
    )

    # Training configuration arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help='Weight decay'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)'
    )

    return parser.parse_args()


def load_graph_datasets(
    train_dir: str,
    val_dir: Optional[str],
    batch_size: int
) -> Tuple[Any, Optional[Any]]:
    """
    Load molecular graph datasets from directories.

    Args:
        train_dir (str): Directory containing training graph files.
        val_dir (Optional[str]): Directory containing validation graph files,
            or None if no validation data.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple[DataLoader, Optional[DataLoader]]: Training and validation
            DataLoaders. Validation loader is None if val_dir is None.

    Raises:
        FileNotFoundError: If training directory doesn't exist or contains
            no graph files.
        RuntimeError: If graph files cannot be loaded.
    """
    # Load training dataset
    train_files = glob.glob(os.path.join(train_dir, GRAPH_FILE_PATTERN))
    if not train_files:
        raise FileNotFoundError(
            f'No training graph files found in {train_dir}'
        )

    try:
        train_graphs = [torch.load(f) for f in train_files]
        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True
        )
        print(f'Loaded {len(train_graphs)} training graphs')
    except Exception as e:
        raise RuntimeError(f'Error loading training graphs: {e}')

    # Load validation dataset if provided
    val_loader = None
    if val_dir:
        val_files = glob.glob(os.path.join(val_dir, GRAPH_FILE_PATTERN))
        if val_files:
            try:
                val_graphs = [torch.load(f) for f in val_files]
                val_loader = DataLoader(val_graphs, batch_size=batch_size)
                print(f'Loaded {len(val_graphs)} validation graphs')
            except Exception as e:
                print(f'Warning: Error loading validation graphs: {e}')
        else:
            print(f'Warning: No validation graph files found in {val_dir}')

    return train_loader, val_loader


def create_model_config(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """
    Create model and training configuration dictionary from arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Dict[str, Any]]: Configuration dictionary with 'model'
            and 'training' sections.
    """
    return {
        'model': {
            'hidden_dim': args.hidden_dim,
            'n_blocks': args.n_blocks,
            'layers_per_block': args.layers_per_block,
            'dropout': args.dropout,
            'jk_mode': args.jk_mode,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'device': args.device,
        },
    }


def print_configuration_summary(config: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of the model and training configuration.

    Args:
        config (Dict[str, Dict[str, Any]]): Configuration dictionary.
    """
    device = config['training'].get('device') or (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device: {device}')
    print(
        f'Model configuration: {config["model"]["n_blocks"]} blocks '
        f'with {config["model"]["layers_per_block"]} layers each'
    )
    print(f'JK mode: {config["model"]["jk_mode"]}')
    print(f'Training for {config["training"]["epochs"]} epochs')


def save_training_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save training configuration to JSON file for reference.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        output_dir (str): Directory where configuration will be saved.

    Raises:
        OSError: If configuration file cannot be written.
    """
    config_path = os.path.join(output_dir, CONFIG_FILENAME)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f'Configuration saved to {config_path}')


def display_training_results(history: Optional[Dict[str, List[float]]]) -> None:
    """
    Display final training metrics from training history.

    Args:
        history (Optional[Dict[str, List[float]]]): Training history
            dictionary containing loss values, or None if no history.
    """
    if history:
        if 'node_loss' in history and history['node_loss']:
            print(f'Final node loss: {history["node_loss"][-1]:.6f}')
        if 'graph_loss' in history and history['graph_loss']:
            print(f'Final graph loss: {history["graph_loss"][-1]:.6f}')
        if 'train_loss' in history and history['train_loss']:
            print(f'Final training loss: {history["train_loss"][-1]:.6f}')
        if 'val_loss' in history and history['val_loss']:
            print(f'Final validation loss: {history["val_loss"][-1]:.6f}')
    else:
        print('No training history available')


def print_usage_example(output_dir: str) -> None:
    """
    Print example usage for making predictions with the trained model.

    Args:
        output_dir (str): Directory where model was saved.
    """
    print('\nExample of prediction usage:')
    print('from moml.models.mgnn.evaluation import create_predictor')
    print(f'config = {{"device": "cpu"}}')
    print(f'predictor = create_predictor(config, model_path="{output_dir}/model.pt")')
    print('predictions = predictor.predict_from_file("molecule.mol")')
    print('print(predictions)')


def main() -> int:
    """
    Main entry point for MGNN model training.

    Parses command-line arguments, loads datasets, configures and trains
    the model, saves checkpoints and configuration, and displays results.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        args = parse_args()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Create configuration dictionary
        config = create_model_config(args)

        # Print configuration summary
        print_configuration_summary(config)

        # Load datasets and create DataLoaders
        train_loader, val_loader = load_graph_datasets(
            args.train_dir,
            args.val_dir,
            config['training']['batch_size']
        )

        # Initialize trainer with configuration
        trainer = create_trainer(
            config=config,
            train_loader=train_loader
        )

        # Define loss functions (MSE for regression tasks)
        node_loss_fn = nn.MSELoss()
        graph_loss_fn = nn.MSELoss()

        # Train model
        print(f'Starting training for {config["training"]["epochs"]} epochs...')
        history = trainer.train(epochs=config["training"]["epochs"])

        print(f'Training completed. Model saved to {args.output_dir}')

        # Display final training metrics
        display_training_results(history)

        # Save configuration for reference
        save_training_config(config, args.output_dir)

        # Print usage example
        print_usage_example(args.output_dir)

        return 0

    except (FileNotFoundError, RuntimeError, OSError) as e:
        print(f'Error: {e}')
        return 1
    except KeyboardInterrupt:
        print('\nTraining cancelled by user')
        return 1


if __name__ == '__main__':
    exit(main())
