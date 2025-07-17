"""
examples/prediction_examples/run_predict.py

A command-line interface for making molecular property predictions using trained MGNN models with support for both single molecule and batch processing modes.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

from moml.models.mgnn.evaluation import create_predictor, batch_predict_from_files

# Constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_FILE_PATTERN = '*.mol,*.sdf'
DEFAULT_OUTPUT_DIR_NAME = 'predictions'
SUPPORTED_DEVICES = ['cpu', 'cuda']
PREDICTION_DISPLAY_LIMIT = 5


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for molecular property prediction.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_path: Path to trained model checkpoint
            - mol_file: Path to molecule file or directory
            - charges_file: Optional path to charges file
            - output_file: Output path for predictions
            - batch_mode: Flag for batch processing mode
            - file_pattern: Pattern for batch file selection
            - batch_size: Batch size for inference
            - device: Computing device selection
    """
    parser = argparse.ArgumentParser(
        description='Run molecular property inference with trained MGNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single molecule prediction
  %(prog)s --model_path model.pt --mol_file molecule.mol --output_file pred.json
  
  # Single molecule with charges
  %(prog)s --model_path model.pt --mol_file mol.mol --charges_file charges.txt
  
  # Batch processing
  %(prog)s --model_path model.pt --mol_file ./molecules/ --batch_mode
  
  # Batch with custom pattern and output
  %(prog)s --model_path model.pt --mol_file ./data/ --batch_mode \
           --file_pattern "*.sdf" --output_file ./results/
        """
    )

    # Model and input arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--mol_file',
        type=str,
        required=True,
        help='Path to molecule file (.mol/.sdf) or directory containing files'
    )
    parser.add_argument(
        '--charges_file',
        type=str,
        default=None,
        help='Path to partial charges file (optional, single file mode only)'
    )

    # Output arguments
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file path or directory for saving predictions'
    )

    # Batch processing arguments
    parser.add_argument(
        '--batch_mode',
        action='store_true',
        help='Enable batch processing mode for directory input'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default=DEFAULT_FILE_PATTERN,
        help='Comma-separated file patterns for batch mode (e.g., "*.mol,*.sdf")'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for inference processing'
    )

    # Hardware arguments
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=SUPPORTED_DEVICES,
        help='Computing device for inference (cpu/cuda)'
    )

    return parser.parse_args()


def run_batch_prediction(args: argparse.Namespace) -> None:
    """
    Execute batch prediction on multiple molecule files.

    Processes all molecule files in the specified directory matching the
    given pattern and saves predictions to the output directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            batch processing parameters.

    Raises:
        FileNotFoundError: If input directory doesn't exist.
        ValueError: If no matching files found in directory.
        OSError: If output directory cannot be created.
    """
    input_dir = args.mol_file
    print(f'Running batch prediction on files in {input_dir}')

    # Determine output directory
    if args.output_file:
        output_dir = args.output_file
    else:
        parent_dir = os.path.dirname(input_dir.rstrip('/'))
        output_dir = os.path.join(parent_dir, DEFAULT_OUTPUT_DIR_NAME)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Execute batch prediction
    print(f'Processing files with pattern: {args.file_pattern}')
    print(f'Using batch size: {args.batch_size}')
    
    try:
        results = batch_predict_from_files(
            model_path=args.model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=args.file_pattern,
            batch_size=args.batch_size,
            device=args.device,
        )

        print(f'\nBatch prediction completed successfully!')
        print(f'Predictions saved to: {output_dir}')
        print(f'Processed {len(results)} files')

    except Exception as e:
        print(f'Error during batch prediction: {e}')
        raise


def run_single_prediction(args: argparse.Namespace) -> None:
    """
    Execute prediction on a single molecule file.

    Loads the trained model, processes the molecule file with optional
    partial charges, and displays/saves the predictions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            single prediction parameters.

    Raises:
        FileNotFoundError: If molecule or charges file doesn't exist.
        ValueError: If prediction fails or model cannot be loaded.
    """
    print('Creating predictor...')
    try:
        config = {'device': args.device} if args.device else {}
        predictor = create_predictor(config, model_path=args.model_path)
        print(f'Model loaded successfully from {args.model_path}')
    except Exception as e:
        print(f'Error loading model: {e}')
        raise

    print(f'Running inference on {args.mol_file}')
    
    try:
        predictions = predictor.predict_from_file(
            args.mol_file,
            args.charges_file
        )
        
        if predictions is None:
            raise ValueError('Prediction returned None - check input file format')

        # Display predictions
        _display_predictions(predictions, args.mol_file)

        # Save predictions if output path specified
        if args.output_file:
            predictor.save_predictions(predictions, args.output_file)
            print(f'Predictions saved to: {args.output_file}')

    except Exception as e:
        print(f'Error during prediction for {args.mol_file}: {e}')
        raise


def _display_predictions(predictions: Dict[str, Any], mol_file: str) -> None:
    """
    Display prediction results in a formatted manner.

    Args:
        predictions (Dict[str, Any]): Dictionary containing prediction tensors.
        mol_file (str): Path to the molecule file for context.
    """
    print(f'\nPrediction Results for {os.path.basename(mol_file)}:')
    print('-' * 50)

    # Convert tensors to lists for display
    serializable_preds = {}
    for key, value in predictions.items():
        if hasattr(value, 'tolist'):
            serializable_preds[key] = value.tolist()
        else:
            serializable_preds[key] = value

    # Display graph-level predictions
    if 'graph_pred' in serializable_preds:
        graph_pred = serializable_preds['graph_pred']
        print(f'Graph-level prediction: {graph_pred}')

    # Display node-level predictions (limited for readability)
    if 'node_pred' in serializable_preds:
        node_pred = serializable_preds['node_pred']
        if isinstance(node_pred, list) and len(node_pred) > 0:
            display_count = min(len(node_pred), PREDICTION_DISPLAY_LIMIT)
            print(f'Node-level predictions (showing first {display_count} of '
                  f'{len(node_pred)} nodes):')
            for i in range(display_count):
                print(f'  Node {i+1}: {node_pred[i]}')
            
            if len(node_pred) > PREDICTION_DISPLAY_LIMIT:
                print(f'  ... ({len(node_pred) - PREDICTION_DISPLAY_LIMIT} '
                      'more nodes)')

    # Display edge-level predictions if present
    if 'edge_pred' in serializable_preds:
        edge_pred = serializable_preds['edge_pred']
        if isinstance(edge_pred, list) and len(edge_pred) > 0:
            display_count = min(len(edge_pred), PREDICTION_DISPLAY_LIMIT)
            print(f'Edge-level predictions (showing first {display_count} of '
                  f'{len(edge_pred)} edges):')
            for i in range(display_count):
                print(f'  Edge {i+1}: {edge_pred[i]}')

    # Display additional prediction types
    for key, value in serializable_preds.items():
        if key not in ['graph_pred', 'node_pred', 'edge_pred']:
            print(f'{key}: {value}')


def main() -> int:
    """
    Main entry point for molecular property prediction CLI.

    Parses command-line arguments and executes either single molecule
    or batch prediction based on the specified mode and input type.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        args = parse_args()

        print('MoML-CA Molecular Property Prediction')
        print('=' * 40)
        print(f'Model: {args.model_path}')
        print(f'Device: {args.device or "auto"}')

        # Determine prediction mode
        is_batch_mode = args.batch_mode or os.path.isdir(args.mol_file)

        if is_batch_mode:
            run_batch_prediction(args)
        else:
            run_single_prediction(args)

        return 0

    except (FileNotFoundError, ValueError, OSError) as e:
        print(f'Error: {e}')
        return 1
    except KeyboardInterrupt:
        print('\nPrediction cancelled by user')
        return 1
    except Exception as e:
        print(f'Unexpected error: {e}')
        return 1


if __name__ == '__main__':
    exit(main())
