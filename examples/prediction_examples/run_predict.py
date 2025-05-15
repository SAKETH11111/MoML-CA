#!/usr/bin/env python
"""
Command-line script for making predictions with trained models.

This script provides a command-line interface to the MGNNPredictor class,
allowing users to make predictions on single molecules or directories of molecules.
"""

import os
import argparse

from moml.models.mgnn.evaluation import create_predictor, batch_predict_from_files


def parse_args():
    """Parse command line arguments for prediction."""
    # Create parser with model arguments only
    parser = argparse.ArgumentParser(description="Run inference with a trained MGNN model")

    # Add I/O arguments specific to prediction
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument(
        "--mol_file", type=str, required=True, help="Path to molecule file (.mol or .sdf) or directory of files"
    )
    parser.add_argument(
        "--charges_file", type=str, default=None, help="Path to charges file (optional, for single file mode only)"
    )
    parser.add_argument("--output_file", type=str, default=None, help="Output file or directory to save predictions")
    parser.add_argument(
        "--batch_mode", action="store_true", help="Whether to interpret mol_file as a directory of molecule files"
    )
    parser.add_argument("--file_pattern", type=str, default="*.mol,*.sdf", help="File pattern to use in batch mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference in batch mode")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    return parser.parse_args()


def main():
    """Run prediction from command line arguments."""
    args = parse_args()

    # Set device
    device = args.device

    print(f"Loading model from {args.model_path}")

    if args.batch_mode or os.path.isdir(args.mol_file):
        # Batch prediction mode - directory of molecule files
        input_dir = args.mol_file
        print(f"Running batch prediction on files in {input_dir}")

        # Create output directory if needed
        output_dir = args.output_file
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(input_dir), "predictions")

        # Run batch prediction
        results = batch_predict_from_files(
            model_path=args.model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=args.file_pattern,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"Predictions saved to {output_dir}")
        print(f"Processed {len(results)} files")

    else:
        # Single molecule prediction mode
        print("Creating predictor...")
        predictor = create_predictor(args.model_path, device=device)

        print(f"Running inference on {args.mol_file}")
        predictions = predictor.predict_from_file(args.mol_file, args.charges_file)

        # Convert tensors to lists for display
        serializable_preds = {}
        for key, value in predictions.items():
            serializable_preds[key] = value.tolist()

        # Print predictions
        print("\nPredictions:")
        if "graph_pred" in serializable_preds:
            graph_pred = serializable_preds["graph_pred"]
            print(f"Graph-level prediction: {graph_pred}")

        if "node_pred" in serializable_preds:
            node_pred = serializable_preds["node_pred"]
            print(f"Node-level predictions (first 5 nodes): {node_pred[:5]}")

        # Save predictions to file if requested
        if args.output_file:
            predictor.save_predictions(predictions, args.output_file)
            print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
