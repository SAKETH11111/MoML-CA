#!/usr/bin/env python
"""
Example script for running predictions with a trained MGNN model.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from code.MGNN.evaluation.predictor import create_predictor
from code.MGNN.utils.config import MGNNConfig, create_argparser


def parse_args():
    """Parse command line arguments for prediction example."""
    parser = create_argparser(
        description='Example script for running predictions with a trained MGNN model',
        include_training=False
    )
    
    # Add example-specific arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model checkpoint')
    parser.add_argument('--mol_path', type=str, required=True,
                       help='Path to molecule file(s) or directory containing molecule files')
    parser.add_argument('--charges_path', type=str, default=None,
                       help='Path to charges file(s) or directory (optional)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Directory to save prediction results')
    
    return parser.parse_args()


def main():
    """Main function for prediction example."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    config = MGNNConfig.from_args(args)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create predictor
    print(f"Loading model from {args.model_path}")
    predictor = create_predictor(
        model_path=args.model_path,
        config=config,
        device=device
    )
    
    # Handle input paths (file or directory)
    mol_path = Path(args.mol_path)
    
    if mol_path.is_dir():
        # Process all files in directory
        mol_files = list(mol_path.glob('*.mol')) + list(mol_path.glob('*.sdf'))
        print(f"Found {len(mol_files)} molecule files in directory")
        
        # Process each file
        for mol_file in mol_files:
            print(f"Processing {mol_file.name}")
            
            # Find matching charges file if applicable
            charges_file = None
            if args.charges_path:
                charges_dir = Path(args.charges_path)
                if charges_dir.is_dir():
                    potential_charges_file = charges_dir / f"{mol_file.stem}_charges.txt"
                    if potential_charges_file.exists():
                        charges_file = str(potential_charges_file)
            
            # Make prediction
            predictions = predictor.predict_from_file(
                mol_file=str(mol_file),
                charges_file=charges_file
            )
            
            # Save prediction
            output_file = Path(args.output_dir) / f"{mol_file.stem}_predictions.json"
            predictor.save_predictions(predictions, str(output_file))
            print(f"Saved predictions to {output_file}")
            
    else:
        # Process single file
        print(f"Processing single molecule file: {mol_path.name}")
        
        # Make prediction
        predictions = predictor.predict_from_file(
            mol_file=str(mol_path),
            charges_file=args.charges_path
        )
        
        # Print summary of predictions
        print("\nPredictions:")
        print(f"Graph-level prediction: {predictions['graph_pred']}")
        if len(predictions['node_pred']) > 5:
            print(f"Node-level predictions (first 5 nodes): {predictions['node_pred'][:5]}")
        else:
            print(f"Node-level predictions: {predictions['node_pred']}")
        
        # Save prediction
        output_file = Path(args.output_dir) / f"{mol_path.stem}_predictions.json"
        predictor.save_predictions(predictions, str(output_file))
        print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main() 