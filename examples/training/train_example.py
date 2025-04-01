#!/usr/bin/env python
"""
Example script to train a MGNN model on PFAS molecular data.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional

from code.MGNN.utils.config import create_argparser, get_config_from_args, MGNNConfig
from code.MGNN.training.trainer import create_trainer


def parse_args():
    """Parse command line arguments for training."""
    # Create parser with all argument groups
    parser = create_argparser(description='Train MGNN model on PFAS data')
    
    # Add I/O arguments specific to training
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training molecular graph data')
    io_group.add_argument('--val_dir', type=str, default=None,
                       help='Directory containing validation molecular graph data')
    io_group.add_argument('--output_dir', type=str, default='./model_checkpoints',
                       help='Directory to save model checkpoints')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get unified configuration
    config = get_config_from_args(args)
    
    # Print configuration summary
    print(f"Using device: {config['training'].get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Model configuration: {config['model']['n_blocks']} blocks with {config['model']['layers_per_block']} layers each")
    print(f"JK mode: {config['model']['jk_mode']}")
    
    # Initialize trainer with config
    trainer = create_trainer(
        config=config,
        train_dir=args.train_dir,
        val_dir=args.val_dir
    )
    
    # Define loss functions (MSE for regression tasks)
    node_loss_fn = nn.MSELoss()
    graph_loss_fn = nn.MSELoss()
    
    # Train model
    print(f"Starting training for {config['training']['epochs']} epochs...")
    history = trainer.train(
        node_loss_fn=node_loss_fn,
        graph_loss_fn=graph_loss_fn,
        checkpoint_dir=args.output_dir
    )
    
    print(f"Training completed. Model saved to {args.output_dir}")
    
    # Save configuration for reference
    config_path = os.path.join(args.output_dir, 'training_config.json')
    MGNNConfig.save_config_to_file(config, config_path)
    print(f"Configuration saved to {config_path}")
    
    # Example of how to make predictions with the trained model
    print("\nExample of prediction usage:")
    print("trainer.predict('molecule.mol', 'charges.txt')")
    
    return trainer


if __name__ == "__main__":
    main() 