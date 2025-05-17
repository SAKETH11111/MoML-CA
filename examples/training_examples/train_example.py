#!/usr/bin/env python
"""
Example script to train a MGNN model on molecular data.
"""

import os
import torch
import torch.nn as nn
import argparse
import glob
from torch_geometric.data import DataLoader

from moml.models.mgnn.training import create_trainer


def parse_args():
    """Parse command line arguments for training."""
    # Create parser with all argument groups
    parser = argparse.ArgumentParser(description="Train MGNN model on molecular data")

    # Add I/O arguments specific to training
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Directory containing training molecular graph data"
    )
    parser.add_argument(
        "--val_dir", type=str, default=None, help="Directory containing validation molecular graph data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model_checkpoints", help="Directory to save model checkpoints"
    )

    # Add model configuration arguments
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of GNN blocks")
    parser.add_argument("--layers_per_block", type=int, default=3, help="Number of layers per block")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--jk_mode", type=str, default="cat", choices=["cat", "max", "sum"], help="Jumping knowledge mode"
    )

    # Add training configuration arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create configuration dictionary
    config = {
        "model": {
            "hidden_dim": args.hidden_dim,
            "n_blocks": args.n_blocks,
            "layers_per_block": args.layers_per_block,
            "dropout": args.dropout,
            "jk_mode": args.jk_mode,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
    }

    # Print configuration summary
    print(f"Using device: {config['training'].get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(
        f"Model configuration: {config['model']['n_blocks']} blocks with {config['model']['layers_per_block']} layers each"
    )
    print(f"JK mode: {config['model']['jk_mode']}")

    # Load datasets and wrap in DataLoaders
    train_files = glob.glob(os.path.join(args.train_dir, "*.pt"))
    train_graphs = [torch.load(f) for f in train_files]
    train_loader = DataLoader(train_graphs, batch_size=config["training"]["batch_size"], shuffle=True)
    if args.val_dir:
        val_files = glob.glob(os.path.join(args.val_dir, "*.pt"))
        val_graphs = [torch.load(f) for f in val_files]
        val_loader = DataLoader(val_graphs, batch_size=config["training"]["batch_size"])
    else:
        val_loader = None

    # Initialize trainer with config
    trainer = create_trainer(config=config, train_loader=train_loader, val_loader=val_loader)

    # Define loss functions (MSE for regression tasks)
    node_loss_fn = nn.MSELoss()
    graph_loss_fn = nn.MSELoss()

    # Train model
    print(f"Starting training for {config['training']['epochs']} epochs...")
    history = trainer.train(node_loss_fn=node_loss_fn, graph_loss_fn=graph_loss_fn, checkpoint_dir=args.output_dir)

    print(f"Training completed. Model saved to {args.output_dir}")

    # Display final training metrics
    if history:
        print(f"Final node loss: {history.get('node_loss', [None])[-1]:.6f}")
        print(f"Final graph loss: {history.get('graph_loss', [None])[-1]:.6f}")

    # Save configuration for reference
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        import json

        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Example of how to make predictions with the trained model
    print("\nExample of prediction usage:")
    print("from moml.models.mgnn.evaluation import create_predictor")
    print("predictor = create_predictor(model_path=os.path.join(args.output_dir, 'model.pt'))")
    print("predictions = predictor.predict('molecule.mol', 'charges.txt')")
    print("print(predictions)")

    return trainer


if __name__ == "__main__":
    main()
