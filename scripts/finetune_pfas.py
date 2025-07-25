"""
scripts/finetune_pfas.py

This script fine-tunes a pre-trained DJMGNN model on the PFAS dataset.
It loads a model checkpoint, prepares the PFAS dataset (splitting it into
training and validation sets), and then runs a fine-tuning loop.

The script includes features like:
-   Learning rate scheduling (`ReduceLROnPlateau`).
-   Early stopping to prevent overfitting.
-   Periodic validation and checkpointing of the best model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN
from moml.utils.dataset_utils import SubsetWrapper

NODE_FEATURE_DIM = 29

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_pretrained_model(ckpt_path: str, config: Dict[str, Any], device: torch.device) -> DJMGNN:
    """Loads a pre-trained DJMGNN model from a checkpoint."""
    logger.info(f"Loading pre-trained model from checkpoint: {ckpt_path}")
    mgnn_config = config.get("mgnn", {})
    model = DJMGNN(
        in_node_dim=NODE_FEATURE_DIM,
        in_edge_dim=mgnn_config.get("in_edge_dim", 0),
        node_output_dims=mgnn_config.get("node_output_dims", 3),
        graph_output_dims=mgnn_config.get("graph_output_dims", 19),
        energy_output_dims=mgnn_config.get("energy_output_dims", 1),
        hidden_dim=mgnn_config.get("hidden_channels", 160),
        n_blocks=mgnn_config.get("num_layers", 4),
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def prepare_data_loaders(batch_size: int, data_root: str = "data") -> Tuple[GraphDataLoader, GraphDataLoader]:
    """Prepares training and validation data loaders for the PFAS dataset."""
    logger.info("Loading and splitting PFAS dataset...")
    transform = Compose(
        [CreateEdges(), FeaturizeNodes(), StandardizeTargets(dataset_name="pfas")]
    )
    full_dataset = get_dataset("pfas", root=data_root, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = GraphDataLoader(SubsetWrapper(train_dataset), batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(SubsetWrapper(val_dataset), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def validate_model(
    model: DJMGNN, loader: GraphDataLoader, loss_fn: Any, device: torch.device
) -> float:
    """Evaluates the model on the validation set and returns the average loss."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in loader:
            batch: Batch = batch.to(device)  # type: ignore[attr-defined]
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)  # type: ignore[attr-defined]
            preds = out["graph_pred"]
            targets = batch.y.view(preds.shape)  # type: ignore[attr-defined]
            loss = loss_fn(preds, targets)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)


def save_checkpoint(model, optimizer, step, loss, val_loss, save_path: Path):
    """Saves a model checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "val_loss": val_loss,
        },
        save_path,
    )
    logger.info(f"Saved checkpoint to {save_path}")


def main():
    """Main function to run the fine-tuning script."""
    parser = argparse.ArgumentParser(description="Fine-tune DJMGNN on PFAS dataset.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the pre-trained model checkpoint.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum fine-tuning steps.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for fine-tuning.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu).")
    parser.add_argument("--config_path", type=str, default="config/training_config.template.yaml", help="Path to training config.")
    parser.add_argument("--save_path", type=str, default="checkpoints/finetuned_pfas.pt", help="Path to save the final fine-tuned model.")
    parser.add_argument("--save_every", type=int, default=200, help="Validate and save best model every N steps.")
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup
    model = load_pretrained_model(args.ckpt, config, device)
    train_loader, val_loader = prepare_data_loaders(args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()  # Mean Absolute Error
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5, verbose=True)  # type: ignore[arg-type]

    # Training Loop
    logger.info(f"Starting fine-tuning for up to {args.max_steps} steps...")
    step, best_val_loss, patience_counter = 0, float("inf"), 0
    
    training_complete = False
    while not training_complete:
        for batch in train_loader:
            if step >= args.max_steps:
                training_complete = True
                break

            model.train()
            batch: Batch = batch.to(device)  # type: ignore[attr-defined]
            optimizer.zero_grad()
            
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)  # type: ignore[attr-defined]
            preds = out["graph_pred"]
            targets = batch.y.view(preds.shape)  # type: ignore[attr-defined]
            
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                logger.info(f"Step {step:5d} | Training Loss: {loss.item():.6f}")

            if step > 0 and step % args.save_every == 0:
                avg_val_loss = validate_model(model, val_loader, loss_fn, device)
                logger.info(f"Validation Loss at step {step}: {avg_val_loss:.6f}")
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_path = Path(args.save_path).with_name(f"{Path(args.save_path).stem}_best.pt")
                    save_checkpoint(model, optimizer, step, loss.item(), avg_val_loss, best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping triggered at step {step}.")
                        training_complete = True
                        break
            step += 1

    logger.info("Fine-tuning complete.")
    final_save_path = Path(args.save_path)
    final_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, final_save_path)
    logger.info(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()