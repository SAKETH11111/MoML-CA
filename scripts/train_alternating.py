#!/usr/bin/env python3
"""
Alternating training script for DJMGNN model.
Alternates between graph-level tasks (QM9+PFAS) and node-level tasks (SPICE).
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
import time
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('alternating_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_cycle_iterator(dataloader):
    """Create an infinite cycle iterator from a dataloader."""
    while True:
        for batch in dataloader:
            yield batch


def compute_losses(model, batch, device, lambda_weight=1000.0):
    """
    Compute node and graph losses for a batch.
    
    Args:
        model: DJMGNN model
        batch: Data batch
        device: PyTorch device
        lambda_weight: Weight for balancing node vs graph losses
        
    Returns:
        tuple: (node_loss, graph_loss)
    """
    batch = batch.to(device)
    
    # Forward pass
    node_output, graph_output = model(batch)
    
    # Node loss (forces prediction)
    node_loss = 0.0
    if hasattr(batch, 'forces') and batch.forces is not None:
        forces_flat = batch.forces.view(-1, 3)  # [N_atoms, 3]
        node_loss = nn.MSELoss()(node_output, forces_flat)
    
    # Graph loss (molecular properties)
    graph_loss = 0.0
    if hasattr(batch, 'y') and batch.y is not None:
        # Handle different target shapes
        if batch.y.dim() == 1:
            targets = batch.y.view(-1, 1)  # [batch_size, 1]
        else:
            targets = batch.y  # [batch_size, n_targets]
        
        # Ensure graph_output matches target dimensions
        if graph_output.size(-1) != targets.size(-1):
            if targets.size(-1) == 1:
                # Take first output dimension
                graph_output = graph_output[:, :1]
            else:
                # Pad or truncate as needed
                min_dim = min(graph_output.size(-1), targets.size(-1))
                graph_output = graph_output[:, :min_dim]
                targets = targets[:, :min_dim]
        
        graph_loss = nn.MSELoss()(graph_output, targets)
    
    return node_loss, graph_loss


def train_step(model, batch, optimizer, device, loss_node_weight, loss_graph_weight, lambda_weight=1000.0):
    """
    Perform a single training step.
    
    Args:
        model: DJMGNN model
        batch: Data batch
        optimizer: PyTorch optimizer
        device: PyTorch device
        loss_node_weight: Weight for node loss (0 or 1)
        loss_graph_weight: Weight for graph loss (0 or 1)
        lambda_weight: Lambda weighting factor
        
    Returns:
        dict: Training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Compute losses
    node_loss, graph_loss = compute_losses(model, batch, device, lambda_weight)
    
    # Weighted total loss
    total_loss = loss_node_weight * lambda_weight * node_loss + loss_graph_weight * graph_loss
    
    # Backward pass
    if total_loss > 0:
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    return {
        'total_loss': total_loss.item() if total_loss > 0 else 0.0,
        'node_loss': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss,
        'graph_loss': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else graph_loss,
        'loss_node_weight': loss_node_weight,
        'loss_graph_weight': loss_graph_weight
    }


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Alternating training for DJMGNN')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_weight', type=float, default=1000.0, help='Lambda weighting factor')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting alternating training for DJMGNN")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Create datasets
        print("Loading datasets...")
    
        # Try to load QM9, fall back to just PFAS if not available
        try:
            qm9_dataset = get_dataset("qm9", split="train")
            pfas_dataset = get_dataset("pfas", split="train")
            ds_graph = torch.utils.data.ConcatDataset([qm9_dataset, pfas_dataset])
            print(f"Loaded QM9 ({len(qm9_dataset)}) + PFAS ({len(pfas_dataset)}) datasets")
        except Exception as e:
            print(f"Could not load QM9 dataset: {e}")
            print("Using only PFAS dataset for graph-level tasks")
            ds_graph = get_dataset("pfas", split="train")
        
        graph_loader = GraphDataLoader(
            ds_graph, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=2
        )

        ds_node = get_dataset("spice", split="train")
        node_loader = GraphDataLoader(
            ds_node, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        # Create cycle iterators
        graph_cycle = create_cycle_iterator(graph_loader)
        node_cycle = create_cycle_iterator(node_loader)
        
        # Initialize model
        logger.info("Initializing DJMGNN model...")
        model = DJMGNN(
            node_input_dims=1,      # Atomic numbers
            edge_input_dims=0,      # No edge features
            node_output_dims=3,     # Forces (x, y, z)
            graph_output_dims=19,   # QM9/PFAS properties
            hidden_dims=128,
            num_layers=4
        )
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        logger.info("Starting alternating training...")
        start_time = time.time()
        
        for step in range(args.max_steps):
            # Alternating logic:
            # Even steps: graph tasks (QM9+PFAS) - loss_node=0, loss_graph=1
            # Odd steps: node tasks (SPICE) - loss_node=1, loss_graph=0
            
            if step % 2 == 0:
                # Graph task step
                batch = next(graph_cycle)
                loss_node_weight = 0
                loss_graph_weight = 1
                task_type = "graph"
            else:
                # Node task step  
                batch = next(node_cycle)
                loss_node_weight = 1
                loss_graph_weight = 0
                task_type = "node"
            
            # Training step
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                loss_node_weight=loss_node_weight,
                loss_graph_weight=loss_graph_weight,
                lambda_weight=args.lambda_weight
            )
            
            # Logging
            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Step {step:5d} | Task: {task_type:5s} | "
                    f"Total Loss: {metrics['total_loss']:.6f} | "
                    f"Node Loss: {metrics['node_loss']:.6f} | "
                    f"Graph Loss: {metrics['graph_loss']:.6f} | "
                    f"Time: {elapsed_time:.1f}s"
                )
            
            # Save checkpoint
            if step % args.save_every == 0 and step > 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, step, metrics['total_loss'], args.checkpoint_dir
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Final checkpoint
        final_checkpoint = save_checkpoint(
            model, optimizer, args.max_steps, metrics['total_loss'], args.checkpoint_dir
        )
        logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
