#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN


def setup_logging():
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
    while True:
        for batch in dataloader:
            yield batch


def compute_losses(model, batch, device, logger, lambda_weight=1000.0, task_type: str = "graph"):
    batch = batch.to(device)
    
    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, 'edge_attr', None),
        batch=getattr(batch, 'batch', None),
        dist=getattr(batch, 'dist', None)
    )
    
    node_pred = out.get("node_pred")
    graph_pred = out.get("graph_pred")
    energy_pred = out.get("energy_pred")
    
    node_loss = torch.tensor(0.0, device=device)
    graph_loss = torch.tensor(0.0, device=device)
    energy_loss = torch.tensor(0.0, device=device)

    # Node-level loss
    if task_type == "node" and node_pred is not None and hasattr(batch, 'node_y') and batch.node_y is not None:
        if batch.node_y.numel() > 0 and node_pred.numel() > 0:
            if node_pred.shape == batch.node_y.shape:
                node_loss = nn.MSELoss()(node_pred, batch.node_y)
            else:
                logger.warning(f"Node prediction shape {node_pred.shape} mismatch with target shape {batch.node_y.shape}. Skipping node loss.")

    # Graph-level loss (main)
    if task_type == "graph" and graph_pred is not None and hasattr(batch, 'y_graph') and batch.y_graph is not None:
        if batch.y_graph.numel() > 0 and graph_pred.numel() > 0:
            targets = batch.y_graph
            if targets.dim() == 1:
                targets = targets.view(-1, 1)
            
            if graph_pred.size(-1) == targets.size(-1):
                graph_loss = nn.MSELoss()(graph_pred, targets)
        elif graph_pred is not None:
            logger.warning(f"Graph prediction or target data is missing or empty. Skipping graph loss.")
    
    # Energy loss (SPICE)
    if task_type == "node" and energy_pred is not None and hasattr(batch, 'y_graph') and batch.y_graph is not None:
        if batch.y_graph.numel() > 0 and energy_pred.numel() > 0:
            targets = batch.y_graph
            if targets.dim() == 1:
                targets = targets.view(-1, 1)

            if energy_pred.size(-1) == targets.size(-1):
                energy_loss = nn.MSELoss()(energy_pred, targets)
            else:
                logger.warning(f"SPICE energy prediction dim {energy_pred.size(-1)} mismatch with target dim {targets.size(-1)}. Skipping energy loss.")

    return node_loss, graph_loss, energy_loss


def train_step(model, batch, optimizer, device, loss_node_weight, loss_graph_weight, logger, lambda_weight=1000.0, lambda_energy_weight=1.0, task_type: str = "graph"):
    model.train()
    optimizer.zero_grad()
    
    node_loss, graph_loss, energy_loss = compute_losses(model, batch, device, logger, lambda_weight, task_type=task_type)
    
    total_loss = (loss_node_weight * lambda_weight * node_loss) + \
                 (loss_graph_weight * graph_loss) + \
                 (loss_node_weight * lambda_energy_weight * energy_loss) # energy loss is also on node step
    
    if total_loss > 0:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return {
        'total_loss': total_loss.item() if total_loss > 0 else 0.0,
        'node_loss': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss,
        'graph_loss': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else graph_loss,
        'energy_loss': energy_loss.item() if isinstance(energy_loss, torch.Tensor) else energy_loss,
        'loss_node_weight': loss_node_weight,
        'loss_graph_weight': loss_graph_weight
    }


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir):
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
    parser.add_argument('--dataset', type=str, default='qm9', choices=['qm9', 'pfas', 'spice', 'qm9+pfas'], help='Dataset to use for training')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_weight', type=float, default=1000.0, help='Lambda weighting factor for node loss')
    parser.add_argument('--lambda_energy_weight', type=float, default=1000.0, help='Lambda weighting factor for SPICE energy loss')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--config_path', type=str, default='config/training_config.template.yaml', help='Path to training config YAML file')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting alternating training for DJMGNN")
    logger.info(f"Arguments: {vars(args)}")

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded training configuration from {args.config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration file {args.config_path}: {e}", exc_info=True)
        sys.exit(1)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set in_node_dim based on dataset
    if args.dataset == 'qm9' or args.dataset == 'qm9+pfas':
        in_node_dim_cfg = 11  # QM9 has 11 node features
    elif args.dataset == 'pfas':
        in_node_dim_cfg = 23 # Example value, adjust if necessary
    elif args.dataset == 'spice':
        in_node_dim_cfg = 23 # Example value, adjust if necessary
    else:
        in_node_dim_cfg = config.get('mgnn', {}).get('in_node_dim', 1)
    
    try:
        print("Loading datasets...")
    
        # Graph-level dataset
        if args.dataset == 'qm9+pfas':
            qm9_dataset = get_dataset("qm9", split="train")
            pfas_dataset = get_dataset("pfas", split="train")
            ds_graph = ConcatDataset([qm9_dataset, pfas_dataset])
        else:
            ds_graph = get_dataset(args.dataset, split="train")
        
        graph_loader = GraphDataLoader(
            ds_graph,  # type: ignore
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        # Node-level dataset (always SPICE for alternating training)
        node_loader = None
        try:
            ds_node = get_dataset("spice", split="train")
            node_loader = GraphDataLoader(
                ds_node,  # type: ignore
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2
            )
        except Exception as e:
            logger.warning(f"Could not load SPICE dataset: {e}. Node-level tasks will be skipped.")
        
        graph_cycle = create_cycle_iterator(graph_loader)
        node_cycle = create_cycle_iterator(node_loader) if node_loader else None
        
        logger.info("Initializing DJMGNN model...")

        mgnn_config = config.get('mgnn', {})
        graph_output_dims_cfg = mgnn_config.get('graph_output_dims', 19) 
        node_output_dims_cfg = mgnn_config.get('node_output_dims', 3)   
        energy_output_dims_cfg = mgnn_config.get('energy_output_dims', 1)
        hidden_dim_cfg = mgnn_config.get('hidden_channels', 128) 
        n_blocks_cfg = mgnn_config.get('num_layers', 4) 

        model = DJMGNN(
            in_node_dim=in_node_dim_cfg, 
            in_edge_dim=0,
            node_output_dims=node_output_dims_cfg,     
            graph_output_dims=graph_output_dims_cfg,   
            energy_output_dims=energy_output_dims_cfg,
            hidden_dim=hidden_dim_cfg,        
            n_blocks=n_blocks_cfg             
        )
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        logger.info("Starting alternating training...")
        start_time = time.time()
        
        # Initialize metrics with default values to handle case where max_steps is 0
        metrics = {
            'total_loss': 0.0,
            'node_loss': 0.0,
            'graph_loss': 0.0,
            'energy_loss': 0.0,
            'loss_node_weight': 0,
            'loss_graph_weight': 0
        }
        
        for step in range(args.max_steps):
            if step % 2 == 0 or node_cycle is None:
                batch = next(graph_cycle)
                loss_node_weight = 0
                loss_graph_weight = 1
                task_type = "graph"
            else:
                batch = next(node_cycle)
                loss_node_weight = 1
                loss_graph_weight = 0 # No graph loss on node step
                task_type = "node"
            
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                loss_node_weight=loss_node_weight,
                loss_graph_weight=loss_graph_weight,
                logger=logger,
                lambda_weight=args.lambda_weight,
                lambda_energy_weight=args.lambda_energy_weight,
                task_type=task_type
            )
            
            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Step {step:5d} | Task: {task_type:5s} | "
                    f"Total Loss: {metrics['total_loss']:.6f} | "
                    f"Node Loss: {metrics['node_loss']:.6f} | "
                    f"Graph Loss: {metrics['graph_loss']:.6f} | "
                    f"Energy Loss: {metrics['energy_loss']:.6f} | "
                    f"Time: {elapsed_time:.1f}s"
                )
            
            if step % args.save_every == 0 and step > 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, step, metrics['total_loss'], args.checkpoint_dir
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        final_checkpoint = save_checkpoint(
            model, optimizer, args.max_steps, metrics['total_loss'], args.checkpoint_dir
        )
        logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        if 'model' in locals() and 'optimizer' in locals() and 'step' in locals() and 'metrics' in locals():
            final_checkpoint = save_checkpoint(
                model, optimizer, step, metrics.get('total_loss', 0.0), args.checkpoint_dir
            )
            logger.info(f"Saved partial checkpoint: {final_checkpoint}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
