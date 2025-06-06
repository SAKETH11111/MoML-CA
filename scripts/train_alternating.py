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
    if task_type == "node":
        graph_pred = out.get("graph_pred_spice_energy")
    else:
        graph_pred = out.get("graph_pred_main")
    
    node_loss = torch.tensor(0.0, device=device)
    if node_pred is not None and hasattr(batch, 'node_y') and batch.node_y is not None:
        if batch.node_y.numel() > 0 and node_pred.numel() > 0:
            if node_pred.shape == batch.node_y.shape:
                node_loss = nn.MSELoss()(node_pred, batch.node_y)
            else:
                logger.warning(f"Node prediction shape {node_pred.shape} mismatch with target shape {batch.node_y.shape}. Skipping node loss.")
        elif node_pred.numel() == 0 and batch.node_y.numel() == 0:
            pass
        else:
            logger.warning(f"Node prediction or target is empty while the other is not. Pred empty: {node_pred.numel()==0}, Target empty: {batch.node_y.numel()==0}. Skipping node loss.")

    graph_loss = torch.tensor(0.0, device=device)
    if graph_pred is not None and hasattr(batch, 'y') and batch.y is not None:
        if batch.y.numel() > 0 and graph_pred.numel() > 0:
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.view(-1, 1)
            
            if graph_pred.size(-1) != targets.size(-1):
                logger.warning(
                    f"Graph prediction dim {graph_pred.size(-1)} mismatch with target dim {targets.size(-1)}. "
                    f"Attempting to adjust. Graph pred shape: {graph_pred.shape}, Target shape: {targets.shape}"
                )
                if targets.size(-1) == 1 and graph_pred.size(-1) > 1:
                    graph_pred_adjusted = graph_pred[:, :1]
                    graph_loss = nn.MSELoss()(graph_pred_adjusted, targets)
                elif graph_pred.size(-1) == 1 and targets.size(-1) > 1:
                    logger.warning("Graph prediction is single-dim but target is multi-dim. Skipping graph loss or apply specific logic.")
                else:
                    min_dim = min(graph_pred.size(-1), targets.size(-1))
                    if min_dim > 0:
                        graph_pred_adjusted = graph_pred[:, :min_dim]
                        targets_adjusted = targets[:, :min_dim]
                        graph_loss = nn.MSELoss()(graph_pred_adjusted, targets_adjusted)
                    else:
                        logger.warning("Min dimension for graph loss is 0. Skipping graph loss.")
            else:
                graph_loss = nn.MSELoss()(graph_pred, targets)
        elif graph_pred.numel() == 0 and batch.y.numel() == 0:
            pass
        else:
             logger.warning(f"Graph prediction or target is empty while the other is not. Pred empty: {graph_pred.numel()==0}, Target empty: {batch.y.numel()==0}. Skipping graph loss.")

    return node_loss, graph_loss


def train_step(model, batch, optimizer, device, loss_node_weight, loss_graph_weight, logger, lambda_weight=1000.0, task_type: str = "graph"):
    model.train()
    optimizer.zero_grad()
    
    node_loss, graph_loss = compute_losses(model, batch, device, logger, lambda_weight, task_type=task_type)
    
    total_loss = loss_node_weight * lambda_weight * node_loss + loss_graph_weight * graph_loss
    
    if total_loss > 0:
        total_loss.backward()
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
    
    try:
        print("Loading datasets...")
    
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
        
        graph_cycle = create_cycle_iterator(graph_loader)
        node_cycle = create_cycle_iterator(node_loader)
        
        logger.info("Initializing DJMGNN model...")

        mgnn_config = config.get('mgnn', {})
        in_node_dim_cfg = mgnn_config.get('in_node_dim', 1)
        graph_output_dims_cfg = mgnn_config.get('graph_output_dims', 19) 
        node_output_dims_cfg = mgnn_config.get('node_output_dims', 3)   
        spice_graph_output_dims_cfg = mgnn_config.get('spice_graph_output_dims', 1)
        hidden_dim_cfg = mgnn_config.get('hidden_channels', 128) 
        n_blocks_cfg = mgnn_config.get('num_layers', 4) 

        model = DJMGNN(
            in_node_dim=in_node_dim_cfg, 
            in_edge_dim=0,
            node_output_dims=node_output_dims_cfg,     
            graph_output_dims=graph_output_dims_cfg,   
            spice_graph_output_dims=spice_graph_output_dims_cfg,
            hidden_dim=hidden_dim_cfg,        
            n_blocks=n_blocks_cfg             
        )
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        logger.info("Starting alternating training...")
        start_time = time.time()
        
        for step in range(args.max_steps):
            if step % 2 == 0:
                batch = next(graph_cycle)
                loss_node_weight = 0
                loss_graph_weight = 1
                task_type = "graph"
            else:
                batch = next(node_cycle)
                loss_node_weight = 1
                loss_graph_weight = 1
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
                task_type=task_type
            )
            
            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Step {step:5d} | Task: {task_type:5s} | "
                    f"Total Loss: {metrics['total_loss']:.6f} | "
                    f"Node Loss: {metrics['node_loss']:.6f} | "
                    f"Graph Loss: {metrics['graph_loss']:.6f} | "
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
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
