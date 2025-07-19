"""
scripts/train_hmgnn_standalone.py

Standalone training script for HMGNN (Hierarchical Molecular Graph Neural Network).

This script implements standalone training for the HMGNN model with hierarchical
graph processing, multi-scale attention, and comprehensive evaluation. It serves
as the foundation for hierarchical molecular representation learning before
joint training with DJMGNN.

Key Features:
    - Hierarchical graph coarsening and multi-scale processing
    - Cross-scale attention training
    - Multi-task learning for molecular properties
    - Comprehensive logging and checkpointing
    - GPU acceleration and distributed training support
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.core.hierarchical_processor import HierarchicalDataProcessor, create_hierarchical_processor
from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes, StandardizeTargets)
from moml.models.mgnn.hmgnn import HMGNN, create_hierarchical_mgnn
from moml.models.mgnn.training import MGNNTrainer, EarlyStopping, ModelCheckpoint

DEFAULT_NODE_FEATURE_DIM = 29
LOG_FILE_NAME = "hmgnn_training.log"

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for HMGNN training."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File handler
        try:
            log_file_path = Path(LOG_FILE_NAME).resolve()
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize file logger: {e}")


class HierarchicalDataset:
    """
    Dataset wrapper for hierarchical molecular graphs.
    
    Processes standard molecular datasets to create hierarchical representations
    suitable for HMGNN training.
    """
    
    def __init__(
        self,
        base_dataset: Any,
        hierarchical_processor: HierarchicalDataProcessor,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Initialize hierarchical dataset.
        
        Args:
            base_dataset: Base molecular dataset
            hierarchical_processor: Processor for creating hierarchical graphs
            cache_dir: Optional directory for caching processed data
        """
        self.base_dataset = base_dataset
        self.hierarchical_processor = hierarchical_processor
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get hierarchical representation of a molecule."""
        # Get base data
        base_data = self.base_dataset[idx]
        
        # Generate hierarchical representation
        mol_id = f"mol_{idx}"
        hierarchical_data = self.hierarchical_processor.process_molecule(
            base_data, mol=None, molecule_id=mol_id
        )
        
        # Add original targets
        result = hierarchical_data.copy()
        if hasattr(base_data, 'y'):
            result['targets'] = base_data.y
        if hasattr(base_data, 'node_y'):
            result['node_targets'] = base_data.node_y
        
        return result


class HMGNNTrainer(MGNNTrainer):
    """
    Specialized trainer for HMGNN models.
    
    Extends the base MGNNTrainer to handle hierarchical data processing,
    multi-scale batching, and HMGNN-specific training procedures.
    """
    
    def __init__(
        self,
        model: HMGNN,
        config: Dict[str, Any],
        train_loader: Optional[GraphDataLoader] = None,
        val_loader: Optional[GraphDataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        callbacks: Optional[List] = None,
        hierarchical_processor: Optional[HierarchicalDataProcessor] = None
    ):
        """Initialize HMGNN trainer."""
        super().__init__(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            callbacks=callbacks
        )
        
        self.hierarchical_processor = hierarchical_processor
        self.hmgnn_model = model
    
    def _prepare_hierarchical_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare hierarchical batch for HMGNN forward pass."""
        if not batch_data:
            return {}
        
        # Use hierarchical processor to create batch
        batched_data = self.hierarchical_processor.create_batch(batch_data)
        
        # Move to device
        for scale_data in batched_data['scale_data']:
            for key, value in scale_data.items():
                if isinstance(value, torch.Tensor):
                    scale_data[key] = value.to(self.device)
        
        # Move mappings to device
        batched_data['mappings'] = [
            mapping.to(self.device) if isinstance(mapping, torch.Tensor) else mapping
            for mapping in batched_data['mappings']
        ]
        
        batched_data['cluster_counts'] = [
            counts.to(self.device) if isinstance(counts, torch.Tensor) else counts
            for counts in batched_data['cluster_counts']
        ]
        
        return batched_data
    
    def _compute_hmgnn_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch_data: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute HMGNN-specific loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        num_loss_terms = 0
        
        # Extract targets from batch data
        graph_targets = []
        node_targets = []
        
        for mol_data in batch_data:
            if 'targets' in mol_data:
                graph_targets.append(mol_data['targets'])
            if 'node_targets' in mol_data:
                node_targets.append(mol_data['node_targets'])
        
        # Graph-level loss
        if graph_targets and 'graph_pred' in outputs:
            try:
                graph_targets_tensor = torch.stack(graph_targets).to(self.device)
                if outputs['graph_pred'].shape == graph_targets_tensor.shape:
                    graph_loss = self.loss_fn(outputs['graph_pred'], graph_targets_tensor)
                    total_loss += graph_loss
                    num_loss_terms += 1
            except Exception as e:
                logger.debug(f"Graph loss computation failed: {e}")
        
        # Node-level loss
        if node_targets and 'node_pred' in outputs and outputs['node_pred'] is not None:
            try:
                node_targets_tensor = torch.cat(node_targets).to(self.device)
                if outputs['node_pred'].shape == node_targets_tensor.shape:
                    node_loss = self.loss_fn(outputs['node_pred'], node_targets_tensor)
                    total_loss += node_loss
                    num_loss_terms += 1
            except Exception as e:
                logger.debug(f"Node loss computation failed: {e}")
        
        # Multi-scale losses (if available)
        for scale_idx in range(3):  # Assuming 3 scales
            scale_key = f'scale_{scale_idx}_graph_pred'
            if scale_key in outputs and graph_targets:
                try:
                    graph_targets_tensor = torch.stack(graph_targets).to(self.device)
                    if outputs[scale_key].shape == graph_targets_tensor.shape:
                        scale_loss = self.loss_fn(outputs[scale_key], graph_targets_tensor)
                        total_loss += 0.1 * scale_loss  # Weighted scale loss
                        num_loss_terms += 1
                except Exception as e:
                    logger.debug(f"Scale {scale_idx} loss computation failed: {e}")
        
        # Average loss terms
        if num_loss_terms > 0:
            total_loss = total_loss / num_loss_terms
        
        return total_loss
    
    def train_epoch(self) -> float:
        """Train HMGNN for one epoch."""
        self.hmgnn_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training HMGNN", leave=False)
        
        for batch_idx, raw_batch in enumerate(progress_bar):
            # Call batch begin callbacks
            self._call_callbacks("on_batch_begin", batch_idx)
            
            # Convert raw batch to hierarchical format
            batch_data = []
            for item in raw_batch:
                # Process each item in the batch
                hierarchical_item = self.hierarchical_processor.process_molecule(item)
                if 'targets' not in hierarchical_item and hasattr(item, 'y'):
                    hierarchical_item['targets'] = item.y
                if 'node_targets' not in hierarchical_item and hasattr(item, 'node_y'):
                    hierarchical_item['node_targets'] = item.node_y
                batch_data.append(hierarchical_item)
            
            # Prepare hierarchical batch
            hierarchical_batch = self._prepare_hierarchical_batch(batch_data)
            
            if not hierarchical_batch:
                continue
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.hmgnn_model(
                    scale_data=hierarchical_batch['scale_data'],
                    maps=(hierarchical_batch['mappings'], hierarchical_batch['cluster_counts']),
                    edge_pairs_cs=hierarchical_batch.get('cross_scale_edges'),
                    env_vec=None
                )
                
                # Compute loss
                loss = self._compute_hmgnn_loss(outputs, batch_data)
                
                # Backward pass
                if torch.isfinite(loss) and loss > 0:
                    loss.backward()
                    
                    # Gradient clipping
                    max_grad_norm = self.config.get("max_grad_norm", 1.0)
                    torch.nn.utils.clip_grad_norm_(self.hmgnn_model.parameters(), max_grad_norm)
                    
                    self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                continue
            
            # Call batch end callbacks
            batch_logs = {"loss": loss.item() if 'loss' in locals() else 0.0}
            self._call_callbacks("on_batch_end", batch_idx, logs=batch_logs)
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """Validate HMGNN model."""
        if self.val_loader is None:
            return 0.0
        
        self.hmgnn_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for raw_batch in self.val_loader:
                # Convert to hierarchical format
                batch_data = []
                for item in raw_batch:
                    hierarchical_item = self.hierarchical_processor.process_molecule(item)
                    if 'targets' not in hierarchical_item and hasattr(item, 'y'):
                        hierarchical_item['targets'] = item.y
                    if 'node_targets' not in hierarchical_item and hasattr(item, 'node_y'):
                        hierarchical_item['node_targets'] = item.node_y
                    batch_data.append(hierarchical_item)
                
                # Prepare hierarchical batch
                hierarchical_batch = self._prepare_hierarchical_batch(batch_data)
                
                if not hierarchical_batch:
                    continue
                
                try:
                    # Forward pass
                    outputs = self.hmgnn_model(
                        scale_data=hierarchical_batch['scale_data'],
                        maps=(hierarchical_batch['mappings'], hierarchical_batch['cluster_counts']),
                        edge_pairs_cs=hierarchical_batch.get('cross_scale_edges'),
                        env_vec=None
                    )
                    
                    # Compute loss
                    loss = self._compute_hmgnn_loss(outputs, batch_data)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.debug(f"Validation batch failed: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def create_hmgnn_trainer(
    config: Dict[str, Any],
    train_dataset: Any,
    val_dataset: Optional[Any] = None,
    device: Optional[str] = None
) -> HMGNNTrainer:
    """Create HMGNN trainer with hierarchical processing."""
    
    # Create hierarchical processor
    hierarchical_config = config.get("hierarchical", {})
    hierarchical_processor = create_hierarchical_processor(hierarchical_config)
    
    # Wrap datasets
    hierarchical_train_dataset = HierarchicalDataset(train_dataset, hierarchical_processor)
    hierarchical_val_dataset = None
    if val_dataset:
        hierarchical_val_dataset = HierarchicalDataset(val_dataset, hierarchical_processor)
    
    # Create data loaders
    batch_size = config.get("batch_size", 8)
    train_loader = GraphDataLoader(
        hierarchical_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.get("num_workers", 0)
    )
    
    val_loader = None
    if hierarchical_val_dataset:
        val_loader = GraphDataLoader(
            hierarchical_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 0)
        )
    
    # Create HMGNN model
    hmgnn_config = config.get("hmgnn", {})
    model = create_hierarchical_mgnn(hmgnn_config)
    
    # Create optimizer
    optimizer_type = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0)
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create loss function
    loss_fn = nn.MSELoss()
    
    # Create callbacks
    callbacks = []
    if config.get("early_stopping", {}).get("enabled", False):
        callbacks.append(EarlyStopping(
            patience=config["early_stopping"].get("patience", 10),
            min_delta=config["early_stopping"].get("min_delta", 1e-4)
        ))
    
    if config.get("checkpointing", {}).get("enabled", False):
        callbacks.append(ModelCheckpoint(
            filepath=config["checkpointing"].get("filepath", "hmgnn_checkpoint.pt"),
            save_best_only=config["checkpointing"].get("save_best_only", True)
        ))
    
    # Create trainer
    trainer = HMGNNTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        callbacks=callbacks,
        hierarchical_processor=hierarchical_processor
    )
    
    return trainer


def main():
    """Main function for HMGNN standalone training."""
    parser = argparse.ArgumentParser(
        description="Standalone training for HMGNN on hierarchical molecular graphs"
    )
    
    # Training parameters
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset name")
    parser.add_argument("--data_root", type=str, default="data", help="Data root directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of GNN blocks per scale")
    parser.add_argument("--layers_per_block", type=int, default=3, help="Layers per block")
    parser.add_argument("--n_scales", type=int, default=3, help="Number of hierarchical scales")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_hmgnn", 
                       help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save frequency")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_every", type=int, default=10, help="Logging frequency")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting HMGNN standalone training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Override config with command line arguments
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "checkpoint_dir": args.checkpoint_dir,
        "log_every": args.log_every
    })
    
    # HMGNN specific configuration
    config["hmgnn"] = {
        "scale_dims": [DEFAULT_NODE_FEATURE_DIM] * args.n_scales,
        "hidden_dim": args.hidden_dim,
        "n_blocks": args.n_blocks,
        "layers_per_block": args.layers_per_block,
        "jk_mode": "attention",
        "node_out_dim": 1,
        "graph_out_dim": config.get("graph_output_dims", 19),
        "cross_scale_exchange": True,
        "dropout": 0.2,
        "n_heads_cs": 4,
        "edge_dim_cs": 0,
        "pool_type": "mean"
    }
    
    # Hierarchical processing configuration
    config["hierarchical"] = {
        "coarsener": {
            "n_levels": args.n_scales,
            "clustering_method": "functional_groups",
            "preserve_connectivity": True
        },
        "processor": {
            "include_cross_scale_edges": True,
            "cache_hierarchical": True
        }
    }
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load dataset
    try:
        transform = Compose([
            CreateEdges(),
            FeaturizeNodes(),
            StandardizeTargets(dataset_name=args.dataset)
        ])
        
        dataset = get_dataset(args.dataset, root=args.data_root, transform=transform)
        logger.info(f"Loaded {args.dataset} dataset: {len(dataset)} molecules")
        
        # Split dataset
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create trainer
    try:
        trainer = create_hmgnn_trainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device
        )
        logger.info("Created HMGNN trainer successfully")
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    try:
        logger.info(f"Starting training for {args.epochs} epochs")
        start_time = time.time()
        
        history = trainer.train(epochs=args.epochs, log_interval=args.log_every)
        
        end_time = time.time()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Save final model
        final_checkpoint = os.path.join(args.checkpoint_dir, "hmgnn_final.pt")
        trainer.save_checkpoint(final_checkpoint)
        logger.info(f"Final model saved to: {final_checkpoint}")
        
        # Print training summary
        if history["train_loss"]:
            logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
        if history["val_loss"]:
            logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
            logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    logger.info("HMGNN standalone training completed successfully")


if __name__ == "__main__":
    main()