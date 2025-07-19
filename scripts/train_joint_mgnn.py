"""
scripts/train_joint_mgnn.py

Joint training script for DJMGNN and HMGNN models.

This script implements the coordinated training workflow for the hybrid molecular
machine learning framework, combining Dense Junction Molecular Graph Neural Network 
(DJMGNN) and Hierarchical Molecular Graph Neural Network (HMGNN) through joint 
optimization, cross-model fusion, and multi-task learning.

Key Features:
    - Joint model training with cross-model attention
    - Alternating optimization strategies
    - Multi-task learning for various molecular properties
    - Hierarchical data processing pipeline
    - Pre-training, joint training, and fine-tuning phases
    - Comprehensive evaluation and benchmarking
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

from moml.core.hierarchical_processor import create_hierarchical_processor
from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes, StandardizeTargets)
from moml.models.mgnn import (
    DJMGNN, HMGNN, JointMGNN, create_joint_mgnn,
    JointMGNNTrainer, create_joint_trainer,
    EarlyStopping, ModelCheckpoint
)

DEFAULT_NODE_FEATURE_DIM = 29
LOG_FILE_NAME = "joint_mgnn_training.log"

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for joint training."""
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
    """Dataset wrapper that creates hierarchical representations on-the-fly."""
    
    def __init__(self, base_dataset: Any, hierarchical_processor: Any):
        self.base_dataset = base_dataset
        self.hierarchical_processor = hierarchical_processor
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_data = self.base_dataset[idx]
        
        # Create hierarchical representation
        hierarchical_data = self.hierarchical_processor.process_molecule(
            base_data, mol=None, molecule_id=f"mol_{idx}"
        )
        
        # Add original data and targets
        hierarchical_data['original_data'] = base_data
        if hasattr(base_data, 'y'):
            hierarchical_data['targets'] = base_data.y
        if hasattr(base_data, 'node_y'):
            hierarchical_data['node_targets'] = base_data.node_y
        
        return hierarchical_data


def load_pretrained_models(
    djmgnn_checkpoint: Optional[str],
    hmgnn_checkpoint: Optional[str],
    djmgnn_config: Dict[str, Any],
    hmgnn_config: Dict[str, Any],
    device: str
) -> tuple[Optional[DJMGNN], Optional[HMGNN]]:
    """
    Load pretrained DJMGNN and HMGNN models from checkpoints.
    
    Args:
        djmgnn_checkpoint: Path to DJMGNN checkpoint
        hmgnn_checkpoint: Path to HMGNN checkpoint
        djmgnn_config: DJMGNN configuration
        hmgnn_config: HMGNN configuration
        device: Device to load models on
        
    Returns:
        Tuple of (djmgnn_model, hmgnn_model)
    """
    djmgnn_model = None
    hmgnn_model = None
    
    if djmgnn_checkpoint and os.path.exists(djmgnn_checkpoint):
        logger.info(f"Loading pretrained DJMGNN from {djmgnn_checkpoint}")
        try:
            djmgnn_model = DJMGNN(**djmgnn_config)
            checkpoint = torch.load(djmgnn_checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                djmgnn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                djmgnn_model.load_state_dict(checkpoint)
            
            djmgnn_model.to(device)
            logger.info("Successfully loaded pretrained DJMGNN")
            
        except Exception as e:
            logger.warning(f"Failed to load DJMGNN checkpoint: {e}")
            djmgnn_model = None
    
    if hmgnn_checkpoint and os.path.exists(hmgnn_checkpoint):
        logger.info(f"Loading pretrained HMGNN from {hmgnn_checkpoint}")
        try:
            from moml.models.mgnn.hmgnn import create_hierarchical_mgnn
            hmgnn_model = create_hierarchical_mgnn(hmgnn_config)
            checkpoint = torch.load(hmgnn_checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                hmgnn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                hmgnn_model.load_state_dict(checkpoint)
            
            hmgnn_model.to(device)
            logger.info("Successfully loaded pretrained HMGNN")
            
        except Exception as e:
            logger.warning(f"Failed to load HMGNN checkpoint: {e}")
            hmgnn_model = None
    
    return djmgnn_model, hmgnn_model


def create_joint_data_loaders(
    train_dataset: Any,
    val_dataset: Optional[Any],
    hierarchical_processor: Any,
    batch_size: int = 8,
    num_workers: int = 0
) -> tuple[GraphDataLoader, Optional[GraphDataLoader], Optional[GraphDataLoader], Optional[GraphDataLoader]]:
    """
    Create data loaders for joint training.
    
    Returns:
        Tuple of (standard_train_loader, standard_val_loader, hierarchical_train_loader, hierarchical_val_loader)
    """
    # Standard data loaders
    standard_train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    standard_val_loader = None
    if val_dataset:
        standard_val_loader = GraphDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    # Hierarchical data loaders
    hierarchical_train_dataset = HierarchicalDataset(train_dataset, hierarchical_processor)
    hierarchical_train_loader = GraphDataLoader(
        hierarchical_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    hierarchical_val_loader = None
    if val_dataset:
        hierarchical_val_dataset = HierarchicalDataset(val_dataset, hierarchical_processor)
        hierarchical_val_loader = GraphDataLoader(
            hierarchical_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return (standard_train_loader, standard_val_loader, 
            hierarchical_train_loader, hierarchical_val_loader)


def run_pretraining_phase(
    config: Dict[str, Any],
    train_dataset: Any,
    val_dataset: Optional[Any],
    device: str,
    hierarchical_processor: Any
) -> tuple[Optional[str], Optional[str]]:
    """
    Run pretraining phase for DJMGNN and HMGNN separately.
    
    Returns:
        Tuple of (djmgnn_checkpoint_path, hmgnn_checkpoint_path)
    """
    pretraining_epochs = config.get("pretraining_epochs", 10)
    batch_size = config.get("batch_size", 8)
    
    logger.info(f"Starting pretraining phase for {pretraining_epochs} epochs")
    
    djmgnn_checkpoint = None
    hmgnn_checkpoint = None
    
    # Create data loaders
    standard_train_loader, standard_val_loader, hierarchical_train_loader, hierarchical_val_loader = create_joint_data_loaders(
        train_dataset, val_dataset, hierarchical_processor, batch_size
    )
    
    # Pretrain DJMGNN
    if config.get("pretrain_djmgnn", True):
        logger.info("Pretraining DJMGNN...")
        try:
            from moml.models.mgnn.training import create_trainer
            
            djmgnn_config = config["djmgnn"].copy()
            djmgnn_config["epochs"] = pretraining_epochs
            
            djmgnn_trainer = create_trainer(
                config=djmgnn_config,
                train_loader=standard_train_loader
            )
            djmgnn_trainer.device = device
            djmgnn_trainer.model = djmgnn_trainer.model.to(device)
            
            djmgnn_trainer.train(epochs=pretraining_epochs)
            
            # Save checkpoint
            djmgnn_checkpoint = os.path.join(
                config.get("checkpoint_dir", "checkpoints"), 
                "djmgnn_pretrained.pt"
            )
            djmgnn_trainer.save_checkpoint(djmgnn_checkpoint)
            logger.info(f"DJMGNN pretraining completed, saved to {djmgnn_checkpoint}")
            
        except Exception as e:
            logger.error(f"DJMGNN pretraining failed: {e}")
    
    # Pretrain HMGNN
    if config.get("pretrain_hmgnn", True):
        logger.info("Pretraining HMGNN...")
        try:
            from scripts.train_hmgnn_standalone import create_hmgnn_trainer
            
            hmgnn_config = config.copy()
            hmgnn_config["epochs"] = pretraining_epochs
            
            hmgnn_trainer = create_hmgnn_trainer(
                config=hmgnn_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device
            )
            
            hmgnn_trainer.train(epochs=pretraining_epochs)
            
            # Save checkpoint
            hmgnn_checkpoint = os.path.join(
                config.get("checkpoint_dir", "checkpoints"),
                "hmgnn_pretrained.pt"
            )
            hmgnn_trainer.save_checkpoint(hmgnn_checkpoint)
            logger.info(f"HMGNN pretraining completed, saved to {hmgnn_checkpoint}")
            
        except Exception as e:
            logger.error(f"HMGNN pretraining failed: {e}")
    
    return djmgnn_checkpoint, hmgnn_checkpoint


def main():
    """Main function for joint DJMGNN and HMGNN training."""
    parser = argparse.ArgumentParser(
        description="Joint training for DJMGNN and HMGNN models"
    )
    
    # Configuration
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to joint training configuration file")
    
    # Training phases
    parser.add_argument("--phase", type=str, default="all",
                       choices=["pretraining", "joint", "fine_tuning", "all"],
                       help="Training phase to execute")
    
    # Pretrained models
    parser.add_argument("--djmgnn_checkpoint", type=str,
                       help="Path to pretrained DJMGNN checkpoint")
    parser.add_argument("--hmgnn_checkpoint", type=str,
                       help="Path to pretrained HMGNN checkpoint")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Primary dataset")
    parser.add_argument("--data_root", type=str, default="data", help="Data root directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    
    # Training strategy
    parser.add_argument("--training_strategy", type=str, default="joint",
                       choices=["joint", "alternating"],
                       help="Training strategy for joint phase")
    
    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_joint",
                       help="Directory for saving checkpoints")
    parser.add_argument("--output_dir", type=str, default="output_joint",
                       help="Directory for saving results")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_every", type=int, default=10, help="Logging frequency")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting joint DJMGNN and HMGNN training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Override config with command line arguments
    if args.epochs:
        config["joint_training_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["joint_lr"] = args.lr
    
    config["checkpoint_dir"] = args.checkpoint_dir
    config["training_strategy"] = args.training_strategy
    
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
        
        logger.info(f"Dataset split: {train_size} train, {val_size} validation")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create hierarchical processor
    try:
        hierarchical_config = config.get("hierarchical", {})
        hierarchical_processor = create_hierarchical_processor(hierarchical_config)
        logger.info("Created hierarchical processor")
    except Exception as e:
        logger.error(f"Failed to create hierarchical processor: {e}")
        return
    
    # Initialize model configurations
    djmgnn_config = config.get("djmgnn", {})
    djmgnn_config.setdefault("in_node_dim", DEFAULT_NODE_FEATURE_DIM)
    djmgnn_config.setdefault("hidden_dim", 64)
    djmgnn_config.setdefault("node_output_dims", 1)
    djmgnn_config.setdefault("graph_output_dims", 19)
    
    hmgnn_config = config.get("hmgnn", {})
    hmgnn_config.setdefault("scale_dims", [DEFAULT_NODE_FEATURE_DIM] * 3)
    hmgnn_config.setdefault("hidden_dim", 64)
    hmgnn_config.setdefault("node_out_dim", 1)
    hmgnn_config.setdefault("graph_out_dim", 19)
    
    joint_config = config.get("joint", {})
    joint_config.setdefault("fusion_dim", 128)
    joint_config.setdefault("training_strategy", args.training_strategy)
    joint_config.update(config)  # Include all config for trainer
    
    # Execute training phases
    djmgnn_checkpoint_path = args.djmgnn_checkpoint
    hmgnn_checkpoint_path = args.hmgnn_checkpoint
    
    # Phase 1: Pretraining
    if args.phase in ["pretraining", "all"]:
        logger.info("=== PHASE 1: PRETRAINING ===")
        try:
            djmgnn_checkpoint_path, hmgnn_checkpoint_path = run_pretraining_phase(
                config, train_dataset, val_dataset, device, hierarchical_processor
            )
        except Exception as e:
            logger.error(f"Pretraining phase failed: {e}")
            if args.phase == "pretraining":
                return
    
    # Phase 2: Joint Training
    if args.phase in ["joint", "fine_tuning", "all"]:
        logger.info("=== PHASE 2: JOINT TRAINING ===")
        try:
            # Create data loaders
            standard_train_loader, standard_val_loader, hierarchical_train_loader, hierarchical_val_loader = create_joint_data_loaders(
                train_dataset, val_dataset, hierarchical_processor, 
                config.get("batch_size", 8), config.get("num_workers", 0)
            )
            
            # Create joint trainer
            joint_trainer = create_joint_trainer(
                djmgnn_config=djmgnn_config,
                hmgnn_config=hmgnn_config,
                joint_config=joint_config,
                train_loader=standard_train_loader,
                val_loader=standard_val_loader,
                hierarchical_train_loader=hierarchical_train_loader,
                hierarchical_val_loader=hierarchical_val_loader,
                device=device
            )
            
            # Load pretrained models if available
            if djmgnn_checkpoint_path or hmgnn_checkpoint_path:
                logger.info("Initializing joint model with pretrained components")
                djmgnn_model, hmgnn_model = load_pretrained_models(
                    djmgnn_checkpoint_path, hmgnn_checkpoint_path,
                    djmgnn_config, hmgnn_config, device
                )
                
                # Transfer weights to joint model
                if djmgnn_model:
                    joint_trainer.joint_model.djmgnn.load_state_dict(djmgnn_model.state_dict())
                if hmgnn_model:
                    joint_trainer.joint_model.hmgnn.load_state_dict(hmgnn_model.state_dict())
            
            # Train joint model
            joint_epochs = config.get("joint_training_epochs", 50)
            logger.info(f"Starting joint training for {joint_epochs} epochs")
            
            start_time = time.time()
            history = joint_trainer.train(epochs=joint_epochs, log_interval=args.log_every)
            end_time = time.time()
            
            logger.info(f"Joint training completed in {end_time - start_time:.2f} seconds")
            
            # Save joint model
            joint_checkpoint = os.path.join(args.checkpoint_dir, "joint_mgnn_trained.pt")
            joint_trainer.save_checkpoint(joint_checkpoint)
            logger.info(f"Joint model saved to {joint_checkpoint}")
            
            # Print training summary
            if history["train_loss"]:
                logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
            if history["val_loss"]:
                logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
                logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
            
        except Exception as e:
            logger.error(f"Joint training phase failed: {e}")
            return
    
    # Phase 3: Fine-tuning (PFAS-specific)
    if args.phase in ["fine_tuning", "all"]:
        logger.info("=== PHASE 3: FINE-TUNING ===")
        # This would involve PFAS-specific dataset and fine-tuning
        # Implementation depends on availability of PFAS data
        logger.info("PFAS fine-tuning phase - implementation pending PFAS dataset")
    
    logger.info("Joint training pipeline completed successfully!")
    
    # Save configuration for reproducibility
    config_save_path = os.path.join(args.output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_save_path}")


if __name__ == "__main__":
    main()