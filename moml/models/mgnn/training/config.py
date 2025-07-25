"""
moml/models/mgnn/training/config.py

Comprehensive Training Configuration System for Early Stopping and Monitoring.

This module provides a Hydra-based configuration system for managing complex
training configurations including early stopping, monitoring, and checkpoint
management parameters. It follows modern MLOps best practices for configuration
management with type safety, validation, and hierarchical composition.

Key Features:
    - Type-safe configuration with dataclasses and OmegaConf
    - Hierarchical configuration composition with Hydra
    - Validation and constraint checking
    - Environment variable integration
    - Configuration templates for common use cases
    - Integration with WandB, TensorBoard, and custom monitoring
    - Checkpoint and model saving strategies

Main Components:
    - TrainingConfig: Main training configuration dataclass
    - EarlyStoppingConfig: Early stopping specific configuration
    - MonitoringConfig: Monitoring and visualization configuration
    - CheckpointConfig: Model checkpoint management configuration
    - DataConfig: Data loading and preprocessing configuration
    - ConfigManager: Configuration management and validation utilities

References:
    - Based on Hydra and OmegaConf best practices from 2024
    - Follows MLOps configuration patterns from leading organizations
    - Integrates with modern monitoring and experiment tracking tools
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import logging

import torch

# Configuration management imports
try:
    from omegaconf import OmegaConf, DictConfig
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    OmegaConf = None
    DictConfig = None
    ConfigStore = None

# Validation utilities
try:
    from pydantic import BaseModel, validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = None

logger = logging.getLogger(__name__)


class OptimzerType(Enum):
    """Enumeration for optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class SchedulerType(Enum):
    """Enumeration for learning rate scheduler types."""
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    PLATEAU = "plateau"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"


class ModelType(Enum):
    """Enumeration for model types."""
    DJMGNN = "djmgnn"
    HMGNN = "hmgnn"
    JOINT_MGNN = "joint_mgnn"
    PIMEH = "pimeh"


class MonitorMode(Enum):
    """Enumeration for monitoring modes."""
    MIN = "min"
    MAX = "max"
    AUTO = "auto"


class DeviceType(Enum):
    """Enumeration for device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer settings.
    
    Attributes:
        type: Type of optimizer to use
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum factor (for SGD, RMSprop)
        beta1: Beta1 parameter (for Adam, AdamW)
        beta2: Beta2 parameter (for Adam, AdamW)
        eps: Epsilon parameter for numerical stability
        amsgrad: Whether to use AMSGrad variant (for Adam)
        alpha: Alpha parameter (for RMSprop)
        centered: Whether to use centered variant (for RMSprop)
    """
    type: OptimzerType = OptimzerType.ADAMW
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    alpha: float = 0.99
    centered: bool = False
    
    def __post_init__(self):
        """Validate optimizer configuration."""
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
        if not 0 <= self.momentum <= 1:
            raise ValueError(f"Momentum must be in [0, 1], got {self.momentum}")


@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate scheduler.
    
    Attributes:
        type: Type of scheduler to use
        step_size: Step size for StepLR
        gamma: Multiplicative factor for learning rate decay
        T_max: Maximum number of iterations for CosineAnnealingLR
        eta_min: Minimum learning rate for CosineAnnealingLR
        patience: Patience for ReduceLROnPlateau
        factor: Factor by which to reduce LR for ReduceLROnPlateau
        threshold: Threshold for measuring improvement
        cooldown: Cooldown period after LR reduction
        min_lr: Minimum learning rate
        base_lr: Base learning rate for CyclicLR
        max_lr: Maximum learning rate for CyclicLR
        step_size_up: Number of training iterations in increasing half of cycle
        mode: Mode for CyclicLR ('triangular', 'triangular2', 'exp_range')
        max_lr_one_cycle: Maximum learning rate for OneCycleLR
        pct_start: Percentage of cycle spent increasing learning rate
        anneal_strategy: Annealing strategy for OneCycleLR ('cos', 'linear')
        div_factor: Initial learning rate will be lr/div_factor
        final_div_factor: Final learning rate will be lr/final_div_factor
    """
    type: SchedulerType = SchedulerType.COSINE
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 1e-6
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 1e-8
    base_lr: float = 1e-4
    max_lr: float = 1e-2
    step_size_up: int = 2000
    mode: str = "triangular"
    max_lr_one_cycle: float = 1e-2
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    div_factor: float = 25.0
    final_div_factor: float = 1e4


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping mechanism.
    
    Attributes:
        enabled: Whether to enable early stopping
        monitor: Metric to monitor for early stopping
        mode: Whether to minimize or maximize the monitored metric
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best weights when stopping
        verbose: Verbosity level (0=silent, 1=epoch info, 2=detailed)
        
        # Advanced Features
        warmup_epochs: Number of epochs before early stopping is active
        multiple_metrics: Additional metrics to monitor for stopping
        metric_weights: Weights for combining multiple metrics
        statistical_test: Whether to use statistical significance testing
        confidence_level: Confidence level for statistical testing
        
        # Learning Rate Integration
        reduce_lr_on_plateau: Whether to reduce LR before stopping
        lr_reduction_factor: Factor to reduce learning rate by
        lr_reduction_patience: Patience for learning rate reduction
        min_lr_threshold: Minimum learning rate threshold
        
        # Divergence Detection
        detect_divergence: Whether to detect training divergence
        divergence_threshold: Threshold for detecting divergence
        divergence_patience: Patience for divergence detection
    """
    enabled: bool = True
    monitor: str = "val_loss"
    mode: MonitorMode = MonitorMode.MIN
    patience: int = 15
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    verbose: int = 1
    
    # Advanced Features
    warmup_epochs: int = 5
    multiple_metrics: List[str] = field(default_factory=list)
    metric_weights: Dict[str, float] = field(default_factory=dict)
    statistical_test: bool = False
    confidence_level: float = 0.95
    
    # Learning Rate Integration
    reduce_lr_on_plateau: bool = True
    lr_reduction_factor: float = 0.5
    lr_reduction_patience: int = 5
    min_lr_threshold: float = 1e-8
    
    # Divergence Detection
    detect_divergence: bool = True
    divergence_threshold: float = 10.0
    divergence_patience: int = 3
    
    def __post_init__(self):
        """Validate early stopping configuration."""
        if self.patience <= 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError(f"Confidence level must be in [0.5, 0.99], got {self.confidence_level}")
        if self.lr_reduction_factor <= 0 or self.lr_reduction_factor >= 1:
            raise ValueError(f"LR reduction factor must be in (0, 1), got {self.lr_reduction_factor}")


@dataclass
class CheckpointConfig:
    """
    Configuration for model checkpointing.
    
    Attributes:
        enabled: Whether to enable checkpointing
        save_best_only: Whether to save only the best model
        save_last: Whether to save the last model
        monitor: Metric to monitor for best model
        mode: Whether to minimize or maximize the monitored metric
        save_top_k: Number of best models to keep
        every_n_epochs: Save checkpoint every N epochs
        save_on_train_epoch_end: Whether to save after training epochs
        save_on_validation_epoch_end: Whether to save after validation epochs
        
        # File Management
        dirpath: Directory to save checkpoints
        filename: Filename template for checkpoints
        auto_insert_metric_name: Whether to auto-insert metric name in filename
        
        # Content
        save_weights_only: Whether to save only model weights
        save_optimizer_state: Whether to save optimizer state
        save_lr_scheduler_state: Whether to save LR scheduler state
        save_epoch: Whether to save epoch number
        save_hyperparameters: Whether to save hyperparameters
        
        # Compression
        enable_compression: Whether to enable checkpoint compression
        compression_format: Compression format ('gzip', 'bz2', 'lzma')
    """
    enabled: bool = True
    save_best_only: bool = True
    save_last: bool = True
    monitor: str = "val_loss"
    mode: MonitorMode = MonitorMode.MIN
    save_top_k: int = 3
    every_n_epochs: int = 1
    save_on_train_epoch_end: bool = False
    save_on_validation_epoch_end: bool = True
    
    # File Management
    dirpath: str = "checkpoints"
    filename: str = "epoch_{epoch:02d}_val_loss_{val_loss:.4f}"
    auto_insert_metric_name: bool = True
    
    # Content
    save_weights_only: bool = False
    save_optimizer_state: bool = True
    save_lr_scheduler_state: bool = True
    save_epoch: bool = True
    save_hyperparameters: bool = True
    
    # Compression
    enable_compression: bool = False
    compression_format: str = "gzip"
    
    def __post_init__(self):
        """Validate checkpoint configuration."""
        if self.save_top_k < 0:
            raise ValueError(f"save_top_k must be non-negative, got {self.save_top_k}")
        if self.every_n_epochs <= 0:
            raise ValueError(f"every_n_epochs must be positive, got {self.every_n_epochs}")
        
        # Create checkpoint directory
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)


@dataclass
class MonitoringConfig:
    """
    Configuration for training monitoring and visualization.
    
    Attributes:
        enabled: Whether to enable monitoring
        log_every_n_steps: Log metrics every N training steps
        log_every_n_epochs: Log metrics every N epochs
        
        # Metrics
        primary_metrics: List of primary metrics to monitor prominently
        secondary_metrics: List of secondary metrics for detailed view
        custom_metrics: Dictionary of custom derived metrics to compute
        
        # Visualization
        enable_visualization: Whether to enable real-time visualization
        visualization_theme: Theme for visualizations
        figure_size: Default figure size (width, height) in inches
        update_interval: Update interval for real-time plots (seconds)
        max_points_display: Maximum number of points to display in plots
        show_trend_lines: Whether to show trend lines in plots
        show_moving_averages: Whether to show moving averages
        moving_average_window: Window size for moving averages
        
        # Alerts
        enable_alerts: Whether to enable alert system
        alert_thresholds: Thresholds for different alert levels
        alert_cooldown: Cooldown period between similar alerts (seconds)
        
        # Export
        export_dir: Directory for exporting plots and reports
        auto_export: Whether to automatically export plots
        export_formats: List of formats to export
        
        # Integration
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        wandb_tags: Tags for the WandB run
        wandb_notes: Notes for the WandB run
        log_to_wandb: Whether to log to Weights & Biases
        log_to_tensorboard: Whether to log to TensorBoard
        tensorboard_log_dir: Directory for TensorBoard logs
        create_html_dashboard: Whether to create HTML dashboard
    """
    enabled: bool = True
    log_every_n_steps: int = 10
    log_every_n_epochs: int = 1
    
    # Metrics
    primary_metrics: List[str] = field(default_factory=lambda: ["train_loss", "val_loss"])
    secondary_metrics: List[str] = field(default_factory=lambda: ["learning_rate", "grad_norm"])
    custom_metrics: Dict[str, str] = field(default_factory=dict)
    
    # Visualization
    enable_visualization: bool = True
    visualization_theme: str = "light"
    figure_size: Tuple[int, int] = (12, 8)
    update_interval: float = 5.0
    max_points_display: int = 1000
    show_trend_lines: bool = True
    show_moving_averages: bool = True
    moving_average_window: int = 10
    
    # Alerts
    enable_alerts: bool = True
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    alert_cooldown: float = 300.0
    
    # Export
    export_dir: str = "monitoring_exports"
    auto_export: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["png", "html"])
    
    # Integration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    log_to_wandb: bool = True
    log_to_tensorboard: bool = True
    tensorboard_log_dir: str = "runs"
    create_html_dashboard: bool = True
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive, got {self.log_every_n_steps}")
        if self.log_every_n_epochs <= 0:
            raise ValueError(f"log_every_n_epochs must be positive, got {self.log_every_n_epochs}")
        
        # Create export directory
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default alert thresholds if not provided
        if not self.alert_thresholds:
            self.alert_thresholds = {
                "val_loss": {"warning": 1.0, "error": 5.0, "critical": 10.0},
                "grad_norm": {"warning": 10.0, "error": 100.0, "critical": 1000.0},
            }


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes:
        # Dataset
        dataset_name: Name of the dataset
        data_dir: Directory containing the data
        train_split: Training data split ratio
        val_split: Validation data split ratio
        test_split: Test data split ratio
        random_seed: Random seed for data splitting
        
        # Data Loading
        batch_size: Batch size for training
        val_batch_size: Batch size for validation (if different)
        test_batch_size: Batch size for testing (if different)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for data loading
        persistent_workers: Whether to use persistent workers
        drop_last: Whether to drop the last incomplete batch
        
        # Preprocessing
        normalize_features: Whether to normalize input features
        standardize_targets: Whether to standardize target values
        feature_scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        target_scaling_method: Method for target scaling
        handle_missing_values: How to handle missing values ('drop', 'mean', 'median', 'zero')
        
        # Augmentation
        enable_augmentation: Whether to enable data augmentation
        augmentation_prob: Probability of applying augmentation
        rotation_prob: Probability of molecular rotation
        noise_level: Level of noise to add for augmentation
        
        # Molecular-specific
        max_atoms: Maximum number of atoms per molecule
        atom_features: List of atomic features to include
        bond_features: List of bond features to include
        use_3d_coordinates: Whether to use 3D molecular coordinates
        add_hydrogen: Whether to add explicit hydrogen atoms
    """
    # Dataset
    dataset_name: str = "qm9"
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Data Loading
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    
    # Preprocessing
    normalize_features: bool = True
    standardize_targets: bool = True
    feature_scaling_method: str = "standard"
    target_scaling_method: str = "standard"
    handle_missing_values: str = "mean"
    
    # Augmentation
    enable_augmentation: bool = False
    augmentation_prob: float = 0.5
    rotation_prob: float = 0.3
    noise_level: float = 0.01
    
    # Molecular-specific
    max_atoms: int = 50
    atom_features: List[str] = field(default_factory=lambda: ["atomic_num", "degree", "formal_charge"])
    bond_features: List[str] = field(default_factory=lambda: ["bond_type", "is_conjugated"])
    use_3d_coordinates: bool = True
    add_hydrogen: bool = False
    
    def __post_init__(self):
        """Validate data configuration."""
        # Validate split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
        
        # Set default batch sizes
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size
        
        # Validate batch sizes
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """
    Configuration for model architecture and parameters.
    
    Attributes:
        # Model Type
        model_type: Type of model to use
        model_name: Specific model name/variant
        
        # Architecture
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        num_heads: Number of attention heads (for transformer-based models)
        dropout: Dropout probability
        activation: Activation function
        norm_type: Normalization type ('batch', 'layer', 'instance')
        
        # MGNN-specific
        use_edge_features: Whether to use edge features
        edge_dim: Edge feature dimension
        use_global_features: Whether to use global features
        global_dim: Global feature dimension
        pool_type: Pooling type for graph-level predictions
        
        # Output
        output_dim: Output dimension
        output_activation: Output activation function
        
        # Initialization
        init_type: Weight initialization type
        init_gain: Gain factor for initialization
        
        # Advanced
        use_residual: Whether to use residual connections
        use_batch_norm: Whether to use batch normalization
        use_layer_norm: Whether to use layer normalization
        gradient_clipping: Maximum gradient norm for clipping
    """
    # Model Type
    model_type: ModelType = ModelType.DJMGNN
    model_name: str = "djmgnn_base"
    
    # Architecture
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "relu"
    norm_type: str = "layer"
    
    # MGNN-specific
    use_edge_features: bool = True
    edge_dim: int = 32
    use_global_features: bool = True
    global_dim: int = 64
    pool_type: str = "mean"
    
    # Output
    output_dim: int = 1
    output_activation: Optional[str] = None
    
    # Initialization
    init_type: str = "xavier_uniform"
    init_gain: float = 1.0
    
    # Advanced
    use_residual: bool = True
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    gradient_clipping: float = 1.0
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.num_layers}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be in [0, 1], got {self.dropout}")


@dataclass
class TrainingConfig:
    """
    Main training configuration combining all sub-configurations.
    
    Attributes:
        # Experiment
        experiment_name: Name of the experiment
        run_name: Name of this specific run
        tags: Tags for experiment tracking
        notes: Notes about the experiment
        
        # Training
        max_epochs: Maximum number of training epochs
        min_epochs: Minimum number of training epochs
        max_steps: Maximum number of training steps
        val_check_interval: Validation check interval
        log_every_n_steps: Log metrics every N steps
        
        # Device and Performance
        accelerator: Accelerator type ('auto', 'cpu', 'gpu', 'tpu')
        devices: Number of devices to use
        precision: Training precision ('32', '16', 'bf16')
        strategy: Training strategy ('auto', 'ddp', 'ddp_spawn', 'deepspeed')
        enable_progress_bar: Whether to show progress bar
        
        # Reproducibility
        seed: Random seed for reproducibility
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to enable cudnn benchmarking
        
        # Sub-configurations
        model: Model configuration
        data: Data configuration
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        early_stopping: Early stopping configuration
        checkpoint: Checkpoint configuration
        monitoring: Monitoring configuration
        
        # Environment
        work_dir: Working directory
        log_dir: Logging directory
        cache_dir: Cache directory
        
        # Debugging
        debug: Whether to enable debug mode
        profiler: Profiler type ('simple', 'advanced', 'pytorch')
        detect_anomaly: Whether to detect anomalies
        track_grad_norm: Whether to track gradient norms
    """
    # Experiment
    experiment_name: str = "mgnn_training"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Training
    max_epochs: int = 100
    min_epochs: int = 1
    max_steps: int = -1
    val_check_interval: Union[int, float] = 1.0
    log_every_n_steps: int = 50
    
    # Device and Performance
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    precision: str = "32"
    strategy: str = "auto"
    enable_progress_bar: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment
    work_dir: str = "."
    log_dir: str = "logs"
    cache_dir: str = ".cache"
    
    # Debugging
    debug: bool = False
    profiler: Optional[str] = None
    detect_anomaly: bool = False
    track_grad_norm: bool = True
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.max_epochs <= 0:
            raise ValueError(f"Max epochs must be positive, got {self.max_epochs}")
        if self.min_epochs <= 0:
            raise ValueError(f"Min epochs must be positive, got {self.min_epochs}")
        if self.min_epochs > self.max_epochs:
            raise ValueError(f"Min epochs ({self.min_epochs}) cannot exceed max epochs ({self.max_epochs})")
        
        # Set default run name
        if self.run_name is None:
            import time
            self.run_name = f"{self.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        for dir_path in [self.work_dir, self.log_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate sub-configurations
        # (Post-init methods of sub-configs will be called automatically)


class ConfigManager:
    """
    Configuration management utilities for training configurations.
    
    This class provides utilities for loading, saving, validating, and
    managing training configurations with Hydra integration.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Register configuration schemas if Hydra is available
        if HYDRA_AVAILABLE:
            self._register_configs()
    
    def _register_configs(self) -> None:
        """Register configuration schemas with Hydra ConfigStore."""
        cs = ConfigStore.instance()
        
        # Register main configuration
        cs.store(name="base_config", node=TrainingConfig)
        
        # Register sub-configurations
        cs.store(group="model", name="djmgnn", node=ModelConfig(model_type=ModelType.DJMGNN))
        cs.store(group="model", name="hmgnn", node=ModelConfig(model_type=ModelType.HMGNN))
        cs.store(group="model", name="joint_mgnn", node=ModelConfig(model_type=ModelType.JOINT_MGNN))
        
        cs.store(group="optimizer", name="adam", node=OptimizerConfig(type=OptimzerType.ADAM))
        cs.store(group="optimizer", name="adamw", node=OptimizerConfig(type=OptimzerType.ADAMW))
        cs.store(group="optimizer", name="sgd", node=OptimizerConfig(type=OptimzerType.SGD))
        
        cs.store(group="scheduler", name="cosine", node=SchedulerConfig(type=SchedulerType.COSINE))
        cs.store(group="scheduler", name="step", node=SchedulerConfig(type=SchedulerType.STEP))
        cs.store(group="scheduler", name="plateau", node=SchedulerConfig(type=SchedulerType.PLATEAU))
    
    def load_config(self, config_name: str = "base_config", overrides: Optional[List[str]] = None) -> TrainingConfig:
        """
        Load configuration using Hydra.
        
        Args:
            config_name: Name of the configuration to load
            overrides: List of configuration overrides
            
        Returns:
            Loaded and validated training configuration
        """
        if not HYDRA_AVAILABLE:
            # Fallback to JSON loading when Hydra not available
            logger.warning("Hydra not available, attempting to load from JSON")
            
            # Try to load from JSON file
            json_path = self.config_dir / f"{config_name}.json"
            if json_path.exists():
                try:
                    import json
                    with open(json_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Convert enum strings back to enum objects
                    config_dict = self._deserialize_enums(config_dict)
                    
                    logger.info(f"Loaded configuration from {json_path}")
                    return TrainingConfig(**config_dict)
                    
                except Exception as e:
                    logger.error(f"Error loading configuration from JSON: {e}")
            
            logger.warning("No JSON config found, returning default configuration")
            return TrainingConfig()
        
        try:
            with initialize_config_store(config_path=str(self.config_dir), version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides or [])
                
                # Convert OmegaConf to TrainingConfig
                if isinstance(cfg, DictConfig):
                    config_dict = OmegaConf.to_container(cfg, resolve=True)
                    return TrainingConfig(**config_dict)
                else:
                    return cfg
                    
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Falling back to default configuration")
            return TrainingConfig()
    
    def save_config(self, config: TrainingConfig, filename: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            filename: Output filename
        """
        output_path = self.config_dir / filename
        
        if not HYDRA_AVAILABLE:
            # Fallback to JSON serialization when Hydra not available
            import json
            import dataclasses
            
            logger.warning("Hydra not available, saving configuration as JSON instead")
            try:
                # Convert dataclass to dict
                config_dict = dataclasses.asdict(config)
                
                # Handle enum values
                def serialize_enums(obj):
                    if hasattr(obj, 'value'):
                        return obj.value
                    elif isinstance(obj, dict):
                        return {k: serialize_enums(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_enums(item) for item in obj]
                    return obj
                
                serializable_dict = serialize_enums(config_dict)
                
                # Save as JSON
                json_path = output_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(serializable_dict, f, indent=2)
                
                logger.info(f"Configuration saved to {json_path}")
                return
                
            except Exception as e:
                logger.error(f"Error saving configuration as JSON: {e}")
                return
        
        try:
            # Convert to OmegaConf
            omega_config = OmegaConf.structured(config)
            
            # Save to file
            with open(output_path, 'w') as f:
                OmegaConf.save(omega_config, f)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self, config: TrainingConfig) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            # Basic validation through dataclass post_init
            # This will raise exceptions for critical issues
            pass  # post_init methods already called
            
        except Exception as e:
            issues.append(f"Critical validation error: {e}")
        
        # Additional validation checks
        
        # Check device compatibility
        if config.accelerator == "gpu" and not torch.cuda.is_available():
            issues.append("GPU accelerator requested but CUDA not available")
        
        # Check memory requirements
        estimated_memory = self._estimate_memory_usage(config)
        if estimated_memory > self._get_available_memory():
            issues.append(f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds available memory")
        
        # Check data consistency
        if config.data.batch_size > 1000:
            issues.append("Very large batch size may cause memory issues")
        
        # Check model complexity
        total_params = self._estimate_model_parameters(config.model)
        if total_params > 100e6:  # 100M parameters
            issues.append(f"Very large model ({total_params/1e6:.1f}M parameters) may be slow to train")
        
        return issues
    
    def _deserialize_enums(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enum string values back to enum objects."""
        # Mapping of enum classes by field names/paths
        enum_mappings = {
            'model.model_type': ModelType,
            'optimizer.type': OptimzerType,
            'scheduler.type': SchedulerType,
            'early_stopping.mode': MonitorMode,
            'checkpoint.mode': MonitorMode,
            'accelerator': DeviceType,
        }
        
        def convert_nested(obj, path=""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if new_path in enum_mappings and isinstance(value, str):
                        # Convert string to enum
                        enum_class = enum_mappings[new_path]
                        try:
                            result[key] = enum_class(value)
                        except ValueError:
                            logger.warning(f"Invalid enum value '{value}' for {new_path}")
                            result[key] = value
                    else:
                        result[key] = convert_nested(value, new_path)
                return result
            elif isinstance(obj, list):
                return [convert_nested(item, path) for item in obj]
            else:
                return obj
        
        return convert_nested(config_dict)
    
    def _estimate_memory_usage(self, config: TrainingConfig) -> float:
        """Estimate memory usage in GB."""
        # Rough estimation based on model size and batch size
        model_params = self._estimate_model_parameters(config.model)
        batch_size = config.data.batch_size
        
        # Rough estimates (in GB)
        model_memory = model_params * 4 / 1e9  # 4 bytes per parameter
        batch_memory = batch_size * config.model.hidden_dim * 4 / 1e9
        gradient_memory = model_memory  # Gradients roughly same size as model
        
        return (model_memory + batch_memory + gradient_memory) * 2  # 2x safety factor
    
    def _estimate_model_parameters(self, model_config: ModelConfig) -> int:
        """Estimate number of model parameters."""
        # Very rough estimation
        if model_config.model_type == ModelType.DJMGNN:
            return model_config.hidden_dim ** 2 * model_config.num_layers * 4
        elif model_config.model_type == ModelType.HMGNN:
            return model_config.hidden_dim ** 2 * model_config.num_layers * 6
        else:
            return model_config.hidden_dim ** 2 * model_config.num_layers * 5
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / 1e9
        except ImportError:
            return 8.0  # Assume 8GB if psutil not available
    
    def create_template_configs(self) -> None:
        """Create template configuration files for common use cases."""
        templates = {
            "molecular_property_prediction.yaml": TrainingConfig(
                experiment_name="molecular_property_prediction",
                model=ModelConfig(
                    model_type=ModelType.DJMGNN,
                    hidden_dim=160,
                    num_layers=4,
                    dropout=0.2
                ),
                data=DataConfig(
                    dataset_name="qm9",
                    batch_size=32,
                    use_3d_coordinates=True
                ),
                early_stopping=EarlyStoppingConfig(
                    patience=15,
                    min_delta=1e-4,
                    warmup_epochs=5
                ),
                monitoring=MonitoringConfig(
                    primary_metrics=["train_loss", "val_loss", "val_mae"],
                    enable_alerts=True,
                    log_to_wandb=True
                )
            ),
            
            "joint_training.yaml": TrainingConfig(
                experiment_name="joint_mgnn_training",
                model=ModelConfig(
                    model_type=ModelType.JOINT_MGNN,
                    hidden_dim=128,
                    num_layers=6
                ),
                optimizer=OptimizerConfig(
                    type=OptimzerType.ADAMW,
                    lr=1e-3,
                    weight_decay=1e-4
                ),
                scheduler=SchedulerConfig(
                    type=SchedulerType.COSINE,
                    T_max=100
                ),
                early_stopping=EarlyStoppingConfig(
                    patience=20,
                    reduce_lr_on_plateau=True
                )
            ),
            
            "debug.yaml": TrainingConfig(
                experiment_name="debug_training",
                max_epochs=5,
                debug=True,
                data=DataConfig(batch_size=8),
                early_stopping=EarlyStoppingConfig(enabled=False),
                monitoring=MonitoringConfig(
                    log_every_n_steps=1,
                    enable_visualization=False
                )
            ),
            
            "production.yaml": TrainingConfig(
                experiment_name="production_training",
                max_epochs=200,
                model=ModelConfig(
                    hidden_dim=256,
                    num_layers=8,
                    dropout=0.1
                ),
                data=DataConfig(
                    batch_size=64,
                    num_workers=8
                ),
                optimizer=OptimizerConfig(
                    type=OptimzerType.ADAMW,
                    lr=5e-4,
                    weight_decay=1e-5
                ),
                early_stopping=EarlyStoppingConfig(
                    patience=25,
                    statistical_test=True,
                    detect_divergence=True
                ),
                checkpoint=CheckpointConfig(
                    save_top_k=5,
                    save_last=True
                ),
                monitoring=MonitoringConfig(
                    enable_alerts=True,
                    auto_export=True,
                    log_to_wandb=True,
                    log_to_tensorboard=True
                )
            )
        }
        
        for filename, config in templates.items():
            self.save_config(config, filename)
        
        logger.info(f"Created {len(templates)} template configurations in {self.config_dir}")


# Utility functions for common configuration patterns
def create_molecular_config(
    dataset_name: str = "qm9",
    model_type: str = "djmgnn",
    hidden_dim: int = 128,
    batch_size: int = 32,
    max_epochs: int = 100,
    **kwargs
) -> TrainingConfig:
    """
    Create configuration for molecular property prediction.
    
    Args:
        dataset_name: Name of the molecular dataset
        model_type: Type of model to use
        hidden_dim: Hidden dimension size
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured TrainingConfig instance
    """
    config = TrainingConfig(
        experiment_name=f"{model_type}_{dataset_name}",
        max_epochs=max_epochs,
        model=ModelConfig(
            model_type=ModelType(model_type.lower()),
            hidden_dim=hidden_dim
        ),
        data=DataConfig(
            dataset_name=dataset_name,
            batch_size=batch_size,
            use_3d_coordinates=True
        ),
        early_stopping=EarlyStoppingConfig(
            patience=15,
            warmup_epochs=5
        ),
        monitoring=MonitoringConfig(
            primary_metrics=["train_loss", "val_loss", "val_mae"],
            enable_alerts=True
        )
    )
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_debug_config(max_epochs: int = 3, batch_size: int = 4) -> TrainingConfig:
    """
    Create configuration for debugging and testing.
    
    Args:
        max_epochs: Maximum epochs for debug run
        batch_size: Small batch size for debugging
        
    Returns:
        Debug TrainingConfig instance
    """
    return TrainingConfig(
        experiment_name="debug",
        max_epochs=max_epochs,
        debug=True,
        data=DataConfig(
            batch_size=batch_size,
            num_workers=0
        ),
        early_stopping=EarlyStoppingConfig(enabled=False),
        monitoring=MonitoringConfig(
            log_every_n_steps=1,
            enable_visualization=False,
            log_to_wandb=False
        ),
        checkpoint=CheckpointConfig(enabled=False)
    )


# Example usage and testing
if __name__ == "__main__":
    # Create configuration manager
    config_manager = ConfigManager("configs")
    
    # Create and validate a molecular prediction configuration
    config = create_molecular_config(
        dataset_name="qm9",
        model_type="djmgnn",
        hidden_dim=160,
        batch_size=32,
        max_epochs=100
    )
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    # Save configuration
    config_manager.save_config(config, "example_config.yaml")
    
    # Create template configurations
    config_manager.create_template_configs()
    
    print("Configuration system ready for use!")
    print(f"Configuration directory: {config_manager.config_dir}")