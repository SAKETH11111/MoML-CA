"""
moml/models/mgnn/__init__.py

MoML MGNN Models Package

This package provides molecular graph neural network models for molecular 
property prediction and feature learning. It includes implementations of 
dense GNN architectures with jumping knowledge aggregation, hierarchical 
multi-scale models, and joint training frameworks.

Main Components:
    - DJMGNN: Dense jumping knowledge molecular graph neural network
    - HMGNN: Hierarchical molecular graph neural network for multi-scale learning
    - JointMGNN: Joint framework combining DJMGNN and HMGNN
    - Core Layers: GraphConvLayer, DenseGNNBlock, JKAggregator building blocks
    - Training: Joint training infrastructure and strategies
"""

# Import core model components
from moml.models.mgnn.djmgnn import (
    GraphConvLayer,
    DenseGNNBlock, 
    JKAggregator,
    DJMGNN,
)

from moml.models.mgnn.hmgnn import (
    HMGNN,
    CrossScaleAttentionMH,
    create_hierarchical_mgnn,
)

from moml.models.mgnn.joint_mgnn import (
    JointMGNN,
    CrossModelFusion,
    MultiTaskHead,
    create_joint_mgnn,
)

# Import training utilities (optional, may require extra dependencies)
try:
    from moml.models.mgnn.training import (
        MGNNTrainer,
        create_trainer,
        JointMGNNTrainer,
        AlternatingTrainingStrategy,
        create_joint_trainer,
        EarlyStopping,
        ModelCheckpoint,
        LearningRateScheduler,
    )
except ModuleNotFoundError as e:
    # Gracefully degrade if optional dependencies like matplotlib are missing
    import logging
    logging.getLogger(__name__).warning(
        "Optional MGNN training utilities could not be imported (%s)."
        " Training-related functionality will be unavailable.", e
    )
    MGNNTrainer = create_trainer = JointMGNNTrainer = AlternatingTrainingStrategy = create_joint_trainer = None
    EarlyStopping = ModelCheckpoint = LearningRateScheduler = None

# Import evaluation utilities (optional)
try:
    from moml.models.mgnn.evaluation import MGNNPredictor  # noqa: F401
except ModuleNotFoundError as e:
    import logging
    logging.getLogger(__name__).warning(
        "Optional MGNN evaluation utilities could not be imported (%s).", e
    )
    MGNNPredictor = None

# Model registry for factory functions
MODEL_REGISTRY = {
    'djmgnn': DJMGNN,
    'hmgnn': HMGNN,
    'joint_mgnn': JointMGNN,
}

# Define public API
__all__ = [
    # Core GNN building blocks
    "GraphConvLayer",  # Graph convolution layer based on NNConv
    "DenseGNNBlock",   # Dense GNN block with skip connections
    "JKAggregator",    # Jumping knowledge aggregation layer
    # Complete models
    "DJMGNN",          # Dense jumping knowledge molecular GNN
    "HMGNN",           # Hierarchical molecular GNN
    "JointMGNN",       # Joint DJMGNN+HMGNN framework
    # HMGNN components
    "CrossScaleAttentionMH",
    "create_hierarchical_mgnn",
    # JointMGNN components
    "CrossModelFusion",
    "MultiTaskHead",
    "create_joint_mgnn",
    # Training utilities
    "MGNNTrainer",
    "create_trainer",
    "JointMGNNTrainer",
    "AlternatingTrainingStrategy",
    "create_joint_trainer",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    # Evaluation utilities
    "MGNNPredictor",
    # Registry
    "MODEL_REGISTRY",
]
