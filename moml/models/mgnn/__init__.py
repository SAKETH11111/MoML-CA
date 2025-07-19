"""
moml/models/mgnn/__init__.py

MoML MGNN Models Package

This package provides molecular graph neural network models for molecular 
property prediction and feature learning. It includes implementations of 
dense GNN architectures with jumping knowledge aggregation and hierarchical 
multi-scale models.

Main Components:
    - DJMGNN: Dense jumping knowledge molecular graph neural network
    - HMGNN: Hierarchical molecular graph neural network for multi-scale learning
    - Core Layers: GraphConvLayer, DenseGNNBlock, JKAggregator building blocks
"""

# Import core model components
from moml.models.mgnn.djmgnn import (
    GraphConvLayer,
    DenseGNNBlock, 
    JKAggregator,
    DJMGNN,
)

# Define public API
__all__ = [
    # Core GNN building blocks
    "GraphConvLayer",  # Graph convolution layer based on NNConv
    "DenseGNNBlock",   # Dense GNN block with skip connections
    "JKAggregator",    # Jumping knowledge aggregation layer
    # Complete models
    "DJMGNN",          # Dense jumping knowledge molecular GNN
]
