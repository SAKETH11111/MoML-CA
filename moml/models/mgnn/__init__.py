"""
MoML MGNN Models Package

This package provides molecular graph neural network models for 
molecular property prediction and feature learning.
"""

# Import from djmgnn.py
from moml.models.mgnn.djmgnn import (
    GraphConvLayer,
    DenseGNNBlock,
    JKAggregator,
    DJMGNN
)

__all__ = [
    # Graph Neural Network Layers
    "GraphConvLayer", # Based on NNConv
    "DenseGNNBlock",
    "JKAggregator",
    
    # Models
    "DJMGNN"
]
