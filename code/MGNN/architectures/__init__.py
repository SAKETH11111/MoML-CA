"""
MGNN Architecture Modules

This module contains the core neural network architectures for
molecular graph neural networks specialized for PFAS analysis.
"""

from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder

from code.MGNN.architectures.graph_coarsening import (
    GraphCoarsener,
    FunctionalGroupIdentifier
)

__all__ = [
    'MolecularGraphBuilder',
    'GraphCoarsener',
    'FunctionalGroupIdentifier'
] 