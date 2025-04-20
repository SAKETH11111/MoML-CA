"""
MoML: Molecular machine learning framework for PFAS analysis

Public API:
- __version__: package version
- core: core graph processing and descriptor facilities
- data: data loading, splitting, and molecule‑to‑graph utilities
- models: graph neural network layers, trainers, and predictors
- pipeline: end‑to‑end orchestrators for data⇒QM⇒graph pipelines
- simulation: quantum/MD simulation helpers and parsers
- utils: miscellaneous lightweight helper functions
"""

__version__ = '0.1.0'

# Subpackages
from . import core
from . import data
from . import models
from . import pipeline
from . import simulation
from . import utils

__all__ = [
    '__version__',
    'core',
    'data',
    'models',
    'pipeline',
    'simulation',
    'utils',
]
