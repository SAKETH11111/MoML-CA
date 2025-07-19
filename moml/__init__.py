"""
moml/__init__.py

This package provides a comprehensive framework for applying graph neural networks
and other machine learning techniques to molecular data, with a special focus on
the analysis and property prediction of Per- and polyfluoroalkyl substances (PFAS).

The main subpackages include:
- `core`: Core functionalities for graph processing and feature extraction.
- `data`: Tools for data loading, processing, and dataset creation.
- `models`: Implementations of graph neural network architectures.
- `pipeline`: Orchestration for end-to-end processing and modeling workflows.
- `simulation`: Interfaces for quantum mechanics and molecular dynamics simulations.
- `utils`: General-purpose utilities for data handling and validation.
"""

import importlib
import warnings

__version__ = "0.1.0"

_SUBPACKAGES = [
    "core",
    "data",
    "models",
    "pipeline",
    "simulation",
    "utils",
]

for _pkg in _SUBPACKAGES:
    try:
        globals()[_pkg] = importlib.import_module(f".{_pkg}", __name__)
    except ImportError as e:
        warnings.warn(f"Could not import the '{_pkg}' subpackage: {e}")
        globals()[_pkg] = None

__all__ = ["__version__"] + _SUBPACKAGES
