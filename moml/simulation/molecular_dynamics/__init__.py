"""
moml/simulation/molecular_dynamics/__init__.py

MoML-CA Molecular Dynamics Engine.

This package provides a production-grade Molecular Dynamics engine that integrates with
the MoML-CA simulation stack. It converts graph neural network generated force-field 
parameters into OpenMM simulations and produces time-series data suitable for analysis
and machine learning applications.

The module provides functionality for:
- System building and force field application
- Energy minimization and equilibration protocols
- Production MD simulations with monitoring
- Ensemble simulations (replicas, constant pH)

Classes:
    MDRunner: Main simulation runner for molecular dynamics
    
Functions:
    build_system: Convenience function to build OpenMM systems
    run_ensemble: Run ensemble simulations with multiple replicas
"""

from .builder.system_builder import build_system
from .runner import MDRunner
from .ensemble import run_ensemble

__all__ = ["build_system", "MDRunner", "run_ensemble"]