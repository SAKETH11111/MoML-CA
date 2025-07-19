"""
moml/pipeline/__init__.py

Pipeline orchestration module for MoML-CA.

This module provides the public API for molecular analysis pipeline orchestration,
supporting both general molecular modeling workflows and PFAS-specific analysis
pipelines with enhanced feature extraction and contaminant analysis capabilities.

The orchestration system coordinates complex multi-stage workflows including:
- Data preprocessing and validation
- Quantum mechanical calculations
- Molecular graph generation
- Model training and evaluation

Classes:
    MOMLPipelineOrchestrator: Base orchestrator for general molecular analysis
    PFASPipelineOrchestrator: Specialized orchestrator for PFAS contaminant analysis
"""

from .pipeline_orchestrator import (
    MOMLPipelineOrchestrator,
    PFASPipelineOrchestrator,
    main,
)

__all__ = [
    "MOMLPipelineOrchestrator",
    "PFASPipelineOrchestrator",
    "main",
]
