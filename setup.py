#!/usr/bin/env python3
"""
Setup script for MoML-CA: Molecular Modeling and Machine Learning for Contaminant Analysis

This package provides a hybrid computational approach to analyze PFAS contaminants,
combining quantum mechanical calculations with graph neural networks.
"""

from setuptools import setup, find_packages
import os

# Get the long description from the README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="moml-ca",
    version="0.1.0",
    description="Molecular Modeling and Machine Learning for Contaminant Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saketh/MoML-CA_PFAS",
    author="Saketh",
    author_email="saketh@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="computational chemistry, machine learning, PFAS, contaminants, molecular graphs",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "OpenMM>=7.5.0",
        "mdtraj>=1.9.5",
        "pdb-tools",
        "rdkit>=2022.03.2",
        "deepchem>=2.5.0",
        "mordred>=1.2.0",
        "networkx>=2.6.0",
        "plotly>=5.3.0",
        "pyyaml>=6.0",
        "h5py>=3.6.0",
        "luigi>=3.0.0",
        "dask>=2022.1.0",
        "distributed>=2022.1.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "pytest>=6.2.5",
        "black>=21.12b0",
        "flake8>=4.0.0",
        "isort>=5.10.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.12b0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moml-ca=moml.pipeline.orchestration.pfas_pipeline_orchestrator:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/saketh/MoML-CA_PFAS/issues",
        "Source": "https://github.com/saketh/MoML-CA_PFAS",
    },
) 