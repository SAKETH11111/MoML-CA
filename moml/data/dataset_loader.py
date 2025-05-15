"""
Dataset Loading Utilities

This module provides functions for loading datasets from files and directories.
"""

import os
import glob
import pandas as pd
from typing import Dict, List, Optional, Any

from moml.data.datasets import MolecularGraphDataset, HierarchicalGraphDataset


def load_dataset(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = "*.mol",
    config: Optional[Dict[str, Any]] = None,
) -> MolecularGraphDataset:
    """
    Load a dataset from a directory of molecule files.

    Args:
        data_dir: Directory containing molecule files
        labels_file: Optional path to file containing labels
        file_pattern: Pattern to match molecule files
        config: Configuration for graph processing

    Returns:
        MolecularGraphDataset instance
    """
    # Find all molecule files
    mol_files = []
    for pattern in file_pattern.split(","):
        mol_files.extend(glob.glob(os.path.join(data_dir, pattern.strip())))

    if not mol_files:
        raise ValueError(f"No files found in {data_dir} matching pattern {file_pattern}")

    # Load labels if provided
    labels = None
    if labels_file and os.path.exists(labels_file):
        labels = {}
        try:
            # Try to load as CSV
            try:
                df = pd.read_csv(labels_file)

                # Determine label column
                if "filename" in df.columns and len(df.columns) >= 2:
                    label_col = [col for col in df.columns if col != "filename"][0]

                    for _, row in df.iterrows():
                        filename = row["filename"]
                        label = row[label_col]

                        # Find full path
                        full_path = os.path.join(data_dir, filename)
                        if os.path.exists(full_path):
                            labels[full_path] = label
            except ImportError:
                # Fallback to simple CSV parsing
                with open(labels_file, "r") as f:
                    # Skip header
                    next(f)
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            file_name = parts[0]
                            label = float(parts[1])

                            # Add full path to labels dictionary
                            file_path = os.path.join(data_dir, file_name)
                            if os.path.exists(file_path):
                                labels[file_path] = label

        except Exception as e:
            print(f"Error loading labels file: {e}")

    # Create dataset
    return MolecularGraphDataset(mol_files, labels, config)


def load_datasets_from_splits(
    data_dir: str,
    labels_file: Optional[str] = None,
    file_pattern: str = "*.mol",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, MolecularGraphDataset]:
    """
    Load datasets from pre-split directories (train/val/test).

    Args:
        data_dir: Root directory containing split directories
        labels_file: Optional path to file containing labels
        file_pattern: Pattern to match molecule files
        config: Configuration for graph processing

    Returns:
        Dictionary of datasets for each split
    """
    datasets = {}

    # Check for split directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Load training dataset (required)
    if os.path.isdir(train_dir):
        datasets["train"] = load_dataset(train_dir, labels_file, file_pattern, config)
    else:
        raise ValueError(f"Training directory not found: {train_dir}")

    # Load validation dataset (optional)
    if os.path.isdir(val_dir):
        datasets["val"] = load_dataset(val_dir, labels_file, file_pattern, config)

    # Load test dataset (optional)
    if os.path.isdir(test_dir):
        datasets["test"] = load_dataset(test_dir, labels_file, file_pattern, config)

    return datasets


def load_hierarchical_dataset(
    data_dir: str,
    levels: List[str] = ["atom", "functional_group", "structural_motif"],
    labels_file: Optional[str] = None,
) -> Dict[str, MolecularGraphDataset]:
    """
    Load hierarchical graph datasets from a directory.

    Args:
        data_dir: Directory containing hierarchical graph files
        levels: List of hierarchy levels to include
        labels_file: Optional path to file containing labels

    Returns:
        Dictionary of datasets for each level
    """

    # Load labels if provided
    labels = None
    if labels_file and os.path.exists(labels_file):
        labels = {}
        try:
            df = pd.read_csv(labels_file)
            id_col = df.columns[0]  # Assume first column is molecule ID
            label_col = df.columns[1]  # Assume second column is label

            for _, row in df.iterrows():
                mol_id = row[id_col]
                label = row[label_col]
                labels[mol_id] = label

        except Exception as e:
            print(f"Error loading labels file: {e}")

    # Create datasets for each level
    datasets = {}
    for level in levels:
        datasets[level] = HierarchicalGraphDataset(
            data_dir=data_dir, labels=labels, levels=[level], transform=None  # Only include this level
        )

    return datasets
