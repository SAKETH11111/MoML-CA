#!/usr/bin/env python
"""
Preprocessing Example for MoML-CA.

This script demonstrates how to preprocess molecular structures into graph
representations suitable for machine learning.
"""

import os
import sys
import argparse
import glob
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import our modules
from moml.core.molecular_graph import create_graph_processor


def load_config(config_path):
    """Load configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)


def main():
    """
    Example script for preprocessing molecular data for graph neural networks.

    This script demonstrates how to:
    1. Process a directory of molecular files into graph representations
    2. Save processed graphs to disk for faster training
    3. Generate feature statistics for model configuration
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Molecular Graph Preprocessing Example")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input molecule files")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save processed graphs and statistics"
    )
    parser.add_argument("--labels_file", type=str, help="Optional CSV file with labels (filename, label)")
    parser.add_argument(
        "--file_pattern", type=str, default="*.mol", help='Pattern for molecular files (e.g., "*.mol", "*.sdf")'
    )
    parser.add_argument("--config_path", type=str, help="Path to configuration file (optional)")
    parser.add_argument("--save_config", action="store_true", help="Save configuration file with feature statistics")

    args = parser.parse_args()

    # Load configuration if provided or create default
    config = None
    if args.config_path and os.path.exists(args.config_path):
        config = load_config(args.config_path)
        print(f"Loaded configuration from {args.config_path}")
    else:
        print("Using default configuration.")
        config = {
            "atom_features": {
                "use_symbol": True,
                "use_degree": True,
                "use_hybridization": True,
                "use_aromatic": True,
                "use_in_ring": True,
                "use_chirality": True,
                "use_formal_charge": True,
                "use_num_h": True,
                "use_radical_electrons": True,
                "use_3d_coords": True,
            },
            "bond_features": {"use_bond_type": True, "use_conjugated": True, "use_in_ring": True, "use_stereo": True},
        }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create graph processor
    graph_processor = create_graph_processor(
        {"atom_features": config["atom_features"], "bond_features": config["bond_features"]}
    )

    # Find all molecular files
    mol_files = glob.glob(os.path.join(args.input_dir, args.file_pattern))

    if not mol_files:
        print(f"No molecular files found in {args.input_dir} matching pattern {args.file_pattern}")
        sys.exit(1)

    print(f"Found {len(mol_files)} molecular files for preprocessing")

    # Load labels if provided
    labels = {}
    if args.labels_file and os.path.exists(args.labels_file):
        try:
            labels_df = pd.read_csv(args.labels_file)

            if "filename" in labels_df.columns and len(labels_df.columns) >= 2:
                # Determine label column (assuming first non-filename column is the label)
                label_col = [col for col in labels_df.columns if col != "filename"][0]

                for _, row in labels_df.iterrows():
                    filename = row["filename"]
                    label = row[label_col]

                    # Store full path as key
                    full_path = os.path.join(args.input_dir, filename)
                    if os.path.exists(full_path):
                        labels[full_path] = label

                print(f"Loaded {len(labels)} labels from {args.labels_file}")
            else:
                print(f"Warning: Labels file {args.labels_file} has invalid format. Expected columns: filename, label")

        except Exception as e:
            print(f"Error loading labels file")

    # Process molecular files and collect statistics
    feature_stats = {"num_nodes": [], "num_edges": [], "node_feature_dim": None, "edge_feature_dim": None}

    processed_files = 0

    print("Processing molecular files...")
    for file_path in tqdm(mol_files):
        try:
            # Process molecule into graph representation
            graph = graph_processor.file_to_graph(file_path)

            # Update feature statistics
            feature_stats["num_nodes"].append(graph["num_nodes"])
            feature_stats["num_edges"].append(graph["edge_index"].shape[1] // 2)  # Divide by 2 for undirected graphs

            if feature_stats["node_feature_dim"] is None:
                feature_stats["node_feature_dim"] = graph["x"].shape[1]

            if feature_stats["edge_feature_dim"] is None:
                feature_stats["edge_feature_dim"] = graph["edge_attr"].shape[1]

            # Add label if available
            if file_path in labels:
                graph["label"] = torch.tensor([labels[file_path]], dtype=torch.float)

            # Save processed graph
            output_file = os.path.join(args.output_dir, os.path.basename(file_path).split(".")[0] + "_graph.pt")
            torch.save(graph, output_file)

            processed_files += 1

        except Exception as e:
            print(f"Error processing file {file_path}")

    print(f"Successfully processed {processed_files} files")

    # Calculate and save statistics
    if processed_files > 0:
        stats_df = pd.DataFrame(
            {
                "statistic": [
                    "num_files",
                    "avg_nodes",
                    "min_nodes",
                    "max_nodes",
                    "std_nodes",
                    "avg_edges",
                    "min_edges",
                    "max_edges",
                    "std_edges",
                    "node_feature_dim",
                    "edge_feature_dim",
                ],
                "value": [
                    processed_files,
                    np.mean(feature_stats["num_nodes"]),
                    np.min(feature_stats["num_nodes"]),
                    np.max(feature_stats["num_nodes"]),
                    np.std(feature_stats["num_nodes"]),
                    np.mean(feature_stats["num_edges"]),
                    np.min(feature_stats["num_edges"]),
                    np.max(feature_stats["num_edges"]),
                    np.std(feature_stats["num_edges"]),
                    feature_stats["node_feature_dim"],
                    feature_stats["edge_feature_dim"],
                ],
            }
        )

        stats_file = os.path.join(args.output_dir, "dataset_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"Dataset statistics saved to {stats_file}")

        # Print summary statistics
        print("\nDataset Statistics:")
        print(f"Number of processed files: {processed_files}")
        print(
            f"Nodes: avg={np.mean(feature_stats['num_nodes']):.1f}, "
            f"min={np.min(feature_stats['num_nodes'])}, "
            f"max={np.max(feature_stats['num_nodes'])}"
        )
        print(
            f"Edges: avg={np.mean(feature_stats['num_edges']):.1f}, "
            f"min={np.min(feature_stats['num_edges'])}, "
            f"max={np.max(feature_stats['num_edges'])}"
        )
        print(f"Node feature dimension: {feature_stats['node_feature_dim']}")
        print(f"Edge feature dimension: {feature_stats['edge_feature_dim']}")

        # Update and save configuration file
        if args.save_config:
            # Update config with dataset statistics
            config["node_features"] = feature_stats["node_feature_dim"]
            config["edge_features"] = feature_stats["edge_feature_dim"]

            # Save configuration
            config_file = os.path.join(args.output_dir, "graph_config.json")
            save_config(config, config_file)
            print(f"Configuration with dataset statistics saved to {config_file}")
    else:
        print("No files were processed. Skipping dataset statistics calculation.")

if __name__ == "__main__":
    main()
