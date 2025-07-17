"""
examples/preprocessing_examples/preprocess_example.py

Command-line tool for preprocessing molecular structures into graph representations suitable for machine learning with comprehensive feature statistics generation.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from moml.core import create_graph_processor

# Constants
DEFAULT_FILE_PATTERN = '*.mol'
DEFAULT_CONFIG = {
    'atom_features': {
        'use_symbol': True,
        'use_degree': True,
        'use_hybridization': True,
        'use_aromatic': True,
        'use_in_ring': True,
        'use_chirality': True,
        'use_formal_charge': True,
        'use_num_h': True,
        'use_radical_electrons': True,
        'use_3d_coords': True,
    },
    'bond_features': {
        'use_bond_type': True,
        'use_conjugated': True,
        'use_in_ring': True,
        'use_stereo': True,
    },
}
GRAPH_FILE_SUFFIX = '_graph.pt'
STATS_FILENAME = 'dataset_statistics.csv'
CONFIG_FILENAME = 'graph_config.json'


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to configuration JSON file.

    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary if file exists and
            is valid, None otherwise.

    Raises:
        json.JSONDecodeError: If JSON file is malformed.
        FileNotFoundError: If configuration file doesn't exist.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        output_path (str): Path where configuration will be saved.

    Raises:
        OSError: If file cannot be written.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def load_labels_from_csv(labels_file: str, input_dir: str) -> Dict[str, float]:
    """
    Load labels from CSV file and create mapping to full file paths.

    Args:
        labels_file (str): Path to CSV file containing labels.
        input_dir (str): Directory containing molecule files.

    Returns:
        Dict[str, float]: Mapping from full file paths to label values.

    Raises:
        ValueError: If CSV file has invalid format.
        FileNotFoundError: If labels file doesn't exist.
    """
    labels = {}
    
    try:
        labels_df = pd.read_csv(labels_file)
        
        if 'filename' not in labels_df.columns or len(labels_df.columns) < 2:
            raise ValueError(
                f'Labels file {labels_file} has invalid format. '
                'Expected columns: filename, label'
            )
        
        # Determine label column (first non-filename column)
        label_col = [col for col in labels_df.columns if col != 'filename'][0]
        
        for _, row in labels_df.iterrows():
            filename = str(row['filename'])  # Ensure string type
            label = float(row[label_col])
            
            # Store full path as key
            full_path = os.path.join(input_dir, filename)
            if os.path.exists(full_path):
                labels[full_path] = label
        
        print(f'Loaded {len(labels)} labels from {labels_file}')
        
    except Exception as e:
        print(f'Error loading labels file: {e}')
        raise
    
    return labels


def process_molecular_files(
    mol_files: List[str],
    graph_processor: Any,
    labels: Dict[str, float],
    output_dir: str
) -> Dict[str, List]:
    """
    Process molecular files into graph representations.

    Args:
        mol_files (List[str]): List of molecule file paths to process.
        graph_processor: Graph processor instance for molecule-to-graph conversion.
        labels (Dict[str, float]): Mapping from file paths to labels.
        output_dir (str): Directory to save processed graphs.

    Returns:
        Dict[str, List]: Dictionary containing feature statistics including
            node counts, edge counts, and feature dimensions.

    Raises:
        ValueError: If graph processing fails for critical files.
    """
    feature_stats = {
        'num_nodes': [],
        'num_edges': [],
        'node_feature_dim': None,
        'edge_feature_dim': None
    }
    
    processed_files = 0
    
    print('Processing molecular files...')
    for file_path in tqdm(mol_files):
        try:
            # Process molecule into graph representation
            graph = graph_processor.file_to_graph(file_path)
            
            # Update feature statistics
            feature_stats['num_nodes'].append(graph.num_nodes)
            feature_stats['num_edges'].append(graph.edge_index.shape[1] // 2)
            
            if feature_stats['node_feature_dim'] is None:
                feature_stats['node_feature_dim'] = graph.x.shape[1]
            
            if feature_stats['edge_feature_dim'] is None:
                feature_stats['edge_feature_dim'] = graph.edge_attr.shape[1]
            
            # Add label if available
            if file_path in labels:
                graph.y = torch.tensor([labels[file_path]], dtype=torch.float)
            
            # Save processed graph
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, base_name + GRAPH_FILE_SUFFIX)
            torch.save(graph, output_file)
            
            processed_files += 1
            
        except Exception as e:
            print(f'Error processing file {file_path}: {e}')
            continue
    
    print(f'Successfully processed {processed_files} files')
    return feature_stats


def calculate_and_save_statistics(
    feature_stats: Dict[str, List],
    processed_files: int,
    output_dir: str
) -> None:
    """
    Calculate dataset statistics and save to CSV file.

    Args:
        feature_stats (Dict[str, List]): Feature statistics from processing.
        processed_files (int): Number of successfully processed files.
        output_dir (str): Directory to save statistics file.

    Raises:
        OSError: If statistics file cannot be written.
    """
    if processed_files == 0:
        print('No files were processed. Skipping dataset statistics calculation.')
        return
    
    stats_df = pd.DataFrame({
        'statistic': [
            'num_files',
            'avg_nodes',
            'min_nodes',
            'max_nodes',
            'std_nodes',
            'avg_edges',
            'min_edges',
            'max_edges',
            'std_edges',
            'node_feature_dim',
            'edge_feature_dim',
        ],
        'value': [
            processed_files,
            np.mean(feature_stats['num_nodes']),
            np.min(feature_stats['num_nodes']),
            np.max(feature_stats['num_nodes']),
            np.std(feature_stats['num_nodes']),
            np.mean(feature_stats['num_edges']),
            np.min(feature_stats['num_edges']),
            np.max(feature_stats['num_edges']),
            np.std(feature_stats['num_edges']),
            feature_stats['node_feature_dim'],
            feature_stats['edge_feature_dim'],
        ],
    })
    
    stats_file = os.path.join(output_dir, STATS_FILENAME)
    stats_df.to_csv(stats_file, index=False)
    print(f'Dataset statistics saved to {stats_file}')
    
    # Print summary statistics
    print('\nDataset Statistics:')
    print(f'Number of processed files: {processed_files}')
    print(
        f'Nodes: avg={np.mean(feature_stats["num_nodes"]):.1f}, '
        f'min={np.min(feature_stats["num_nodes"])}, '
        f'max={np.max(feature_stats["num_nodes"])}'
    )
    print(
        f'Edges: avg={np.mean(feature_stats["num_edges"]):.1f}, '
        f'min={np.min(feature_stats["num_edges"])}, '
        f'max={np.max(feature_stats["num_edges"])}'
    )
    print(f'Node feature dimension: {feature_stats["node_feature_dim"]}')
    print(f'Edge feature dimension: {feature_stats["edge_feature_dim"]}')


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for molecular graph preprocessing.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - input_dir: Directory with input molecule files
            - output_dir: Directory for processed graphs and statistics
            - labels_file: Optional CSV file with labels
            - file_pattern: Pattern for molecular files
            - config_path: Optional configuration file path
            - save_config: Flag to save configuration with statistics
    """
    parser = argparse.ArgumentParser(
        description='Molecular Graph Preprocessing Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input_dir ./molecules --output_dir ./processed
  %(prog)s --input_dir ./data --output_dir ./graphs --labels_file labels.csv
  %(prog)s --input_dir ./sdf_files --output_dir ./processed --file_pattern "*.sdf"
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input molecule files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save processed graphs and statistics'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        help='Optional CSV file with labels (filename, label)'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default=DEFAULT_FILE_PATTERN,
        help='Pattern for molecular files (e.g., "*.mol", "*.sdf")'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--save_config',
        action='store_true',
        help='Save configuration file with feature statistics'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for molecular graph preprocessing.

    Demonstrates how to:
    1. Process a directory of molecular files into graph representations
    2. Save processed graphs to disk for faster training
    3. Generate feature statistics for model configuration

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        args = parse_args()
        
        # Load configuration or use default
        config = DEFAULT_CONFIG.copy()
        if args.config_path and os.path.exists(args.config_path):
            loaded_config = load_config(args.config_path)
            if loaded_config:
                config.update(loaded_config)
                print(f'Loaded configuration from {args.config_path}')
        else:
            print('Using default configuration.')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create graph processor
        graph_processor = create_graph_processor({
            'atom_features': config['atom_features'],
            'bond_features': config['bond_features']
        })
        
        # Find all molecular files
        mol_files = glob.glob(os.path.join(args.input_dir, args.file_pattern))
        
        if not mol_files:
            print(
                f'No molecular files found in {args.input_dir} '
                f'matching pattern {args.file_pattern}'
            )
            return 1
        
        print(f'Found {len(mol_files)} molecular files for preprocessing')
        
        # Load labels if provided
        labels = {}
        if args.labels_file and os.path.exists(args.labels_file):
            labels = load_labels_from_csv(args.labels_file, args.input_dir)
        
        # Process molecular files and collect statistics
        feature_stats = process_molecular_files(
            mol_files, graph_processor, labels, args.output_dir
        )
        
        processed_files = len([
            f for f in feature_stats['num_nodes'] if f is not None
        ])
        
        # Calculate and save statistics
        calculate_and_save_statistics(feature_stats, processed_files, args.output_dir)
        
        # Update and save configuration file
        if args.save_config and processed_files > 0:
            # Create new config with statistics for saving
            save_config_dict: Dict[str, Any] = dict(config)
            save_config_dict['node_feature_dim'] = feature_stats['node_feature_dim']
            save_config_dict['edge_feature_dim'] = feature_stats['edge_feature_dim']
            
            config_file = os.path.join(args.output_dir, CONFIG_FILENAME)
            save_config(save_config_dict, config_file)
            print(f'Configuration with dataset statistics saved to {config_file}')
        
        return 0
        
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f'Error: {e}')
        return 1
    except KeyboardInterrupt:
        print('\nProcessing cancelled by user')
        return 1


if __name__ == '__main__':
    exit(main())
