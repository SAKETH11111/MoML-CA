"""
data_loading_example.py

Demonstration script for using PFASDataLoader to load and batch molecular graph data for MGNN training workflows.
"""

import argparse
import os
import sys
from typing import Dict, List, Any, Optional

from moml.data.data_loader import PFASDataLoader

# Constants
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'example_dataset'
)
DEFAULT_ENVIRONMENTAL_FEATURES = ['ph', 'temperature', 'flow_rate']
DEFAULT_LABEL_TYPES = ['force_field_params', 'molecular_properties']
DEFAULT_MOLECULE_IDS = ['molecule1', 'molecule2']
DEFAULT_BATCH_SIZE = 2


def create_loader_config(
    env_features: Optional[List[str]] = None,
    label_types: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Create configuration dictionary for PFASDataLoader.

    Args:
        env_features (Optional[List[str]]): List of environmental feature names.
            Defaults to pH, temperature, and flow rate.
        label_types (Optional[List[str]]): List of label types to load.
            Defaults to force field parameters and molecular properties.

    Returns:
        Dict[str, List[str]]: Configuration dictionary with environmental
            features and label types.
    """
    return {
        'environmental_features': env_features or DEFAULT_ENVIRONMENTAL_FEATURES,
        'label_types': label_types or DEFAULT_LABEL_TYPES,
    }


def demonstrate_single_molecule_loading(
    loader: PFASDataLoader,
    molecule_id: str
) -> bool:
    """
    Demonstrate loading a single molecule by ID and display its properties.

    Args:
        loader (PFASDataLoader): Configured data loader instance.
        molecule_id (str): Identifier of molecule to load.

    Returns:
        bool: True if molecule was loaded successfully, False otherwise.

    Raises:
        Exception: Re-raises any exceptions from the loader for debugging.
    """
    print(f'Attempting to load molecule: {molecule_id}')
    
    try:
        graph, label, env = loader.load_molecule_by_id(molecule_id)
        
        print(f'✓ Successfully loaded {molecule_id}')
        print(f'  Labels: {label}')
        print(f'  Environmental features: {env}')
        print(f'  Number of nodes: {graph.num_nodes}')
        
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            num_edges = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
            print(f'  Number of edges: {num_edges}')
        
        if hasattr(graph, 'x') and graph.x is not None:
            print(f'  Node feature dimension: {graph.x.shape[1]}')
        
        return True
        
    except Exception as e:
        print(f'✗ Failed to load {molecule_id}: {e}')
        return False


def demonstrate_batch_creation(
    loader: PFASDataLoader,
    molecule_ids: List[str],
    batch_size: int
) -> bool:
    """
    Demonstrate creating batches from multiple molecule IDs.

    Args:
        loader (PFASDataLoader): Configured data loader instance.
        molecule_ids (List[str]): List of molecule identifiers for batching.
        batch_size (int): Number of molecules per batch.

    Returns:
        bool: True if batch was created successfully, False otherwise.

    Raises:
        Exception: Re-raises any exceptions from the loader for debugging.
    """
    print(f'Attempting to create batch from molecules: {molecule_ids}')
    
    try:
        batch = loader.get_batch(molecule_ids, batch_size=batch_size)
        
        print(f'✓ Successfully created batch')
        print(f'  Number of graphs in batch: {batch.num_graphs}')
        print(f'  Total nodes in batch: {batch.num_nodes}')
        
        if hasattr(batch, 'batch') and batch.batch is not None:
            print(f'  Batch assignment shape: {batch.batch.shape}')
        
        if hasattr(batch, 'x') and batch.x is not None:
            print(f'  Batched node features shape: {batch.x.shape}')
        
        return True
        
    except Exception as e:
        print(f'✗ Failed to create batch: {e}')
        return False


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data loading demonstration.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - data_dir: Directory containing molecular data
            - molecule_ids: List of molecule IDs to load
            - batch_size: Batch size for demonstration
            - env_features: Environmental features to include
            - label_types: Label types to load
    """
    parser = argparse.ArgumentParser(
        description='Demonstrate PFASDataLoader functionality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --data_dir ./custom_data --molecule_ids mol1 mol2 mol3
  %(prog)s --batch_size 4 --env_features ph temperature
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Directory containing molecular data'
    )
    parser.add_argument(
        '--molecule_ids',
        nargs='+',
        default=DEFAULT_MOLECULE_IDS,
        help='Molecule IDs to load for demonstration'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for batch creation demonstration'
    )
    parser.add_argument(
        '--env_features',
        nargs='*',
        default=DEFAULT_ENVIRONMENTAL_FEATURES,
        help='Environmental features to include'
    )
    parser.add_argument(
        '--label_types',
        nargs='*',
        default=DEFAULT_LABEL_TYPES,
        help='Label types to load'
    )
    
    return parser.parse_args()


def validate_data_directory(data_dir: str) -> bool:
    """
    Validate that the data directory exists and is accessible.

    Args:
        data_dir (str): Path to data directory.

    Returns:
        bool: True if directory is valid, False otherwise.
    """
    if not os.path.exists(data_dir):
        print(f'Error: Data directory does not exist: {data_dir}')
        return False
    
    if not os.path.isdir(data_dir):
        print(f'Error: Path is not a directory: {data_dir}')
        return False
    
    if not os.access(data_dir, os.R_OK):
        print(f'Error: Data directory is not readable: {data_dir}')
        return False
    
    return True


def main() -> int:
    """
    Main entry point for PFASDataLoader demonstration.

    Demonstrates:
    1. Creating and configuring a PFASDataLoader
    2. Loading single molecules by ID
    3. Creating batches from multiple molecule IDs
    4. Handling exceptions during data loading operations

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        args = parse_args()
        
        print('PFASDataLoader Demonstration')
        print('=' * 40)
        
        # Validate data directory
        if not validate_data_directory(args.data_dir):
            return 1
        
        print(f'Using data directory: {args.data_dir}')
        
        # Create loader configuration
        config = create_loader_config(args.env_features, args.label_types)
        print(f'Configuration: {config}')
        
        # Initialize data loader
        try:
            loader = PFASDataLoader(args.data_dir, config=config)
            print('✓ PFASDataLoader initialized successfully')
        except Exception as e:
            print(f'✗ Failed to initialize PFASDataLoader: {e}')
            return 1
        
        print('\n1. Single Molecule Loading Demonstration:')
        print('-' * 40)
        
        # Demonstrate single molecule loading
        successful_loads = 0
        for molecule_id in args.molecule_ids:
            if demonstrate_single_molecule_loading(loader, molecule_id):
                successful_loads += 1
        
        print(f'\nSuccessfully loaded {successful_loads}/{len(args.molecule_ids)} molecules')
        
        print('\n2. Batch Creation Demonstration:')
        print('-' * 40)
        
        # Demonstrate batch creation
        if successful_loads > 0:
            success = demonstrate_batch_creation(
                loader, args.molecule_ids, args.batch_size
            )
            if not success:
                print('Batch creation failed, but single molecule loading worked')
        else:
            print('Skipping batch demonstration (no molecules loaded successfully)')
        
        print('\nDemonstration completed.')
        return 0
        
    except KeyboardInterrupt:
        print('\nDemonstration cancelled by user')
        return 1
    except Exception as e:
        print(f'Unexpected error: {e}')
        return 1


if __name__ == '__main__':
    exit(main())
