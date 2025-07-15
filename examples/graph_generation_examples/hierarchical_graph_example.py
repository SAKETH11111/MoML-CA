"""
hierarchical_graph_example.py

A command-line tool for creating hierarchical molecular graph representations at different granularity levels (atom, functional group, structural motif) with optional quantum properties integration.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
from rdkit import Chem

from moml.core import GraphCoarsener
from moml.utils.visualization_utils.visualization import visualize_molecular_graph

DEFAULT_OUTPUT_DIR = './output'
DEFAULT_CHARGE_TYPE = 'mulliken'
VISUALIZATION_SUBDIR = 'visualizations'
HIGHLIGHT_FEATURES = ['fluorine', 'functional_group']


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for hierarchical graph creation.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - mol_file: Path to molecule file
            - orca_file: Optional ORCA output file path
            - output_dir: Output directory path
            - visualize: Visualization flag
            - use_3d: 3D coordinates flag
    """
    parser = argparse.ArgumentParser(
        description='Create hierarchical molecular graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mol_file molecule.mol --output_dir ./output
  %(prog)s --mol_file molecule.sdf --orca_file output.out --visualize
  %(prog)s --mol_file pfoa.mol --use_3d --visualize
        """
    )
    parser.add_argument(
        '--mol_file',
        type=str,
        required=True,
        help='Path to the molecule file (MOL/SDF format)'
    )
    parser.add_argument(
        '--orca_file',
        type=str,
        default=None,
        help='Path to the ORCA output file (optional, for quantum properties)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to save the hierarchical graphs'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Whether to visualize the hierarchical graphs'
    )
    parser.add_argument(
        '--use_3d',
        action='store_true',
        help='Whether to use 3D coordinates in the graphs'
    )
    return parser.parse_args()


def create_hierarchical_graphs(mol_file: str, orca_file: Optional[str],
                              output_dir: str, use_3d: bool) -> Dict[str, Any]:
    """
    Create hierarchical molecular graphs with optional quantum properties.

    Args:
        mol_file (str): Path to the molecule file.
        orca_file (Optional[str]): Path to ORCA output file for quantum
            properties, or None for molecule-only graphs.
        output_dir (str): Directory to save the generated graphs.
        use_3d (bool): Whether to include 3D coordinates in graphs.

    Returns:
        Dict[str, str]: Dictionary mapping graph level names to file paths
            of the generated graph files.

    Raises:
        OSError: If output directory cannot be created.
        ValueError: If molecule file cannot be processed.
    """
    # Create a GraphCoarsener instance with PFAS-specific features
    coarsener = GraphCoarsener(
        use_3d_coords=use_3d,
        use_pfas_features=True
    )

    if orca_file is None:
        print('Creating hierarchical graphs from molecule file...')
        graph_paths = coarsener.create_from_molecule_file(
            mol_file=mol_file,
            output_dir=output_dir
        )
    else:
        print('Creating hierarchical graphs with quantum properties...')
        graph_paths = coarsener.create_from_orca(
            mol_file=mol_file,
            orca_output=orca_file,
            output_dir=output_dir,
            charge_type=DEFAULT_CHARGE_TYPE,
            use_quantum_properties=True
        )

    return graph_paths


def print_graph_information(graph_paths: Dict[str, Any]) -> None:
    """
    Print detailed information about created hierarchical graphs.

    Args:
        graph_paths (Dict[str, str]): Dictionary mapping level names to
            file paths of the generated graphs.
    """
    print('\nHierarchical graphs created:')
    
    for level, path in graph_paths.items():
        try:
            graph = torch.load(path)
            print(f'- {level.upper()} level: {Path(path).name}')
            print(f'  - {graph.num_nodes} nodes, {graph.num_edges // 2} edges')

            # Identify node feature attributes
            node_feature_keys = []
            if hasattr(graph, 'num_nodes'):
                for key in graph.keys:
                    attr = graph[key]
                    if (torch.is_tensor(attr) and attr.dim() > 0 and
                            attr.shape[0] == graph.num_nodes):
                        node_feature_keys.append(key)

            if node_feature_keys:
                print(f'  - Node attributes (features): {node_feature_keys}')
                for key in node_feature_keys:
                    print(f'    - {key}: shape {graph[key].shape}')
            else:
                print('  - No node attributes (features) found matching '
                      'num_nodes criteria.')

            # Print atomic composition for atom-level graphs
            if (level == 'atom' and hasattr(graph, 'atomic_num') and
                    torch.is_tensor(graph.atomic_num)):
                atom_nums = graph.atomic_num.tolist()
                atom_types = [
                    Chem.GetPeriodicTable().GetElementSymbol(int(num))
                    for num in atom_nums
                ]
                print(f'  - Atoms: {atom_types}')
                
        except Exception as e:
            print(f'Error processing {level} level graph at {path}: {e}')
            continue


def create_visualizations(graph_paths: Dict[str, str], output_dir: str) -> None:
    """
    Create visualizations for all hierarchical graphs.

    Args:
        graph_paths (Dict[str, str]): Dictionary mapping level names to
            graph file paths.
        output_dir (str): Base output directory for visualizations.

    Raises:
        OSError: If visualization directory cannot be created.
    """
    print('\nVisualizing hierarchical graphs...')
    vis_dir = os.path.join(output_dir, VISUALIZATION_SUBDIR)
    os.makedirs(vis_dir, exist_ok=True)

    for level, path in graph_paths.items():
        try:
            graph = torch.load(path)
            
            # Create visualizations for different highlight features
            for feature in HIGHLIGHT_FEATURES:
                vis_path = os.path.join(
                    vis_dir,
                    f'{level}_{Path(path).stem}_{feature}.png'
                )
                visualize_molecular_graph(
                    graph,
                    vis_path,
                    highlight_feature=feature
                )
                print(f'- Created visualization: {Path(vis_path).name}')
                
        except Exception as e:
            print(f'Error visualizing {level} level graph at {path}: {e}')


def main() -> int:
    """
    Main entry point for hierarchical graph creation tool.

    Parses command-line arguments, creates hierarchical molecular graphs,
    prints detailed information, and optionally generates visualizations.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        # Parse command-line arguments
        args = parse_args()

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        print(f'Processing molecule: {args.mol_file}')
        print(f'Output directory: {args.output_dir}')

        # Create hierarchical graphs
        graph_paths = create_hierarchical_graphs(
            args.mol_file,
            args.orca_file,
            args.output_dir,
            args.use_3d
        )

        # Print information about created graphs
        print_graph_information(graph_paths)

        # Create visualizations if requested
        if args.visualize:
            create_visualizations(graph_paths, args.output_dir)

        print('\nDone!')
        return 0
        
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f'Error: {e}')
        return 1


if __name__ == '__main__':
    exit(main())
