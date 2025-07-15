"""
molecular_graph_gen.py

A unified command-line tool for generating various types of molecular graph
representations including atomic-level, hierarchical, and functional group
analysis with optional quantum enhancement.
"""

import argparse
import os
from typing import Dict, Optional, Union, Any

import torch
from rdkit import Chem
from torch_geometric.data import Data as PyGData

from moml.core import (
    create_graph_processor,
    FunctionalGroupDetector,
    GraphCoarsener,
)
from moml.simulation.qm.parser.orca_parser import parse_orca_output
from moml.utils.visualization_utils.visualization import visualize_molecular_graph

DEFAULT_OUTPUT_DIR = './output'
DEFAULT_HIGHLIGHT_FEATURE = 'fluorine'
DEFAULT_CHARGE_TYPE = 'mulliken'
VISUALIZATION_SUBDIR = 'visualizations'
DEFAULT_USE_PFAS_FEATURES = True
DEFAULT_USE_QUANTUM_PROPERTIES = True
DEFAULT_USE_3D_COORDS = True
SUPPORTED_MOL_EXTENSIONS = ['.mol', '.sdf', '.mol2']
ORCA_OUTPUT_EXTENSION = '.out'
PFAS_FLUORINE_THRESHOLD = 10.0


def _print_graph_statistics(graph: Any, level_name: str) -> None:
    """
    Print detailed statistics about a molecular graph.

    Args:
        graph (PyGData): PyTorch Geometric Data object.
        level_name (str): Name of the graph level for display.
    """
    print(f'  {level_name} level statistics:')
    print(f'    Nodes: {graph.num_nodes}')
    print(f'    Edges: {graph.num_edges // 2}')

    if hasattr(graph, 'x') and graph.x is not None:
        print(f'    Node features: {graph.x.shape[1]}')
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        print(f'    Edge features: {graph.edge_attr.shape[1]}')


def _create_single_visualization(
    graph: Any,
    mol_name: str,
    level: str,
    highlight_feature: str,
    output_dir: Optional[str]
) -> None:
    """
    Create a single graph visualization.

    Args:
        graph (PyGData): PyTorch Geometric Data object to visualize.
        mol_name (str): Base name of the molecule.
        level (str): Graph level name.
        highlight_feature (str): Feature to highlight.
        output_dir (Optional[str]): Output directory.
    """
    try:
        if output_dir:
            vis_dir = os.path.join(output_dir, VISUALIZATION_SUBDIR)
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(
                vis_dir,
                f'{mol_name}_{level}_{highlight_feature}.png'
            )
        else:
            vis_path = f'{mol_name}_{level}_{highlight_feature}.png'

        visualize_molecular_graph(
            graph,
            vis_path,
            highlight_feature=highlight_feature
        )
        print(f'  Visualization saved: {os.path.basename(vis_path)}')

    except Exception as e:
        print(f'  Warning: Visualization failed for {level} graph: {e}')


def create_and_visualize_atomic_graph(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    highlight_feature: str = DEFAULT_HIGHLIGHT_FEATURE,
    use_pfas_features: bool = DEFAULT_USE_PFAS_FEATURES,
    use_quantum_properties: bool = DEFAULT_USE_QUANTUM_PROPERTIES,
) -> str:
    """
    Create and optionally visualize an atomic-level molecular graph.

    Generates a PyTorch Geometric Data object representing the molecular
    structure at the atomic level, with optional quantum properties from
    ORCA calculations and visualization capabilities.

    Args:
        mol_file (str): Path to molecule file (MOL/SDF format).
        orca_output (str): Path to ORCA output file for quantum properties.
        output_dir (Optional[str]): Directory to save outputs. If None,
            uses current directory.
        visualize (bool): Whether to create graph visualizations.
        highlight_feature (str): Feature to highlight in visualizations
            ('fluorine', 'functional_group', etc.).
        use_pfas_features (bool): Whether to include PFAS-specific features.
        use_quantum_properties (bool): Whether to include quantum properties
            from ORCA output.

    Returns:
        str: Path to the saved graph file.

    Raises:
        FileNotFoundError: If molecule or ORCA output file doesn't exist.
        ValueError: If graph creation fails.
        OSError: If output directory cannot be created.
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    processor_config = {
        'use_pfas_features': use_pfas_features,
        'use_quantum_properties': use_quantum_properties,
        'use_3d_coords': DEFAULT_USE_3D_COORDS,
        'charge_type': DEFAULT_CHARGE_TYPE,
    }
    processor = create_graph_processor(processor_config)

    print(f'Parsing ORCA output: {orca_output}')
    try:
        orca_data = parse_orca_output(orca_output)
        print('ORCA parsing successful')
    except Exception as e:
        print(f'Warning: ORCA parsing failed: {e}')
        orca_data = None

    print(f'Creating atomic graph from: {mol_file}')
    mol = Chem.MolFromMolFile(mol_file)
    if mol is None:
        raise ValueError(f'Could not load molecule from {mol_file}')

    graph = processor.mol_to_graph(mol)

    if graph is None:
        raise ValueError('Graph creation failed')

    mol_name = os.path.splitext(os.path.basename(mol_file))[0]
    if output_dir:
        graph_path = os.path.join(output_dir, f'{mol_name}_atomic_graph.pt')
    else:
        graph_path = f'{mol_name}_atomic_graph.pt'

    torch.save(graph, graph_path)
    print(f'Atomic graph saved to: {graph_path}')

    _print_graph_statistics(graph, 'Atomic')

    if visualize:
        _create_single_visualization(
            graph, mol_name, 'atomic', highlight_feature, output_dir
        )

    return graph_path


def create_and_visualize_hierarchical_graphs(
    mol_file: str,
    orca_output: Optional[str] = None,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    highlight_feature: str = DEFAULT_HIGHLIGHT_FEATURE,
) -> Dict[str, Union[str, PyGData]]:
    """
    Create and optionally visualize hierarchical molecular graphs.

    Generates multi-level graph representations (atom, functional group,
    structural motif) using the GraphCoarsener class, with optional
    quantum property enhancement from ORCA calculations.

    Args:
        mol_file (str): Path to molecule file (MOL/SDF format).
        orca_output (Optional[str]): Path to ORCA output file. If None,
            creates graphs without quantum properties.
        output_dir (Optional[str]): Directory to save outputs. If None,
            uses current directory.
        visualize (bool): Whether to create graph visualizations.
        highlight_feature (str): Feature to highlight in visualizations.

    Returns:
        Dict[str, Union[str, PyGData]]: Dictionary mapping level names to graph
            file paths or in-memory PyGData objects.

    Raises:
        FileNotFoundError: If molecule file doesn't exist.
        ValueError: If graph creation fails.
        OSError: If output directory cannot be created.
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f'Creating hierarchical graphs from: {mol_file}')

    coarsener = GraphCoarsener(
        use_3d_coords=DEFAULT_USE_3D_COORDS,
        use_pfas_features=DEFAULT_USE_PFAS_FEATURES
    )

    if orca_output is not None:
        print(f'Including quantum properties from: {orca_output}')
        graph_paths = coarsener.create_from_orca(
            mol_file=mol_file,
            orca_output=orca_output,
            output_dir=output_dir or '.',
            charge_type=DEFAULT_CHARGE_TYPE,
            use_quantum_properties=DEFAULT_USE_QUANTUM_PROPERTIES,
        )
    else:
        print('Creating graphs without quantum properties')
        graph_paths = coarsener.create_from_molecule_file(
            mol_file=mol_file,
            output_dir=output_dir or '.'
        )

    print('\nHierarchical graphs created:')
    for level, path in graph_paths.items():
        try:
            if isinstance(path, str):
                graph = torch.load(path)
                print(f'- {level.upper()}: {os.path.basename(path)}')
                _print_graph_statistics(graph, level.capitalize())
            else:
                print(f'- {level.upper()}: (in-memory graph object)')
                _print_graph_statistics(path, level.capitalize())
        except Exception as e:
            print(f'Error loading {level} graph: {e}')

    if visualize:
        mol_name = os.path.splitext(os.path.basename(mol_file))[0]
        for level, path in graph_paths.items():
            try:
                if isinstance(path, str):
                    graph = torch.load(path)
                    _create_single_visualization(
                        graph, mol_name, level, highlight_feature, output_dir
                    )
                else:
                    print(f'Warning: Skipping visualization for {level} graph '
                          '(in-memory object)')
            except Exception as e:
                print(f'Error visualizing {level} graph: {e}')

    return graph_paths


def analyze_functional_groups(
    mol_file: str,
    output_dir: Optional[str] = None
) -> Dict[str, int]:
    """
    Analyze and report functional groups present in a molecule.

    Uses the FunctionalGroupDetector to identify and count functional
    groups, with special emphasis on PFAS-relevant groups and fluorine
    content analysis.

    Args:
        mol_file (str): Path to molecule file (MOL/SDF format).
        output_dir (Optional[str]): Directory to save analysis results.
            If None, only prints to console.

    Returns:
        Dict[str, int]: Dictionary mapping functional group names to counts.

    Raises:
        FileNotFoundError: If molecule file doesn't exist.
        ValueError: If molecule cannot be loaded.
    """
    print(f'Analyzing functional groups in: {mol_file}')

    mol = Chem.MolFromMolFile(mol_file)
    if mol is None:
        raise ValueError(f'Could not load molecule from {mol_file}')

    detector = FunctionalGroupDetector()

    raw_functional_groups = detector.get_all_functional_groups(mol)

    functional_group_counts: Dict[str, int] = {}
    for group_name, group_data in raw_functional_groups.items():
        if isinstance(group_data, list) or isinstance(group_data, set):
            functional_group_counts[group_name] = len(group_data)
        else:
            functional_group_counts[group_name] = 1 if group_data else 0

    fluorine_count = len([
        atom for atom in mol.GetAtoms()
        if atom.GetSymbol() == 'F'
    ])

    if fluorine_count > 0:
        functional_group_counts['fluorine_atoms'] = fluorine_count

    print('\nFunctional Group Analysis:')
    print('-' * 30)
    if functional_group_counts:
        for group, count in functional_group_counts.items():
            print(f'{group}: {count}')
    else:
        print('No functional groups detected')

    total_atoms = mol.GetNumAtoms()
    fluorine_percentage = (
        (fluorine_count / total_atoms) * 100 if total_atoms > 0 else 0
    )

    print(f'\nPFAS Analysis:')
    print(f'Total atoms: {total_atoms}')
    print(f'Fluorine atoms: {fluorine_count}')
    print(f'Fluorine percentage: {fluorine_percentage:.1f}%')

    is_pfas_like = fluorine_percentage > PFAS_FLUORINE_THRESHOLD
    print(f'PFAS-like characteristics: {"Yes" if is_pfas_like else "No"}')

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        mol_name = os.path.splitext(os.path.basename(mol_file))[0]
        analysis_file = os.path.join(
            output_dir, f'{mol_name}_functional_groups.txt'
        )

        with open(analysis_file, 'w') as f:
            f.write(f'Functional Group Analysis for {mol_name}\n')
            f.write('=' * 50 + '\n\n')
            f.write('Detected Functional Groups:\n')
            for group, count in functional_group_counts.items():
                f.write(f'{group}: {count}\n')
            f.write(f'\nPFAS Analysis:\n')
            f.write(f'Total atoms: {total_atoms}\n')
            f.write(f'Fluorine atoms: {fluorine_count}\n')
            f.write(f'Fluorine percentage: {fluorine_percentage:.1f}%\n')
            f.write(f'PFAS-like: {"Yes" if is_pfas_like else "No"}\n')

        print(f'Analysis saved to: {analysis_file}')

    return functional_group_counts


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        description='Generate molecular graph representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s atomic --mol molecule.mol --orca output.out --visualize
  %(prog)s hierarchical --mol pfoa.mol --output_dir ./graphs
  %(prog)s analyze --mol molecule.mol --output_dir ./analysis
        """
    )

    subparsers = parser.add_subparsers(
        dest='mode',
        help='Graph generation mode',
        required=True
    )

    atomic_parser = subparsers.add_parser(
        'atomic',
        help='Generate atomic-level molecular graph'
    )
    atomic_parser.add_argument(
        '--mol',
        required=True,
        help='Path to molecule file (MOL/SDF format)'
    )
    atomic_parser.add_argument(
        '--orca',
        required=True,
        help='Path to ORCA output file'
    )
    atomic_parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory for graphs and visualizations'
    )
    atomic_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create graph visualizations'
    )
    atomic_parser.add_argument(
        '--highlight',
        default=DEFAULT_HIGHLIGHT_FEATURE,
        choices=['fluorine', 'functional_group', 'charge'],
        help='Feature to highlight in visualizations'
    )

    hierarchical_parser = subparsers.add_parser(
        'hierarchical',
        help='Generate hierarchical molecular graphs'
    )
    hierarchical_parser.add_argument(
        '--mol',
        required=True,
        help='Path to molecule file (MOL/SDF format)'
    )
    hierarchical_parser.add_argument(
        '--orca',
        default=None,
        help='Path to ORCA output file (optional)'
    )
    hierarchical_parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory for graphs and visualizations'
    )
    hierarchical_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create graph visualizations'
    )
    hierarchical_parser.add_argument(
        '--highlight',
        default=DEFAULT_HIGHLIGHT_FEATURE,
        choices=['fluorine', 'functional_group', 'charge'],
        help='Feature to highlight in visualizations'
    )

    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze functional groups in molecule'
    )
    analyze_parser.add_argument(
        '--mol',
        required=True,
        help='Path to molecule file (MOL/SDF format)'
    )
    analyze_parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory for analysis results'
    )

    return parser


def main() -> int:
    """
    Main entry point for molecular graph generation tool.

    Parses command-line arguments and executes the appropriate graph
    generation or analysis function based on the selected mode.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        parser = create_parser()
        args = parser.parse_args()

        if args.mode == 'atomic':
            create_and_visualize_atomic_graph(
                mol_file=args.mol,
                orca_output=args.orca,
                output_dir=args.output_dir,
                visualize=args.visualize,
                highlight_feature=args.highlight
            )
        elif args.mode == 'hierarchical':
            create_and_visualize_hierarchical_graphs(
                mol_file=args.mol,
                orca_output=args.orca,
                output_dir=args.output_dir,
                visualize=args.visualize,
                highlight_feature=args.highlight
            )
        elif args.mode == 'analyze':
            analyze_functional_groups(
                mol_file=args.mol,
                output_dir=args.output_dir
            )

        print('\nGraph generation completed successfully!')
        return 0

    except (FileNotFoundError, ValueError, OSError) as e:
        print(f'Error: {e}')
        return 1
    except KeyboardInterrupt:
        print('\nOperation cancelled by user')
        return 1


if __name__ == '__main__':
    exit(main())
