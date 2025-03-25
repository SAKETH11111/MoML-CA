#!/usr/bin/env python
"""
PFAS Molecular Graph Generator

A unified tool for generating molecular graph representations for PFAS molecules:
- Regular atomic-level graphs
- Hierarchical graphs (atom, functional group, and structural motif levels)
- Graph visualization
- Functional group analysis

Usage:
    # Generate a regular molecular graph
    python molecular_graph_gen.py atomic --mol molecule.mol --orca orca_output.out

    # Generate hierarchical graphs
    python molecular_graph_gen.py hierarchical --mol molecule.mol --orca orca_output.out

    # Visualize graphs
    python molecular_graph_gen.py atomic --mol molecule.mol --orca orca_output.out --visualize
    python molecular_graph_gen.py hierarchical --mol molecule.mol --orca orca_output.out --visualize

    # Analyze functional groups
    python molecular_graph_gen.py analyze --mol molecule.mol
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import our modules
from rdkit import Chem
from code.MGNN.utils.mol_graph_generator import create_graph_from_orca_data
from code.MGNN.utils.graph_coarsening_utils import create_hierarchical_graphs_from_orca
from code.MGNN.utils.visualization import (
    visualize_molecular_graph,
    print_graph_statistics,
    visualize_hierarchical_graphs
)
from code.MGNN.architectures.graph_coarsening import FunctionalGroupIdentifier


def create_and_visualize_atomic_graph(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    highlight_feature: str = 'fluorine',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True
) -> str:
    """Create and optionally visualize an atomic-level molecular graph."""
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine output file path
    output_file = None
    if output_dir is not None:
        base_name = os.path.splitext(os.path.basename(mol_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_graph.pt")
    
    # Create the graph
    graph_path = create_graph_from_orca_data(
        mol_file=mol_file,
        orca_output=orca_output,
        output_file=output_file,
        use_pfas_features=use_pfas_features,
        use_quantum_properties=use_quantum_properties
    )
    
    # Load and print graph statistics
    graph = torch.load(graph_path)
    print_graph_statistics(graph)
    
    # Visualize if requested
    if visualize:
        viz_path = os.path.splitext(graph_path)[0] + "_visualization.png"
        visualize_molecular_graph(graph, viz_path, highlight_feature)
    
    return graph_path


def create_and_visualize_hierarchical_graphs(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True
) -> Dict[str, str]:
    """Create and optionally visualize hierarchical graphs."""
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create hierarchical graphs
    hierarchical_graph_paths = create_hierarchical_graphs_from_orca(
        mol_file=mol_file,
        orca_output=orca_output,
        output_dir=output_dir,
        use_pfas_features=use_pfas_features,
        use_quantum_properties=use_quantum_properties
    )
    
    # Visualize if requested
    if visualize:
        viz_dir = os.path.join(output_dir, "visualizations") if output_dir else None
        visualize_hierarchical_graphs(hierarchical_graph_paths, viz_dir)
    
    return hierarchical_graph_paths


def analyze_functional_groups(mol_file: str) -> None:
    """
    Analyze the functional groups in a molecule.
    
    Args:
        mol_file: Path to the molecule file
    """
    try:
        # Load molecule
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        if mol is None:
            print(f"Error: Could not load molecule from {mol_file}")
            return
        
        # Identify functional groups
        identifier = FunctionalGroupIdentifier()
        cf_groups, functional_groups = identifier.identify_all_functional_groups(mol)
        
        # Print results
        print(f"\nFunctional Group Analysis for {os.path.basename(mol_file)}:")
        
        print("\nFluorinated Groups:")
        if cf_groups:
            for atom_idx, group_type in cf_groups.items():
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                print(f"  {group_type} group at atom {atom_idx} ({symbol})")
        else:
            print("  No fluorinated groups found")
        
        print("\nOther Functional Groups:")
        if functional_groups:
            for i, group in enumerate(functional_groups):
                group_type = identifier.identify_functional_group_type(mol, group)
                atoms_str = ", ".join([str(idx) for idx in group])
                print(f"  Group {i+1} ({group_type}): atoms {atoms_str}")
        else:
            print("  No other functional groups found")
        
    except Exception as e:
        print(f"Error analyzing functional groups: {e}")


def main():
    """Parse arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description='PFAS Molecular Graph Generator')
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Command to run')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--mol', type=str, required=True,
                              help='Path to molecule file (MOL/SDF)')
    
    # Common arguments for graph generation
    graph_parser = argparse.ArgumentParser(add_help=False, parents=[common_parser])
    graph_parser.add_argument('--orca', type=str, required=True,
                            help='Path to ORCA output file')
    graph_parser.add_argument('--output-dir', type=str,
                            help='Directory to save generated graphs')
    graph_parser.add_argument('--visualize', action='store_true',
                            help='Visualize the generated graph')
    graph_parser.add_argument('--no-pfas-features', action='store_false', dest='use_pfas_features',
                            help='Disable PFAS-specific features')
    graph_parser.add_argument('--no-quantum-properties', action='store_false', dest='use_quantum_properties',
                            help='Disable quantum properties')
    
    # Atomic graph command
    atomic_parser = subparsers.add_parser('atomic', parents=[graph_parser],
                                         help='Generate atomic-level molecular graph')
    atomic_parser.add_argument('--highlight', type=str, default='fluorine',
                              choices=['fluorine', 'partial_charge', 'functional_group', 'head_group'],
                              help='Feature to highlight in visualization')
    
    # Hierarchical graph command
    hierarchical_parser = subparsers.add_parser('hierarchical', parents=[graph_parser],
                                              help='Generate hierarchical graphs')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', parents=[common_parser],
                                         help='Analyze functional groups')
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'atomic':
        create_and_visualize_atomic_graph(
            mol_file=args.mol,
            orca_output=args.orca,
            output_dir=args.output_dir,
            visualize=args.visualize,
            highlight_feature=args.highlight,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties
        )
    
    elif args.command == 'hierarchical':
        create_and_visualize_hierarchical_graphs(
            mol_file=args.mol,
            orca_output=args.orca,
            output_dir=args.output_dir,
            visualize=args.visualize,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties
        )
    
    elif args.command == 'analyze':
        analyze_functional_groups(args.mol)


if __name__ == '__main__':
    main() 