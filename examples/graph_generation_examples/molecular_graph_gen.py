#!/usr/bin/env python
"""
Molecular Graph Generator

A unified tool for generating molecular graph representations for molecules:
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
import argparse
import torch
from typing import Dict, Optional
from rdkit import Chem

# Import modules
from moml.core import create_graph_processor, GraphCoarsener, FunctionalGroupDetector

from moml.simulation.qm.parser.orca_parser import parse_orca_output

from moml.utils.visualization.visualization import visualize_molecular_graph, print_graph_statistics


def create_and_visualize_atomic_graph(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    highlight_feature: str = "fluorine",
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True,
) -> str:
    """Create and optionally visualize an atomic-level molecular graph."""
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine output file path
    output_file = None
    if output_dir is not None:
        base_name = os.path.splitext(os.path.basename(mol_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_graph.pt")

    # Create the graph processor
    graph_processor = create_graph_processor(
        {
            "use_pfas_specific_features": use_pfas_features,
            "use_3d_coords": True,
            "use_partial_charges": use_quantum_properties,
        }
    )

    # Parse ORCA output for partial charges
    partial_charges = None
    if use_quantum_properties and orca_output and os.path.exists(orca_output):
        orca_data = parse_orca_output(orca_output)
        # ORCA parser uses the key `mulliken_charges`
        if orca_data and "mulliken_charges" in orca_data:
            partial_charges = orca_data["mulliken_charges"]

    # Create the graph
    graph = graph_processor.file_to_graph(
        mol_file, additional_features={"partial_charges": partial_charges} if partial_charges else None
    )

    # Check if graph was successfully created
    if graph is None:
        print(f"Error: Failed to create graph from {mol_file}. The molecule may be unreadable or invalid.")
        return None

    # Save the graph
    if output_file:
        torch.save(graph, output_file)

    # Load and print graph statistics
    print_graph_statistics(graph)

    # Visualize if requested
    if visualize:
        viz_path = None
        if output_file:
            viz_path = os.path.splitext(output_file)[0] + "_visualization.png"
        visualize_molecular_graph(graph, viz_path, highlight_feature)

    return output_file


def create_and_visualize_hierarchical_graphs(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True,
) -> Dict[str, str]:
    """Create and optionally visualize hierarchical graphs."""
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a GraphCoarsener instance
    coarsener = GraphCoarsener(use_3d_coords=True, use_pfas_features=use_pfas_features)

    # Create hierarchical graphs
    hierarchical_graph_paths = coarsener.create_from_orca(
        mol_file=mol_file,
        orca_output=orca_output,
        output_dir=output_dir,
        charge_type="mulliken",
        use_quantum_properties=use_quantum_properties,
    )

    # Print statistics for each graph
    for level, path in hierarchical_graph_paths.items():
        graph = torch.load(path)
        print(f"\n{level.upper()} LEVEL GRAPH:")
        print_graph_statistics(graph)

    # Visualize if requested
    if visualize:
        if output_dir is None:
            output_dir = os.path.dirname(mol_file)

        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        for level, path in hierarchical_graph_paths.items():
            graph = torch.load(path)

            # Create a visualization for fluorine highlighting
            viz_path = os.path.join(viz_dir, f"{os.path.basename(path).split('.')[0]}_fluorine.png")
            visualize_molecular_graph(graph, viz_path, highlight_feature="fluorine")

            # Create a visualization for functional groups
            viz_path = os.path.join(viz_dir, f"{os.path.basename(path).split('.')[0]}_functional.png")
            visualize_molecular_graph(graph, viz_path, highlight_feature="functional_group")

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
        identifier = FunctionalGroupDetector()
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
    parser = argparse.ArgumentParser(description="Molecular Graph Generator")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--mol", type=str, required=True, help="Path to molecule file (MOL/SDF)")

    # Common arguments for graph generation
    graph_parser = argparse.ArgumentParser(add_help=False, parents=[common_parser])
    graph_parser.add_argument("--orca", type=str, help="Path to ORCA output file")
    graph_parser.add_argument("--output-dir", type=str, help="Directory to save generated graphs")
    graph_parser.add_argument("--visualize", action="store_true", help="Visualize the generated graph")
    graph_parser.add_argument(
        "--no-pfas-features", action="store_false", dest="use_pfas_features", help="Disable PFAS-specific features"
    )
    graph_parser.add_argument(
        "--no-quantum-properties",
        action="store_false",
        dest="use_quantum_properties",
        help="Disable quantum properties",
    )

    # Atomic graph command
    atomic_parser = subparsers.add_parser(
        "atomic", parents=[graph_parser], help="Generate atomic-level molecular graph"
    )
    atomic_parser.add_argument(
        "--highlight",
        type=str,
        default="fluorine",
        choices=["fluorine", "partial_charge", "functional_group", "head_group"],
        help="Feature to highlight in visualization",
    )

    # Hierarchical graph command
    subparsers.add_parser(
        "hierarchical", parents=[graph_parser], help="Generate hierarchical graphs"
    )

    # Analyze command
    subparsers.add_parser("analyze", parents=[common_parser], help="Analyze functional groups")

    args = parser.parse_args()

    # Check if ORCA file is required but not provided
    if args.command in ["atomic", "hierarchical"] and args.use_quantum_properties and args.orca is None:
        print("Quantum properties are enabled but no ORCA file provided. Proceeding without quantum properties.")
        args.use_quantum_properties = False

    # Execute the appropriate command
    if args.command == "atomic":
        create_and_visualize_atomic_graph(
            mol_file=args.mol,
            orca_output=args.orca,
            output_dir=args.output_dir,
            visualize=args.visualize,
            highlight_feature=args.highlight,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties,
        )
    elif args.command == "hierarchical":
        create_and_visualize_hierarchical_graphs(
            mol_file=args.mol,
            orca_output=args.orca,
            output_dir=args.output_dir,
            visualize=args.visualize,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties,
        )
    elif args.command == "analyze":
        analyze_functional_groups(args.mol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
