#!/usr/bin/env python
"""
Hierarchical Graph Creation Example

This script demonstrates how to use the GraphCoarsener class to create
hierarchical representations of PFAS molecules at different levels of granularity.
"""

import os
import torch
import argparse
from rdkit import Chem
from pathlib import Path

from moml.core.graph_coarsening import GraphCoarsener
from moml.core.molecular_graph import create_graph_processor
from moml.utils.visualization.visualization import visualize_molecular_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Create hierarchical molecular graphs")
    parser.add_argument(
        "--mol_file", 
        type=str, 
        required=True,
        help="Path to the molecule file (MOL/SDF format)"
    )
    parser.add_argument(
        "--orca_file", 
        type=str, 
        default=None,
        help="Path to the ORCA output file (optional, for quantum properties)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output",
        help="Directory to save the hierarchical graphs"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Whether to visualize the hierarchical graphs"
    )
    parser.add_argument(
        "--use_3d", 
        action="store_true",
        help="Whether to use 3D coordinates in the graphs"
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing molecule: {args.mol_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Create a GraphCoarsener instance
    coarsener = GraphCoarsener(
        use_3d_coords=args.use_3d,
        use_pfas_features=True  # Enable PFAS-specific features
    )
    
    # Method 1: Create hierarchical graphs from molecule file only
    if args.orca_file is None:
        print("Creating hierarchical graphs from molecule file...")
        graph_paths = coarsener.create_from_molecule_file(
            mol_file=args.mol_file,
            output_dir=args.output_dir
        )
    
    # Method 2: Create hierarchical graphs with quantum properties from ORCA
    else:
        print("Creating hierarchical graphs with quantum properties...")
        graph_paths = coarsener.create_from_orca(
            mol_file=args.mol_file,
            orca_output=args.orca_file,
            output_dir=args.output_dir,
            charge_type='mulliken',
            use_quantum_properties=True
        )
    
    # Print information about the created graphs
    print("\nHierarchical graphs created:")
    for level, path in graph_paths.items():
        graph = torch.load(path)
        print(f"- {level.upper()} level: {Path(path).name}")
        print(f"  - {graph.num_nodes} nodes, {graph.num_edges // 2} edges")
        
        # Print node feature information
        print(f"  - Node features: {list(graph.ndata.keys())}")
        
        # For atom level, print atom types
        if level == 'atom' and 'atomic_num' in graph.ndata:
            atom_nums = graph.ndata['atomic_num'].tolist()
            atom_types = [Chem.GetPeriodicTable().GetElementSymbol(int(num)) for num in atom_nums]
            print(f"  - Atoms: {atom_types}")
    
    # Visualize the graphs if requested
    if args.visualize:
        print("\nVisualizing hierarchical graphs...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        for level, path in graph_paths.items():
            graph = torch.load(path)
            
            # Create a visualization for fluorine highlighting
            vis_path = os.path.join(vis_dir, f"{Path(path).stem}_fluorine.png")
            visualize_molecular_graph(graph, vis_path, highlight_feature='fluorine')
            print(f"- Created visualization: {Path(vis_path).name}")
            
            # Create a visualization for functional groups
            vis_path = os.path.join(vis_dir, f"{Path(path).stem}_functional.png")
            visualize_molecular_graph(graph, vis_path, highlight_feature='functional_group')
            print(f"- Created visualization: {Path(vis_path).name}")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 