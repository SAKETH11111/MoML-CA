#!/usr/bin/env python
"""
Batch Hierarchical Graph Creation Example

This script demonstrates how to use the GraphCoarsener class to batch process
multiple molecules and create hierarchical representations.
"""

import os
import argparse
from pathlib import Path

from moml import GraphCoarsener


def parse_args():
    parser = argparse.ArgumentParser(description="Batch create hierarchical molecular graphs")
    parser.add_argument(
        "--mol_dir", 
        type=str, 
        required=True,
        help="Directory containing molecule files (MOL/SDF format)"
    )
    parser.add_argument(
        "--orca_dir", 
        type=str, 
        default=None,
        help="Directory containing ORCA output files (optional, for quantum properties)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./batch_output",
        help="Directory to save the hierarchical graphs"
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
    
    print(f"Processing molecules from: {args.mol_dir}")
    if args.orca_dir:
        print(f"Using ORCA outputs from: {args.orca_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create a GraphCoarsener instance
    coarsener = GraphCoarsener(
        use_3d_coords=args.use_3d,
        use_pfas_features=True  # Enable PFAS-specific features
    )
    
    # Method 1: Batch process molecule files only (no quantum properties)
    if args.orca_dir is None:
        print("\nBatch processing molecule files without quantum properties...")
        
        # Find all molecule files
        mol_files = []
        for ext in ['.mol', '.sdf']:
            mol_files.extend(list(Path(args.mol_dir).glob(f'*{ext}')))
        
        if not mol_files:
            print("No molecule files found!")
            return
        
        print(f"Found {len(mol_files)} molecule files")
        
        # Process each molecule file individually
        results = {}
        for mol_file in mol_files:
            mol_name = mol_file.stem
            mol_output_dir = os.path.join(args.output_dir, mol_name)
            
            try:
                print(f"Processing {mol_name}...")
                graph_paths = coarsener.create_from_molecule_file(
                    mol_file=str(mol_file),
                    output_dir=mol_output_dir
                )
                results[mol_name] = graph_paths
                print(f"  Created graphs at {len(graph_paths)} levels")
            except Exception as e:
                print(f"  Error processing {mol_name}: {e}")
    
    # Method 2: Batch process with ORCA outputs (with quantum properties)
    else:
        print("\nBatch processing molecules with quantum properties...")
        try:
            all_results = coarsener.batch_create_from_directories(
                mol_dir=args.mol_dir,
                orca_dir=args.orca_dir,
                output_dir=args.output_dir,
                charge_type='mulliken',
                use_quantum_properties=True
            )
            
            # Print summary of results
            print(f"\nProcessed {len(all_results)} molecules:")
            for mol_name, graph_paths in all_results.items():
                levels = list(graph_paths.keys())
                print(f"- {mol_name}: Created {len(levels)} levels ({', '.join(levels)})")
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 