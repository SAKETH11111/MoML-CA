#!/usr/bin/env python
"""
Preprocessing Example for MoML-CA.

This script demonstrates how to preprocess molecule files 
to create graph representations for machine learning.
"""

import os
import argparse
from moml import preprocess_molecules


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Molecule preprocessing example")
    
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing molecule files (.mol or .sdf)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed graph files (.pt)')
    parser.add_argument('--charges_dir', type=str, default=None,
                      help='Optional directory containing partial charge files')
    parser.add_argument('--file_pattern', type=str, default="*.mol,*.sdf",
                      help='Pattern(s) to match molecule files')
    
    return parser.parse_args()


def main():
    """Run the preprocessing example."""
    # Parse command-line arguments
    args = parse_args()
    
    print("MoML-CA Preprocessing Example")
    print("============================")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create configuration dictionary
    config = {
        # Atom features
        'use_atom_symbol': True,
        'use_atom_charge': True,
        'use_atom_hybridization': True,
        'use_atom_is_aromatic': True,
        'use_atom_is_in_ring': True,
        
        # Bond features
        'use_bond_type': True,
        'use_bond_is_conjugated': True,
        'use_bond_is_in_ring': True,
        
        # 3D features
        'use_3d_coords': True,
        
        # PFAS-specific features
        'use_pfas_specific_features': True,
        
        # Partial charges
        'use_partial_charges': True if args.charges_dir else False,
    }
    
    # Process the molecules
    results = preprocess_molecules(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        charges_dir=args.charges_dir,
        file_pattern=args.file_pattern
    )
    
    # Report results
    print("\nPreprocessing Results:")
    print(f"Total processed files: {len(results['all_files'])}")
    
    # Print example of one of the processed files
    if results['all_files']:
        example_file = results['all_files'][0]
        print(f"\nExample processed file: {example_file}")
        print("To load this processed graph:")
        print(f"  graph = torch.load('{example_file}')")
    
    print("\nNext steps:")
    print("1. Use moml.data.dataset.load_dataset() to load these graphs")
    print("2. Use moml.data.dataset.split_dataset() to create train/val/test splits")
    print("3. Use moml.data.dataset.prepare_dataloaders() to create dataloaders")
    
    print("\nPreprocessing example completed successfully!")


if __name__ == "__main__":
    main() 