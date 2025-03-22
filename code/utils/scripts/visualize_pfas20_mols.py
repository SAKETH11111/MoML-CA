#!/usr/bin/env python3
"""
Visualize the standardized PFAS-20 molecular structures.

This script generates 2D visualizations of the 20 PFAS molecules that were
standardized in the data processing step.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path to import project modules
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Updated imports to match directory structure
from code.utils.plotting.molecules.molecule_visualization import (
    load_molecules_from_pickle,
    visualize_dataset
)


def main():
    """Generate visualizations for PFAS-20 molecules."""
    parser = argparse.ArgumentParser(description="Visualize PFAS-20 molecules")
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/chemical_list/pfas20_rdkit_mols.pkl",
        help="Path to pickle file with processed molecules"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/processed/visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--grid-filename",
        "-g",
        default="pfas20_compounds_grid.png",
        help="Filename for the molecule grid image"
    )
    parser.add_argument(
        "--columns",
        "-c",
        type=int,
        default=4,
        help="Number of columns in the molecule grid"
    )
    args = parser.parse_args()
    
    # Resolve paths relative to project root if not absolute
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    # Load molecules
    print(f"Loading PFAS-20 molecules from: {input_path}")
    molecules = load_molecules_from_pickle(input_path)
    print(f"Loaded {len(molecules)} molecules")
    
    # Generate visualizations
    print(f"Generating PFAS-20 visualizations in: {output_dir}")
    visualization_paths = visualize_dataset(
        molecules,
        output_dir,
        grid_filename=args.grid_filename,
        n_cols=args.columns
    )
    
    # Print summary
    print("\nVisualization Summary:")
    print(f"- Grid image: {visualization_paths['grid']}")
    print(f"- Individual images: {len(visualization_paths['individual'])} files in {os.path.dirname(list(visualization_paths['individual'].values())[0])}")


if __name__ == "__main__":
    main() 