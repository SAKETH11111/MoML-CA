#!/usr/bin/env python
"""
PFAS Data Pipeline

This script provides a comprehensive data pipeline for processing PFAS molecules:
1. Extract quantum chemical data from ORCA output files
2. Generate molecular graphs with PFAS-specific features
3. Create hierarchical graph representations at multiple scales
4. Visualize and analyze the results

Usage:
    python pfas_pipeline.py --mol-dir /path/to/molecules --orca-dir /path/to/orca_outputs --output-dir /path/to/output

Options:
    --hierarchical         Generate hierarchical graphs (default: False)
    --visualize            Visualize the generated graphs (default: False)
    --analyze              Analyze functional groups (default: False)
    --charge-type          Type of partial charges to use ('mulliken' or 'loewdin')
"""

import os
import sys
import argparse
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from rdkit import Chem

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import our modules
from code.MGNN.utils.orca_parser import batch_extract_charges
from code.MGNN.utils.mol_graph_generator import batch_create_graphs_from_orca
from code.MGNN.utils.graph_coarsening_utils import batch_create_hierarchical_graphs
from code.MGNN.utils.visualization import (
    visualize_molecular_graph, 
    visualize_hierarchical_graphs, 
    print_graph_statistics
)
from code.MGNN.architectures.graph_coarsening import FunctionalGroupIdentifier


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


def run_pipeline(
    mol_dir: str,
    orca_dir: str,
    output_dir: str,
    hierarchical: bool = False,
    visualize: bool = False,
    analyze: bool = False,
    charge_type: str = 'mulliken'
) -> Dict[str, List[str]]:
    """
    Run the complete PFAS data processing pipeline.
    
    Args:
        mol_dir: Directory containing molecule files (MOL/SDF)
        orca_dir: Directory containing ORCA output files
        output_dir: Directory to save generated graphs
        hierarchical: Whether to generate hierarchical graphs
        visualize: Whether to visualize the graphs
        analyze: Whether to analyze functional groups
        charge_type: Type of partial charges to use
        
    Returns:
        Dictionary mapping graph types to lists of output file paths
    """
    print("Starting PFAS data pipeline...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Extract partial charges from ORCA outputs
    print("\nExtracting partial charges from ORCA outputs...")
    charges_dir = os.path.join(output_dir, "charges")
    batch_extract_charges(orca_dir, charges_dir, charge_type=charge_type)
    
    # Find molecule files
    mol_files = [os.path.join(mol_dir, f) for f in os.listdir(mol_dir) 
                 if f.endswith('.mol') or f.endswith('.sdf')]
    
    # Analyze functional groups if requested
    if analyze:
        print("\nAnalyzing functional groups...")
        for mol_file in tqdm(mol_files, desc="Analyzing molecules"):
            print(f"\nAnalyzing {os.path.basename(mol_file)}:")
            analyze_functional_groups(mol_file)
    
    # Generate graphs
    output_paths = {}
    
    if hierarchical:
        print("\nGenerating hierarchical graphs...")
        graph_dir = os.path.join(output_dir, "hierarchical_graphs")
        output_paths = batch_create_hierarchical_graphs(
            mol_dir=mol_dir,
            orca_dir=orca_dir,
            output_dir=graph_dir,
            charge_type=charge_type
        )
        
        if visualize:
            print("\nVisualizing hierarchical graphs...")
            viz_dir = os.path.join(output_dir, "visualizations")
            
            # Select one graph from each level for visualization
            if all(output_paths.values()):
                sample_graphs = {level: paths[0] for level, paths in output_paths.items() if paths}
                visualize_hierarchical_graphs(sample_graphs, viz_dir)
    else:
        print("\nGenerating atomic-level graphs...")
        graph_dir = os.path.join(output_dir, "atomic_graphs")
        atom_graphs = batch_create_graphs_from_orca(
            mol_dir=mol_dir,
            orca_dir=orca_dir,
            output_dir=graph_dir,
            charge_type=charge_type
        )
        output_paths = {'atom': atom_graphs}
        
        if visualize:
            print("\nVisualizing atomic-level graphs...")
            viz_dir = os.path.join(output_dir, "visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Visualize a few graphs for reference
            for i, graph_path in enumerate(atom_graphs[:min(3, len(atom_graphs))]):
                graph = torch.load(graph_path)
                base_name = os.path.splitext(os.path.basename(graph_path))[0]
                viz_path = os.path.join(viz_dir, f"{base_name}_visualization.png")
                visualize_molecular_graph(graph, viz_path, 'fluorine')
    
    # Print summary
    print("\nPipeline completed successfully!")
    print(f"Output directory: {output_dir}")
    
    for level, paths in output_paths.items():
        print(f"  {level.replace('_', ' ').title()} graphs: {len(paths)}")
    
    return output_paths


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='PFAS Data Pipeline')
    parser.add_argument('--mol-dir', required=True, help='Directory containing molecule files (MOL/SDF)')
    parser.add_argument('--orca-dir', required=True, help='Directory containing ORCA output files')
    parser.add_argument('--output-dir', required=True, help='Directory to save generated graphs')
    parser.add_argument('--hierarchical', action='store_true', help='Generate hierarchical graphs')
    parser.add_argument('--visualize', action='store_true', help='Visualize the graphs')
    parser.add_argument('--analyze', action='store_true', help='Analyze functional groups')
    parser.add_argument('--charge-type', default='mulliken', choices=['mulliken', 'loewdin'],
                       help='Type of partial charges to use')
    
    args = parser.parse_args()
    
    run_pipeline(
        mol_dir=args.mol_dir,
        orca_dir=args.orca_dir,
        output_dir=args.output_dir,
        hierarchical=args.hierarchical,
        visualize=args.visualize,
        analyze=args.analyze,
        charge_type=args.charge_type
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 