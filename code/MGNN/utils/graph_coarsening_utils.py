"""
Graph Coarsening Utilities for PFAS Molecular Structures

This module provides utility functions for creating hierarchical graph representations
of PFAS molecules at different levels of coarseness:
1. Atom level (original graph)
2. Functional group level (intermediate coarseness)
3. Structural motif level (highest coarseness)
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from rdkit import Chem

from code.MGNN.architectures.graph_coarsening import GraphCoarsener
from code.MGNN.utils.unified_graph_generator import create_graph_from_orca
from code.utils.quantum.orca_parser import parse_orca_output


def create_hierarchical_graphs_from_orca(
    mol_file: str,
    orca_output: str,
    output_dir: Optional[str] = None,
    charge_type: str = 'mulliken',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True,
    use_3d_coords: bool = True
) -> Dict[str, str]:
    """
    Create hierarchical graph representations from a molecule file and ORCA output.
    
    Args:
        mol_file: Path to the molecule file (MOL/SDF)
        orca_output: Path to the ORCA output file
        output_dir: Directory to save graph files (default: same directory as mol_file)
        charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
        use_3d_coords: Whether to use 3D coordinates
    
    Returns:
        Dictionary mapping level names to paths of saved graph files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(mol_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load molecule
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load molecule from {mol_file}")
    
    # Parse ORCA output for quantum data
    qm_data = None
    if use_quantum_properties and os.path.exists(orca_output):
        qm_data = parse_orca_output(orca_output)
    
    # Create base atom-level graph
    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    atom_graph_path = os.path.join(output_dir, f"{base_name}_atom_graph.pt")
    
    # Use unified graph generator to create atom-level graph
    atom_graph_data = create_graph_from_orca(
        mol_file=mol_file,
        orca_file=orca_output if use_quantum_properties else None,
        output_file=atom_graph_path,
        charge_type=charge_type,
        use_pfas_features=use_pfas_features,
        use_quantum_properties=use_quantum_properties
    )
    
    # Load the saved graph if it was saved, otherwise use returned data
    if os.path.exists(atom_graph_path):
        atom_graph_data = torch.load(atom_graph_path)
    
    # Create graph coarsener
    coarsener = GraphCoarsener(use_3d_coords=use_3d_coords)
    
    # Generate hierarchical graphs
    hierarchical_graphs = coarsener.create_hierarchical_graphs(atom_graph_data, mol)
    
    # Save each level of graph
    graph_paths = {}
    graph_paths['atom'] = atom_graph_path
    
    for level, graph in hierarchical_graphs.items():
        if level == 'atom':
            continue  # Already saved
        
        # Save this level
        graph_path = os.path.join(output_dir, f"{base_name}_{level}_graph.pt")
        torch.save(graph, graph_path)
        graph_paths[level] = graph_path
    
    return graph_paths


def batch_create_hierarchical_graphs(
    mol_dir: str,
    orca_dir: str,
    output_dir: str,
    charge_type: str = 'mulliken',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Batch process multiple molecules to create hierarchical graph representations.
    
    Args:
        mol_dir: Directory containing molecule files
        orca_dir: Directory containing ORCA output files
        output_dir: Directory to save graph files
        charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
    
    Returns:
        Dictionary mapping molecule names to dictionaries of graph paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_graph_paths = {}
    
    # Find all molecule files
    mol_files = {}
    for filename in os.listdir(mol_dir):
        if filename.endswith('.mol') or filename.endswith('.sdf'):
            base_name = os.path.splitext(filename)[0]
            mol_files[base_name] = os.path.join(mol_dir, filename)
    
    # Find all ORCA output files
    orca_files = {}
    for filename in os.listdir(orca_dir):
        if filename.endswith('.out') or filename.endswith('.log'):
            base_name = os.path.splitext(filename)[0]
            orca_files[base_name] = os.path.join(orca_dir, filename)
    
    # Process each molecule with matching ORCA output
    for base_name, mol_file in mol_files.items():
        mol_output_dir = os.path.join(output_dir, base_name)
        
        # Find matching ORCA output
        orca_file = None
        if base_name in orca_files:
            orca_file = orca_files[base_name]
        else:
            # Look for similar names
            for orca_name, orca_path in orca_files.items():
                if base_name in orca_name or orca_name in base_name:
                    orca_file = orca_path
                    break
        
        if orca_file is None:
            print(f"No matching ORCA output found for {base_name}, skipping")
            continue
        
        try:
            # Create hierarchical graphs
            graph_paths = create_hierarchical_graphs_from_orca(
                mol_file=mol_file,
                orca_output=orca_file,
                output_dir=mol_output_dir,
                charge_type=charge_type,
                use_pfas_features=use_pfas_features,
                use_quantum_properties=use_quantum_properties
            )
            
            all_graph_paths[base_name] = graph_paths
            print(f"Created hierarchical graphs for {base_name}")
            
        except Exception as e:
            print(f"Error creating hierarchical graphs for {base_name}: {e}")
    
    return all_graph_paths


def visualize_graph_hierarchy(
    graph_paths: Dict[str, str],
    output_dir: Optional[str] = None,
    highlight_features: List[str] = ['fluorine', 'functional_group']
):
    """
    Visualize a set of hierarchical graphs with different highlighting options.
    
    Args:
        graph_paths: Dictionary mapping level names to graph file paths
        output_dir: Directory to save visualizations
        highlight_features: List of features to highlight in separate visualizations
    """
    from code.MGNN.utils.visualization import visualize_molecular_graph, visualize_hierarchical_graphs
    
    # Use hierarchical visualization if available
    if visualize_hierarchical_graphs is not None:
        visualize_hierarchical_graphs(graph_paths, output_dir)
    else:
        # Fallback to individual visualizations
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for level, graph_path in graph_paths.items():
            try:
                graph = torch.load(graph_path)
                
                for feature in highlight_features:
                    if output_dir:
                        base_name = os.path.splitext(os.path.basename(graph_path))[0]
                        viz_path = os.path.join(output_dir, f"{base_name}_{feature}.png")
                    else:
                        viz_path = None
                    
                    visualize_molecular_graph(graph, viz_path, highlight_feature=feature)
                    
            except Exception as e:
                print(f"Error visualizing {level} graph: {e}") 