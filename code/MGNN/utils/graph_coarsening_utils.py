"""
Graph Coarsening Utilities for PFAS Compounds

This module provides utility functions for creating and saving hierarchical
graph representations of PFAS molecules at different levels of coarseness.
"""

import os
import torch
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from torch_geometric.data import Data

from code.MGNN.architectures.graph_coarsening import GraphCoarsener
from code.MGNN.utils.mol_graph_generator import create_graph_from_orca_data


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
    Create hierarchical graph representations from molecule file and ORCA output.
    
    Args:
        mol_file: Path to the molecule file (SDF, MOL)
        orca_output: Path to the ORCA output file
        output_dir: Directory to save graph files (default is directory of mol_file)
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
        use_3d_coords: Whether to include 3D coordinates in the graphs
    
    Returns:
        Dictionary mapping hierarchy level to output file path
    """
    # Create base atom-level graph first
    base_output_file = None
    if output_dir is not None:
        base_name = os.path.splitext(os.path.basename(mol_file))[0]
        base_output_file = os.path.join(output_dir, f"{base_name}_atom_graph.pt")
    
    # Create the atom-level graph
    atom_graph_path = create_graph_from_orca_data(
        mol_file=mol_file,
        orca_output=orca_output,
        output_file=base_output_file,
        charge_type=charge_type,
        use_pfas_features=use_pfas_features,
        use_quantum_properties=use_quantum_properties
    )
    
    # Set up output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(atom_graph_path)
    
    # Load the atom-level graph
    atom_graph = torch.load(atom_graph_path)
    
    # Load the molecule
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {mol_file}")
    
    # Initialize graph coarsener
    coarsener = GraphCoarsener(use_3d_coords=use_3d_coords)
    
    # Create hierarchical graphs
    hierarchical_graphs = coarsener.create_hierarchical_graphs(atom_graph, mol)
    
    # Save the coarsened graphs
    output_paths = {
        'atom': atom_graph_path
    }
    
    base_name = os.path.splitext(os.path.basename(atom_graph_path))[0].replace("_atom_graph", "")
    
    # Save functional group level graph
    fg_output_path = os.path.join(output_dir, f"{base_name}_functional_group_graph.pt")
    torch.save(hierarchical_graphs['functional_group'], fg_output_path)
    output_paths['functional_group'] = fg_output_path
    print(f"Functional group graph saved to {fg_output_path}")
    
    # Save structural motif level graph
    sm_output_path = os.path.join(output_dir, f"{base_name}_structural_motif_graph.pt")
    torch.save(hierarchical_graphs['structural_motif'], sm_output_path)
    output_paths['structural_motif'] = sm_output_path
    print(f"Structural motif graph saved to {sm_output_path}")
    
    return output_paths


def batch_create_hierarchical_graphs(
    mol_dir: str,
    orca_dir: str,
    output_dir: str,
    charge_type: str = 'mulliken',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True,
    use_3d_coords: bool = True
) -> Dict[str, List[str]]:
    """
    Batch create hierarchical graphs for multiple PFAS molecules.
    
    Args:
        mol_dir: Directory containing molecule files
        orca_dir: Directory containing ORCA output files
        output_dir: Directory to save graph files
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
        use_3d_coords: Whether to include 3D coordinates in the graphs
    
    Returns:
        Dictionary mapping hierarchy level to list of output file paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_paths = {
        'atom': [],
        'functional_group': [],
        'structural_motif': []
    }
    
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
    
    # Process each matched pair of molecule and ORCA output files
    for base_name, mol_file in mol_files.items():
        # Look for exact match
        if base_name in orca_files:
            orca_file = orca_files[base_name]
            
            try:
                # Create hierarchical graphs
                paths = create_hierarchical_graphs_from_orca(
                    mol_file=mol_file,
                    orca_output=orca_file,
                    output_dir=output_dir,
                    charge_type=charge_type,
                    use_pfas_features=use_pfas_features,
                    use_quantum_properties=use_quantum_properties,
                    use_3d_coords=use_3d_coords
                )
                
                # Store output paths
                for level, path in paths.items():
                    output_paths[level].append(path)
                
                print(f"Created hierarchical graphs for {base_name}")
            except Exception as e:
                print(f"Error creating hierarchical graphs for {base_name}: {e}")
        else:
            # Try to find similar names
            found = False
            for orca_name, orca_file in orca_files.items():
                if base_name in orca_name or orca_name in base_name:
                    try:
                        # Create hierarchical graphs
                        paths = create_hierarchical_graphs_from_orca(
                            mol_file=mol_file,
                            orca_output=orca_file,
                            output_dir=output_dir,
                            charge_type=charge_type,
                            use_pfas_features=use_pfas_features,
                            use_quantum_properties=use_quantum_properties,
                            use_3d_coords=use_3d_coords
                        )
                        
                        # Store output paths
                        for level, path in paths.items():
                            output_paths[level].append(path)
                        
                        print(f"Created hierarchical graphs for {base_name} using {orca_name} ORCA output")
                        found = True
                        break
                    except Exception as e:
                        print(f"Error creating hierarchical graphs for {base_name} using {orca_name} ORCA output: {e}")
            
            if not found:
                print(f"No matching ORCA output found for {base_name}")
    
    return output_paths


def visualize_hierarchical_graphs(
    graph_paths: Dict[str, str],
    output_dir: Optional[str] = None,
    highlight_feature: str = 'functional_group'
) -> Dict[str, str]:
    """
    Visualize hierarchical graphs at different levels of coarseness.
    
    Args:
        graph_paths: Dictionary mapping hierarchy level to graph file path
        output_dir: Directory to save visualization files (default is same as graph files)
        highlight_feature: Feature to highlight in visualization
        
    Returns:
        Dictionary mapping hierarchy level to visualization file path
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError:
        print("Visualization requires matplotlib, networkx, and PyTorch Geometric. Please install these packages.")
        return {}
    
    visualization_paths = {}
    
    for level, graph_path in graph_paths.items():
        # Load the graph
        graph = torch.load(graph_path)
        
        # Convert to NetworkX graph
        G = to_networkx(graph, to_undirected=True)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get node positions from the graph if available
        if graph.pos is not None:
            pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.num_nodes)}
        else:
            # Use spring layout if 3D positions not available
            pos = nx.spring_layout(G, seed=42)
        
        # Define node colors based on hierarchy level
        if level == 'atom':
            # Color atoms by type for atom-level graph
            node_colors = []
            for i in range(graph.num_nodes):
                # Check if atom is F, C, or other
                is_f = graph.x[i][8].item() > 0.5  # Index 8 is the fluorine flag
                is_cf = graph.x[i][9].item() > 0.5  # Index 9 is the carbon-fluorine flag
                
                if is_f:
                    node_colors.append('green')  # Fluorine atoms
                elif is_cf:
                    node_colors.append('orange')  # Carbon atoms bonded to fluorine
                else:
                    node_colors.append('lightblue')  # Other atoms
        
        elif level == 'functional_group':
            # Color functional groups
            node_colors = ['purple' if i < 10 else 'red' for i in range(graph.num_nodes)]
        
        elif level == 'structural_motif':
            # For structural motifs, use different colors for head and tail
            node_colors = ['blue' if i == 0 else 'yellow' for i in range(graph.num_nodes)]
        
        # Draw the graph
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=600,
                width=2.0,
                with_labels=True,
                font_size=12,
                font_weight='bold')
        
        # Add title
        plt.title(f'{level.replace("_", " ").title()} Level Graph', fontsize=16)
        
        # Save or show the figure
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name = os.path.splitext(os.path.basename(graph_path))[0]
            viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            plt.savefig(viz_path, bbox_inches='tight', dpi=300)
            visualization_paths[level] = viz_path
            print(f"{level.title()} visualization saved to {viz_path}")
        else:
            # Save in the same directory as the graph file
            base_name = os.path.splitext(graph_path)[0]
            viz_path = f"{base_name}_visualization.png"
            plt.savefig(viz_path, bbox_inches='tight', dpi=300)
            visualization_paths[level] = viz_path
            print(f"{level.title()} visualization saved to {viz_path}")
        
        plt.close()
    
    return visualization_paths 