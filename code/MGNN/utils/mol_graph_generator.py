"""
Molecular Graph Generator for PFAS Compounds

This module provides a command-line interface and functions for generating
molecular graph representations from molecule files and ORCA output files.
"""

import os
import argparse
import torch
from typing import List, Optional, Tuple

# Import from our modules
# Update these imports based on your actual module structure
from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder, mol_file_to_graph
from code.MGNN.utils.orca_parser import extract_partial_charges_from_orca, extract_orbital_contributions_from_orca, extract_electrostatic_potential_from_orca


def create_graph_from_orca_data(mol_file: str, 
                               orca_output: str, 
                               output_file: Optional[str] = None,
                               charge_type: str = 'mulliken',
                               use_pfas_features: bool = True,
                               use_quantum_properties: bool = True) -> str:
    """
    Create a molecular graph from a molecule file and ORCA output with enhanced PFAS features.
    
    Args:
        mol_file: Path to the molecule file (SDF, MOL)
        orca_output: Path to the ORCA output file
        output_file: Path to save the graph data (default is mol_file with _graph.pt extension)
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
    
    Returns:
        Path to the saved graph file
    """
    # Extract partial charges from ORCA output
    partial_charges = extract_partial_charges_from_orca(orca_output, charge_type)
    
    # Extract additional quantum properties if requested
    homo_lumo_contributions = None
    electrostatic_potential = None
    
    if use_quantum_properties:
        # Try to extract HOMO/LUMO contributions
        try:
            homo_lumo_contributions = extract_orbital_contributions_from_orca(orca_output)
        except Exception as e:
            print(f"Warning: Could not extract HOMO/LUMO contributions: {e}")
        
        # Try to extract electrostatic potential values
        try:
            electrostatic_potential = extract_electrostatic_potential_from_orca(orca_output)
        except Exception as e:
            print(f"Warning: Could not extract electrostatic potential values: {e}")
    
    # Create molecular graph builder
    builder = MolecularGraphBuilder(
        use_partial_charges=True,
        use_3d_coords=True,
        use_pfas_specific_features=use_pfas_features
    )
    
    # Read molecule from file
    from rdkit import Chem
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {mol_file}")
    
    # Convert to graph
    graph = builder.mol_to_graph(mol, partial_charges, homo_lumo_contributions)
    
    # Add electrostatic potential values as a separate attribute if available
    if electrostatic_potential is not None:
        if len(electrostatic_potential) == mol.GetNumAtoms():
            graph.esp = torch.tensor(electrostatic_potential, dtype=torch.float)
        else:
            print(f"Warning: Number of ESP values ({len(electrostatic_potential)}) does not match "
                  f"number of atoms ({mol.GetNumAtoms()}). ESP values will not be included.")
    
    # Save graph to file
    if output_file is None:
        base_name = os.path.splitext(mol_file)[0]
        output_file = f"{base_name}_graph.pt"
    
    torch.save(graph, output_file)
    print(f"Graph saved to {output_file}")
    
    return output_file


def batch_create_graphs_from_orca(mol_dir: str, 
                                  orca_dir: str, 
                                  output_dir: str,
                                  charge_type: str = 'mulliken',
                                  use_pfas_features: bool = True,
                                  use_quantum_properties: bool = True) -> List[str]:
    """
    Batch create molecular graphs from molecule files and ORCA outputs with enhanced PFAS features.
    
    Args:
        mol_dir: Directory containing molecule files
        orca_dir: Directory containing ORCA output files
        output_dir: Directory to save graph files
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        use_pfas_features: Whether to include PFAS-specific features
        use_quantum_properties: Whether to include quantum properties from ORCA
    
    Returns:
        List of paths to created graph files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    created_files = []
    
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
    
    # Match molecule files with ORCA outputs
    for base_name, mol_file in mol_files.items():
        # Look for exact match
        if base_name in orca_files:
            orca_file = orca_files[base_name]
            output_file = os.path.join(output_dir, f"{base_name}_graph.pt")
            
            try:
                create_graph_from_orca_data(
                    mol_file, 
                    orca_file, 
                    output_file, 
                    charge_type,
                    use_pfas_features=use_pfas_features,
                    use_quantum_properties=use_quantum_properties
                )
                created_files.append(output_file)
                print(f"Created graph for {base_name}")
            except Exception as e:
                print(f"Error creating graph for {base_name}: {e}")
        else:
            # Look for similar names (with prefixes/suffixes)
            found = False
            for orca_name, orca_file in orca_files.items():
                if base_name in orca_name or orca_name in base_name:
                    output_file = os.path.join(output_dir, f"{base_name}_graph.pt")
                    
                    try:
                        create_graph_from_orca_data(
                            mol_file, 
                            orca_file, 
                            output_file, 
                            charge_type,
                            use_pfas_features=use_pfas_features,
                            use_quantum_properties=use_quantum_properties
                        )
                        created_files.append(output_file)
                        print(f"Created graph for {base_name} using {orca_name} ORCA output")
                        found = True
                        break
                    except Exception as e:
                        print(f"Error creating graph for {base_name} using {orca_name} ORCA output: {e}")
            
            if not found:
                print(f"No matching ORCA output found for {base_name}")
    
    return created_files


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Generate molecular graphs from molecule files and ORCA outputs')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single molecule graph creation
    single_parser = subparsers.add_parser('single', help='Create a graph for a single molecule')
    single_parser.add_argument('--mol', required=True, help='Path to molecule file (SDF, MOL)')
    single_parser.add_argument('--orca', required=True, help='Path to ORCA output file')
    single_parser.add_argument('--output', required=False, help='Path to save graph file')
    single_parser.add_argument('--charge-type', default='mulliken', choices=['mulliken', 'loewdin'],
                              help='Type of charges to extract')
    single_parser.add_argument('--use-pfas-features', action='store_true', default=True,
                              help='Include PFAS-specific features')
    single_parser.add_argument('--no-pfas-features', action='store_false', dest='use_pfas_features',
                              help='Do not include PFAS-specific features')
    single_parser.add_argument('--use-quantum-properties', action='store_true', default=True,
                              help='Include quantum properties from ORCA')
    single_parser.add_argument('--no-quantum-properties', action='store_false', dest='use_quantum_properties',
                              help='Do not include quantum properties from ORCA')
    
    # Batch molecule graph creation
    batch_parser = subparsers.add_parser('batch', help='Create graphs for multiple molecules')
    batch_parser.add_argument('--mol-dir', required=True, help='Directory containing molecule files')
    batch_parser.add_argument('--orca-dir', required=True, help='Directory containing ORCA output files')
    batch_parser.add_argument('--output-dir', required=True, help='Directory to save graph files')
    batch_parser.add_argument('--charge-type', default='mulliken', choices=['mulliken', 'loewdin'],
                             help='Type of charges to extract')
    batch_parser.add_argument('--use-pfas-features', action='store_true', default=True,
                             help='Include PFAS-specific features')
    batch_parser.add_argument('--no-pfas-features', action='store_false', dest='use_pfas_features',
                             help='Do not include PFAS-specific features')
    batch_parser.add_argument('--use-quantum-properties', action='store_true', default=True,
                             help='Include quantum properties from ORCA')
    batch_parser.add_argument('--no-quantum-properties', action='store_false', dest='use_quantum_properties',
                             help='Do not include quantum properties from ORCA')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        create_graph_from_orca_data(
            args.mol, 
            args.orca, 
            args.output, 
            args.charge_type,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties
        )
    elif args.command == 'batch':
        batch_create_graphs_from_orca(
            args.mol_dir, 
            args.orca_dir, 
            args.output_dir, 
            args.charge_type,
            use_pfas_features=args.use_pfas_features,
            use_quantum_properties=args.use_quantum_properties
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 