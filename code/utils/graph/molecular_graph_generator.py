"""
Molecular Graph Generator

This module provides functions to generate molecular graphs from mol files without quantum properties.
This is used as a fallback when QM calculations are skipped.
"""

import os
import json
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Configure logging
logger = logging.getLogger(__name__)

def generate_atom_features(atom, use_pfas_features=True) -> Dict[str, Any]:
    """
    Generate features for an atom in a molecule.
    
    Args:
        atom: RDKit atom object
        use_pfas_features: Whether to include PFAS-specific features
        
    Returns:
        Dictionary of atom features
    """
    # Basic atom features
    features = {
        "atomic_num": atom.GetAtomicNum(),
        "formal_charge": atom.GetFormalCharge(),
        "hybridization": int(atom.GetHybridization()),
        "num_hydrogens": atom.GetTotalNumHs(),
        "is_aromatic": int(atom.GetIsAromatic()),
        "is_in_ring": int(atom.IsInRing()),
        "degree": atom.GetDegree(),
        "implicit_valence": atom.GetImplicitValence(),
        "explicit_valence": atom.GetExplicitValence(),
    }
    
    # Add PFAS-specific features
    if use_pfas_features:
        features.update({
            "is_halogen": int(atom.GetAtomicNum() in [9, 17, 35, 53]),  # F, Cl, Br, I
            "is_fluorine": int(atom.GetAtomicNum() == 9),
            "is_carbon": int(atom.GetAtomicNum() == 6),
            "is_oxygen": int(atom.GetAtomicNum() == 8),
            "is_sulfur": int(atom.GetAtomicNum() == 16),
            "is_nitrogen": int(atom.GetAtomicNum() == 7),
            "is_phosphorus": int(atom.GetAtomicNum() == 15),
        })
    
    return features

def generate_bond_features(bond) -> Dict[str, Any]:
    """
    Generate features for a bond in a molecule.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        Dictionary of bond features
    """
    bond_type = bond.GetBondType()
    
    # Convert bond type to integer
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_type_int = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_type_int = 2
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_type_int = 3
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_type_int = 4
    else:
        bond_type_int = 0
    
    # Basic bond features
    features = {
        "bond_type": bond_type_int,
        "is_conjugated": int(bond.GetIsConjugated()),
        "is_in_ring": int(bond.IsInRing()),
        "is_aromatic": int(bond.GetIsAromatic()),
    }
    
    return features

def get_molecule_descriptors(mol) -> Dict[str, float]:
    """
    Calculate basic molecular descriptors for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of molecular descriptors
    """
    descriptors = {
        "mol_weight": Descriptors.MolWt(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "qed": Descriptors.qed(mol),
        "fraction_sp3": Descriptors.FractionCSP3(mol),
    }
    
    return descriptors

def create_molecular_graph(mol_file: str, output_dir: str, use_pfas_features: bool = True) -> Optional[str]:
    """
    Create a molecular graph from a mol file without quantum properties.
    
    Args:
        mol_file: Path to the mol file
        output_dir: Directory to save the graph
        use_pfas_features: Whether to include PFAS-specific features
        
    Returns:
        Path to the generated graph file if successful, None otherwise
    """
    try:
        mol_id = os.path.splitext(os.path.basename(mol_file))[0]
        logger.info(f"Creating graph for {mol_id}")
        
        # Load molecule from file
        mol = Chem.MolFromMolFile(mol_file)
        if mol is None:
            logger.error(f"Failed to load molecule from {mol_file}")
            return None
        
        # Get atom features
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append({
                "idx": atom.GetIdx(),
                "features": generate_atom_features(atom, use_pfas_features),
                "coords": mol.GetConformer().GetAtomPosition(atom.GetIdx()).__dict__,
            })
        
        # Get bond features
        bonds = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            bonds.append({
                "begin_atom_idx": begin_idx,
                "end_atom_idx": end_idx,
                "features": generate_bond_features(bond),
            })
        
        # Calculate molecule descriptors
        descriptors = get_molecule_descriptors(mol)
        
        # Create graph
        graph = {
            "mol_id": mol_id,
            "atoms": atoms,
            "bonds": bonds,
            "descriptors": descriptors,
            "quantum_properties": {},  # Empty since quantum properties are not available
        }
        
        # Save graph to file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{mol_id}_graph.json")
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"Created graph for {mol_id}: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error creating graph for {mol_file}: {str(e)}")
        return None

def batch_create_graphs_from_molecules(mol_dir: str, output_dir: str, 
                                       use_pfas_features: bool = True, 
                                       max_workers: int = 4) -> List[str]:
    """
    Create molecular graphs for all molecules in a directory.
    
    Args:
        mol_dir: Directory containing mol files
        output_dir: Directory to save the graphs
        use_pfas_features: Whether to include PFAS-specific features
        max_workers: Maximum number of worker processes to use
        
    Returns:
        List of paths to the generated graph files
    """
    # Check if directories exist
    if not os.path.exists(mol_dir):
        logger.error(f"Molecule directory does not exist: {mol_dir}")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of mol files
    mol_files = [os.path.join(mol_dir, f) for f in os.listdir(mol_dir) if f.endswith('.mol')]
    if not mol_files:
        logger.warning(f"No mol files found in {mol_dir}")
        return []
    
    logger.info(f"Processing {len(mol_files)} molecules in parallel with {max_workers} workers")
    
    # Process molecules in parallel
    graph_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                create_molecular_graph, 
                mol_file, 
                output_dir, 
                use_pfas_features
            ): mol_file for mol_file in mol_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            mol_file = futures[future]
            mol_id = os.path.splitext(os.path.basename(mol_file))[0]
            
            try:
                graph_file = future.result()
                if graph_file:
                    graph_files.append(graph_file)
                else:
                    logger.warning(f"Failed to create graph for {mol_id}")
            except Exception as e:
                logger.error(f"Error processing {mol_id}: {str(e)}")
    
    logger.info(f"Created {len(graph_files)} molecular graphs")
    return graph_files

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    import argparse
    parser = argparse.ArgumentParser(description="Create molecular graphs from mol files")
    parser.add_argument("--mol-dir", required=True, help="Directory containing mol files")
    parser.add_argument("--output-dir", required=True, help="Directory to save the graphs")
    parser.add_argument("--use-pfas-features", action="store_true", help="Include PFAS-specific features")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes")
    
    args = parser.parse_args()
    
    # Create graphs
    graph_files = batch_create_graphs_from_molecules(
        args.mol_dir, 
        args.output_dir, 
        args.use_pfas_features, 
        args.max_workers
    )
    
    print(f"Created {len(graph_files)} molecular graphs") 