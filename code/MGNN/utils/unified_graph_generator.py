"""
Unified Graph Generator for MGNN

This module provides functionality to generate molecular graphs for the MGNN pipeline.
It supports graph creation from different sources:
1. SMILES strings
2. RDKit molecules
3. ORCA output files with quantum mechanical data

Graphs can be enriched with:
- Basic molecular features
- Quantum mechanical features (partial charges, etc.)
- PFAS-specific features
"""

import os
import sys
import logging
from pathlib import Path
import glob
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import concurrent.futures
import time
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

try:
    from code.utils.quantum.orca_parser import parse_orca_output
    from code.utils.helper_functions.molecular.molecule_processing import validate_smiles
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("graph_generator")

#-------------------------------------
# Graph Generation Classes
#-------------------------------------

class AtomFeatureExtractor:
    """Extract atom features for graph generation."""
    
    # Atom feature dictionaries
    ATOM_FEATURES = {
        'atomic_num': [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
        'is_aromatic': [False, True],
        'is_in_ring': [False, True]
    }
    
    # PFAS-specific feature flags
    PFAS_FEATURES = {
        'is_fluorine': [False, True],
        'is_carbon': [False, True],
        'is_f_bonded_to_c': [False, True],
        'is_part_of_cf2': [False, True],
        'is_part_of_cf3': [False, True]
    }
    
    def __init__(self, use_pfas_features: bool = True):
        """
        Initialize atom feature extractor.
        
        Args:
            use_pfas_features: Whether to use PFAS-specific features
        """
        self.use_pfas_features = use_pfas_features
    
    def _get_atom_features(self, atom: Chem.Atom, mol: Chem.Mol = None) -> Dict[str, Any]:
        """
        Extract features for a single atom.
        
        Args:
            atom: RDKit atom object
            mol: Parent molecule (used for PFAS features)
            
        Returns:
            Dictionary of atom features
        """
        features = {}
        
        # Basic atom features
        features['atomic_num'] = atom.GetAtomicNum()
        features['degree'] = atom.GetDegree()
        features['formal_charge'] = atom.GetFormalCharge()
        features['hybridization'] = atom.GetHybridization()
        features['is_aromatic'] = atom.GetIsAromatic()
        features['is_in_ring'] = atom.IsInRing()
        
        # PFAS-specific features
        if self.use_pfas_features and mol is not None:
            features['is_fluorine'] = atom.GetAtomicNum() == 9
            features['is_carbon'] = atom.GetAtomicNum() == 6
            
            # Check if F is bonded to C
            features['is_f_bonded_to_c'] = False
            if features['is_fluorine']:
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetAtomicNum() == 6:
                        features['is_f_bonded_to_c'] = True
                        break
            
            # Check if part of CF2 or CF3 group
            features['is_part_of_cf2'] = False
            features['is_part_of_cf3'] = False
            
            if features['is_carbon']:
                f_count = 0
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetAtomicNum() == 9:
                        f_count += 1
                
                features['is_part_of_cf2'] = f_count == 2
                features['is_part_of_cf3'] = f_count == 3
        
        return features
    
    def _one_hot_encode(self, value: Any, allowable_values: List[Any]) -> List[int]:
        """
        One-hot encode a value based on allowable values.
        
        Args:
            value: Value to encode
            allowable_values: List of allowable values
            
        Returns:
            One-hot encoded list
        """
        encoding = [0] * len(allowable_values)
        if value in allowable_values:
            encoding[allowable_values.index(value)] = 1
        return encoding
    
    def encode_atom(self, atom: Chem.Atom, mol: Chem.Mol = None) -> np.ndarray:
        """
        Encode an atom's features as a vector.
        
        Args:
            atom: RDKit atom object
            mol: Parent molecule (used for PFAS features)
            
        Returns:
            Numpy array of atom features
        """
        features = self._get_atom_features(atom, mol)
        feature_vector = []
        
        # Encode basic features
        for name, allowable_values in self.ATOM_FEATURES.items():
            feature_vector.extend(self._one_hot_encode(features[name], allowable_values))
        
        # Encode PFAS features if requested
        if self.use_pfas_features:
            for name, allowable_values in self.PFAS_FEATURES.items():
                feature_vector.extend(self._one_hot_encode(features.get(name, False), allowable_values))
        
        return np.array(feature_vector, dtype=np.float32)


class BondFeatureExtractor:
    """Extract bond features for graph generation."""
    
    # Bond feature dictionaries
    BOND_FEATURES = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'is_conjugated': [False, True],
        'is_in_ring': [False, True],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]
    }
    
    def __init__(self):
        """Initialize bond feature extractor."""
        pass
    
    def _get_bond_features(self, bond: Chem.Bond) -> Dict[str, Any]:
        """
        Extract features for a single bond.
        
        Args:
            bond: RDKit bond object
            
        Returns:
            Dictionary of bond features
        """
        features = {}
        
        features['bond_type'] = bond.GetBondType()
        features['is_conjugated'] = bond.GetIsConjugated()
        features['is_in_ring'] = bond.IsInRing()
        features['stereo'] = bond.GetStereo()
        
        return features
    
    def _one_hot_encode(self, value: Any, allowable_values: List[Any]) -> List[int]:
        """
        One-hot encode a value based on allowable values.
        
        Args:
            value: Value to encode
            allowable_values: List of allowable values
            
        Returns:
            One-hot encoded list
        """
        encoding = [0] * len(allowable_values)
        if value in allowable_values:
            encoding[allowable_values.index(value)] = 1
        return encoding
    
    def encode_bond(self, bond: Chem.Bond) -> np.ndarray:
        """
        Encode a bond's features as a vector.
        
        Args:
            bond: RDKit bond object
            
        Returns:
            Numpy array of bond features
        """
        features = self._get_bond_features(bond)
        feature_vector = []
        
        for name, allowable_values in self.BOND_FEATURES.items():
            feature_vector.extend(self._one_hot_encode(features[name], allowable_values))
        
        return np.array(feature_vector, dtype=np.float32)


class MolecularGraphGenerator:
    """Generate molecular graphs for MGNN."""
    
    def __init__(self, use_pfas_features: bool = True, use_3d_coords: bool = True):
        """
        Initialize molecular graph generator.
        
        Args:
            use_pfas_features: Whether to use PFAS-specific features
            use_3d_coords: Whether to use 3D coordinates
        """
        self.atom_featurizer = AtomFeatureExtractor(use_pfas_features=use_pfas_features)
        self.bond_featurizer = BondFeatureExtractor()
        self.use_3d_coords = use_3d_coords
    
    def mol_to_graph(self, mol: Chem.Mol, mol_id: str = None, qm_data: Dict = None) -> Dict:
        """
        Convert a molecule to a graph representation.
        
        Args:
            mol: RDKit molecule
            mol_id: Molecule identifier
            qm_data: Quantum mechanical data from ORCA
            
        Returns:
            Dictionary with graph data
        """
        if mol is None:
            logger.error("Cannot create graph from None molecule")
            return None
        
        try:
            # Basic graph data
            num_atoms = mol.GetNumAtoms()
            atom_features = []
            atomic_numbers = []
            
            # Node positions (3D coordinates)
            positions = np.zeros((num_atoms, 3), dtype=np.float32)
            has_positions = mol.GetNumConformers() > 0
            
            # Partial charges from QM data
            atom_charges = np.zeros(num_atoms, dtype=np.float32)
            if qm_data and 'mulliken_charges' in qm_data and len(qm_data['mulliken_charges']) == num_atoms:
                atom_charges = np.array(qm_data['mulliken_charges'], dtype=np.float32)
            
            # Process atoms
            for atom_idx in range(num_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                
                # Get atom features
                atom_feature = self.atom_featurizer.encode_atom(atom, mol)
                atom_features.append(atom_feature)
                atomic_numbers.append(atom.GetAtomicNum())
                
                # Get 3D coordinates if available
                if has_positions and self.use_3d_coords:
                    pos = mol.GetConformer().GetAtomPosition(atom_idx)
                    positions[atom_idx] = [pos.x, pos.y, pos.z]
                # If no positions in mol but QM data has optimized geometry
                elif qm_data and 'optimized_geometry' in qm_data and len(qm_data['optimized_geometry']) == num_atoms:
                    coords = qm_data['optimized_geometry'][atom_idx].get('coordinates', [0, 0, 0])
                    positions[atom_idx] = coords
            
            # Edge indices and features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Get bond features
                bond_feature = self.bond_featurizer.encode_bond(bond)
                
                # Add edges in both directions for undirected graph
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                # Same features for both directions
                edge_features.append(bond_feature)
                edge_features.append(bond_feature)
            
            # Convert to tensors
            atom_features = torch.tensor(np.array(atom_features), dtype=torch.float)
            edge_indices = torch.tensor(np.array(edge_indices), dtype=torch.long).t().contiguous()
            edge_features = torch.tensor(np.array(edge_features), dtype=torch.float)
            positions = torch.tensor(positions, dtype=torch.float)
            atom_charges = torch.tensor(atom_charges, dtype=torch.float)
            
            # Create graph data dict
            data = {
                'x': atom_features,
                'edge_index': edge_indices,
                'edge_attr': edge_features,
                'pos': positions,
                'charges': atom_charges,
                'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
                'mol_id': mol_id or "unknown"
            }
            
            # Add global features if available
            if qm_data:
                global_features = []
                
                # Add HOMO-LUMO gap if available
                if 'homo_lumo_gap' in qm_data and qm_data['homo_lumo_gap'] is not None:
                    global_features.append(qm_data['homo_lumo_gap'])
                else:
                    global_features.append(0.0)
                
                # Add dipole moment if available
                if 'dipole_moment' in qm_data and qm_data['dipole_moment'] is not None:
                    global_features.extend(qm_data['dipole_moment'])  # [dx, dy, dz, total]
                else:
                    global_features.extend([0.0, 0.0, 0.0, 0.0])
                
                data['global_features'] = torch.tensor(global_features, dtype=torch.float)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating graph for {mol_id}: {str(e)}")
            return None
    
    def smiles_to_graph(self, smiles: str, mol_id: str = None) -> Dict:
        """
        Convert a SMILES string to a graph representation.
        
        Args:
            smiles: SMILES string
            mol_id: Molecule identifier
            
        Returns:
            Dictionary with graph data
        """
        # Validate SMILES
        is_valid, canonical_smiles, error = validate_smiles(smiles)
        if not is_valid:
            logger.error(f"Invalid SMILES: {smiles}, Error: {error}")
            return None
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            logger.error(f"Failed to create molecule from SMILES: {smiles}")
            return None
        
        # Add hydrogen atoms
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates if requested
        if self.use_3d_coords:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                logger.warning(f"Failed to generate 3D coordinates for {smiles}: {str(e)}")
        
        # Convert to graph
        return self.mol_to_graph(mol, mol_id)
    
    def molfile_to_graph(self, mol_file: str, qm_file: str = None, charge_type: str = 'mulliken') -> Dict:
        """
        Convert a molecule file to a graph representation, optionally with QM data.
        
        Args:
            mol_file: Path to molecule file (MOL format)
            qm_file: Path to quantum mechanical data file (ORCA output)
            charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
            
        Returns:
            Dictionary with graph data
        """
        # Get molecule ID from filename
        mol_id = os.path.basename(mol_file).split('.')[0]
        
        # Load molecule
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        if mol is None:
            logger.error(f"Failed to load molecule from {mol_file}")
            return None
        
        # Load QM data if available
        qm_data = None
        if qm_file and os.path.exists(qm_file):
            try:
                qm_data = parse_orca_output(qm_file)
            except Exception as e:
                logger.warning(f"Failed to parse ORCA output {qm_file}: {str(e)}")
        
        # Convert to graph
        return self.mol_to_graph(mol, mol_id, qm_data)


#-------------------------------------
# Graph Generation Functions
#-------------------------------------

def create_graph_from_smiles(
    smiles: str, 
    output_file: str = None, 
    mol_id: str = None,
    use_pfas_features: bool = True
) -> Dict:
    """
    Create a molecular graph from a SMILES string.
    
    Args:
        smiles: SMILES string
        output_file: Path to save graph
        mol_id: Molecule identifier
        use_pfas_features: Whether to use PFAS-specific features
        
    Returns:
        Dictionary with graph data
    """
    generator = MolecularGraphGenerator(use_pfas_features=use_pfas_features)
    graph_data = generator.smiles_to_graph(smiles, mol_id)
    
    if graph_data is not None and output_file:
        torch.save(graph_data, output_file)
    
    return graph_data


def create_graph_from_orca(
    mol_file: str, 
    orca_file: str, 
    output_file: str = None,
    charge_type: str = 'mulliken',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True
) -> Dict:
    """
    Create a molecular graph from a molecule file and ORCA output file.
    
    Args:
        mol_file: Path to molecule file (MOL format)
        orca_file: Path to ORCA output file
        output_file: Path to save graph
        charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
        use_pfas_features: Whether to use PFAS-specific features
        use_quantum_properties: Whether to use quantum mechanical properties
        
    Returns:
        Dictionary with graph data
    """
    generator = MolecularGraphGenerator(use_pfas_features=use_pfas_features)
    qm_file = orca_file if use_quantum_properties else None
    graph_data = generator.molfile_to_graph(mol_file, qm_file, charge_type)
    
    if graph_data is not None and output_file:
        torch.save(graph_data, output_file)
    
    return graph_data


def batch_create_graphs_from_orca(
    mol_dir: str, 
    orca_dir: str, 
    output_dir: str,
    charge_type: str = 'mulliken',
    use_pfas_features: bool = True,
    use_quantum_properties: bool = True,
    max_workers: int = 4,
    batch_size: int = 32,
    cache_file: str = None
) -> List[str]:
    """
    Batch create molecular graphs from molecule files and ORCA outputs.
    
    Args:
        mol_dir: Directory with molecule files
        orca_dir: Directory with ORCA output files
        output_dir: Directory to save graph files
        charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
        use_pfas_features: Whether to use PFAS-specific features
        use_quantum_properties: Whether to use quantum mechanical properties
        max_workers: Maximum number of workers for parallel processing
        batch_size: Size of batches for processing
        cache_file: Path to cache file for molecule-ORCA file mapping
        
    Returns:
        List of paths to generated graph files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    logger.info(f"Batch creating graphs from {mol_dir} and {orca_dir}")
    
    # Get molecule files
    mol_files = glob.glob(os.path.join(mol_dir, "*.mol"))
    if not mol_files:
        logger.warning(f"No molecule files found in {mol_dir}")
        return []
    
    # Create molecule ID to file mapping for faster lookups
    molecule_mapping = {}
    orca_outputs = {}
    
    # Try to load from cache if available
    cache_loaded = False
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                molecule_mapping = cached_data.get('molecule_mapping', {})
                orca_outputs = cached_data.get('orca_outputs', {})
                cache_loaded = True
                logger.info(f"Loaded molecule mapping from cache: {len(molecule_mapping)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache file: {str(e)}")
    
    # Create mapping if not loaded from cache
    if not cache_loaded:
        logger.info("Creating molecule ID to file mapping...")
        for mol_file in mol_files:
            mol_id = os.path.basename(mol_file).split('.')[0]
            molecule_mapping[mol_id] = mol_file
            
            # Try to find matching ORCA output
            orca_file = os.path.join(orca_dir, f"{mol_id}.out")
            if os.path.exists(orca_file):
                orca_outputs[mol_id] = orca_file
        
        # Save to cache if requested
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'molecule_mapping': molecule_mapping,
                        'orca_outputs': orca_outputs
                    }, f)
                logger.info(f"Saved molecule mapping to cache: {len(molecule_mapping)} entries")
            except Exception as e:
                logger.warning(f"Failed to save cache file: {str(e)}")
    
    # Match molecule files with ORCA outputs
    matched_files = []
    for mol_id, mol_file in molecule_mapping.items():
        if mol_id in orca_outputs:
            matched_files.append((mol_id, mol_file, orca_outputs[mol_id]))
    
    logger.info(f"Found {len(matched_files)} matching molecule-ORCA file pairs")
    
    # Skip already processed files
    existing_graphs = set()
    for file in os.listdir(output_dir):
        if file.endswith('.pt'):
            existing_graphs.add(file.split('.')[0])
    
    to_process = []
    for mol_id, mol_file, orca_file in matched_files:
        if mol_id not in existing_graphs:
            to_process.append((mol_id, mol_file, orca_file))
    
    logger.info(f"Processing {len(to_process)} new files")
    
    # Create batches for more efficient processing
    batches = [to_process[i:i+batch_size] for i in range(0, len(to_process), batch_size)]
    
    generated_files = []
    
    # Process in parallel if max_workers > 1
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(
                    _process_graph_batch,
                    batch,
                    output_dir,
                    charge_type,
                    use_pfas_features,
                    use_quantum_properties,
                    batch_idx
                )
                future_to_batch[future] = batch_idx
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_files = future.result()
                    generated_files.extend(batch_files)
                    logger.info(f"Completed batch {batch_idx+1}/{len(batches)} ({len(batch_files)} graphs)")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
    else:
        # Process sequentially
        for batch_idx, batch in enumerate(batches):
            try:
                batch_files = _process_graph_batch(
                    batch,
                    output_dir,
                    charge_type,
                    use_pfas_features,
                    use_quantum_properties,
                    batch_idx
                )
                generated_files.extend(batch_files)
                logger.info(f"Completed batch {batch_idx+1}/{len(batches)} ({len(batch_files)} graphs)")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
    
    # Add existing files to output list
    for mol_id in existing_graphs:
        graph_file = os.path.join(output_dir, f"{mol_id}.pt")
        if graph_file not in generated_files and os.path.exists(graph_file):
            generated_files.append(graph_file)
    
    # Report statistics
    total_time = time.time() - start_time
    logger.info(f"Generated {len(generated_files)} graphs in {total_time:.2f} seconds")
    if generated_files:
        logger.info(f"Average time per graph: {total_time/len(generated_files):.2f} seconds")
    
    return generated_files


def _process_graph_batch(
    batch: List[Tuple[str, str, str]],
    output_dir: str,
    charge_type: str,
    use_pfas_features: bool,
    use_quantum_properties: bool,
    batch_idx: int
) -> List[str]:
    """
    Process a batch of graphs (helper function for parallelization).
    
    Args:
        batch: List of (mol_id, mol_file, orca_file) tuples
        output_dir: Directory to save graph files
        charge_type: Type of partial charges to use
        use_pfas_features: Whether to use PFAS-specific features
        use_quantum_properties: Whether to use quantum mechanical properties
        batch_idx: Batch index for logging
        
    Returns:
        List of paths to generated graph files
    """
    generated_files = []
    
    # Process each file in the batch
    for mol_id, mol_file, orca_file in batch:
        output_file = os.path.join(output_dir, f"{mol_id}.pt")
        
        try:
            # Create graph
            graph_data = create_graph_from_orca(
                mol_file,
                orca_file,
                output_file,
                charge_type,
                use_pfas_features,
                use_quantum_properties
            )
            
            if graph_data is not None:
                generated_files.append(output_file)
        except Exception as e:
            logger.error(f"Error creating graph for {mol_id}: {str(e)}")
    
    return generated_files 