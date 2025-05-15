"""
Molecular Graph Representation for PFAS Molecules

This module provides functionality to convert PFAS molecules into graph
representations suitable for graph neural networks using PyTorch Geometric.
It also supports JSON-based graph representations for interoperability.
"""

import os
import json
import torch
import logging
import glob # Added import
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import QED
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import concurrent.futures


from moml.core.molecular_feature_extraction import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor
)

# Configure logging
logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    'MolecularGraphProcessor',
    
    # Factory functions
    'create_graph_processor',
    
    # Utility functions
    'mol_file_to_graph', # Note: This is a standalone, consider if it should use MolecularGraphProcessor
    'graph_to_device',
    'collate_graphs',
    'find_charges_file',
    'read_charges_from_file',
    'create_molecular_graph_json', # Note: This is a standalone
    'batch_create_graphs_from_molecules' # Note: This is a standalone
]

class MolecularGraphProcessor:
    """
    A comprehensive class for processing molecular graphs.
    
    This class handles the creation and processing of molecular graphs from different
    input formats, providing a consistent interface for graph operations used in MGNN.
    """
    
    # Define ATOM_FEATURES at the class level for reference by atom_feature_dim property
    # and _get_atom_features method.
    ATOM_FEATURES_DEFAULTS = {
        'atomic_num': [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, -1],  # Common non-metals + F, Cl, Br, I, Other
        'degree': [0, 1, 2, 3, 4, 5, 6, -1],  # Max 6, Other
        'formal_charge': [-2, -1, 0, 1, 2, -999],  # Common charges, Other
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            -1 # Other
        ],
        'is_aromatic': [0, 1], # False, True
        'is_in_ring': [0, 1], # False, True
        # Numerical features (not one-hot)
        'num_hydrogens': None, # Placeholder, will be a single value
        'is_fluorine': None, # Placeholder, will be 0 or 1
        'is_carbon_bonded_to_fluorine': None, # Placeholder, will be 0 or 1
        'num_fluorine_neighbors': None, # Placeholder
        'is_in_carboxylic_group': None,
        'is_in_sulfonic_group': None,
        'is_in_phosphonic_group': None,
        'partial_charge': None, # If use_partial_charges is True
        'dist_to_cf3': None, # If use_pfas_specific_features and use_3d_coords
        'dist_to_functional_group': None, # If use_pfas_specific_features and use_3d_coords
        'is_head_group_atom': None, # If use_pfas_specific_features and use_3d_coords
        'homo_contribution': None, # If additional_features provided
        'lumo_contribution': None, # If additional_features provided
    }

    BOND_FEATURES_DEFAULTS = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            -1 # Other
        ],
        'is_conjugated': [0, 1], # False, True
        'is_in_ring': [0, 1], # False, True
        # Numerical features
        'is_cf_bond': None,
        'is_cf_cf_bond': None, # If use_pfas_specific_features
        'is_fluorinated_tail_bond': None, # If use_pfas_specific_features
        'is_functional_group_bond': None, # If use_pfas_specific_features
        'bond_length': None, # If use_3d_coords
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MolecularGraphProcessor.
        
        Args:
            config: Configuration dictionary with processing parameters.
                    Expected keys:
                    - 'atom_feature_schemes': List of atom feature names to include.
                                              If None, all default features are used.
                    - 'bond_feature_schemes': List of bond feature names to include.
                                              If None, all default features are used.
                    - 'use_partial_charges': bool (default: True)
                    - 'use_3d_coords': bool (default: True)
                    - 'use_pfas_specific_features': bool (default: True)
        """
        self.config = config or {}
        self.use_partial_charges = self.config.get('use_partial_charges', True)
        self.use_3d_coords = self.config.get('use_3d_coords', True)
        self.use_pfas_specific_features = self.config.get('use_pfas_specific_features', True)
        
        # Determine active feature schemes
        self.atom_feature_schemes = self.config.get('atom_feature_schemes')
        if self.atom_feature_schemes is None: # Use all defined defaults if not specified
            self.atom_feature_schemes = list(self.ATOM_FEATURES_DEFAULTS.keys())

        self.bond_feature_schemes = self.config.get('bond_feature_schemes')
        if self.bond_feature_schemes is None:
            self.bond_feature_schemes = list(self.BOND_FEATURES_DEFAULTS.keys())

        self.feature_extractor = MolecularFeatureExtractor()
        self.functional_group_detector = FunctionalGroupDetector()

    @property
    def atom_feature_dim(self) -> int:
        """Dynamically calculate atom feature dimension based on active schemes."""
        dim = 0
        for scheme in self.atom_feature_schemes:
            if scheme not in self.ATOM_FEATURES_DEFAULTS:
                logger.warning(f"Atom feature scheme '{scheme}' not recognized. Skipping.")
                continue
            
            choices = self.ATOM_FEATURES_DEFAULTS[scheme]
            if isinstance(choices, list): # One-hot encoded
                dim += len(choices)
            else: # Numerical feature
                # Conditional features based on config
                if scheme == 'partial_charge' and not self.use_partial_charges:
                    continue
                if scheme in ['dist_to_cf3', 'dist_to_functional_group', 'is_head_group_atom']:
                    if not (self.use_pfas_specific_features and self.use_3d_coords):
                        continue
                # These are other PFAS-specific numerical features
                if scheme in ['num_fluorine_neighbors', 'is_in_carboxylic_group',
                              'is_in_sulfonic_group', 'is_in_phosphonic_group']:
                    if not self.use_pfas_specific_features:
                        continue
                
                # For homo/lumo, count them if the scheme is active.
                # The actual generation in _get_atom_features depends on additional_features.
                # To make atom_feature_dim reflect the baseline dimension (no additional_features),
                # we should not count them here by default.
                # Tests for additional_features will verify the increased dimension separately.
                if scheme in ['homo_contribution', 'lumo_contribution']:
                    # These are only added if additional_features provide them.
                    # So, for the baseline dimension, they are not counted.
                    continue
                dim += 1
        return dim

    @property
    def bond_feature_dim(self) -> int:
        """Dynamically calculate bond feature dimension based on active schemes."""
        dim = 0
        for scheme in self.bond_feature_schemes:
            if scheme not in self.BOND_FEATURES_DEFAULTS:
                logger.warning(f"Bond feature scheme '{scheme}' not recognized. Skipping.")
                continue

            choices = self.BOND_FEATURES_DEFAULTS[scheme]
            if isinstance(choices, list): # One-hot encoded
                dim += len(choices)
            else: # Numerical feature
                if scheme == 'bond_length' and not self.use_3d_coords:
                    continue
                if scheme in ['is_cf_cf_bond', 'is_fluorinated_tail_bond', 'is_functional_group_bond'] and \
                   not self.use_pfas_specific_features:
                    continue
                dim += 1
        return dim
    
    @staticmethod
    def _one_hot_encoding(value: Any, choices: List[Any]) -> List[int]:
        """
        Create a one-hot encoding of a value from a list of choices.
        
        Args:
            value: The value to encode
            choices: List of possible values
            
        Returns:
            One-hot encoded list
        """
        encoding = [0] * len(choices)
        try:
            idx = choices.index(value)
            encoding[idx] = 1
        except ValueError:
            # If value not in choices, check if -1 or -999 (Other) is the last choice
            if choices and choices[-1] in [-1, -999]: # Check if 'Other' category exists
                encoding[-1] = 1 # Encode as 'Other'
            # else: logger.debug(f"Value {value} not in choices {choices}, not using 'Other' encoding.")
        return encoding

    def _is_in_carboxylic_group(self, atom: Chem.Atom) -> bool:
        return self.functional_group_detector.is_in_carboxylic_group(atom)
    
    def _is_in_sulfonic_group(self, atom: Chem.Atom) -> bool:
        return self.functional_group_detector.is_in_sulfonic_group(atom)

    def _is_in_phosphonic_group(self, atom: Chem.Atom) -> bool:
        return self.functional_group_detector.is_in_phosphonic_group(atom)

    def _get_atom_features(self, atom: Chem.Atom, mol: Chem.Mol,
                           partial_charge_val: Optional[float] = None, 
                           distance_features_map: Optional[Dict[int, Dict[str, float]]] = None,
                           homo_lumo_contrib_val: Optional[List[float]] = None) -> List[float]:
        """
        Generate a feature vector for an atom based on active feature schemes.
        """
        features = []
        atom_idx = atom.GetIdx()

        # PFAS basic calculations (needed by some schemes)
        is_f_atom = atom.GetAtomicNum() == 9
        is_cf_atom = False
        num_f_neighbors_atom = 0
        if atom.GetAtomicNum() == 6:  # Carbon
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 9:
                    is_cf_atom = True
                    num_f_neighbors_atom += 1
        
        atom_dist_feats = distance_features_map.get(atom_idx, {}) if distance_features_map else {}

        for scheme in self.atom_feature_schemes:
            if scheme == 'atomic_num':
                features.extend(self._one_hot_encoding(atom.GetAtomicNum(), self.ATOM_FEATURES_DEFAULTS['atomic_num']))
            elif scheme == 'degree':
                features.extend(self._one_hot_encoding(atom.GetDegree(), self.ATOM_FEATURES_DEFAULTS['degree']))
            elif scheme == 'formal_charge':
                features.extend(self._one_hot_encoding(atom.GetFormalCharge(), self.ATOM_FEATURES_DEFAULTS['formal_charge']))
            elif scheme == 'hybridization':
                features.extend(self._one_hot_encoding(atom.GetHybridization(), self.ATOM_FEATURES_DEFAULTS['hybridization']))
            elif scheme == 'is_aromatic':
                features.extend(self._one_hot_encoding(1 if atom.GetIsAromatic() else 0, self.ATOM_FEATURES_DEFAULTS['is_aromatic']))
            elif scheme == 'is_in_ring':
                features.extend(self._one_hot_encoding(1 if atom.IsInRing() else 0, self.ATOM_FEATURES_DEFAULTS['is_in_ring']))
            elif scheme == 'num_hydrogens':
                features.append(atom.GetTotalNumHs())
            elif scheme == 'is_fluorine':
                features.append(float(is_f_atom))
            elif scheme == 'is_carbon_bonded_to_fluorine':
                features.append(float(is_cf_atom))
            elif scheme == 'num_fluorine_neighbors':
                 if self.use_pfas_specific_features: # This scheme implies use_pfas_specific_features
                    features.append(float(num_f_neighbors_atom))
            elif scheme == 'is_in_carboxylic_group':
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_carboxylic_group(atom)))
            elif scheme == 'is_in_sulfonic_group':
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_sulfonic_group(atom)))
            elif scheme == 'is_in_phosphonic_group':
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_phosphonic_group(atom)))
            elif scheme == 'partial_charge':
                if self.use_partial_charges and partial_charge_val is not None:
                    features.append(partial_charge_val)
            elif scheme == 'dist_to_cf3':
                if self.use_pfas_specific_features and self.use_3d_coords:
                    features.append(float(atom_dist_feats.get('dist_to_cf3', -1.0)))
            elif scheme == 'dist_to_functional_group':
                if self.use_pfas_specific_features and self.use_3d_coords:
                    features.append(float(atom_dist_feats.get('dist_to_functional', -1.0)))
            elif scheme == 'is_head_group_atom':
                if self.use_pfas_specific_features and self.use_3d_coords: # Assuming head group def needs 3D context
                    features.append(float(atom_dist_feats.get('is_head_group', 0.0)))
            elif scheme == 'homo_contribution':
                if homo_lumo_contrib_val is not None and len(homo_lumo_contrib_val) > 0:
                    features.append(homo_lumo_contrib_val[0])
            elif scheme == 'lumo_contribution':
                if homo_lumo_contrib_val is not None and len(homo_lumo_contrib_val) > 1:
                    features.append(homo_lumo_contrib_val[1])
            # else: logger.debug(f"Atom feature scheme '{scheme}' defined in ATOM_FEATURES_DEFAULTS but not handled in _get_atom_features.")
        return features

    def _get_bond_features(self, bond: Chem.Bond, bond_lengths_map: Optional[Dict[Tuple[int, int], float]] = None) -> List[float]:
        """
        Generate a feature vector for a bond based on active feature schemes.
        """
        features = []
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # PFAS basic calculations
        is_cf_bond_val = (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 9) or \
                         (begin_atom.GetAtomicNum() == 9 and end_atom.GetAtomicNum() == 6)
        
        is_cf_cf_bond_val = False
        if self.use_pfas_specific_features and begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 6:
            begin_f_count = sum(1 for n in begin_atom.GetNeighbors() if n.GetAtomicNum() == 9)
            end_f_count = sum(1 for n in end_atom.GetNeighbors() if n.GetAtomicNum() == 9)
            is_cf_cf_bond_val = begin_f_count > 0 and end_f_count > 0
        
        is_fluorinated_tail_bond_val = is_cf_bond_val or is_cf_cf_bond_val if self.use_pfas_specific_features else False
        
        is_func_group_bond_val = False
        if self.use_pfas_specific_features:
            is_func_group_bond_val = (self._is_in_carboxylic_group(begin_atom) or 
                                      self._is_in_carboxylic_group(end_atom) or
                                      self._is_in_sulfonic_group(begin_atom) or
                                      self._is_in_sulfonic_group(end_atom) or
                                      self._is_in_phosphonic_group(begin_atom) or
                                      self._is_in_phosphonic_group(end_atom))

        bond_len_val = None
        if self.use_3d_coords and bond_lengths_map:
            bond_len_val = bond_lengths_map.get((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if bond_len_val is None: # Check reverse tuple if not found
                 bond_len_val = bond_lengths_map.get((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))


        for scheme in self.bond_feature_schemes:
            if scheme == 'bond_type':
                features.extend(self._one_hot_encoding(bond.GetBondType(), self.BOND_FEATURES_DEFAULTS['bond_type']))
            elif scheme == 'is_conjugated':
                features.extend(self._one_hot_encoding(1 if bond.GetIsConjugated() else 0, self.BOND_FEATURES_DEFAULTS['is_conjugated']))
            elif scheme == 'is_in_ring':
                features.extend(self._one_hot_encoding(1 if bond.IsInRing() else 0, self.BOND_FEATURES_DEFAULTS['is_in_ring']))
            elif scheme == 'is_cf_bond':
                features.append(float(is_cf_bond_val))
            elif scheme == 'is_cf_cf_bond':
                if self.use_pfas_specific_features:
                    features.append(float(is_cf_cf_bond_val))
            elif scheme == 'is_fluorinated_tail_bond':
                if self.use_pfas_specific_features:
                    features.append(float(is_fluorinated_tail_bond_val))
            elif scheme == 'is_functional_group_bond':
                if self.use_pfas_specific_features:
                    features.append(float(is_func_group_bond_val))
            elif scheme == 'bond_length':
                if self.use_3d_coords and bond_len_val is not None:
                    features.append(bond_len_val)
            # else: logger.debug(f"Bond feature scheme '{scheme}' defined but not handled.")
        return features

    def mol_to_graph(self, 
                    mol: Chem.Mol, 
                    additional_features: Optional[Dict[str, List[float]]] = None) -> Data:
        if self.use_3d_coords and mol.GetNumConformers() == 0:
            # Try to generate 3D conformer if missing
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                AllChem.UFFOptimizeMolecule(mol)
                if mol.GetNumConformers() == 0: # Check again
                    raise ValueError("Failed to generate 3D coordinates for molecule.")
            except Exception as e:
                logger.error(f"Error generating 3D conformer: {e}. Proceeding without 3D if possible or features might be missing.")
                # If use_3d_coords is True but we fail, subsequent features might be missing.
                # This could be handled by turning off 3D-dependent features for this molecule,
                # or by raising the error if 3D is strictly required.
                # For now, let it proceed, features requiring 3D will be absent.
                # raise ValueError(f"Molecule does not have 3D coordinates, and generation failed, but use_3d_coords is True: {e}")


        partial_charges_list = additional_features.get('partial_charges') if additional_features else None
        homo_contribs = additional_features.get('homo_contributions') if additional_features else None
        lumo_contribs = additional_features.get('lumo_contributions') if additional_features else None
        
        bond_lengths_map = self.feature_extractor.calculate_bond_lengths(mol) if self.use_3d_coords and mol.GetNumConformers() > 0 else {}
        
        distance_features_map = None
        if self.use_pfas_specific_features and self.use_3d_coords and mol.GetNumConformers() > 0:
            distance_features_map = self.feature_extractor.calculate_distance_features(mol)
        
        x_features = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pc = partial_charges_list[i] if partial_charges_list and i < len(partial_charges_list) else None
            hlc = None
            if homo_contribs and i < len(homo_contribs) and lumo_contribs and i < len(lumo_contribs):
                hlc = [homo_contribs[i], lumo_contribs[i]]
            
            atom_feats = self._get_atom_features(atom, mol, pc, distance_features_map, hlc)
            x_features.append(atom_feats)
        
        x = torch.tensor(x_features, dtype=torch.float) if x_features else torch.empty((0, self.atom_feature_dim), dtype=torch.float)

        edge_indices = []
        edge_attrs_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feats = self._get_bond_features(bond, bond_lengths_map)
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs_list.extend([bond_feats, bond_feats])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float) if edge_attrs_list else torch.empty((0, self.bond_feature_dim), dtype=torch.float)

        pos = None
        if self.use_3d_coords and mol.GetNumConformers() > 0:
            conformer = mol.GetConformer()
            positions = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
            pos = torch.tensor([[p.x, p.y, p.z] for p in positions], dtype=torch.float)
        
        # Add other graph-level properties from config if needed
        data_dict = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'num_nodes': mol.GetNumAtoms()}
        if pos is not None:
            data_dict['pos'] = pos
        
        # Example of adding a graph-level label if provided (e.g. for training)
        if additional_features and 'label' in additional_features:
            data_dict['y'] = torch.tensor([additional_features['label']], dtype=torch.float)

        return Data(**data_dict)

    def file_to_graph(self, file_path: str, additional_features: Optional[Dict[str, List[float]]] = None) -> Optional[Data]:
        if not os.path.exists(file_path):
            # logger.error(f"Molecule file not found: {file_path}") # Logging can be removed if raising
            raise FileNotFoundError(f"Molecule file not found: {file_path}")
        try:
            if file_path.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(file_path, removeHs=False)
                mol = next(iter(suppl), None)
            elif file_path.endswith('.mol2'):
                mol = Chem.MolFromMol2File(file_path, removeHs=False)
            elif file_path.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(file_path, removeHs=False)
            elif file_path.endswith('.mol'): # Added .mol handling
                mol = Chem.MolFromMolFile(file_path, removeHs=False)
            else:
                logger.error(f"Unsupported molecule file format: {file_path}")
                return None

            if mol is None:
                logger.error(f"Failed to read molecule from {file_path}")
                return None
            
            return self.mol_to_graph(mol, additional_features)
        except Exception as e:
            logger.error(f"Error processing molecule file {file_path}: {e}")
            return None

    def smiles_to_graph(self, smiles: str, additional_features: Optional[Dict[str, List[float]]] = None) -> Optional[Data]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES string: {smiles}")
                return None
            # Generate 3D coordinates for SMILES
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol) # Optionally remove Hs after embedding if not needed for features
            return self.mol_to_graph(mol, additional_features)
        except Exception as e:
            logger.error(f"Error processing SMILES string {smiles}: {e}")
            return None

    # Batch processing can be added here if needed, e.g. batch_files_to_graphs
    # For now, datasets will call file_to_graph or smiles_to_graph iteratively.

    def mol_to_json_graph(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Converts an RDKit molecule to a JSON-serializable graph dictionary."""
        # This is a simplified JSON representation, not directly a PyG Data object.
        # For full PyG Data serialization, torch.save is typically used.
        graph_dict = {
            "nodes": [],
            "edges": [],
            "descriptors": self._get_molecule_descriptors(mol)
        }

        bond_lengths_map = self.feature_extractor.calculate_bond_lengths(mol) if self.use_3d_coords and mol.GetNumConformers() > 0 else {}

        for atom in mol.GetAtoms():
            # Using a simplified feature set for JSON for readability
            atom_data = {
                "id": atom.GetIdx(),
                "atomic_num": atom.GetAtomicNum(),
                "symbol": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "num_hs": atom.GetTotalNumHs()
            }
            if self.use_3d_coords and mol.GetNumConformers() > 0:
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                atom_data["pos"] = {"x": pos.x, "y": pos.y, "z": pos.z}
            graph_dict["nodes"].append(atom_data)

        for bond in mol.GetBonds():
            bond_data = {
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()),
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing()
            }
            if self.use_3d_coords and bond_lengths_map:
                 length = bond_lengths_map.get((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                 if length is None: length = bond_lengths_map.get((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
                 if length is not None: bond_data["length"] = length
            graph_dict["edges"].append(bond_data)
        
        return graph_dict

    def _get_molecule_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Helper to get some basic molecular descriptors."""
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "num_h_donors": Lipinski.NumHDonors(mol),
            "num_h_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "tpsa": Descriptors.TPSA(mol)
        }

    def file_to_json_graph(self, file_path: str, output_dir: Optional[str] = None, output_filename: Optional[str] = None) -> Optional[str]:
        """Reads a molecule file, converts to JSON graph, and saves it."""
        try:
            mol: Optional[Chem.Mol] = None # Ensure mol is defined before the block
            if not os.path.exists(file_path): # Check existence first
                logger.error(f"Molecule file not found for JSON conversion: {file_path}")
                return None

            if file_path.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=True)
                mol = next(iter(suppl), None)
            elif file_path.endswith('.mol2'):
                mol = Chem.MolFromMol2File(file_path, removeHs=False, sanitize=True)
            elif file_path.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=True)
            elif file_path.endswith('.mol'): # Add .mol support
                mol = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=True)
            else:
                logger.error(f"Unsupported molecule file format for JSON conversion: {file_path}")
                return None

            if mol is None:
                logger.error(f"Failed to read RDKit molecule from {file_path} for JSON conversion.")
                return None

            graph_dict = self.mol_to_json_graph(mol)

            if output_dir and output_filename:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(graph_dict, f, indent=2)
                logger.info(f"Saved JSON graph to {output_path}")
                return output_path
            # If output_dir/filename not provided, could return dict, but signature implies path
            logger.warning("output_dir or output_filename not provided for file_to_json_graph, JSON not saved.")
            return None # Or return graph_dict if that's useful
        except Exception as e:
            logger.error(f"Error in file_to_json_graph for {file_path}: {e}")
            return None

# Factory function
def create_graph_processor(config: Dict[str, Any] = None) -> MolecularGraphProcessor:
    """
    Factory function to create a MolecularGraphProcessor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MolecularGraphProcessor instance
    """
    return MolecularGraphProcessor(config=config)


# Standalone utility functions (might be deprecated or refactored to use MolecularGraphProcessor)

def mol_file_to_graph(mol_file_path: str,
                      use_partial_charges: bool = True,
                      charges_file_path: Optional[str] = None,
                      use_3d_coords: bool = True,
                      use_pfas_specific_features: bool = True,
                      config: Optional[Dict[str, Any]] = None) -> Optional[Data]:
    """
    Converts a molecule file (SDF, MOL2, PDB) to a PyTorch Geometric Data object.
    This is a standalone version. Consider using MolecularGraphProcessor class for more features.
    """
    # This standalone function should ideally use an instance of MolecularGraphProcessor
    # to ensure consistency with feature generation.
    # For now, it replicates some logic, which is not ideal.
    
    # Default config if none provided
    effective_config = config or {}
    effective_config.setdefault('use_partial_charges', use_partial_charges)
    effective_config.setdefault('use_3d_coords', use_3d_coords)
    effective_config.setdefault('use_pfas_specific_features', use_pfas_specific_features)
    
    processor = MolecularGraphProcessor(config=effective_config)
    
    additional_processor_features = {}
    if use_partial_charges and charges_file_path:
        qm_charges = read_charges_from_file(charges_file_path)
        if qm_charges:
            additional_processor_features['partial_charges'] = qm_charges
            
    return processor.file_to_graph(mol_file_path, additional_features=additional_processor_features)


def graph_to_device(graph: Union[Data, Dict[str, torch.Tensor]], device: torch.device) -> Union[Data, Dict[str, torch.Tensor]]:
    """Moves a graph or dictionary of tensors to the specified device."""
    if isinstance(graph, Data):
        return graph.to(device)
    elif isinstance(graph, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in graph.items()}
    return graph # Or raise error

def collate_graphs(graphs: List[Data]) -> Data:
    """Collates a list of PyG Data objects into a single Batch object."""
    from torch_geometric.loader import DataLoader # Local import to avoid circularity if Batch was imported globally
    # DataLoader can collate. A bit heavy if just for collation.
    # PyG Batch.from_data_list is the standard way.
    from torch_geometric.data import Batch
    if not graphs:
        return Batch() # Return an empty batch
    return Batch.from_data_list(graphs)


def find_charges_file(mol_file: str, charges_dir: str) -> Optional[str]:
    """
    Finds a corresponding .charges file for a given molecule file.
    Assumes charge file has the same base name but with .charges extension.
    """
    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    charge_file = os.path.join(charges_dir, base_name + ".charges")
    if os.path.exists(charge_file):
        return charge_file
    
    # Try .txt as well, as per ORCA output
    charge_file_txt = os.path.join(charges_dir, base_name + ".txt")
    if os.path.exists(charge_file_txt):
        return charge_file_txt
        
    logger.warning(f"No .charges or .txt file found for {mol_file} in {charges_dir}")
    return None

def read_charges_from_file(charge_file: str) -> Optional[List[float]]:
    """
    Reads partial charges from a file.
    Supports simple text files with one charge per line, or ORCA .prop files.
    """
    if not os.path.exists(charge_file):
        # logger.error(f"Charge file not found: {charge_file}") # Test expects FileNotFoundError
        raise FileNotFoundError(f"Charges file not found: {charge_file}")
    
    charges = []
    try:
        if charge_file.endswith(".json"):
            with open(charge_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                charges = [float(c) for c in data]
            elif isinstance(data, dict) and "esp_charges" in data:
                charges = [float(c) for c in data["esp_charges"]]
            else:
                logger.error(f"Unsupported JSON structure in {charge_file}")
                return None # Or raise ValueError
            return charges

        elif charge_file.endswith(".chg") or charge_file.endswith(".txt"):
            with open(charge_file, 'r') as f:
                lines = f.readlines()
            
            # Try ORCA .prop format (typically for .chg)
            in_orca_section = False
            parsed_orca_charges = []
            if charge_file.endswith(".chg") and (any("MULLIKEN ATOMIC CHARGES" in line for line in lines) or \
               any("LOEWDIN ATOMIC CHARGES" in line for line in lines)):
                for line in lines:
                    if "MULLIKEN ATOMIC CHARGES" in line or "LOEWDIN ATOMIC CHARGES" in line:
                        in_orca_section = True
                        continue
                    if in_orca_section and line.strip() == "":
                        in_orca_section = False
                        # Break here as ORCA charge section is typically contiguous and then ends.
                        # If simple list format follows, it will be caught by the next loop.
                        break
                    if in_orca_section and ":" in line:
                        try:
                            parts = line.split(':')
                            charge_val = float(parts[1].strip())
                            parsed_orca_charges.append(charge_val)
                        except (IndexError, ValueError):
                            logger.warning(f"Could not parse ORCA charge line: {line.strip()} in {charge_file}")
                            continue # Skip malformed line
                if parsed_orca_charges: # If ORCA parsing yielded results, use them
                    return parsed_orca_charges

            # If not ORCA, or ORCA parsing yielded no charges, or it's a .txt file, try simple list format
            # This re-iterates 'lines' which is fine.
            for line_content in lines:
                line_content = line_content.strip()
                if not line_content or line_content.startswith('#'): # Skip empty/comment
                    continue
                try:
                    charges.append(float(line_content))
                except ValueError:
                    logger.warning(f"Could not parse charge from line: '{line_content}' in file {charge_file}")
                    return None # Malformed simple format line
            # If file was empty or only comments, charges list will be empty.
            # For .txt, an empty list is a valid outcome if the file is empty.
            # For .chg, if ORCA parsing failed and simple parsing also yields empty, it's also an empty list.
            return charges
        
        else: # Should not be reached if extensions are .json, .chg, .txt
            logger.warning(f"File extension not explicitly handled by charge reading logic: {charge_file}")
            return None

    except Exception as e:
        logger.error(f"Error reading charge file {charge_file}: {e}")
        return None


def create_molecular_graph_json(mol_file: str, 
                                output_dir: str, 
                                charges_file: Optional[str] = None,
                                use_3d_coords: bool = True,
                                use_pfas_specific_features: bool = True,
                                config: Optional[Dict[str, Any]] = None
                                ) -> Optional[str]:
    """
    Creates a JSON representation of a molecular graph from a molecule file.
    This function is for generating a JSON file, not for creating a PyG Data object from JSON.
    """
    effective_config = config or {}
    effective_config.setdefault('use_3d_coords', use_3d_coords)
    effective_config.setdefault('use_pfas_specific_features', use_pfas_specific_features)
    # Partial charges are not directly used by file_to_json_graph, but by mol_to_graph if it were called.
    # This function primarily uses the MolecularGraphProcessor's file_to_json_graph method.

    processor = MolecularGraphProcessor(config=effective_config)
    
    if not os.path.exists(mol_file):
        logger.error(f"Input molecule file not found: {mol_file}")
        return None

    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    output_filename = f"{base_name}_graph.json"
    
    # The file_to_json_graph method of the processor handles reading the mol_file
    # and converting it to its JSON dict representation, then saving.
    # It does not currently take additional_features like charges directly for JSON output.
    # If charges need to be in the JSON, mol_to_json_graph would need modification.
    
    try:
        output_path = processor.file_to_json_graph(mol_file, output_dir, output_filename)
        return output_path
    except Exception as e:
        logger.error(f"Failed to create JSON graph for {mol_file}: {e}")
        return None


def batch_create_graphs_from_molecules(mol_dir: str, 
                                       output_dir: str, 
                                       file_format: str = 'sdf',
                                       use_3d_coords: bool = True,
                                       use_pfas_specific_features: bool = True,
                                       max_workers: Optional[int] = None,
                                       config: Optional[Dict[str, Any]] = None
                                       ) -> List[str]:
    """
    Processes multiple molecule files from a directory into PyG Data objects and saves them.
    Returns a list of paths to the saved .pt files.
    This is a standalone utility.
    """
    effective_config = config or {}
    effective_config.setdefault('use_3d_coords', use_3d_coords)
    effective_config.setdefault('use_pfas_specific_features', use_pfas_specific_features)
    processor = MolecularGraphProcessor(config=effective_config)

    if not os.path.isdir(mol_dir):
        logger.error(f"Molecule directory not found: {mol_dir}")
        return []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    mol_files = glob.glob(os.path.join(mol_dir, f"*.{file_format}"))
    if not mol_files:
        logger.warning(f"No .{file_format} files found in {mol_dir}")
        return []

    saved_graph_paths = []

    def process_file(mol_file_path):
        try:
            graph_data = processor.file_to_graph(mol_file_path) # additional_features can be passed here if available
            if graph_data:
                base_name = os.path.splitext(os.path.basename(mol_file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_graph.pt")
                torch.save(graph_data, output_path)
                return output_path
            return None
        except Exception as e:
            logger.error(f"Failed to process {mol_file_path}: {e}")
            return None

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_mol_file = {executor.submit(process_file, mol_file): mol_file for mol_file in mol_files}
        for future in concurrent.futures.as_completed(future_to_mol_file):
            mol_file_path_completed = future_to_mol_file[future]
            try:
                result_path = future.result()
                if result_path:
                    saved_graph_paths.append(result_path)
                    logger.info(f"Successfully processed and saved graph for {mol_file_path_completed} to {result_path}")
            except Exception as exc:
                logger.error(f'{mol_file_path_completed} generated an exception: {exc}')
    
    return saved_graph_paths
