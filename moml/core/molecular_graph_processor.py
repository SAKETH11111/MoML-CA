"""
moml/core/molecular_graph_processor.py

Molecular graph representation for PFAS molecules using PyTorch Geometric.
"""

import concurrent.futures
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Batch, Data
    HAS_TORCH_GEOMETRIC: bool
else:
    try:
        from torch_geometric.data import Batch, Data
        HAS_TORCH_GEOMETRIC = True
    except ImportError:
        HAS_TORCH_GEOMETRIC = False

        class Data:
            """Dummy Data when torch_geometric is not installed."""
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class Batch:
            """Dummy Batch when torch_geometric is not installed."""
            @staticmethod
            def from_data_list(data_list):
                return data_list

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, Descriptors, Lipinski, QED
HAS_RDKIT = True

# Local imports
from moml.core.molecular_feature_extraction import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor,
)

# Configure logging
logger = logging.getLogger(__name__)


def _process_single_mol_file_for_batch(
    args: Tuple[str, Optional[Dict[str, Any]], str]
) -> Optional[str]:
    """
    Process a single molecule file for batch processing.
    
    This function is designed to work with multiprocessing and handles
    the conversion of a single molecule file to a graph representation.
    
    Args:
        args: Tuple containing (mol_file_path, config_dict, output_dir_for_pt)
        
    Returns:
        Path to saved .pt file if successful, None otherwise
    """
    mol_file_path, config_dict, output_dir_for_pt = args
    try:
        graph_data = mol_file_to_graph(mol_file_path, config=config_dict)
        if graph_data:
            base_name = os.path.splitext(os.path.basename(mol_file_path))[0]
            output_pt_path = os.path.join(output_dir_for_pt, f"{base_name}.pt")
            torch.save(graph_data, output_pt_path)
            return output_pt_path
        else:
            logger.error(
                f"Graph generation failed for {mol_file_path}, not saving .pt file."
            )
            return None
    except Exception as e:
        logger.error(
            f"Error processing file {mol_file_path} in batch processing: {e}"
        )
        return None


__all__ = [
    "MolecularGraphProcessor",
    "create_graph_processor",
    "mol_file_to_graph",
    "graph_to_device",
    "collate_graphs",
    "find_charges_file",
    "read_charges_from_file",
    "create_molecular_graph_json",
    "batch_create_graphs_from_molecules",
]


class MolecularGraphProcessor:
    """
    Processes molecules into graph representations for machine learning.
    
    This class converts molecular structures into PyTorch Geometric Data objects
    with comprehensive feature extraction for atoms and bonds. It includes
    specialized handling for PFAS molecules with relevant chemical features.
    
    Attributes:
        ATOM_FEATURES_DEFAULTS: Default feature schemes for atom representations
        BOND_FEATURES_DEFAULTS: Default feature schemes for bond representations
    """
    
    ATOM_FEATURES_DEFAULTS = {
        "atomic_num": [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, -1],
        "degree": [0, 1, 2, 3, 4, 5, 6, -1],
        "formal_charge": [-2, -1, 0, 1, 2, -999],
        "hybridization": [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,
            rdchem.HybridizationType.UNSPECIFIED,
            -1,
        ],
        "is_aromatic": [0, 1],
        "is_in_ring": [0, 1],
        "num_hydrogens": None,
        "is_fluorine": None,
        "is_carbon_bonded_to_fluorine": None,
        "num_fluorine_neighbors": None,
        "is_in_carboxylic_group": None,
        "is_in_sulfonic_group": None,
        "is_in_phosphonic_group": None,
        "partial_charge": None,
        "dist_to_cf3": None,
        "dist_to_functional_group": None,
        "is_head_group_atom": None,
        "homo_contribution": None,
        "lumo_contribution": None,
    }

    BOND_FEATURES_DEFAULTS = {
        "bond_type": [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            -1,
        ],
        "is_conjugated": [0, 1],
        "is_in_ring": [0, 1],
        "is_cf_bond": None,
        "is_cf_cf_bond": None,
        "is_fluorinated_tail_bond": None,
        "is_functional_group_bond": None,
        "bond_length": None,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the MolecularGraphProcessor.
        
        Args:
            config: Configuration dictionary with processing options
            
        Raises:
            ImportError: If RDKit is not available
        """
        if not HAS_RDKIT:
            raise ImportError(
                "RDKit is required for MolecularGraphProcessor. "
                "Please install rdkit."
            )
        
        self.config = config or {}
        self.use_partial_charges = self.config.get("use_partial_charges", True)
        self.use_3d_coords = self.config.get("use_3d_coords", True)
        self.use_pfas_specific_features = self.config.get(
            "use_pfas_specific_features", True
        )

        self.atom_feature_schemes = (
            self.config.get("atom_feature_schemes") 
            or list(self.ATOM_FEATURES_DEFAULTS.keys())
        )
        self.bond_feature_schemes = (
            self.config.get("bond_feature_schemes") 
            or list(self.BOND_FEATURES_DEFAULTS.keys())
        )

        self.feature_extractor = MolecularFeatureExtractor()
        self.functional_group_detector = FunctionalGroupDetector()

    @property
    def atom_feature_dim(self) -> int:
        """
        Calculate the dimensionality of atom feature vectors.
        
        Returns:
            Integer representing the total feature dimension for atoms
        """
        dim = 0
        for scheme in self.atom_feature_schemes:
            if scheme not in self.ATOM_FEATURES_DEFAULTS:
                continue
            choices = self.ATOM_FEATURES_DEFAULTS[scheme]
            if isinstance(choices, list):
                dim += len(choices)
            else:
                if scheme == "partial_charge" and not self.use_partial_charges:
                    continue
                if scheme in [
                    "dist_to_cf3", 
                    "dist_to_functional_group", 
                    "is_head_group_atom"
                ] and not (self.use_pfas_specific_features and self.use_3d_coords):
                    continue
                if scheme in [
                    "num_fluorine_neighbors",
                    "is_in_carboxylic_group",
                    "is_in_sulfonic_group",
                    "is_in_phosphonic_group",
                ] and not self.use_pfas_specific_features:
                    continue
                if scheme in ["homo_contribution", "lumo_contribution"]:
                    continue
                dim += 1
        return dim

    @property
    def bond_feature_dim(self) -> int:
        """
        Calculate the dimensionality of bond feature vectors.
        
        Returns:
            Integer representing the total feature dimension for bonds
        """
        dim = 0
        for scheme in self.bond_feature_schemes:
            if scheme not in self.BOND_FEATURES_DEFAULTS:
                continue
            choices = self.BOND_FEATURES_DEFAULTS[scheme]
            if isinstance(choices, list):
                dim += len(choices)
            else:
                if scheme == "bond_length" and not self.use_3d_coords:
                    continue
                if scheme in [
                    "is_cf_cf_bond", 
                    "is_fluorinated_tail_bond", 
                    "is_functional_group_bond"
                ] and not self.use_pfas_specific_features:
                    continue
                dim += 1
        return dim

    @staticmethod
    def _one_hot_encoding(value: Any, choices: List[Any]) -> List[int]:
        """
        Create one-hot encoding for a given value.
        
        Args:
            value: Value to encode
            choices: List of possible values
            
        Returns:
            One-hot encoded list
        """
        encoding = [0] * len(choices)
        try:
            idx = choices.index(value)
            encoding[idx] = 1
        except ValueError:
            if choices and choices[-1] in [-1, -999]:
                encoding[-1] = 1
        return encoding

    def _is_in_carboxylic_group(self, atom: Chem.Atom) -> bool:
        """Check if atom is part of a carboxylic acid group."""
        return self.functional_group_detector.is_in_carboxylic_group(atom)

    def _is_in_sulfonic_group(self, atom: Chem.Atom) -> bool:
        """Check if atom is part of a sulfonic acid group."""
        return self.functional_group_detector.is_in_sulfonic_group(atom)

    def _is_in_phosphonic_group(self, atom: Chem.Atom) -> bool:
        """Check if atom is part of a phosphonic acid group."""
        return self.functional_group_detector.is_in_phosphonic_group(atom)

    def _get_atom_features(
        self,
        atom: Chem.Atom,
        mol: Chem.Mol,
        partial_charge_val: Optional[float] = None,
        distance_features_map: Optional[Dict[int, Dict[str, float]]] = None,
        homo_lumo_contrib_val: Optional[List[float]] = None,
    ) -> List[Union[int, float]]:
        """
        Extract comprehensive features for a single atom.
        
        Args:
            atom: RDKit Atom object
            mol: Parent molecule
            partial_charge_val: Optional partial charge value
            distance_features_map: Optional distance-based features
            homo_lumo_contrib_val: Optional HOMO/LUMO contributions
            
        Returns:
            List of extracted features (mixed int/float types)
        """
        features = []
        atom_idx = atom.GetIdx()
        is_f_atom = atom.GetAtomicNum() == 9
        is_cf_atom = False
        num_f_neighbors_atom = 0
        
        if atom.GetAtomicNum() == 6:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 9:
                    is_cf_atom = True
                    num_f_neighbors_atom += 1
                    
        atom_dist_feats = (
            distance_features_map.get(atom_idx, {}) 
            if distance_features_map 
            else {}
        )

        for scheme in self.atom_feature_schemes:
            if scheme == "atomic_num":
                features.extend(self._one_hot_encoding(atom.GetAtomicNum(), self.ATOM_FEATURES_DEFAULTS["atomic_num"]))
            elif scheme == "degree":
                features.extend(self._one_hot_encoding(atom.GetDegree(), self.ATOM_FEATURES_DEFAULTS["degree"]))
            elif scheme == "formal_charge":
                features.extend(
                    self._one_hot_encoding(atom.GetFormalCharge(), self.ATOM_FEATURES_DEFAULTS["formal_charge"])
                )
            elif scheme == "hybridization":
                features.extend(
                    self._one_hot_encoding(atom.GetHybridization(), self.ATOM_FEATURES_DEFAULTS["hybridization"])
                )
            elif scheme == "is_aromatic":
                features.extend(
                    self._one_hot_encoding(1 if atom.GetIsAromatic() else 0, self.ATOM_FEATURES_DEFAULTS["is_aromatic"])
                )
            elif scheme == "is_in_ring":
                features.extend(
                    self._one_hot_encoding(1 if atom.IsInRing() else 0, self.ATOM_FEATURES_DEFAULTS["is_in_ring"])
                )
            elif scheme == "num_hydrogens":
                features.append(atom.GetTotalNumHs())
            elif scheme == "is_fluorine":
                features.append(float(is_f_atom))
            elif scheme == "is_carbon_bonded_to_fluorine":
                features.append(float(is_cf_atom))
            elif scheme == "num_fluorine_neighbors":
                if self.use_pfas_specific_features:
                    features.append(float(num_f_neighbors_atom))
            elif scheme == "is_in_carboxylic_group":
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_carboxylic_group(atom)))
            elif scheme == "is_in_sulfonic_group":
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_sulfonic_group(atom)))
            elif scheme == "is_in_phosphonic_group":
                if self.use_pfas_specific_features:
                    features.append(float(self._is_in_phosphonic_group(atom)))
            elif scheme == "partial_charge":
                if self.use_partial_charges:
                    features.append(partial_charge_val if partial_charge_val is not None else 0.0)
            elif scheme == "dist_to_cf3":
                if self.use_pfas_specific_features and self.use_3d_coords:
                    features.append(float(atom_dist_feats.get("dist_to_cf3", -1.0)))
            elif scheme == "dist_to_functional_group":
                if self.use_pfas_specific_features and self.use_3d_coords:
                    features.append(float(atom_dist_feats.get("dist_to_functional", -1.0)))
            elif scheme == "is_head_group_atom":
                if self.use_pfas_specific_features and self.use_3d_coords:
                    features.append(float(atom_dist_feats.get("is_head_group", 0.0)))
            elif scheme == "homo_contribution":
                if homo_lumo_contrib_val is not None and len(homo_lumo_contrib_val) > 0:
                    features.append(homo_lumo_contrib_val[0])
            elif scheme == "lumo_contribution":
                if homo_lumo_contrib_val is not None and len(homo_lumo_contrib_val) > 1:
                    features.append(homo_lumo_contrib_val[1])
        return features

    def _get_bond_features(
        self, bond: Chem.Bond, bond_lengths_map: Optional[Dict[Tuple[int, int], float]] = None
    ) -> List[Union[int, float]]:
        """
        Extract comprehensive features for a single bond.
        
        Args:
            bond: RDKit Bond object
            bond_lengths_map: Optional dictionary mapping bond indices to lengths
            
        Returns:
            List of extracted bond features (mixed int/float types)
        """
        features = []
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        
        # Check if this is a C-F bond
        is_cf_bond_val = (
            (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 9) or
            (begin_atom.GetAtomicNum() == 9 and end_atom.GetAtomicNum() == 6)
        )
        
        # Check if this is a bond between two fluorinated carbons
        is_cf_cf_bond_val = False
        if (self.use_pfas_specific_features and 
            begin_atom.GetAtomicNum() == 6 and 
            end_atom.GetAtomicNum() == 6):
            is_cf_cf_bond_val = (
                sum(1 for n in begin_atom.GetNeighbors() if n.GetAtomicNum() == 9) > 0
                and sum(1 for n in end_atom.GetNeighbors() if n.GetAtomicNum() == 9) > 0
            )
            
        is_fluorinated_tail_bond_val = (
            is_cf_bond_val or is_cf_cf_bond_val 
            if self.use_pfas_specific_features 
            else False
        )
        
        # Check if bond involves functional group atoms
        is_func_group_bond_val = False
        if self.use_pfas_specific_features:
            functional_group_checks = [
                self._is_in_carboxylic_group, 
                self._is_in_sulfonic_group, 
                self._is_in_phosphonic_group
            ]
            is_func_group_bond_val = any(
                fn(atm)
                for fn in functional_group_checks
                for atm in [begin_atom, end_atom]
            )
            
        # Get bond length if 3D coordinates are available
        bond_len_val = None
        if self.use_3d_coords and bond_lengths_map:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_key: Tuple[int, int] = (i, j) if i < j else (j, i)
            bond_len_val = bond_lengths_map.get(bond_key)

        for scheme in self.bond_feature_schemes:
            if scheme == "bond_type":
                features.extend(self._one_hot_encoding(bond.GetBondType(), self.BOND_FEATURES_DEFAULTS["bond_type"]))
            elif scheme == "is_conjugated":
                features.extend(
                    self._one_hot_encoding(
                        1 if bond.GetIsConjugated() else 0, self.BOND_FEATURES_DEFAULTS["is_conjugated"]
                    )
                )
            elif scheme == "is_in_ring":
                features.extend(
                    self._one_hot_encoding(1 if bond.IsInRing() else 0, self.BOND_FEATURES_DEFAULTS["is_in_ring"])
                )
            elif scheme == "is_cf_bond":
                features.append(float(is_cf_bond_val))
            elif scheme == "is_cf_cf_bond":
                if self.use_pfas_specific_features:
                    features.append(float(is_cf_cf_bond_val))
            elif scheme == "is_fluorinated_tail_bond":
                if self.use_pfas_specific_features:
                    features.append(float(is_fluorinated_tail_bond_val))
            elif scheme == "is_functional_group_bond":
                if self.use_pfas_specific_features:
                    features.append(float(is_func_group_bond_val))
            elif scheme == "bond_length":
                if self.use_3d_coords and bond_len_val is not None:
                    features.append(bond_len_val)
        return features

    def mol_to_graph(
        self, 
        mol: Chem.Mol, 
        additional_features: Optional[Dict[str, List[float]]] = None
    ) -> Data:
        """
        Convert an RDKit molecule to a PyTorch Geometric Data object.
        
        This method performs comprehensive processing including:
        - Adding hydrogens and generating 3D coordinates if needed
        - Extracting atom and bond features
        - Computing distance-based features for PFAS molecules
        - Including partial charges and quantum mechanical properties
        
        Args:
            mol: RDKit molecule object
            additional_features: Optional dictionary containing:
                - partial_charges: List of partial charges per atom
                - homo_contributions: HOMO orbital contributions per atom
                - lumo_contributions: LUMO orbital contributions per atom
                - label: Target property value
                
        Returns:
            PyTorch Geometric Data object with node and edge features
        """
        mol_processed = Chem.Mol(mol) 
        initial_atom_count = mol_processed.GetNumAtoms()
        original_smiles_for_log = (
            Chem.MolToSmiles(mol_processed) if mol_processed else "None"
        )

        try:
            mol_with_hs_attempt = Chem.AddHs(
                mol_processed, 
                explicitOnly=False, 
                addCoords=(mol_processed.GetNumConformers() > 0)
            )

            # Check if hydrogens were actually added or if the molecule inherently has no implicit Hs to add (e.g. H2)
            # or if AddHs failed to change atom count when it should have.
            if (
                mol_with_hs_attempt.GetNumAtoms() > initial_atom_count
                or (initial_atom_count > 0 and Descriptors.HeavyAtomCount(mol_processed) < initial_atom_count) # type: ignore
                or (initial_atom_count == 0 and mol_with_hs_attempt.GetNumAtoms() > 0)
            ):  # handles H2, or if it was already all H
                mol_processed = mol_with_hs_attempt
            elif Descriptors.HeavyAtomCount(mol_processed) > 0:  # If it has heavy atoms but Hs weren't added
                logger.info(
                    f"AddHs with addCoords=True did not increase atom count for {original_smiles_for_log}. Trying AddHs with addCoords=False."
                )
                mol_processed = Chem.AddHs(mol_processed, explicitOnly=False, addCoords=False)  # Fallback
            # If still no change, mol_processed remains as is (e.g. noble gas, or already fully H-specified)

        except Exception as e:
            logger.warning(
                f"Failed to add/process hydrogens for molecule: {original_smiles_for_log}. Error: {e}. Proceeding with mol_processed in its current state (atom count: {mol_processed.GetNumAtoms()})."
            )

        if self.use_3d_coords and mol_processed.GetNumConformers() == 0:
            try:
                logger.info(f"Attempting to generate 3D coordinates for molecule: {Chem.MolToSmiles(mol_processed)}")
                AllChem.EmbedMolecule(mol_processed, AllChem.ETKDGv3()) # type: ignore
                AllChem.UFFOptimizeMolecule(mol_processed) # type: ignore
                if mol_processed.GetNumConformers() == 0:
                    logger.warning(
                        f"Failed to generate 3D coordinates for {Chem.MolToSmiles(mol_processed)}. Creating dummy 2D-like conformer."
                    )
                    conf = Chem.Conformer(mol_processed.GetNumAtoms())
                    for i in range(mol_processed.GetNumAtoms()):
                        conf.SetAtomPosition(i, (float(i) * 0.1, 0.0, 0.0))
                    mol_processed.AddConformer(conf, assignId=True)
            except Exception as e:
                logger.error(
                    f"Error generating 3D conformer for {Chem.MolToSmiles(mol_processed)}: {e}. Proceeding with current conformer state."
                )

        partial_charges_list = additional_features.get("partial_charges") if additional_features else None
        homo_contribs = additional_features.get("homo_contributions") if additional_features else None
        lumo_contribs = additional_features.get("lumo_contributions") if additional_features else None

        bond_lengths_map = (
            self.feature_extractor.calculate_bond_lengths(mol_processed)
            if self.use_3d_coords and mol_processed.GetNumConformers() > 0
            else {}
        )

        distance_features_map = None
        if self.use_pfas_specific_features and self.use_3d_coords and mol_processed.GetNumConformers() > 0:
            distance_features_map = self.feature_extractor.calculate_distance_features(mol_processed)

        x_features = []
        for i in range(mol_processed.GetNumAtoms()):
            atom = mol_processed.GetAtomWithIdx(i)
            pc = partial_charges_list[i] if partial_charges_list and i < len(partial_charges_list) else None
            hlc = None
            if homo_contribs and i < len(homo_contribs) and lumo_contribs and i < len(lumo_contribs):
                hlc = [homo_contribs[i], lumo_contribs[i]]

            atom_feats = self._get_atom_features(atom, mol_processed, pc, distance_features_map, hlc)
            x_features.append(atom_feats)

        x = (
            torch.tensor(x_features, dtype=torch.float)
            if x_features
            else torch.empty((0, self.atom_feature_dim), dtype=torch.float)
        )

        edge_indices = []
        edge_attrs_list = []
        for bond in mol_processed.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feats = self._get_bond_features(bond, bond_lengths_map)
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs_list.extend([bond_feats, bond_feats])

        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor(edge_attrs_list, dtype=torch.float)
            if edge_attrs_list
            else torch.empty((0, self.bond_feature_dim), dtype=torch.float)
        )

        pos = None
        if self.use_3d_coords and mol_processed.GetNumConformers() > 0:
            conformer = mol_processed.GetConformer()
            positions = [conformer.GetAtomPosition(i) for i in range(mol_processed.GetNumAtoms())]
            pos = torch.tensor([[p.x, p.y, p.z] for p in positions], dtype=torch.float)

        data_dict = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "num_nodes": mol_processed.GetNumAtoms()}
        if pos is not None:
            data_dict["pos"] = pos

        if additional_features and "label" in additional_features:
            data_dict["y"] = torch.tensor([additional_features["label"]], dtype=torch.float)
        else:
            # Calculate and add global molecular descriptors if not already provided as a 'label'
            descriptors_dict = self._get_molecule_descriptors(mol_processed)
            # Ensure a consistent order for the tensor
            descriptor_values = [
                descriptors_dict.get("mol_weight", 0.0),
                descriptors_dict.get("logp", 0.0),
                descriptors_dict.get("num_h_acceptors", 0.0),
                descriptors_dict.get("num_h_donors", 0.0),
                descriptors_dict.get("num_rotatable_bonds", 0.0),
                descriptors_dict.get("tpsa", 0.0),
                descriptors_dict.get("qed", 0.0),
            ]
            data_dict["y"] = torch.tensor(descriptor_values, dtype=torch.float)  # Shape [num_descriptors]

        return Data(**data_dict)

    def file_to_graph(
        self, file_path: str, additional_features: Optional[Dict[str, List[float]]] = None
    ) -> Optional[Data]:
        if not os.path.exists(file_path):
            msg = f"Molecule file not found: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            mol = None
            if file_path.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(file_path, removeHs=False)
                mol = next(iter(suppl), None)
            elif file_path.endswith(".mol2"):
                mol = Chem.MolFromMol2File(file_path, removeHs=False)
            elif file_path.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(file_path, removeHs=False)
            elif file_path.endswith(".mol"):
                mol = Chem.MolFromMolFile(file_path, removeHs=False)
            else:
                logger.error(f"Unsupported molecule file format: {file_path}")
                return None

            if mol is None:
                logger.error(f"Failed to read molecule from {file_path}")
                return None

            # mol_to_graph will handle AddHs internally with mol_processed
            return self.mol_to_graph(mol, additional_features)
        except Exception as e:
            logger.error(f"Error processing molecule file {file_path}")
            return None

    def smiles_to_graph(
        self, smiles: str, additional_features: Optional[Dict[str, List[float]]] = None
    ) -> Optional[Data]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Could not parse SMILES: {smiles}")
                return None
            # mol_to_graph will handle AddHs internally with mol_processed
            return self.mol_to_graph(mol, additional_features)
        except Exception as e:
            logger.error(f"Error converting SMILES '{smiles}' to graph")
            return None

    def _get_adjacency_matrix(self, mol: Chem.Mol) -> np.ndarray:
        # Ensure mol has explicit hydrogens for consistent adjacency matrix
        mol_with_hs = Chem.AddHs(Chem.Mol(mol))
        adj = Chem.GetAdjacencyMatrix(mol_with_hs)
        return np.array(adj, dtype=int)

    def mol_to_json_graph(self, mol: Chem.Mol, partial_charges: Optional[List[float]] = None) -> Dict[str, Any]:
        mol_with_hs = Chem.AddHs(Chem.Mol(mol))
        nodes = []
        for atom in mol_with_hs.GetAtoms():
            node_data = {
                "id": atom.GetIdx(),
                "atomic_num": atom.GetAtomicNum(),
                "symbol": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "num_hydrogens": atom.GetTotalNumHs(),
                "degree": atom.GetDegree(),
            }
            
            # Add partial charge if available
            if partial_charges and atom.GetIdx() < len(partial_charges):
                node_data["partial_charge"] = partial_charges[atom.GetIdx()]
            
            nodes.append(node_data)
        bond_lengths_map = {}
        if mol_with_hs.GetNumConformers() > 0:
            bond_lengths_map = self.feature_extractor.calculate_bond_lengths(mol_with_hs)

        edges = []
        for bond in mol_with_hs.GetBonds():
            bond_idx_tuple = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            length = bond_lengths_map.get(bond_idx_tuple)
            edges.append(
                {
                    "source": bond.GetBeginAtomIdx(),
                    "target": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                    "is_conjugated": bond.GetIsConjugated(),
                    "is_in_ring": bond.IsInRing(),
                    "length": length,
                }
            )
        descriptors = self._get_molecule_descriptors(mol_with_hs)
        return {"nodes": nodes, "edges": edges, "descriptors": descriptors, "smiles": Chem.MolToSmiles(mol_with_hs)}

    def _get_molecule_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        return {
            "mol_weight": Descriptors.MolWt(mol), # type: ignore
            "logp": Descriptors.MolLogP(mol), # type: ignore
            "num_h_acceptors": Lipinski.NumHAcceptors(mol), # type: ignore
            "num_h_donors": Lipinski.NumHDonors(mol), # type: ignore
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol), # type: ignore
            "tpsa": Descriptors.TPSA(mol), # type: ignore
            "qed": QED.qed(mol), # type: ignore
        }

    def file_to_json_graph(
        self, file_path: str, output_dir: Optional[str] = None, output_filename: Optional[str] = None, partial_charges: Optional[List[float]] = None
    ) -> Optional[str]:
        if not os.path.exists(file_path):
            logger.error(f"Molecule file not found: {file_path}")
            return None
        try:
            mol = None
            if file_path.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(file_path, removeHs=False)
                mol = next(iter(suppl), None)
            elif file_path.endswith(".mol2"):
                mol = Chem.MolFromMol2File(file_path, removeHs=False)
            elif file_path.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(file_path, removeHs=False)
            elif file_path.endswith(".mol"):
                mol = Chem.MolFromMolFile(file_path, removeHs=False)
            else:
                logger.error(f"Unsupported molecule file format: {file_path}")
                return None
            if mol is None:
                logger.error(f"Failed to read molecule from {file_path}")
                return None

            json_data = self.mol_to_json_graph(mol, partial_charges)  # mol_to_json_graph handles AddHs

            output_dir = output_dir or os.path.dirname(file_path)
            if not output_filename:
                output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_graph.json"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=4)
            return output_path
        except Exception as e:
            logger.error(f"Error processing molecule file {file_path} to JSON")
            return None


def create_graph_processor(config: Optional[Dict[str, Any]] = None) -> MolecularGraphProcessor:
    return MolecularGraphProcessor(config=config)


def mol_file_to_graph(
    mol_file_path: str,
    config: Optional[Dict[str, Any]] = None,
    additional_features: Optional[Dict[str, List[float]]] = None,
) -> Optional[Data]:
    try:
        processor = create_graph_processor(config)
        return processor.file_to_graph(mol_file_path, additional_features)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to convert {mol_file_path} to graph")
        return None


def graph_to_device(
    graph: Union[Data, Dict[str, torch.Tensor]], device: torch.device
) -> Union[Data, Dict[str, torch.Tensor]]:
    if isinstance(graph, Data):
        return graph.to(device)  # type: ignore
    elif isinstance(graph, dict):
        return {level: g.to(device) for level, g in graph.items()}  # type: ignore
    return graph


def collate_graphs(graphs: List[Data]) -> Data:
    from torch_geometric.loader import DataLoader

    if not graphs:
        return Batch()

    class DummyDataset(torch.utils.data.Dataset[Data]):
        def __init__(self, data_list: List[Data]):
            self.data_list = data_list

        def __len__(self) -> int:
            return len(self.data_list)

        def __getitem__(self, idx: int) -> Data:
            return self.data_list[idx]

    temp_loader = DataLoader(DummyDataset(graphs), batch_size=len(graphs) if graphs else 1)  # type: ignore
    try:
        return next(iter(temp_loader))
    except StopIteration:  # Handle empty graphs list fed to DummyDataset then DataLoader
        return Batch()


def find_charges_file(mol_file: str, charges_dir: Optional[str] = None) -> Optional[str]:
    """
    Find a corresponding charges file for a given molecule file.
    Looks for .charges, .chg, _charges.txt, .txt, or .log in the specified directory or mol_file's directory.
    """
    if charges_dir is None:
        charges_dir = os.path.dirname(mol_file)
    base_name = os.path.splitext(os.path.basename(mol_file))[0]

    patterns = [
        os.path.join(charges_dir, f"{base_name}.charges"),
        os.path.join(charges_dir, f"{base_name}.CHG"),
        os.path.join(charges_dir, f"{base_name}.chg"),
        os.path.join(charges_dir, f"{base_name}_charges.txt"),
        os.path.join(charges_dir, f"{base_name}.txt"),
        os.path.join(charges_dir, f"{base_name}.log"),
    ]
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    logger.debug(f"No standard charges file found for {mol_file} in {charges_dir}")
    return None


def read_charges_from_file(charge_file: str) -> Optional[List[float]]:
    charges = []
    try:
        with open(charge_file, "r") as f:
            lines = f.readlines()
        file_ext = os.path.splitext(charge_file)[1].lower()

        if file_ext in [".charges", ".chg"]:
            for line in lines:
                try:
                    charges.append(float(line.split()[-1]))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse charge from line: '{line.strip()}' in {charge_file}")
        elif file_ext == ".txt":  # ORCA Mulliken or similar
            for line_idx, line in enumerate(lines):
                if ":" in line:  # Common for "Atom Symbol : Charge"
                    try:
                        charges.append(float(line.split(":")[-1].strip()))
                    except ValueError:
                        logger.warning(f"Could not parse charge from .txt line: '{line.strip()}' in {charge_file}")
                elif (
                    line_idx < 20 and len(line.split()) == 1
                ):  # Attempt to parse if it's just numbers (e.g. simple list)
                    try:
                        charges.append(float(line.strip()))
                    except ValueError:
                        pass  # Ignore if not a float, might be header
        elif file_ext == ".log":  # ORCA ESP charges
            esp_section_found = False
            for line in lines:
                if "ESP Fit Center  Symbol    Charge (e)" in line:
                    esp_section_found = True
                    continue
                if esp_section_found:
                    if not line.strip() or "----" in line:
                        if charges:
                            break
                            continue
                    parts = line.split()
                    if len(parts) >= 3:  # Index, Symbol, Charge
                        try:
                            charges.append(float(parts[-1]))
                        except ValueError:
                            pass
        elif file_ext == ".json":
            try:
                data = json.loads("".join(lines))  # Read all lines and parse
                if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
                    charges = [float(c) for c in data]
                elif isinstance(data, dict) and "charges" in data and isinstance(data["charges"], list):
                    charges = [float(c) for c in data["charges"]]
                else:
                    logger.warning(
                        f"'.json' charges file {charge_file} does not contain a 'charges' list or is not a direct list of charges."
                    )
            except json.JSONDecodeError:
                logger.error(f"Could not decode JSON from {charge_file}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing charges from JSON file {charge_file}")
        else:
            raise ValueError(f"Unsupported charges file format: {file_ext}")
        return charges if charges else None
    except FileNotFoundError:
        logger.error(f"Charges file not found: {charge_file}")
        raise
    except Exception as e:
        logger.error(f"Error reading charges from {charge_file}")
        return None


def create_molecular_graph_json(
    mol_file: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    charges_file: Optional[str] = None,
) -> Optional[str]:
    """
    Create a molecular graph JSON from a molecule file with optional partial charges.
    
    Args:
        mol_file: Path to the molecule file
        output_dir: Directory to save the JSON file 
        output_filename: Name of the output JSON file
        config: Configuration dictionary for the graph processor
        charges_file: Optional path to file containing partial charges
        
    Returns:
        Path to the created JSON file, or None if failed
    """
    try:
        processor = create_graph_processor(config)
        
        # Read partial charges if charges_file is provided
        partial_charges = None
        if charges_file:
            try:
                partial_charges = read_charges_from_file(charges_file)
                if partial_charges:
                    logger.info(f"Loaded {len(partial_charges)} partial charges from {charges_file}")
                else:
                    logger.warning(f"No charges could be read from {charges_file}")
            except Exception as e:
                logger.error(f"Failed to read charges from {charges_file}: {e}")
                # Continue without charges rather than failing completely
        
        return processor.file_to_json_graph(mol_file, output_dir, output_filename, partial_charges)
    except Exception as e:
        logger.error(f"Failed to create JSON graph for {mol_file}: {e}")
        return None


def batch_create_graphs_from_molecules(
    mol_dir: str,
    output_dir: str,
    file_format: str = "sdf",
    config: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
    use_3d_coords: bool = True,
    use_pfas_specific_features: bool = True,
) -> List[str]:
    if not os.path.isdir(mol_dir):
        logger.error(f"Input directory not found: {mol_dir}")
        return []
    os.makedirs(output_dir, exist_ok=True)
    mol_files = glob.glob(os.path.join(mol_dir, f"*.{file_format}"))
    if not mol_files:
        logger.warning(f"No .{file_format} files found in {mol_dir}")
        return []
    logger.info(f"Found {len(mol_files)} files to process: {mol_files}")

    worker_config = (config or {}).copy()
    worker_config.setdefault("use_3d_coords", use_3d_coords)
    worker_config.setdefault("use_pfas_specific_features", use_pfas_specific_features)
    worker_config.setdefault("use_partial_charges", worker_config.get("use_partial_charges", True))

    tasks = [(f, worker_config, output_dir) for f in mol_files]
    saved_graph_paths = []

    if not tasks:
        return []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_process_single_mol_file_for_batch, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(future_to_file):
            mol_file_path_key = future_to_file[future]
            try:
                result_path = future.result()
                if result_path:
                    saved_graph_paths.append(result_path)
                    logger.info(f"Successfully processed and saved graph for: {mol_file_path_key} -> {result_path}")
                # else: Error already logged by worker or _process_single_mol_file_for_batch
            except Exception as exc:
                logger.error(f"{mol_file_path_key} generated an exception during main thread future processing: {exc}")
    return saved_graph_paths
