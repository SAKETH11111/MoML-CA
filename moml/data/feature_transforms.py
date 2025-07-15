import torch
from torch_geometric.data import Data
from rdkit import Chem
import logging # Added for error logging
import yaml # Moved import to top

from moml.core.molecular_feature_extraction import MolecularFeatureExtractor


class CreateEdges:
    """A transform that creates edges based on a distance cutoff if they don't exist."""
    def __init__(self, cutoff: float = 3.0):
        self.cutoff = cutoff

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            return data  # Edges already exist

        if hasattr(data, 'pos') and data.pos is not None:
            dist_matrix = torch.cdist(data.pos, data.pos)
            mask = (dist_matrix < self.cutoff) & (dist_matrix > 0)
            data.edge_index = mask.nonzero().t().contiguous()
        return data

class FeaturizeNodes:
    """A transform that correctly adds node features (data.x) to a Data object."""
    def __call__(self, data: Data) -> Data:
        if not hasattr(data, 'z') or data.z is None:
            return data

        mol = Chem.RWMol()
        for atomic_num in data.z:
            mol.AddAtom(Chem.Atom(int(atomic_num)))

        if data.edge_index is not None:
            rows, cols = data.edge_index
            for i, j in zip(rows.tolist(), cols.tolist()):
                if i < j:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        
        # It's okay if sanitization fails for some structures,
        # we extract features that don't rely on it.
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.GetAtomicNum(), MolecularFeatureExtractor.ATOM_FEATURES['atomic_num']
            ))
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.GetDegree(), MolecularFeatureExtractor.ATOM_FEATURES['degree']
            ))
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.GetFormalCharge(), MolecularFeatureExtractor.ATOM_FEATURES['formal_charge']
            ))
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.GetHybridization(), MolecularFeatureExtractor.ATOM_FEATURES['hybridization']
            ))
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.GetIsAromatic(), [False, True]
            ))
            features.extend(MolecularFeatureExtractor.one_hot_encoding(
                atom.IsInRing(), [False, True]
            ))
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)
        
        # If the feature vector is 11-dimensional (original QM9), pad it to 29
        if x.size(-1) == 11:
            pad = torch.zeros(x.size(0), 18, device=x.device)
            x = torch.cat([x, pad], dim=-1)
            
        data.x = x
        return data
class PadQM9Features:
    """A transform to pad QM9 features to a target dimension."""
    def __init__(self, target_dim=29):
        self.target_dim = target_dim

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, 'x') or data.x is None:
            return data
        
        num_nodes, current_dim = data.x.shape
        if current_dim < self.target_dim:
            padding_dim = self.target_dim - current_dim
            padding = torch.zeros((num_nodes, padding_dim), dtype=data.x.dtype, device=data.x.device)
            data.x = torch.cat([data.x, padding], dim=1)
        
        return data
class StandardizeTargets:
    """A transform to standardize targets using pre-computed statistics."""
    def __init__(self, stats_path="data/target_stats.yaml", dataset_name="qm9"):
        try:
            with open(stats_path, 'r') as f:
                stats = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Statistics file not found at {stats_path}")
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {stats_path}: {e}")
            raise ValueError(f"Error parsing YAML file: {stats_path} - {e}")
        
        if dataset_name not in stats:
            raise KeyError(f"Statistics for dataset '{dataset_name}' not found in {stats_path}")

        self.mean = torch.tensor(stats[dataset_name]['mean'])
        self.std = torch.tensor(stats[dataset_name]['std'])
        self.dataset_name = dataset_name
    def __call__(self, data: Data) -> Data:
        target_attr = 'y' if self.dataset_name in ['qm9', 'pfas'] else 'y_graph'
        if hasattr(data, target_attr) and getattr(data, target_attr) is not None:
            setattr(data, target_attr, (getattr(data, target_attr) - self.mean) / self.std)
        return data