import torch
from torch_geometric.data import Data
from rdkit import Chem

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

        data.x = torch.tensor(atom_features, dtype=torch.float)
        return data