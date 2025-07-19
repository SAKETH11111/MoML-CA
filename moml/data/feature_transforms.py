"""
moml/data/feature_transforms.py

Feature transformation classes for molecular graph data preprocessing.

This module provides various transformation classes that can be applied to
molecular graph data during dataset loading or preprocessing. These transforms
handle feature engineering, edge creation, standardization, and other data
preparation tasks commonly needed for molecular machine learning.

Classes:
    CreateEdges: Creates graph edges based on distance cutoffs
    FeaturizeNodes: Adds atomic features to graph nodes
    PadQM9Features: Pads feature vectors to target dimensions
    StandardizeTargets: Standardizes target values using statistics

Example:
    Basic usage with a data loader:
    
    >>> from torch_geometric.transforms import Compose
    >>> transforms = Compose([
    ...     CreateEdges(cutoff=3.5),
    ...     FeaturizeNodes(),
    ...     PadQM9Features(target_dim=32)
    ... ])
    >>> dataset = SomeDataset(transform=transforms)
"""

import logging
import os

import torch
import yaml
from torch_geometric.data import Data

# Constants
DEFAULT_DISTANCE_CUTOFF = 3.0
DEFAULT_TARGET_DIM = 29
DEFAULT_STATS_PATH = "data/target_stats.yaml"
DEFAULT_DATASET_NAME = "qm9"

# Configure module logger
logger = logging.getLogger(__name__)

# Conditional imports for optional dependencies
try:
    from rdkit import Chem  # type: ignore
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Molecular featurization will be limited.")
    
    # Create dummy Chem module for fallback
    class Chem:  # type: ignore
        """Dummy Chem class when RDKit is not available."""
        
        class RWMol:
            """Dummy RWMol class."""
            def AddAtom(self, atom): pass
            def AddBond(self, i, j, bond_type): pass
            def GetAtoms(self): return []
        
        class Atom:
            """Dummy Atom class."""
            def __init__(self, atomic_num): pass
        
        class BondType:
            """Dummy BondType class."""
            SINGLE = 1
        
        @staticmethod
        def SanitizeMol(mol): pass

try:
    from moml.core.molecular_feature_extraction import MolecularFeatureExtractor  # type: ignore
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False
    logger.warning("MolecularFeatureExtractor not available. Using fallback features.")
    
    # Create dummy feature extractor for fallback
    class MolecularFeatureExtractor:  # type: ignore
        """Dummy MolecularFeatureExtractor when not available."""
        
        ATOM_FEATURES = {
            "atomic_num": list(range(1, 119)),
            "degree": list(range(11)),
            "formal_charge": list(range(-5, 6)),
            "hybridization": list(range(8))
        }
        
        @staticmethod
        def one_hot_encoding(value, allowed_values):
            """Dummy one-hot encoding."""
            encoding = [0] * len(allowed_values)
            if value in allowed_values:
                encoding[allowed_values.index(value)] = 1
            return encoding


class CreateEdges:
    """
    Transform that creates graph edges based on distance cutoffs.
    
    This transform creates edges between atoms/nodes that are within a specified
    distance cutoff. If edges already exist in the data, they are preserved.
    This is useful for molecular graphs where connectivity information may be
    missing or incomplete.
    
    Args:
        cutoff: Maximum distance for creating edges between nodes (Angstroms)
    
    Attributes:
        cutoff: Distance cutoff value
    
    Example:
        >>> transform = CreateEdges(cutoff=3.5)
        >>> # Apply to data with positional coordinates
        >>> transformed_data = transform(molecular_graph_data)
    
    Note:
        Requires the input data to have a 'pos' attribute containing 3D
        coordinates of the atoms/nodes.
    """

    def __init__(self, cutoff: float = DEFAULT_DISTANCE_CUTOFF):
        """
        Initialize the edge creation transform.
        
        Args:
            cutoff: Maximum distance for edge creation in Angstroms
            
        Raises:
            ValueError: If cutoff is not positive
        """
        if cutoff <= 0:
            raise ValueError(f"Cutoff must be positive, got {cutoff}")
        
        self.cutoff = cutoff
        logger.debug(f"Initialized CreateEdges with cutoff={cutoff}")

    def __call__(self, data: Data) -> Data:
        """
        Apply edge creation transform to molecular graph data.
        
        Args:
            data: PyTorch Geometric Data object containing molecular graph
            
        Returns:
            Data object with edges added if they didn't exist
            
        Raises:
            AttributeError: If required attributes are missing
        """
        # Check if edges already exist
        if hasattr(data, "edge_index") and data.edge_index is not None:
            logger.debug("Edges already exist, skipping edge creation")
            return data

        # Check for positional coordinates
        if not hasattr(data, "pos") or data.pos is None:
            logger.warning("No positional coordinates found, cannot create edges")
            return data

        try:
            # Calculate distance matrix between all atom pairs
            dist_matrix = torch.cdist(data.pos, data.pos)
            
            # Create mask for distances within cutoff (excluding self-connections)
            mask = (dist_matrix < self.cutoff) & (dist_matrix > 0)
            
            # Create edge index from mask
            data.edge_index = mask.nonzero().t().contiguous()
            
            logger.debug(
                f"Created {data.edge_index.shape[1]} edges with cutoff {self.cutoff}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create edges: {e}")
            # Return data unchanged if edge creation fails
        
        return data


class FeaturizeNodes:
    """
    Transform that adds atomic features to graph nodes.
    
    This transform converts atomic numbers and connectivity information into
    rich feature vectors for each node/atom. Features include atomic properties,
    degree, formal charge, hybridization, aromaticity, and ring membership.
    
    The transform creates a molecular representation from atomic numbers and
    edges, then extracts comprehensive atomic features using RDKit and the
    MolecularFeatureExtractor.
    
    Attributes:
        None (stateless transform)
    
    Example:
        >>> transform = FeaturizeNodes()
        >>> # Apply to data with atomic numbers (z) and edges
        >>> featured_data = transform(molecular_graph_data)
        >>> print(featured_data.x.shape)  # [num_atoms, num_features]
    
    Note:
        Requires the input data to have 'z' (atomic numbers) and optionally
        'edge_index' attributes. Creates 'x' attribute with node features.
    
    Raises:
        ImportError: If RDKit or MolecularFeatureExtractor are not available
    """

    def __init__(self):
        """
        Initialize the node featurization transform.
        
        Raises:
            ImportError: If required dependencies are not available
        """
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for FeaturizeNodes transform")
        
        if not HAS_FEATURE_EXTRACTOR:
            raise ImportError(
                "MolecularFeatureExtractor is required for FeaturizeNodes transform"
            )
        
        logger.debug("Initialized FeaturizeNodes transform")

    def __call__(self, data: Data) -> Data:
        """
        Apply node featurization to molecular graph data.
        
        Args:
            data: PyTorch Geometric Data object with atomic numbers
            
        Returns:
            Data object with node features added as 'x' attribute
            
        Raises:
            AttributeError: If required atomic number data is missing
        """
        # Check for atomic numbers
        if not hasattr(data, "z") or data.z is None:
            logger.warning("No atomic numbers found, cannot featurize nodes")
            return data

        try:
            # Create RDKit molecule from atomic numbers and connectivity
            mol = self._create_molecule_from_data(data)
            
            # Extract atomic features
            atom_features = self._extract_atomic_features(mol)
            
            # Convert to tensor
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Handle QM9 feature padding for backward compatibility
            x = self._handle_qm9_padding(x)
            
            data.x = x
            logger.debug(f"Added node features with shape {x.shape}")
            
        except Exception as e:
            logger.error(f"Failed to featurize nodes: {e}")
            # Return data unchanged if featurization fails
        
        return data

    def _create_molecule_from_data(self, data: Data):
        """
        Create RDKit molecule from graph data.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            RDKit molecule object
        """
        mol = Chem.RWMol()
        
        # Add atoms
        for atomic_num in data.z:
            atom = Chem.Atom(int(atomic_num))
            mol.AddAtom(atom)
        
        # Add bonds if edges exist
        if hasattr(data, "edge_index") and data.edge_index is not None:
            rows, cols = data.edge_index
            added_bonds = set()  # Track added bonds to avoid duplicates
            
            for i, j in zip(rows.tolist(), cols.tolist()):
                bond_key = tuple(sorted([i, j]))
                if i != j and bond_key not in added_bonds:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    added_bonds.add(bond_key)
        
        # Attempt sanitization (non-critical if it fails)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            logger.debug("Molecule sanitization failed, proceeding with unsanitized molecule")
        
        return mol

    def _extract_atomic_features(self, mol):
        """
        Extract comprehensive atomic features from molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of feature vectors for each atom
        """
        atom_features = []
        
        for atom in mol.GetAtoms():
            features = []
            
            # Atomic number one-hot encoding
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.GetAtomicNum(),
                    MolecularFeatureExtractor.ATOM_FEATURES["atomic_num"]
                )
            )
            
            # Degree one-hot encoding
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.GetDegree(),
                    MolecularFeatureExtractor.ATOM_FEATURES["degree"]
                )
            )
            
            # Formal charge one-hot encoding
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.GetFormalCharge(),
                    MolecularFeatureExtractor.ATOM_FEATURES["formal_charge"]
                )
            )
            
            # Hybridization one-hot encoding
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.GetHybridization(),
                    MolecularFeatureExtractor.ATOM_FEATURES["hybridization"]
                )
            )
            
            # Aromaticity boolean feature
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.GetIsAromatic(), [False, True]
                )
            )
            
            # Ring membership boolean feature
            features.extend(
                MolecularFeatureExtractor.one_hot_encoding(
                    atom.IsInRing(), [False, True]
                )
            )
            
            atom_features.append(features)
        
        return atom_features

    def _handle_qm9_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Handle QM9 feature padding for backward compatibility.
        
        Args:
            x: Feature tensor
            
        Returns:
            Padded feature tensor if needed
        """
        # If the feature vector is 11-dimensional (original QM9), pad to 29
        if x.size(-1) == 11:
            padding_size = DEFAULT_TARGET_DIM - 11
            pad = torch.zeros(x.size(0), padding_size, device=x.device)
            x = torch.cat([x, pad], dim=-1)
            logger.debug("Applied QM9 feature padding from 11 to 29 dimensions")
        
        return x


class PadQM9Features:
    """
    Transform to pad feature vectors to a target dimension.
    
    This transform is useful for ensuring consistent feature dimensions across
    different datasets or when combining datasets with different feature sizes.
    It pads the feature vectors with zeros to reach the target dimension.
    
    Args:
        target_dim: Target number of features per node
    
    Attributes:
        target_dim: Target feature dimension
    
    Example:
        >>> transform = PadQM9Features(target_dim=32)
        >>> # Apply to data with node features
        >>> padded_data = transform(molecular_graph_data)
        >>> print(padded_data.x.shape[1])  # 32
    
    Note:
        Only pads features if current dimension is smaller than target.
        Does nothing if features are already at or above target dimension.
    """

    def __init__(self, target_dim: int = DEFAULT_TARGET_DIM):
        """
        Initialize the feature padding transform.
        
        Args:
            target_dim: Target feature dimension
            
        Raises:
            ValueError: If target_dim is not positive
        """
        if target_dim <= 0:
            raise ValueError(f"Target dimension must be positive, got {target_dim}")
        
        self.target_dim = target_dim
        logger.debug(f"Initialized PadQM9Features with target_dim={target_dim}")

    def __call__(self, data: Data) -> Data:
        """
        Apply feature padding to molecular graph data.
        
        Args:
            data: PyTorch Geometric Data object with node features
            
        Returns:
            Data object with padded node features
        """
        # Check for existing features
        if not hasattr(data, "x") or data.x is None:
            logger.debug("No node features found, skipping padding")
            return data

        try:
            num_nodes, current_dim = data.x.shape
            
            if current_dim < self.target_dim:
                padding_dim = self.target_dim - current_dim
                padding = torch.zeros(
                    (num_nodes, padding_dim),
                    dtype=data.x.dtype,
                    device=data.x.device
                )
                data.x = torch.cat([data.x, padding], dim=1)
                
                logger.debug(
                    f"Padded features from {current_dim} to {self.target_dim} dimensions"
                )
            elif current_dim > self.target_dim:
                logger.warning(
                    f"Current feature dimension ({current_dim}) exceeds target "
                    f"({self.target_dim}). No padding applied."
                )
            
        except Exception as e:
            logger.error(f"Failed to pad features: {e}")
            # Return data unchanged if padding fails
        
        return data


class StandardizeTargets:
    """
    Transform to standardize target values using pre-computed statistics.
    
    This transform applies z-score normalization to target values using
    pre-computed mean and standard deviation statistics. This is essential
    for neural network training stability and convergence.
    
    Args:
        stats_path: Path to YAML file containing dataset statistics
        dataset_name: Name of dataset in the statistics file
    
    Attributes:
        mean: Mean values for standardization
        std: Standard deviation values for standardization
        dataset_name: Dataset identifier
    
    Example:
        >>> transform = StandardizeTargets(
        ...     stats_path="data/target_stats.yaml",
        ...     dataset_name="qm9"
        ... )
        >>> standardized_data = transform(molecular_graph_data)
    
    Raises:
        FileNotFoundError: If statistics file doesn't exist
        KeyError: If dataset statistics are not found
        ValueError: If statistics file is malformed
    """

    def __init__(
        self,
        stats_path: str = DEFAULT_STATS_PATH,
        dataset_name: str = DEFAULT_DATASET_NAME
    ):
        """
        Initialize the target standardization transform.
        
        Args:
            stats_path: Path to statistics YAML file
            dataset_name: Dataset name in statistics file
            
        Raises:
            FileNotFoundError: If statistics file doesn't exist
            KeyError: If dataset not found in statistics
            ValueError: If statistics file is malformed
        """
        self.dataset_name = dataset_name
        
        # Load statistics from file
        stats = self._load_statistics(stats_path)
        
        # Extract dataset-specific statistics
        if dataset_name not in stats:
            raise KeyError(
                f"Statistics for dataset '{dataset_name}' not found in {stats_path}"
            )
        
        dataset_stats = stats[dataset_name]
        
        # Validate required fields
        if "mean" not in dataset_stats or "std" not in dataset_stats:
            raise ValueError(
                f"Missing 'mean' or 'std' in statistics for dataset '{dataset_name}'"
            )
        
        # Convert to tensors
        self.mean = torch.tensor(dataset_stats["mean"], dtype=torch.float)
        self.std = torch.tensor(dataset_stats["std"], dtype=torch.float)
        
        # Validate statistics
        if torch.any(self.std <= 0):
            logger.warning(
                "Some standard deviation values are zero or negative. "
                "This may cause numerical issues."
            )
        
        logger.info(
            f"Initialized StandardizeTargets for dataset '{dataset_name}' "
            f"with mean={self.mean.tolist()}, std={self.std.tolist()}"
        )

    def _load_statistics(self, stats_path: str) -> dict:
        """
        Load statistics from YAML file.
        
        Args:
            stats_path: Path to statistics file
            
        Returns:
            Dictionary containing statistics
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is malformed
        """
        # Check if file exists
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        
        try:
            with open(stats_path, "r") as f:
                stats = yaml.safe_load(f)
            
            if not isinstance(stats, dict):
                raise ValueError("Statistics file must contain a dictionary")
            
            return stats
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {stats_path}: {e}")
            raise ValueError(f"Error parsing YAML file: {stats_path}") from e
        except Exception as e:
            logger.error(f"Error reading statistics file {stats_path}: {e}")
            raise

    def __call__(self, data: Data) -> Data:
        """
        Apply target standardization to molecular graph data.
        
        Args:
            data: PyTorch Geometric Data object with target values
            
        Returns:
            Data object with standardized target values
        """
        # Determine target attribute name based on dataset
        target_attr = self._get_target_attribute()
        
        # Check if target exists
        if not hasattr(data, target_attr):
            logger.debug(f"No target attribute '{target_attr}' found, skipping standardization")
            return data
        
        target_values = getattr(data, target_attr)
        if target_values is None:
            logger.debug("Target values are None, skipping standardization")
            return data
        
        try:
            # Move tensors to same device if needed
            device = target_values.device
            mean = self.mean.to(device)
            std = self.std.to(device)
            
            # Apply standardization: (x - mean) / std
            standardized_values = (target_values - mean) / std
            setattr(data, target_attr, standardized_values)
            
            logger.debug(f"Applied standardization to {target_attr}")
            
        except Exception as e:
            logger.error(f"Failed to standardize targets: {e}")
            # Return data unchanged if standardization fails
        
        return data

    def _get_target_attribute(self) -> str:
        """
        Determine the target attribute name based on dataset.
        
        Returns:
            Name of target attribute
        """
        # Use different target attributes for different datasets
        if self.dataset_name in ["qm9", "pfas"]:
            return "y"
        else:
            return "y_graph"
