"""
moml/data/datasets.py

Dataset classes for working with molecular data.

This module provides specialized dataset classes for handling molecular
graphs, hierarchical molecular graphs, and PFAS compounds. Each dataset
class is designed to work with different types of molecular data and
provides appropriate preprocessing and transformation capabilities.

Classes:
    MolecularGraphDataset: Basic molecular graph dataset with labels
    HierarchicalGraphDataset: Multi-level hierarchical molecular graphs
    PFASDataset: Specialized dataset for PFAS compounds with features

Example:
    Basic usage with molecular graph dataset:
    
    >>> mol_files = ['mol1.sdf', 'mol2.sdf']
    >>> labels = {'mol1.sdf': 1.0, 'mol2.sdf': 0.5}
    >>> dataset = MolecularGraphDataset(mol_files, labels=labels)
    >>> print(len(dataset))
    2
"""

import glob
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch.utils.data import Dataset
from tqdm import tqdm

# Constants
DEFAULT_LEVELS = ["atom", "functional_group", "structural_motif"]
GRAPH_CACHE_EXTENSION = ".pt"
GRAPH_JSON_EXTENSION = "_graph.json"
DEFAULT_SMILES_COLUMN = "smiles"

# Configure module logger
logger = logging.getLogger(__name__)

# Conditional imports for optional dependencies
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Some functionality will be limited.")
    
    # Create dummy Chem module for fallback
    class Chem:  # type: ignore
        """Dummy Chem class when RDKit is not available."""
        
        @staticmethod
        def MolFromSmiles(smiles: str):
            """Dummy method that returns None."""
            return None
        
        @staticmethod
        def MolToSmiles(mol):
            """Dummy method that returns empty string."""
            return ""
    
    class AllChem:  # type: ignore
        """Dummy AllChem class when RDKit is not available."""
        
        @staticmethod
        def EmbedMolecule(mol):
            """Dummy method that returns -1."""
            return -1
        
        @staticmethod
        def ETKDGv3():
            """Dummy method for embedding parameters."""
            return None
        
        @staticmethod
        def UFFOptimizeMolecule(mol):
            """Dummy method for optimization."""
            return None

try:
    from moml.core import create_graph_processor  # type: ignore
    from moml.core.molecular_graph_processor import MolecularGraphProcessor  # type: ignore
    HAS_GRAPH_PROCESSOR = True
except ImportError:
    HAS_GRAPH_PROCESSOR = False
    logger.warning("Graph processor not available. Graph creation will fail.")
    
    def create_graph_processor(*args, **kwargs):  # type: ignore
        """Raise ImportError when graph processor is not available."""
        raise ImportError(
            "Graph processor requires additional dependencies. "
            "Please install moml.core package."
        )
    
    class MolecularGraphProcessor:  # type: ignore
        """Dummy MolecularGraphProcessor when not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MolecularGraphProcessor requires additional dependencies. "
                "Please install moml.core package."
            )

try:
    from torch_geometric.data import Data  # type: ignore
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not available. Using dummy Data class.")
    
    # Create dummy Data class for fallback
    class Data:  # type: ignore
        """Dummy Data class when PyTorch Geometric is not available."""
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class MolecularGraphDataset(Dataset):
    """
    Dataset for molecular graphs with optional labels.
    
    This dataset processes molecule files into graph representations using
    a configured graph processor. It supports both batch and individual
    processing modes, caching, and optional transformations.
    
    Args:
        mol_files: List of paths to molecule files (SDF, MOL, etc.)
        labels: Optional dictionary mapping filenames to float labels
        config: Configuration dictionary for graph processing
        transform: Optional callable to transform graphs after creation
    
    Attributes:
        mol_files: List of molecule file paths
        labels: Dictionary of labels (if provided)
        config: Graph processor configuration
        transform: Transform function
        graphs: List of processed molecular graphs
        graph_processor: Instance of MolecularGraphProcessor
    
    Example:
        >>> mol_files = ['molecule1.sdf', 'molecule2.sdf']
        >>> labels = {'molecule1.sdf': 1.0, 'molecule2.sdf': 0.5}
        >>> config = {'include_conformers': True}
        >>> dataset = MolecularGraphDataset(mol_files, labels, config)
        >>> graph = dataset[0]  # Get first molecular graph
    
    Raises:
        ImportError: If required graph processing dependencies are missing
        FileNotFoundError: If molecule files cannot be found
    """

    def __init__(
        self,
        mol_files: List[str],
        labels: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        """Initialize the molecular graph dataset."""
        self.mol_files = mol_files
        self.labels = labels
        self.config = config or {}
        self.transform = transform
        self.graphs = []

        # Validate inputs
        if not mol_files:
            raise ValueError("mol_files cannot be empty")
        
        if not HAS_GRAPH_PROCESSOR:
            raise ImportError(
                "Graph processor dependencies required for MolecularGraphDataset"
            )

        # Create graph processor
        try:
            self.graph_processor = create_graph_processor(self.config)
        except Exception as e:
            logger.error(f"Failed to create graph processor: {e}")
            raise

        # Process graphs
        self._process_graphs()

    def _process_graphs(self) -> None:
        """
        Process all molecule files into graphs.
        
        Uses batch processing if available, otherwise falls back to
        individual file processing. Handles caching, labeling, and
        transformations.
        
        Raises:
            Exception: If critical processing errors occur
        """
        # Use batch processing if available
        if hasattr(self.graph_processor, "batch_files_to_graphs"):
            self._process_graphs_batch()
        else:
            self._process_graphs_individual()

    def _process_graphs_batch(self) -> None:
        """Process graphs using batch processing."""
        try:
            self.graphs = self.graph_processor.batch_files_to_graphs(
                self.mol_files
            )
            
            # Add labels to graphs if available
            self._assign_labels_batch()
            
            # Apply transforms
            self._apply_transforms()
            
        except Exception as e:
            logger.error(f"Batch graph processing failed: {e}")
            # Fall back to individual processing
            self._process_graphs_individual()

    def _assign_labels_batch(self) -> None:
        """Assign labels to graphs in batch mode."""
        if self.labels is None:
            # Remove any existing labels
            for graph in self.graphs:
                self._remove_labels(graph)
            return

        # Assign labels to corresponding graphs
        for i, file_path in enumerate(self.mol_files):
            if i >= len(self.graphs):
                break
                
            if file_path in self.labels:
                self._assign_label(self.graphs[i], self.labels[file_path])
            else:
                self._remove_labels(self.graphs[i])

    def _process_graphs_individual(self) -> None:
        """Process graphs individually with caching support."""
        processed_graphs = []
        
        for file_path in tqdm(self.mol_files, desc="Processing molecular graphs"):
            graph = self._process_single_file(file_path)
            
            if graph is not None:
                processed_graphs.append(graph)
            else:
                logger.warning(
                    f"Failed to process {file_path}, excluding from dataset"
                )
        
        self.graphs = processed_graphs

    def _process_single_file(self, file_path: str):
        """
        Process a single molecule file.
        
        Args:
            file_path: Path to molecule file
            
        Returns:
            Processed graph or None if processing failed
        """
        graph = None
        
        try:
            # Check for cached graph
            cache_path = file_path + GRAPH_CACHE_EXTENSION
            if os.path.exists(cache_path):
                graph = self._load_cached_graph(cache_path)
                if graph is not None:
                    logger.debug(f"Loaded cached graph: {cache_path}")
            
            # Process file if no cache
            if graph is None:
                graph = self.graph_processor.file_to_graph(file_path)
            
            # Handle labeling
            if graph is not None:
                if self.labels is not None and file_path in self.labels:
                    self._assign_label(graph, self.labels[file_path])
                elif self.labels is None:
                    self._remove_labels(graph)
            
            # Apply transform
            if graph is not None and self.transform is not None:
                graph = self._apply_transform(graph, file_path)
                
        except FileNotFoundError:
            logger.error(f"Molecule file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return graph

    def _load_cached_graph(self, cache_path: str):
        """
        Load a cached graph file.
        
        Args:
            cache_path: Path to cached graph file
            
        Returns:
            Loaded graph or None if loading failed
        """
        try:
            if version.parse(torch.__version__) >= version.parse("2.1"):
                return torch.load(cache_path, weights_only=False)
            else:
                return torch.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cached graph {cache_path}: {e}")
            return None

    def _assign_label(self, graph, label_value: float) -> None:
        """
        Assign a label to a graph.
        
        Args:
            graph: Graph object to label
            label_value: Label value to assign
        """
        label_tensor = torch.tensor([label_value], dtype=torch.float)
        
        if isinstance(graph, dict):
            graph["y"] = label_tensor
            graph.pop("label", None)  # Use 'y' consistently
        else:  # Assume PyG Data object
            setattr(graph, "y", label_tensor)
            if hasattr(graph, "label"):
                delattr(graph, "label")

    def _remove_labels(self, graph) -> None:
        """
        Remove labels from a graph.
        
        Args:
            graph: Graph object to remove labels from
        """
        if isinstance(graph, dict):
            graph.pop("y", None)
            graph.pop("label", None)
        else:  # Assume PyG Data object
            if hasattr(graph, "y"):
                delattr(graph, "y")
            if hasattr(graph, "label"):
                delattr(graph, "label")

    def _apply_transforms(self) -> None:
        """Apply transforms to all graphs."""
        if self.transform is None:
            return
        
        transformed_graphs = []
        for i, graph in enumerate(self.graphs):
            transformed_graph = self._apply_transform(
                graph, self.mol_files[i] if i < len(self.mol_files) else "unknown"
            )
            if transformed_graph is not None:
                transformed_graphs.append(transformed_graph)
        
        self.graphs = transformed_graphs

    def _apply_transform(self, graph, file_path: str):
        """
        Apply transform to a single graph.
        
        Args:
            graph: Graph to transform
            file_path: Original file path (for error reporting)
            
        Returns:
            Transformed graph or None if transform failed
        """
        if self.transform is None:
            return graph
        try:
            return self.transform(graph)
        except Exception as e:
            logger.error(f"Transform failed for {file_path}: {e}")
            return None

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int):
        """
        Get a graph by index.
        
        Args:
            idx: Index of graph to retrieve
            
        Returns:
            Molecular graph at the specified index
            
        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.graphs) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.graphs)}")
        
        return self.graphs[idx]


class HierarchicalGraphDataset(Dataset):
    """
    Dataset for hierarchical molecular graphs with multiple levels.
    
    This dataset handles molecular representations at different levels of
    abstraction (e.g., atom-level, functional group level, structural motif
    level). Each molecule is represented by multiple graphs corresponding
    to different hierarchical levels.
    
    Args:
        data_dir: Directory containing hierarchical graph files organized
                 by molecule ID subdirectories
        labels: Optional dictionary mapping molecule IDs to float labels
        levels: List of hierarchy level names to load
        transform: Optional callable to transform graphs after loading
    
    Attributes:
        data_dir: Path to data directory
        labels: Dictionary of labels (if provided)
        levels: List of hierarchy levels
        transform: Transform function
        molecule_dirs: List of molecule directory paths
        molecule_ids: List of molecule IDs
    
    Example:
        >>> data_dir = '/path/to/hierarchical/graphs'
        >>> labels = {'mol_001': 1.0, 'mol_002': 0.5}
        >>> levels = ['atom', 'functional_group']
        >>> dataset = HierarchicalGraphDataset(data_dir, labels, levels)
        >>> graphs_dict = dataset[0]  # Get hierarchical graphs for first molecule
        >>> atom_graph = graphs_dict['atom']
        >>> fg_graph = graphs_dict['functional_group']
    
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no valid molecule directories found
    """

    def __init__(
        self,
        data_dir: str,
        labels: Optional[Dict[str, float]] = None,
        levels: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        """Initialize the hierarchical graph dataset."""
        if levels is None:
            levels = DEFAULT_LEVELS.copy()
            
        self.data_dir = data_dir
        self.labels = labels
        self.levels = levels
        self.transform = transform

        # Validate data directory
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory not found: {data_dir}")
            self.molecule_dirs = []
            self.molecule_ids = []
        elif not os.path.isdir(data_dir):
            raise ValueError(f"Data path is not a directory: {data_dir}")
        else:
            # Find all molecule directories
            self.molecule_dirs = [
                d for d in glob.glob(os.path.join(data_dir, "*")) 
                if os.path.isdir(d)
            ]
            
            if not self.molecule_dirs:
                logger.warning(f"No molecule directories found in {data_dir}")
                self.molecule_dirs = []
                self.molecule_ids = []
            else:
                # Extract molecule IDs
                self.molecule_ids = [
                    os.path.basename(d) for d in self.molecule_dirs
                ]
        
        logger.info(
            f"Initialized HierarchicalGraphDataset with {len(self.molecule_ids)} "
            f"molecules and {len(self.levels)} levels"
        )

    def __len__(self) -> int:
        """Return the number of molecules in the dataset."""
        return len(self.molecule_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get hierarchical graphs for a molecule by index.
        
        Args:
            idx: Index of molecule to retrieve
            
        Returns:
            Dictionary mapping level names to corresponding graphs
            
        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.molecule_dirs) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.molecule_dirs)}"
            )
        
        mol_dir = self.molecule_dirs[idx]
        mol_id = self.molecule_ids[idx]

        # Load graphs for each level
        graphs = {}
        for level in self.levels:
            graph = self._load_level_graph(mol_dir, level)
            if graph is not None:
                graphs[level] = graph
            elif level in DEFAULT_LEVELS:
                # Default levels should be included as None when missing
                graphs[level] = None
            # Custom levels are omitted entirely when missing

        # Add labels if available
        if self.labels is not None and mol_id in self.labels:
            self._assign_labels_to_graphs(graphs, self.labels[mol_id], mol_id)

        # Apply transforms if available
        if self.transform is not None:
            graphs = self._apply_transforms_to_graphs(graphs, mol_id)

        return graphs

    def _load_level_graph(self, mol_dir: str, level: str):
        """
        Load a graph for a specific hierarchical level.
        
        Args:
            mol_dir: Path to molecule directory
            level: Hierarchical level name
            
        Returns:
            Loaded graph or None if not found/failed to load
        """
        # Try PyTorch format first
        graph_path = os.path.join(mol_dir, f"{level}_graph.pt")
        if os.path.exists(graph_path):
            try:
                if version.parse(torch.__version__) >= version.parse("2.1"):
                    return torch.load(graph_path, weights_only=False)
                else:
                    return torch.load(graph_path)
            except Exception as e:
                logger.error(f"Failed to load graph from {graph_path}: {e}")

        # Try JSON format
        json_path = os.path.join(mol_dir, f"{level}{GRAPH_JSON_EXTENSION}")
        if os.path.exists(json_path):
            return self._load_graph_from_json(json_path)

        logger.warning(f"No graph found for level '{level}' in {mol_dir}")
        return None

    def _load_graph_from_json(self, json_path: str):
        """
        Load a graph from JSON representation.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Graph object or None if loading failed
        """
        try:
            with open(json_path, 'r') as json_file:
                graph_dict = json.load(json_file)
            
            if isinstance(graph_dict, dict):
                return Data(**graph_dict)
            else:
                logger.error(f"JSON at {json_path} is not a dict")
                return None
                
        except Exception as e:
            logger.error(f"Error loading graph from JSON {json_path}: {e}")
            return None

    def _assign_labels_to_graphs(
        self, 
        graphs: Dict[str, Any], 
        label_value: float, 
        mol_id: str
    ) -> None:
        """
        Assign labels to all graphs in the hierarchy.
        
        Args:
            graphs: Dictionary of graphs by level
            label_value: Label value to assign
            mol_id: Molecule ID (for logging)
        """
        label_tensor = torch.tensor([label_value], dtype=torch.float)
        
        for level, graph in graphs.items():
            if graph is not None:
                graph.y = label_tensor
            else:
                logger.warning(
                    f"Cannot assign label to None graph for level '{level}' "
                    f"of molecule '{mol_id}'"
                )

    def _apply_transforms_to_graphs(
        self, 
        graphs: Dict[str, Any], 
        mol_id: str
    ) -> Dict[str, Any]:
        """
        Apply transforms to all graphs in the hierarchy.
        
        Args:
            graphs: Dictionary of graphs by level
            mol_id: Molecule ID (for logging)
            
        Returns:
            Dictionary of transformed graphs
        """
        transformed_graphs = {}
        
        for level, graph in graphs.items():
            if graph is not None:
                if self.transform is not None:
                    try:
                        transformed_graphs[level] = self.transform(graph)
                    except Exception as e:
                        logger.error(
                            f"Transform failed for level {level} of molecule "
                            f"{mol_id}: {e}"
                        )
                        transformed_graphs[level] = None
                else:
                    transformed_graphs[level] = graph
            else:
                logger.warning(
                    f"Skipping transform for level {level} as graph is None"
                )
                transformed_graphs[level] = None
        
        return transformed_graphs


class PFASDataset(Dataset):
    """
    Dataset specifically for PFAS (Per- and Polyfluoroalkyl Substances) compounds.
    
    This dataset loads PFAS data from CSV files, processes SMILES strings into
    molecular graphs, and optionally includes additional molecular features and
    targets for machine learning tasks.
    
    Args:
        data_path: Path to CSV file containing PFAS data
        feature_columns: List of column names to use as additional features
        target_column: Column name to use as regression/classification target
        smiles_column: Column name containing SMILES strings
        transform: Optional callable to transform graphs after creation
    
    Attributes:
        df: Pandas DataFrame containing the loaded data
        features: NumPy array of additional features (if specified)
        targets: NumPy array of target values (if specified)
        smiles: List of SMILES strings
        graphs: List of processed molecular graphs
        transform: Transform function
    
    Example:
        >>> data_path = 'pfas_compounds.csv'
        >>> features = ['molecular_weight', 'logp']
        >>> target = 'bioaccumulation_factor'
        >>> dataset = PFASDataset(
        ...     data_path,
        ...     feature_columns=features,
        ...     target_column=target
        ... )
        >>> graph = dataset[0]  # Get first PFAS graph with target
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
        ImportError: If RDKit or graph processor dependencies are missing
    """

    def __init__(
        self,
        data_path: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        smiles_column: str = DEFAULT_SMILES_COLUMN,
        transform: Optional[Callable] = None,
    ):
        """Initialize the PFAS dataset."""
        self.transform = transform

        # Validate dependencies
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for PFASDataset")
        
        if not HAS_GRAPH_PROCESSOR:
            raise ImportError(
                "Graph processor dependencies required for PFASDataset"
            )

        # Load and validate data
        self.df = self._load_and_validate_data(
            data_path, feature_columns, target_column, smiles_column
        )

        # Extract features and targets
        self.features = self._extract_features(feature_columns)
        self.targets = self._extract_targets(target_column)
        self.smiles: Optional[List[str]] = self._extract_smiles(smiles_column)

        # Create molecular graphs
        self.graphs = self._create_molecular_graphs()

    def _load_and_validate_data(
        self,
        data_path: str,
        feature_columns: Optional[List[str]],
        target_column: Optional[str],
        smiles_column: str,
    ) -> pd.DataFrame:
        """
        Load and validate the CSV data.
        
        Args:
            data_path: Path to CSV file
            feature_columns: Feature column names
            target_column: Target column name
            smiles_column: SMILES column name
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {data_path}: {e}")

        # Validate required columns
        if smiles_column not in df.columns:
            logger.warning(f"SMILES column '{smiles_column}' not found. Dataset will be empty.")
            return pd.DataFrame()  # Return empty DataFrame

        if feature_columns:
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(
                    f"Feature columns not found: {missing_features}"
                )

        if target_column and target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found. Proceeding without labels.")

        logger.info(f"Loaded PFAS data with {len(df)} compounds")
        return df

    def _extract_features(
        self, 
        feature_columns: Optional[List[str]]
    ) -> Optional[np.ndarray]:
        """
        Extract feature columns as NumPy array.
        
        Args:
            feature_columns: List of feature column names
            
        Returns:
            Feature array or None if no features specified
        """
        if not feature_columns:
            return None

        try:
            return self.df[feature_columns].values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract features: {e}")
            return None

    def _extract_targets(
        self, 
        target_column: Optional[str]
    ) -> Optional[np.ndarray]:
        """
        Extract target column as NumPy array.
        
        Args:
            target_column: Target column name
            
        Returns:
            Target array or None if no target specified
        """
        if not target_column or target_column not in self.df.columns:
            return None

        try:
            return self.df[target_column].values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract targets: {e}")
            return None

    def _extract_smiles(self, smiles_column: str) -> Optional[List[str]]:
        """
        Extract SMILES strings as list.
        
        Args:
            smiles_column: SMILES column name
            
        Returns:
            List of SMILES strings or None if column missing
        """
        if self.df.empty or smiles_column not in self.df.columns:
            return None
        return self.df[smiles_column].tolist()

    def _create_molecular_graphs(self) -> List[Any]:
        """
        Create molecular graphs from SMILES strings.
        
        Returns:
            List of processed molecular graphs
        """
        if self.smiles is None or not self.smiles:
            logger.warning("No SMILES strings found")
            return []

        # Convert SMILES to RDKit molecules
        molecules, valid_indices = self._smiles_to_molecules()
        
        if not molecules:
            logger.warning("No valid molecules could be created from SMILES")
            return []

        # Process molecules to graphs
        graphs = self._molecules_to_graphs(molecules, valid_indices)
        
        logger.info(
            f"Successfully created {len(graphs)} graphs from "
            f"{len(self.smiles)} SMILES strings"
        )
        
        return graphs

    def _smiles_to_molecules(self) -> tuple:
        """
        Convert SMILES strings to RDKit molecules.
        
        Returns:
            Tuple of (molecules list, valid indices list)
        """
        molecules = []
        valid_indices = []
        
        for i, smiles in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate 3D coordinates if needed
                self._add_3d_coordinates(mol, smiles)
                molecules.append(mol)
                valid_indices.append(i)
            else:
                logger.warning(f"Could not parse SMILES: {smiles}")
        
        return molecules, valid_indices

    def _add_3d_coordinates(self, mol, smiles: str) -> None:
        """
        Add 3D coordinates to molecule if not present.
        
        Args:
            mol: RDKit molecule object
            smiles: Original SMILES string (for logging)
        """
        if mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # type: ignore
                AllChem.UFFOptimizeMolecule(mol)  # type: ignore
            except Exception as e:
                logger.warning(
                    f"Could not generate 3D coordinates for SMILES {smiles}: {e}"
                )

    def _molecules_to_graphs(
        self, 
        molecules: List[Any], 
        valid_indices: List[int]
    ) -> List[Any]:
        """
        Convert RDKit molecules to graph representations.
        
        Args:
            molecules: List of RDKit molecule objects
            valid_indices: List of original DataFrame indices
            
        Returns:
            List of processed graphs
        """
        # Initialize graph processor
        processor_config = getattr(self, "config", {})
        processor = MolecularGraphProcessor(config=processor_config)

        processed_graphs = []
        successful_indices = []

        for i, mol in enumerate(molecules):
            original_idx = valid_indices[i]
            
            try:
                graph = processor.mol_to_graph(mol)  # type: ignore
                
                # Handle targets
                if graph is not None:
                    self._assign_target_to_graph(graph, original_idx)
                    processed_graphs.append(graph)
                    successful_indices.append(original_idx)
                    
            except Exception as e:
                logger.error(
                    f"Failed to process molecule at index {original_idx} "
                    f"(SMILES: {self.smiles[original_idx]}): {e}"
                )

        return processed_graphs

    def _assign_target_to_graph(self, graph, original_idx: int) -> None:
        """
        Assign target value to graph if targets are available.
        
        Args:
            graph: Graph object
            original_idx: Original DataFrame index
        """
        # Remove existing labels first
        if hasattr(graph, "label"):
            del graph.label

        # Assign targets if available
        if self.targets is not None and original_idx < len(self.targets):
            graph.y = torch.tensor(
                [self.targets[original_idx]], 
                dtype=torch.float
            )
        elif hasattr(graph, "y"):
            # Remove y if no targets provided
            del graph.y

    def __len__(self) -> int:
        """Return the number of compounds in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int):
        """
        Get a compound by index.
        
        Args:
            idx: Index of compound to retrieve
            
        Returns:
            Molecular graph (optionally transformed)
            
        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.graphs) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.graphs)}"
            )
        
        graph = self.graphs[idx]

        # Apply transform if available
        if self.transform is not None:
            try:
                graph = self.transform(graph)
            except Exception as e:
                logger.error(f"Transform failed for index {idx}: {e}")
                # Return original graph if transform fails
        
        return graph