"""
moml/data/data_loader.py

Unified PFAS data loader for molecular machine learning applications.

This module provides the PFASDataLoader class which integrates molecular graph
processing utilities with optional quantum mechanical features and environmental
context information. It returns PyTorch Geometric Data objects ready for
training MGNN models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from rdkit import Chem
from torch_geometric.data import Data

from moml.core import GraphCoarsener, MolecularGraphProcessor, collate_graphs
from moml.simulation.qm.parser.orca_parser import parse_orca_output

# Constants
DEFAULT_ENV_FEATURES = [
    "ph",
    "temperature", 
    "ionic_strength",
    "flow_rate",
    "residence_time",
    "pressure",
    "dissolved_oxygen",
]

DEFAULT_ENV_VALUES = {
    "ph": 7.0,
    "temperature": 298.15,
    "ionic_strength": 0.0,
    "flow_rate": 0.0,
    "residence_time": 0.0,
    "pressure": 101325.0,
    "dissolved_oxygen": 0.0,
}

SUPPORTED_MOL_FORMATS = (".mol", ".sdf", ".pdb", ".mol2")


class PFASDataLoader:
    """
    Load PFAS molecules and create graph data objects.

    This class provides comprehensive data loading capabilities for PFAS
    molecular datasets, including support for quantum mechanical properties,
    environmental context features, and hierarchical graph representations.

    Args:
        data_dir: Base directory containing molecule files and optional metadata.
        config: Configuration dictionary controlling loader behavior. The
            'graph' key is passed directly to MolecularGraphProcessor.
            If 'hierarchical' is set to True, the loader will construct
            hierarchical graphs using GraphCoarsener.

    Attributes:
        data_dir: Path to the data directory.
        config: Configuration dictionary for the loader.
        graph_processor: Molecular graph processor instance.
        coarsener: Optional hierarchical graph coarsener.
        env_features: List of environmental feature names.
        label_types: Optional label type configuration.
        cache_enabled: Whether graph caching is enabled.
        validation_mode: Whether validation mode is active.
    """

    def __init__(
        self, 
        data_dir: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the PFASDataLoader with data directory and configuration.

        Sets up molecular graph processing, optional hierarchical graph
        construction, environmental feature keys, label types, caching,
        and validation mode. Loads environment and label data from JSON
        files and indexes available molecule files.

        Args:
            data_dir: Directory containing molecule files and metadata.
            config: Optional configuration dictionary for customizing
                loader behavior and graph processing parameters.
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}

        # Initialize graph processor
        graph_cfg = self.config.get("graph", {})
        self.graph_processor = MolecularGraphProcessor(config=graph_cfg)

        # Initialize hierarchical graph support
        self.coarsener: Optional[GraphCoarsener] = None
        if self.config.get("hierarchical", False):
            self.coarsener = GraphCoarsener(
                use_3d_coords=graph_cfg.get("use_3d_coords", True),
                use_pfas_features=graph_cfg.get("use_pfas_specific_features", True),
            )

        # Configure environmental and label settings
        self.env_features = self.config.get(
            "environmental_features", DEFAULT_ENV_FEATURES
        )
        self.label_types = self.config.get("label_types")
        self.cache_enabled = self.config.get("cache_graphs", True)
        self.validation_mode = self.config.get("validation_mode", False)

        # Set up directory paths
        self.mol_dir = self.data_dir / "molecules"
        self.qm_dir = self.data_dir / "qm"
        self.env_path = self.data_dir / "environment.json"
        self.labels_path = self.data_dir / "labels.json"

        # Load metadata and create file index
        self.environment = self._load_json(self.env_path)
        self.labels = self._load_json(self.labels_path)
        self.index = self._find_molecule_files()
        self.cache: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """
        Safely load and parse a JSON file from the specified path.

        Args:
            path: Path to the JSON file to load.

        Returns:
            Dictionary containing the parsed JSON data, or empty
            dictionary if file doesn't exist or parsing fails.
        """
        if not path.exists():
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # TODO: Log error in production code
            return {}

    def _find_molecule_files(self) -> Dict[str, str]:
        """
        Index molecule files by searching multiple directories.

        Searches for supported molecule file formats across multiple
        potential directories and creates a mapping from molecule IDs
        (filenames without extension) to their full file paths.

        Returns:
            Dictionary mapping molecule IDs to their file paths for all
            found molecule files with supported extensions.
        """
        search_paths = [
            self.data_dir / "molecules",
            self.data_dir / "mol_files", 
            self.data_dir / "structures",
            self.data_dir,
        ]

        index: Dict[str, str] = {}
        for search_path in search_paths:
            if not search_path.is_dir():
                continue

            for path in search_path.iterdir():
                if (
                    path.is_file() 
                    and path.suffix.lower() in SUPPORTED_MOL_FORMATS
                ):
                    mol_id = path.stem
                    index[mol_id] = str(path)

        return index

    def _assign_labels_to_graph(
        self, 
        graph: Data, 
        labels_data: Any
    ) -> None:
        """
        Assign labels to a graph object based on label data structure.

        Args:
            graph: PyTorch Geometric Data object to assign labels to.
            labels_data: Label data which can be a dictionary with specific
                label types or a single numeric value.
        """
        if isinstance(labels_data, dict):
            # Handle structured label data
            if "force_field_params" in labels_data:
                graph.y_ff = torch.tensor(
                    labels_data["force_field_params"], dtype=torch.float
                )
            if "molecular_properties" in labels_data:
                graph.y_props = torch.tensor(
                    labels_data["molecular_properties"], dtype=torch.float
                )
            if "adsorption_potential" in labels_data:
                graph.y_ads = torch.tensor(
                    [labels_data["adsorption_potential"]], dtype=torch.float
                )
        elif labels_data is not None:
            # Handle simple numeric labels
            graph.y = torch.tensor([labels_data], dtype=torch.float)

    def add_environmental_features(
        self, 
        graph_data: Data, 
        env_params: Dict[str, float]
    ) -> Data:
        """
        Add environmental context features to a molecular graph.

        Encodes environmental variables such as pH, temperature, ionic
        strength, flow rate, residence time, pressure, and dissolved
        oxygen into a fixed-order vector and attaches it as the 'u'
        attribute of the graph.

        Args:
            graph_data: The molecular graph to augment with environmental
                features.
            env_params: Dictionary of environmental variables to encode.

        Returns:
            The graph with added environmental features as global attributes.
        """
        if not env_params:
            return graph_data

        # Build feature vector in consistent order
        feature_vector = []
        for feature_name in self.env_features:
            default_value = DEFAULT_ENV_VALUES.get(feature_name, 0.0)
            value = env_params.get(feature_name, default_value)
            feature_vector.append(value)

        graph_data.u = torch.tensor(feature_vector, dtype=torch.float)

        # Store co-contaminants separately if present
        if "co_contaminants" in env_params:
            graph_data.co_contaminants = env_params["co_contaminants"]

        return graph_data

    def load_molecule_by_id(
        self, 
        mol_id: str
    ) -> Tuple[Union[Data, Dict[str, Data]], Any, Dict[str, Any]]:
        """
        Load a molecule graph by its identifier with optional enrichment.

        Loads molecular graph data, optionally enriching it with quantum
        mechanical charges, environmental features, and labels. If
        hierarchical graph construction is enabled, returns a dictionary
        of hierarchical graphs; otherwise returns a single atom-level graph.

        Args:
            mol_id: The identifier of the molecule to load.

        Returns:
            Tuple containing:
                - Graph data (atom-level Data object or dictionary of
                  hierarchical Data objects)
                - Label data if available
                - Environmental context dictionary if available

        Raises:
            ValueError: If no molecule file exists for the given ID.
        """
        # Check cache first
        if self.cache_enabled and mol_id in self.cache:
            return self.cache[mol_id]

        # Find molecule file
        mol_path = self.index.get(mol_id)
        if not mol_path:
            raise ValueError(f"No molecule file found for ID: {mol_id}")

        # Load quantum mechanical data if available
        additional_features: Dict[str, List[float]] = {}
        qm_file = self.qm_dir / f"{mol_id}.out"
        if qm_file.exists():
            try:
                qm_data = parse_orca_output(str(qm_file))
                charges = qm_data.get("mulliken_charges")
                if charges:
                    additional_features["partial_charges"] = list(charges)
            except Exception:
                # QM parsing failed, continue without charges
                pass

        # Process molecule to graph
        graph = self.graph_processor.file_to_graph(mol_path, additional_features)
        if graph is None:
            raise ValueError(f"Failed to process molecule file: {mol_path}")

        # Add environmental features
        env_data = self.environment.get(mol_id, {})
        if env_data:
            graph = self.add_environmental_features(graph, env_data)

        # Assign labels
        labels_data = self.labels.get(mol_id)
        if labels_data is not None:
            self._assign_labels_to_graph(graph, labels_data)

        # Create hierarchical graphs if enabled
        result: Union[Data, Dict[str, Data]] = graph
        if self.coarsener is not None:
            mol = Chem.MolFromMolFile(mol_path, removeHs=False)
            if mol is not None:
                hier_graphs = self.coarsener.create_hierarchical_graphs(
                    graph, mol
                )
                
                # Add environmental features and labels to hierarchical graphs
                for level_graph in hier_graphs.values():
                    if env_data:
                        self.add_environmental_features(level_graph, env_data)
                    if labels_data is not None:
                        self._assign_labels_to_graph(level_graph, labels_data)
                
                result = hier_graphs

        # Cache result
        if self.cache_enabled:
            self.cache[mol_id] = (result, labels_data, env_data)

        return result, labels_data, env_data

    def get_batch(
        self, 
        mol_ids: List[str], 
        batch_size: int = 32
    ) -> Data:
        """
        Load and batch molecular graphs for a list of molecule IDs.

        For each molecule ID, loads the corresponding graph (extracting
        the atom-level graph if hierarchical graphs are present) and
        collates them into a single batched PyTorch Geometric Data object.

        Args:
            mol_ids: List of molecule identifiers to load and batch.
            batch_size: Maximum number of molecules to include in the batch.

        Returns:
            A batched PyTorch Geometric Data object containing up to
            batch_size molecular graphs.
        """
        graphs = []
        for mol_id in mol_ids[:batch_size]:
            try:
                graph, _, _ = self.load_molecule_by_id(mol_id)
                
                # Extract atom-level graph if hierarchical
                if isinstance(graph, dict):
                    graph = graph.get("atom")
                
                if graph is not None:
                    graphs.append(graph)
            except (ValueError, Exception):
                # Skip molecules that fail to load
                continue

        if not graphs:
            raise ValueError("No valid graphs could be loaded from the provided IDs")

        return collate_graphs(graphs)

    def load_dataset(self, split: str = "train") -> List[Union[Data, Dict[str, Data]]]:
        """
        Load all molecule graphs from a specified dataset split directory.

        Scans the split directory for molecule files or molecule IDs,
        loads each molecule graph by its identifier, and returns a list
        of graph objects suitable for model training or evaluation.

        Args:
            split: Name of the dataset split to load (e.g., "train", "test").

        Returns:
            List of graph objects corresponding to the molecules in the
            specified split.

        Raises:
            ValueError: If the split directory doesn't exist.
        """
        split_dir = self.data_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Find molecule IDs in split directory
        mol_ids = []
        
        # First, look for molecule files directly
        for path in split_dir.iterdir():
            if (
                path.is_file() 
                and path.suffix.lower() in SUPPORTED_MOL_FORMATS
            ):
                mol_ids.append(path.stem)
        
        # If no files found, look for ID references
        if not mol_ids:
            for item in split_dir.iterdir():
                if item.name in self.index:
                    mol_ids.append(item.name)

        if not mol_ids:
            raise ValueError(f"No molecule files or IDs found in {split_dir}")

        # Load all molecules in the split
        dataset = []
        for mol_id in mol_ids:
            try:
                graph, _, _ = self.load_molecule_by_id(mol_id)
                dataset.append(graph)
            except (ValueError, Exception):
                # Skip molecules that fail to load
                continue

        return dataset
