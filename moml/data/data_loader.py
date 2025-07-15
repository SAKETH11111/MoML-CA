"""Unified PFAS data loader for MoML-CA.

This module provides the :class:`PFASDataLoader` which integrates the
existing molecular graph processing utilities with optional quantum
mechanical features and environmental context information.  It returns
PyTorch Geometric ``Data`` objects ready for training MGNN models.
"""

from __future__ import annotations

import os
import glob
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from rdkit import Chem

from moml.core import (
    MolecularGraphProcessor,
    GraphCoarsener,
    collate_graphs,
)
from moml.simulation.qm.parser.orca_parser import parse_orca_output


class PFASDataLoader:
    """Load PFAS molecules and create graph data objects.

    Parameters
    ----------
    data_dir : str
        Base directory containing molecule files and optional metadata.
    config : dict, optional
        Configuration dictionary controlling loader behaviour.  The
        ``graph`` key is passed directly to
        :class:`MolecularGraphProcessor`.  If ``hierarchical`` is set to
        ``True`` the loader will also construct hierarchical graphs using
        :class:`GraphCoarsener`.
    """

    def __init__(self, data_dir: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the PFASDataLoader with data directory and configuration.
        
        Sets up molecular graph processing, optional hierarchical graph construction, environmental feature keys, label types, caching, and validation mode. Loads environment and label data from JSON files, indexes available molecule files, and prepares internal cache.
        """
        self.data_dir = data_dir
        self.config = config or {}

        # Graph processor configuration
        graph_cfg = self.config.get("graph", {})
        self.graph_processor = MolecularGraphProcessor(config=graph_cfg)

        # Hierarchical graph support
        self.coarsener: Optional[GraphCoarsener] = None

        # Environmental and label configuration
        self.env_features = self.config.get(
            "environmental_features",
            [
                "ph",
                "temperature",
                "ionic_strength",
                "flow_rate",
                "residence_time",
                "pressure",
                "dissolved_oxygen",
            ],
        )
        self.label_types = self.config.get("label_types")
        self.cache_enabled = self.config.get("cache_graphs", True)
        self.validation_mode = self.config.get("validation_mode", False)

        self.mol_dir = os.path.join(self.data_dir, "molecules")
        self.qm_dir = os.path.join(self.data_dir, "qm")
        self.env_path = os.path.join(self.data_dir, "environment.json")
        self.labels_path = os.path.join(self.data_dir, "labels.json")

        self.environment = self._load_json(self.env_path)
        self.labels = self._load_json(self.labels_path)
        self.index = self._find_molecule_files()
        self.cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _assign_labels_to_graph(self, graph, labels_data):
        """Helper method to assign labels to a graph."""
        if isinstance(labels_data, dict):
            if "force_field_params" in labels_data:
                graph.y_ff = torch.tensor(labels_data["force_field_params"], dtype=torch.float)
            if "molecular_properties" in labels_data:
                graph.y_props = torch.tensor(labels_data["molecular_properties"], dtype=torch.float)
            if "adsorption_potential" in labels_data:
                graph.y_ads = torch.tensor([labels_data["adsorption_potential"]], dtype=torch.float)
        elif labels_data is not None:
            graph.y = torch.tensor([labels_data], dtype=torch.float)
        if self.config.get("hierarchical"):
            graph_cfg = self.config.get("graph", {}) # Access graph_cfg from self.config
            self.coarsener = GraphCoarsener(
                use_3d_coords=graph_cfg.get("use_3d_coords", True),
                use_pfas_features=graph_cfg.get("use_pfas_specific_features", True),
            )

    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        Safely loads and parses a JSON file from the specified path.
        
        Returns an empty dictionary if the file does not exist or if parsing fails.
        """
        if os.path.exists(path):
            with open(path) as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}

    def _find_molecule_files(self) -> Dict[str, str]:
        """
        Indexes molecule files by searching multiple directories for supported file types.
        
        Returns:
            A dictionary mapping molecule IDs (filenames without extension) to their file paths for all found molecule files with extensions .mol, .sdf, .pdb, or .mol2.
        """
        search_paths = [
            os.path.join(self.data_dir, "molecules"),
            os.path.join(self.data_dir, "mol_files"),
            os.path.join(self.data_dir, "structures"),
            self.data_dir,
        ]
        idx: Dict[str, str] = {}
        for spath in search_paths:
            if not os.path.isdir(spath):
                continue
            for path in glob.glob(os.path.join(spath, "*")):
                if os.path.isfile(path) and path.lower().endswith((".mol", ".sdf", ".pdb", ".mol2")):
                    base = os.path.splitext(os.path.basename(path))[0]
                    idx[base] = path
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_environmental_features(self, graph_data: Data, env_params: Dict[str, float]) -> Data:
        """
        Adds environmental context features to a molecular graph as global attributes.
        
        The function encodes environmental variables such as pH, temperature, ionic strength, flow rate, residence time, pressure, and dissolved oxygen into a fixed-order vector and attaches it as the `u` attribute of the graph. If co-contaminants are present in the environment parameters, they are added as a separate attribute.
        
        Args:
            graph_data: The molecular graph to augment.
            env_params: Dictionary of environmental variables to encode.
        
        Returns:
            The graph with added environmental features.
        """

        if not env_params:
            return graph_data

        features = [
            env_params.get("ph", 7.0),
            env_params.get("temperature", 298.15),
            env_params.get("ionic_strength", 0.0),
            env_params.get("flow_rate", 0.0),
            env_params.get("residence_time", 0.0),
            env_params.get("pressure", 101325.0),
            env_params.get("dissolved_oxygen", 0.0),
        ]

        feature_order = [
            "ph",
            "temperature",
            "ionic_strength",
            "flow_rate",
            "residence_time",
            "pressure",
            "dissolved_oxygen",
        ]
        vec = []
        for name in self.env_features:
            if name in feature_order:
                idx = feature_order.index(name)
                vec.append(features[idx])
        graph_data.u = torch.tensor(vec, dtype=torch.float)

        # Store co-contaminants separately if present
        if "co_contaminants" in env_params:
            graph_data.co_contaminants = env_params["co_contaminants"]

        return graph_data

    def load_molecule_by_id(self, mol_id: str) -> Tuple[Any, Optional[float], Optional[Dict[str, float]]]:
        """
        Loads a molecule graph by its identifier, optionally enriching it with quantum mechanical charges, environmental features, and labels.
        
        If hierarchical graph construction is enabled, returns a dictionary of hierarchical graphs with corresponding features and labels. Otherwise, returns a single atom-level graph. Also returns the label data and environmental context dictionary if available.
        
        Args:
            mol_id: The identifier of the molecule to load.
        
        Returns:
            A tuple containing the graph data (atom-level or hierarchical), the label data (if available), and the environmental context dictionary.
        """
        if self.cache_enabled and mol_id in self.cache:
            return self.cache[mol_id]

        mol_path = self.index.get(mol_id)
        if not mol_path:
            raise ValueError(f"No molecule file for id {mol_id}")

        additional: Dict[str, List[float]] = {}
        qm_file = os.path.join(self.qm_dir, f"{mol_id}.out")
        if os.path.exists(qm_file):
            try:
                qm = parse_orca_output(qm_file)
                charges = qm.get("mulliken_charges")
                if charges:
                    additional["partial_charges"] = list(charges) # Ensure it's a list of floats
            except Exception:
                pass

        graph = self.graph_processor.file_to_graph(mol_path, additional)
        env = self.environment.get(mol_id, {})
        if env and graph is not None: # Ensure graph is not None before adding features
            graph = self.add_environmental_features(graph, env)
        labels_data = self.labels.get(mol_id)
        label = labels_data
        self._assign_labels_to_graph(graph, labels_data)

        result: Any = graph
        if self.coarsener is not None:
            mol = Chem.MolFromMolFile(mol_path, removeHs=False)
            if graph is not None: # Ensure graph is not None before passing to create_hierarchical_graphs
                hier_graphs = self.coarsener.create_hierarchical_graphs(graph, mol)
                if env:
                    for g in hier_graphs.values():
                        self.add_environmental_features(g, env)
                for g in hier_graphs.values():
                    self._assign_labels_to_graph(g, labels_data)
                result = hier_graphs
            else:
                # Handle the case where graph is None, perhaps log a warning or return an empty dict
                # For now, let's assume we skip hierarchical graph creation if atom-level graph is None
                hier_graphs = {}
                result = graph # Or handle as appropriate for your application

        if self.cache_enabled:
            self.cache[mol_id] = (result, label, env)
        return result, label, env

    def get_batch(self, mol_ids: List[str], batch_size: int = 32) -> Data:
        """
        Loads and batches molecular graphs for a list of molecule IDs.
        
        For each molecule ID, loads the corresponding graph (extracting the atom-level graph if hierarchical graphs are present) and collates them into a single batched PyTorch Geometric Data object.
        
        Args:
            mol_ids: List of molecule identifiers to load and batch.
            batch_size: Maximum number of molecules to include in the batch.
        
        Returns:
            A batched PyTorch Geometric Data object containing up to batch_size molecular graphs.
        """
        graphs = []
        for mid in mol_ids:
            graph, _, _ = self.load_molecule_by_id(mid)
            if isinstance(graph, dict):
                # If hierarchical, take atom level for batching
                graph = graph.get("atom")
            graphs.append(graph)
        return collate_graphs(graphs[:batch_size])

    def load_dataset(self, split: str = "train") -> List[Any]:
        """
        Loads all molecule graphs from a specified dataset split directory.
        
        Scans the split directory for molecule files or molecule IDs, loads each molecule graph by its identifier, and returns a list of graph objects suitable for model training or evaluation.
        
        Args:
            split: Name of the dataset split to load (e.g., "train", "test").
        
        Returns:
            A list of graph objects corresponding to the molecules in the specified split.
        """
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        mol_ids = []
        for path in glob.glob(os.path.join(split_dir, "*")):
            if os.path.isfile(path) and path.lower().endswith((".mol", ".sdf", ".pdb", ".mol2")):
                mol_ids.append(os.path.splitext(os.path.basename(path))[0])
        if not mol_ids:
            # fallback to index intersection if split dir contains ids only
            for mid in os.listdir(split_dir):
                if mid in self.index:
                    mol_ids.append(mid)
        dataset = []
        for mid in mol_ids:
            dataset.append(self.load_molecule_by_id(mid)[0])
        return dataset
