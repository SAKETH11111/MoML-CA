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
from moml.simulation.quantum_mechanics.parser.orca_parser import parse_orca_output


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
        self.data_dir = data_dir
        self.config = config or {}
        graph_cfg = self.config.get("graph", {})
        self.graph_processor = MolecularGraphProcessor(config=graph_cfg)
        self.coarsener: Optional[GraphCoarsener] = None
        if self.config.get("hierarchical"):
            self.coarsener = GraphCoarsener(
                use_3d_coords=graph_cfg.get("use_3d_coords", True),
                use_pfas_features=graph_cfg.get("use_pfas_specific_features", True),
            )

        self.mol_dir = os.path.join(self.data_dir, "molecules")
        self.qm_dir = os.path.join(self.data_dir, "qm")
        self.env_path = os.path.join(self.data_dir, "environment.json")
        self.labels_path = os.path.join(self.data_dir, "labels.json")

        self.environment = self._load_json(self.env_path)
        self.labels = self._load_json(self.labels_path)
        self.index = self._create_index()
        self.cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _load_json(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path) as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}

    def _create_index(self) -> Dict[str, str]:
        idx = {}
        if not os.path.isdir(self.mol_dir):
            return idx
        for path in glob.glob(os.path.join(self.mol_dir, "*")):
            if os.path.isfile(path) and path.lower().endswith((".mol", ".sdf", ".pdb", ".mol2")):
                base = os.path.splitext(os.path.basename(path))[0]
                idx[base] = path
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_environmental_features(self, graph_data: Data, env_params: Dict[str, float]) -> Data:
        """Attach environmental context to a graph as global features."""
        if not env_params:
            return graph_data
        vec = torch.tensor(
            [
                float(env_params.get("ph", 7.0)),
                float(env_params.get("temperature", 298.15)),
                float(env_params.get("ionic_strength", 0.0)),
            ],
            dtype=torch.float,
        )
        graph_data.u = vec
        return graph_data

    def load_molecule_by_id(self, mol_id: str) -> Tuple[Any, Optional[float], Optional[Dict[str, float]]]:
        """Load a molecule by identifier.

        Returns the graph data (atom level or hierarchical dict), the label if
        available and the environmental context dictionary.
        """
        if mol_id in self.cache:
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
                    additional["partial_charges"] = charges
            except Exception:
                pass

        graph = self.graph_processor.file_to_graph(mol_path, additional)
        env = self.environment.get(mol_id, {})
        if env:
            graph = self.add_environmental_features(graph, env)
        label = self.labels.get(mol_id)
        if label is not None:
            graph.y = torch.tensor([label], dtype=torch.float)

        result: Any = graph
        if self.coarsener is not None:
            mol = Chem.MolFromMolFile(mol_path, removeHs=False)
            hier_graphs = self.coarsener.create_hierarchical_graphs(graph, mol)
            if env:
                for g in hier_graphs.values():
                    self.add_environmental_features(g, env)
            if label is not None:
                for g in hier_graphs.values():
                    g.y = torch.tensor([label], dtype=torch.float)
            result = hier_graphs

        self.cache[mol_id] = (result, label, env)
        return result, label, env

    def get_batch(self, mol_ids: List[str], batch_size: int = 32) -> Data:
        """Return a batched PyG ``Data`` object for a list of molecule ids."""
        graphs = []
        for mid in mol_ids:
            graph, _, _ = self.load_molecule_by_id(mid)
            if isinstance(graph, dict):
                # If hierarchical, take atom level for batching
                graph = graph.get("atom")
            graphs.append(graph)
        return collate_graphs(graphs[:batch_size])

    def load_dataset(self, split: str = "train") -> List[Any]:
        """Load all molecules from a split directory.

        The split directories are expected to be located under
        ``<data_dir>/<split>`` and contain molecule files named by the
        molecule identifier.
        """
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        mol_ids = []
        for path in glob.glob(os.path.join(split_dir, "*")):
            if os.path.isfile(path) and path.lower().endswith((".mol", ".sdf", ".pdb", ".mol2")):
                mol_ids.append(os.path.splitext(os.path.basename(path))[0])
        dataset = []
        for mid in mol_ids:
            dataset.append(self.load_molecule_by_id(mid)[0])
        return dataset
