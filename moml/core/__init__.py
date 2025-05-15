# -*- coding: utf-8 -*-
#
# Copyright 2025 MoML-CA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
#
# MoML-CA core package for graph processing and feature extraction.
"""
MoML Core Package

Public API:
- MolecularGraphProcessor: build and configure graph processors
- mol_file_to_graph, create_molecular_graph_json, batch_create_graphs_from_molecules: file‑based graph builders
- collate_graphs, graph_to_device, find_charges_file, read_charges_from_file: graph utilities
- GraphCoarsener: hierarchical coarsening of PFAS graphs
- FunctionalGroupDetector, MolecularFeatureExtractor: feature‑extraction helpers
- calculate_molecular_descriptors, extract_fingerprints: single‑molecule descriptor utilities
"""

# Import core processing classes and functions
from .molecular_graph_processor import (
    MolecularGraphProcessor,
    create_graph_processor,
    mol_file_to_graph,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules,
    collate_graphs,
    graph_to_device,
    find_charges_file,
    read_charges_from_file,
)

# Hierarchical graph coarsening
from .hierarchical_graph_coarsener import GraphCoarsener

# Feature extraction and descriptor utilities
from .molecular_feature_extraction import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor,
    calculate_molecular_descriptors,
    extract_fingerprints,
    # validate_smiles is provided in utils
)

__all__ = [
    "MolecularGraphProcessor",
    "create_graph_processor",
    "mol_file_to_graph",
    "create_molecular_graph_json",
    "batch_create_graphs_from_molecules",
    "collate_graphs",
    "graph_to_device",
    "find_charges_file",
    "read_charges_from_file",
    "GraphCoarsener",
    "FunctionalGroupDetector",
    "MolecularFeatureExtractor",
    "calculate_molecular_descriptors",
    "extract_fingerprints",
    # 'validate_smiles', removed as part of core, use utils.validate_smiles
]
