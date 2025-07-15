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

# Import core processing classes and functions with error handling
try:
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
except ImportError as e:
    # Capture the exception message in a local variable
    processor_import_error_msg = str(e)

    # Create dummy functions when dependencies are not available
    def MolecularGraphProcessor(*args, **kwargs):
        raise ImportError(
            f"MolecularGraphProcessor requires additional dependencies: {processor_import_error_msg}"
        )

    def create_graph_processor(*args, **kwargs):
        raise ImportError(
            f"create_graph_processor requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def mol_file_to_graph(*args, **kwargs):
        raise ImportError(
            f"mol_file_to_graph requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def create_molecular_graph_json(*args, **kwargs):
        raise ImportError(
            f"create_molecular_graph_json requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def batch_create_graphs_from_molecules(*args, **kwargs):
        raise ImportError(
            f"batch_create_graphs_from_molecules requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def collate_graphs(*args, **kwargs):
        raise ImportError(
            f"collate_graphs requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def graph_to_device(*args, **kwargs):
        raise ImportError(
            f"graph_to_device requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def find_charges_file(*args, **kwargs):
        raise ImportError(
            f"find_charges_file requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )

    def read_charges_from_file(*args, **kwargs):
        raise ImportError(
            f"read_charges_from_file requires additional dependencies (torch_geometric, rdkit): {processor_import_error_msg}"
        )


# Hierarchical graph coarsening
try:
    from .hierarchical_graph_coarsener import GraphCoarsener
except ImportError as e:
    # Capture the exception message in a local variable
    coarsener_import_error_msg = str(e)

    def GraphCoarsener(*args, **kwargs):
        raise ImportError(
            f"GraphCoarsener requires additional dependencies (torch_geometric, rdkit): {coarsener_import_error_msg}"
        )


# Feature extraction and descriptor utilities
try:
    from .molecular_feature_extraction import (
        FunctionalGroupDetector,
        MolecularFeatureExtractor,
        calculate_molecular_descriptors,
        extract_fingerprints,
        # validate_smiles is provided in utils
    )
except ImportError as e:
    # Capture the exception message in a local variable
    feature_import_error_msg = str(e)

    def FunctionalGroupDetector(*args, **kwargs):
        raise ImportError(
            f"FunctionalGroupDetector requires additional dependencies (rdkit): {feature_import_error_msg}"
        )

    def MolecularFeatureExtractor(*args, **kwargs):
        raise ImportError(
            f"MolecularFeatureExtractor requires additional dependencies (rdkit): {feature_import_error_msg}"
        )

    def calculate_molecular_descriptors(*args, **kwargs):
        raise ImportError(
            f"calculate_molecular_descriptors requires additional dependencies (rdkit): {feature_import_error_msg}"
        )

    def extract_fingerprints(*args, **kwargs):
        raise ImportError(f"extract_fingerprints requires additional dependencies (rdkit): {feature_import_error_msg}")


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
