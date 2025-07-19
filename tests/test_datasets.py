"""
tests/test_datasets.py

Unit tests for the Dataset classes in moml.data.datasets.
"""

import json
import logging
import os
import shutil
import tempfile
from typing import List, Dict, Generator

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

from moml.data.datasets import MolecularGraphDataset, HierarchicalGraphDataset, PFASDataset


def create_dummy_mol_file(dir_path: str, filename: str, smiles: str) -> str:
    """
    Create a dummy .sdf file in the given directory.

    Args:
        dir_path (str): Directory path.
        filename (str): File name.
        smiles (str): SMILES string.

    Returns:
        str: Path to the created file.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # type: ignore[attr-defined]
    if embed_result == -1:
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)  # type: ignore[attr-defined]
        if embed_result == -1:
            print(f"Warning: Could not generate 3D coordinates for {smiles}")
    filepath = os.path.join(dir_path, filename)
    writer = Chem.SDWriter(filepath)
    writer.write(mol)
    writer.close()
    return filepath


def create_dummy_pt_graph_file(dir_path: str, filename: str, num_nodes: int = 5) -> str:
    """
    Create a dummy .pt (PyTorch Geometric Data) file.

    Args:
        dir_path (str): Directory path.
        filename (str): File name.
        num_nodes (int): Number of nodes.

    Returns:
        str: Path to the created file.
    """
    filepath = os.path.join(dir_path, filename)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    if num_nodes == 0:
        x = torch.empty(0, 16)
        edge_index = torch.empty(2, 0, dtype=torch.long)
    elif num_nodes < 5:
        x = torch.randn(num_nodes, 16)
        edge_index = (
            torch.tensor(
                [[i % num_nodes for i in range(num_nodes - 1)], [(i + 1) % num_nodes for i in range(num_nodes - 1)]],
                dtype=torch.long,
            )
            if num_nodes > 1
            else torch.empty(2, 0, dtype=torch.long)
        )
    else:
        x = torch.randn(num_nodes, 16)
    data = Data(x=x, edge_index=edge_index.contiguous())
    data.y = torch.tensor([0.5], dtype=torch.float)
    torch.save(data, filepath)
    return filepath


def create_dummy_json_graph_file(dir_path: str, filename: str) -> str:
    """
    Create a dummy .json graph file.

    Args:
        dir_path (str): Directory path.
        filename (str): File name.

    Returns:
        str: Path to the created file.
    """
    filepath = os.path.join(dir_path, filename)
    graph_data = {
        "nodes": [{"id": i, "features": [0.1 * j for j in range(5)]} for i in range(3)],
        "edges": [
            {"source": 0, "target": 1, "features": [0.2, 0.3]},
            {"source": 1, "target": 2, "features": [0.4, 0.5]},
        ],
        "graph_features": {"label": 0.75},
    }
    with open(filepath, "w") as f:
        json.dump(graph_data, f)
    return filepath


@pytest.fixture(scope="module")
def temp_data_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for dataset files.

    Yields:
        str: Path to the temporary directory.
    """
    dir_path = tempfile.mkdtemp(prefix="moml_test_datasets_")
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def dummy_mol_files_sdf(temp_data_dir: str) -> List[str]:
    """
    Create a few dummy SDF molecule files.

    Args:
        temp_data_dir (str): Path to the temporary data directory.

    Returns:
        List[str]: List of file paths to the dummy SDF files.
    """
    files = [
        create_dummy_mol_file(temp_data_dir, "mol1.sdf", "CCO"),
        create_dummy_mol_file(temp_data_dir, "mol2.sdf", "C1=CC=CC=C1"),
        create_dummy_mol_file(temp_data_dir, "mol3.sdf", "CC(C)C"),
    ]
    return files


@pytest.fixture
def dummy_labels_for_mol_files(dummy_mol_files_sdf: List[str]) -> Dict[str, float]:
    """
    Create dummy labels for the molecule files.

    Args:
        dummy_mol_files_sdf (List[str]): List of molecule file paths.

    Returns:
        Dict[str, float]: Dictionary mapping file paths to labels.
    """
    return {f: float(i + 1) for i, f in enumerate(dummy_mol_files_sdf)}


@pytest.fixture
def dummy_hierarchical_data_dir(temp_data_dir: str) -> str:
    """
    Create a dummy directory structure for HierarchicalGraphDataset.

    Args:
        temp_data_dir (str): Path to the temporary data directory.

    Returns:
        str: Path to the hierarchical data directory.
    """
    hier_dir = os.path.join(temp_data_dir, "hier_data")
    os.makedirs(hier_dir, exist_ok=True)

    mol1_dir = os.path.join(hier_dir, "molA")
    mol2_dir = os.path.join(hier_dir, "molB")
    mol3_dir = os.path.join(hier_dir, "molC_empty")
    os.makedirs(mol1_dir, exist_ok=True)
    os.makedirs(mol2_dir, exist_ok=True)
    os.makedirs(mol3_dir, exist_ok=True)

    create_dummy_pt_graph_file(mol1_dir, "atom_graph.pt")
    create_dummy_pt_graph_file(mol1_dir, "functional_group_graph.pt")
    create_dummy_json_graph_file(mol1_dir, "structural_motif_graph.json")

    create_dummy_pt_graph_file(mol2_dir, "atom_graph.pt")
    create_dummy_json_graph_file(mol2_dir, "structural_motif_graph.json")
    create_dummy_pt_graph_file(mol3_dir, "atom_graph.pt")
    return hier_dir


@pytest.fixture
def dummy_labels_for_hier_data() -> Dict[str, float]:
    """
    Create dummy labels for hierarchical data.

    Returns:
        Dict[str, float]: Dictionary mapping names to labels.
    """
    return {"molA": 10.0, "molB": 20.0, "molC_empty": 30.0}


@pytest.fixture
def dummy_pfas_csv_file(temp_data_dir: str) -> str:
    """
    Create a dummy CSV file for PFASDataset.

    Args:
        temp_data_dir (str): Path to the temporary data directory.

    Returns:
        str: Path to the CSV file.
    """
    csv_path = os.path.join(temp_data_dir, "pfas_data.csv")
    data = {
        "smiles": ["CC(F)(F)C(=O)O", "CCC(F)(F)C(=O)O", "InvalidSMILES", "CCCC", "C"],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "target_property": [100.0, 200.0, 300.0, 400.0, 500.0],
        "another_target": [1.1, 2.2, 3.3, 4.4, 5.5],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


def simple_transform(graph: Data) -> Data:
    """
    A simple transform function for testing.

    Args:
        graph (Data): PyTorch Geometric Data object.

    Returns:
        Data: Transformed Data object.
    """
    if hasattr(graph, "x") and graph.x is not None:
        graph.x = graph.x * 2
    return graph


class TestMolecularGraphDataset:
    def test_initialization_and_len(self, dummy_mol_files_sdf: List[str], dummy_labels_for_mol_files: Dict[str, float]):
        dataset = MolecularGraphDataset(mol_files=dummy_mol_files_sdf, labels=dummy_labels_for_mol_files)
        assert len(dataset) == len(dummy_mol_files_sdf)
        assert len(dataset.graphs) == len(dummy_mol_files_sdf)

    def test_getitem(self, dummy_mol_files_sdf: List[str], dummy_labels_for_mol_files: Dict[str, float]):
        dataset = MolecularGraphDataset(mol_files=dummy_mol_files_sdf, labels=dummy_labels_for_mol_files)
        graph_item = dataset[0]
        assert isinstance(graph_item, Data)
        assert hasattr(graph_item, "y")
        expected_label = dummy_labels_for_mol_files[dummy_mol_files_sdf[0]]
        if isinstance(graph_item.y, torch.Tensor):
            assert torch.allclose(graph_item.y, torch.tensor([expected_label], dtype=torch.float))

    def test_transform(self, dummy_mol_files_sdf: List[str]):
        dataset_no_transform = MolecularGraphDataset(mol_files=[dummy_mol_files_sdf[0]])
        original_x = dataset_no_transform[0].x.clone() if isinstance(dataset_no_transform[0].x, torch.Tensor) else None

        dataset_with_transform = MolecularGraphDataset(mol_files=[dummy_mol_files_sdf[0]], transform=simple_transform)
        transformed_graph = dataset_with_transform[0]
        if original_x is not None and isinstance(transformed_graph.x, torch.Tensor):
            assert torch.allclose(transformed_graph.x, original_x * 2)
        else:
            assert transformed_graph.x is None

    def test_caching_behavior(self, dummy_mol_files_sdf: List[str], temp_data_dir: str):
        """Test that .pt cache files are used if they exist."""
        mol_file_path = dummy_mol_files_sdf[0]
        cache_file_path = mol_file_path + ".pt"

        # Create a dummy graph and save it as if it were cached
        # Ensure this cached graph is different from what would be processed
        cached_data = Data(x=torch.ones(3, 3) * 123, edge_index=torch.tensor([[0], [1]]))
        torch.save(cached_data, cache_file_path)

        dataset = MolecularGraphDataset(mol_files=[mol_file_path])
        loaded_graph = dataset[0]
        if isinstance(loaded_graph.x, torch.Tensor) and isinstance(cached_data.x, torch.Tensor):
            assert torch.allclose(loaded_graph.x, cached_data.x)  # Check if cached version was loaded
        os.remove(cache_file_path)

    def test_init_no_labels(self, dummy_mol_files_sdf: List[str]):
        """Test MolecularGraphDataset initialization without labels."""
        dataset = MolecularGraphDataset(mol_files=dummy_mol_files_sdf)
        assert len(dataset) == len(dummy_mol_files_sdf)
        for graph in dataset.graphs:
            if hasattr(graph, 'y'):
                assert graph.y is None  # Check if y is None after deletion attempt

    def test_init_with_config(self, dummy_mol_files_sdf: List[str], temp_data_dir: str):
        """Test MolecularGraphDataset with a custom config for graph processor."""
        # Example: config that changes feature dimensions or type
        custom_config = {"atom_feature_schemes": ["atomic_num", "formal_charge"]}
        dataset = MolecularGraphDataset(mol_files=[dummy_mol_files_sdf[0]], config=custom_config)
        graph = dataset[0]
        if isinstance(graph.x, torch.Tensor):
            assert graph.x.shape[1] == 17

    def test_error_mol_file_not_found(self, temp_data_dir: str, caplog):
        """Test MolecularGraphDataset with a non-existent molecule file."""
        non_existent_file = os.path.join(temp_data_dir, "not_here.sdf")

        with caplog.at_level(logging.WARNING):  # Capture WARNING and ERROR (default for ERROR)
            dataset = MolecularGraphDataset(mol_files=[non_existent_file])

        assert len(dataset) == 0  # Should skip the file

        # Check for expected log messages
        # MolecularGraphProcessor logs ERROR: "Molecule file not found: {file_path}"
        # MolecularGraphDataset logs WARNING: "Failed to process {file_path}, excluding from dataset"

        processor_log_found = any(
            f"Molecule file not found: {non_existent_file}" in record.message and record.levelname == "ERROR"
            for record in caplog.records
        )
        dataset_log_found = any(
            f"Failed to process {non_existent_file}, excluding from dataset" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )

        assert processor_log_found, "Expected 'Molecule file not found' log from processor not found."
        assert dataset_log_found, "Expected 'Failed to process ... excluding from dataset' log from dataset not found."

    def test_error_invalid_mol_file(self, temp_data_dir: str, caplog):
        """Test MolecularGraphDataset with a malformed molecule file."""
        invalid_sdf_path = os.path.join(temp_data_dir, "invalid.sdf")
        with open(invalid_sdf_path, "w") as f: f.write("This is not an SDF file")

        with caplog.at_level(logging.WARNING):  # Capture WARNING and ERROR
            dataset = MolecularGraphDataset(mol_files=[invalid_sdf_path])

        assert len(dataset) == 0  # Should skip

        # Check for expected log messages
        # MolecularGraphProcessor logs ERROR: "Failed to read molecule from {file_path}"
        # MolecularGraphDataset logs WARNING: "Failed to process {file_path}, excluding from dataset"

        processor_log_found = any(
            f"Failed to read molecule from {invalid_sdf_path}" in record.message and record.levelname == "ERROR"
            for record in caplog.records
        )
        dataset_log_found = any(
            f"Failed to process {invalid_sdf_path}, excluding from dataset" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )

        assert processor_log_found, "Expected 'Failed to read molecule' log from processor not found."
        assert dataset_log_found, "Expected 'Failed to process ... excluding from dataset' log from dataset not found."


class TestHierarchicalGraphDataset:
    def test_initialization_and_len(self, dummy_hierarchical_data_dir: str):
        dataset = HierarchicalGraphDataset(data_dir=dummy_hierarchical_data_dir)
        assert len(dataset) == 3  # molA, molB, molC_empty

    def test_getitem(self, dummy_hierarchical_data_dir: str, dummy_labels_for_hier_data: Dict[str, float]):
        """
        Tests retrieval of hierarchical graph data for molecules, verifying that each returned item is a dictionary containing graph objects for available levels (e.g., "atom", "functional_group", "structural_motif"). Ensures that present graphs are instances of `Data`, missing levels are handled gracefully, and label tensors `y` match expected values from the dummy labels.
        """
        dataset = HierarchicalGraphDataset(data_dir=dummy_hierarchical_data_dir, labels=dummy_labels_for_hier_data)

        # Test molA
        item_a = dataset[dataset.molecule_ids.index("molA")]
        assert isinstance(item_a, dict)
        assert "atom" in item_a and isinstance(item_a["atom"], Data)
        assert "functional_group" in item_a and isinstance(item_a["functional_group"], Data)

        assert "structural_motif" in item_a and (
            isinstance(item_a["structural_motif"], Data) or item_a["structural_motif"] is None
        )
        if isinstance(item_a["atom"].y, torch.Tensor):
            assert torch.allclose(item_a["atom"].y, torch.tensor([10.0], dtype=torch.float))
        if item_a["structural_motif"] is not None and hasattr(item_a["structural_motif"], "y") and isinstance(item_a["structural_motif"].y, torch.Tensor):
            assert torch.allclose(item_a["structural_motif"].y, torch.tensor([10.0], dtype=torch.float))

        # Test molB
        item_b = dataset[dataset.molecule_ids.index("molB")]
        assert isinstance(item_b, dict)
        assert "atom" in item_b and isinstance(item_b["atom"], Data)
        assert "functional_group" in item_b and item_b["functional_group"] is None
        assert "structural_motif" in item_b and (
            isinstance(item_b["structural_motif"], Data) or item_b["structural_motif"] is None
        )
        if isinstance(item_b["atom"].y, torch.Tensor):
            assert torch.allclose(item_b["atom"].y, torch.tensor([20.0], dtype=torch.float))
        if item_b["structural_motif"] is not None and hasattr(item_b["structural_motif"], "y") and isinstance(item_b["structural_motif"].y, torch.Tensor):
            assert torch.allclose(item_b["structural_motif"].y, torch.tensor([20.0], dtype=torch.float))

    def test_transform_hierarchical(self, dummy_hierarchical_data_dir: str):  # Renamed to avoid conflict
        dataset_no_transform = HierarchicalGraphDataset(data_dir=dummy_hierarchical_data_dir)
        idx_A = dataset_no_transform.molecule_ids.index("molA")
        original_item_A = dataset_no_transform[idx_A]
        original_x_atom_A = None
        if original_item_A["atom"].x is not None:
            original_x_atom_A = original_item_A["atom"].x.clone()

        dataset_with_transform = HierarchicalGraphDataset(
            data_dir=dummy_hierarchical_data_dir, transform=simple_transform
        )
        transformed_item_A = dataset_with_transform[idx_A]

        if original_x_atom_A is not None: assert torch.allclose(transformed_item_A["atom"].x, original_x_atom_A * 2)

    def test_init_data_dir_not_found(self, temp_data_dir: str):
        """Test HierarchicalGraphDataset with a non-existent data_dir."""
        non_existent_dir = os.path.join(temp_data_dir, "no_such_dir")
        dataset = HierarchicalGraphDataset(data_dir=non_existent_dir)
        assert len(dataset) == 0

    def test_getitem_missing_level_file(self, dummy_hierarchical_data_dir: str):
        """Test HierarchicalGraphDataset __getitem__ when a level file is missing."""
        dataset = HierarchicalGraphDataset(data_dir=dummy_hierarchical_data_dir, levels=["atom", "non_existent_level"])
        idx_C = dataset.molecule_ids.index("molC_empty")
        item_c = dataset[idx_C]
        assert "atom" in item_c
        assert "non_existent_level" not in item_c  # Should be skipped gracefully

    def test_init_different_levels_specified(self, dummy_hierarchical_data_dir: str):
        """Test HierarchicalGraphDataset loads only specified levels."""
        dataset = HierarchicalGraphDataset(data_dir=dummy_hierarchical_data_dir, levels=["atom"])
        idx_A = dataset.molecule_ids.index("molA")
        item_a = dataset[idx_A]
        assert "atom" in item_a
        assert "functional_group" not in item_a
        assert "structural_motif" not in item_a


class TestPFASDataset:
    def test_initialization_and_len(self, dummy_pfas_csv_file: str, caplog):
        """Test PFASDataset initialization and length."""
        with caplog.at_level(logging.WARNING):  # To catch SMILES parsing warnings
            dataset = PFASDataset(
                data_path=dummy_pfas_csv_file,
                smiles_column="smiles",
                feature_columns=["feature1", "feature2"],
                target_column="target_property",
            )
        # 1 invalid SMILES, so 4 valid molecules
        assert len(dataset) == 4

    def test_getitem_and_labels(self, dummy_pfas_csv_file: str, caplog):
        """Test PFASDataset getitem and labels."""
        dataset = PFASDataset(data_path=dummy_pfas_csv_file, smiles_column="smiles", target_column="target_property")
        # Corresponding targets for valid SMILES
        expected_targets = [100.0, 200.0, 400.0, 500.0]
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, Data)
            assert hasattr(item, "y")
            if isinstance(item.y, torch.Tensor):
                assert torch.allclose(item.y, torch.tensor([expected_targets[i]], dtype=torch.float))

    def test_features_extraction(self, dummy_pfas_csv_file: str, caplog):
        """Test features extraction from PFAS dataset."""
        with caplog.at_level(logging.WARNING):
            dataset = PFASDataset(
                data_path=dummy_pfas_csv_file,
                smiles_column="smiles",
                feature_columns=["feature1", "feature2"],
            )
        assert hasattr(dataset, "features")
        assert dataset.features.shape == (5, 2)  # 5 rows in CSV, 2 feature columns

    def test_transform_pfas(self, dummy_pfas_csv_file: str, caplog):
        """Test transform functionality for PFAS dataset."""
        with caplog.at_level(logging.WARNING):
            dataset_no_transform = PFASDataset(data_path=dummy_pfas_csv_file, smiles_column="smiles")
            original_x_0 = dataset_no_transform[0].x.clone() if isinstance(dataset_no_transform[0].x, torch.Tensor) else None

            dataset_with_transform = PFASDataset(
                data_path=dummy_pfas_csv_file, smiles_column="smiles", transform=simple_transform
            )
            transformed_item_0 = dataset_with_transform[0]

            if original_x_0 is not None:
                assert torch.allclose(transformed_item_0.x, original_x_0 * 2)

    def test_init_csv_not_found(self, temp_data_dir: str):
        """Test PFASDataset with a non-existent CSV file."""
        non_existent_csv = os.path.join(temp_data_dir, "no_data.csv")
        with pytest.raises(FileNotFoundError):
            PFASDataset(data_path=non_existent_csv, smiles_column="smiles")

    def test_init_missing_smiles_column(self, temp_data_dir: str, caplog):
        """Test PFASDataset when smiles_column is missing from CSV."""
        csv_path = os.path.join(temp_data_dir, "no_smiles_col.csv")
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        df.to_csv(csv_path, index=False)
        dataset = PFASDataset(data_path=csv_path, smiles_column="wrong_smiles_col")
        assert len(dataset) == 0
        assert dataset.smiles is None

    def test_init_missing_target_column(self, dummy_pfas_csv_file: str, caplog):
        """Test PFASDataset when target_column is specified but missing."""
        with caplog.at_level(logging.WARNING):  # To catch SMILES parsing warnings
            dataset = PFASDataset(
                data_path=dummy_pfas_csv_file,
                smiles_column="smiles",
                target_column="non_existent_target",
            )
        # Graphs are still created, just without labels
        assert len(dataset) == 4
        for i in range(len(dataset)):
            graph = dataset[i]
            assert graph.y is None

    def test_init_no_feature_columns(self, dummy_pfas_csv_file: str, caplog):
        """Test PFASDataset initialization without feature_columns."""
        with caplog.at_level(logging.WARNING):
            dataset = PFASDataset(data_path=dummy_pfas_csv_file, smiles_column="smiles")
        assert dataset.features is None
        assert len(dataset) == 4  # Graphs should still be created

    def test_graph_content_basic(self, dummy_pfas_csv_file: str, caplog):
        """Test basic content of graphs generated by PFASDataset."""
        with caplog.at_level(logging.WARNING):
            dataset = PFASDataset(data_path=dummy_pfas_csv_file, smiles_column="smiles")

        for i in range(len(dataset)):
            graph = dataset[i]
            assert isinstance(graph, Data)
            assert hasattr(graph, "x")
            assert hasattr(graph, "edge_index")
            if isinstance(graph.x, torch.Tensor):
                assert graph.x.dtype == torch.float
                assert graph.x.shape[0] > 0  # Should have at least one atom
                assert graph.x.shape[1] == 47
            if hasattr(graph, 'edge_index') and isinstance(graph.edge_index, torch.Tensor):
                assert graph.edge_index.dtype == torch.long
                assert graph.edge_index.shape[0] == 2
                # edge_index can be empty for single-atom molecules if no self-loops
                if graph.x.shape[0] == 1: assert graph.edge_index.shape[1] == 0  # e.g. for "C"
                elif graph.x.shape[0] > 1: assert graph.edge_index.shape[1] > 0
