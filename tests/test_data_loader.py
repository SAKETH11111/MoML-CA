"""
tests/test_data_loader.py

Unit tests for the PFASDataLoader class in moml.data.data_loader.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, Generator

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from moml.data.data_loader import PFASDataLoader


def _create_mol(smiles: str, path: str) -> None:
    """
    Create a 3D-embedded RDKit molecule from a SMILES string and save it to a file.
    
    The molecule is generated with explicit hydrogens, embedded in 3D space using a fixed random seed,
    and geometry-optimized with the UFF force field before being written to the specified file path in MOL format.
    
    Args:
        smiles (str): SMILES string representation of the molecule.
        path (str): File path where the molecule will be saved.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    AllChem.UFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    Chem.MolToMolFile(mol, path)


@pytest.fixture()
def temp_dataset_dir() -> Generator[str, None, None]:
    """
    Pytest fixture that creates a temporary dataset directory for testing.
    
    The fixture generates a directory containing molecular files for water and methane, environment and label JSON files,
    and a 'train' subdirectory with copies of the molecule files. The directory is cleaned up after the test completes.
    
    Yields:
        str: The path to the temporary dataset directory.
    """
    tmp = tempfile.mkdtemp(prefix="pfas_loader_test_")
    mol_dir = os.path.join(tmp, "molecules")
    os.makedirs(mol_dir, exist_ok=True)

    _create_mol("O", os.path.join(mol_dir, "water.mol"))
    _create_mol("C", os.path.join(mol_dir, "methane.mol"))

    env = {
        "water": {"ph": 7.0, "temperature": 298.15, "ionic_strength": 0.1},
        "methane": {"ph": 8.0, "temperature": 300.0, "ionic_strength": 0.05},
    }
    with open(os.path.join(tmp, "environment.json"), "w") as f:
        json.dump(env, f)

    labels = {"water": 0.1, "methane": 0.2}
    with open(os.path.join(tmp, "labels.json"), "w") as f:
        json.dump(labels, f)

    train_dir = os.path.join(tmp, "train")
    os.makedirs(train_dir, exist_ok=True)
    shutil.copy(os.path.join(mol_dir, "water.mol"), os.path.join(train_dir, "water.mol"))
    shutil.copy(os.path.join(mol_dir, "methane.mol"), os.path.join(train_dir, "methane.mol"))

    yield tmp
    shutil.rmtree(tmp)


def test_load_molecule_by_id(temp_dataset_dir: str) -> None:
    """
    Test that loading the "water" molecule by ID returns the correct graph, label, and environment data.
    
    Asserts that the label and environment values match expected results, the graph is a torch_geometric Data object,
    and required attributes are present.
    
    Args:
        temp_dataset_dir (str): Path to the temporary dataset directory.
    """
    loader = PFASDataLoader(temp_dataset_dir)
    graph, label, env = loader.load_molecule_by_id("water")

    assert pytest.approx(label, rel=1e-6) == 0.1
    assert env["ph"] == 7.0
    assert isinstance(graph, Data)
    assert hasattr(graph, "u")
    assert graph.u.shape[0] >= 3


def test_get_batch(temp_dataset_dir: str) -> None:
    """ 
    Test that PFASDataLoader correctly loads a batch of molecular graphs by IDs.
    
    Asserts that the returned batch contains the expected batch attribute and the correct number of graphs.
    
    Args:
        temp_dataset_dir (str): Path to the temporary dataset directory.
    """
    loader = PFASDataLoader(temp_dataset_dir)
    batch = loader.get_batch(["water", "methane"], batch_size=2)
    assert hasattr(batch, "batch")
    assert batch.num_graphs == 2


def test_load_dataset(temp_dataset_dir: str) -> None:
    """
    Test that the PFASDataLoader correctly loads all molecules in the 'train' dataset split.
    
    Asserts that the loaded dataset contains the expected number of entries.
    
    Args:
        temp_dataset_dir (str): Path to the temporary dataset directory.
    """
    loader = PFASDataLoader(temp_dataset_dir)
    dataset = loader.load_dataset("train")
    assert len(dataset) == 2


def test_water_molecule_validation(temp_dataset_dir: str) -> None:
    """
    Validate the structure and features of the loaded water molecule graph.
    
    Ensures the water molecule graph has the correct number of nodes, edges, and node feature dimensions,
    and checks the presence and size of the node-level attribute 'u' if available.
    
    Args:
        temp_dataset_dir (str): Path to the temporary dataset directory.
    """
    loader = PFASDataLoader(temp_dataset_dir)
    graph, label, env = loader.load_molecule_by_id("water")

    assert graph.num_nodes == 3
    assert graph.edge_index.size(1) // 2 == 2
    assert graph.x.size(1) == loader.graph_processor.atom_feature_dim
    if hasattr(graph, "u"):
        assert graph.u.size(0) >= 3


def test_batch_loading_performance(temp_dataset_dir: str) -> None:
    """
    Test that batch loading returns the correct number of graphs and consistent node feature sizes.
    
    Asserts that loading a batch of "water" and "methane" molecules yields two graphs and that the number of
    node features matches the batch assignment size.
    
    Args:
        temp_dataset_dir (str): Path to the temporary dataset directory.
    """
    loader = PFASDataLoader(temp_dataset_dir)
    batch = loader.get_batch(["water", "methane"], batch_size=2)
    assert batch.num_graphs == 2
    assert batch.x.size(0) == batch.batch.size(0)
