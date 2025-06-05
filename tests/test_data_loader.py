import os
import json
import shutil
import tempfile

import pytest
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

from moml.data.data_loader import PFASDataLoader


def _create_mol(smiles: str, path: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, path)


@pytest.fixture()
def temp_dataset_dir():
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


def test_load_molecule_by_id(temp_dataset_dir):
    loader = PFASDataLoader(temp_dataset_dir)
    graph, label, env = loader.load_molecule_by_id("water")

    assert pytest.approx(label, rel=1e-6) == 0.1
    assert env["ph"] == 7.0
    assert isinstance(graph, Data)
    assert hasattr(graph, "u")
    assert graph.u.shape[0] >= 3


def test_get_batch(temp_dataset_dir):
    loader = PFASDataLoader(temp_dataset_dir)
    batch = loader.get_batch(["water", "methane"], batch_size=2)
    assert hasattr(batch, "batch")
    assert batch.num_graphs == 2


def test_load_dataset(temp_dataset_dir):
    loader = PFASDataLoader(temp_dataset_dir)
    dataset = loader.load_dataset("train")
    assert len(dataset) == 2


def test_water_molecule_validation(temp_dataset_dir):
    loader = PFASDataLoader(temp_dataset_dir)
    graph, label, env = loader.load_molecule_by_id("water")

    assert graph.num_nodes == 3
    assert graph.edge_index.size(1) // 2 == 2
    assert graph.x.size(1) == loader.graph_processor.atom_feature_dim
    if hasattr(graph, "u"):
        assert graph.u.size(0) >= 3


def test_batch_loading_performance(temp_dataset_dir):
    loader = PFASDataLoader(temp_dataset_dir)
    batch = loader.get_batch(["water", "methane"], batch_size=2)
    assert batch.num_graphs == 2
    assert batch.x.size(0) == batch.batch.size(0)
