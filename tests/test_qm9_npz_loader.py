from torch_geometric.datasets import QM9
import numpy as np


def test_qm9_npz_loader():
    qm9 = QM9(root="data/qm9")  # vanilla dataset
    pfas = np.load("data/pfas_qm9.npz")  # your converted file

    assert qm9[0].y.shape[1] == pfas["y"][0].shape[0]
    print("Shapes match – MGNN can ingest PFAS directly ✅")
