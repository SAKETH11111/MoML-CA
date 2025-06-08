"""
Build a rigid graphene-like slab inside the current working directory
and return paths & indices needed by the MD builder.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from openmm.app import PDBFile, Topology
from openmm.app.element import carbon

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    """Return (pdb_path, topology_obj, surface_atom_indices)."""
    box_x, box_y, slab_z = cfg["slab_dims_nm"]
    a = 0.142  # nm C–C
    # create simple hexagonal sheet – minimal until full pore model arrives
    nx = int(box_x / (a * np.sqrt(3)))
    ny = int(box_y / (a * 1.5))
    positions = []
    topology = Topology()
    chain = topology.addChain()
    res   = topology.addResidue("GAC", chain)
    idx_list = []

    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5 * (j % 2)) * a * np.sqrt(3)
            y = j * a * 1.5
            positions.append([x, y, slab_z / 2])
            atom = topology.addAtom(f"C{i}_{j}", carbon, res)
            idx_list.append(atom.index)

    pdb_path = tmp_dir / "gac_slab.pdb"
    PDBFile.writeFile(topology, positions, open(pdb_path, "w"))
    # tag unit cell
    topology.setUnitCellDimensions((box_x, box_y, slab_z))
    return pdb_path, topology, idx_list
