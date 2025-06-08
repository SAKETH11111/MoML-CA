from pathlib import Path
from typing import Tuple, List
import numpy as np
from openmm.app import PDBFile, Modeller, Topology
from openff.toolkit.topology import Molecule

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    bead_count      = cfg["bead_count"]
    bead_radius_nm  = cfg["bead_radius_nm"]

    # 1. Load styrene-divinylbenzene-backbone monomer with quaternary amine
    monomer = Molecule.from_smiles("[CH2]C(=O)O[C@H]1CC[N+](C)(C)C1")  # placeholder
    # 2. Pack monomers randomly inside a sphere with Packmol, polymerise manually
    positions, topology = _pack_bead(monomer, bead_radius_nm, bead_count)

    pdb_out = tmp_dir / "ix_bead.pdb"
    PDBFile.writeFile(topology, positions, open(pdb_out, "w"))
    return pdb_out, topology, list(range(topology.getNumAtoms()))

def _pack_bead(monomer: Molecule, radius_nm: float, count: int):
    # placeholder random sphere fill
    return [], Topology()
