from pathlib import Path
from typing import Tuple, List
from openmm.app import PDBFile, Topology
from openff.toolkit.topology import Molecule

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    bead_count      = cfg["bead_count"]
    bead_radius_nm  = cfg["bead_radius_nm"]

    # 1. Load styrene-divinylbenzene-backbone monomer with quaternary amine
    # Correct SMILES for para-benzyl trimethylammonium styrene unit (with chloride counterion)
    monomer = Molecule.from_smiles("C[N+](C)(C)Cc1ccc(C=C)cc1.[Cl-]")
    # 2. Pack monomers randomly inside a sphere with Packmol, polymerise manually
    positions, topology = _pack_bead(monomer, bead_radius_nm, bead_count)

    pdb_out = tmp_dir / "ix_bead.pdb"
    with open(pdb_out, "w") as f: # Use context manager for file handling
        PDBFile.writeFile(topology, positions, f)
    return pdb_out, topology, list(range(topology.getNumAtoms()))

def _pack_bead(monomer: Molecule, radius_nm: float, count: int):
    """
    Placeholder for molecular packing of polymer beads.
    A proper implementation would use a packing algorithm (e.g., Packmol)
    to generate accurate positions and topology data.
    """
    # placeholder random sphere fill
    return [], Topology()
