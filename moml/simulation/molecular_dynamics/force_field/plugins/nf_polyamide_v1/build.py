from pathlib import Path
from typing import Tuple, List
from openmm.app import PDBFile, Modeller, Topology
from openff.toolkit.topology import Molecule
import packmol_runner   # hypothetical utility used elsewhere in project

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    reps = cfg["repeat_units"]
    # Load monomer PDB (stored in resources) and polymerise with OpenFF
    monomer = Molecule.from_file(Path(__file__).with_suffix(".sdf"))
    polymer = monomer.to_topology().convert_to_openmm()
    modeller = Modeller(polymer, polymer.positions)
    modeller.addSolvent(forcefield=None, model=None, boxSize=(5,5,5))  # placeholder water removal

    # TODO: Packmol-based tiling for real membrane
    pdb_out = tmp_dir / "nf_polyamide.pdb"
    PDBFile.writeFile(modeller.topology, modeller.positions, open(pdb_out, "w"))
    all_ix = list(range(modeller.topology.getNumAtoms()))
    return pdb_out, modeller.topology, all_ix
