from pathlib import Path
from typing import Tuple, List
from openmm.app import PDBFile, Modeller, Topology
from openff.toolkit.topology import Molecule
import packmol_runner   # hypothetical utility used elsewhere in project

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    reps = cfg["repeat_units"]
    # Load monomer PDB (stored in resources) and polymerise with OpenFF
    sdf_path = Path(__file__).with_suffix(".sdf")
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    monomer = Molecule.from_file(sdf_path)
    polymer = monomer.to_topology().convert_to_openmm()
    modeller = Modeller(polymer, polymer.positions)
    modeller.addSolvent(forcefield=None, model=None, boxSize=(5,5,5))  # placeholder water removal

    # TODO: Packmol-based tiling for real membrane
    pdb_out = tmp_dir / "nf_polyamide.pdb"
    with open(pdb_out, "w") as f: # Use context manager for file handling
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    all_ix = list(range(modeller.topology.getNumAtoms()))
    return pdb_out, modeller.topology, all_ix
