from importlib import import_module
from pathlib import Path
from typing import Tuple, List
from openmm.app import Topology

# Re-use NF builder then scale box / adjust density
_nf_builder = import_module("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build")

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    pdb_path, topology, idx = _nf_builder.build(tmp_dir, cfg)
    # optional: compress box further or add support layer
    return pdb_path, topology, idx
