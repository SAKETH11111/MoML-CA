"""
moml/simulation/molecular_dynamics/force_field/plugins/ro_v1/build.py

Reverse Osmosis (RO) Membrane Surface Builder Module

This plugin re-uses the nanofiltration (NF) polyamide builder provided in
`nf_polyamide_v1` and exposes it under an RO-specific surface ID.  In future the
RO membrane may deviate from the NF implementation (e.g., by compressing the
simulation box or introducing an additional support layer).  For now, the
module simply delegates the heavy lifting to the NF builder and returns the
same artefacts.
"""

from importlib import import_module
from pathlib import Path
from typing import Tuple, List
from openmm.app import Topology

# Re-use NF membrane builder and expose as RO
_nf_builder = import_module(
    "moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build"
)

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    """Build an RO polyamide membrane.

    The function simply calls the NF membrane builder from
    `nf_polyamide_v1.build` and returns its output.  It exists to maintain a
    stable public surface ID (`ro_v1`) while permitting future divergence in the
    implementation.

    Args:
        tmp_dir: Directory where temporary artefacts (e.g., PDB file) should be
                  written.
        cfg:     Configuration dictionary to be forwarded verbatim to the NF
                  builder.  See the NF plugin documentation for accepted keys.

    Returns
    -------
    Tuple consisting of:
        Path      – path to the generated PDB file
        Topology  – OpenMM topology describing the membrane
        List[int] – atom indices belonging to the membrane (useful for applying
                     restraints or analysis)
    """
    pdb_path, topology, idx = _nf_builder.build(tmp_dir, cfg)
    # TODO: Potential future post-processing (e.g., box compression) goes here.
    return pdb_path, topology, idx
