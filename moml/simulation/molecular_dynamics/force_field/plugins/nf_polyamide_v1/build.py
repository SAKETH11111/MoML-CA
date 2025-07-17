"""
moml/simulation/molecular_dynamics/force_field/plugins/nf_polyamide_v1/build.py

Builder for the *nf_polyamide_v1* force-field plugin.

This helper constructs a simple polyamide membrane patch for use in molecular
simulations. It relies on OpenMM and the OpenFF Toolkit when available.
All heavy dependencies are imported lazily with fall-backs so that the
repository can be imported even when optional packages are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from openmm.app import Modeller, PDBFile, Topology

try:
    from openff.toolkit.topology import Molecule  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    logging.warning(
        "OpenFF Toolkit not available. Polyamide builder will raise if executed."
    )

    class Molecule:  # type: ignore
        """Stub Molecule class provided when the OpenFF Toolkit is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "OpenFF Toolkit is required for building nf_polyamide membranes."
            )

        @classmethod  # type: ignore
        def from_file(cls, *args, **kwargs):  # noqa: D401 – stub method
            raise ImportError("OpenFF Toolkit is required to read molecule files.")

try:
    import packmol_runner  # type: ignore
except ImportError:  # pragma: no cover – optional utility
    packmol_runner = None  # type: ignore
    logging.debug("packmol_runner not found – skipping dense packing step.")

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    """Construct a minimal polyamide membrane system.

    Parameters
    ----------
    tmp_dir
        Temporary directory where intermediate and output files are written.
    cfg
        Configuration dictionary. Must contain the key ``"repeat_units"`` to
        indicate how many monomer units to polymerise (currently unused but
        reserved for future extensions).

    Returns
    Tuple[Path, Topology, List[int]]
        * Path to the generated PDB file.
        * The corresponding OpenMM ``Topology`` object.
        * A list of all atom indices (useful for downstream selection).

    Notes:
    TODO: This implementation is intentionally lightweight. In production one would
    build an actual polymer and optionally pack it with Packmol. Those steps
    are elided here to keep the example self-contained.
    """

    reps = int(cfg.get("repeat_units", 1))  # noqa: F841 – reserved for future use

    sdf_path = Path(__file__).with_suffix(".sdf")
    if not sdf_path.exists():
        raise FileNotFoundError(f"Monomer SDF file not found: {sdf_path}")

    monomer = Molecule.from_file(sdf_path)  # type: ignore[attr-defined]
    polymer_top = monomer.to_topology().convert_to_openmm()  # type: ignore[attr-defined]
    modeller = Modeller(polymer_top, polymer_top.positions)

    # Add placeholder solvent box (dimensions arbitrary for stub)
    modeller.addSolvent(forcefield=None, model=None, boxSize=(5.0, 5.0, 5.0))  # type: ignore[arg-type]

    # Optionally pack with Packmol
    if packmol_runner is not None:
        logging.debug("Packmol packing would occur here if implemented.")

    # Write output
    pdb_out = tmp_dir / "nf_polyamide.pdb"
    with pdb_out.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)

    atom_indices = list(range(modeller.topology.getNumAtoms()))
    return pdb_out, modeller.topology, atom_indices
