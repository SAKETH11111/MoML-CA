"""
moml/simulation/molecular_dynamics/force_field/plugins/gac_v1/build.py

Granular Activated Carbon (GAC) Surface Builder Module

This module provides functionality to build a rigid graphene-like slab representing
a granular activated carbon (GAC) surface for molecular dynamics simulations.
It creates a hexagonal carbon lattice with specified dimensions that can be used
as an adsorption surface in PFAS treatment simulations.

The module generates both the 3D structure and topology information needed by the
MD simulation builder, with options to control the size and properties of the
carbon surface.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from openmm.app import PDBFile, Topology
from openmm.app.element import carbon

def build(tmp_dir: Path, cfg: dict) -> Tuple[Path, Topology, List[int]]:
    """
    Build a rigid graphene-like GAC surface and return required simulation components.
    
    This function creates a hexagonal carbon lattice representing a granular activated
    carbon surface based on the provided configuration. It generates the PDB file,
    topology, and atom indices needed for molecular dynamics simulations.
    
    Args:
        tmp_dir: Path to temporary directory where output files will be written
        cfg: Configuration dictionary containing:
            - slab_dims_nm: List[float] with [x, y, z] dimensions in nanometers
            - pore_fraction: Float indicating porosity (0.0 = solid slab)
            - freeze_atoms: Boolean indicating whether to fix atom positions
    
    Returns:
        Tuple[Path, Topology, List[int]]: A tuple containing:
            - Path: Path to the generated PDB file
            - Topology: OpenMM Topology object for the surface
            - List[int]: List of atom indices comprising the surface
    """
    box_x, box_y, slab_z = cfg["slab_dims_nm"]
    a = 0.142  # nm C–C
    # create simple hexagonal sheet – minimal until full pore model arrives
    # Validate box dimensions to ensure at least one lattice unit
    min_box_x = a * np.sqrt(3)
    min_box_y = a * 1.5
    if box_x < min_box_x:
        raise ValueError(f"box_x ({box_x:.4f} nm) is too small to produce a lattice unit. Must be at least {min_box_x:.4f} nm.")
    if box_y < min_box_y:
        raise ValueError(f"box_y ({box_y:.4f} nm) is too small to produce a lattice unit. Must be at least {min_box_y:.4f} nm.")

    nx = int(box_x / min_box_x)
    ny = int(box_y / min_box_y)
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
    with open(pdb_path, "w") as f: # Use context manager for file handling
        PDBFile.writeFile(topology, positions, f)
    # tag unit cell
    topology.setUnitCellDimensions((box_x, box_y, slab_z))
    return pdb_path, topology, idx_list
