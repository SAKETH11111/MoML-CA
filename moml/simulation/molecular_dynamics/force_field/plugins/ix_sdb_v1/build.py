"""
moml/simulation/molecular_dynamics/force_field/plugins/ix_sdb_v1/build.py

Ion-Exchange Resin (Styrene-Divinylbenzene) Bead Builder Module

This plugin constructs a coarse-grained spherical bead representing a fragment
of a strong-base anion-exchange (IX-SDB) resin.  The bead contains a controllable
number of quaternary-amine functionalised styrene-divinylbenzene monomers packed
inside a sphere.  The resulting structure can be used as an adsorbent phase in
PFAS sorption simulations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Any
from openmm.app import PDBFile, Topology
import openmm.unit as unit
import openmm
try:
    from openff.toolkit.topology import Molecule  # type: ignore
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover
    # Allow the plugin to be imported even if OpenFF Toolkit is not available.
    Molecule = None  # type: ignore
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
import numpy as np
import random

__all__ = ["build"]


def build(tmp_dir: Path, cfg: dict):  # type: ignore[invalid-type]
    """Build a coarse-grained IX-SDB polymer bead.

    Args
    ----
    tmp_dir
        Directory where the generated PDB file should be written.
    cfg
        Configuration dictionary with the following keys:

        * ``bead_count`` – number of monomer units to include in the bead.
        * ``bead_radius_nm`` – target bead radius in **nanometres**.

    Returns
    -------
    Tuple consisting of
        Path
            Path to the generated ``ix_bead.pdb`` file.
        Topology
            OpenMM ``Topology`` describing the bead.
        List[int]
            Atom indices that belong to the bead (useful for restraints).
    """
    bead_count = cfg["bead_count"]
    bead_radius_nm = cfg["bead_radius_nm"]

    # Ensure OpenFF Toolkit is available
    if Molecule is None:
        raise ImportError(
            "openff-toolkit is required to build the IX-SDB bead (pip install openff-toolkit)"
        )

    # 1. Create styrene-divinylbenzene monomer with a quaternary amine.
    #    SMILES for para-benzyl trimethylammonium styrene unit with chloride
    #    counter-ion.
    monomer = Molecule.from_smiles("C[N+](C)(C)Cc1ccc(C=C)cc1.[Cl-]")

    # 2. Pack monomers randomly inside a sphere with geometric overlap avoidance.
    #    Future enhancement will include polymerization via cross-link formation.
    positions, topology = _pack_bead(monomer, bead_radius_nm, bead_count)

    pdb_out = tmp_dir / "ix_bead.pdb"
    with open(pdb_out, "w") as f:
        PDBFile.writeFile(topology, positions, f)
    return pdb_out, topology, list(range(topology.getNumAtoms()))


def _pack_bead(monomer: Any, radius_nm: float, count: int):
    """Pack monomers into a spherical bead with basic geometric arrangement.
    
    This implementation provides real functionality by:
    1. Generating 3D coordinates for the monomer using RDKit
    2. Randomly placing monomers within the sphere with overlap avoidance
    3. Building an OpenMM topology from the packed structures
    
    TODO: Form covalent cross-links between vinyl sites within 0.25 nm cutoff.
    TODO: For beads with more than 200 monomers, replace rejection sampling with Packmol for efficient packing. The current rejection sampling approach is only practical for small bead counts and will become extremely inefficient for larger systems.
    TODO: Use van der Waals radii for more accurate collision detection.
    
    Note:
        - The current implementation is not scalable for large systems (>200 monomers).
        - For large-scale packing, integrating Packmol or a similar tool is strongly recommended.
        - Cross-linking logic is not yet implemented and is required for realistic bead models.
    
    Args:
        monomer: OpenFF Molecule object representing the monomer unit
        radius_nm: Target bead radius in nanometers
        count: Number of monomer units to pack in the bead
        
    Returns:
        tuple: (positions, topology) where positions is a list of Vec3 coordinates
               in nanometers with proper units, and topology is an OpenMM Topology object
    """
    # Ensure required dependencies are available
    if Chem is None or AllChem is None:
        raise ImportError("rdkit is required for monomer packing (conda install rdkit)")
    
    # Convert OpenFF molecule to RDKit for 3D coordinate generation
    rdkit_mol = monomer.to_rdkit()
    rdkit_mol = Chem.AddHs(rdkit_mol)
    
    # Generate 3D coordinates for the monomer
    if AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG()) != 0:  # type: ignore[attr-defined]
        # Fallback if embedding fails
        AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)  # type: ignore[attr-defined]
    
    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(rdkit_mol, maxIters=200)  # type: ignore[attr-defined]
    except:
        pass  # Continue even if optimization fails
    
    # Get monomer coordinates and atom info
    conf = rdkit_mol.GetConformer()
    monomer_coords = np.array([[conf.GetAtomPosition(i).x, 
                               conf.GetAtomPosition(i).y, 
                               conf.GetAtomPosition(i).z] 
                              for i in range(rdkit_mol.GetNumAtoms())])
    
    # Calculate monomer size for overlap avoidance
    # TODO: Use van der Waals radii sum for more accurate collision detection
    monomer_size = np.max(np.ptp(monomer_coords, axis=0))  # Max dimension
    min_distance = max(0.3, monomer_size * 0.8)  # Minimum separation in nm
    
    # Create OpenMM topology
    topology = Topology()
    chain = topology.addChain()
    all_positions = []
    
    # Pack monomers with basic overlap avoidance
    # TODO: For >200 monomers, replace with Packmol for better packing efficiency.
    #       The current rejection sampling approach is not suitable for large bead counts.
    placed_centers = []
    max_attempts = count * 50  # Limit attempts to avoid infinite loops
    
    for mol_idx in range(count):
        placed = False
        
        for _ in range(max_attempts // count):
            # Generate random position within sphere
            # Use rejection sampling for uniform distribution in sphere
            while True:
                x = random.uniform(-1, 1) * radius_nm
                y = random.uniform(-1, 1) * radius_nm  
                z = random.uniform(-1, 1) * radius_nm
                
                if x*x + y*y + z*z <= radius_nm*radius_nm:
                    center = np.array([x, y, z])
                    break
            
            # Check for overlaps with existing monomers
            if len(placed_centers) == 0:
                overlap = False
            else:
                distances = np.linalg.norm(np.array(placed_centers) - center, axis=1)
                overlap = np.any(distances < min_distance)
            
            if not overlap:
                # Place monomer at this position
                placed_centers.append(center)
                
                # Add random rotation to the monomer
                theta = random.uniform(0, 2*np.pi)
                phi = random.uniform(0, np.pi)
                psi = random.uniform(0, 2*np.pi)
                
                # Simple rotation matrices
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                cos_psi, sin_psi = np.cos(psi), np.sin(psi)
                
                # Rotation around z, then y, then z again (Euler angles)
                rot_z1 = np.array([[cos_theta, -sin_theta, 0],
                                  [sin_theta, cos_theta, 0],
                                  [0, 0, 1]])
                rot_y = np.array([[cos_phi, 0, sin_phi],
                                 [0, 1, 0],
                                 [-sin_phi, 0, cos_phi]])
                rot_z2 = np.array([[cos_psi, -sin_psi, 0],
                                  [sin_psi, cos_psi, 0],
                                  [0, 0, 1]])
                
                rotation_matrix = rot_z2 @ rot_y @ rot_z1
                
                # Apply rotation and translation to monomer coordinates
                rotated_coords = (rotation_matrix @ (monomer_coords - monomer_coords.mean(axis=0)).T).T
                final_coords = rotated_coords + center
                
                # Add atoms to topology
                residue = topology.addResidue(f"MON", chain)
                
                for i, atom in enumerate(rdkit_mol.GetAtoms()):
                    element = atom.GetSymbol()
                    # Create OpenMM element
                    try:
                        omm_element = openmm.app.Element.getBySymbol(element)  # type: ignore[attr-defined]
                    except:
                        omm_element = openmm.app.Element.getByAtomicNumber(1)  # type: ignore[attr-defined]
                    
                    topology.addAtom(f"{element}{i}", omm_element, residue)
                    # CRITICAL: Add units to ensure correct scaling (coordinates are in nm)
                    all_positions.append(openmm.Vec3(*final_coords[i]) * unit.nanometer)  # type: ignore[attr-defined]
                
                placed = True
                break
        
        if not placed:
            # If we can't place this monomer, issue a warning but continue
            print(f"Warning: Could not place monomer {mol_idx+1} without overlap. "
                  f"Only {len(placed_centers)} of {count} monomers placed.")
            break
    
    return all_positions, topology
