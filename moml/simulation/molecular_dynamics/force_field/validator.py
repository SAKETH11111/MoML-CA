import openmm.app as app
import openmm.unit as unit
import openmm
from openmm import VerletIntegrator, Context
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

"""
Force field validation module for ensuring completeness and physical validity
of force field parameters before simulation.
"""

class ForceFieldValidator:
    """Validates force field parameters for completeness and physical validity."""
    
    REQUIRED_SECTIONS = {
        'HarmonicBondForce',
        'HarmonicAngleForce',
        'PeriodicTorsionForce',
        'NonbondedForce'
    }
    
    def __init__(self, xml_path: Path):
        """Initialize validator with path to force field XML."""
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        
    def validate_completeness(self) -> List[str]:
        """Check if all required force field sections are present."""
        missing = []
        for section in self.REQUIRED_SECTIONS:
            if self.root.find(section) is None:
                missing.append(f"Missing required section: {section}")
        return missing
    
    def validate_ranges(self) -> List[str]:
        """Validate physical ranges of force field parameters."""
        errors = []
        
        # Check bond parameters
        for bond in self.root.findall('.//HarmonicBondForce/Bond'):
            length = float(bond.get('length', 0))
            k = float(bond.get('k', 0))
            if length <= 0 or length > 0.5:  # nm
                errors.append(f"Invalid bond length: {length} nm")
            if k <= 0 or k > 500000:  # kJ/mol/nm²
                errors.append(f"Invalid bond force constant: {k} kJ/mol/nm²")
        
        # Check angle parameters
        for angle in self.root.findall('.//HarmonicAngleForce/Angle'):
            angle_val = float(angle.get('angle', 0))
            k = float(angle.get('k', 0))
            if angle_val <= 0 or angle_val > np.pi:
                errors.append(f"Invalid angle: {angle_val} rad")
            if k <= 0 or k > 1000:  # kJ/mol/rad²
                errors.append(f"Invalid angle force constant: {k} kJ/mol/rad²")
        
        # Check torsion parameters
        for torsion in self.root.findall('.//PeriodicTorsionForce/Proper'):
            k = float(torsion.get('k', 0))
            if k < 0 or k > 100:  # kJ/mol
                errors.append(f"Invalid torsion barrier: {k} kJ/mol")
        
        return errors
    
    def smoke_test(self, steps: int = 100, pdb_path: Optional[Path] = None) -> List[str]:
        """Run a short vacuum simulation to check for instabilities.
        
        Args:
            steps: Number of simulation steps to run.
            pdb_path: Optional path to a PDB file to use for the test system.
                      If not provided or file not found, a simple water molecule
                      system will be generated.
        Returns:
            A list of error messages if instabilities are detected.
        """
        errors = []
        try:
            topology = None
            positions = None

            if pdb_path and pdb_path.exists():
                try:
                    pdb = app.PDBFile(str(pdb_path))
                    topology = pdb.topology
                    positions = pdb.positions
                except Exception as e:
                    errors.append(f"Failed to load PDB file {pdb_path}")
                    return errors
            
            if topology is None or positions is None:
                # Generate a simple water molecule system if no valid PDB is provided
                # This requires a minimal force field for water, e.g., TIP3P
                # For a generic smoke test, we can create a single atom system
                # or assume a basic force field is loaded.
                # For now, let's create a simple system with one atom.
                # This might need a more robust way to get atom types from the FF.
                # For a true smoke test, we need a valid molecule.
                # Let's create a simple system with a single particle.
                system = openmm.System()
                system.addParticle(1.0 * unit.amu) # Add a dummy particle with some mass
                topology = app.Topology()
                chain = topology.addChain()
                residue = topology.addResidue("DUM", chain)
                topology.addAtom("DUM", app.Element.getByAtomicNumber(1), residue)
                positions = [openmm.Vec3(0,0,0) * unit.nanometer]

                # Add a nonbonded force to the system for the particle
                nonbonded_force = openmm.NonbondedForce()
                # Add a particle to the nonbonded force (charge, sigma, epsilon)
                # These values are arbitrary for a smoke test, just to make it valid
                nonbonded_force.addParticle(0.0, 1.0 * unit.angstrom, 0.0 * unit.kilojoules_per_mole)
                system.addForce(nonbonded_force)

            forcefield = app.ForceField(str(self.xml_path))
            system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
            
            integrator = VerletIntegrator(0.001 * unit.picoseconds)
            context = Context(system, integrator)
            context.setPositions(positions)
            
            for _ in range(steps):
                integrator.step(1)
                state = context.getState(getEnergy=True)
                if state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) > 1e6:
                    errors.append("Energy explosion detected in smoke test")
                    break
                    
        except Exception as e:
            errors.append(f"Smoke test failed: {str(e)}")
            
        return errors
    
    def validate(self) -> Dict[str, List[str]]:
        """Run all validation checks and return results."""
        return {
            'completeness': self.validate_completeness(),
            'ranges': self.validate_ranges(),
            'smoke_test': self.smoke_test()
        } 