"""
Force field validation module for ensuring completeness and physical validity
of force field parameters before simulation.
"""

from typing import Dict, List
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from openmm import app, VerletIntegrator, Context
from openmm import unit

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
    
    def smoke_test(self, steps: int = 100) -> List[str]:
        """Run a short vacuum simulation to check for instabilities."""
        errors = []
        try:
            # Create a minimal system
            pdb = app.PDBFile(str(self.xml_path.parent / 'test.pdb'))
            forcefield = app.ForceField(str(self.xml_path))
            system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
            
            # Setup simulation
            integrator = VerletIntegrator(0.001 * unit.picoseconds)
            context = Context(system, integrator)
            context.setPositions(pdb.positions)
            
            # Run short simulation
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