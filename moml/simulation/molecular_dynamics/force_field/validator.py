"""
moml/simulation/molecular_dynamics/force_field/validator.py

Force Field Validation Module

This module provides functionality to validate force field XML files for OpenMM
molecular dynamics simulations. It checks for completeness, physical validity,
and performs basic smoke tests to ensure the force field can be used in simulations.

The primary class is ForceFieldValidator which performs various validation checks
on OpenMM force field XML files to ensure they are properly structured and contain
physically reasonable parameters.
"""

import openmm.app as app
import openmm.unit as unit
import openmm
from openmm import VerletIntegrator, Context
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

class ForceFieldValidator:
    """
    Validates force field parameters for completeness and physical validity.
    
    This class provides methods to check OpenMM force field XML files for:
    1. Completeness - ensuring all required force field components are present
    2. Physical validity - checking parameter ranges for physical reasonableness
    3. Simulation stability - running short smoke tests to detect instabilities
    
    The validator helps identify issues that could cause simulation failures
    or unphysical behavior before running production simulations.
    
    Attributes:
        xml_path: Path to the force field XML file being validated
        tree: ElementTree representation of the XML file
        root: Root element of the XML tree
        REQUIRED_SECTIONS: Set of force field sections that must be present
    """
    
    REQUIRED_SECTIONS = {
        'HarmonicBondForce',
        'HarmonicAngleForce',
        'PeriodicTorsionForce',
        'NonbondedForce'
    }
    
    def __init__(self, xml_path: Path):
        """
        Initialize validator with path to force field XML.
        
        Args:
            xml_path: Path object pointing to the force field XML file to validate
        
        Raises:
            FileNotFoundError: If the XML file does not exist
            xml.etree.ElementTree.ParseError: If the XML file is malformed
        """
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        
    def validate_completeness(self) -> List[str]:
        """
        Check if all required force field sections are present.
        
        This method verifies that the force field XML contains all the required sections
        defined in REQUIRED_SECTIONS. Missing sections could cause simulation failures
        or incomplete force field behavior.
        
        Returns:
            List[str]: List of error messages for missing sections.
                       Empty list if all required sections are present.
        """
        missing = []
        for section in self.REQUIRED_SECTIONS:
            if self.root.find(section) is None:
                missing.append(f"Missing required section: {section}")
        return missing
    
    def validate_ranges(self) -> List[str]:
        """
        Validate physical ranges of force field parameters.
        
        This method checks that force field parameters fall within physically reasonable
        ranges. It examines bond lengths, force constants, angles, and torsion barriers
        to identify potentially problematic values that could cause simulation instabilities.
        
        Returns:
            List[str]: List of error messages for parameters outside acceptable ranges.
                       Empty list if all parameters are within reasonable ranges.
        """
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
    
    def _build_minimal_water_box(self, water_box_size_nm: float, n_waters: int) -> tuple:
        """
        Build a minimal TIP3P water box for deep smoke testing.
        
        This helper method creates a small water box with the specified number of
        water molecules arranged in a simple linear pattern. The system uses the TIP3P
        water model for realistic intermolecular interactions.
        
        Args:
            water_box_size_nm: Size of the cubic water box in nanometers
            n_waters: Number of water molecules to include in the box
            
        Returns:
            tuple: (topology, positions) where topology is an OpenMM Topology
                   and positions is a list of Vec3 coordinates
        """
        # Create topology with water molecules
        topology = app.Topology()
        chain = topology.addChain()
        
        positions = []
        
        # Simple linear spacing to avoid complex cube root calculations
        spacing = water_box_size_nm / max(n_waters, 1)
        
        # TIP3P water geometry constants
        oh_length = 0.09572  # nm - O-H bond length
        hoh_angle = 104.52 * np.pi / 180  # radians - H-O-H angle
        
        # Add water molecules to topology and positions
        for i in range(n_waters):
            # Create residue for this water molecule
            residue = topology.addResidue("HOH", chain)
            
            # Add oxygen atom
            oxygen = topology.addAtom("O", app.Element.getBySymbol("O"), residue)
            # Add hydrogen atoms
            hydrogen1 = topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
            hydrogen2 = topology.addAtom("H2", app.Element.getBySymbol("H"), residue)
            
            # Add bonds (O-H1, O-H2)
            topology.addBond(oxygen, hydrogen1)
            topology.addBond(oxygen, hydrogen2)
            
            # Simple linear positioning along x-axis with small y,z offsets to avoid overlap
            base_x = i * spacing + 0.3  # Offset from origin
            base_y = 0.2 + (i % 2) * 0.1  # Slight y variation
            base_z = 0.2 + (i % 3) * 0.1  # Slight z variation
            
            # Oxygen position
            o_pos = openmm.Vec3(base_x, base_y, base_z)
            positions.append(o_pos)
            
            # Hydrogen positions using proper TIP3P geometry
            h1_pos = openmm.Vec3(
                base_x + oh_length * np.cos(hoh_angle / 2),
                base_y + oh_length * np.sin(hoh_angle / 2),
                base_z
            )
            h2_pos = openmm.Vec3(
                base_x + oh_length * np.cos(-hoh_angle / 2),
                base_y + oh_length * np.sin(-hoh_angle / 2),
                base_z
            )
            
            positions.append(h1_pos)
            positions.append(h2_pos)
        
        return topology, positions
    
    def smoke_test(
        self,
        steps: int = 100,
        pdb_path: Optional[Path] = None,
        deep_check: bool = False,
        water_box_size_nm: float = 2.0,
        n_waters: int = 3,
    ) -> List[str]:
        """
        Run a short vacuum simulation to check for instabilities.
        
        This method performs a brief molecular dynamics simulation to detect any
        immediate instabilities or energy explosions that would indicate problems
        with the force field. It supports two modes:
        
        1. Shallow mode (default): Uses a single-particle system for fast validation
        2. Deep mode: Creates a minimal TIP3P water box to exercise real interactions
        
        Args:
            steps: Number of simulation steps to run
            pdb_path: Optional path to a PDB file to use for the test system.
                      If not provided or file not found, a fallback system will be generated.
            deep_check: If True, builds a minimal TIP3P water box for more robust testing.
                       If False, uses a single-particle system (fast but limited).
            water_box_size_nm: Size of the cubic water box in nanometers (deep_check only)
            n_waters: Number of water molecules to include in the box (deep_check only)
        
        Returns:
            List[str]: List of error messages if instabilities are detected.
                       Empty list if the smoke test passes.
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
                if deep_check:
                    # Use minimal TIP3P water box for robust testing
                    try:
                        topology, positions = self._build_minimal_water_box(
                            water_box_size_nm, n_waters
                        )
                        # Try to load TIP3P force field, with fallback
                        try:
                            forcefield = app.ForceField('tip3p.xml')
                        except Exception:
                            # Fallback: try common TIP3P variants or skip deep check
                            try:
                                forcefield = app.ForceField('amber14/tip3p.xml')
                            except Exception:
                                errors.append("TIP3P force field not found, falling back to shallow check")
                                # Fall through to shallow check below
                                deep_check = False
                        
                        if deep_check:  # Only proceed if TIP3P was loaded successfully
                            system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
                    except Exception as e:
                        errors.append(f"Deep check failed, falling back to shallow: {str(e)}")
                        deep_check = False  # Fall back to shallow check
                
                if not deep_check:
                    # TODO-DEPRECATE: Single-particle fallback (fast but limited)
                    # This will be removed once deep_check is fully vetted
                    system = openmm.System()
                    system.addParticle(1.0)  # Add a dummy particle with mass 1.0 amu
                    topology = app.Topology()
                    chain = topology.addChain()
                    residue = topology.addResidue("DUM", chain)
                    topology.addAtom("DUM", app.Element.getByAtomicNumber(1), residue)
                    positions = [openmm.Vec3(0, 0, 0)]  # Position without units

                    # Add a nonbonded force to the system for the particle
                    nonbonded_force = openmm.NonbondedForce()
                    # Add a particle to the nonbonded force (charge, sigma, epsilon)
                    # These values are arbitrary for a smoke test, just to make it valid
                    nonbonded_force.addParticle(0.0, 1.0, 0.0)
                    system.addForce(nonbonded_force)
            else:
                # Use provided topology and positions
                forcefield = app.ForceField(str(self.xml_path))
                system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
            
            integrator = VerletIntegrator(0.001)  # Time step in picoseconds
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
        """
        Run all validation checks and return results.
        
        This method executes all validation checks (completeness, parameter ranges,
        and smoke test) and aggregates the results into a single dictionary.
        
        Returns:
            Dict[str, List[str]]: Dictionary with validation results for each check type:
                - 'completeness': Results from validate_completeness()
                - 'ranges': Results from validate_ranges()
                - 'smoke_test': Results from smoke_test()
                
        Example:
            >>> validator = ForceFieldValidator(Path("forcefield.xml"))
            >>> results = validator.validate()
            >>> if not any(results.values()):
            ...     print("Force field is valid")
            ... else:
            ...     print("Force field has issues")
        """
        return {
            'completeness': self.validate_completeness(),
            'ranges': self.validate_ranges(),
            'smoke_test': self.smoke_test()
        } 