"""
moml/simulation/molecular_dynamics/builder/system_builder.py

System builder for molecular dynamics simulations.

This module provides functionality for building OpenMM systems from molecular
structures and force field parameters. It handles force field generation,
validation, solvent addition, and system setup for molecular dynamics simulations.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from openmm import Platform, System, unit
from openmm import app
from rdkit import Chem

from ..config import MDConfig
from ..force_field.mapper import ForceFieldMapper
from ..force_field.plugins import load_plugin
from ..force_field.validator import ForceFieldValidator


def build_system(
    pdb_path: Path,
    ff_params: Dict,
    config: MDConfig,
    surface_name: Optional[str] = None,
    solvent_name: Optional[str] = None,
    net_charge_e: int = 0,
    ionic_strength_m: float = 0.1
) -> System:
    """
    Build an OpenMM system from components.
    
    This is a convenience function that creates a SystemBuilder instance
    and calls its build_system method.
    
    Args:
        pdb_path: Path to PDB file containing molecular structure
        ff_params: Dictionary of force field parameters
        config: Configuration parameters for the system
        surface_name: Optional name of surface plugin to use
        solvent_name: Optional name of solvent plugin to use
        net_charge_e: Net charge of the system in elementary charge units
        ionic_strength_m: Ionic strength for adding counterions in molar units
        
    Returns:
        OpenMM System object ready for simulation
        
    Raises:
        ValueError: If system building fails
    """
    builder = SystemBuilder(config)
    return builder.build_system(
        pdb_path=pdb_path,
        ff_params=ff_params,
        surface_name=surface_name,
        solvent_name=solvent_name,
        net_charge_e=net_charge_e,
        ionic_strength_m=ionic_strength_m
    )


class SystemBuilder:
    """
    Builds OpenMM systems from force field components.
    
    This class handles the creation of OpenMM System objects from molecular
    structures and force field parameters. It supports custom force fields,
    surfaces, and solvents through a plugin system.
    
    Attributes:
        config: Configuration parameters for the system
        validator: Validator for checking force field files
        force_field_mapper: Mapper for converting ML predictions to force fields
    """
    
    def __init__(self, config: Optional[MDConfig] = None) -> None:
        """
        Initialize the system builder.
        
        Args:
            config: Configuration parameters for the system
        """
        self.config = config or MDConfig()
        self.validator = None  # Will be set when force field is loaded
        self.force_field_mapper = ForceFieldMapper()
    
    def build_system(
        self, 
        pdb_path: Path,
        ff_params: Dict,
        surface_name: Optional[str] = None,
        solvent_name: Optional[str] = None,
        net_charge_e: int = 0,
        ionic_strength_m: float = 0.1
    ) -> System:
        """
        Build an OpenMM system from components.
        
        Args:
            pdb_path: Path to PDB file containing molecular structure
            ff_params: Dictionary of force field parameters
            surface_name: Optional name of surface plugin to use
            solvent_name: Optional name of solvent plugin to use
            net_charge_e: Net charge of the system in elementary charge units
            ionic_strength_m: Ionic strength for adding counterions in molar units
            
        Returns:
            OpenMM System object ready for simulation
            
        Raises:
            ValueError: If system building fails
            FileNotFoundError: If PDB file doesn't exist
        """
        # Convert PDB to RDKit molecule
        mol = Chem.MolFromPDBFile(str(pdb_path))
        if mol is None:
            raise ValueError(f"Failed to load molecule from {pdb_path}")
        
        # Write force field XML using ForceFieldMapper
        ff_xml_path = pdb_path.parent / 'ff_params.xml'
        success, results = self.force_field_mapper.convert_mgnn_predictions_to_force_field(
            mol=mol,
            node_predictions=ff_params,
            output_dir=str(ff_xml_path.parent),
            base_filename=ff_xml_path.stem
        )
        
        if not success:
            raise ValueError("Failed to generate force field XML")
        
        # Validate force field
        self.validator = ForceFieldValidator(ff_xml_path)
        validation_results = self.validator.validate()
        
        # Check for validation errors
        errors = []
        for msgs in validation_results.values():
            if msgs:
                errors.extend(msgs)
        
        if errors:
            raise ValueError("Force field validation failed:\n" + "\n".join(errors))
        
        # Load PDB
        pdb = app.PDBFile(str(pdb_path))
        
        # Create force field
        forcefield = app.ForceField(str(ff_xml_path))
        
        # Add surface if specified
        if surface_name:
            try:
                surface_cfg, surface_build = load_plugin(surface_name)
                # Add surface to force field using the build module
                forcefield.registerTemplateGenerator(surface_build.get_xml)
            except Exception as e:
                raise ValueError(f"Failed to load surface plugin {surface_name}") from e
        
        # Add solvent if specified
        if solvent_name:
            try:
                solvent_cfg, solvent_build = load_plugin(solvent_name)
                # Add solvent to force field using the build module
                forcefield.registerTemplateGenerator(solvent_build.get_xml)
            except Exception as e:
                raise ValueError(f"Failed to load solvent plugin {solvent_name}") from e
        
        # Create modeller and add ions
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addIons(  # type: ignore
            forcefield, 
            ionicStrength=ionic_strength_m * unit.molar, # type: ignore
            neutralize=True
        )
        
        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,  # type: ignore
            nonbondedCutoff=1.0 * unit.nanometers,  # type: ignore
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005
        )
        
        # Apply HMR if enabled
        if self.config.integration.hmr_enabled:
            self._apply_hmr(system, modeller.topology)
        
        return system
    
    def _apply_hmr(self, system: System, topology: app.Topology) -> None:
        """
        Apply hydrogen mass repartitioning to the system.
        
        This method increases the mass of hydrogen atoms and decreases the mass
        of connected heavy atoms to allow for larger integration timesteps
        without affecting the overall dynamics.
        
        Args:
            system: OpenMM system to modify
            topology: OpenMM topology containing bond information
            
        Raises:
            ValueError: If mass repartitioning fails
        """
        factor = self.config.integration.hmr_factor
        
        # Store original masses to ensure conservation
        original_masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]
        
        # Identify heavy atoms and their bonded hydrogens
        heavy_atom_to_hydrogens = {}
        for bond in topology.bonds():
            atom1 = bond.atom1
            atom2 = bond.atom2
            
            # Check for heavy atom - hydrogen bond
            if atom1.element.mass.value_in_unit(unit.dalton) > 2.0 and atom2.element.mass.value_in_unit(unit.dalton) < 2.0:
                heavy_atom_to_hydrogens.setdefault(atom1.index, []).append(atom2.index)
            elif atom2.element.mass.value_in_unit(unit.dalton) > 2.0 and atom1.element.mass.value_in_unit(unit.dalton) < 2.0:
                heavy_atom_to_hydrogens.setdefault(atom2.index, []).append(atom1.index)
        
        # Apply HMR
        for i in range(system.getNumParticles()):
            mass = original_masses[i]
            if mass.value_in_unit(unit.dalton) < 2.0:  # type: ignore # Hydrogen
                system.setParticleMass(i, mass * factor)
            elif i in heavy_atom_to_hydrogens:  # Heavy atom bonded to hydrogens
                # Calculate total mass transferred to hydrogens from this heavy atom
                total_transferred_mass = unit.Quantity(0.0, unit.dalton)
                for h_idx in heavy_atom_to_hydrogens[i]:
                    h_mass = original_masses[h_idx]
                    total_transferred_mass += (h_mass * factor) - h_mass
                
                # Subtract the total transferred mass from the heavy atom
                system.setParticleMass(i, mass - total_transferred_mass)
            # else: heavy atom not bonded to hydrogens, mass remains unchanged
    
    def get_platform(self) -> Platform:
        """
        Get the OpenMM platform based on configuration.
        
        Returns:
            OpenMM Platform object configured according to settings
        """
        platform_name = self.config.platform
        platform = Platform.getPlatformByName(platform_name)
        
        if platform_name == "CUDA":
            if self.config.device_index is not None:
                platform.setPropertyDefaultValue('DeviceIndex', str(self.config.device_index))
            platform.setPropertyDefaultValue('Precision', 'mixed')
        
        return platform 
