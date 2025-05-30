"""
System builder for molecular dynamics simulations.
"""

from pathlib import Path
from typing import Dict, Optional
from openmm import app, System, Platform, unit
from rdkit import Chem

from ..config import MDConfig
from ..force_field.validator import ForceFieldValidator
from ..force_field.mapper import ForceFieldMapper
from ..force_field.plugins import load_plugin

def build_system(
    pdb_path: Path,
    ff_params: Dict,
    config: MDConfig,
    surface_name: Optional[str] = None,
    solvent_name: Optional[str] = None,
    net_charge_e: int = 0,
    ionic_strength_m: float = 0.1
) -> System:
    """Build an OpenMM system from components.
    
    This is a convenience function that creates a SystemBuilder instance
    and calls its build_system method.
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
    """Builds OpenMM systems from force field components."""
    
    def __init__(self, config: MDConfig | None = None):
        self.config = config or MDConfig()
        self.validator = None  # Will be set when force field is loaded
        self.force_field_mapper = ForceFieldMapper()
    
    def build_system(self, 
                    pdb_path: Path,
                    ff_params: Dict,
                    surface_name: Optional[str] = None,
                    solvent_name: Optional[str] = None,
                    net_charge_e: int = 0,
                    ionic_strength_m: float = 0.1) -> System:
        """Build an OpenMM system from components."""
        
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
        for category, msgs in validation_results.items():
            if msgs:
                errors.extend(msgs)
        
        if errors:
            raise ValueError(f"Force field validation failed:\n" + "\n".join(errors))
        
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
                raise ValueError(f"Failed to load surface plugin {surface_name}: {e}")
        
        # Add solvent if specified
        if solvent_name:
            try:
                solvent_cfg, solvent_build = load_plugin(solvent_name)
                # Add solvent to force field using the build module
                forcefield.registerTemplateGenerator(solvent_build.get_xml)
            except Exception as e:
                raise ValueError(f"Failed to load solvent plugin {solvent_name}: {e}")
        
        # Create modeller and add ions
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addIons(forcefield, ionicStrength=ionic_strength_m*unit.molar, neutralize=True)
        
        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005
        )
        
        # Apply HMR if enabled
        if self.config.integration.hmr_enabled:
            self._apply_hmr(system)
        
        return system
    
    def _apply_hmr(self, system: System):
        """Apply hydrogen mass repartitioning to the system."""
        factor = self.config.integration.hmr_factor
        
        for i in range(system.getNumParticles()):
            mass = system.getParticleMass(i)
            if mass.value_in_unit(unit.dalton) < 2.0:  # Hydrogen
                system.setParticleMass(i, mass * factor)
            else:  # Heavy atom
                system.setParticleMass(i, mass - (factor - 1) * unit.dalton)
    
    def get_platform(self) -> Platform:
        """Get the OpenMM platform based on configuration."""
        platform_name = self.config.platform
        platform = Platform.getPlatformByName(platform_name)
        
        if platform_name == "CUDA":
            if self.config.device_index is not None:
                platform.setPropertyDefaultValue('DeviceIndex', str(self.config.device_index))
            platform.setPropertyDefaultValue('Precision', 'mixed')
        
        return platform 
