"""
System equilibration module for preparing systems for production MD.
Implements a deterministic protocol: minimization → NVT → NPT.
"""

from typing import Tuple
from openmm import app, Context, VerletIntegrator, MonteCarloBarostat, System, LocalEnergyMinimizer, CustomExternalForce, AndersenThermostat
from openmm import unit
import structlog

from .config.schema import MDConfig
from .builder.system_builder import SystemBuilder

logger = structlog.get_logger()

class EquilibrationProtocol:
    """Handles system equilibration through a series of stages."""
    
    def __init__(self, config: MDConfig, system_builder: SystemBuilder):
        self.config = config
        self.system_builder = system_builder
        self.platform = system_builder.get_platform()
    
    def run(self, system: System, positions: unit.Quantity) -> Tuple[System, unit.Quantity]:
        """Run the full equilibration protocol."""
        logger.info("starting_equilibration")
        
        # Minimization
        positions = self._minimize(system, positions)
        
        # NVT equilibration
        positions = self._nvt_equilibration(system, positions)
        
        # NPT equilibration
        positions = self._npt_equilibration(system, positions)
        
        logger.info("equilibration_complete")
        return system, positions
    
    def _minimize(self, system: System, positions: unit.Quantity) -> unit.Quantity:
        """Run energy minimization."""
        logger.info("starting_minimization", steps=self.config.equilibration.minimization_steps)
        
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        context = Context(system, integrator, self.platform)
        context.setPositions(positions)
        
        # Minimize
        context.setVelocitiesToTemperature(0 * unit.kelvin)
        LocalEnergyMinimizer.minimize(context,
                                        maxIterations=self.config.equilibration.minimization_steps)
        
        # Get minimized positions and energy
        state = context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        logger.info("minimization_complete", energy=energy)
        return positions
    
    def _nvt_equilibration(self, system: System, positions: unit.Quantity) -> unit.Quantity:
        """Run NVT equilibration with position restraints."""
        logger.info("starting_nvt", steps=self.config.equilibration.nvt_steps)
        
        # Create NVT system with restraints
        nvt_system = self._add_position_restraints(system, positions)
        
        # Setup NVT simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)
        integrator.setConstraintTolerance(1e-5)
        nvt_system.addForce(AndersenThermostat(self.config.system.temperature * unit.kelvin, 1.0 / unit.picoseconds))
        context = Context(nvt_system, integrator, self.platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)
        
        # Run NVT
        for i in range(self.config.equilibration.nvt_steps):
            integrator.step(1)
            if i % 1000 == 0:
                state = context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                logger.info("nvt_progress", step=i, energy=energy)
        
        # Get final positions
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        
        logger.info("nvt_complete")
        return positions
    
    def _npt_equilibration(self, system: System, positions: unit.Quantity) -> unit.Quantity:
        """Run NPT equilibration."""
        logger.info("starting_npt", steps=self.config.equilibration.npt_steps)
        
        # Add barostat
        npt_system = self._add_barostat(system)
        
        # Setup NPT simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)
        context = Context(npt_system, integrator, self.platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)
        
        # Run NPT
        for i in range(self.config.equilibration.npt_steps):
            integrator.step(1)
            if i % 1000 == 0:
                state = context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometers**3)
                logger.info("npt_progress", step=i, energy=energy, volume=volume)
        
        # Get final positions
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        
        logger.info("npt_complete")
        return positions
    
    def _add_position_restraints(self, system: System, positions: unit.Quantity) -> System:
        """Add position restraints to the system."""
        force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addPerParticleParameter("k")
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        
        # Add restraints to all atoms
        for i in range(system.getNumParticles()):
            pos = positions[i]
            force.addParticle(i, [
                self.config.equilibration.restraint_force * unit.kilojoules_per_mole/unit.nanometers**2,
                pos[0].value_in_unit(unit.nanometers),
                pos[1].value_in_unit(unit.nanometers),
                pos[2].value_in_unit(unit.nanometers)
            ])
        
        system.addForce(force)
        return system
    
    def _add_barostat(self, system: System) -> System:
        """Add barostat to the system."""
        barostat = MonteCarloBarostat(
            self.config.system.pressure * unit.atmospheres,
            self.config.system.temperature * unit.kelvin
        )
        system.addForce(barostat)
        return system 