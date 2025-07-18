"""
moml/simulation/molecular_dynamics/equilibration.py

System equilibration module for molecular dynamics simulations.

This module implements a deterministic multi-stage equilibration protocol:
1. Energy minimization to remove bad contacts
2. NVT equilibration with position restraints to equilibrate velocities
3. NPT equilibration to equilibrate system density and volume

The protocol ensures proper system preparation before production simulations
and helps prevent simulation instabilities.
"""

from typing import Tuple
from copy import deepcopy

import structlog
from openmm import (
    AndersenThermostat,
    Context,
    CustomExternalForce,
    LocalEnergyMinimizer,
    MonteCarloBarostat,
    System,
    VerletIntegrator,
    unit,
)

from .builder.system_builder import SystemBuilder
from .config.schema import MDConfig

# Configure logger
logger = structlog.get_logger()


class EquilibrationProtocol:
    """
    Handles system equilibration through a series of stages.
    
    This class implements a standard equilibration protocol for molecular systems:
    minimization, followed by NVT with position restraints, and finally NPT
    equilibration. Each stage prepares the system for production MD.
    
    Attributes:
        config: Configuration parameters for the equilibration
        system_builder: Builder for creating OpenMM systems
        platform: OpenMM platform to use for calculations
    """
    
    def __init__(self, config: MDConfig, system_builder: SystemBuilder) -> None:
        """
        Initialize the equilibration protocol.
        
        Args:
            config: Configuration parameters for the equilibration
            system_builder: Builder for creating OpenMM systems
        """
        self.config = config
        self.system_builder = system_builder
        self.platform = system_builder.get_platform()
    
    def run(self, system: System, positions: unit.Quantity) -> Tuple[System, unit.Quantity]:
        """
        Run the full equilibration protocol.
        
        Executes all stages of equilibration in sequence: minimization, NVT, and NPT.
        
        Args:
            system: OpenMM system to equilibrate
            positions: Initial atomic positions
            
        Returns:
            Tuple containing the equilibrated system and positions
            
        Raises:
            ValueError: If equilibration fails due to instability
        """
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
        """
        Run energy minimization to remove bad contacts.
        
        Args:
            system: OpenMM system to minimize
            positions: Initial atomic positions
            
        Returns:
            Minimized positions
        """
        logger.info("starting_minimization", steps=self.config.equilibration.minimization_steps)
        
        integrator = VerletIntegrator(0.001 * unit.picoseconds)  # type: ignore
        context = Context(system, integrator, self.platform)
        context.setPositions(positions)
        
        # Minimize
        context.setVelocitiesToTemperature(0 * unit.kelvin)  # type: ignore
        LocalEnergyMinimizer.minimize(
            context,
            maxIterations=self.config.equilibration.minimization_steps
        )
        
        # Get minimized positions and energy
        state = context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # type: ignore
        
        logger.info("minimization_complete", energy=energy)
        return positions
    
    def _nvt_equilibration(self, system: System, positions: unit.Quantity) -> unit.Quantity:
        """
        Run NVT equilibration with position restraints.
        
        This stage equilibrates velocities while keeping positions restrained
        to prevent large conformational changes.
        
        Args:
            system: OpenMM system to equilibrate
            positions: Minimized atomic positions
            
        Returns:
            Equilibrated positions after NVT
        """
        logger.info("starting_nvt", steps=self.config.equilibration.nvt_steps)
        
        # Create a deep copy of the system so we don't modify the caller's system object.
        nvt_system = self._add_position_restraints(deepcopy(system), positions)
        
        # Setup NVT simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)  # type: ignore
        integrator.setConstraintTolerance(1e-5)
        nvt_system.addForce(AndersenThermostat(
            self.config.system.temperature * unit.kelvin,  # type: ignore
            1.0 / unit.picoseconds  # type: ignore
        ))
        context = Context(nvt_system, integrator, self.platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)  # type: ignore
        
        # Run NVT
        for i in range(self.config.equilibration.nvt_steps):
            integrator.step(1)
            if i % 1000 == 0:
                state = context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # type: ignore
                logger.info("nvt_progress", step=i, energy=energy)
        
        # Get final positions
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        
        logger.info("nvt_complete")
        return positions
    
    def _npt_equilibration(self, system: System, positions: unit.Quantity) -> unit.Quantity:
        """
        Run NPT equilibration to equilibrate density and volume.
        
        Args:
            system: OpenMM system to equilibrate
            positions: Positions from NVT equilibration
            
        Returns:
            Fully equilibrated positions
        """
        logger.info("starting_npt", steps=self.config.equilibration.npt_steps)
        
        # Add barostat
        npt_system = self._add_barostat(system)
        
        # Setup NPT simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)  # type: ignore
        context = Context(npt_system, integrator, self.platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)  # type: ignore
        
        # Run NPT
        for i in range(self.config.equilibration.npt_steps):
            integrator.step(1)
            if i % 1000 == 0:
                state = context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # type: ignore
                volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)  # type: ignore
                logger.info("npt_progress", step=i, energy=energy, volume=volume)
        
        # Get final positions
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        
        logger.info("npt_complete")
        return positions
    
    def _add_position_restraints(self, system: System, positions: unit.Quantity) -> System:
        """
        Add harmonic position restraints to all atoms in the system.
        
        Args:
            system: OpenMM system to modify
            positions: Reference positions for restraints
            
        Returns:
            Modified system with position restraints
        """
        force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addPerParticleParameter("k")
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        
        # Add restraints to all atoms
        for i in range(system.getNumParticles()):
            pos = positions[i]
            force.addParticle(i, [
                self.config.equilibration.restraint_force * unit.kilojoules_per_mole/unit.nanometer**2,  # type: ignore
                pos[0].value_in_unit(unit.nanometer),  # type: ignore
                pos[1].value_in_unit(unit.nanometer),  # type: ignore
                pos[2].value_in_unit(unit.nanometer)  # type: ignore
            ])
        
        system.addForce(force)
        return system
    
    def _add_barostat(self, system: System) -> System:
        """
        Add Monte Carlo barostat to the system for NPT simulation.
        
        Args:
            system: OpenMM system to modify
            
        Returns:
            Modified system with barostat
        """
        barostat_frequency = getattr(
            getattr(self.config, "equilibration", None), "barostat_frequency", 25
        )
        barostat = MonteCarloBarostat(
            self.config.system.pressure * unit.atmospheres,  # type: ignore
            self.config.system.temperature * unit.kelvin,  # type: ignore
            barostat_frequency,  # Frequency
        )
        system.addForce(barostat)
        return system 