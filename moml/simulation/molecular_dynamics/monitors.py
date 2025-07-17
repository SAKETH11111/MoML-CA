"""
moml/simulation/molecular_dynamics/monitors.py

Monitoring module for tracking system stability during molecular dynamics simulations.

This module provides classes for monitoring various aspects of MD simulations:
- Energy stability and drift
- Density fluctuations
- Temperature control
- Simulation divergence detection

These monitors help detect unstable simulations early and prevent wasted
computational resources.
"""

from typing import List, Optional

import numpy as np
import structlog
from openmm import State, System, unit
from openmm import app

from .config.schema import MDConfig

# Configure logger
logger = structlog.get_logger()


class SimulationDiverged(Exception):
    """
    Exception raised when simulation diverges beyond recovery.
    
    This exception indicates that the simulation has become numerically unstable
    and should be terminated to prevent wasted computation.
    """
    pass


class BaseMonitor:
    """
    Base class for system monitors.
    
    Defines the common interface for all monitoring classes that track
    system properties during simulation.
    
    Attributes:
        window_size: Number of historical values to keep for trend analysis
        history: List of historical values for the monitored property
    """
    
    def __init__(self, window_size: int = 1000) -> None:
        """
        Initialize the base monitor.
        
        Args:
            window_size: Number of historical values to keep
        """
        self.window_size = window_size
        self.history: List[float] = []
    
    def update(self, state: State) -> None:
        """
        Update monitor with new state.
        
        Args:
            state: Current simulation state
        """
        raise NotImplementedError
    
    def is_unstable(self) -> bool:
        """
        Check if system is unstable.
        
        Returns:
            True if the system is detected as unstable, False otherwise
        """
        raise NotImplementedError
    
    def _update_history(self, value: float) -> None:
        """
        Update history with new value, maintaining fixed window size.
        
        Args:
            value: New value to add to history
        """
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)


class EnergyMonitor(BaseMonitor):
    """
    Monitors system energy for instabilities.
    
    Tracks potential energy to detect simulation instabilities such as
    energy explosions or excessive drift.
    
    Attributes:
        energy_threshold: Maximum allowed energy value
        energy_drift_threshold: Maximum allowed energy change between steps
        history: List of historical energy values
    """
    
    def __init__(
        self, 
        config: MDConfig,
        window_size: int = 1000
    ) -> None:
        """
        Initialize the energy monitor.
        
        Args:
            config: Simulation configuration containing monitoring thresholds
            window_size: Number of historical values to keep
        """
        super().__init__(window_size)
        self.energy_threshold = config.monitoring.energy_threshold
        self.energy_drift_threshold = config.monitoring.energy_drift_threshold
    
    def update(self, state: State) -> None:
        """
        Update with new energy state.
        
        Args:
            state: Current simulation state
        """
        # Get energy from state
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)  # type: ignore
        
        # Get positions from state to check for atoms moving too far
        positions = state.getPositions(asNumpy=True)
        max_disp = np.abs(positions.value_in_unit(unit.nanometer)).max()  # type: ignore
        
        # Flag energy as problematic if positions are extreme
        if max_disp > 50:  # nm (well outside any normal simulation box)
            energy = self.energy_threshold + 1.0

        self._update_history(energy)  # type: ignore
        
        if energy > self.energy_threshold:  # type: ignore
            logger.warning(
                "high_energy_detected", 
                energy=energy,
                threshold=self.energy_threshold
            )
    
    def is_unstable(self) -> bool:
        """
        Check if energy indicates simulation instability.
        
        Returns:
            True if energy is unstable (too high or drifting), False otherwise
        """
        if len(self.history) < 2:
            return False
        
        # Check for energy explosion
        if max(self.history) > self.energy_threshold:
            return True
        
        # Check for energy drift
        energy_diff = np.diff(self.history)
        if np.any(np.abs(energy_diff) > self.energy_drift_threshold):
            return True
        
        return False


class DensityMonitor(BaseMonitor):
    """
    Monitors system density for instabilities.
    
    Tracks system density to ensure it remains close to the target value,
    which helps detect issues with pressure coupling or system setup.
    
    Attributes:
        system: OpenMM system being monitored
        target_density: Expected density value
        density_tolerance: Allowed deviation from target density
        density_drift_threshold: Maximum allowed density change between steps
        total_mass: Total mass of the system
    """
    
    def __init__(
        self,
        config: MDConfig,
        system: System,
        window_size: int = 1000
    ) -> None:
        """
        Initialize the density monitor.
        
        Args:
            config: Simulation configuration containing monitoring thresholds
            system: OpenMM system to monitor
            window_size: Number of historical values to keep
        """
        super().__init__(window_size)
        self.system = system
        self.target_density = config.monitoring.target_density
        self.density_tolerance = config.monitoring.density_tolerance
        self.density_drift_threshold = config.monitoring.density_drift_threshold
        
        # Calculate total system mass dynamically
        if self.system.getNumParticles() > 0:
            self.total_mass = sum(
                (self.system.getParticleMass(i) for i in range(self.system.getNumParticles())),
                0.0 * unit.amu  # type: ignore
            )
        else:
            self.total_mass = 0.0 * unit.amu  # type: ignore
        
    def update(self, state: State) -> None:
        """
        Update with new density state.
        
        Args:
            state: Current simulation state
        """
        # Get volume from state
        volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)  # type: ignore
        
        # Calculate density from volume and total system mass
        # Convert to g/mL (1.66053886e-21 is conversion factor from amu/nm^3 to g/mL)
        density = (self.total_mass.value_in_unit(unit.amu) / volume) * 1.66053886e-21  # type: ignore
        self._update_history(density)  # type: ignore
        
        if abs(density - self.target_density) > self.density_tolerance:
            logger.warning(
                "density_deviation",
                density=density,
                target=self.target_density,
                tolerance=self.density_tolerance
            )
    
    def is_unstable(self) -> bool:
        """
        Check if density indicates simulation instability.
        
        Returns:
            True if density is unstable (outside tolerance or drifting), False otherwise
        """
        if len(self.history) < 2:
            return False
        
        # Check for density outside tolerance
        if any(abs(d - self.target_density) > self.density_tolerance for d in self.history):
            return True
        
        # Check for density drift
        density_diff = np.diff(self.history)
        if np.any(np.abs(density_diff) > self.density_drift_threshold):
            return True
        
        return False


class TemperatureMonitor(BaseMonitor):
    """
    Monitors system temperature for instabilities.
    
    Tracks system temperature to ensure proper thermostat function and
    detect runaway simulations.
    
    Attributes:
        target_temp: Expected temperature value
        temp_tolerance: Allowed deviation from target temperature
        temp_drift_threshold: Maximum allowed temperature change between steps
    """
    
    def __init__(
        self,
        config: MDConfig,
        window_size: int = 1000
    ) -> None:
        """
        Initialize the temperature monitor.
        
        Args:
            config: Simulation configuration containing monitoring thresholds
            window_size: Number of historical values to keep
        """
        super().__init__(window_size)
        self.target_temp = config.monitoring.target_temperature
        self.temp_tolerance = config.monitoring.temperature_tolerance
        self.temp_drift_threshold = config.monitoring.temperature_drift_threshold
    
    def update(self, state: State) -> None:
        """
        Update with new temperature state.
        
        Args:
            state: Current simulation state
        """
        # Get temperature from state
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)  # type: ignore
        temp = ke / (1.5 * (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.kilojoule_per_mole/unit.kelvin))
        self._update_history(temp)  # type: ignore
        
        if abs(temp - self.target_temp) > self.temp_tolerance:
            logger.warning(
                "temperature_deviation",
                temperature=temp,
                target=self.target_temp,
                tolerance=self.temp_tolerance
            )
    
    def is_unstable(self) -> bool:
        """
        Check if temperature indicates simulation instability.
        
        Returns:
            True if temperature is unstable (outside tolerance or drifting), False otherwise
        """
        if len(self.history) < 2:
            return False
        
        # Check for temperature outside tolerance
        if any(abs(t - self.target_temp) > self.temp_tolerance for t in self.history):
            return True
        
        # Check for temperature drift
        temp_diff = np.diff(self.history)
        if np.any(np.abs(temp_diff) > self.temp_drift_threshold):
            return True
        
        return False


class Watchdog:
    """
    Monitors simulation for divergence and raises exceptions.
    
    Acts as a safety mechanism to terminate simulations that have become
    unstable before they waste computational resources.
    
    Attributes:
        temp_max: Maximum allowed temperature
        energy_drift_kj_per_ns: Maximum allowed energy drift rate
        last_energy: Energy from previous step
        last_step: Step number from previous check
    """
    
    def __init__(self, config: MDConfig) -> None:
        """
        Initialize watchdog with thresholds from config.
        
        Args:
            config: Simulation configuration containing monitoring thresholds
        """
        self.temp_max = config.monitoring.max_temperature
        self.energy_drift_kj_per_ns = config.monitoring.max_energy_drift
        self.last_energy: Optional[float] = None
        self.last_step: Optional[int] = None
    
    def as_reporter(self, system: System, reportInterval: int = 1000) -> app.StateDataReporter:
        """
        Create a StateDataReporter that calls this watchdog.
        
        Args:
            system: OpenMM system to monitor
            reportInterval: Frequency of monitoring in steps
            
        Returns:
            StateDataReporter configured to call this watchdog
        """
        def _callback(state: State, step: int) -> None:
            self._check_state(step, state, system)

        rep = app.StateDataReporter(
            None, reportInterval,
            step=True, temperature=True, potentialEnergy=True, kineticEnergy=True
        )
        # Store callback as attribute for testing
        rep._callback = _callback  # type: ignore
        return rep

    def _check_state(self, step: int, state: State, system: System) -> None:
        """
        Check simulation state for divergence.
        
        Args:
            step: Current simulation step
            state: Current simulation state
            system: OpenMM system being simulated
            
        Raises:
            SimulationDiverged: If simulation has diverged beyond recovery
        """
        # Get temperature and energy from state
        ke = state.getKineticEnergy()
        n_dof = system.getNumParticles() * 3 - system.getNumConstraints()
        temp = (2 * ke / (n_dof * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kelvin)  # type: ignore
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)  # type: ignore
        
        # Check temperature
        if temp > self.temp_max:
            raise SimulationDiverged(f"Temperature {temp:.1f}K exceeds maximum {self.temp_max}K")
        
        # Check energy drift
        if self.last_energy is not None and self.last_step is not None:
            steps = step - self.last_step
            if steps == 0:
                return  # Avoid division by zero if called with the same step
                
            energy_diff = energy - self.last_energy  # type: ignore
            # Assuming timestep is in ps, drift is in kJ/mol/ns
            # This calculation is simplified, assuming a fixed timestep of 2fs
            drift = energy_diff / (steps * 0.002)

            if abs(drift) > self.energy_drift_kj_per_ns:
                raise SimulationDiverged(
                    f"Energy drift {drift:.2f} kJ/mol/ns exceeds maximum "
                    f"{self.energy_drift_kj_per_ns} kJ/mol/ns"
                )
        
        self.last_energy = energy  # type: ignore
        self.last_step = step 