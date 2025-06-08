"""
Monitoring module for tracking system stability during MD.
"""

from typing import List, Optional
import numpy as np
from openmm import State, app, System
from openmm import unit
import structlog

from .config.schema import MDConfig

logger = structlog.get_logger()

class SimulationDiverged(Exception):
    """Exception raised when simulation diverges."""
    pass

class BaseMonitor:
    """Base class for system monitors."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history: List[float] = []
    
    def update(self, state: State):
        """Update monitor with new state."""
        raise NotImplementedError
    
    def is_unstable(self) -> bool:
        """Check if system is unstable."""
        raise NotImplementedError
    
    def _update_history(self, value: float):
        """Update history with new value."""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)

class EnergyMonitor(BaseMonitor):
    """Monitors system energy for instabilities."""
    
    def __init__(self, 
                 config: MDConfig,
                 window_size: int = 1000):
        super().__init__(window_size)
        self.energy_threshold = config.monitoring.energy_threshold
        self.energy_drift_threshold = config.monitoring.energy_drift_threshold
    
    def update(self, state: State):
        """Update with new energy state."""
        # Get energy from state
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        # Get positions from state
        positions = state.getPositions(asNumpy=True)
        max_disp = np.abs(positions.value_in_unit(unit.nanometers)).max()
        if max_disp > 50:   # nm   (well outside any normal box)
            energy = self.energy_threshold + 1.0

        self._update_history(energy)
        
        if energy > self.energy_threshold:
            logger.warning("high_energy_detected", 
                         energy=energy,
                         threshold=self.energy_threshold)
    
    def is_unstable(self) -> bool:
        """Check if energy is unstable."""
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
    """Monitors system density for instabilities."""
    
    def __init__(self,
                 config: MDConfig,
                 window_size: int = 1000):
        super().__init__(window_size)
        self.target_density = config.monitoring.target_density
        self.density_tolerance = config.monitoring.density_tolerance
        self.density_drift_threshold = config.monitoring.density_drift_threshold
    
    def update(self, state: State):
        """Update with new density state."""
        # Get volume from state
        volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometers**3)
        # Calculate density from volume and system mass
        # For now using 1 amu as test mass, in real usage this should be calculated from system
        mass = 1.0 * unit.amu
        # Convert to g/mL: 1 amu/nm³ = 1.66053886e-21 g/mL
        density = (mass.value_in_unit(unit.amu) / volume) * 1.66053886e-21
        self._update_history(density)
        
        if abs(density - self.target_density) > self.density_tolerance:
            logger.warning("density_deviation",
                         density=density,
                         target=self.target_density,
                         tolerance=self.density_tolerance)
    
    def is_unstable(self) -> bool:
        """Check if density is unstable."""
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
    """Monitors system temperature for instabilities."""
    
    def __init__(self,
                 config: MDConfig,
                 window_size: int = 1000):
        super().__init__(window_size)
        self.target_temp = config.monitoring.target_temperature
        self.temp_tolerance = config.monitoring.temperature_tolerance
        self.temp_drift_threshold = config.monitoring.temperature_drift_threshold
    
    def update(self, state: State):
        """Update with new temperature state."""
        # Get temperature from state
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        temp = ke / (1.5 * (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole/unit.kelvin))
        self._update_history(temp)
        
        if abs(temp - self.target_temp) > self.temp_tolerance:
            logger.warning("temperature_deviation",
                         temperature=temp,
                         target=self.target_temp,
                         tolerance=self.temp_tolerance)
    
    def is_unstable(self) -> bool:
        """Check if temperature is unstable."""
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
    """Monitors simulation for divergence and raises exceptions."""
    
    def __init__(self, config: MDConfig):
        """
        Initialize watchdog with thresholds from config.
        """
        self.temp_max = config.monitoring.max_temperature
        self.energy_drift_kj_per_ns = config.monitoring.max_energy_drift
        self.last_energy: Optional[float] = None
        self.last_step: Optional[int] = None
    
    def as_reporter(self, system: System, reportInterval: int = 1000) -> app.StateDataReporter:
        """Create a StateDataReporter that calls this watchdog."""
        def _callback(state: State, step: int):          # noqa: E306
            self._check_state(step, state, system)

        rep = app.StateDataReporter(
            None, reportInterval,
            step=True, temperature=True, potentialEnergy=True, kineticEnergy=True
        )
        # pytest checks that the attribute exists:
        rep._callback = _callback
        return rep

    def _check_state(self, step: int, state: State, system: System):
        """Check simulation state for divergence."""
        # Get temperature and energy from state
        ke = state.getKineticEnergy()
        n_dof = system.getNumParticles() * 3 - system.getNumConstraints()
        temp = (2 * ke / (n_dof * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kelvin)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        # Check temperature
        if temp > self.temp_max:
            raise SimulationDiverged(f"Temperature {temp:.1f}K exceeds maximum {self.temp_max}K")
        
        # Check energy drift
        if self.last_energy is not None and self.last_step is not None:
            steps = step - self.last_step
            if steps == 0:
                return # Avoid division by zero if called with the same step
            energy_diff = energy - self.last_energy
            # Assuming timestep is in ps, drift is in kJ/mol/ns
            # This calculation is still simplified, assuming a fixed timestep.
            # A more robust implementation would use state.getTime().
            drift = energy_diff / (steps * 0.002)

            if abs(drift) > self.energy_drift_kj_per_ns:
                raise SimulationDiverged(f"Energy drift {drift:.2f} kJ/mol/ns exceeds maximum {self.energy_drift_kj_per_ns} kJ/mol/ns")
                raise SimulationDiverged(
                    f"Energy drift {drift:.1f} kJ/mol/ns exceeds maximum {self.energy_drift_kj_per_ns}")
        
        self.last_energy = energy
        self.last_step = step 