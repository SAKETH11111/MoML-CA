"""
Monitoring module for tracking system stability during MD.
"""

from typing import List, Optional
import numpy as np
from openmm import State, app
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
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
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
        density = state.getDensity().value_in_unit(unit.grams_per_milliliter)
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
        temp = state.getTemperature().value_in_unit(unit.kelvin)
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
    
    def as_reporter(self, reportInterval: int = 1000) -> app.StateDataReporter:
        """Create a StateDataReporter that calls this watchdog."""
        def callback(state: State, step: int):
            self._check_state(step, state)
        
        return app.StateDataReporter(None, reportInterval,
                                   step=True,
                                   temperature=True,
                                   potentialEnergy=True,
                                   callback=callback)
    
    def _check_state(self, step: int, state: State):
        """Check simulation state for divergence."""
        temp = state.getTemperature().value_in_unit(unit.kelvin)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        # Check temperature
        if temp > self.temp_max:
            raise SimulationDiverged(f"Temperature {temp:.1f}K exceeds maximum {self.temp_max}K")
        
        # Check energy drift
        if self.last_energy is not None and self.last_step is not None:
            steps = step - self.last_step
            energy_diff = energy - self.last_energy
            drift = energy_diff / (steps * 0.002)  # kJ/mol/ns assuming 2fs timestep
            
            if abs(drift) > self.energy_drift_kj_per_ns:
                raise SimulationDiverged(
                    f"Energy drift {drift:.1f} kJ/mol/ns exceeds maximum {self.energy_drift_kj_per_ns}")
        
        self.last_energy = energy
        self.last_step = step 