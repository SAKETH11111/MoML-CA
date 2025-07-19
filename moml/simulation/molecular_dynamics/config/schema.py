"""
moml/simulation/molecular_dynamics/config/schema.py

Configuration schema for Molecular Dynamics simulations.

This module defines the configuration schema for molecular dynamics simulations
using Pydantic for type validation and default values. It provides a comprehensive
set of configuration classes that cover all aspects of MD simulations, with
validation to ensure that parameters are within reasonable ranges.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SystemConfig(BaseModel):
    """
    Configuration for molecular system setup.
    
    This class defines parameters related to the physical system being simulated,
    including thermodynamic conditions and boundary conditions.
    
    Attributes:
        temperature: System temperature in Kelvin
        pressure: System pressure in atmospheres
        box_size: Box dimensions in nanometers [x, y, z]
        periodic: Whether to use periodic boundary conditions
        ph: System pH if using constant pH
    """
    temperature: float = Field(default=300.0, description="System temperature in Kelvin")
    pressure: float = Field(default=1.0, description="System pressure in atmospheres")
    box_size: List[float] = Field(default_factory=lambda: [5.0, 5.0, 5.0], description="Box dimensions in nanometers")
    periodic: bool = Field(default=True, description="Whether to use periodic boundary conditions")
    ph: Optional[float] = Field(default=None, description="System pH if using constant pH")
    
    @field_validator('box_size')
    @classmethod
    def validate_box_size(cls, v: List[float]) -> List[float]:
        """
        Validate box size dimensions.
        
        Args:
            v: Box size dimensions [x, y, z]
            
        Returns:
            Validated box size dimensions
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if len(v) != 3:
            raise ValueError("Box size must have exactly 3 dimensions")
        if any(x <= 0 for x in v):
            raise ValueError("Box dimensions must be positive")
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """
        Validate temperature value.
        
        Args:
            v: Temperature in Kelvin
            
        Returns:
            Validated temperature
            
        Raises:
            ValueError: If temperature is invalid
        """
        if v <= 0:
            raise ValueError("Temperature must be greater than 0 Kelvin.")
        if v > 1000:  # Upper limit for typical simulations
            raise ValueError("Temperature seems unusually high. Please verify.")
        return v

    @field_validator('pressure')
    @classmethod
    def validate_pressure(cls, v: float) -> float:
        """
        Validate pressure value.
        
        Args:
            v: Pressure in atmospheres
            
        Returns:
            Validated pressure
            
        Raises:
            ValueError: If pressure is invalid
        """
        if v < 0:
            raise ValueError("Pressure cannot be negative.")
        if v > 1000:  # Upper limit for typical simulations
            raise ValueError("Pressure seems unusually high. Please verify.")
        return v


class IntegrationConfig(BaseModel):
    """
    Configuration for MD integration parameters.
    
    This class defines parameters related to the numerical integration
    of the equations of motion during simulation.
    
    Attributes:
        timestep: Integration timestep in femtoseconds
        hmr_enabled: Whether to use hydrogen mass repartitioning
        hmr_factor: HMR mass scaling factor
        constraint_tolerance: Constraint solver tolerance
    """
    timestep: float = Field(default=2.0, description="Integration timestep in femtoseconds")
    hmr_enabled: bool = Field(default=False, description="Whether to use hydrogen mass repartitioning")
    hmr_factor: float = Field(default=4.0, description="HMR mass scaling factor")
    constraint_tolerance: float = Field(default=1e-5, description="Constraint solver tolerance")
    
    @field_validator('timestep')
    @classmethod
    def validate_timestep(cls, v: float) -> float:
        """
        Validate timestep value.
        
        Args:
            v: Timestep in femtoseconds
            
        Returns:
            Validated timestep
            
        Raises:
            ValueError: If timestep is invalid
        """
        if v <= 0 or v > 4.0:
            raise ValueError("Timestep must be between 0 and 4 fs")
        return v


class EquilibrationConfig(BaseModel):
    """
    Configuration for system equilibration.
    
    This class defines parameters for the equilibration protocol,
    including minimization, NVT, and NPT stages.
    
    Attributes:
        minimization_steps: Number of minimization steps
        nvt_steps: Number of NVT equilibration steps
        npt_steps: Number of NPT equilibration steps
        restraint_force: Position restraint force constant in kJ/mol/nm²
    """
    minimization_steps: int = Field(default=1000, description="Number of minimization steps")
    nvt_steps: int = Field(default=10000, description="Number of NVT equilibration steps")
    npt_steps: int = Field(default=10000, description="Number of NPT equilibration steps")
    restraint_force: float = Field(default=1000.0, description="Position restraint force constant in kJ/mol/nm²")


class ProductionConfig(BaseModel):
    """
    Configuration for production MD simulation.
    
    This class defines parameters for the production simulation,
    including run length and output frequency.
    
    Attributes:
        total_steps: Total number of production steps
        checkpoint_interval: Steps between checkpoints
        trajectory_interval: Steps between trajectory frames
        energy_interval: Steps between energy reports
    """
    total_steps: int = Field(default=1000000, description="Total number of production steps")
    checkpoint_interval: int = Field(default=10000, description="Steps between checkpoints")
    trajectory_interval: int = Field(default=1000, description="Steps between trajectory frames")
    energy_interval: int = Field(default=100, description="Steps between energy reports")


class MLflowConfig(BaseModel):
    """
    MLflow tracking configuration.
    
    This class defines parameters for MLflow experiment tracking,
    allowing simulation results to be logged and compared.
    
    Attributes:
        tracking_uri: MLflow tracking server URI
        experiment_name: MLflow experiment name
        tags: Additional MLflow tags
    """
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    experiment_name: str = Field(default="md_simulations", description="MLflow experiment name")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Additional MLflow tags")


class MonitoringConfig(BaseModel):
    """
    Configuration for simulation monitoring.
    
    This class defines parameters for monitoring simulation stability,
    including thresholds for energy, density, and temperature.
    
    Attributes:
        energy_threshold: Maximum allowed energy in kJ/mol
        energy_drift_threshold: Maximum allowed energy drift in kJ/mol
        target_density: Target system density in g/cm³
        density_tolerance: Allowed density deviation from target
        density_drift_threshold: Maximum allowed density drift
        target_temperature: Target system temperature in Kelvin
        temperature_tolerance: Allowed temperature deviation from target
        temperature_drift_threshold: Maximum allowed temperature drift
        max_temperature: Maximum allowed temperature in Kelvin
        max_energy_drift: Maximum allowed energy drift per step
    """
    energy_threshold: float = Field(default=10000.0, description="Maximum allowed energy in kJ/mol")
    energy_drift_threshold: float = Field(default=100.0, description="Maximum allowed energy drift in kJ/mol")
    target_density: float = Field(default=1.0, description="Target system density in g/cm³")
    density_tolerance: float = Field(default=0.1, description="Allowed density deviation from target")
    density_drift_threshold: float = Field(default=0.01, description="Maximum allowed density drift")
    target_temperature: float = Field(default=300.0, description="Target system temperature in Kelvin")
    temperature_tolerance: float = Field(default=10.0, description="Allowed temperature deviation from target")
    temperature_drift_threshold: float = Field(default=1.0, description="Maximum allowed temperature drift")
    max_temperature: float = Field(default=1000.0, description="Maximum allowed temperature in Kelvin")
    max_energy_drift: float = Field(default=5.0, description="Maximum allowed energy drift per step")


class MDConfig(BaseModel):
    """
    Complete MD configuration.
    
    This class combines all configuration components into a single
    comprehensive configuration for molecular dynamics simulations.
    
    Attributes:
        system: System configuration
        integration: Integration configuration
        equilibration: Equilibration configuration
        production: Production configuration
        monitoring: Monitoring configuration
        mlflow: MLflow tracking configuration
        random_seed: Random seed for reproducibility
        platform: OpenMM platform to use
        device_index: GPU device index
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    equilibration: EquilibrationConfig = Field(default_factory=EquilibrationConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # Optional configurations
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    platform: str = Field(default="CUDA", description="OpenMM platform to use")
    device_index: Optional[int] = Field(default=None, description="GPU device index")

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """
        Validate the OpenMM platform.
        
        Args:
            v: The platform string.
            
        Returns:
            The validated platform string.
            
        Raises:
            ValueError: If the platform is not supported.
        """
        if v.upper() not in ["CUDA", "OPENCL", "CPU"]:
            raise ValueError(f"Unsupported platform: {v}. Supported platforms are CUDA, OPENCL, CPU.")
        return v.upper() 