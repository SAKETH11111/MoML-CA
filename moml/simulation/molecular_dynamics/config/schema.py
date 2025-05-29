"""
Configuration schema for Molecular Dynamics simulations.
Uses Pydantic for type validation and default values.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

class SystemConfig(BaseModel):
    """Configuration for system setup"""
    temperature: float = Field(300.0, description="System temperature in Kelvin")
    pressure: float = Field(1.0, description="System pressure in atmospheres")
    box_size: List[float] = Field([5.0, 5.0, 5.0], description="Box dimensions in nanometers")
    periodic: bool = Field(True, description="Whether to use periodic boundary conditions")
    ph: Optional[float] = Field(default=None, description="System pH if using constant pH")
    
    @field_validator('box_size')
    @classmethod
    def validate_box_size(cls, v):
        if len(v) != 3:
            raise ValueError("Box size must have exactly 3 dimensions")
        if any(x <= 0 for x in v):
            raise ValueError("Box dimensions must be positive")
        return v

class IntegrationConfig(BaseModel):
    """Configuration for MD integration"""
    timestep: float = Field(2.0, description="Integration timestep in femtoseconds")
    hmr_enabled: bool = Field(False, description="Whether to use hydrogen mass repartitioning")
    hmr_factor: float = Field(4.0, description="HMR mass scaling factor")
    constraint_tolerance: float = Field(1e-5, description="Constraint solver tolerance")
    
    @field_validator('timestep')
    @classmethod
    def validate_timestep(cls, v):
        if v <= 0 or v > 4.0:
            raise ValueError("Timestep must be between 0 and 4 fs")
        return v

class EquilibrationConfig(BaseModel):
    """Configuration for system equilibration"""
    minimization_steps: int = Field(1000, description="Number of minimization steps")
    nvt_steps: int = Field(10000, description="Number of NVT equilibration steps")
    npt_steps: int = Field(10000, description="Number of NPT equilibration steps")
    restraint_force: float = Field(1000.0, description="Position restraint force constant in kJ/mol/nm²")

class ProductionConfig(BaseModel):
    """Configuration for production MD"""
    total_steps: int = Field(1000000, description="Total number of production steps")
    checkpoint_interval: int = Field(10000, description="Steps between checkpoints")
    trajectory_interval: int = Field(1000, description="Steps between trajectory frames")
    energy_interval: int = Field(100, description="Steps between energy reports")

class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    experiment_name: str = Field(default="md_simulations", description="MLflow experiment name")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Additional MLflow tags")

class MonitoringConfig(BaseModel):
    """Configuration for simulation monitoring"""
    energy_threshold: float = Field(10000.0, description="Maximum allowed energy in kJ/mol")
    energy_drift_threshold: float = Field(100.0, description="Maximum allowed energy drift in kJ/mol")
    target_density: float = Field(1.0, description="Target system density in g/cm³")
    density_tolerance: float = Field(0.1, description="Allowed density deviation from target")
    density_drift_threshold: float = Field(0.01, description="Maximum allowed density drift")
    target_temperature: float = Field(300.0, description="Target system temperature in Kelvin")
    temperature_tolerance: float = Field(10.0, description="Allowed temperature deviation from target")
    temperature_drift_threshold: float = Field(1.0, description="Maximum allowed temperature drift")
    max_temperature: float = Field(1000.0, description="Maximum allowed temperature in Kelvin")
    max_energy_drift: float = Field(5.0, description="Maximum allowed energy drift per step")

class MDConfig(BaseModel):
    """Complete MD configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    equilibration: EquilibrationConfig = Field(default_factory=EquilibrationConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # Optional configurations
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    platform: str = Field("CUDA", description="OpenMM platform to use")
    device_index: Optional[int] = Field(None, description="GPU device index") 