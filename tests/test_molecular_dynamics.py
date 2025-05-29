"""
Tests for the molecular dynamics module.
"""

import pytest
import numpy as np
from pathlib import Path
from openmm import app, System, Context, VerletIntegrator, MonteCarloBarostat
from openmm import unit
import tempfile
import shutil

from moml.simulation.molecular_dynamics.runner import MDRunner
from moml.simulation.molecular_dynamics.equilibration import EquilibrationProtocol
from moml.simulation.molecular_dynamics.monitors import (
    EnergyMonitor, DensityMonitor, TemperatureMonitor, Watchdog,
    SimulationDiverged, BaseMonitor
)
from moml.simulation.molecular_dynamics.config import (
    MDConfig, SystemConfig, IntegrationConfig, EquilibrationConfig,
    ProductionConfig, MonitoringConfig, MLflowConfig
)
from moml.simulation.molecular_dynamics.builder.system_builder import SystemBuilder

# Test fixtures
@pytest.fixture
def md_config():
    """Create a test MD configuration."""
    return MDConfig(
        system=SystemConfig(
            temperature=300.0,
            pressure=1.0
        ),
        integration=IntegrationConfig(
            timestep=2.0
        ),
        equilibration=EquilibrationConfig(
            minimization_steps=100,
            nvt_steps=1000,
            npt_steps=1000,
            restraint_force=1000.0
        ),
        production=ProductionConfig(
            total_steps=10000,
            trajectory_interval=100,
            energy_interval=100,
            checkpoint_interval=1000
        ),
        monitoring=MonitoringConfig(
            energy_threshold=10000.0,
            energy_drift_threshold=100.0,
            target_density=1.0,
            density_tolerance=0.1,
            density_drift_threshold=0.01,
            target_temperature=300.0,
            temperature_tolerance=10.0,
            temperature_drift_threshold=1.0,
            max_temperature=1000.0,
            max_energy_drift=5.0
        ),
        mlflow=MLflowConfig(
            tracking_uri="file:./mlruns",
            experiment_name="test_md",
            tags={}
        )
    )

@pytest.fixture
def system_builder():
    """Create a test system builder."""
    return SystemBuilder()

@pytest.fixture
def test_system():
    """Create a simple test system."""
    system = System()
    system.addParticle(1.0 * unit.amu)
    system.addParticle(1.0 * unit.amu)
    return system

@pytest.fixture
def test_positions():
    """Create test positions."""
    positions = unit.Quantity(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]), unit.nanometers)
    return positions

@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

# Tests for MDRunner
def test_md_runner_initialization(md_config, system_builder):
    """Test MDRunner initialization."""
    runner = MDRunner(md_config, system_builder)
    assert runner.config == md_config
    assert runner.system_builder == system_builder
    assert isinstance(runner.energy_monitor, EnergyMonitor)
    assert isinstance(runner.density_monitor, DensityMonitor)
    assert isinstance(runner.watchdog, Watchdog)

def test_md_runner_checkpoint_verification(md_config, system_builder):
    """Test checkpoint verification."""
    runner = MDRunner(md_config, system_builder)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid checkpoint
        system = System()
        system.addParticle(1.0 * unit.amu)
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        context = Context(system, integrator)
        context.setPositions(unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometers))
        
        checkpoint_path = Path(tmpdir) / "test.chk"
        with open(checkpoint_path, 'wb') as f:
            f.write(context.createCheckpoint())
        
        assert runner.verify_checkpoint(checkpoint_path)
        
        # Test invalid checkpoint
        invalid_path = Path(tmpdir) / "invalid.chk"
        with open(invalid_path, 'wb') as f:
            f.write(b'invalid data')
        
        assert not runner.verify_checkpoint(invalid_path)
        
        # Test non-existent checkpoint
        non_existent_path = Path(tmpdir) / "nonexistent.chk"
        assert not runner.verify_checkpoint(non_existent_path)

def test_md_runner_full_simulation(md_config, system_builder, test_system, test_positions, output_dir):
    """Test full MD simulation with checkpointing and monitoring."""
    runner = MDRunner(md_config, system_builder)
    
    # Run simulation
    metadata = runner.run(test_system, test_positions, output_dir)
    
    # Verify output files
    assert (output_dir / "trajectory.dcd").exists()
    assert (output_dir / "energies.csv").exists()
    assert (output_dir / "final.pdb").exists()
    assert (output_dir / "metadata.json").exists()
    
    # Verify metadata
    assert isinstance(metadata, dict)
    assert "total_steps" in metadata
    assert "elapsed_time" in metadata
    assert "final_energy" in metadata
    assert "final_temperature" in metadata
    assert "final_density" in metadata
    
    # Verify checkpoint files
    checkpoints = list(output_dir.glob("checkpoint_*.chk"))
    assert len(checkpoints) > 0
    
    # Verify checkpoint integrity
    for checkpoint in checkpoints:
        assert runner.verify_checkpoint(checkpoint)

def test_md_runner_simulation_recovery(md_config, system_builder, test_system, test_positions, output_dir):
    """Test simulation recovery from checkpoint."""
    runner = MDRunner(md_config, system_builder)
    
    # Run initial simulation
    initial_metadata = runner.run(test_system, test_positions, output_dir)
    
    # Find latest checkpoint
    checkpoints = sorted(output_dir.glob("checkpoint_*.chk"))
    latest_checkpoint = checkpoints[-1]
    
    # Run recovery simulation
    recovery_metadata = runner.run(test_system, test_positions, output_dir, latest_checkpoint)
    
    # Verify recovery
    assert recovery_metadata["total_steps"] == initial_metadata["total_steps"]
    assert abs(recovery_metadata["final_energy"] - initial_metadata["final_energy"]) < 1e-6

# Tests for EquilibrationProtocol
def test_equilibration_protocol_initialization(md_config, system_builder):
    """Test EquilibrationProtocol initialization."""
    protocol = EquilibrationProtocol(md_config, system_builder)
    assert protocol.config == md_config
    assert protocol.system_builder == system_builder

def test_minimization(md_config, system_builder, test_system, test_positions):
    """Test energy minimization."""
    protocol = EquilibrationProtocol(md_config, system_builder)
    minimized_positions = protocol._minimize(test_system, test_positions)
    
    assert isinstance(minimized_positions, unit.Quantity)
    assert minimized_positions.shape == test_positions.shape
    
    # Verify energy decrease
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(test_system, integrator)
    context.setPositions(test_positions)
    initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
    
    context.setPositions(minimized_positions)
    final_energy = context.getState(getEnergy=True).getPotentialEnergy()
    
    assert final_energy < initial_energy

def test_nvt_equilibration(md_config, system_builder, test_system, test_positions):
    """Test NVT equilibration."""
    protocol = EquilibrationProtocol(md_config, system_builder)
    nvt_positions = protocol._nvt_equilibration(test_system, test_positions)
    
    assert isinstance(nvt_positions, unit.Quantity)
    assert nvt_positions.shape == test_positions.shape
    
    # Verify temperature control
    integrator = VerletIntegrator(md_config.integration.timestep * unit.femtoseconds)
    context = Context(test_system, integrator)
    context.setPositions(nvt_positions)
    context.setVelocitiesToTemperature(md_config.system.temperature * unit.kelvin)
    
    state = context.getState(getTemperature=True)
    temperature = state.getTemperature().value_in_unit(unit.kelvin)
    assert abs(temperature - md_config.system.temperature) < md_config.monitoring.temperature_tolerance

def test_npt_equilibration(md_config, system_builder, test_system, test_positions):
    """Test NPT equilibration."""
    protocol = EquilibrationProtocol(md_config, system_builder)
    npt_positions = protocol._npt_equilibration(test_system, test_positions)
    
    assert isinstance(npt_positions, unit.Quantity)
    assert npt_positions.shape == test_positions.shape
    
    # Verify pressure control
    npt_system = protocol._add_barostat(test_system)
    integrator = VerletIntegrator(md_config.integration.timestep * unit.femtoseconds)
    context = Context(npt_system, integrator)
    context.setPositions(npt_positions)
    context.setVelocitiesToTemperature(md_config.system.temperature * unit.kelvin)
    
    # Check for barostat
    has_barostat = any(isinstance(force, MonteCarloBarostat) for force in npt_system.getForces())
    assert has_barostat

# Tests for Monitors
def test_base_monitor():
    """Test BaseMonitor functionality."""
    monitor = BaseMonitor(window_size=5)
    
    # Test history management
    for i in range(10):
        monitor._update_history(float(i))
    
    assert len(monitor.history) == 5
    assert monitor.history == [5.0, 6.0, 7.0, 8.0, 9.0]

def test_energy_monitor(md_config):
    """Test EnergyMonitor."""
    monitor = EnergyMonitor(md_config)
    
    # Create test state
    system = System()
    system.addParticle(1.0 * unit.amu)
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator)
    context.setPositions(unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometers))
    
    # Test normal energy
    state = context.getState(getEnergy=True)
    monitor.update(state)
    assert not monitor.is_unstable()
    
    # Test high energy
    context.setPositions(unit.Quantity(np.array([[100.0, 100.0, 100.0]]), unit.nanometers))
    state = context.getState(getEnergy=True)
    monitor.update(state)
    assert monitor.is_unstable()
    
    # Test energy drift
    monitor.history = [0.0] * 1000
    monitor.history[-1] = 1000.0  # Sudden energy jump
    assert monitor.is_unstable()

def test_density_monitor(md_config):
    """Test DensityMonitor."""
    monitor = DensityMonitor(md_config)
    
    # Create test state
    system = System()
    system.addParticle(1.0 * unit.amu)
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator)
    context.setPositions(unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometers))
    
    # Test normal density
    state = context.getState(getEnergy=True, getDensity=True)
    monitor.update(state)
    assert not monitor.is_unstable()
    
    # Test unstable density
    for _ in range(1000):
        monitor._update_history(2.0)  # Far from target density
    assert monitor.is_unstable()
    
    # Test density drift
    monitor.history = [1.0] * 1000
    monitor.history[-1] = 2.0  # Sudden density change
    assert monitor.is_unstable()

def test_temperature_monitor(md_config):
    """Test TemperatureMonitor."""
    monitor = TemperatureMonitor(md_config)
    
    # Create test state
    system = System()
    system.addParticle(1.0 * unit.amu)
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator)
    context.setPositions(unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometers))
    
    # Test normal temperature
    state = context.getState(getEnergy=True, getTemperature=True)
    monitor.update(state)
    assert not monitor.is_unstable()
    
    # Test unstable temperature
    for _ in range(1000):
        monitor._update_history(1000.0)  # Far from target temperature
    assert monitor.is_unstable()
    
    # Test temperature drift
    monitor.history = [300.0] * 1000
    monitor.history[-1] = 1000.0  # Sudden temperature change
    assert monitor.is_unstable()

def test_watchdog(md_config):
    """Test Watchdog."""
    watchdog = Watchdog(md_config)
    
    # Create test state
    system = System()
    system.addParticle(1.0 * unit.amu)
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    context = Context(system, integrator)
    context.setPositions(unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometers))
    
    # Test normal state
    state = context.getState(getEnergy=True, getTemperature=True)
    watchdog._check_state(0, state)
    
    # Test high temperature
    context.setVelocitiesToTemperature(2000.0 * unit.kelvin)
    state = context.getState(getEnergy=True, getTemperature=True)
    with pytest.raises(SimulationDiverged):
        watchdog._check_state(1, state)
    
    # Test energy drift
    context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    state = context.getState(getEnergy=True, getTemperature=True)
    watchdog._check_state(1, state)
    
    # Simulate energy drift
    context.setPositions(unit.Quantity(np.array([[100.0, 100.0, 100.0]]), unit.nanometers))
    state = context.getState(getEnergy=True, getTemperature=True)
    with pytest.raises(SimulationDiverged):
        watchdog._check_state(1000, state)

def test_watchdog_reporter(md_config):
    """Test Watchdog reporter creation."""
    watchdog = Watchdog(md_config)
    reporter = watchdog.as_reporter(reportInterval=100)
    
    assert isinstance(reporter, app.StateDataReporter)
    assert reporter._reportInterval == 100
    # The callback is now set through the reporter's constructor
    assert hasattr(reporter, '_callback') 