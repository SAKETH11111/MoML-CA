"""
tests/test_molecular_dynamics.py

Unit tests for the molecular dynamics simulation module, including MDRunner,
EquilibrationProtocol, and various monitoring components.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
from openmm import Context, NonbondedForce, Platform, System, VerletIntegrator, app, unit
from openmm import MonteCarloBarostat # Import MonteCarloBarostat explicitly

from moml.simulation.molecular_dynamics.config import (
    MDConfig,
    MLflowConfig,
    MonitoringConfig,
    ProductionConfig,
    SystemConfig,
)
from moml.simulation.molecular_dynamics.equilibration import EquilibrationProtocol
from moml.simulation.molecular_dynamics.monitors import (
    BaseMonitor,
    DensityMonitor,
    EnergyMonitor,
    SimulationDiverged,
    TemperatureMonitor,
    Watchdog,
)
from moml.simulation.molecular_dynamics.runner import MDRunner
from moml.simulation.molecular_dynamics.builder.system_builder import SystemBuilder
from moml.simulation.molecular_dynamics.config import IntegrationConfig, EquilibrationConfig


@pytest.fixture
def md_config() -> MDConfig:
    """
    Create a test MD configuration.

    Returns:
        MDConfig: A configured MDConfig object for testing.
    """
    return MDConfig(
        platform="CPU",
        system=SystemConfig(temperature=300.0, pressure=1.0),
        integration=IntegrationConfig(timestep=2.0),
        equilibration=EquilibrationConfig(
            minimization_steps=100, nvt_steps=1000, npt_steps=100, restraint_force=1000.0
        ),
        production=ProductionConfig(total_steps=100, trajectory_interval=10, energy_interval=10, checkpoint_interval=50),
        monitoring=MonitoringConfig(
            energy_threshold=10000.0,
            energy_drift_threshold=100.0,
            target_density=1.0,
            density_tolerance=0.1,
            density_drift_threshold=0.01,
            target_temperature=300.0,
            temperature_tolerance=150.0,
            temperature_drift_threshold=10.0,
            max_temperature=1000.0,
            max_energy_drift=500.0,
        ),
        mlflow=MLflowConfig(tracking_uri="file:./mlruns", experiment_name="test_md", tags={}),
    )


@pytest.fixture
def system_builder(md_config: MDConfig) -> SystemBuilder:
    """
    Create a SystemBuilder instance with the test config.

    Args:
        md_config (MDConfig): The MD configuration fixture.

    Returns:
        SystemBuilder: A configured SystemBuilder instance.
    """
    return SystemBuilder(config=md_config)


@pytest.fixture
def test_system() -> System:
    """
    Create a simple, non-periodic test system with two particles and a nonbonded force.

    Returns:
        System: An OpenMM System object.
    """
    system = System()
    system.addParticle(1.0 * unit.amu) # type: ignore
    system.addParticle(1.0 * unit.amu) # type: ignore
    force = NonbondedForce()
    force.addParticle(0.0, 1.0, 0.0)
    force.addParticle(0.0, 1.0, 0.0)
    system.addForce(force)
    return system


@pytest.fixture
def test_positions() -> unit.Quantity:
    """
    Create test positions for two particles.

    Returns:
        unit.Quantity: OpenMM Quantity object representing particle positions.
    """
    positions = unit.Quantity(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), unit.nanometers) # type: ignore
    return positions


@pytest.fixture
def output_dir() -> Generator[Path, None, None]:
    """
    Create a temporary output directory for simulation results.

    Yields:
        Path: Path object for the temporary directory.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestMDRunner:
    """
    Test suite for the MDRunner class.
    """

    def test_md_runner_initialization(self, md_config: MDConfig, system_builder: SystemBuilder) -> None:
        """
        Test MDRunner initialization.
        """
        runner = MDRunner(md_config, system_builder)
        assert runner.config == md_config
        assert runner.system_builder == system_builder

    def test_md_runner_checkpoint_verification(self, md_config: MDConfig, system_builder: SystemBuilder) -> None:
        """
        Test checkpoint verification functionality.
        """
        runner = MDRunner(md_config, system_builder)
        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            platform = Platform.getPlatformByName("CPU")
            integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
            context = Context(system, integrator, platform)
            context.setPositions(unit.Quantity(np.zeros((1, 3)), unit.nanometers)) # type: ignore

            checkpoint_path = tmpdir_path / "test.chk"
            with open(checkpoint_path, "wb") as f:
                f.write(context.createCheckpoint())

            assert runner.verify_checkpoint(system, checkpoint_path)

            invalid_path = tmpdir_path / "invalid.chk"
            with open(invalid_path, "wb") as f:
                f.write(b"this is not a checkpoint")

            assert not runner.verify_checkpoint(system, invalid_path)

    def test_md_runner_full_simulation(
        self, md_config: MDConfig, system_builder: SystemBuilder, test_system: System, test_positions: unit.Quantity, output_dir: Path
    ) -> None:
        """
        Test a full MD simulation run.
        """
        runner = MDRunner(md_config, system_builder)

        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("HOH", chain)
        topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
        topology.addAtom("H2", app.Element.getBySymbol("H"), residue)

        vecs = unit.Quantity(np.eye(3) * 5, unit.nanometers) # type: ignore
        test_system.setDefaultPeriodicBoxVectors(*vecs)

        metadata = runner.run(topology, test_system, test_positions, output_dir)

        assert (output_dir / "trajectory.dcd").exists()
        assert (output_dir / "energies.csv").exists()
        assert (output_dir / "final.pdb").exists()
        assert (output_dir / "metadata.json").exists()
        assert isinstance(metadata, dict)

    def test_md_runner_simulation_recovery(
        self, md_config: MDConfig, system_builder: SystemBuilder, test_system: System, test_positions: unit.Quantity, output_dir: Path
    ) -> None:
        """
        Test simulation recovery from a checkpoint.
        """
        runner = MDRunner(md_config, system_builder)

        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("HOH", chain)
        topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
        topology.addAtom("H2", app.Element.getBySymbol("H"), residue)

        vecs = unit.Quantity(np.eye(3) * 5, unit.nanometers) # type: ignore
        test_system.setDefaultPeriodicBoxVectors(*vecs)

        runner.run(topology, test_system, test_positions, output_dir)

        checkpoints = sorted(output_dir.glob("*.chk"))
        assert len(checkpoints) > 0
        latest_checkpoint = checkpoints[-1]

        runner.run(topology, test_system, test_positions, output_dir, latest_checkpoint)


class TestEquilibrationProtocol:
    """
    Test suite for the EquilibrationProtocol class.
    """

    def test_minimization(
        self, md_config: MDConfig, system_builder: SystemBuilder, test_system: System, test_positions: unit.Quantity
    ) -> None:
        """
        Test energy minimization step.
        """
        protocol = EquilibrationProtocol(md_config, system_builder)

        integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
        context = Context(test_system, integrator)
        context.setPositions(test_positions)
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

        minimized_positions = protocol._minimize(test_system, test_positions)

        context.setPositions(minimized_positions)
        final_energy = context.getState(getEnergy=True).getPotentialEnergy()

        assert final_energy <= initial_energy

    def test_nvt_equilibration(
        self, md_config: MDConfig, system_builder: SystemBuilder, test_system: System, test_positions: unit.Quantity
    ) -> None:
        """
        Test NVT (constant Number, Volume, Temperature) equilibration.
        """
        protocol = EquilibrationProtocol(md_config, system_builder)
        nvt_positions = protocol._nvt_equilibration(test_system, test_positions)

        integrator = VerletIntegrator(md_config.integration.timestep * unit.femtoseconds) # type: ignore
        context = Context(test_system, integrator)
        context.setPositions(nvt_positions)
        context.setVelocitiesToTemperature(md_config.system.temperature * unit.kelvin) # type: ignore

        state = context.getState(getEnergy=True)
        ke = state.getKineticEnergy()
        n_dof = 3 * test_system.getNumParticles() - test_system.getNumConstraints()
        temperature = (2 * ke / (n_dof * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)).value_in_unit( # type: ignore
            unit.kelvin # type: ignore
        )
        assert abs(temperature - md_config.system.temperature) < 300.0

    def test_npt_equilibration(
        self, md_config: MDConfig, system_builder: SystemBuilder, test_system: System, test_positions: unit.Quantity
    ) -> None:
        """
        Test NPT (constant Number, Pressure, Temperature) equilibration.
        """
        vecs = unit.Quantity(np.eye(3) * 5, unit.nanometers) # type: ignore
        test_system.setDefaultPeriodicBoxVectors(*vecs)
        test_system.getForce(0).setNonbondedMethod(NonbondedForce.PME)

        protocol = EquilibrationProtocol(md_config, system_builder)
        protocol._npt_equilibration(test_system, test_positions)

        has_barostat = any(isinstance(force, MonteCarloBarostat) for force in test_system.getForces())
        assert has_barostat


class TestMonitors:
    """
    Test suite for various simulation monitoring classes.
    """

    def test_energy_monitor(self, md_config: MDConfig) -> None:
        """
        Test EnergyMonitor functionality.
        """
        monitor = EnergyMonitor(md_config)

        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore
        integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
        context = Context(system, integrator)
        context.setPositions(unit.Quantity(np.zeros((1, 3)), unit.nanometers)) # type: ignore

        state = context.getState(getEnergy=True, getPositions=True)
        monitor.update(state)
        assert not monitor.is_unstable()

    def test_density_monitor(self, md_config: MDConfig) -> None:
        """
        Test DensityMonitor functionality.
        """
        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore
        vecs = unit.Quantity(np.eye(3) * 2, unit.nanometers) # type: ignore
        system.setDefaultPeriodicBoxVectors(*vecs)

        monitor = DensityMonitor(md_config, system)

        integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
        context = Context(system, integrator)
        context.setPositions(unit.Quantity(np.zeros((1, 3)), unit.nanometers)) # type: ignore

        state = context.getState(getPositions=True)
        volume_nm3 = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) # type: ignore
        mass_daltons = system.getParticleMass(0).value_in_unit(unit.dalton) # type: ignore
        density = (mass_daltons / volume_nm3) * 1.66054
        monitor._update_history(density)
        assert not monitor.is_unstable()

    def test_temperature_monitor(self, md_config: MDConfig) -> None:
        """
        Test TemperatureMonitor functionality.
        """
        monitor = TemperatureMonitor(md_config)

        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore
        integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
        context = Context(system, integrator)
        context.setPositions(unit.Quantity(np.zeros((1, 3)), unit.nanometers)) # type: ignore

        state = context.getState(getEnergy=True)
        monitor.update(state)
        assert not monitor.is_unstable()

    def test_watchdog(self, md_config: MDConfig) -> None:
        """
        Test Watchdog functionality for detecting simulation divergence.
        """
        watchdog = Watchdog(md_config)
        watchdog.last_energy = None
        watchdog.last_step = None

        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore
        integrator = VerletIntegrator(0.001 * unit.picoseconds) # type: ignore
        context = Context(system, integrator)
        context.setPositions(unit.Quantity(np.zeros((1, 3)), unit.nanometers)) # type: ignore

        state = context.getState(getEnergy=True, getVelocities=True)
        watchdog._check_state(0, state, system)

        context.setVelocitiesToTemperature(5000.0 * unit.kelvin) # type: ignore
        integrator.step(1)
        state = context.getState(getEnergy=True, getVelocities=True)
        try:
            watchdog._check_state(1, state, system) # type: ignore
            pytest.fail("Expected SimulationDiverged exception but none was raised")
        except SimulationDiverged as e:
            assert "Temperature" in str(e), f"Expected temperature error but got: {e}"

        context.setVelocitiesToTemperature(300.0 * unit.kelvin) # type: ignore
        integrator.step(1)
        state = context.getState(getEnergy=True, getVelocities=True)
        watchdog._check_state(2, state, system) # type: ignore

        watchdog.last_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) # type: ignore
        watchdog.last_step = 2

        class MockState:
            """A mock OpenMM State object for testing."""

            def __init__(self, energy_val: unit.Quantity, ke_val: unit.Quantity) -> None: # type: ignore
                self._energy = energy_val
                self._ke = ke_val

            def getPotentialEnergy(self) -> unit.Quantity: # type: ignore
                """Returns mock potential energy."""
                return self._energy

            def getKineticEnergy(self) -> unit.Quantity: # type: ignore
                """Returns mock kinetic energy."""
                return self._ke

        ke = state.getKineticEnergy()
        drift_energy = watchdog.last_energy + md_config.monitoring.max_energy_drift * 2 # type: ignore

        with pytest.raises(SimulationDiverged, match="Energy drift"):
            watchdog._check_state(3, MockState(drift_energy * unit.kilojoules_per_mole, ke), system) # type: ignore

    def test_watchdog_reporter(self, md_config: MDConfig) -> None:
        """
        Test Watchdog reporter creation.
        """
        watchdog = Watchdog(md_config)
        system = System()
        system.addParticle(1.0 * unit.amu) # type: ignore
        reporter = watchdog.as_reporter(system, reportInterval=100)

        assert isinstance(reporter, app.StateDataReporter)
        assert reporter._reportInterval == 100
        assert hasattr(reporter, "_callback")
