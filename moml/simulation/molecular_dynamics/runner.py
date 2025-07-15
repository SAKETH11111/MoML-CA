"""
MD simulation runner.
"""
import time
import json
from pathlib import Path
from typing import Dict, Optional
import logging

from openmm import unit

from .config import MDConfig
from .builder.system_builder import SystemBuilder
from .equilibration import EquilibrationProtocol
from .monitors import EnergyMonitor, DensityMonitor, Watchdog, SimulationDiverged


class MDRunner:
    """
    Runs molecular dynamics simulations.

    This class orchestrates the simulation process, including system setup,
    equilibration, production, and monitoring.
    """

    def __init__(self, config: MDConfig, system_builder: SystemBuilder):
        self.config = config
        self.system_builder = system_builder
        self.platform = system_builder.get_platform()
        self.context = None # Will be created in run

        # Setup monitors
        self.energy_monitor = EnergyMonitor(config)
        self.density_monitor = DensityMonitor(config)
        self.watchdog = Watchdog(config)

    def run(self,
            topology: app.Topology,
            system: System,
            positions: unit.Quantity,
            output_dir: Path,
            checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Run a simulation with optional recovery from checkpoint.

        Args:
            topology: The OpenMM topology of the system.
            system: The OpenMM system to simulate.
            positions: Initial positions of the system.
            output_dir: Directory to save simulation outputs.
            checkpoint_path: Optional path to a checkpoint file to resume from.

        Returns:
            Dictionary with simulation metadata.
        """
        output_dir.mkdir(exist_ok=True)
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint(system, output_dir)

        try:
            return self._run_simulation(topology, system, positions, output_dir, checkpoint_path)
        except SimulationDiverged as e:
            logging.error(f"Simulation diverged: {e}")
            # Optionally, try to recover from an earlier checkpoint
            return {"status": "failed", "reason": str(e)}

    def _run_simulation(self,
                       topology: app.Topology,
                       system: System,
                       positions: unit.Quantity,
                       output_dir: Path,
                       checkpoint_path: Optional[Path]) -> Dict:
        """Run a single simulation attempt."""

        # Setup simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)
        simulation = app.Simulation(topology, system, integrator, self.platform)

        # Set positions and load checkpoint if available
        if checkpoint_path and checkpoint_path.exists():
            logging.info("loading_checkpoint from %s", checkpoint_path)
            simulation.loadCheckpoint(str(checkpoint_path))
        else:
            simulation.context.setPositions(positions)
            simulation.context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)

        # Setup reporters
        trajectory_path = output_dir / "trajectory.dcd"
        energy_path = output_dir / "energies.csv"

        simulation.reporters.append(app.DCDReporter(str(trajectory_path),
                                           self.config.production.trajectory_interval))
        simulation.reporters.append(app.StateDataReporter(str(energy_path),
                                              self.config.production.energy_interval,
                                              step=True,
                                              potentialEnergy=True,
                                              kineticEnergy=True,
                                              totalEnergy=True,
                                              temperature=True,
                                              volume=True,
                                              density=True))

        # Add watchdog reporter
        simulation.reporters.append(self.watchdog.as_reporter(
            system, reportInterval=self.config.production.energy_interval))

        # Add checkpoint reporter
        checkpoint_path = output_dir / "checkpoint.chk"
        simulation.reporters.append(app.CheckpointReporter(str(checkpoint_path),
                                               self.config.production.checkpoint_interval))

        # Run production
        logging.info("starting_production for %d steps",
                   self.config.production.total_steps)

        start_time = time.time()

        simulation.step(self.config.production.total_steps)

        # Save final state
        final_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        app.PDBFile.writeFile(simulation.topology,
                            final_state.getPositions(),
                            open(output_dir / "final.pdb", 'w'))

        # Save run metadata
        metadata = {
            "total_steps": self.config.production.total_steps,
            "elapsed_time": time.time() - start_time,
            "final_energy": final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            "final_temperature": 0.0,
        }
        try:
            metadata["final_density"] = final_state.getDensity().value_in_unit(unit.grams_per_milliliter)
        except Exception:
            metadata["final_density"] = "N/A"


        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info("production_complete. metadata: %s", metadata)
        return metadata

    def _find_latest_checkpoint(self, system: System, output_dir: Path) -> Optional[Path]:
        """Find the latest valid checkpoint in the output directory."""
        checkpoints = sorted(output_dir.glob("*.chk"))
        for checkpoint in reversed(checkpoints):
            if self.verify_checkpoint(system, checkpoint):
                return checkpoint
        return None

    def verify_checkpoint(self, system: System, path: Path) -> bool:
        """Verify the integrity of a checkpoint file by trying to load it."""
        if not path.exists():
            return False
        try:
            # Create a dummy context to load the checkpoint into
            integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)
            context = Context(system, integrator, self.platform)
            with open(path, 'rb') as f:
                context.loadCheckpoint(f.read())
            return True
        except Exception as e:
            logging.warning(f"Checkpoint verification failed for {path}: {e}")
            return False