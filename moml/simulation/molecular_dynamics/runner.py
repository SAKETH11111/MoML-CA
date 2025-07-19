"""
moml/simulation/molecular_dynamics/runner.py

MD simulation runner for molecular dynamics simulations.

This module provides the core functionality for running molecular dynamics
simulations using OpenMM. It handles simulation setup, checkpoint management,
trajectory and energy output, and monitoring for simulation stability.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from openmm import Context, System, VerletIntegrator, unit
from openmm import app

from .builder.system_builder import SystemBuilder
from .config import MDConfig
from .equilibration import EquilibrationProtocol
from .monitors import DensityMonitor, EnergyMonitor, SimulationDiverged, Watchdog


class MDRunner:
    """
    Runs molecular dynamics simulations.

    This class orchestrates the simulation process, including system setup,
    equilibration, production, and monitoring. It supports checkpoint-based
    recovery for interrupted simulations and provides comprehensive output
    for analysis.
    
    Attributes:
        config: Configuration parameters for the simulation
        system_builder: Builder for creating OpenMM systems
        platform: OpenMM platform to use for calculations
        energy_monitor: Monitor for system energy stability
        density_monitor: Monitor for system density stability
        watchdog: Monitor for detecting simulation divergence
    """

    def __init__(self, config: MDConfig, system_builder: SystemBuilder) -> None:
        """
        Initialize the MD runner.
        
        Args:
            config: Configuration parameters for the simulation
            system_builder: Builder for creating OpenMM systems
        """
        self.config = config
        self.system_builder = system_builder
        self.platform = system_builder.get_platform()
        self.context = None  # Will be created in run

        # Setup monitors
        self.energy_monitor = EnergyMonitor(config)
        # density_monitor will be initialized in run() when system is available
        self.density_monitor = None
        self.watchdog = Watchdog(config)

    def run(
        self,
        topology: app.Topology,
        system: System,
        positions: unit.Quantity,
        output_dir: Path,
        checkpoint_path: Optional[Path] = None
    ) -> Dict:
        """
        Run a simulation with optional recovery from checkpoint.

        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            checkpoint_path: Optional path to a checkpoint file to resume from

        Returns:
            Dictionary with simulation metadata and results
            
        Raises:
            ValueError: If the simulation parameters are invalid
            SimulationDiverged: If the simulation becomes unstable
        """
        output_dir.mkdir(exist_ok=True)
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint(system, output_dir)

        try:
            return self._run_simulation(topology, system, positions, output_dir, checkpoint_path)
        except SimulationDiverged as e:
            logging.error("Simulation diverged: %s", str(e))
            return {"status": "failed", "reason": str(e)}

    def _run_simulation(
        self,
        topology: app.Topology,
        system: System,
        positions: unit.Quantity,
        output_dir: Path,
        checkpoint_path: Optional[Path]
    ) -> Dict:
        """
        Run a single simulation attempt.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            checkpoint_path: Optional path to a checkpoint file to resume from
            
        Returns:
            Dictionary with simulation metadata and results
        """
        # Initialize density monitor now that system is available
        if self.density_monitor is None:
            self.density_monitor = DensityMonitor(self.config, system)

        # Setup simulation
        integrator = VerletIntegrator(
            self.config.integration.timestep * unit.femtoseconds  # type: ignore
        )
        simulation = app.Simulation(topology, system, integrator, self.platform)

        # Set positions and load checkpoint if available
        if checkpoint_path and checkpoint_path.exists():
            logging.info("Loading checkpoint from %s", checkpoint_path)
            simulation.loadCheckpoint(str(checkpoint_path))
        else:
            simulation.context.setPositions(positions)
            simulation.context.setVelocitiesToTemperature(
                self.config.system.temperature * unit.kelvin  # type: ignore
            )

        # Setup reporters
        trajectory_path = output_dir / "trajectory.dcd"
        energy_path = output_dir / "energies.csv"

        simulation.reporters.append(
            app.DCDReporter(
                str(trajectory_path),
                self.config.production.trajectory_interval
            )
        )
        simulation.reporters.append(
            app.StateDataReporter(
                str(energy_path),
                self.config.production.energy_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True
            )
        )

        # Add watchdog reporter
        simulation.reporters.append(
            self.watchdog.as_reporter(
                system, 
                reportInterval=self.config.production.energy_interval
            )
        )

        # Add checkpoint reporter
        checkpoint_path = output_dir / "checkpoint.chk"
        simulation.reporters.append(
            app.CheckpointReporter(
                str(checkpoint_path),
                self.config.production.checkpoint_interval
            )
        )

        # Run production
        logging.info(
            "Starting production simulation for %d steps",
            self.config.production.total_steps
        )

        start_time = time.time()
        simulation.step(self.config.production.total_steps)

        # Save final state
        final_state = simulation.context.getState(
            getPositions=True, 
            getVelocities=True, 
            getEnergy=True
        )
        
        with open(output_dir / "final.pdb", 'w') as f:
            app.PDBFile.writeFile(
                simulation.topology,
                final_state.getPositions(),
                f
            )

        # Save run metadata
        metadata = {
            "total_steps": self.config.production.total_steps,
            "elapsed_time": time.time() - start_time,
            "final_energy": final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),  # type: ignore
            "final_temperature": (
                2 * final_state.getKineticEnergy() / 
                (system.getNumParticles() * 3 - system.getNumConstraints()) / 
                (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
            ).value_in_unit(unit.kelvin),  # type: ignore
        }
        
        try:
            metadata["final_density"] = final_state.getDensity().value_in_unit(unit.gram / unit.milliliter)  # type: ignore
        except Exception:
            metadata["final_density"] = "N/A"

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info("Production complete. Metadata: %s", metadata)
        return metadata

    def _find_latest_checkpoint(self, system: System, output_dir: Path) -> Optional[Path]:
        """
        Find the latest valid checkpoint in the output directory.
        
        Args:
            system: The OpenMM system to verify checkpoints against
            output_dir: Directory containing checkpoint files
            
        Returns:
            Path to the latest valid checkpoint, or None if no valid checkpoint exists
        """
        checkpoints = sorted(output_dir.glob("*.chk"))
        for checkpoint in reversed(checkpoints):
            if self.verify_checkpoint(system, checkpoint):
                return checkpoint
        return None

    def verify_checkpoint(self, system: System, path: Path) -> bool:
        """
        Verify the integrity of a checkpoint file by trying to load it.
        
        Args:
            system: The OpenMM system to verify checkpoints against
            path: Path to the checkpoint file
            
        Returns:
            True if the checkpoint is valid, False otherwise
        """
        if not path.exists():
            return False
            
        try:
            # Create a dummy context to load the checkpoint into
            integrator = VerletIntegrator(
                self.config.integration.timestep * unit.femtoseconds  # type: ignore
            )
            context = Context(system, integrator, self.platform)
            with open(path, 'rb') as f:
                context.loadCheckpoint(f.read())
            return True
        except Exception as e:
            logging.warning("Checkpoint verification failed for %s: %s", path, str(e))
            return False