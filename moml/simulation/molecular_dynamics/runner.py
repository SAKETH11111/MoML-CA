"""
Production MD runner with checkpointing and monitoring.
"""

from pathlib import Path
from typing import Optional, Dict
import time
import json
import hashlib
import structlog
import mlflow
from openmm import app, System, Context, VerletIntegrator
from openmm import unit

from .config.schema import MDConfig
from .builder.system_builder import SystemBuilder
from .equilibration import EquilibrationProtocol
from .monitors import EnergyMonitor, DensityMonitor, Watchdog, SimulationDiverged

logger = structlog.get_logger()

def run_job(config: MDConfig, system_builder: SystemBuilder, output_dir: Path) -> Dict:
    """
    Run a complete MD job including system building, equilibration and production.
    
    Args:
        config: MD configuration
        system_builder: System builder instance
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing run metadata
    """
    # Create runner
    runner = MDRunner(config, system_builder)
    
    # Build system
    system, positions = system_builder.build()
    
    # Run equilibration if needed
    if config.equilibration.enabled:
        equilibration = EquilibrationProtocol(config.equilibration)
        system, positions = equilibration.run(system, positions, output_dir / "equilibration")
    
    # Run production
    return runner.run(system, positions, output_dir / "production")

class MDRunner:
    """Handles production MD with checkpointing and monitoring."""
    
    def __init__(self, config: MDConfig, system_builder: SystemBuilder):
        self.config = config
        self.system_builder = system_builder
        self.platform = system_builder.get_platform()
        
        # Setup monitors
        self.energy_monitor = EnergyMonitor()
        self.density_monitor = DensityMonitor()
        self.watchdog = Watchdog(temp_max=1000, energy_drift_kj_per_ns=5)
        
        # MLflow tracking
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
    
    def run(self, 
            system: System,
            positions: unit.Quantity,
            output_dir: Path,
            checkpoint_path: Optional[Path] = None,
            max_retries: int = 3) -> Dict:
        """Run production MD with checkpointing and automatic recovery."""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"md_run_{int(time.time())}"):
            # Log parameters
            mlflow.log_params(self.config.dict())
            mlflow.set_tags(self.config.mlflow.tags)
            
            retry_count = 0
            last_checkpoint = checkpoint_path
            
            while retry_count < max_retries:
                try:
                    return self._run_simulation(system, positions, output_dir, last_checkpoint)
                except SimulationDiverged as e:
                    retry_count += 1
                    logger.warning("simulation_diverged_retry",
                                 error=str(e),
                                 retry_count=retry_count,
                                 max_retries=max_retries)
                    
                    if retry_count >= max_retries:
                        logger.error("max_retries_exceeded")
                        raise
                    
                    # Find latest valid checkpoint
                    last_checkpoint = self._find_latest_checkpoint(output_dir)
                    if last_checkpoint is None:
                        logger.error("no_valid_checkpoint_found")
                        raise
                    
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
    
    def _run_simulation(self,
                       system: System,
                       positions: unit.Quantity,
                       output_dir: Path,
                       checkpoint_path: Optional[Path]) -> Dict:
        """Run a single simulation attempt."""
        
        # Setup simulation
        integrator = VerletIntegrator(self.config.integration.timestep * unit.femtoseconds)
        context = Context(system, integrator, self.platform)
        
        # Load checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            logger.info("loading_checkpoint", path=str(checkpoint_path))
            with open(checkpoint_path, 'rb') as f:
                context.loadCheckpoint(f.read())
        else:
            context.setPositions(positions)
            context.setVelocitiesToTemperature(self.config.system.temperature * unit.kelvin)
        
        # Setup reporters
        trajectory_path = output_dir / "trajectory.dcd"
        energy_path = output_dir / "energies.csv"
        
        trajectory_reporter = app.DCDReporter(str(trajectory_path), 
                                            self.config.production.trajectory_interval)
        energy_reporter = app.StateDataReporter(str(energy_path),
                                              self.config.production.energy_interval,
                                              step=True,
                                              potentialEnergy=True,
                                              kineticEnergy=True,
                                              totalEnergy=True,
                                              temperature=True,
                                              volume=True,
                                              density=True)
        
        # Add watchdog reporter
        context.addReporter(self.watchdog.as_reporter(
            reportInterval=self.config.production.energy_interval))
        
        # Run production
        logger.info("starting_production", 
                   total_steps=self.config.production.total_steps)
        
        start_time = time.time()
        
        for i in range(0, self.config.production.total_steps, 
                      self.config.production.checkpoint_interval):
            # Run chunk
            integrator.step(self.config.production.checkpoint_interval)
            
            # Get state
            state = context.getState(getEnergy=True, getPositions=True)
            
            # Update monitors
            self.energy_monitor.update(state)
            self.density_monitor.update(state)
            
            # Check for instabilities
            if self.energy_monitor.is_unstable():
                logger.error("energy_instability_detected")
                raise SimulationDiverged("Energy instability detected")
            
            if self.density_monitor.is_unstable():
                logger.error("density_instability_detected")
                raise SimulationDiverged("Density instability detected")
            
            # Save checkpoint
            checkpoint_path = output_dir / f"checkpoint_{i}.chk"
            with open(checkpoint_path, 'wb') as f:
                f.write(context.createCheckpoint())
            
            # Log progress
            elapsed = time.time() - start_time
            logger.info("production_progress",
                      step=i,
                      elapsed=elapsed,
                      energy=state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
            
            # Log to MLflow
            mlflow.log_metrics({
                "potential_energy": state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                "kinetic_energy": state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
                "temperature": state.getTemperature().value_in_unit(unit.kelvin),
                "density": state.getDensity().value_in_unit(unit.grams_per_milliliter)
            }, step=i)
        
        # Save final state
        final_state = context.getState(getPositions=True, getVelocities=True)
        app.PDBFile.writeFile(system.getTopology(), 
                            final_state.getPositions(),
                            open(output_dir / "final.pdb", 'w'))
        
        # Save run metadata
        metadata = {
            "total_steps": self.config.production.total_steps,
            "elapsed_time": time.time() - start_time,
            "final_energy": final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            "final_temperature": final_state.getTemperature().value_in_unit(unit.kelvin),
            "final_density": final_state.getDensity().value_in_unit(unit.grams_per_milliliter)
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("production_complete", **metadata)
        return metadata
    
    def _find_latest_checkpoint(self, output_dir: Path) -> Optional[Path]:
        """Find the latest valid checkpoint in the output directory."""
        checkpoints = sorted(output_dir.glob("checkpoint_*.chk"))
        for checkpoint in reversed(checkpoints):
            if self.verify_checkpoint(checkpoint):
                return checkpoint
        return None
    
    def verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint integrity."""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = f.read()
                # Verify checksum
                checksum = hashlib.sha256(data).hexdigest()
                return len(data) > 0 and checksum is not None
        except Exception as e:
            logger.error("checkpoint_verification_failed", error=str(e))
            return False 