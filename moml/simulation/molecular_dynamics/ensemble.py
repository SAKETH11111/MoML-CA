"""
moml/simulation/molecular_dynamics/ensemble.py

Ensemble module for managing multiple MD replicas and parameter sweeps.

This module provides functionality for running ensemble simulations, which are
collections of related simulations that differ in parameters or initial conditions.
Supported ensemble types include:
- NPT ensemble with multiple replicas for statistical sampling
- Constant pH ensemble for studying pH-dependent properties

The module includes resource management to prevent overloading the system
when running multiple simulations in parallel.
"""

from pathlib import Path
from typing import Dict, List, Optional

import concurrent.futures
import mlflow
import psutil
import structlog
import time
from openmm import System, unit
from openmm import app

from .builder.system_builder import SystemBuilder
from .config.schema import MDConfig
from .equilibration import EquilibrationProtocol
from .runner import MDRunner

# Configure logger
logger = structlog.get_logger()


class ResourceLimits:
    """
    Resource limits for parallel execution.
    
    This class manages system resource limits to prevent overloading
    the system when running multiple simulations in parallel.
    
    Attributes:
        max_cpu_percent: Maximum CPU utilization percentage
        max_memory_percent: Maximum memory utilization percentage
        max_workers: Maximum number of parallel workers
    """

    def __init__(
        self, 
        max_cpu_percent: float = 80.0, 
        max_memory_percent: float = 80.0, 
        max_workers: Optional[int] = None
    ) -> None:
        """
        Initialize resource limits.
        
        Args:
            max_cpu_percent: Maximum CPU utilization percentage (0-100)
            max_memory_percent: Maximum memory utilization percentage (0-100)
            max_workers: Maximum number of parallel workers (defaults to CPU count)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_workers = max_workers or psutil.cpu_count()

    def get_available_workers(self) -> int:
        """
        Get number of available workers based on current system resources.
        
        Returns:
            Number of workers that can be safely used without overloading the system
        """
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Ensure max_workers is always an int, handle case where psutil.cpu_count() returns None
        if self.max_workers is not None:
            max_workers = self.max_workers
        else:
            cpu_count = psutil.cpu_count()
            max_workers = cpu_count if cpu_count is not None else 1

        # If system is already heavily loaded, reduce worker count
        if cpu_percent > self.max_cpu_percent or memory_percent > self.max_memory_percent:
            return max(1, max_workers // 2)
            
        return max_workers


class EnsembleStrategy:
    """
    Base class for ensemble simulation strategies.
    
    This abstract class defines the interface for all ensemble simulation
    strategies. Concrete subclasses implement specific ensemble types.
    
    Attributes:
        config: Configuration parameters for the simulations
        system_builder: Builder for creating OpenMM systems
        resource_limits: Resource limits for parallel execution
    """

    def __init__(
        self, 
        config: MDConfig, 
        system_builder: SystemBuilder, 
        resource_limits: Optional[ResourceLimits] = None
    ) -> None:
        """
        Initialize the ensemble strategy.
        
        Args:
            config: Configuration parameters for the simulations
            system_builder: Builder for creating OpenMM systems
            resource_limits: Resource limits for parallel execution
        """
        self.config = config
        self.system_builder = system_builder
        self.resource_limits = resource_limits or ResourceLimits()

    def run(
        self, 
        topology: app.Topology, 
        system: System, 
        positions: unit.Quantity, 
        output_dir: Path
    ) -> Dict:
        """
        Run ensemble simulation.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            
        Returns:
            Dictionary with ensemble results
        """
        raise NotImplementedError


class NPTEnsemble(EnsembleStrategy):
    """
    NPT ensemble with multiple replicas.
    
    This ensemble runs multiple identical simulations with different random
    seeds to improve statistical sampling of thermodynamic properties.
    
    Attributes:
        num_replicas: Number of replica simulations to run
    """

    def __init__(
        self,
        config: MDConfig,
        system_builder: SystemBuilder,
        num_replicas: int = 4,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> None:
        """
        Initialize the NPT ensemble.
        
        Args:
            config: Configuration parameters for the simulations
            system_builder: Builder for creating OpenMM systems
            num_replicas: Number of replica simulations to run
            resource_limits: Resource limits for parallel execution
        """
        super().__init__(config, system_builder, resource_limits)
        self.num_replicas = num_replicas

    def run(
        self, 
        topology: app.Topology, 
        system: System, 
        positions: unit.Quantity, 
        output_dir: Path
    ) -> Dict:
        """
        Run NPT ensemble simulation with multiple replicas.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            
        Returns:
            Dictionary with ensemble results including individual replica results
        """
        logger.info("starting_npt_ensemble", num_replicas=self.num_replicas)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        ensemble_results: Dict = {}
        # Start MLflow run
        with mlflow.start_run(run_name=f"npt_ensemble_{int(time.time())}"):
            # Log parameters
            mlflow.log_params({**self.config.model_dump(), "num_replicas": self.num_replicas})

            # Get available workers
            max_workers = min(self.num_replicas, self.resource_limits.get_available_workers())
            logger.info("resource_allocation", max_workers=max_workers)

            # Run replicas in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(self.num_replicas):
                    replica_dir = output_dir / f"replica_{i}"
                    futures.append(
                        executor.submit(
                            self._run_replica, 
                            topology, 
                            system, 
                            positions, 
                            replica_dir, 
                            i
                        )
                    )

                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error("replica_failed", error=str(e))

            # Aggregate results
            ensemble_results = {
                "num_replicas": self.num_replicas,
                "completed_replicas": len(results),
                "replica_results": results,
            }

            # Log ensemble metrics
            failed_count = self.num_replicas - len(results)
            mlflow.log_metrics({
                "completed_replicas": len(results),
                "failed_replicas": failed_count,
                "success_rate": len(results) / self.num_replicas
            })

            logger.info("ensemble_complete", **ensemble_results)

        return ensemble_results

    def _run_replica(
        self, 
        topology: app.Topology, 
        system: System, 
        positions: unit.Quantity, 
        output_dir: Path, 
        replica_id: int
    ) -> Dict:
        """
        Run a single replica simulation.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            replica_id: Unique identifier for this replica
            
        Returns:
            Dictionary with simulation results
        """
        logger.info("starting_replica", replica_id=replica_id)

        # Create replica-specific config with unique random seed
        replica_config = self.config.model_copy()
        if self.config.random_seed is not None:
            replica_config.random_seed = self.config.random_seed + replica_id

        # Setup and run
        equilibration = EquilibrationProtocol(replica_config, self.system_builder)
        runner = MDRunner(replica_config, self.system_builder)

        # Equilibrate
        system, positions = equilibration.run(system, positions)

        # Run production
        result = runner.run(topology, system, positions, output_dir)

        logger.info("replica_complete", replica_id=replica_id, **result)
        return result


class CpHEnsemble(EnsembleStrategy):
    """
    Constant pH ensemble simulation.
    
    This ensemble runs simulations at different pH values to study
    pH-dependent properties of the system.
    
    Attributes:
        ph_values: List of pH values to simulate
    """

    def __init__(
        self,
        config: MDConfig,
        system_builder: SystemBuilder,
        ph_values: List[float],
        resource_limits: Optional[ResourceLimits] = None,
    ) -> None:
        """
        Initialize the constant pH ensemble.
        
        Args:
            config: Configuration parameters for the simulations
            system_builder: Builder for creating OpenMM systems
            ph_values: List of pH values to simulate
            resource_limits: Resource limits for parallel execution
        """
        super().__init__(config, system_builder, resource_limits)
        self.ph_values = ph_values

    def run(
        self, 
        topology: app.Topology, 
        system: System, 
        positions: unit.Quantity, 
        output_dir: Path
    ) -> Dict:
        """
        Run constant pH ensemble simulation.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            
        Returns:
            Dictionary with ensemble results including results at each pH
        """
        logger.info("starting_cph_ensemble", ph_values=self.ph_values)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        ensemble_results: Dict = {}
        # Start MLflow run
        with mlflow.start_run(run_name=f"cph_ensemble_{int(time.time())}"):
            # Log parameters
            mlflow.log_params({**self.config.model_dump(), "ph_values": self.ph_values})

            # Get available workers
            max_workers = min(len(self.ph_values), self.resource_limits.get_available_workers())
            logger.info("resource_allocation", max_workers=max_workers)

            # Run pH replicas in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for ph in self.ph_values:
                    ph_dir = output_dir / f"ph_{ph:.1f}"
                    futures.append(
                        executor.submit(
                            self._run_ph_replica, 
                            topology, 
                            system, 
                            positions, 
                            ph_dir, 
                            ph
                        )
                    )

                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error("ph_replica_failed", error=str(e))

            # Aggregate results
            ensemble_results = {
                "ph_values": self.ph_values,
                "completed_replicas": len(results),
                "replica_results": results,
            }

            # Log ensemble metrics
            mlflow.log_metrics({
                "completed_replicas": len(results), 
                "success_rate": len(results) / len(self.ph_values)
            })

            logger.info("ensemble_complete", **ensemble_results)

        return ensemble_results

    def _run_ph_replica(
        self, 
        topology: app.Topology, 
        system: System, 
        positions: unit.Quantity, 
        output_dir: Path, 
        ph: float
    ) -> Dict:
        """
        Run a simulation at a specific pH value.
        
        Args:
            topology: The OpenMM topology of the system
            system: The OpenMM system to simulate
            positions: Initial positions of the system
            output_dir: Directory to save simulation outputs
            ph: pH value for this simulation
            
        Returns:
            Dictionary with simulation results
        """
        logger.info("starting_ph_replica", ph=ph)

        # Create pH-specific config
        ph_config = self.config.model_copy()
        ph_config.system.ph = ph

        # Setup and run
        equilibration = EquilibrationProtocol(ph_config, self.system_builder)
        runner = MDRunner(ph_config, self.system_builder)

        # Equilibrate
        system, positions = equilibration.run(system, positions)

        # Run production
        result = runner.run(topology, system, positions, output_dir)

        logger.info("ph_replica_complete", ph=ph, **result)
        return result


def run_ensemble(
    topology: app.Topology, 
    system: System, 
    positions: unit.Quantity, 
    output_dir: Path, 
    strategy: str = "npt", 
    **kwargs
) -> Dict:
    """
    Run ensemble simulation with specified strategy.
    
    This is a convenience function that creates the appropriate ensemble
    strategy and runs it with the provided parameters.
    
    Args:
        topology: The OpenMM topology of the system
        system: The OpenMM system to simulate
        positions: Initial positions of the system
        output_dir: Directory to save simulation outputs
        strategy: Ensemble strategy to use ("npt" or "cph")
        **kwargs: Additional keyword arguments for the ensemble strategy
        
    Returns:
        Dictionary with ensemble results
        
    Raises:
        ValueError: If an unknown ensemble strategy is specified
    """
    config = kwargs.pop("config", MDConfig(random_seed=42, platform="CUDA", device_index=0))
    system_builder = kwargs.pop("system_builder", SystemBuilder())
    resource_limits = kwargs.pop("resource_limits", ResourceLimits())

    if strategy == "npt":
        num_replicas = kwargs.pop("num_replicas", 4)
        ensemble = NPTEnsemble(config, system_builder, num_replicas, resource_limits)
    elif strategy == "cph":
        ph_values = kwargs.pop("ph_values", [7.0])
        ensemble = CpHEnsemble(config, system_builder, ph_values, resource_limits)
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    return ensemble.run(topology, system, positions, output_dir)
