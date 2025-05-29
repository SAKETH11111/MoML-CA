"""
Timeseries extractor for converting MD trajectories to LSTM-ready data.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import mdtraj as md
import yaml
import structlog

logger = structlog.get_logger()

class TimeseriesExtractor:
    """Extracts timeseries data from MD trajectories."""
    
    def __init__(self, metrics_config: Optional[Path] = None):
        self.metrics_config = metrics_config or Path(__file__).parent / 'metrics.yaml'
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load metrics configuration."""
        with open(self.metrics_config) as f:
            return yaml.safe_load(f)
    
    def extract(self,
               trajectory_path: Path,
               topology_path: Path,
               output_path: Path) -> Dict:
        """Extract timeseries data from trajectory."""
        logger.info("starting_extraction",
                   trajectory=str(trajectory_path),
                   topology=str(topology_path))
        
        # Load trajectory
        traj = md.load(str(trajectory_path), top=str(topology_path))
        
        # Extract metrics
        metrics = {}
        for metric_name, metric_config in self.metrics.items():
            try:
                value = self._compute_metric(traj, metric_config)
                metrics[metric_name] = value
            except Exception as e:
                logger.error("metric_extraction_failed",
                           metric=metric_name,
                           error=str(e))
        
        # Save metrics
        np.save(output_path, metrics)
        
        logger.info("extraction_complete",
                   output=str(output_path),
                   metrics=list(metrics.keys()))
        return metrics
    
    def _compute_metric(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute a single metric from trajectory."""
        metric_type = config['type']
        
        if metric_type == 'rmsd':
            return self._compute_rmsd(traj, config)
        elif metric_type == 'rmsf':
            return self._compute_rmsf(traj, config)
        elif metric_type == 'rg':
            return self._compute_rg(traj, config)
        elif metric_type == 'sasa':
            return self._compute_sasa(traj, config)
        elif metric_type == 'hbonds':
            return self._compute_hbonds(traj, config)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def _compute_rmsd(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute RMSD."""
        ref_frame = config.get('ref_frame', 0)
        selection = config.get('selection', 'protein')
        
        return md.rmsd(traj, traj, ref_frame, selection)
    
    def _compute_rmsf(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute RMSF."""
        selection = config.get('selection', 'protein')
        
        return md.rmsf(traj, traj, selection)
    
    def _compute_rg(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute radius of gyration."""
        selection = config.get('selection', 'protein')
        
        return md.compute_rg(traj, selection)
    
    def _compute_sasa(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute solvent accessible surface area."""
        selection = config.get('selection', 'protein')
        probe_radius = config.get('probe_radius', 0.14)  # nm
        
        return md.shrake_rupley(traj, selection, probe_radius)
    
    def _compute_hbonds(self, traj: md.Trajectory, config: Dict) -> np.ndarray:
        """Compute hydrogen bonds."""
        selection = config.get('selection', 'protein')
        distance_cutoff = config.get('distance_cutoff', 0.3)  # nm
        angle_cutoff = config.get('angle_cutoff', 120)  # degrees
        
        return md.baker_hubbard(traj, selection, distance_cutoff, angle_cutoff)
    
    def validate_metrics(self) -> List[str]:
        """Validate metrics configuration."""
        errors = []
        
        for metric_name, config in self.metrics.items():
            # Check required fields
            if 'type' not in config:
                errors.append(f"Missing type for metric: {metric_name}")
                continue
            
            # Validate type
            if config['type'] not in ['rmsd', 'rmsf', 'rg', 'sasa', 'hbonds']:
                errors.append(f"Invalid type for metric {metric_name}: {config['type']}")
            
            # Validate parameters
            if config['type'] == 'rmsd':
                if 'ref_frame' in config and not isinstance(config['ref_frame'], int):
                    errors.append(f"Invalid ref_frame for {metric_name}: {config['ref_frame']}")
            
            elif config['type'] == 'sasa':
                if 'probe_radius' in config and not isinstance(config['probe_radius'], (int, float)):
                    errors.append(f"Invalid probe_radius for {metric_name}: {config['probe_radius']}")
            
            elif config['type'] == 'hbonds':
                if 'distance_cutoff' in config and not isinstance(config['distance_cutoff'], (int, float)):
                    errors.append(f"Invalid distance_cutoff for {metric_name}: {config['distance_cutoff']}")
                if 'angle_cutoff' in config and not isinstance(config['angle_cutoff'], (int, float)):
                    errors.append(f"Invalid angle_cutoff for {metric_name}: {config['angle_cutoff']}")
        
        return errors 