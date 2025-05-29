import pytest
import numpy as np
import mdtraj as md
from pathlib import Path
import tempfile
import yaml
from moml.simulation.molecular_dynamics.postprocessing.timeseries_extractor import TimeseriesExtractor

@pytest.fixture
def sample_metrics_config():
    """Create a sample metrics configuration for testing."""
    config = {
        'rmsd': {
            'type': 'rmsd',
            'ref_frame': 0,
            'selection': 'protein'
        },
        'rmsf': {
            'type': 'rmsf',
            'selection': 'protein'
        },
        'rg': {
            'type': 'rg',
            'selection': 'protein'
        },
        'sasa': {
            'type': 'sasa',
            'selection': 'protein',
            'probe_radius': 0.14
        },
        'hbonds': {
            'type': 'hbonds',
            'selection': 'protein',
            'distance_cutoff': 0.3,
            'angle_cutoff': 120
        }
    }
    return config

@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    # Create a simple trajectory with 10 frames
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue('ALA', chain)
    for _ in range(10):  # 10 atoms
        topology.add_atom('CA', md.element.carbon, residue)
    
    # Create random coordinates
    xyz = np.random.randn(10, 10, 3)  # 10 frames, 10 atoms, 3 coordinates
    return md.Trajectory(xyz, topology)

@pytest.fixture
def temp_files(sample_trajectory):
    """Create temporary files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        traj_path = tmp_path / 'test.xtc'
        top_path = tmp_path / 'test.pdb'
        output_path = tmp_path / 'output.npy'
        
        # Save trajectory and topology
        sample_trajectory.save(str(traj_path))
        sample_trajectory.save_pdb(str(top_path))
        
        yield traj_path, top_path, output_path

def test_timeseries_extractor_initialization(sample_metrics_config):
    """Test TimeseriesExtractor initialization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_metrics_config, f)
        f.flush()
        f.close()  # Close the file before using it
        
        extractor = TimeseriesExtractor(metrics_config=Path(f.name))
        assert extractor.metrics == sample_metrics_config
        
        # Clean up
        Path(f.name).unlink()

def test_metrics_validation(sample_metrics_config):
    """Test metrics configuration validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_metrics_config, f)
        f.flush()
        f.close()  # Close the file before using it
        
        extractor = TimeseriesExtractor(metrics_config=Path(f.name))
        errors = extractor.validate_metrics()
        assert len(errors) == 0
        
        # Clean up
        Path(f.name).unlink()

def test_invalid_metrics_validation():
    """Test validation of invalid metrics configuration."""
    invalid_config = {
        'invalid_metric': {
            'type': 'unknown_type'
        },
        'invalid_rmsd': {
            'type': 'rmsd',
            'ref_frame': 'not_an_integer'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        f.flush()
        f.close()  # Close the file before using it
        
        extractor = TimeseriesExtractor(metrics_config=Path(f.name))
        errors = extractor.validate_metrics()
        assert len(errors) > 0
        assert any('Invalid type for metric' in error for error in errors)
        assert any('Invalid ref_frame' in error for error in errors)
        
        # Clean up
        Path(f.name).unlink()

def test_extract_metrics(temp_files, sample_metrics_config):
    """Test metric extraction from trajectory."""
    traj_path, top_path, output_path = temp_files
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_metrics_config, f)
        f.flush()
        f.close()  # Close the file before using it
        
        extractor = TimeseriesExtractor(metrics_config=Path(f.name))
        metrics = extractor.extract(traj_path, top_path, output_path)
        
        assert isinstance(metrics, dict)
        assert all(metric_name in metrics for metric_name in sample_metrics_config.keys())
        assert output_path.exists()
        
        # Load saved metrics and verify
        saved_metrics = np.load(output_path, allow_pickle=True).item()
        assert saved_metrics.keys() == metrics.keys()
        
        # Clean up
        Path(f.name).unlink()

def test_compute_metric_rmsd(sample_trajectory, sample_metrics_config):
    """Test RMSD computation."""
    extractor = TimeseriesExtractor()
    rmsd = extractor._compute_rmsd(sample_trajectory, sample_metrics_config['rmsd'])
    assert isinstance(rmsd, np.ndarray)
    assert rmsd.shape == (sample_trajectory.n_frames,)

def test_compute_metric_rmsf(sample_trajectory, sample_metrics_config):
    """Test RMSF computation."""
    extractor = TimeseriesExtractor()
    rmsf = extractor._compute_rmsf(sample_trajectory, sample_metrics_config['rmsf'])
    assert isinstance(rmsf, np.ndarray)
    assert rmsf.shape == (sample_trajectory.n_atoms,)

def test_compute_metric_rg(sample_trajectory, sample_metrics_config):
    """Test radius of gyration computation."""
    extractor = TimeseriesExtractor()
    rg = extractor._compute_rg(sample_trajectory, sample_metrics_config['rg'])
    assert isinstance(rg, np.ndarray)
    assert rg.shape == (sample_trajectory.n_frames,)

def test_compute_metric_sasa(sample_trajectory, sample_metrics_config):
    """Test SASA computation."""
    extractor = TimeseriesExtractor()
    sasa = extractor._compute_sasa(sample_trajectory, sample_metrics_config['sasa'])
    assert isinstance(sasa, np.ndarray)
    assert sasa.shape == (sample_trajectory.n_frames,)

def test_compute_metric_hbonds(sample_trajectory, sample_metrics_config):
    """Test hydrogen bonds computation."""
    extractor = TimeseriesExtractor()
    hbonds = extractor._compute_hbonds(sample_trajectory, sample_metrics_config['hbonds'])
    assert isinstance(hbonds, np.ndarray)

def test_invalid_metric_type(sample_trajectory):
    """Test handling of invalid metric type."""
    extractor = TimeseriesExtractor()
    with pytest.raises(ValueError, match="Unknown metric type"):
        extractor._compute_metric(sample_trajectory, {'type': 'invalid_type'})

def test_nonexistent_trajectory(temp_files, sample_metrics_config):
    """Test handling of nonexistent trajectory file."""
    _, _, output_path = temp_files
    nonexistent_path = Path('nonexistent.xtc')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump(sample_metrics_config, f)
        f.flush()
        
        extractor = TimeseriesExtractor(metrics_config=Path(f.name))
        with pytest.raises(Exception):
            extractor.extract(nonexistent_path, nonexistent_path, output_path) 