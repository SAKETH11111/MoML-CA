import importlib
from importlib.resources import files
import yaml

_PLUGIN_ROOT = files("moml.simulation.molecular_dynamics.force_field.plugins")

def load(surface_id: str):
    """Load a surface plugin's config and build module.
    
    Args:
        surface_id: The ID of the surface plugin to load
        
    Returns:
        Tuple of (config dict, build module)
        
    Raises:
        FileNotFoundError: If config.yaml is missing
        ImportError: If build module cannot be imported
        yaml.YAMLError: If config.yaml is invalid
    """
    pkg = _PLUGIN_ROOT.joinpath(surface_id)
    if not pkg.exists():
        raise FileNotFoundError(f"Plugin directory not found: {surface_id}")
        
    config_path = pkg / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in plugin: {surface_id}")
        
    cfg = yaml.safe_load(config_path.read_text())
    build_mod = importlib.import_module(
        f"moml.simulation.molecular_dynamics.force_field.plugins.{surface_id}.build"
    )
    return cfg, build_mod

