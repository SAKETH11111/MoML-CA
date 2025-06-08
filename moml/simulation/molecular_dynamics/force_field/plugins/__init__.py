"""
Plugin loader for force field components.
"""

import importlib
from pathlib import Path
from typing import Tuple, Dict, Any

def load_plugin(plugin_name: str) -> Tuple[Dict[str, Any], Any]:
    """Load a force field plugin by name.
    
    Args:
        plugin_name: Name of the plugin to load (e.g. 'nf_polyamide_v1')
        
    Returns:
        Tuple containing:
        - Dict: Plugin configuration from config.yaml
        - Any: Plugin build module with get_xml method
        
    Raises:
        ImportError: If plugin module cannot be loaded
        ValueError: If plugin is missing required files
    """
    # Get the plugin directory
    plugin_dir = Path(__file__).parent / plugin_name
    
    if not plugin_dir.exists():
        raise ValueError(f"Plugin directory not found: {plugin_name}")
    
    # Check for required files
    build_file = plugin_dir / "build.py"
    config_file = plugin_dir / "config.yaml"
    
    if not build_file.exists():
        raise ValueError(f"Plugin missing build.py: {plugin_name}")
    if not config_file.exists():
        raise ValueError(f"Plugin missing config.yaml: {plugin_name}")
    
    # Import the build module
    module_path = f"..force_field.plugins.{plugin_name}.build"
    try:
        build_module = importlib.import_module(module_path, package=__package__)
    except ImportError as e:
        raise ImportError(f"Failed to load plugin module {plugin_name}: {e}")
    
    # Load config
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    return config, build_module 