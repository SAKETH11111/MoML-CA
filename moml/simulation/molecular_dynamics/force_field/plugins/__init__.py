"""
__init__.py

Force Field Plugin System Module

This module provides a plugin system for force field components in molecular dynamics
simulations. It enables dynamic loading of different surface types and materials
that can be used in simulations, with each plugin providing its own configuration
and implementation.

The module handles plugin discovery, loading, and validation to ensure that all
required components are present and properly structured. This modular approach
allows for easy extension of the simulation capabilities with new surface types.
"""

import importlib
from pathlib import Path
from typing import Tuple, Dict, Any

def load_plugin(plugin_name: str) -> Tuple[Dict[str, Any], Any]:
    """
    Load a force field plugin by name.
    
    This function locates and loads a force field plugin, validating that it has
    the required files and interface. It returns both the plugin's configuration
    and the build module that contains the implementation code.
    
    Args:
        plugin_name: Name of the plugin to load (e.g. 'nf_polyamide_v1', 'gac_v1')
        
    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing:
            - Dict[str, Any]: Plugin configuration loaded from config.yaml
            - Any: Plugin build module with the 'build' function
        
    Raises:
        ValueError: If plugin directory is missing or required files are not found
        ImportError: If plugin module cannot be loaded
        AttributeError: If plugin module lacks required interface
        
    Example:
        >>> config, build_module = load_plugin('gac_v1')
        >>> # Use the build module to create a surface
        >>> surface = build_module.build(temp_dir, config)
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
        raise ImportError(f"Failed to load plugin module {plugin_name}") from e
    
    # Validate that the build_module has the expected interface
    if not hasattr(build_module, 'build') or not callable(build_module.build):
        raise AttributeError(
            f"Plugin module {plugin_name} does not have a callable 'build' function. "
            "Please ensure the plugin module provides the required interface."
        )

    # Load config
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    return config, build_module