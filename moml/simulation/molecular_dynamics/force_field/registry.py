"""
moml/simulation/molecular_dynamics/force_field/registry.py

Force Field Plugin Registry Module

This module provides functionality to load force field plugins from the plugins directory.
It handles the dynamic loading of plugin configurations and build modules based on surface IDs.
Plugins are used to define specific surface types for molecular dynamics simulations.

The module uses Python's importlib to dynamically load plugin modules and YAML
for configuration file parsing.
"""

import importlib
from importlib.resources import files, is_resource, read_text
import yaml
import os

def load(surface_id: str):
    """
    Load a surface plugin's config and build module.
    
    This function dynamically loads a plugin based on its surface_id, returning both
    the configuration data from config.yaml and the build module that contains
    the implementation code. This enables the dynamic creation of different surface
    types in molecular dynamics simulations.
    
    Args:
        surface_id: The ID of the surface plugin to load (e.g., 'gac_v1', 'ro_v1')
        
    Returns:
        tuple: A tuple containing:
            - dict: The plugin configuration loaded from config.yaml
            - module: The build module containing plugin implementation
        
    Raises:
        FileNotFoundError: If the plugin directory or config.yaml is missing
        ImportError: If the build module cannot be imported
        yaml.YAMLError: If config.yaml contains invalid YAML
        
    Example:
        >>> config, build_module = load('gac_v1')
        >>> pdb_path, topology, indices = build_module.build(tmp_dir, config)
    """
    # Check if the plugin exists by trying to import it
    plugin_package = f"moml.simulation.molecular_dynamics.force_field.plugins.{surface_id}"
    try:
        importlib.import_module(plugin_package)
    except ImportError:
        raise FileNotFoundError(f"Plugin directory not found: {surface_id}")
    
    # Check if config.yaml exists and load it
    config_resource = f"{plugin_package}.config"
    try:
        config_text = read_text(plugin_package, "config.yaml")
        cfg = yaml.safe_load(config_text)
    except (FileNotFoundError, IsADirectoryError):
        raise FileNotFoundError(f"config.yaml not found in plugin: {surface_id}")
    
    # Import the build module
    build_mod = importlib.import_module(f"{plugin_package}.build")
    return cfg, build_mod

