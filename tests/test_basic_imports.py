"""
Basic import tests that don't require heavy dependencies.
"""
import pytest
import sys
import os

def test_basic_imports():
    """Test that basic Python modules can be imported."""
    import numpy as np
    import torch
    assert np.__version__
    assert torch.__version__

def test_moml_package_structure():
    """Test that the moml package structure is correct."""
    # Add the project root to the path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test basic package structure
    import moml
    assert hasattr(moml, '__version__')
    
    # Test that subpackages exist (but don't import them to avoid dependency issues)
    moml_path = os.path.join(project_root, 'moml')
    assert os.path.exists(os.path.join(moml_path, 'core'))
    assert os.path.exists(os.path.join(moml_path, 'data'))
    assert os.path.exists(os.path.join(moml_path, 'models'))
    assert os.path.exists(os.path.join(moml_path, 'pipeline'))
    assert os.path.exists(os.path.join(moml_path, 'simulation'))

def test_torch_geometric_availability():
    """Test if torch_geometric is available."""
    try:
        import torch_geometric
        assert torch_geometric.__version__
        print(f"torch_geometric version: {torch_geometric.__version__}")
    except ImportError:
        pytest.skip("torch_geometric not available")

def test_rdkit_availability():
    """Test if rdkit is available."""
    try:
        from rdkit import Chem
        # Test basic functionality
        mol = Chem.MolFromSmiles('CCO')
        assert mol is not None
        print("RDKit is available and functional")
    except ImportError:
        pytest.skip("RDKit not available")