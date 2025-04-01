# MGNN Test Suite

This directory contains tests for the Molecular Graph Neural Network (MGNN) module, which is a key component of the PFAS analysis pipeline.

## Test Structure

The test suite is organized to verify the functionality of:

- **Molecular Graph Generation**: Creating graph representations of molecules with appropriate atom and bond features
- **Graph Coarsening**: Converting atom-level graphs to hierarchical representations at functional group and structural motif levels
- **Utility Functions**: Operations for creating, transforming, and visualizing hierarchical molecular graphs

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest -xvs .

# Run a specific test file
pytest -xvs test_graph_coarsening.py

# Run a specific test case
pytest -xvs test_graph_coarsening.py::TestFunctionalGroupIdentifier
```

## Test Files

- `test_graph_coarsening.py`: Comprehensive test suite for graph coarsening functionality
- `test_molecular_graph.py`: Tests for molecular graph generation

## Dependencies

Tests require the following packages:
- pytest
- torch
- torch_geometric
- rdkit
- numpy

These should be installed as part of the overall project environment. 