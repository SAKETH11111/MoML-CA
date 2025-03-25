# MoML-CA: MGNN Module Quick Start Guide

This guide provides a quick introduction to using the Molecular Graph Neural Networks (MGNN) module for PFAS analysis.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MoML-CA.git
cd MoML-CA

# Install the package and dependencies
pip install -e .
```

## Data Pipeline

The easiest way to get started is using the built-in data pipeline:

```bash
# Process a set of PFAS molecules with ORCA outputs
pfas-pipeline --mol-dir data/molecules --orca-dir data/orca_outputs --output-dir results

# Generate hierarchical graphs and visualize them
pfas-pipeline --mol-dir data/molecules --orca-dir data/orca_outputs --output-dir results --hierarchical --visualize

# Analyze functional groups in molecules
pfas-pipeline --mol-dir data/molecules --orca-dir data/orca_outputs --output-dir results --analyze
```

## Working with Individual Molecules

You can also process individual molecules using the graph generator tool:

```bash
# Create an atomic-level graph
pfas-graph atomic --mol data/molecules/GenX.mol --orca data/orca_outputs/GenX.out

# Create hierarchical graphs
pfas-graph hierarchical --mol data/molecules/GenX.mol --orca data/orca_outputs/GenX.out

# Analyze functional groups
pfas-graph analyze --mol data/molecules/GenX.mol

# Visualize with different highlighting options
pfas-graph atomic --mol data/molecules/GenX.mol --orca data/orca_outputs/GenX.out --visualize --highlight functional_group
```

## Python API

You can use the MGNN module in your Python code:

```python
import torch
from rdkit import Chem
from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder
from code.MGNN.utils import create_graph_from_orca_data, create_hierarchical_graphs_from_orca
from code.MGNN.utils.visualization import visualize_molecular_graph, print_graph_statistics

# Create a molecular graph from ORCA data
graph_path = create_graph_from_orca_data(
    mol_file="data/molecules/GenX.mol",
    orca_output="data/orca_outputs/GenX.out",
    use_pfas_features=True,
    use_quantum_properties=True
)

# Load and use the graph
graph = torch.load(graph_path)
print_graph_statistics(graph)

# Visualize the graph
visualize_molecular_graph(graph, "genx_visualization.png", highlight_feature="fluorine")

# Create a hierarchical representation
hierarchical_graphs = create_hierarchical_graphs_from_orca(
    mol_file="data/molecules/GenX.mol",
    orca_output="data/orca_outputs/GenX.out"
)
```

## Key Features

- **Enhanced PFAS-specific features**: Fluorine atom flags, distance-to-functional group metrics, CF group detection
- **Quantum properties integration**: Partial charges, HOMO/LUMO contributions, electrostatic potential
- **Multi-scale representation**: Atom level, functional group level, structural motif level
- **Visualization tools**: Highlight different features (fluorine atoms, partial charges, functional groups)
- **Batch processing**: Process entire directories of molecules and ORCA outputs

## Next Steps

Refer to the README.md and the example scripts in `code/MGNN/examples/` for more detailed usage information. 