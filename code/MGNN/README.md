# Molecular Graph Neural Networks (MGNN) for PFAS

This module provides functionality for creating and using molecular graph representations of PFAS compounds, integrating quantum chemical data from ORCA calculations.

## Overview

The MGNN module enables:

1. Creation of molecular graph representations from PFAS structures
2. Integration of quantum chemical properties (e.g., partial charges, HOMO/LUMO contributions) from ORCA calculations
3. PFAS-specific node and edge features that capture fluorine-carbon interactions
4. Multi-scale graph representations through graph coarsening at functional group and structural motif levels
5. Utilities for batch processing of molecular structures

## Module Structure

- `architectures/`: Graph neural network architectures
  - `molecular_graph.py`: Molecular graph representation implementation
  - `graph_coarsening.py`: Hierarchical graph coarsening functionality
- `utils/`: Utility functions
  - `orca_parser.py`: Functions for parsing ORCA output files
  - `mol_graph_generator.py`: Functions for generating molecular graphs
  - `graph_coarsening_utils.py`: Utilities for creating hierarchical graphs
- `examples/`: Example scripts
  - `create_genx_graph.py`: Example script for creating a graph from GenX molecule
  - `create_hierarchical_graphs.py`: Example script for creating hierarchical graphs
- `tests/`: Unit and integration tests

## Enhanced PFAS-Specific Features

### Node (Atom) Features

The implementation now includes specialized features for PFAS compounds:

- **Basic Atomic Features**:
  - Atomic number (one-hot encoded)
  - Degree (one-hot encoded)
  - Formal charge (one-hot encoded)
  - Hybridization (one-hot encoded)
  - Aromaticity flag
  - Ring membership flag
  - Number of hydrogen atoms

- **PFAS-Specific Features**:
  - Fluorine atom flag
  - Carbon-fluorine bond flag
  - Number of fluorine neighbors
  - Functional group identification (carboxylic, sulfonic, phosphonic)
  - Distance to nearest CF3 group
  - Distance to nearest functional group
  - Head group vs. fluorinated tail position

- **Quantum-Derived Features**:
  - Partial charges from ORCA calculations
  - HOMO/LUMO contributions
  - Electrostatic potential values

### Edge (Bond) Features

- Bond type (single, double, triple, aromatic)
- Conjugation flag
- Ring membership flag 
- Carbon-fluorine bond flag
- Bond length (from 3D coordinates)
- Bond between two carbons with fluorine atoms
- Bond in fluorinated tail vs. head group
- Bond in functional group

### Global (Molecule) Features

- Molecular weight
- Topological polar surface area
- Number of hydrogen bond donors
- Number of hydrogen bond acceptors
- LogP (octanol-water partition coefficient)
- Total atom count
- Fluorine atom count
- CF3 group count
- Functional group counts (carboxylic, sulfonic, phosphonic)
- Distribution of fluorinated carbons (CF, CF2, CF3)

## Multi-Scale Graph Coarsening

The MGNN module now supports hierarchical graph representations at multiple scales:

### Atom Level (Original)
- Each node represents an individual atom
- Full atomic detail with all features described above

### Functional Group Level
- Atoms are clustered into functional groups like:
  - CF, CF2, CF3 (fluorinated carbon groups)
  - COOH (carboxylic acid group)
  - SO3H (sulfonic acid group)
  - PO3H2 (phosphonic acid group)
- Features are aggregated from constituent atoms
- Reduces graph complexity while preserving chemical significance

### Structural Motif Level
- Functional groups are further clustered into larger motifs:
  - Head group (typically containing the functional groups)
  - Fluorinated tail (containing CF, CF2, CF3 groups)
- Captures the dual nature of PFAS compounds (hydrophilic head, hydrophobic tail)
- Highest level of abstraction for very large molecules

This multi-scale approach allows models to learn at different levels of chemical abstraction, potentially improving performance on various prediction tasks.

## Usage Examples

### Creating a Molecular Graph for a PFAS Compound

```python
# Import the necessary modules
from code.MGNN.utils import create_graph_from_orca_data

# Create a molecular graph with PFAS-specific features and quantum properties
create_graph_from_orca_data(
    mol_file="path/to/molecule.mol",
    orca_output="path/to/orca_output.out",
    output_file="molecule_graph.pt",
    use_pfas_features=True,
    use_quantum_properties=True
)
```

### Creating Hierarchical Graphs at Multiple Scales

```python
# Import the necessary modules
from code.MGNN.utils import create_hierarchical_graphs_from_orca

# Create hierarchical graph representations
graph_paths = create_hierarchical_graphs_from_orca(
    mol_file="path/to/molecule.mol",
    orca_output="path/to/orca_output.out",
    output_dir="path/to/output_dir",
    use_pfas_features=True,
    use_quantum_properties=True
)

# Access the paths to the saved graphs
atom_graph_path = graph_paths['atom']
functional_group_graph_path = graph_paths['functional_group']
structural_motif_graph_path = graph_paths['structural_motif']
```

### Batch Processing Multiple Molecules

```python
from code.MGNN.utils import batch_create_hierarchical_graphs

# Batch process multiple molecules with hierarchical graphs
graph_paths = batch_create_hierarchical_graphs(
    mol_dir="path/to/molecule_files",
    orca_dir="path/to/orca_outputs",
    output_dir="path/to/output_graphs",
    use_pfas_features=True,
    use_quantum_properties=True
)
```

### Command-line Script for Hierarchical Graph Creation

```bash
# Create hierarchical graphs and visualize them
python -m code.MGNN.examples.create_hierarchical_graphs --mol path/to/molecule.mol --orca path/to/orca.out --visualize

# Analyze functional groups in the molecule
python -m code.MGNN.examples.create_hierarchical_graphs --mol path/to/molecule.mol --orca path/to/orca.out --analyze-groups
```

## Dependencies

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Matplotlib (for visualization)
- NetworkX (for visualization) 