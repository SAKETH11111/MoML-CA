# Molecular Graph Utilities

> **Important Note**: The graph generation functionality has been consolidated in the `moml.core.molecular_graph` module. Please use that module for all graph-related operations.

This directory once contained utilities for molecular graph generation. The functionality has been moved to the core module to reduce redundancy and improve maintainability.

## Migration Guide

If you were previously using functions from:

- `moml.utils.graph.molecular_graph_generator`
- `moml.utils.graph.qm_graph_generator`

Please update your code to use the equivalent functions from `moml.core.molecular_graph`:

### Replacements

| Old Function                                                   | New Function                                                |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| `molecular_graph_generator.generate_atom_features`             | `MolecularGraphProcessor._get_atom_features_dict`           |
| `molecular_graph_generator.generate_bond_features`             | `MolecularGraphProcessor._get_bond_features_dict`           |
| `molecular_graph_generator.get_molecule_descriptors`           | `MolecularGraphProcessor._get_molecule_descriptors`         |
| `molecular_graph_generator.create_molecular_graph`             | `create_molecular_graph_json`                               |
| `molecular_graph_generator.batch_create_graphs_from_molecules` | `batch_create_graphs_from_molecules`                        |
| `qm_graph_generator.batch_create_graphs_from_orca`             | Use `create_graph_processor` with appropriate configuration |

## Example Code

```python
from moml.core.molecular_graph import (
    MolecularGraphProcessor,
    create_graph_processor,
    create_molecular_graph_json,
    batch_create_graphs_from_molecules
)

# Create a processor with QM support
config = {
    'use_pfas_specific_features': True,
    'use_partial_charges': True
}
processor = create_graph_processor(config)

# Process a single molecule
graph = processor.file_to_graph('molecule.mol')

# Process a directory of molecules
graph_files = batch_create_graphs_from_molecules(
    mol_dir='molecules/',
    output_dir='graphs/'
)
```

## Graph Format

The generated graph files are in JSON format with the following structure:

```json
{
  "mol_id": "PFOA",
  "atoms": [
    {
      "idx": 0,
      "features": {
        "atomic_num": 9,
        "formal_charge": 0,
        "hybridization": 4,
        "num_hydrogens": 0,
        "is_aromatic": 0,
        "is_in_ring": 0,
        "degree": 1,
        "implicit_valence": 0,
        "explicit_valence": 1,
        "is_halogen": 1,
        "is_fluorine": 1,
        "is_carbon": 0,
        "is_oxygen": 0,
        "is_sulfur": 0,
        "is_nitrogen": 0,
        "is_phosphorus": 0
      },
      "coords": {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0
      }
    },
    ...
  ],
  "bonds": [
    {
      "begin_atom_idx": 0,
      "end_atom_idx": 1,
      "features": {
        "bond_type": 1,
        "is_conjugated": 0,
        "is_in_ring": 0,
        "is_aromatic": 0
      }
    },
    ...
  ],
  "descriptors": {
    "mol_weight": 414.0,
    "num_atoms": 29,
    "num_heavy_atoms": 15,
    ...
  },
  "quantum_properties": {
    // QM properties when available
  }
}
```

## Integration with Pipeline

The graph generation module is integrated with the PFAS pipeline orchestrator, which can be configured to skip QM calculations and use direct graph generation:

```shell
# Run pipeline with QM calculations skipped (use direct graph generation)
python run_pipeline.py --skip-qm

# Run pipeline with both QM and graph generation skipped
python run_pipeline.py --skip-qm --skip-graphs

# Run pipeline with default settings (perform QM and graph generation)
python run_pipeline.py
```
