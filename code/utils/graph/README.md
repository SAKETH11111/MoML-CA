# Molecular Graph Generation Module

This module provides functionality for generating molecular graphs from different sources:

1. `molecular_graph_generator.py`: Creates molecular graphs directly from mol files without using quantum properties. This is used as a fallback when QM calculations are skipped.

2. `qm_graph_generator.py`: Creates molecular graphs incorporating quantum mechanical properties from ORCA calculations (placeholder implementation).

## Usage

### Direct Graph Generation (No QM)

```python
from code.utils.graph.molecular_graph_generator import batch_create_graphs_from_molecules

# Generate graphs from mol files
graph_files = batch_create_graphs_from_molecules(
    mol_dir="/path/to/mol/files",
    output_dir="/path/to/output",
    use_pfas_features=True,  # Include PFAS-specific features
    max_workers=4  # Number of parallel workers
)
```

### QM-Based Graph Generation

```python
from code.utils.graph.qm_graph_generator import batch_create_graphs_from_orca

# Generate graphs using QM properties
graph_files = batch_create_graphs_from_orca(
    mol_dir="/path/to/mol/files",
    orca_dir="/path/to/orca/outputs",
    output_dir="/path/to/output",
    charge_type="mulliken",  # or "loewdin"
    use_pfas_features=True,
    use_quantum_properties=True
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