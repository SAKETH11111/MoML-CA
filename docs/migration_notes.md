# Migration Notes - Removing Deprecated Modules

## Modules Removed

- `code/MGNN/utils/graph_coarsening_utils.py` - Functionality moved to `GraphCoarsener` class
- `code/MGNN/utils/graph_utils.py` - Functionality moved to `architectures/molecular_graph.py`

## Modules That Still Need to Be Kept

The following modules are deprecated but could not be completely removed yet due to dependencies:

1. `code/MGNN/architectures/molecular_graph.py` (MolecularGraphBuilder)
   - ~~Needed by~~:
     - ~~`code/tests/test_simple_graph.py`~~ UPDATED ✓
     - ~~`code/MGNN/tests/test_molecular_graph.py`~~ UPDATED ✓
   - While tests have been updated to use MolecularGraphProcessor, the MolecularGraphBuilder class is still kept for any external code that might depend on it.

## Migration Plan

1. ~~Update all remaining tests to use the new MolecularGraphProcessor API~~ COMPLETED ✓
2. Remove the deprecated modules after all tests are migrated
3. Update any dependencies in integration code to use the new API

## Migration Progress

- [x] Updated core MGNN package to use MolecularGraphProcessor
- [x] Updated `__init__.py` files to remove deprecated imports
- [x] Updated example files to use new API
- [x] Created new tests for MolecularGraphProcessor
- [x] Updated test_graph_coarsening.py to use new API
- [x] Removed `code/MGNN/utils/graph_utils.py` and consolidated functionality
- [x] Updated test_simple_graph.py
- [x] Updated test_molecular_graph.py
- [ ] Remove remaining deprecated files after all tests pass

## Modernized API Usage

For any code still using the deprecated classes, here are the migration paths:

### From MolecularGraphBuilder to MolecularGraphProcessor

```python
# Old approach
from code.MGNN.architectures.molecular_graph import MolecularGraphBuilder
builder = MolecularGraphBuilder(use_partial_charges=True)
graph = builder.mol_to_graph(mol, partial_charges)

# New approach
from code.MGNN.architectures.molecular_graph import create_graph_processor
processor = create_graph_processor({'use_partial_charges': True})
graph = processor.mol_to_graph(mol, additional_features={'partial_charges': partial_charges})
```

### From graph_coarsening_utils to GraphCoarsener

```python
# Old approach
from code.MGNN.utils.graph_coarsening_utils import create_hierarchical_graphs_from_orca
graphs = create_hierarchical_graphs_from_orca('molecule.mol', 'output.out')

# New approach
from code.MGNN.architectures.graph_coarsening import GraphCoarsener
coarsener = GraphCoarsener()
graphs = coarsener.create_from_orca('molecule.mol', 'output.out')
```

# Migration Guide: Molecular Graph Implementation

## Overview of Changes

We have consolidated multiple implementations of molecular graph processing into a single, comprehensive implementation. The goal was to:

1. Remove redundancy between `molecular_graph.py` and `graph_utils.py`
2. Create a cleaner API with better organized code
3. Maintain backward compatibility through deprecation warnings

## Key Changes

### Consolidated Files

- `code/MGNN/architectures/molecular_graph.py` is now the primary source for molecular graph processing
- `code/MGNN/utils/graph_utils.py` has been deprecated and now redirects to `molecular_graph.py`

### Updated Classes

- `MolecularGraphProcessor` has been moved to `molecular_graph.py` with enhanced functionality
- The deprecated `MolecularGraphBuilder` class is maintained for backward compatibility

### API Changes

- Import paths have changed:
  - Use `from code.MGNN.architectures.molecular_graph import MolecularGraphProcessor`
  - Instead of `from code.MGNN.utils.graph_utils import MolecularGraphProcessor`

## Migration Steps

### For New Code

Use the new consolidated API:

```python
from code.MGNN.architectures.molecular_graph import (
    MolecularGraphProcessor,
    create_graph_processor
)

# Create a processor with specific configuration
processor = create_graph_processor({
    'use_partial_charges': True,
    'use_3d_coords': True,
    'use_pfas_specific_features': True
})

# Process a molecule file
graph = processor.file_to_graph('molecule.mol')

# Or from SMILES
graph = processor.smiles_to_graph('CCC')
```

### For Existing Code

Existing code will continue to work with deprecation warnings. To update:

1. Find all imports of `graph_utils` and update to `molecular_graph`:

```python
# Old
from code.MGNN.utils.graph_utils import MolecularGraphProcessor

# New
from code.MGNN.architectures.molecular_graph import MolecularGraphProcessor
```

2. Update method calls if needed:

```python
# Some method names have changed for clarity:
# Old: processor.file_to_graph(...)
# New: processor.file_to_graph(...)

# Old: processor.smiles_to_graph(...)
# New: processor.smiles_to_graph(...)
```

## Feature Comparison

| Feature                 | Old Implementation | New Implementation |
| ----------------------- | ------------------ | ------------------ |
| Basic graph creation    | ✅                 | ✅                 |
| PFAS-specific features  | ⚠️ (Limited)       | ✅ (Enhanced)      |
| 3D coordinate support   | ⚠️ (Limited)       | ✅ (Full)          |
| QM property integration | ⚠️ (Limited)       | ✅ (Enhanced)      |
| Batch processing        | ✅                 | ✅                 |

## Future Work

- In a future release, deprecated modules will be removed
- Any tests still using the deprecated implementations should be updated
- Documentation should be expanded with more examples for the new API
