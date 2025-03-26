# MoML-CA: Molecular Modeling & Machine Learning for Contaminant Analysis

This repository contains a hybrid computational framework for analyzing PFAS (Per- and Polyfluoroalkyl Substances) contaminants by combining quantum mechanical calculations with graph neural networks. The framework enables the prediction of PFAS properties and environmental behavior based on molecular structure.

## Overview

MoML-CA integrates multiple computational techniques:

1. **Molecular Processing**: Validation of SMILES strings and calculation of basic molecular descriptors
2. **Quantum Mechanical Calculations**: Using ORCA to obtain electronic properties
3. **Molecular Graph Generation**: Creating graph representations of molecules with quantum-enriched features
4. **Machine Learning**: Training and applying Graph Neural Networks for property prediction

## Installation

### Prerequisites

- Python 3.8 or higher
- RDKit
- PyTorch 1.12 or higher
- ORCA 5.0 or higher (optional, for quantum calculations)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/saketh/MoML-CA_PFAS.git
cd MoML-CA_PFAS

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Installing with Additional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For graph neural networks
pip install -e ".[mgnn]"

# For documentation
pip install -e ".[docs]"
```

### ORCA Installation (Optional)

To use quantum mechanical calculations, you need to install ORCA:

1. Download ORCA from [the official website](https://orcaforum.kofo.mpg.de)
2. Extract the files to a directory of your choice
3. Add the ORCA directory to your PATH
4. Set the `orca_path` parameter in the pipeline configuration

## Usage

### Quick Start

```bash
# Run the complete pipeline on a dataset
moml-ca --config config/pipeline_config.json --input data/raw/pfas_dataset.csv --stage all

# Run specific stages
moml-ca --stage preprocess --input data/raw/pfas_dataset.csv
moml-ca --stage orca --input data/processed/pfas_processed.csv
moml-ca --stage graphs

# Skip quantum mechanical calculations (faster execution)
moml-ca --config config/pipeline_config.json --skip-qm

# Skip both QM calculations and graph generation
moml-ca --config config/pipeline_config.json --skip-qm --skip-graphs

# Resume from the last successful stage
moml-ca --config config/pipeline_config.json --resume
```

### Pipeline Configuration

The pipeline uses a JSON configuration file to customize its behavior. 

1. Copy the template configuration file:
```bash
cp config/pipeline_config_template.json config/pipeline_config.json
```

2. Edit the configuration file to match your environment:
```json
{
  "data_dir": "/path/to/data",
  "output_dir": "/path/to/output",
  "working_dir": "/path/to/working",
  "orca_path": "/path/to/orca",
  "parallel": {
    "enabled": true,
    "max_workers": 4
  },
  "qm": {
    "functional": "B3LYP",
    "basis_set": "6-31G*",
    "num_procs": 4,
    "memory": 4000
  },
  "graph": {
    "charge_type": "mulliken",
    "use_pfas_features": true,
    "use_quantum_properties": true
  },
  "execution": {
    "skip_qm": false,
    "skip_graph_generation": false,
    "force_rerun": false,
    "cache_intermediates": true
  }
}
```

Note: The actual configuration file (`config/pipeline_config.json`) is excluded from version control to avoid committing personal paths and settings.

### Python API

You can also use the pipeline programmatically:

```python
from code.integration.orchestration.pfas_pipeline_orchestrator import PFASPipelineOrchestrator

# Initialize the orchestrator
orchestrator = PFASPipelineOrchestrator(
    config_file="config/pipeline_config.json",
    cache_intermediates=True
)

# Run the full pipeline
results = orchestrator.run_full_pipeline("data/raw/pfas_dataset.csv")

# Or run individual stages
df = orchestrator.preprocess_data("data/raw/pfas_dataset.csv")
orca_results = orchestrator.run_orca_calculations(df)
graph_files = orchestrator.generate_molecular_graphs()

# Resume a previously interrupted pipeline
orchestrator.resume_pipeline("data/raw/pfas_dataset.csv")
```

## Pipeline Stages

### 1. Data Preprocessing

- Validates SMILES strings
- Calculates basic molecular descriptors
- Creates 3D structures
- Outputs a processed dataset

### 2. Quantum Mechanical Calculations

- Performs quantum mechanical calculations using ORCA
- Extracts partial charges, orbital energies, and other electronic properties
- Provides quantum-enriched molecular representations

### 3. Molecular Graph Generation

- Creates graph representations of molecules
- Incorporates quantum mechanical data as node and edge features
- Generates PyTorch Geometric compatible graph objects
- Supports hierarchical graph coarsening for PFAS analysis

### 4. Machine Learning (MGNN)

- Trains Graph Neural Networks on the generated molecular graphs
- Predicts PFAS properties and environmental behavior
- Provides interpretable insights into structure-property relationships

## Directory Structure

```
MoML-CA_PFAS/
├── code/
│   ├── integration/
│   │   ├── orchestration/      # Pipeline orchestration
│   │   └── data_pipeline/      # Data processing pipeline
│   ├── MGNN/                   # Molecular Graph Neural Networks
│   │   ├── architectures/      # GNN model architectures
│   │   ├── utils/              # Graph generation utilities
│   │   └── tests/              # MGNN tests
│   ├── tests/                  # General tests
│   └── utils/
│       ├── helper_functions/   # Molecular processing functions
│       └── quantum/            # Quantum calculation utilities
├── config/                     # Configuration files
├── data/
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed datasets
├── output/                     # Output files
└── working/                    # Temporary working files
```

## Performance Optimization

This implementation includes several optimizations:

- **Memory Efficiency**: Optimized ORCA parsing to handle large output files
- **Parallel Processing**: Batch processing of molecules with configurable parallelism
- **Checkpointing**: Save and resume pipeline states to recover from failures
- **Caching**: Intermediate results caching to avoid redundant calculations
- **Modularity**: Independently enable/disable pipeline stages
- **Skip Options**: Skip compute-intensive steps like quantum mechanical calculations or graph generation

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest code/tests/test_smiles_validation.py
pytest code/MGNN/tests/test_graph_coarsening.py
```

### Code Formatting

```bash
# Format code with Black
black code/

# Sort imports
isort code/

# Run linter
flake8 code/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{MoML-CA2025,
  author = {Saketh Baddam & Daniel Umemezie},
  title = {MoML-CA: Molecular Modeling \& Machine Learning for Contaminant Analysis},
  year = {2025},
  url = {https://github.com/saketh/MoML-CA_PFAS}
}
```

## Acknowledgments

This project builds upon several open-source libraries, including RDKit, ORCA, PyTorch, and PyTorch Geometric.