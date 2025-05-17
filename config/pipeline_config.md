# MoML-CA Pipeline Configuration (`pipeline_config.template.json`)

This document provides a detailed explanation of the parameters found in the `pipeline_config.template.json` file. This configuration file is crucial for controlling the behavior of the MoML-CA pipeline, from data input and output locations to the specifics of quantum mechanics calculations and graph generation.

## General Structure

The configuration is a JSON object with several top-level keys, each grouping related parameters:

-   `data_dir`, `output_dir`, `working_dir`: Define essential file system paths.
-   `parallel`: Configures parallel execution settings.
-   `qm`: Specifies parameters for Quantum Mechanics (QM) calculations.
-   `graph`: Controls aspects of molecular graph generation.
-   `execution`: Manages the pipeline's execution flow, including skipping steps and caching.

---

## Path Configurations

These parameters define the key directories used by the pipeline. **It is essential to replace the placeholder paths (e.g., `/path/to/project/...`) with actual, valid paths on your system.**

### `data_dir`
-   **Purpose**: Specifies the root directory where input data for the pipeline is located. This typically includes molecular structure files (e.g., SDF, PDB), raw datasets, or any other initial data required by the pipeline.
-   **Type**: `string` (Path)
-   **Example**:
    ```json
    "data_dir": "/home/user/projects/moml_ca_project/input_data"
    ```
    On Windows:
    ```json
    "data_dir": "C:\\Users\\YourUser\\Documents\\MoML-CA\\data"
    ```

### `output_dir`
-   **Purpose**: Specifies the directory where the final outputs of the pipeline will be saved. This can include trained models, prediction results, generated reports, and other artifacts.
-   **Type**: `string` (Path)
-   **Example**:
    ```json
    "output_dir": "/home/user/projects/moml_ca_project/pipeline_outputs"
    ```
    On Windows:
    ```json
    "output_dir": "C:\\Users\\YourUser\\Documents\\MoML-CA\\outputs"
    ```

### `working_dir`
-   **Purpose**: Specifies a directory for storing intermediate files, logs, cached data, and temporary files generated during pipeline execution. This helps in organizing temporary data and can be useful for debugging or resuming interrupted runs if `cache_intermediates` is enabled.
-   **Type**: `string` (Path)
-   **Example**:
    ```json
    "working_dir": "/home/user/projects/moml_ca_project/working_files"
    ```
    On Windows:
    ```json
    "working_dir": "C:\\Users\\YourUser\\Documents\\MoML-CA\\work"
    ```

---

## Parallel Processing (`parallel`)

This section configures settings related to parallel execution of computationally intensive tasks within the pipeline.

### `parallel.enabled`
-   **Purpose**: A boolean flag to enable or disable parallel processing for tasks that support it (e.g., running multiple QM calculations simultaneously).
-   **Type**: `boolean`
-   **Valid Values**: `true`, `false`
-   **Default (in template)**: `true`
-   **Example**:
    ```json
    "enabled": true
    ```

### `parallel.max_workers`
-   **Purpose**: An integer specifying the maximum number of worker processes or threads to use when parallel processing is enabled. The optimal value depends on the number of CPU cores available on your system and the nature of the parallelizable tasks.
-   **Type**: `integer`
-   **Default (in template)**: `4`
-   **Example**:
    ```json
    "max_workers": 8
    ```
    *Note: Setting this too high relative to available resources might not improve performance and could lead to resource contention.*

---

## Quantum Mechanics (QM) Settings (`qm`)

This section defines parameters for the Quantum Mechanics calculations performed by software like ORCA.

### `qm.functional`
-   **Purpose**: A string specifying the Density Functional Theory (DFT) functional to be used for the QM calculations. The choice of functional can impact accuracy and computational cost.
-   **Type**: `string`
-   **Default (in template)**: `"B3LYP"`
-   **Valid Values**: Any functional supported by the underlying QM software (e.g., ORCA). Common examples include:
    -   `"B3LYP"`
    -   `"PBE"`
    -   `"PBE0"`
    -   `"TPSS"`
    -   `"M06"`
    -   `"M06-2X"`
    -   `"wB97X-D"`
    -   `"BLYP"`
-   **Example**:
    ```json
    "functional": "wB97X-D3"
    ```

### `qm.basis_set`
-   **Purpose**: A string specifying the basis set to be used for the QM calculations. The basis set determines the set of functions used to represent the electronic wavefunctions.
-   **Type**: `string`
-   **Default (in template)**: `"6-31G*"`
-   **Valid Values**: Any basis set supported by the underlying QM software. Common examples include:
    -   `"STO-3G"`
    -   `"3-21G"`
    -   `"6-31G*"` (aliased as `6-31G(d)`)
    -   `"6-31+G*"` (aliased as `6-31+G(d)`)
    -   `"6-311G**"` (aliased as `6-311G(d,p)`)
    -   `"cc-pVDZ"`, `"aug-cc-pVDZ"`
    -   `"def2-SVP"`, `"def2-TZVP"`
-   **Example**:
    ```json
    "basis_set": "def2-SVP"
    ```

### `qm.num_procs`
-   **Purpose**: An integer specifying the number of processors (CPU cores) to allocate for each individual QM calculation. This is relevant if the QM software itself supports internal parallelization for a single job.
-   **Type**: `integer`
-   **Default (in template)**: `2`
-   **Example**:
    ```json
    "num_procs": 4
    ```
    *Note: This is distinct from `parallel.max_workers`, which controls how many QM jobs run concurrently.*

### `qm.memory`
-   **Purpose**: An integer specifying the amount of memory (in Megabytes, MB) to allocate for each individual QM calculation.
-   **Type**: `integer`
-   **Default (in template)**: `4000` (i.e., 4 GB)
-   **Example**:
    ```json
    "memory": 8000
    ```

---

## Graph Generation Settings (`graph`)

This section controls parameters related to the construction of molecular graphs, which are used as input for Graph Neural Networks (GNNs).

### `graph.charge_type`
-   **Purpose**: A string specifying the method or type of partial atomic charges to be calculated and/or used as features in the molecular graph.
-   **Type**: `string`
-   **Default (in template)**: `"mulliken"`
-   **Valid Values**: The valid options depend on the capabilities of the QM software and the MoML-CA pipeline's implementation. Common types include:
    -   `"mulliken"`: Charges derived from Mulliken population analysis.
    -   `"hirshfeld"`: Charges derived from Hirshfeld population analysis.
    -   `"lowdin"`: Charges derived from Löwdin population analysis.
    -   `"esp"`: Charges fitted to reproduce the electrostatic potential (e.g., CHELPG, Merz-Kollman).
    -   `"nbo"`: Charges from Natural Bond Orbital analysis (if supported and NBO analysis is performed).
    -   `"gasteiger"`: Empirical charges, typically not from QM but can be an option for faster, less accurate estimations.
-   **Example**:
    ```json
    "charge_type": "hirshfeld"
    ```

### `graph.use_pfas_features`
-   **Purpose**: A boolean flag indicating whether to include specialized PFAS-specific atomic features during graph construction. These features might highlight the unique properties of fluorine atoms and C-F bonds.
-   **Type**: `boolean`
-   **Default (in template)**: `true`
-   **Example**:
    ```json
    "use_pfas_features": true
    ```

### `graph.use_quantum_properties`
-   **Purpose**: A boolean flag indicating whether to incorporate global or atomic properties derived from QM calculations (e.g., HOMO/LUMO energies, dipole moment, atomic Fukui indices) as features in the molecular graph nodes or edges.
-   **Type**: `boolean`
-   **Default (in template)**: `true`
-   **Example**:
    ```json
    "use_quantum_properties": true
    ```

---

## Execution Control (`execution`)

This section provides parameters to manage the pipeline's execution flow, such as skipping certain stages or forcing reruns.

### `execution.skip_qm`
-   **Purpose**: A boolean flag. If set to `true`, the pipeline will attempt to skip the QM calculation step. This is useful if QM calculations have already been performed and their results are available in the `working_dir` (or a specified QM output directory), allowing the pipeline to proceed directly to subsequent stages like graph generation.
-   **Type**: `boolean`
-   **Default (in template)**: `false`
-   **Example**:
    ```json
    "skip_qm": true
    ```

### `execution.skip_graph_generation`
-   **Purpose**: A boolean flag. If set to `true`, the pipeline will attempt to skip the molecular graph generation step. This assumes that graph files have been previously generated and are available for use.
-   **Type**: `boolean`
-   **Default (in template)**: `false`
-   **Example**:
    ```json
    "skip_graph_generation": true
    ```

### `execution.force_rerun`
-   **Purpose**: A boolean flag. If set to `true`, this forces the pipeline to rerun all steps from the beginning, ignoring any cached intermediate files or previous outputs. This will override `skip_qm` and `skip_graph_generation` if they are set to `true`.
-   **Type**: `boolean`
-   **Default (in template)**: `false`
-   **Example**:
    ```json
    "force_rerun": true
    ```

### `execution.cache_intermediates`
-   **Purpose**: A boolean flag. If set to `true`, the pipeline will save intermediate results (such as QM output files, generated graph objects) to the `working_dir`. This allows for faster subsequent runs if parts of the pipeline are re-executed, as these cached results can be reused (unless `force_rerun` is `true`).
-   **Type**: `boolean`
-   **Default (in template)**: `true`
-   **Example**:
    ```json
    "cache_intermediates": true
    ```

---

## Environment Variable Substitution

Currently, the `pipeline_config.template.json` does not explicitly define a syntax for environment variable substitution (e.g., using `${MY_DATA_PATH}/input.sdf`). Paths are expected to be specified directly.

If environment variable substitution is required, you would typically need to:
1.  Modify the pipeline's configuration loading mechanism to parse and substitute such variables.
2.  Alternatively, pre-process the JSON configuration file using a script or tool (like `envsubst` on Linux/macOS) before passing it to the pipeline.

For example, if the pipeline were enhanced to support this, you might write:
```json
// Hypothetical example if env var substitution was supported
"data_dir": "${PROJECT_ROOT}/data",
"output_dir": "${PROJECT_ROOT}/output"
```
And then ensure `PROJECT_ROOT` is set in your shell environment before running the pipeline. However, for the current template, provide absolute or relative paths directly.