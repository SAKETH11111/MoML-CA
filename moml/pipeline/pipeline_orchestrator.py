"""
moml/pipeline/pipeline_orchestrator.py

Molecular analysis pipeline orchestration for MoML-CA.

This module provides comprehensive orchestration for molecular modeling and machine
learning workflows, specifically designed for contaminant analysis. It coordinates
the execution of multiple pipeline stages including data preprocessing, quantum
mechanical calculations, and molecular graph generation.

The orchestrator supports both general molecular analysis and PFAS-specific
workflows with enhanced feature extraction and analysis capabilities.

Classes:
    MOMLPipelineOrchestrator: Base orchestrator for molecular analysis pipelines
    PFASPipelineOrchestrator: PFAS-specific pipeline with enhanced analysis

Functions:
    main: Command-line entry point for pipeline execution
"""

import argparse
import concurrent.futures
import functools
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from moml.core import (
    calculate_molecular_descriptors,
    create_graph_processor,
)
from moml.data import graph_batch_process, process_dataset, process_mol_file_to_graph
from moml.simulation.qm.parser.orca_parser import batch_process_molecules

# Constants
DEFAULT_QM_FUNCTIONAL = "B3LYP"
DEFAULT_QM_BASIS_SET = "6-31G*"
DEFAULT_QM_NUM_PROCS = 4
DEFAULT_QM_MEMORY = 4000
DEFAULT_MAX_WORKERS = 4
DEFAULT_CHECKPOINT_INTERVAL = 10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("moml_orchestrator")


class MOMLPipelineOrchestrator:
    """
    Orchestrator for molecular analysis pipelines.
    
    This class provides a unified interface for coordinating complex molecular
    modeling workflows including data preprocessing, quantum mechanical calculations,
    and molecular graph generation. It manages pipeline state, handles dependencies
    between stages, and provides options for resuming interrupted workflows.
    
    Attributes:
        base_dir (str): Base directory for the project
        config (Dict[str, Any]): Configuration dictionary with all pipeline settings
        dirs (Dict[str, str]): Dictionary mapping logical names to directory paths
        state (Dict[str, Any]): Pipeline execution state tracking
        
    Example:
        >>> orchestrator = MOMLPipelineOrchestrator(
        ...     config_file="config.json",
        ...     data_dir="/path/to/data"
        ... )
        >>> results = orchestrator.run_full_pipeline("molecules.csv")
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the pipeline orchestrator.

        Args:
            config_file: Path to JSON configuration file containing pipeline settings.
                If not provided, uses default configuration.
            data_dir: Path to data directory. Overrides config file setting if provided.
            output_dir: Path to output directory. Overrides config file setting if provided.
            working_dir: Path to working directory. Overrides config file setting if provided.
            
        Raises:
            FileNotFoundError: If config_file is specified but doesn't exist
            OSError: If required directories cannot be created
        """
        # Project root is three levels up from this module's location
        self.base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Default configuration provides sensible defaults for all pipeline stages
        self.config: Dict[str, Any] = {
            "data_dir": os.path.join(self.base_dir, "data"),
            "output_dir": os.path.join(self.base_dir, "output"),
            "working_dir": os.path.join(self.base_dir, "working"),
            "orca_path": None,  # Auto-detected from system PATH during execution
            "parallel": {"enabled": False, "max_workers": DEFAULT_MAX_WORKERS},
            "qm": {
                "functional": DEFAULT_QM_FUNCTIONAL,
                "basis_set": DEFAULT_QM_BASIS_SET,
                "num_procs": DEFAULT_QM_NUM_PROCS,
                "memory": DEFAULT_QM_MEMORY,
            },
            "graph": {
                "charge_type": "mulliken",
                "use_specific_features": True,
                "use_quantum_properties": True,
            },
            "execution": {
                "skip_qm": False,
                "skip_graph_generation": False,
                "force_rerun": False,
                "cache_intermediates": True,
            },
        }

        # Merge user configuration file with defaults, preserving nested structure
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                file_config = json.load(f)
                # Deep merge preserves nested dictionaries rather than replacing them
                self._deep_update(self.config, file_config)

        self.config_file_path = config_file  # Store the original path used for loading
        self.config_path = config_file  # Alias for tests that might expect 'config_path'

        # Command-line arguments take precedence over configuration file settings
        if data_dir:
            self.config["data_dir"] = data_dir
        if output_dir:
            self.config["output_dir"] = output_dir
        if working_dir:
            self.config["working_dir"] = working_dir

        # Ensure all required directories exist before pipeline execution
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["working_dir"], exist_ok=True)

        # Organize data flow with separate directories for each pipeline stage
        self.dirs = {
            "raw_data": os.path.join(self.config["data_dir"], "raw"),
            "processed_data": os.path.join(self.config["data_dir"], "processed"),
            "orca_input": os.path.join(self.config["working_dir"], "orca_input"),
            "orca_output": os.path.join(self.config["working_dir"], "orca_output"),
            "molecule_files": os.path.join(self.config["working_dir"], "molecules"),
            "molecular_graphs": os.path.join(self.config["output_dir"], "molecular_graphs"),
            "analysis": os.path.join(self.config["output_dir"], "analysis"),
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Track pipeline execution state for resume capability and error recovery
        self.state = {
            "preprocessing_completed": False,
            "orca_calculated": False,
            "graphs_generated": False,
            "last_run": None,
            "molecules_processed": 0,
            "orca_success_count": 0,
            "orca_error_count": 0,
            "graph_count": 0,
            "errors": [],
        }

        logger.info("MOML Pipeline Orchestrator initialized")
        logger.info(f"Data directory: {self.config['data_dir']}")
        logger.info(f"Output directory: {self.config['output_dir']}")
        logger.info(f"Working directory: {self.config['working_dir']}")

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update dictionary d with values from dictionary u.
        
        This method performs a deep merge of two dictionaries, where nested
        dictionaries are merged recursively rather than being replaced entirely.

        Args:
            d: Target dictionary to update in-place
            u: Source dictionary containing new values to merge

        Returns:
            The updated target dictionary (same object as input d)
            
        Example:
            >>> base = {"a": {"x": 1, "y": 2}, "b": 3}
            >>> update = {"a": {"y": 20, "z": 30}, "c": 4}
            >>> result = self._deep_update(base, update)
            >>> result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}
            True
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _process_molecule_features(
        self, df: pd.DataFrame, molecule_id_column: str = "common_name"
    ) -> pd.DataFrame:
        """
        Extract molecular descriptors for valid molecules in the dataset.
        
        This method centralizes feature extraction logic to avoid redundancy across
        different pipeline stages. It processes only molecules with valid SMILES
        representations and calculates various molecular descriptors.

        Args:
            df: DataFrame containing molecular data with 'is_valid_smiles' and 
                'rdkit_mol' columns
            molecule_id_column: Name of column containing unique molecule identifiers

        Returns:
            The input DataFrame with additional molecular descriptor columns added
            
        Raises:
            KeyError: If required columns ('is_valid_smiles', 'rdkit_mol') are missing
            ValueError: If no valid molecules are found in the dataset
        """
        if "is_valid_smiles" not in df.columns:
            raise KeyError("DataFrame must contain 'is_valid_smiles' column")
        if "rdkit_mol" not in df.columns:
            raise KeyError("DataFrame must contain 'rdkit_mol' column")
            
        valid_mask = df["is_valid_smiles"]
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            logger.warning("No valid molecules found for feature extraction")
            return df
            
        logger.info(f"Extracting features for {valid_count} valid molecules")

        for idx, row in df[valid_mask].iterrows():
            try:
                descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
                for name, value in descriptors.items():
                    df.at[idx, name] = value
            except Exception as e:
                logger.warning(
                    f"Failed to extract features for molecule at index {idx}: {e}"
                )

        return df

    def preprocess_data(
        self,
        input_file: str,
        smiles_col: str = "SMILES",
        id_col: str = "common_name",
        force_rerun: bool = False,
    ) -> pd.DataFrame:
        """
        Preprocess molecular data by validating SMILES and calculating descriptors.
        
        This method performs comprehensive preprocessing of molecular datasets including
        SMILES validation, RDKit molecule object creation, and molecular descriptor
        calculation. Results are cached to avoid reprocessing.

        Args:
            input_file: Path to input CSV file containing molecular data
            smiles_col: Name of column containing SMILES string representations
            id_col: Name of column containing unique molecule identifiers
            force_rerun: If True, reprocess data even if cached results exist

        Returns:
            Processed DataFrame with validation results, RDKit molecules, and
            calculated molecular descriptors
            
        Raises:
            FileNotFoundError: If input_file does not exist
            ValueError: If required columns are missing from the input file
            OSError: If output directory cannot be created or written to
        """
        logger.info(f"Preprocessing data from {input_file}")

        # Process dataset using consolidated function
        df = process_dataset(input_file, smiles_col=smiles_col, id_col=id_col)

        # Calculate descriptors for valid molecules
        valid_mask = df["is_valid_smiles"]

        for idx, row in df[valid_mask].iterrows():
            descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
            for name, value in descriptors.items():
                df.at[idx, name] = value

        # Cache processed data for downstream pipeline stages
        output_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
        df.to_csv(output_file, index=False)

        logger.info(f"Preprocessed {len(df)} molecules, saved to {output_file}")

        # Mark preprocessing as complete for pipeline state tracking
        self.state["preprocessing_completed"] = True
        self.state["molecules_processed"] = len(df)

        return df

    def run_orca_calculations(
        self,
        df: Optional[pd.DataFrame] = None,
        input_file: Optional[str] = None,
        smiles_col: str = "SMILES",
        id_col: str = "common_name",
        force_rerun: bool = False,
    ) -> pd.DataFrame:
        """
        Execute ORCA quantum mechanical calculations for molecular dataset.
        
        This method runs quantum mechanical calculations using the ORCA software
        package for molecules in the dataset. It supports both single-threaded
        and parallel execution modes and handles calculation failures gracefully.

        Args:
            df: DataFrame containing molecular data. If None, attempts to load
                from previously processed data or from input_file
            input_file: Path to input CSV file. Used if df is None and no
                processed data is available
            smiles_col: Name of column containing SMILES string representations
            id_col: Name of column containing unique molecule identifiers
            force_rerun: If True, recalculate even if results already exist

        Returns:
            DataFrame containing calculation results with status information
            and quantum mechanical properties for successful calculations
            
        Raises:
            ValueError: If no input data is provided and no processed data exists
            FileNotFoundError: If input_file is specified but doesn't exist
            RuntimeError: If ORCA executable is not found or accessible
        """
        # Determine data source with fallback hierarchy: provided → cached → raw input
        if df is None:
            processed_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
            if os.path.exists(processed_file):
                logger.info(f"Loading processed data from {processed_file}")
                df = pd.read_csv(processed_file)
            elif input_file and os.path.exists(input_file):
                logger.info(f"Preprocessing data from {input_file}")
                df = self.preprocess_data(input_file, smiles_col, id_col)
            else:
                raise ValueError("No data provided and no processed data found")

        # Process only molecules with valid SMILES to avoid ORCA failures
        valid_df = df[df["is_valid_smiles"]].copy()
        logger.info(f"Running ORCA calculations for {len(valid_df)} molecules")

        # Execute calculations with configured quantum chemistry settings
        qm_config = self.config["qm"]
        parallel_config = self.config["parallel"]

        orca_results = batch_process_molecules(
            molecules_df=valid_df,  # type: ignore
            output_dir=self.dirs["orca_output"],
            functional=qm_config["functional"],
            basis_set=qm_config["basis_set"],
            num_procs=qm_config["num_procs"],
            memory_mb=qm_config["memory"],
            orca_executable=self.config["orca_path"],
            max_workers=parallel_config["max_workers"] if parallel_config["enabled"] else 1,
            smiles_col=smiles_col,
            id_col=id_col,
        )

        # Persist calculation results for downstream graph generation
        results_file = os.path.join(self.dirs["orca_output"], "orca_results.csv")
        orca_results.to_csv(results_file, index=False)

        # Update pipeline state with calculation statistics
        self.state["orca_calculated"] = True
        self.state["orca_success_count"] = sum(orca_results["status"] == "completed")
        self.state["orca_error_count"] = sum(orca_results["status"] == "error")

        logger.info(
            f"ORCA calculations completed: {self.state['orca_success_count']} successful, {self.state['orca_error_count']} failed"
        )

        return orca_results

    def generate_molecular_graphs(
        self,
        mol_dir: Optional[str] = None,
        orca_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        force_rerun: bool = False,
    ) -> List[str]:
        """
        Generate molecular graph representations from molecular data and QM results.
        
        This method creates molecular graph objects incorporating both structural
        information from molecular geometries and quantum mechanical properties
        from ORCA calculations. The graphs are suitable for use with graph neural
        networks and other ML models.

        Args:
            mol_dir: Directory containing molecular structure files. Defaults to
                configured molecule files directory if not provided
            orca_dir: Directory containing ORCA quantum calculation outputs.
                Defaults to configured ORCA output directory if not provided
            output_dir: Directory to save generated graph files. Defaults to
                configured molecular graphs directory if not provided
            force_rerun: If True, regenerate graphs even if they already exist

        Returns:
            List of file paths to successfully generated molecular graph files
            
        Raises:
            FileNotFoundError: If specified directories don't exist
            ValueError: If no molecular data files are found
            OSError: If graph files cannot be written to output directory
        """
        mol_dir = mol_dir or self.dirs["molecule_files"]
        orca_dir = orca_dir or self.dirs["orca_output"]
        output_dir = output_dir or self.dirs["molecular_graphs"]

        graph_config = self.config["graph"]

        logger.info(f"Generating molecular graphs from {mol_dir} and {orca_dir}")

        # Find molecule files
        mol_files = [f for f in os.listdir(mol_dir) if f.endswith(".mol")] if os.path.exists(mol_dir) else []

        # Check if QM calculations were skipped and we need to generate molecule files
        skip_qm = self.config.get("execution", {}).get("skip_qm", False)
        if skip_qm and not mol_files:
            logger.info("QM calculations were skipped, generating molecule files from SMILES")
            # Ensure directory exists
            os.makedirs(mol_dir, exist_ok=True)

            processed_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
            if not os.path.exists(processed_file):
                logger.warning(f"Processed data file not found: {processed_file}")
                return []

            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                import pandas as pd

                df = pd.read_csv(processed_file)
                valid_df = df[df["is_valid_smiles"]].copy()

                mol_files = []
                for idx, row in valid_df.iterrows():
                    smiles = row.get("canonical_smiles", row.get("SMILES"))
                    mol_id = row.get("common_name", f"molecule_{idx}")

                    try:
                        # Generate 3D conformer for molecular dynamics compatibility
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore
                            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore

                            # Export structure for downstream QM calculations
                            mol_file = os.path.join(mol_dir, f"{mol_id}.mol")
                            Chem.MolToMolFile(mol, mol_file)
                            mol_files.append(f"{mol_id}.mol")
                            logger.info(f"Generated molecule file for {mol_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate molecule file for {smiles}: {str(e)}")

                logger.info(f"Generated {len(mol_files)} molecule files")
            except ImportError:
                logger.error("RDKit is required for generating molecule files")
                return []

        if not mol_files:
            logger.warning(f"No molecule files found in {mol_dir}")
            return []

        # Find or create mock ORCA outputs if QM calculations were skipped
        orca_files = [f for f in os.listdir(orca_dir) if f.endswith(".out")] if os.path.exists(orca_dir) else []

        if skip_qm and not orca_files:
            logger.info("QM calculations were skipped, creating placeholder ORCA outputs")
            # Ensure directory exists
            os.makedirs(orca_dir, exist_ok=True)

            # Generate placeholder QM results for testing without ORCA
            orca_files = []
            for mol_file in mol_files:
                mol_id = os.path.splitext(mol_file)[0]
                orca_file = os.path.join(orca_dir, f"{mol_id}.out")

                # Minimal output format with neutral atomic charges
                with open(orca_file, "w") as f:
                    f.write(f"ORCA PLACEHOLDER OUTPUT FILE FOR {mol_id}\n")
                    f.write("CALCULATION COMPLETED\n")

                orca_files.append(f"{mol_id}.out")

            logger.info(f"Created {len(orca_files)} placeholder ORCA output files")

        if not orca_files:
            logger.warning(f"No ORCA output files found in {orca_dir}")
            return []

        logger.info(f"Found {len(mol_files)} molecule files and {len(orca_files)} ORCA output files")

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Simplified graph generation when quantum calculations are bypassed
            if skip_qm:
                logger.info("Using simplified graph generation without quantum properties")
                graph_files = graph_batch_process(
                    input_dir=mol_dir,
                    output_dir=output_dir,
                    config={
                        "use_specific_features": graph_config["use_specific_features"],
                        "use_quantum_properties": False,
                    },
                    file_pattern="*.mol",
                )
            else:
                # Enhanced graph generation with quantum mechanical properties integration
                # Process each molecule with QM properties
                graph_files = []
                mol_file_paths = [os.path.join(mol_dir, f) for f in mol_files]
                
                # Define the configuration for the graph processor
                config = {
                    "use_specific_features": graph_config["use_specific_features"],
                    "use_quantum_properties": True,
                    "charge_type": graph_config.get("charge_type", "mulliken")
                }

                # Function to process a single molecule
                def process_single_molecule(mol_file: str) -> Optional[str]:
                    # Instantiate processor inside worker to avoid non-picklable objects
                    processor = create_graph_processor(config)
                    try:

                        mol_id = os.path.splitext(os.path.basename(mol_file))[0]
                        logger.info(f"Processing molecule: {mol_id}")

                        # Find corresponding ORCA output file
                        orca_file = None
                        for ext in [".out", ".log", f"_{graph_config['charge_type']}.txt"]:
                            potential_file = os.path.join(orca_dir, f"{mol_id}{ext}")
                            if os.path.exists(potential_file):
                                orca_file = potential_file
                                break

                        # Use our new processor function
                        output_file = os.path.join(output_dir, f"{mol_id}_graph.pt")
                        process_mol_file_to_graph(
                            mol_file=mol_file, output_file=output_file, processor=processor, charges_file=orca_file
                        )

                        return output_file
                    except Exception as e:
                        logger.error(f"Error processing {mol_file}")
                        return None

                # Process molecules in parallel if enabled
                parallel_config = self.config.get("parallel", {})
                max_workers = parallel_config.get("max_workers", 4) if parallel_config.get("enabled", False) else 1

                if max_workers > 1:
                    logger.info(f"Processing {len(mol_file_paths)} molecules in parallel with {max_workers} workers")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Pass config explicitly to process_single_molecule
                        results = list(executor.map(process_single_molecule, mol_file_paths))
                        graph_files = [f for f in results if f is not None]
                else:
                    logger.info(f"Processing {len(mol_file_paths)} molecules sequentially")
                    for mol_file in mol_file_paths:
                        result = process_single_molecule(mol_file)
                        if result:
                            graph_files.append(result)

                logger.info(f"Created {len(graph_files)} molecular graphs with QM properties")
        except ImportError as e:
            logger.error(f"Failed to import graph generation modules: {str(e)}")
            return []

        self.state["graphs_generated"] = True
        self.state["graph_count"] = len(graph_files)

        logger.info(f"Generated {len(graph_files)} molecular graphs")

        return graph_files

    def run_full_pipeline(
        self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete molecular analysis pipeline from data preprocessing to graph generation.

        Args:
            input_file: Path to input CSV file with SMILES data
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers
            force_rerun: Force rerun of all steps

        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        self.state["last_run"] = datetime.now().isoformat()

        try:
            # Step 1: Preprocess data
            logger.info("Step 1: Preprocessing data")
            df = self.preprocess_data(input_file, smiles_col, id_col, force_rerun)

            # Step 2: Run ORCA calculations
            logger.info("Step 2: Running ORCA calculations")
            if not self.state.get("orca_calculated") or force_rerun:
                logger.info("Running ORCA calculations...")
                if self.config["execution"].get("skip_qm"):
                    logger.info("Skipping ORCA calculations as per configuration")
                else:
                    self.run_orca_calculations(
                        df, smiles_col=smiles_col, id_col=id_col, force_rerun=force_rerun
                    )
            else:
                logger.info("ORCA calculations already completed, skipping.")

            # Step 3: Generate molecular graphs
            logger.info("Step 3: Generating molecular graphs")
            if not self.state.get("graphs_generated") or force_rerun:
                logger.info("Generating molecular graphs...")
                if self.config["execution"].get("skip_graph_generation"):
                    logger.info("Skipping graph generation as per configuration")
                else:
                    self.generate_molecular_graphs(force_rerun=force_rerun)

            # Collect results
            pipeline_results = {
                "molecules_processed": len(df),
                "valid_molecules": sum(df["is_valid_smiles"]),
                "orca_success": self.state.get("orca_success_count", 0),
                "orca_errors": self.state.get("orca_error_count", 0),
                "graphs_generated": self.state.get("graph_count", 0),
                "execution_time": time.time() - start_time,
            }

            # Save pipeline state and results
            self._save_state()

            logger.info(f"Pipeline completed successfully in {pipeline_results['execution_time']:.2f} seconds")
            logger.info(f"Results: {pipeline_results}")

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.state["errors"].append(str(e))
            self._save_state()
            raise

    def _save_state(self) -> None:
        """Save the current pipeline state to a file."""
        state_file = os.path.join(self.config["output_dir"], "pipeline_state.json")
        with open(state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def resume_pipeline(self, input_file: Optional[str] = None, smiles_col: str = "SMILES", id_col: str = "common_name") -> Dict[str, Any]:
        """
        Resume the pipeline from the last successful stage.

        Args:
            input_file: Path to input CSV file (required if preprocessing hasn't been done)
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers

        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        self.state["last_run"] = datetime.now().isoformat()

        try:
            # Check if steps should be skipped
            skip_qm = self.config.get("execution", {}).get("skip_qm", False)
            skip_graphs = self.config.get("execution", {}).get("skip_graph_generation", False)

            # Determine what stages have been completed
            if not self.state["preprocessing_completed"]:
                if not input_file:
                    raise ValueError("Input file required to resume pipeline from preprocessing")
                logger.info("Resuming from preprocessing stage")
                df = self.preprocess_data(input_file, smiles_col, id_col)

                # Check if we should skip QM calculations
                if skip_qm:
                    logger.info("Skipping ORCA calculations as configured")
                    orca_results = None
                else:
                    orca_results = self.run_orca_calculations(df, smiles_col=smiles_col, id_col=id_col)

                # Check if we should skip graph generation
                if skip_graphs:
                    logger.info("Skipping molecular graph generation as configured")
                    graph_files = []
                else:
                    graph_files = self.generate_molecular_graphs()

            elif not self.state["orca_calculated"] and not skip_qm:
                logger.info("Resuming from ORCA calculation stage")
                if not input_file:
                    raise ValueError("Input file required to resume from ORCA calculation stage")
                df = self.preprocess_data(input_file, smiles_col, id_col)
                orca_results = self.run_orca_calculations(df, smiles_col=smiles_col, id_col=id_col)

                # Check if we should skip graph generation
                if skip_graphs:
                    logger.info("Skipping molecular graph generation as configured")
                    graph_files = []
                else:
                    graph_files = self.generate_molecular_graphs()

            elif not self.state["graphs_generated"] and not skip_graphs:
                logger.info("Resuming from graph generation stage")
                if not input_file:
                    raise ValueError("Input file required to resume from graph generation stage")
                df = self.preprocess_data(input_file, smiles_col, id_col)

                # Check if we should skip QM calculations
                if skip_qm:
                    logger.info("Skipping ORCA calculations as configured")
                    orca_results = None
                else:
                    self.run_orca_calculations(df, smiles_col=smiles_col, id_col=id_col)

                self.generate_molecular_graphs()
            else:
                logger.info("All pipeline stages already completed or skipped as configured")
                if not input_file:
                    raise ValueError("Input file required when no processed data exists")
                df = self.preprocess_data(input_file, smiles_col, id_col)

            # Collect results
            pipeline_results = {
                "molecules_processed": len(df) if isinstance(df, pd.DataFrame) else 0,
                "valid_molecules": sum(df["is_valid_smiles"]) if isinstance(df, pd.DataFrame) else 0,
                "orca_success": self.state.get("orca_success_count", 0),
                "orca_errors": self.state.get("orca_error_count", 0),
                "graphs_generated": self.state.get("graph_count", 0),
                "execution_time": time.time() - start_time,
            }

            # Save pipeline state and results
            self._save_state()

            logger.info(
                f"Pipeline resumed and completed successfully in {pipeline_results['execution_time']:.2f} seconds"
            )
            logger.info(f"Results: {pipeline_results}")

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline failed during resume: {str(e)}")
            self.state["errors"].append(str(e))
            self._save_state()
            raise


class PFASPipelineOrchestrator(MOMLPipelineOrchestrator):
    """
    Specialized orchestrator for PFAS molecular analysis.
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        cache_intermediates: bool = True,
    ):
        """
        Initialize the PFAS pipeline orchestrator.

        Args:
            config_file: Path to JSON configuration file
            data_dir: Path to data directory (overrides config file)
            output_dir: Path to output directory (overrides config file)
            working_dir: Path to working directory (overrides config file)
            cache_intermediates: Whether to cache intermediate results in memory
        """
        super().__init__(config_file, data_dir, output_dir, working_dir)

        # Handle backward compatibility for state key migration
        if "preprocessed" in self.state:
            if self.state["preprocessed"] and not self.state.get("preprocessing_completed"):
                self.state["preprocessing_completed"] = True
                logger.info("Migrated 'preprocessed: True' state to 'preprocessing_completed: True' in PFAS orchestrator.")
            del self.state["preprocessed"]
            logger.info("Removed 'preprocessed' key from state in PFAS orchestrator after superclass initialization.")
        
        # Ensure consistent state initialization across different entry points
        if "preprocessing_completed" not in self.state:
            self.state["preprocessing_completed"] = False
        # PFAS-specific analysis configuration with fluorine detection thresholds
        pfas_config = {
            "pfas": {
                "categorize_types": True,
                "identify_groups": True,
                "calculate_statistics": True,
                "min_f_atoms": 1,  # Minimum fluorine atoms for PFAS classification
                "min_f_c_ratio": 0.05,  # Minimum F:C ratio for PFAS classification
            },
            "execution": {"cache_intermediates": cache_intermediates},
        }

        # PFAS-specific configuration enhances analysis for fluorinated compounds
        self._deep_update(self.config, pfas_config)

        # Extended directory structure supports PFAS-specific outputs and checkpointing
        self.dirs["pfas_results"] = os.path.join(self.config["output_dir"], "pfas_analysis")
        self.dirs["checkpoints"] = os.path.join(self.config["working_dir"], "checkpoints")

        os.makedirs(self.dirs["pfas_results"], exist_ok=True)
        os.makedirs(self.dirs["checkpoints"], exist_ok=True)

        # In-memory caching reduces I/O overhead for large datasets
        self.cache = {"processed_df": None, "orca_results": None, "graph_results": None}

        logger.info("PFAS Pipeline Orchestrator initialized")

    def run_preprocessing_stage(
        self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess PFAS data with specialized PFAS-specific features.

        Args:
            input_file: Path to input CSV file
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers
            force_rerun: Force rerun even if processed data exists

        Returns:
            Processed DataFrame with PFAS-specific features
        """
        logger.info(
            f"Starting PFAS-specific preprocessing for {input_file} (run_preprocessing_stage), force_rerun={force_rerun}"
        )

        # Define the expected output path for this stage
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        processed_file_path = os.path.join(
            self.dirs["processed_data"], f"{base_name}_pfas_processed.csv"
        )  # Consistent naming

        # Check if already processed and not forcing rerun
        if (
            not force_rerun
            and os.path.exists(processed_file_path)
            and self.state.get("preprocessed_files", {}).get(input_file) == processed_file_path
        ):
            logger.info(
                f"Loading existing processed data from {processed_file_path} for {input_file} as force_rerun is False and state indicates completion."
            )
            try:
                df = pd.read_csv(processed_file_path)
                self.cache["processed_df"] = df  # type: ignore
                self.state["preprocessing_completed"] = True  # Ensure state is set when loading
                return df
            except Exception as e:
                logger.warning(f"Could not load existing processed file {processed_file_path}: {e}. Re-processing.")

        # If not resuming, proceed with actual processing:
        # This was previously: df = super().preprocess_data(input_file, smiles_col, id_col, force_rerun)
        # The PFAS orchestrator seems to want to do its own specific sequence.
        logger.info(f"Performing full preprocessing for {input_file}")
        df_initial_processed = process_dataset(input_file, smiles_col=smiles_col, id_col=id_col)
        df_with_descriptors = self._process_molecule_features(df_initial_processed, molecule_id_column=id_col)

        # Placeholder for PFAS-specific feature engineering or analysis
        df_final = df_with_descriptors  # Assuming no extra PFAS steps for now beyond base descriptors

        df_final.to_csv(processed_file_path, index=False)
        logger.info(f"PFAS-specific processed data saved to {processed_file_path}")

        self.state["preprocessing_completed"] = True
        if "preprocessed_files" not in self.state:
            self.state["preprocessed_files"] = {}
        self.state["preprocessed_files"][input_file] = processed_file_path  # Track specific file
        self.state["molecules_processed"] = len(df_final)  # This might need to be cumulative or per file
        self.cache["processed_df"] = df_final  # type: ignore
        self._save_state()
        return df_final

    # Reuse parent class implementation instead of redefining
    def preprocess_data(
        self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess data, delegating to the PFAS-specific method.
        """
        return self.run_preprocessing_stage(input_file, smiles_col, id_col, force_rerun)

    def run_orca_calculations(
        self, df: Optional[pd.DataFrame] = None, input_file: Optional[str] = None, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Run ORCA quantum mechanical calculations for PFAS molecules.

        Args:
            df: DataFrame containing SMILES strings (if None, will load from processed data)
            input_file: Path to input CSV file (used if df is None and no processed data exists)
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers
            force_rerun: Force rerun of calculations even if they already exist

        Returns:
            DataFrame with calculation results
        """
        # Check if we should skip QM calculations
        if self.config["execution"]["skip_qm"]:
            logger.info("ORCA calculations skipped according to configuration")
            return pd.DataFrame()  # Return empty DataFrame

        # Check if we have cached results and not forcing rerun
        if (
            not force_rerun
            and self.cache["orca_results"] is not None
            and self.config["execution"]["cache_intermediates"]
        ):
            logger.info("Using cached ORCA results")
            return self.cache["orca_results"]

        # Run base ORCA calculations
        orca_results = super().run_orca_calculations(df, input_file, smiles_col, id_col, force_rerun)

        # Store results in memory cache for performance optimization
        if self.config["execution"]["cache_intermediates"]:
            self.cache["orca_results"] = orca_results  # type: ignore

        return orca_results

    def analyze_pfas_dataset(self, df: pd.DataFrame, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Perform comprehensive PFAS-specific dataset analysis and classification.
        
        This method analyzes PFAS compounds for structural features, fluorine content,
        and functional groups, generating detailed statistics and classifications
        for contaminant analysis workflows.

        Args:
            df: Processed DataFrame containing molecular data with descriptors
            input_file: Original input file path for reference tracking

        Returns:
            DataFrame with PFAS analysis results and classifications added
            
        Raises:
            ValueError: If required PFAS analysis columns are missing
            OSError: If analysis output directory cannot be created
        """
        logger.info("Starting comprehensive PFAS dataset analysis.")

        analysis_dir = os.path.join(self.dirs["pfas_results"], "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Extract PFAS compounds from dataset if classification exists
        pfas_df = df
        if "is_pfas" in df.columns:
            pfas_df = df[df["is_pfas"]].copy()
            logger.info(f"Identified {len(pfas_df)} PFAS compounds out of {len(df)} total compounds")
            
            pfas_list_path = os.path.join(analysis_dir, "pfas_compounds_list.csv")
            pfas_df.to_csv(pfas_list_path, index=False)
            logger.info(f"Saved PFAS compounds list to {pfas_list_path}")
        else:
            logger.warning("'is_pfas' column not found, assuming all compounds are PFAS for analysis")

        if len(pfas_df) > 0:
            # Fluorine content analysis for PFAS characterization
            if "F_Count" in pfas_df.columns and "C_Count" in pfas_df.columns:
                f_c_ratios = pfas_df["F_Count"] / pfas_df["C_Count"].replace(0, float('nan'))  # type: ignore
                f_c_stats = {
                    "mean_f_c_ratio": f_c_ratios.mean(),
                    "median_f_c_ratio": f_c_ratios.median(),  # type: ignore
                    "min_f_c_ratio": f_c_ratios.min(),
                    "max_f_c_ratio": f_c_ratios.max()
                }
                logger.info(f"F:C ratio statistics: {f_c_stats}")
                
                with open(os.path.join(analysis_dir, "f_c_ratio_stats.json"), "w") as f:
                    json.dump(f_c_stats, f, indent=2)
            
            # Structural feature analysis for PFAS categorization
            struct_columns = ["Is_Aromatic", "Has_Rings", "Is_Cyclic", "Is_Branched", "Chain_Length"]
            struct_stats = {}
            for col in struct_columns:
                if col in pfas_df.columns:
                    struct_stats[col] = pfas_df[col].value_counts().to_dict()  # type: ignore
            
            if struct_stats:
                logger.info(f"PFAS structural statistics: {struct_stats}")
                with open(os.path.join(analysis_dir, "structural_stats.json"), "w") as f:
                    json.dump(struct_stats, f, indent=2)
            
            # Functional group analysis for environmental impact assessment
            func_group_cols = ["num_cf3_groups", "num_cf2_groups", "num_cf_groups"]
            if all(col in pfas_df.columns for col in func_group_cols):
                func_group_stats = {
                    col: {
                        "mean": pfas_df[col].mean(),
                        "median": pfas_df[col].median(),  # type: ignore
                        "max": pfas_df[col].max()
                    } 
                    for col in func_group_cols
                }
                logger.info(f"Functional group statistics: {func_group_stats}")
                with open(os.path.join(analysis_dir, "functional_group_stats.json"), "w") as f:
                    json.dump(func_group_stats, f, indent=2)

            # Chain length-based classification for persistence and bioaccumulation assessment
            if "Chain_Length" in pfas_df.columns:
                def classify_pfas(row):
                    if pd.isna(row["Chain_Length"]):
                        return "Unknown"
                    length = row["Chain_Length"]
                    if length <= 3:
                        return "Short-chain PFAS"
                    elif length <= 7:
                        return "Medium-chain PFAS"
                    else:
                        return "Long-chain PFAS"
                
                pfas_df["pfas_class"] = pfas_df.apply(classify_pfas, axis=1)
                class_stats = pfas_df["pfas_class"].value_counts().to_dict()  # type: ignore
                logger.info(f"PFAS class distribution: {class_stats}")
        
        logger.info("PFAS dataset analysis completed")
        return pfas_df  # type: ignore

    def execute_pipeline(self, input_file: Optional[str] = None, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False) -> Dict[str, Any]:
        """
        Run the complete PFAS analysis pipeline.

        Args:
            input_file: Path to input CSV file
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers
            force_rerun: Force rerun of all stages

        Returns:
            Dictionary with results from all pipeline stages
        """
        results = {}

        # 1. Preprocessing stage
        logger.info("Starting preprocessing stage")
        df = self.preprocess_data(input_file, smiles_col, id_col, force_rerun)  # type: ignore
        results["preprocessing"] = {
            "total_compounds": len(df),
            "valid_compounds": df["is_valid_smiles"].sum() if "is_valid_smiles" in df.columns else 0,
            "pfas_compounds": df["is_pfas"].sum() if "is_pfas" in df.columns else 0,
        }

        # 2. ORCA calculations stage (if not skipped)
        if not self.config["execution"]["skip_qm"]:
            logger.info("Starting ORCA calculations stage")
            orca_results = self.run_orca_calculations(df, input_file, smiles_col, id_col, force_rerun)
            results["orca"] = {
                "compounds_calculated": len(orca_results),
                "success_count": self.state["orca_success_count"],
                "error_count": self.state["orca_error_count"],
            }
        else:
            logger.info("Skipping ORCA calculations stage")
            results["orca"] = {"skipped": True}

        # 3. Molecular graph generation stage (if not skipped)
        if not self.config["execution"]["skip_graph_generation"]:
            logger.info("Starting molecular graph generation stage")
            results["graph_generation"] = {"compounds": 0} # Actual graph generation is handled by base or skipped
        else:
            logger.info("Skipping molecular graph generation stage")
            results["graph_generation"] = {"skipped": True}

        # 4. PFAS analysis stage
        logger.info("Starting PFAS analysis stage")
        pfas_results = self.analyze_pfas_dataset(df, input_file)
        results["pfas_analysis"] = {
            "total_pfas_compounds": len(pfas_results) if isinstance(pfas_results, pd.DataFrame) else 0
        }

        # Store final results
        if isinstance(df, pd.DataFrame):
            results["final_data"] = {
                "total_compounds": len(df),
                "valid_compounds": df["is_valid_smiles"].sum() if "is_valid_smiles" in df.columns else 0,
                "preprocessing_status": "completed" if self.state.get("preprocessing_completed") else "pending/failed",
            }

        # Store any errors
        results["errors"] = self.state["errors"]

        logger.info("Full pipeline completed")
        return results

    def _save_state(self) -> None:
        """
        Save the current pipeline state and PFAS-specific checkpoints.
        Overrides the base class method to add specific checkpointing.
        """
        super()._save_state()

        # PFAS-specific checkpointing enables fine-grained recovery for long-running analyses
        if self.state.get("preprocessing_completed"):
            checkpoint_file = os.path.join(self.dirs["checkpoints"], "preprocessing_checkpoint.pkl")
            # Directory creation here handles race conditions in parallel execution
            os.makedirs(self.dirs["checkpoints"], exist_ok=True)

            data_to_save = {
                "preprocessing_completed": self.state.get("preprocessing_completed", False),
                "molecules_processed": self.state.get("molecules_processed", 0),
                "valid_molecules": self.state.get("valid_molecules", 0),
            }
            try:
                with open(checkpoint_file, "wb") as f_pkl:
                    pickle.dump(data_to_save, f_pkl)
                logger.info(f"Saved PFAS preprocessing checkpoint to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save PFAS preprocessing checkpoint {checkpoint_file}")

        # Additional PFAS-specific checkpoints can be added here for other stages


def main() -> None:
    """
    Command-line interface for molecular analysis pipeline execution.
    
    This function provides a comprehensive CLI for running molecular modeling
    and machine learning pipelines. It supports both general molecular analysis
    and PFAS-specific workflows with configurable execution options.
    
    The CLI supports various execution modes:
    - Full pipeline execution (default)
    - Individual stage execution (preprocess, orca, graphs)
    - Resume interrupted pipelines
    - Force rerun with cached data invalidation
    
    Examples:
        Run full PFAS pipeline:
        $ moml-ca --pfas --input pfas_data.csv
        
        Run only preprocessing stage:
        $ moml-ca --stage preprocess --input molecules.csv
        
        Resume interrupted pipeline:
        $ moml-ca --stage resume --input molecules.csv
        
    Raises:
        SystemExit: On argument parsing errors or pipeline execution failures
    """
    parser = argparse.ArgumentParser(
        description="MoML-CA: Molecular Modeling and Machine Learning for Contaminant Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pfas --input pfas_molecules.csv
  %(prog)s --stage preprocess --input molecules.csv
  %(prog)s --stage resume --config my_config.json
        """,
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to JSON configuration file with pipeline settings",
    )
    parser.add_argument(
        "--input",
        metavar="FILE", 
        help="Path to input CSV file containing molecular data with SMILES",
    )
    
    # Directory options
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        help="Path to data directory (overrides config file setting)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Path to output directory (overrides config file setting)",
    )
    parser.add_argument(
        "--working-dir",
        metavar="DIR", 
        help="Path to working directory (overrides config file setting)",
    )
    
    # Execution control
    parser.add_argument(
        "--stage",
        choices=["preprocess", "orca", "graphs", "all", "resume"],
        default="all",
        help="Pipeline stage to execute (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun stages even if cached results exist",
    )
    parser.add_argument(
        "--skip-qm",
        action="store_true",
        help="Skip quantum mechanical calculations (use existing results)",
    )
    parser.add_argument(
        "--skip-graphs",
        action="store_true", 
        help="Skip molecular graph generation",
    )
    parser.add_argument(
        "--pfas",
        action="store_true",
        help="Use PFAS-specific pipeline with enhanced contaminant analysis",
    )

    args = parser.parse_args()

    # Select orchestrator based on analysis type (PFAS vs general molecular)
    if args.pfas:
        logger.info("Initializing PFAS-specific pipeline orchestrator")
        orchestrator = PFASPipelineOrchestrator(
            config_file=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            working_dir=args.working_dir,
            cache_intermediates=True,
        )
    else:
        logger.info("Initializing general molecular pipeline orchestrator")
        orchestrator = MOMLPipelineOrchestrator(
            config_file=args.config, data_dir=args.data_dir, output_dir=args.output_dir, working_dir=args.working_dir
        )

    # Command-line flags override configuration file settings
    if args.skip_qm:
        orchestrator.config["execution"]["skip_qm"] = True
    if args.skip_graphs:
        orchestrator.config["execution"]["skip_graph_generation"] = True

    # Execute pipeline based on selected stage
    if args.stage == "preprocess":
        if not args.input:
            parser.error("--input is required for preprocess stage")
        orchestrator.preprocess_data(args.input, force_rerun=args.force)

    elif args.stage == "orca":
        orchestrator.run_orca_calculations(input_file=args.input, force_rerun=args.force)

    elif args.stage == "graphs":
        orchestrator.generate_molecular_graphs(force_rerun=args.force)

    elif args.stage == "resume":
        if not args.input and not os.path.exists(
            os.path.join(orchestrator.dirs["processed_data"], "molecules_processed.csv")
        ):
            parser.error("--input is required for resume when no processed data exists")

        results = orchestrator.resume_pipeline(args.input)

        logger.info("Pipeline resumed and completed with results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

    elif args.stage == "all":
        if not args.input:
            parser.error("--input is required for full pipeline")

        results = orchestrator.run_full_pipeline(args.input, force_rerun=args.force)

        logger.info("Pipeline completed with results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        exit(1)
    else:
        exit(0)
