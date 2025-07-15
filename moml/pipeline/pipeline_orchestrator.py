#!/usr/bin/env python3
"""
MOML Pipeline Orchestrator

This module provides a complete orchestration layer for the molecular analysis pipeline.
It coordinates the execution of all pipeline stages:
1. Data preprocessing (SMILES validation, molecular property calculation)
2. Quantum mechanical calculations (ORCA)
3. Molecular graph generation
4. Model preparation

The orchestrator handles dependencies between stages, ensures proper data flow,
and provides options for executing the full pipeline or specific stages.
"""

import os
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import time
import pickle
from datetime import datetime
import concurrent.futures
import functools # Added for partial

from moml.core import (
    calculate_molecular_descriptors,
    create_graph_processor,
)
from moml.simulation.qm.parser.orca_parser import batch_process_molecules
from moml.data import process_dataset, process_mol_file_to_graph, graph_batch_process

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("moml_orchestrator")


class MOMLPipelineOrchestrator:
    """
    Orchestrator for the molecular analysis pipeline.
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config_file: Path to JSON configuration file
            data_dir: Path to data directory (overrides config file)
            output_dir: Path to output directory (overrides config file)
            working_dir: Path to working directory (overrides config file)
        """
        # Set up default paths
        self.base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Initialize configuration
        self.config = {
            "data_dir": os.path.join(self.base_dir, "data"),
            "output_dir": os.path.join(self.base_dir, "output"),
            "working_dir": os.path.join(self.base_dir, "working"),
            "orca_path": None,  # Will be auto-detected
            "parallel": {"enabled": False, "max_workers": 4},
            "qm": {"functional": "B3LYP", "basis_set": "6-31G*", "num_procs": 4, "memory": 4000},
            "graph": {"charge_type": "mulliken", "use_specific_features": True, "use_quantum_properties": True},
            "execution": {
                "skip_qm": False,
                "skip_graph_generation": False,
                "force_rerun": False,
                "cache_intermediates": True,
            },
        }

        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                file_config = json.load(f)
                # Update config with file values (nested update)
                self._deep_update(self.config, file_config)

        self.config_file_path = config_file  # Store the original path used for loading
        self.config_path = config_file  # Alias for tests that might expect 'config_path'

        # Override with provided paths if any
        if data_dir:
            self.config["data_dir"] = data_dir
        if output_dir:
            self.config["output_dir"] = output_dir
        if working_dir:
            self.config["working_dir"] = working_dir

        # Create required directories
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["working_dir"], exist_ok=True)

        # Set up stage-specific directories
        self.dirs = {
            "raw_data": os.path.join(self.config["data_dir"], "raw"),
            "processed_data": os.path.join(self.config["data_dir"], "processed"),
            "orca_input": os.path.join(self.config["working_dir"], "orca_input"),
            "orca_output": os.path.join(self.config["working_dir"], "orca_output"),
            "molecule_files": os.path.join(self.config["working_dir"], "molecules"),
            "molecular_graphs": os.path.join(self.config["output_dir"], "molecular_graphs"),
            "analysis": os.path.join(self.config["output_dir"], "analysis"),
        }

        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Initialize pipeline state
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
        Deep update dictionary d with values from dictionary u.

        Args:
            d: Dictionary to update
            u: Dictionary with new values

        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _process_molecule_features(self, df: pd.DataFrame, molecule_id_column: str = "common_name") -> pd.DataFrame:
        """
        Common method to process molecule features
        This method centralizes feature extraction logic to avoid redundancy.

        Args:
            df: DataFrame with molecule data
            molecule_id_column: Column containing molecule identifiers

        Returns:
            DataFrame with extracted features
        """
        valid_mask = df["is_valid_smiles"]

        for idx, row in df[valid_mask].iterrows():
            descriptors = calculate_molecular_descriptors(row["rdkit_mol"])
            for name, value in descriptors.items():
                df.at[idx, name] = value

        return df

    def preprocess_data(
        self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess molecular data from CSV file, validating SMILES and calculating descriptors.

        Args:
            input_file: Path to input CSV file
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers

        Returns:
            Processed DataFrame
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

        # Save processed data
        output_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
        df.to_csv(output_file, index=False)

        logger.info(f"Preprocessed {len(df)} molecules, saved to {output_file}")

        # Update state
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
        Run ORCA quantum mechanical calculations for molecules.

        Args:
            df: DataFrame containing SMILES strings (if None, will load from processed data)
            input_file: Path to input CSV file (used if df is None and no processed data exists)
            smiles_col: Column name containing SMILES strings
            id_col: Column name containing molecule identifiers
            force_rerun: Force rerun of calculations even if they already exist

        Returns:
            DataFrame with calculation results
        """
        # Get input data
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

        # Filter to only valid SMILES
        valid_df = df[df["is_valid_smiles"]].copy()
        logger.info(f"Running ORCA calculations for {len(valid_df)} molecules")

        # Run batch processing
        qm_config = self.config["qm"]
        parallel_config = self.config["parallel"]

        orca_results = batch_process_molecules(
            molecules_df=valid_df,
            output_dir=self.dirs["orca_output"],
            functional=qm_config["functional"],
            basis_set=qm_config["basis_set"],
            num_procs=qm_config["num_procs"],
            memory=qm_config["memory"],
            orca_path=self.config["orca_path"],
            max_workers=parallel_config["max_workers"] if parallel_config["enabled"] else 1,
            smiles_col=smiles_col,
            id_col=id_col,
        )

        # Save results
        results_file = os.path.join(self.dirs["orca_output"], "orca_results.csv")
        orca_results.to_csv(results_file, index=False)

        # Update state
        self.state["orca_calculated"] = True
        self.state["orca_success_count"] = sum(orca_results["status"] == "completed")
        self.state["orca_error_count"] = sum(orca_results["status"] == "error")

        logger.info(
            f"ORCA calculations completed: {self.state['orca_success_count']} successful, {self.state['orca_error_count']} failed"
        )

        return orca_results

    def generate_molecular_graphs(
        self, mol_dir: Optional[str] = None, orca_dir: Optional[str] = None, output_dir: Optional[str] = None, force_rerun: bool = False
    ) -> List[str]:
        """
        Generate molecular graphs from molecule files and ORCA outputs.

        Args:
            mol_dir: Directory containing molecule files (defaults to self.dirs["molecule_files"])
            orca_dir: Directory containing ORCA outputs (defaults to self.dirs["orca_output"])
            output_dir: Directory to save graphs (defaults to self.dirs["molecular_graphs"])
            force_rerun: Force regeneration of graphs even if they already exist

        Returns:
            List of paths to generated graph files
        """
        # Set default directories if not provided
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

            # Load processed data
            processed_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
            if not os.path.exists(processed_file):
                logger.warning(f"Processed data file not found: {processed_file}")
                return []

            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                import pandas as pd

                # Load processed data
                df = pd.read_csv(processed_file)
                valid_df = df[df["is_valid_smiles"]].copy()

                # Generate molecule files
                mol_files = []
                for idx, row in valid_df.iterrows():
                    smiles = row.get("canonical_smiles", row.get("SMILES"))
                    mol_id = row.get("common_name", f"molecule_{idx}")

                    try:
                        # Create RDKit molecule and generate 3D coordinates
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Add hydrogens and generate 3D coordinates
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol, randomSeed=42)
                            AllChem.MMFFOptimizeMolecule(mol)

                            # Save molecule file
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

            # Create placeholder ORCA outputs with neutral charges
            orca_files = []
            for mol_file in mol_files:
                mol_id = os.path.splitext(mol_file)[0]
                orca_file = os.path.join(orca_dir, f"{mol_id}.out")

                # Create a minimal ORCA output file with neutral charges
                with open(orca_file, "w") as f:
                    f.write(f"ORCA PLACEHOLDER OUTPUT FILE FOR {mol_id}\n")
                    f.write("CALCULATION COMPLETED\n")

                orca_files.append(f"{mol_id}.out")

            logger.info(f"Created {len(orca_files)} placeholder ORCA output files")

        if not orca_files:
            logger.warning(f"No ORCA output files found in {orca_dir}")
            return []

        logger.info(f"Found {len(mol_files)} molecule files and {len(orca_files)} ORCA output files")

        # Generate graphs
        os.makedirs(output_dir, exist_ok=True)

        try:
            # For the case when QM is skipped, use a simplified graph generation
            if skip_qm:
                logger.info("Using simplified graph generation without quantum properties")
                # Use our new batch processor from data module
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
                # Create a QM-aware processor
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
                        logger.error(f"Error processing {mol_file}: {e}")
                        return None

                # Process molecules in parallel if enabled
                parallel_config = self.config.get("parallel", {})
                max_workers = parallel_config.get("max_workers", 4) if parallel_config.get("enabled", False) else 1

                if max_workers > 1:
                    logger.info(f"Processing {len(mol_file_paths)} molecules in parallel with {max_workers} workers")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Pass config explicitly to process_single_molecule
                        partial_process_single_molecule = functools.partial(process_single_molecule, config=config)
                        results = list(executor.map(partial_process_single_molecule, mol_file_paths))
                        graph_files = [f for f in results if f is not None]
                else:
                    logger.info(f"Processing {len(mol_file_paths)} molecules sequentially")
                    for mol_file in mol_file_paths:
                        result = process_single_molecule(mol_file, config=config)
                        if result:
                            graph_files.append(result)

                logger.info(f"Created {len(graph_files)} molecular graphs with QM properties")
        except ImportError as e:
            logger.error(f"Failed to import graph generation modules: {str(e)}")
            return []

        # Update state
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
        # Initialize base class
        super().__init__(config_file, data_dir, output_dir, working_dir)

        # Migrate state from "preprocessed" (from base class init or loaded general state)
        # to "preprocessing_completed" and remove the old key.
        if "preprocessed" in self.state:
            if self.state["preprocessed"] and not self.state.get("preprocessing_completed"):
                # If old key is true and new key isn't already true (e.g. from a more specific checkpoint load),
                # transfer status. This handles cases where super state might have been loaded with "preprocessed": True.
                self.state["preprocessing_completed"] = True
                logger.info("Migrated 'preprocessed: True' state to 'preprocessing_completed: True' in PFAS orchestrator.")
            del self.state["preprocessed"]
            logger.info("Removed 'preprocessed' key from state in PFAS orchestrator after superclass initialization.")
        
        # Ensure 'preprocessing_completed' exists in state, defaulting to False if not set by migration or superclass.
        # This is important if super().__init__ didn't set "preprocessing_completed" and no checkpoint was loaded.
        if "preprocessing_completed" not in self.state:
            self.state["preprocessing_completed"] = False
        # Add PFAS-specific configuration
        pfas_config = {
            "pfas": {
                "categorize_types": True,
                "identify_groups": True,
                "calculate_statistics": True,
                "min_f_atoms": 1,  # Minimum number of F atoms to be considered PFAS
                "min_f_c_ratio": 0.05,  # Minimum F:C ratio to be considered PFAS
            },
            "execution": {"cache_intermediates": cache_intermediates},
        }

        # Update config with PFAS defaults
        self._deep_update(self.config, pfas_config)

        # Add PFAS-specific directories
        self.dirs["pfas_results"] = os.path.join(self.config["output_dir"], "pfas_analysis")
        self.dirs["checkpoints"] = os.path.join(self.config["working_dir"], "checkpoints")

        # Create the directories
        os.makedirs(self.dirs["pfas_results"], exist_ok=True)
        os.makedirs(self.dirs["checkpoints"], exist_ok=True)

        # Initialize cache for better performance
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
                self.cache["processed_df"] = df  # Update cache
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
        self.cache["processed_df"] = df_final  # Update cache
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

        # Cache results if requested
        if self.config["execution"]["cache_intermediates"]:
            self.cache["orca_results"] = orca_results

        return orca_results

    def analyze_pfas_dataset(self, df: pd.DataFrame, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Perform PFAS-specific dataset analysis.

        Args:
            df: Processed DataFrame containing molecular data.
            input_file: Original input file path (optional).

        Returns:
            DataFrame with PFAS analysis results.
        """
        logger.info("Starting comprehensive PFAS dataset analysis.")

        # Ensure output directories exist
        analysis_dir = os.path.join(self.dirs["pfas_results"], "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Filter to PFAS compounds if the flag exists
        pfas_df = df
        if "is_pfas" in df.columns:
            pfas_df = df[df["is_pfas"]].copy()
            logger.info(f"Identified {len(pfas_df)} PFAS compounds out of {len(df)} total compounds")
            
            # Save list of PFAS compounds for reference
            pfas_list_path = os.path.join(analysis_dir, "pfas_compounds_list.csv")
            pfas_df.to_csv(pfas_list_path, index=False)
            logger.info(f"Saved PFAS compounds list to {pfas_list_path}")
        else:
            logger.warning("'is_pfas' column not found, assuming all compounds are PFAS for analysis")

        # 2. Calculate key PFAS metrics and add them to the dataframe
        if len(pfas_df) > 0:
            # 2.1. Analyze fluorine content and distribution
            if "F_Count" in pfas_df.columns and "C_Count" in pfas_df.columns:
                # Calculate F:C ratio statistics
                f_c_ratios = pfas_df["F_Count"] / pfas_df["C_Count"].replace(0, float('nan'))
                f_c_stats = {
                    "mean_f_c_ratio": f_c_ratios.mean(),
                    "median_f_c_ratio": f_c_ratios.median(),
                    "min_f_c_ratio": f_c_ratios.min(),
                    "max_f_c_ratio": f_c_ratios.max()
                }
                logger.info(f"F:C ratio statistics: {f_c_stats}")
                
                # Save F:C ratio statistics
                with open(os.path.join(analysis_dir, "f_c_ratio_stats.json"), "w") as f:
                    json.dump(f_c_stats, f, indent=2)
            
            # 2.2. Categorize PFAS by structural features
            struct_columns = ["Is_Aromatic", "Has_Rings", "Is_Cyclic", "Is_Branched", "Chain_Length"]
            struct_stats = {}
            for col in struct_columns:
                if col in pfas_df.columns:
                    struct_stats[col] = pfas_df[col].value_counts().to_dict()
            
            if struct_stats:
                logger.info(f"PFAS structural statistics: {struct_stats}")
                with open(os.path.join(analysis_dir, "structural_stats.json"), "w") as f:
                    json.dump(struct_stats, f, indent=2)
            
            # 2.3. Analyze functional groups if data is available
            func_group_cols = ["num_cf3_groups", "num_cf2_groups", "num_cf_groups"]
            if all(col in pfas_df.columns for col in func_group_cols):
                func_group_stats = {
                    col: {
                        "mean": pfas_df[col].mean(),
                        "median": pfas_df[col].median(),
                        "max": pfas_df[col].max()
                    } 
                    for col in func_group_cols
                }
                logger.info(f"Functional group statistics: {func_group_stats}")
                with open(os.path.join(analysis_dir, "functional_group_stats.json"), "w") as f:
                    json.dump(func_group_stats, f, indent=2)

            # 2.4. Create classification of PFAS based on chain length and functional groups
            if "Chain_Length" in pfas_df.columns:
                # Define PFAS classes based on chain length
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
                class_stats = pfas_df["pfas_class"].value_counts().to_dict()
                logger.info(f"PFAS class distribution: {class_stats}")
        
        logger.info("PFAS dataset analysis completed")
        return pfas_df

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
        df = self.preprocess_data(input_file, smiles_col, id_col, force_rerun)
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
        super()._save_state()  # Call base class method to save general state

        # Save PFAS-specific preprocessing checkpoint
        if self.state.get("preprocessing_completed"): # Use the new, correct key
            checkpoint_file = os.path.join(self.dirs["checkpoints"], "preprocessing_checkpoint.pkl")
            # Ensure the 'checkpoints' directory exists, as it's defined in PFASPipelineOrchestrator.__init__
            # and _save_state might be called before other directory creation logic in some scenarios.
            os.makedirs(self.dirs["checkpoints"], exist_ok=True)

            data_to_save = {
                "preprocessing_completed": self.state.get("preprocessing_completed", False), # Save the new key
                "molecules_processed": self.state.get("molecules_processed", 0),
                "valid_molecules": self.state.get("valid_molecules", 0),
            }
            try:
                with open(checkpoint_file, "wb") as f_pkl:
                    pickle.dump(data_to_save, f_pkl)
                logger.info(f"Saved PFAS preprocessing checkpoint to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save PFAS preprocessing checkpoint {checkpoint_file}: {e}")

        # Future: Add other PFAS-specific checkpoints here if needed.


def main() -> None:
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Molecular Analysis Pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input", help="Path to input CSV file with SMILES data")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--output-dir", help="Path to output directory")
    parser.add_argument("--working-dir", help="Path to working directory")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "orca", "graphs", "all", "resume"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun even if already processed")
    parser.add_argument("--skip-qm", action="store_true", help="Skip quantum mechanical calculations")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph generation")
    parser.add_argument("--pfas", action="store_true", help="Use PFAS-specific pipeline with enhanced analysis")

    args = parser.parse_args()

    # Initialize orchestrator
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

    # Update configuration based on command-line arguments
    if args.skip_qm:
        orchestrator.config["execution"]["skip_qm"] = True
    if args.skip_graphs:
        orchestrator.config["execution"]["skip_graph_generation"] = True

    # Run requested stage
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

        # Resume pipeline
        results = orchestrator.resume_pipeline(args.input)

        # Print summary
        logger.info("Pipeline resumed and completed with results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

    elif args.stage == "all":
        if not args.input:
            parser.error("--input is required for full pipeline")

        # Run full pipeline
        results = orchestrator.run_full_pipeline(args.input, force_rerun=args.force)

        # Print summary
        logger.info("Pipeline completed with results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
