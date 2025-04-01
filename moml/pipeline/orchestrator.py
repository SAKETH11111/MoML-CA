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
from typing import Dict, List, Optional
import json
import time
from datetime import datetime

# Import consolidated core functionality
from moml.core import (
    calculate_molecular_descriptors,
    create_graph_processor,
)

# Import consolidated data functionality
from moml.data import (
    process_dataset,
    batch_process_molecules,
    process_mol_file_to_graph
)

# For parallel processing
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("moml_orchestrator")


class MOMLPipelineOrchestrator:
    """
    Orchestrator for the molecular analysis pipeline.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 working_dir: Optional[str] = None):
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
            "parallel": {
                "enabled": False,
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
                "use_specific_features": True,
                "use_quantum_properties": True
            },
            "execution": {
                "skip_qm": False,
                "skip_graph_generation": False,
                "force_rerun": False,
                "cache_intermediates": True
            }
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Update config with file values (nested update)
                self._deep_update(self.config, file_config)
        
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
            "analysis": os.path.join(self.config["output_dir"], "analysis")
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize pipeline state
        self.state = {
            "preprocessed": False,
            "orca_calculated": False,
            "graphs_generated": False,
            "last_run": None,
            "molecules_processed": 0,
            "orca_success_count": 0,
            "orca_error_count": 0,
            "graph_count": 0,
            "errors": []
        }
        
        logger.info(f"MOML Pipeline Orchestrator initialized")
        logger.info(f"Data directory: {self.config['data_dir']}")
        logger.info(f"Output directory: {self.config['output_dir']}")
        logger.info(f"Working directory: {self.config['working_dir']}")
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
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
    
    def preprocess_data(self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False) -> pd.DataFrame:
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
        valid_mask = df['is_valid_smiles']
        
        for idx, row in df[valid_mask].iterrows():
            descriptors = calculate_molecular_descriptors(row['rdkit_mol'])
            for name, value in descriptors.items():
                df.at[idx, name] = value
        
        # Save processed data
        output_file = os.path.join(self.dirs["processed_data"], "molecules_processed.csv")
        df.to_csv(output_file, index=False)
        
        logger.info(f"Preprocessed {len(df)} molecules, saved to {output_file}")
        
        # Update state
        self.state["preprocessed"] = True
        self.state["molecules_processed"] = len(df)
        
        return df
    
    def run_orca_calculations(self, 
                             df: pd.DataFrame = None, 
                             input_file: str = None,
                             smiles_col: str = "SMILES", 
                             id_col: str = "common_name",
                             force_rerun: bool = False) -> pd.DataFrame:
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
        valid_df = df[df['is_valid_smiles']].copy()
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
            id_col=id_col
        )
        
        # Save results
        results_file = os.path.join(self.dirs["orca_output"], "orca_results.csv")
        orca_results.to_csv(results_file, index=False)
        
        # Update state
        self.state["orca_calculated"] = True
        self.state["orca_success_count"] = sum(orca_results["status"] == "completed")
        self.state["orca_error_count"] = sum(orca_results["status"] == "error")
        
        logger.info(f"ORCA calculations completed: {self.state['orca_success_count']} successful, {self.state['orca_error_count']} failed")
        
        return orca_results
    
    def generate_molecular_graphs(self, 
                                 mol_dir: str = None, 
                                 orca_dir: str = None,
                                 output_dir: str = None,
                                 force_rerun: bool = False) -> List[str]:
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
        mol_files = [f for f in os.listdir(mol_dir) if f.endswith('.mol')] if os.path.exists(mol_dir) else []
        
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
                valid_df = df[df['is_valid_smiles']].copy()
                
                # Generate molecule files
                mol_files = []
                for idx, row in valid_df.iterrows():
                    smiles = row.get('canonical_smiles', row.get('SMILES'))
                    mol_id = row.get('common_name', f"molecule_{idx}")
                    
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
        orca_files = [f for f in os.listdir(orca_dir) if f.endswith('.out')] if os.path.exists(orca_dir) else []
        
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
                with open(orca_file, 'w') as f:
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
                graph_files = batch_process_molecules(
                    input_dir=mol_dir,
                    output_dir=output_dir,
                    config={
                        'use_specific_features': graph_config["use_specific_features"],
                        'use_quantum_properties': False
                    },
                    file_pattern="*.mol"
                )
            else:
                # Create a QM-aware processor
                logger.info("Using graph generation with quantum properties")
                config = {
                    'use_specific_features': graph_config["use_specific_features"],
                    'use_partial_charges': graph_config["use_quantum_properties"]
                }
                processor = create_graph_processor(config)
                
                # Process each molecule with QM properties
                graph_files = []
                mol_file_paths = [os.path.join(mol_dir, f) for f in mol_files]
                
                # Function to process a single molecule
                def process_single_molecule(mol_file):
                    try:
                        import torch
                        from rdkit import Chem
                        from rdkit.Chem import AllChem
                        
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
                            mol_file=mol_file,
                            output_file=output_file,
                            processor=processor,
                            charges_file=orca_file
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
        
        # Update state
        self.state["graphs_generated"] = True
        self.state["graph_count"] = len(graph_files)
        
        logger.info(f"Generated {len(graph_files)} molecular graphs")
        
        return graph_files
    
    def run_full_pipeline(self, input_file: str, smiles_col: str = "SMILES", id_col: str = "common_name", force_rerun: bool = False) -> Dict:
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
            # Check if we should skip QM calculations
            if self.config.get("execution", {}).get("skip_qm", False):
                logger.info("Skipping ORCA calculations as configured")
                orca_results = None
            else:
                orca_results = self.run_orca_calculations(df, smiles_col=smiles_col, id_col=id_col, force_rerun=force_rerun)
            
            # Step 3: Generate molecular graphs
            logger.info("Step 3: Generating molecular graphs")
            # Check if we should skip graph generation
            if self.config.get("execution", {}).get("skip_graph_generation", False):
                logger.info("Skipping molecular graph generation as configured")
                graph_files = []
            else:
                graph_files = self.generate_molecular_graphs(force_rerun=force_rerun)
            
            # Collect results
            pipeline_results = {
                "molecules_processed": len(df),
                "valid_molecules": sum(df['is_valid_smiles']),
                "orca_success": self.state.get("orca_success_count", 0),
                "orca_errors": self.state.get("orca_error_count", 0),
                "graphs_generated": self.state.get("graph_count", 0),
                "execution_time": time.time() - start_time
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
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def resume_pipeline(self, 
                       input_file: str = None, 
                       smiles_col: str = "SMILES", 
                       id_col: str = "common_name") -> Dict:
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
            if not self.state["preprocessed"]:
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
                    orca_results = self.run_orca_calculations(df, smiles_col=smiles_col, id_col=id_col)
                    
                graph_files = self.generate_molecular_graphs()
            else:
                logger.info("All pipeline stages already completed or skipped as configured")
                df = self.preprocess_data(input_file, smiles_col, id_col)
            
            # Collect results
            pipeline_results = {
                "molecules_processed": len(df) if isinstance(df, pd.DataFrame) else 0,
                "valid_molecules": sum(df['is_valid_smiles']) if isinstance(df, pd.DataFrame) else 0,
                "orca_success": self.state.get("orca_success_count", 0),
                "orca_errors": self.state.get("orca_error_count", 0),
                "graphs_generated": self.state.get("graph_count", 0),
                "execution_time": time.time() - start_time
            }
            
            # Save pipeline state and results
            self._save_state()
            
            logger.info(f"Pipeline resumed and completed successfully in {pipeline_results['execution_time']:.2f} seconds")
            logger.info(f"Results: {pipeline_results}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed during resume: {str(e)}")
            self.state["errors"].append(str(e))
            self._save_state()
            raise


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Molecular Analysis Pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input", help="Path to input CSV file with SMILES data")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--output-dir", help="Path to output directory")
    parser.add_argument("--working-dir", help="Path to working directory")
    parser.add_argument("--stage", choices=["preprocess", "orca", "graphs", "all", "resume"], default="all",
                      help="Pipeline stage to run")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already processed")
    parser.add_argument("--skip-qm", action="store_true", help="Skip quantum mechanical calculations")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph generation")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MOMLPipelineOrchestrator(
        config_file=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        working_dir=args.working_dir
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
        orchestrator.resume_pipeline(input_file=args.input)
    
    elif args.stage == "all":
        if not args.input:
            parser.error("--input is required for full pipeline")
        orchestrator.run_full_pipeline(args.input, force_rerun=args.force)


if __name__ == "__main__":
    main() 