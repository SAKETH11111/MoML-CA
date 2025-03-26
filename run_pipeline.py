#!/usr/bin/env python3
"""
MoML-CA Pipeline Runner

This script demonstrates running the complete PFAS analysis pipeline
on a sample dataset. It showcases the optimized pipeline orchestration,
caching mechanisms, and the ability to resume from interruptions.

Usage:
  python run_pipeline.py [--config CONFIG] [--input INPUT] [--skip-qm]

Options:
  --config CONFIG   Path to pipeline configuration file
  --input INPUT     Path to input CSV file with SMILES data
  --skip-qm         Skip quantum mechanical calculations (for faster testing)
  --skip-graphs     Skip graph generation
  --resume          Resume from the last successful stage
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import pipeline components
from code.integration.orchestration.pfas_pipeline_orchestrator import PFASPipelineOrchestrator

# Default paths
DEFAULT_CONFIG = os.path.join(project_root, "config", "pipeline_config.json")
DEFAULT_INPUT = os.path.join(project_root, "data", "raw", "pfas_sample.csv")

# Sample PFAS data for demo purposes
SAMPLE_DATA = [
    {
        "SMILES": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "common_name": "PFOA",
        "cas_rn": "335-67-1"
    },
    {
        "SMILES": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "common_name": "PFHxA",
        "cas_rn": "307-24-4"
    },
    {
        "SMILES": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "common_name": "PFBA",
        "cas_rn": "375-22-4"
    },
    {
        "SMILES": "FC(F)(F)S(=O)(=O)[O-].[Na+]",
        "common_name": "PFBS",
        "cas_rn": "375-73-5"
    },
    {
        "SMILES": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-].[Na+]",
        "common_name": "PFHpS",
        "cas_rn": "375-92-8"
    }
]

def create_default_config():
    """Create default pipeline configuration."""
    config = {
        "data_dir": os.path.join(project_root, "data"),
        "output_dir": os.path.join(project_root, "output"),
        "working_dir": os.path.join(project_root, "working"),
        "parallel": {
            "enabled": True,
            "max_workers": 4
        },
        "qm": {
            "functional": "B3LYP",
            "basis_set": "6-31G*",
            "num_procs": 2,
            "memory": 4000
        },
        "graph": {
            "charge_type": "mulliken",
            "use_pfas_features": True,
            "use_quantum_properties": True
        },
        "execution": {
            "skip_qm": False,
            "skip_graph_generation": False,
            "force_rerun": False,
            "cache_intermediates": True
        }
    }
    
    # Ensure config directory exists
    os.makedirs(os.path.dirname(DEFAULT_CONFIG), exist_ok=True)
    
    # Save config to file
    with open(DEFAULT_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)
    
    return DEFAULT_CONFIG

def create_sample_dataset():
    """Create a sample PFAS dataset for demonstration."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DEFAULT_INPUT), exist_ok=True)
    
    # Check if sample dataset already exists
    if os.path.exists(DEFAULT_INPUT):
        return DEFAULT_INPUT
    
    # Create sample dataset
    import pandas as pd
    df = pd.DataFrame(SAMPLE_DATA)
    df.to_csv(DEFAULT_INPUT, index=False)
    
    return DEFAULT_INPUT

def setup_environment():
    """Ensure required directories and files exist."""
    # Create required directories
    for dir_path in ["data", "data/raw", "data/processed", "output", "working", "config"]:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)
    
    # Create sample dataset if it doesn't exist
    input_file = create_sample_dataset()
    
    # Create default config if it doesn't exist
    config_file = DEFAULT_CONFIG
    if not os.path.exists(config_file):
        config_file = create_default_config()
    
    return config_file, input_file

def run_pipeline(config_file=None, input_file=None, skip_qm=False, skip_graphs=False, resume=False):
    """Run the PFAS analysis pipeline."""
    start_time = time.time()
    
    if not config_file or not os.path.exists(config_file):
        config_file, _ = setup_environment()
    
    if not input_file or not os.path.exists(input_file):
        _, input_file = setup_environment()
    
    print(f"{'='*80}")
    print(f"MoML-CA: PFAS Analysis Pipeline")
    print(f"{'='*80}")
    print(f"Configuration: {config_file}")
    print(f"Input dataset: {input_file}")
    print(f"Skip QM calculations: {skip_qm}")
    print(f"Skip graph generation: {skip_graphs}")
    print(f"Resume from last stage: {resume}")
    print(f"{'='*80}\n")
    
    # Initialize pipeline orchestrator
    orchestrator = PFASPipelineOrchestrator(config_file=config_file)
    
    # Configure pipeline
    if skip_qm:
        orchestrator.config["execution"]["skip_qm"] = True
    if skip_graphs:
        orchestrator.config["execution"]["skip_graph_generation"] = True
    
    try:
        # Run pipeline
        if resume:
            print("Resuming pipeline from last successful stage...")
            results = orchestrator.resume_pipeline(input_file)
        else:
            print("Running full pipeline...")
            results = orchestrator.run_full_pipeline(input_file)
        
        # Print results
        print("\n" + "="*80)
        print("PIPELINE RESULTS")
        print("="*80)
        print(f"Molecules processed: {results['molecules_processed']}")
        print(f"Valid SMILES: {results['valid_molecules']}")
        print(f"ORCA calculations successful: {results['orca_success']}")
        print(f"ORCA calculations failed: {results['orca_errors']}")
        print(f"Molecular graphs generated: {results['graphs_generated']}")
        print(f"Total execution time: {results['execution_time']:.2f} seconds")
        
        print("\n✅ Pipeline completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        total_time = time.time() - start_time
        print(f"\nTotal script execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PFAS Analysis Pipeline")
    parser.add_argument("--config", help="Path to pipeline configuration file")
    parser.add_argument("--input", help="Path to input CSV file with SMILES data")
    parser.add_argument("--skip-qm", action="store_true", help="Skip quantum mechanical calculations")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph generation")
    parser.add_argument("--resume", action="store_true", help="Resume from the last successful stage")
    
    args = parser.parse_args()
    
    sys.exit(run_pipeline(args.config, args.input, args.skip_qm, args.skip_graphs, args.resume)) 