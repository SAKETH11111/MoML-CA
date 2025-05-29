#!/bin/bash
#SBATCH --job-name=md_run
#SBATCH --output=md_run_%j.out
#SBATCH --error=md_run_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

# Load required modules
module load cuda/12.2
module load openmm/8.0

# Set environment variables
export OPENMM_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc
export OPENMM_CPU_THREADS=1
export OPENMM_DEVICE_INDEX=0

# Activate conda environment
source /path/to/conda/bin/activate moml-ca

# Set up logging
export STRUCTLOG_LEVEL=INFO
export STRUCTLOG_FORMAT=json

# Run MD simulation
python -m moml.simulation.molecular_dynamics.runner \
    --pdb ${PDB_PATH} \
    --ff-params ${FF_PARAMS} \
    --output-dir ${OUTPUT_DIR} \
    --config ${CONFIG_PATH} \
    --checkpoint ${CHECKPOINT_PATH:-} \
    --surface ${SURFACE_NAME:-} \
    --solvent ${SOLVENT_NAME:-}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Simulation completed successfully"
    exit 0
else
    echo "Simulation failed"
    exit 1
fi 