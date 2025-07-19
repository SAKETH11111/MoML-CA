#!/bin/bash
# scripts/run_joint_training.sh
# 
# Complete workflow for joint DJMGNN and HMGNN training
# This script implements the pipeline described in the implementation plan

set -e  # Exit on any error

# Configuration
DATA_DIR="data"
OUTPUT_DIR="output_joint"
CHECKPOINT_DIR="checkpoints_joint"
CONFIG_FILE="config/joint_training.yaml"
LOG_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create directories
log "Setting up directories..."
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# Check if configuration exists
if [ ! -f "$CONFIG_FILE" ]; then
    error "Configuration file not found: $CONFIG_FILE"
    error "Please create the configuration file or update the CONFIG_FILE variable"
    exit 1
fi

# Check Python environment
log "Checking Python environment..."
if ! python -c "import torch, torch_geometric, yaml" 2>/dev/null; then
    error "Required Python packages not found. Please install:"
    error "  pip install torch torch-geometric pyyaml tqdm"
    exit 1
fi

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    log "Activating conda environment: $CONDA_ENV"
    source activate "$CONDA_ENV"
fi

# Stage 1: Data Preparation and Preprocessing
log "=== STAGE 1: DATA PREPARATION ==="

if [ ! -d "$DATA_DIR/qm9" ]; then
    log "Downloading and preparing QM9 dataset..."
    python -c "
from moml.data.dataset import get_dataset
from torchvision.transforms import Compose
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets

transform = Compose([
    CreateEdges(),
    FeaturizeNodes(),
    StandardizeTargets(dataset_name='qm9')
])

dataset = get_dataset('qm9', root='$DATA_DIR', transform=transform)
print(f'QM9 dataset prepared: {len(dataset)} molecules')
"
    success "QM9 dataset prepared"
else
    log "QM9 dataset already exists, skipping download"
fi

# Generate hierarchical representations
log "Preprocessing hierarchical graph representations..."
python -c "
import yaml
from moml.core.hierarchical_processor import create_hierarchical_processor

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

processor = create_hierarchical_processor(config['hierarchical'])
print('Hierarchical processor initialized successfully')
" 2>&1 | tee "$LOG_DIR/preprocessing.log"

success "Data preprocessing completed"

# Stage 2: Pre-training (Optional)
log "=== STAGE 2: PRE-TRAINING ==="

# Check if pre-training is enabled in config
PRETRAIN_ENABLED=$(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('pretrain_djmgnn', True) or config.get('pretrain_hmgnn', True))
")

if [ "$PRETRAIN_ENABLED" = "True" ]; then
    log "Running pre-training phase..."
    
    # Pre-train DJMGNN
    log "Pre-training DJMGNN..."
    python scripts/train_alternating.py \
        --config_path "$CONFIG_FILE" \
        --max_steps 1000 \
        --checkpoint_dir "$CHECKPOINT_DIR/djmgnn_pretrain" \
        --log_every 50 \
        --device auto \
        2>&1 | tee "$LOG_DIR/djmgnn_pretrain.log"
    
    if [ $? -eq 0 ]; then
        success "DJMGNN pre-training completed"
    else
        warning "DJMGNN pre-training failed or skipped"
    fi
    
    # Pre-train HMGNN
    log "Pre-training HMGNN..."
    python scripts/train_hmgnn_standalone.py \
        --config "$CONFIG_FILE" \
        --epochs 10 \
        --checkpoint_dir "$CHECKPOINT_DIR/hmgnn_pretrain" \
        --log_every 5 \
        --device auto \
        2>&1 | tee "$LOG_DIR/hmgnn_pretrain.log"
    
    if [ $? -eq 0 ]; then
        success "HMGNN pre-training completed"
    else
        warning "HMGNN pre-training failed or skipped"
    fi
    
else
    log "Pre-training disabled in configuration, skipping..."
fi

# Stage 3: Joint Training
log "=== STAGE 3: JOINT TRAINING ==="

log "Starting joint DJMGNN and HMGNN training..."

# Find pre-trained checkpoints if they exist
DJMGNN_CHECKPOINT=""
HMGNN_CHECKPOINT=""

if [ -d "$CHECKPOINT_DIR/djmgnn_pretrain" ]; then
    DJMGNN_CHECKPOINT=$(find "$CHECKPOINT_DIR/djmgnn_pretrain" -name "*.pt" -type f | head -1)
    if [ ! -z "$DJMGNN_CHECKPOINT" ]; then
        log "Found DJMGNN checkpoint: $DJMGNN_CHECKPOINT"
    fi
fi

if [ -d "$CHECKPOINT_DIR/hmgnn_pretrain" ]; then
    HMGNN_CHECKPOINT=$(find "$CHECKPOINT_DIR/hmgnn_pretrain" -name "*.pt" -type f | head -1)
    if [ ! -z "$HMGNN_CHECKPOINT" ]; then
        log "Found HMGNN checkpoint: $HMGNN_CHECKPOINT"
    fi
fi

# Build joint training command
JOINT_CMD="python scripts/train_joint_mgnn.py \
    --config '$CONFIG_FILE' \
    --phase joint \
    --dataset qm9 \
    --data_root '$DATA_DIR' \
    --checkpoint_dir '$CHECKPOINT_DIR' \
    --output_dir '$OUTPUT_DIR' \
    --training_strategy joint \
    --log_level INFO"

# Add checkpoint arguments if available
if [ ! -z "$DJMGNN_CHECKPOINT" ]; then
    JOINT_CMD="$JOINT_CMD --djmgnn_checkpoint '$DJMGNN_CHECKPOINT'"
fi

if [ ! -z "$HMGNN_CHECKPOINT" ]; then
    JOINT_CMD="$JOINT_CMD --hmgnn_checkpoint '$HMGNN_CHECKPOINT'"
fi

# Execute joint training
log "Executing: $JOINT_CMD"
eval $JOINT_CMD 2>&1 | tee "$LOG_DIR/joint_training.log"

if [ $? -eq 0 ]; then
    success "Joint training completed successfully"
else
    error "Joint training failed"
    exit 1
fi

# Stage 4: Evaluation and Testing
log "=== STAGE 4: EVALUATION ==="

log "Running model evaluation..."
python -c "
import torch
import yaml
from moml.models.mgnn import JointMGNN, create_joint_mgnn

# Load configuration
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Find latest checkpoint
import os
import glob
checkpoint_pattern = '$CHECKPOINT_DIR/joint_mgnn_*.pt'
checkpoints = glob.glob(checkpoint_pattern)

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f'Found joint model checkpoint: {latest_checkpoint}')
    
    # Load model
    joint_model = create_joint_mgnn(
        djmgnn_config=config['djmgnn'],
        hmgnn_config=config['hmgnn'],
        joint_config=config.get('joint', {})
    )
    
    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        joint_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        joint_model.load_state_dict(checkpoint)
    
    print('Model loaded successfully')
    print(f'Total parameters: {sum(p.numel() for p in joint_model.parameters()):,}')
    
    # Quick evaluation
    joint_model.eval()
    print('Model evaluation passed')
    
else:
    print('No joint model checkpoints found')
" 2>&1 | tee "$LOG_DIR/evaluation.log"

# Stage 5: Results Summary
log "=== STAGE 5: RESULTS SUMMARY ==="

# Generate training summary
log "Generating training summary..."
python -c "
import os
import glob

print('Joint DJMGNN and HMGNN Training Summary')
print('=' * 50)

# Check for log files
log_files = {
    'Preprocessing': '$LOG_DIR/preprocessing.log',
    'DJMGNN Pre-training': '$LOG_DIR/djmgnn_pretrain.log',
    'HMGNN Pre-training': '$LOG_DIR/hmgnn_pretrain.log',
    'Joint Training': '$LOG_DIR/joint_training.log',
    'Evaluation': '$LOG_DIR/evaluation.log'
}

for phase, log_file in log_files.items():
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f'{phase}: ✓ (log: {size} bytes)')
    else:
        print(f'{phase}: ✗ (no log found)')

print()

# Check for checkpoints
checkpoint_dirs = [
    '$CHECKPOINT_DIR/djmgnn_pretrain',
    '$CHECKPOINT_DIR/hmgnn_pretrain',
    '$CHECKPOINT_DIR'
]

print('Checkpoints:')
for checkpoint_dir in checkpoint_dirs:
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        if checkpoints:
            print(f'  {checkpoint_dir}: {len(checkpoints)} checkpoint(s)')
        else:
            print(f'  {checkpoint_dir}: no checkpoints')
    else:
        print(f'  {checkpoint_dir}: directory not found')

print()

# Check output directory
if os.path.exists('$OUTPUT_DIR'):
    output_files = os.listdir('$OUTPUT_DIR')
    print(f'Output files: {len(output_files)} file(s) in $OUTPUT_DIR')
else:
    print('Output directory not found')
"

success "Joint training pipeline completed successfully!"

# Final instructions
log "=== NEXT STEPS ==="
echo ""
echo "Training pipeline completed. You can now:"
echo "1. Check training logs in: $LOG_DIR/"
echo "2. Find model checkpoints in: $CHECKPOINT_DIR/"
echo "3. View outputs in: $OUTPUT_DIR/"
echo "4. Run evaluation scripts to analyze performance"
echo "5. Use the trained joint model for predictions"
echo ""
echo "For PFAS-specific fine-tuning:"
echo "  python scripts/train_joint_mgnn.py --phase fine_tuning --config $CONFIG_FILE"
echo ""
echo "For model inference:"
echo "  python examples/joint_training_example.py"

log "Pipeline execution completed at $(date)"