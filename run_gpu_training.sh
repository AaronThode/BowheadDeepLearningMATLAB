#!/bin/bash
# Quick GPU training launcher for garibaldi
# Usage: ./run_gpu_training.sh [epochs] [latent_dim] [channels]

# Default parameters
EPOCHS=${1:-100}
LATENT_DIM=${2:-32}
CHANNELS=${3:-32}
BATCH_SIZE=${4:-64}
SEED=${5:-42}

# Remote settings
REMOTE_USER="oboulais"
REMOTE_HOST="garibaldi.ucsd.edu"
REMOTE_DIR="~/BowheadDeepLearningMATLAB"
VENV_NAME=".venv_garibaldi"

# Dataset on garibaldi (combined dataset with 99,933 samples)
DATASET_AUTO="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir"
DATASET_MANUAL="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir"

echo "========================================================================="
echo "GPU TRAINING ON GARIBALDI"
echo "========================================================================="
echo "Parameters:"
echo "  Epochs:      $EPOCHS"
echo "  Latent Dim:  $LATENT_DIM"
echo "  Channels:    $CHANNELS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  Seed:        $SEED"
echo ""
echo "Dataset: Combined (99,933 samples)"
echo "  - AutoWithAirguns: ~49,963 samples"
echo "  - MostlyManual:    ~49,970 samples"
echo "========================================================================="
echo ""

# Step 1: Sync code to remote
echo "Step 1: Syncing code to garibaldi..."
rsync -avz --exclude '.venv*' --exclude 'results/' --exclude 'models/' --exclude '__pycache__/' \
  Pytorch_scripts/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/Pytorch_scripts/

if [ $? -ne 0 ]; then
    echo "❌ Failed to sync code"
    exit 1
fi
echo "✓ Code synced"
echo ""

# Step 2: Start training on remote with GPU
echo "Step 2: Starting training on garibaldi (with GPU)..."
echo "This will run in background - you can close your laptop!"
echo ""

ssh ${REMOTE_USER}@${REMOTE_HOST} << ENDSSH
cd ${REMOTE_DIR}/Pytorch_scripts
source ${VENV_NAME}/bin/activate

# Check GPU availability
echo "Checking GPU..."
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Kill any existing training
pkill -f "Autoencoder_v02"

# Start new training with nohup
echo "Starting training process..."
nohup python -u Autoencoder_v02_20251118.py \\
  --data-dir ${DATASET_AUTO} ${DATASET_MANUAL} \\
  --epochs ${EPOCHS} \\
  --latent-dim ${LATENT_DIM} \\
  --channels ${CHANNELS} \\
  --batch-size ${BATCH_SIZE} \\
  --seed ${SEED} \\
  --output-samples 30 \\
  --tsne-samples 100000 \\
  > training_gpu_\$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started! PID: \$!"
echo ""
echo "Monitor with:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  tail -f ${REMOTE_DIR}/Pytorch_scripts/training_gpu_*.log"
echo ""
ENDSSH

echo ""
echo "========================================================================="
echo "✓ TRAINING LAUNCHED ON GPU"
echo "========================================================================="
echo ""
echo "Monitor progress:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}/Pytorch_scripts"
echo "  tail -f training_gpu_*.log"
echo ""
echo "Check GPU usage:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'nvidia-smi'"
echo ""
echo "Stop training:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'pkill -f Autoencoder_v02'"
echo ""
echo "========================================================================="
