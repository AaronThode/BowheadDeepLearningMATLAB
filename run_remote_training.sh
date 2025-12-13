#!/bin/bash
# Training script for garibaldi.ucsd.edu
# Run this on the remote server after SSH

set -e

# Choose your dataset (uncomment the one you want):
DATASET="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_With_Airguns.dir"
# DATASET="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_No_Airguns.dir"
# DATASET="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_MostlyManual_100K.dir"

# Activate virtual environment
source /home/oboulais/BowheadDeepLearningMATLAB/Pytorch_scripts/.venv_py31018/bin/activate

# Go to script directory
cd /home/oboulais/BowheadDeepLearningMATLAB/Pytorch_scripts

# Run training
python Autoencoder_v02_20251118.py \
  --data-dir "$DATASET" \
  --epochs 100 \
  --latent-dim 32 \
  --channels 64 \
  --batch-size 32 \
  --seed 42

echo ""
echo "Training complete! Results saved in:"
echo "  /home/oboulais/BowheadDeepLearningMATLAB/results/"
