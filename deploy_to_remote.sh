#!/bin/bash
# Deployment script for remote GPU training
# Usage: ./deploy_to_remote.sh username@remote-server.edu /path/to/remote/dataset

set -e

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <remote-user@host> <remote-dataset-path>"
    echo "Example: $0 username@gpu.server.edu /data/whale_spectrograms"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_DATASET=$2
REMOTE_DIR="~/autoencoder_training"

echo "=========================================="
echo "Deploying Autoencoder to Remote GPU"
echo "=========================================="
echo "Remote host: $REMOTE_HOST"
echo "Dataset path: $REMOTE_DATASET"
echo ""

# Step 1: Create remote directory
echo "Step 1: Creating remote directory..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Step 2: Copy training script
echo "Step 2: Copying training script..."
scp Pytorch_scripts/Autoencoder_v02_20251118.py $REMOTE_HOST:$REMOTE_DIR/

# Step 3: Copy requirements
echo "Step 3: Copying requirements.txt..."
scp requirements.txt $REMOTE_HOST:$REMOTE_DIR/

# Step 4: Copy helper scripts
echo "Step 4: Copying helper scripts..."
scp Pytorch_scripts/replot_reconstructions.py $REMOTE_HOST:$REMOTE_DIR/ 2>/dev/null || echo "Skipping replot script"
scp Pytorch_scripts/map_embeddings_to_files.py $REMOTE_HOST:$REMOTE_DIR/ 2>/dev/null || echo "Skipping mapping script"

# Step 5: Create setup script on remote
echo "Step 5: Creating remote setup script..."
cat > /tmp/remote_setup.sh << 'SETUP_EOF'
#!/bin/bash
set -e

echo "Setting up Python environment..."

# Check if conda exists
if command -v conda &> /dev/null; then
    echo "Using conda..."
    conda create -n autoencoder python=3.10 -y || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate autoencoder
    pip install -r requirements.txt
elif command -v python3 &> /dev/null; then
    echo "Using venv..."
    python3 -m venv venv_autoencoder
    source venv_autoencoder/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "ERROR: Neither conda nor python3 found!"
    exit 1
fi

echo "Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate autoencoder  (if using conda)"
echo "  source venv_autoencoder/bin/activate  (if using venv)"
SETUP_EOF

scp /tmp/remote_setup.sh $REMOTE_HOST:$REMOTE_DIR/
ssh $REMOTE_HOST "chmod +x $REMOTE_DIR/remote_setup.sh"

# Step 6: Create training launch script
echo "Step 6: Creating training launch script..."
cat > /tmp/run_training.sh << TRAIN_EOF
#!/bin/bash
# Training launch script

# Activate environment
if command -v conda &> /dev/null; then
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate autoencoder
else
    source venv_autoencoder/bin/activate
fi

# Run training
python Autoencoder_v02_20251118.py \\
  --data-dir $REMOTE_DATASET \\
  --epochs 100 \\
  --latent-dim 32 \\
  --channels 64 \\
  --batch-size 32 \\
  --seed 42

echo "Training complete! Results saved in results/"
TRAIN_EOF

scp /tmp/run_training.sh $REMOTE_HOST:$REMOTE_DIR/
ssh $REMOTE_HOST "chmod +x $REMOTE_DIR/run_training.sh"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to remote server:"
echo "   ssh $REMOTE_HOST"
echo ""
echo "2. Navigate to project directory:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Set up Python environment (first time only):"
echo "   ./remote_setup.sh"
echo ""
echo "4. Run training:"
echo "   ./run_training.sh"
echo ""
echo "5. Or run in background with nohup:"
echo "   nohup ./run_training.sh > training.log 2>&1 &"
echo "   tail -f training.log"
echo ""
echo "6. Download results when done (from local machine):"
echo "   scp -r $REMOTE_HOST:$REMOTE_DIR/results/ ./results_from_remote/"
echo ""
