# Remote Training on Garibaldi - Quick Reference

## Setup (One-time)
✅ SSH keys configured - no password needed
✅ Virtual environment (.venv_garibaldi) created on garibaldi
✅ PyTorch and dependencies installed

## Start Training (Close Laptop Safe!)

### Option 1: Use the automated script
```bash
cd /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts
./start_remote_training.sh [epochs] [latent_dim] [channels] [batch_size] [seed]
```

**Examples:**
```bash
./start_remote_training.sh                    # Default: 100 epochs, 32 latent, 64 channels
./start_remote_training.sh 50                 # 50 epochs, other defaults
./start_remote_training.sh 200 32 64 32 42   # Full custom parameters
```

### Option 2: Manual start
```bash
ssh oboulais@garibaldi.ucsd.edu
cd ~/BowheadDeepLearningMATLAB/Pytorch_scripts
source .venv_garibaldi/bin/activate

nohup python -u Autoencoder_v02_20251118.py \
  --data-dir /home/oboulais/BCB_Whale_Datasets/Unsupervised_database_MostlyManual_100K.dir \
  --epochs 100 --latent-dim 32 --channels 64 --batch-size 32 --seed 42 \
  > training.log 2>&1 &

exit  # Close laptop now!
```

## Monitor Training Progress

### Quick check (from your MacBook)
```bash
./monitor_training.sh
```

This shows:
- Whether training is running
- Last 30 lines of log
- Results directory size
- GPU usage

### Watch live updates
```bash
ssh oboulais@garibaldi.ucsd.edu 'tail -f ~/BowheadDeepLearningMATLAB/Pytorch_scripts/training.log'
```
Press `Ctrl+C` to stop watching (training continues)

### Check specific info
```bash
# Just see current epoch
ssh oboulais@garibaldi.ucsd.edu "grep 'Epoch' ~/BowheadDeepLearningMATLAB/Pytorch_scripts/training.log | tail -5"

# See if process is running
ssh oboulais@garibaldi.ucsd.edu "pgrep -af python"

# Check GPU usage
ssh oboulais@garibaldi.ucsd.edu "nvidia-smi"
```

## Download Results

```bash
./download_results.sh
```

Results will be saved to: `/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results_from_garibaldi/`

## Stop Training (if needed)

```bash
ssh oboulais@garibaldi.ucsd.edu "pkill -f 'python.*Autoencoder'"
```

## Training Features

✅ **Autonomous:** Runs without intervention after start
✅ **Laptop-safe:** Close laptop anytime, training continues
✅ **Progress tracking:** Every epoch prints ETA
✅ **Checkpoints:** Saves model every 10 epochs for recovery
✅ **Robust:** Recreates output directories if deleted
✅ **Logged:** All output captured in training.log

## Troubleshooting

### Training not starting?
```bash
ssh oboulais@garibaldi.ucsd.edu
cd ~/BowheadDeepLearningMATLAB/Pytorch_scripts
source .venv_garibaldi/bin/activate
python Autoencoder_v02_20251118.py --help  # Check if script works
```

### Check for errors
```bash
ssh oboulais@garibaldi.ucsd.edu "tail -100 ~/BowheadDeepLearningMATLAB/Pytorch_scripts/training.log | grep -i error"
```

### Disk space issues?
```bash
ssh oboulais@garibaldi.ucsd.edu "df -h ~"
```

## Training Output

Each training run creates a timestamped directory with:
- `autoencoder_clean.pth` - Final trained model
- `checkpoint_epoch*.pth` - Checkpoints every 10 epochs
- `latent_embeddings.mat` - Latent vectors + filenames
- `tsne_latent.png` - t-SNE visualization
- `reconstructions.png` - Sample reconstructions
- `training_loss.png` - Loss curve
- Various reconstruction panels (.jpg)

## Expected Timeline

- **100 epochs on 50K samples:** ~2-4 hours (depends on GPU)
- **Checkpoints:** Every 10 epochs (~12-24 minutes)
- **ETA shown:** After each epoch

---

**Remember:** You can close your laptop immediately after starting training. The process runs autonomously on garibaldi!
