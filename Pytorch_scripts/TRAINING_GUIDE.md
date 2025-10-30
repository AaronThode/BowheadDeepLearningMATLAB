# Fresh Autoencoder Training Guide

## Quick Start

1. **Use the new improved script:**
```bash
cd Pytorch_scripts
python Bowhead_Train_Autoencoder_Fresh.py --epochs 20 --data-dir /path/to/your/spectrograms
```

2. **Or use the updated original script:**
```bash
python Bowhead_Train_Autoencoder.py --epochs 20 --data-dir /path/to/your/spectrograms
```

## Key Improvements Made

### Architecture Changes
- **Removed sigmoid activation**: No longer constrains outputs to [0,1]
- **Increased latent dimension**: 64D instead of 16D for better representation
- **More channels**: 128 max channels instead of 16 for better capacity
- **Added batch normalization**: Stabilizes training
- **Added dropout**: Prevents overfitting

### Training Improvements
- **Better loss functions**: Option for L1 loss (`--l1-loss`)
- **Learning rate scheduling**: Reduces LR when validation loss plateaus
- **Data normalization**: Optional normalization (`--normalize`)
- **Early stopping**: Saves best model based on validation loss
- **Better monitoring**: More detailed TensorBoard logging

## Command Line Options

```bash
python Bowhead_Train_Autoencoder_Fresh.py \
  --data-dir /path/to/mat/files \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --latent-dim 64 \
  --normalize \
  --l1-loss
```

## What to Expect

### Before (Original Model)
- Blurry reconstructions due to sigmoid bottleneck
- Poor fine detail preservation
- Limited representation capacity (16D latent)

### After (Improved Model)
- Sharp reconstructions with unbounded outputs
- Better fine detail preservation
- Richer representation capacity (64D latent)
- Stable training with batch normalization

## Troubleshooting

1. **"No .mat files found"**: Update `--data-dir` to point to your spectrogram folder
2. **Memory errors**: Reduce `--batch-size` to 16 or 8
3. **Slow training**: Reduce `--epochs` for quick tests
4. **Poor reconstruction**: Try `--normalize` and `--l1-loss` flags

## Output Files

- `improved_*_best_model.pth`: Best model based on validation loss
- `improved_*_final_model.pth`: Final model after all epochs
- `improved_*_training_plot.png`: Loss curves
- `improved_*_final_comparison.png`: Reconstruction examples

## TensorBoard Monitoring

```bash
# From repo root
tensorboard --logdir runs
# Open http://localhost:6006
```

View:
- **Scalars**: Training/validation loss curves
- **Images**: Periodic reconstruction comparisons
- **Text**: Configuration details