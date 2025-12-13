# Combined Dataset Training Guide

## Overview

The `Autoencoder_v02_20251118.py` script has been modified to load and train on multiple datasets simultaneously using PyTorch's `ConcatDataset`.

## Key Changes

### 1. **Import Addition**
- Added `ConcatDataset` from `torch.utils.data`

### 2. **Dataset Loading Logic**
The script now supports both single and multiple directories:

**Single Directory (backward compatible):**
```python
python Autoencoder_v02_20251118.py --data-dir /path/to/single/dataset
```

**Multiple Directories (new default):**
```python
python Autoencoder_v02_20251118.py --data-dir /path/to/dataset1 /path/to/dataset2
```

### 3. **Default Behavior**
By default, the script now loads BOTH datasets:
- `Unsupervised_database_AutoWithAirguns.dir` (~50K samples)
- `Unsupervised_database_MostlyManual.dir` (~50K samples)
- **Total: ~100K combined samples**

### 4. **Version Tag Updated**
Default version tag changed to: `08_100E_32LD_CombinedDatasets_100K`

## Usage Examples

### Run with Both Datasets (Default)
```bash
python Autoencoder_v02_20251118.py
```

### Run with Single Dataset
```bash
# AutoWithAirguns only
python Autoencoder_v02_20251118.py --data-dir /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir

# MostlyManual only
python Autoencoder_v02_20251118.py --data-dir /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir
```

### Run with Both Datasets (Explicit)
```bash
python Autoencoder_v02_20251118.py \
  --data-dir \
    /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir \
    /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir \
  --epochs 100 \
  --latent-dim 32
```

### Run with Three or More Datasets
```bash
python Autoencoder_v02_20251118.py \
  --data-dir /path/to/dataset1 /path/to/dataset2 /path/to/dataset3
```

## Output

The script will print dataset loading information:

```
Loading data from 2 directories:
  [1] /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir
      Loaded 49963 samples
  [2] /Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir
      Loaded 49970 samples

Total combined samples: 99933
```

## How It Works

1. **SNRDataset**: Each directory is loaded as a separate `SNRDataset` instance
2. **ConcatDataset**: PyTorch's `ConcatDataset` concatenates multiple datasets into one
3. **DataLoader**: The combined dataset is fed to the DataLoader for training
4. **Shuffling**: Individual datasets can be shuffled during loading (controlled by `seed`)
5. **Shape Validation**: Each dataset validates that all samples have the same shape

## Benefits

- ✅ **No Data Duplication**: Datasets remain in separate directories
- ✅ **Memory Efficient**: Samples loaded on-demand from disk
- ✅ **Flexible**: Can use 1, 2, or more datasets
- ✅ **Backward Compatible**: Still works with single directory
- ✅ **Reproducible**: Seed controls shuffling across all datasets

## Training Time Estimates (on GPU)

- **Single dataset (~50K samples)**: ~2 hours for 100 epochs
- **Combined datasets (~100K samples)**: ~4 hours for 100 epochs
- **Per epoch**: ~75-150 seconds depending on dataset size

## Remote Training (garibaldi)

Update your `launch_training.sh` or run directly:

```bash
ssh garibaldi
cd /path/to/BowheadDeepLearningMATLAB/Pytorch_scripts

# Activate environment
source .venv_garibaldi/bin/activate

# Run with both datasets (default)
nohup python Autoencoder_v02_20251118.py --epochs 100 --latent-dim 32 > training.log 2>&1 &

# Monitor
tail -f training.log
```

## File Paths

The two dataset directories are located at:
```
/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/
├── Unsupervised_database_AutoWithAirguns.dir/
│   ├── S310D0T20100815T000050_Type0.mat
│   ├── S310D0T20100815T000247_Type0.mat
│   └── ... (~50K .mat files)
└── Unsupervised_database_MostlyManual.dir/
    ├── S310D0T20100815T011815_Type3.mat
    ├── S310D0T20100815T012124_Type3.mat
    └── ... (~50K .mat files)
```

Each `.mat` file contains a `SNR_gram` field with a 121×104 spectrogram.

## Expected Results

Training on combined datasets should:
- Provide more diverse whale call examples
- Improve generalization of the autoencoder
- Create a more comprehensive latent space representation
- Show both manually-detected and automatically-detected calls

The t-SNE visualization should reveal any systematic differences between the two datasets.
