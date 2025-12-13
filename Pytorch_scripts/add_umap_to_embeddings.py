#!/usr/bin/env python3
"""
Add UMAP Embeddings to Existing Latent Embeddings Files
OPTIMIZED FOR 100K SAMPLES - FAST VERSION

This script:
1. Loads existing latent_embeddings.mat files
2. Computes UMAP embeddings with optimized parameters for speed
3. Saves UMAP results back to the same .mat files

Performance Optimizations:
- Uses lower dimensional approximation for initial steps
- Reduced n_epochs for faster convergence
- Optimized n_neighbors for 100K dataset
- Multi-threaded processing when available

Usage:
    python add_umap_to_embeddings.py
"""

import numpy as np
from scipy.io import loadmat, savemat
import umap
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory - MERGED DATASET ONLY
DATASET_DIR = '/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v09_100E_16LD_CombinedDatasets_100K_Date20251209-122650.dir'

# UMAP Parameters - OPTIMIZED FOR SPEED WITH 100K SAMPLES
UMAP_PARAMS = {
    'n_neighbors': 30,           # Lower for speed (15-30 good for large datasets)
    'min_dist': 0.1,             # Standard value
    'n_components': 2,           # 2D output
    'metric': 'euclidean',       # Fastest metric
    'random_state': 42,          # Reproducibility
    'n_epochs': 200,             # Reduced from default (~500 for 100k) for speed
    'learning_rate': 1.0,        # Standard
    'spread': 1.0,               # Standard
    'low_memory': False,         # Use more memory for speed
    'verbose': True,             # Progress updates
    'n_jobs': -1                 # Use all CPU cores
}

print("="*70)
print("FAST UMAP FOR 100K MERGED DATASET")
print("="*70)
print(f"\nDataset: {DATASET_DIR}")
print(f"\nUMAP Parameters (Optimized for Speed):")
for key, value in UMAP_PARAMS.items():
    if key != 'verbose':
        print(f"  {key:20s}: {value}")
print()
print("Speed optimizations:")
print("  • Lower n_neighbors (30 vs 50)")
print("  • Reduced epochs (200 vs ~500 default)")
print("  • Multi-threaded processing (n_jobs=-1)")
print("  • Euclidean metric (fastest)")
print()
print("Expected time: 2-5 minutes for 100K samples")
print("="*70)

# ============================================================================
# PROCESS DATASET
# ============================================================================

mat_file_path = os.path.join(DATASET_DIR, 'latent_embeddings.mat')

# Check if file exists
if not os.path.exists(mat_file_path):
    print(f"\n❌ ERROR: File not found: {mat_file_path}")
    print(f"Please check the directory path.")
    exit(1)

# ========================================================================
# STEP 1: Load existing data
# ========================================================================
print("\nSTEP 1: Loading existing latent embeddings...")
start_load = datetime.now()
data = loadmat(mat_file_path)
load_time = (datetime.now() - start_load).total_seconds()

# Check what's in the file
print(f"  Current contents:")
for key in data.keys():
    if not key.startswith('__'):
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"    {key:25s}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"    {key:25s}: {type(item)}")

# Get latent embeddings
latent_embeddings = data['latent_embeddings']
n_samples = latent_embeddings.shape[0]
n_dims = latent_embeddings.shape[1]

print(f"\n  ✓ Loaded {n_samples:,} samples with {n_dims}D latent vectors in {load_time:.1f}s")
print(f"  Memory footprint: ~{latent_embeddings.nbytes / (1024**2):.1f} MB")

# ========================================================================
# STEP 2: Compute UMAP embeddings
# ========================================================================
print(f"\nSTEP 2: Computing UMAP embeddings...")
print(f"  Processing {n_samples:,} samples...")
print(f"  Using all available CPU cores for parallel processing")
print()

start_time = datetime.now()

# Create UMAP reducer with optimized parameters
reducer = umap.UMAP(**UMAP_PARAMS)

# Fit and transform
umap_embeddings = reducer.fit_transform(latent_embeddings)

elapsed_time = (datetime.now() - start_time).total_seconds()
samples_per_sec = n_samples / elapsed_time

print(f"\n  ✓ UMAP complete in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"  ✓ Processing speed: {samples_per_sec:.0f} samples/second")
print(f"  ✓ Output shape: {umap_embeddings.shape}")
print(f"  ✓ Output range:")
print(f"      X: [{umap_embeddings[:, 0].min():.2f}, {umap_embeddings[:, 0].max():.2f}]")
print(f"      Y: [{umap_embeddings[:, 1].min():.2f}, {umap_embeddings[:, 1].max():.2f}]")

# ========================================================================
# STEP 3: Save updated data
# ========================================================================
print(f"\nSTEP 3: Saving updated file with UMAP results...")

# Add UMAP results to existing data
data['umap_embeddings'] = umap_embeddings
data['umap_n_neighbors'] = UMAP_PARAMS['n_neighbors']
data['umap_min_dist'] = UMAP_PARAMS['min_dist']
data['umap_metric'] = UMAP_PARAMS['metric']
data['umap_n_epochs'] = UMAP_PARAMS['n_epochs']
data['umap_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Save back to same file
start_save = datetime.now()
savemat(mat_file_path, data)
save_time = (datetime.now() - start_save).total_seconds()

file_size_mb = os.path.getsize(mat_file_path) / (1024 * 1024)

print(f"  ✓ Saved to: {mat_file_path}")
print(f"  ✓ File size: {file_size_mb:.1f} MB")
print(f"  ✓ Save time: {save_time:.1f}s")

# ============================================================================
# SUMMARY
# ============================================================================
total_time = load_time + elapsed_time + save_time

print()
print("="*70)
print("✓ COMPLETED SUCCESSFULLY!")
print("="*70)
print()
print(f"Dataset: {os.path.basename(DATASET_DIR)}")
print(f"  Samples:          {n_samples:,}")
print(f"  Latent dims:      {n_dims}D")
print(f"  UMAP output:      {umap_embeddings.shape}")
print()
print(f"Timing:")
print(f"  Load time:        {load_time:.1f}s")
print(f"  UMAP time:        {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print(f"  Save time:        {save_time:.1f}s")
print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
print()
print(f"File updated: latent_embeddings.mat ({file_size_mb:.1f} MB)")
print()
print("The file now contains:")
print(f"  - latent_embeddings:  {n_dims}D latent vectors")
print(f"  - tsne_embeddings:    t-SNE 2D (if present)")
print(f"  - umap_embeddings:    UMAP 2D ✓ NEW")
print(f"  - umap_n_neighbors:   {UMAP_PARAMS['n_neighbors']}")
print(f"  - umap_min_dist:      {UMAP_PARAMS['min_dist']}")
print(f"  - umap_metric:        {UMAP_PARAMS['metric']}")
print(f"  - umap_n_epochs:      {UMAP_PARAMS['n_epochs']}")
print()
print("="*70)
