#!/usr/bin/env python3
"""
Generate UMAP and t-SNE Embeddings for v07 Dataset
OPTIMIZED FOR SPEED - 50K SAMPLES

This script:
1. Loads latent embeddings from v07 model
2. Computes UMAP embeddings (fast, 1-3 min)
3. Computes t-SNE embeddings (slower, 5-10 min)
4. Saves both to a new combined .mat file

Speed optimizations:
- Parallel processing (all CPU cores)
- Reduced iterations for faster convergence
- Optimized hyperparameters for large datasets

Expected total time: 8-15 minutes for 50K samples
"""

import numpy as np
from scipy.io import loadmat, savemat
import umap
from sklearn.manifold import TSNE
import os
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory - v07 AutoWithAirguns 50K
DATASET_DIR = '/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830.dir'
INPUT_FILE = 'latent_embeddings.mat'
OUTPUT_FILE = 'umap_tsne_embeddings.mat'

# UMAP Parameters - Optimized for speed with 50K samples
UMAP_PARAMS = {
    # n_neighbors: Size of local neighborhood (in terms of number of neighboring points)
    #   - Controls balance between local vs global structure
    #   - Small (5-15): Captures fine local details, more clusters
    #   - Medium (30-50): Balanced view, recommended for large datasets
    #   - Large (100+): Emphasizes broad global structure
    #   - Here: 30 = fast computation while preserving meaningful local structure
    'n_neighbors': 30,
    
    # min_dist: Minimum distance between points in the low-dimensional space
    #   - Controls how tightly UMAP packs points together
    #   - 0.0: Points can be arbitrarily close (tight clumps)
    #   - 0.1: Small distances allowed (good balance)
    #   - 0.5-1.0: Forces points apart (more diffuse)
    #   - Here: 0.1 = preserves local structure with reasonable separation
    'min_dist': 0.1,
    
    # n_components: Dimensionality of the output embedding space
    #   - 2: Standard for 2D visualization plots
    #   - 3: For 3D interactive visualizations
    #   - Higher: For downstream ML tasks (e.g., dimensionality reduction before classification)
    'n_components': 2,
    
    # metric: Distance function used to measure similarity in high-dimensional space
    #   - 'euclidean': Standard L2 distance (fastest, most common)
    #   - 'cosine': Angle-based similarity (good for normalized vectors)
    #   - 'manhattan': L1 distance
    #   - 'correlation': Pearson correlation distance
    #   - Here: euclidean = fastest and works well for spectral features
    'metric': 'euclidean',
    
    # random_state: Seed for random number generator
    #   - Ensures reproducible results across runs
    #   - Use same value to get identical embeddings
    'random_state': 42,
    
    # n_epochs: Number of optimization iterations
    #   - None: Auto-determined based on dataset size (~500 for 50k)
    #   - More epochs = better optimization but slower
    #   - Here: 200 = significant speedup (2.5x faster) with minimal quality loss
    'n_epochs': 200,
    
    # learning_rate: Initial learning rate for gradient descent optimization
    #   - Controls step size during optimization
    #   - 1.0 is standard default, works well for most cases
    #   - Higher (>1.0): Faster convergence but may overshoot
    #   - Lower (<1.0): More careful optimization but slower
    'learning_rate': 1.0,
    
    # spread: Effective scale of embedded points
    #   - Works with min_dist to control cluster dispersion
    #   - Determines how "spread out" the final embedding is
    #   - 1.0 is standard default
    #   - Higher: More spread out points
    #   - Lower: More compact embedding
    'spread': 1.0,
    
    # low_memory: Trade-off between memory usage and speed
    #   - False: Use more RAM for faster computation (recommended for speed)
    #   - True: Use less RAM but slower (use if running out of memory)
    #   - Here: False = maximize speed since we have sufficient RAM
    'low_memory': False,
    
    # verbose: Print progress messages during computation
    #   - True: See progress updates
    #   - False: Silent operation
    'verbose': True,
    
    # n_jobs: Number of CPU cores to use for parallel processing
    #   - -1: Use all available CPU cores (maximum speed)
    #   - 1: Single-threaded (slower)
    #   - n: Use n specific cores
    #   - Here: -1 = use all cores for maximum parallelization
    'n_jobs': -1
}

# t-SNE Parameters - Optimized for speed with 50K samples
TSNE_PARAMS = {
    'n_components': 2,           # 2D output
    'perplexity': 30,            # Standard for large datasets
    'learning_rate': 200,        # 'auto' would be ~4166 for 50k, 200 is faster
    'max_iter': 500,             # Reduced from 1000 default for speed
    'random_state': 42,          # Reproducibility
    'verbose': 2,                # Print progress every 50 iterations
    'method': 'barnes_hut',      # Faster than exact for large datasets
    'angle': 0.5,                # Speed vs accuracy tradeoff (0.2-0.8)
    'n_jobs': -1                 # Use all CPU cores (scikit-learn >=1.0)
}

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("="*80)
print("GENERATE UMAP AND t-SNE EMBEDDINGS FOR v07 DATASET")
print("="*80)
print(f"\nDataset: {os.path.basename(DATASET_DIR)}")
print(f"Input:   {INPUT_FILE}")
print(f"Output:  {OUTPUT_FILE}")
print()

# ============================================================================
# STEP 1: Load existing data
# ============================================================================
print("STEP 1: Loading latent embeddings...")
print("-" * 80)

input_path = os.path.join(DATASET_DIR, INPUT_FILE)
if not os.path.exists(input_path):
    print(f"\n❌ ERROR: File not found: {input_path}")
    exit(1)

start_load = datetime.now()
data = loadmat(input_path)
load_time = (datetime.now() - start_load).total_seconds()

print(f"\nCurrent file contents:")
for key in data.keys():
    if not key.startswith('__'):
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"  {key:25s}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key:25s}: {type(item)}")

# Get latent embeddings
latent_embeddings = data['latent_embeddings']
n_samples = latent_embeddings.shape[0]
n_dims = latent_embeddings.shape[1]

print(f"\n✓ Loaded {n_samples:,} samples with {n_dims}D latent vectors in {load_time:.1f}s")
print(f"  Memory footprint: ~{latent_embeddings.nbytes / (1024**2):.1f} MB")
print()

# ============================================================================
# STEP 2: Compute UMAP embeddings
# ============================================================================
print("STEP 2: Computing UMAP embeddings...")
print("-" * 80)
print(f"\nUMAP Parameters (optimized for speed):")
for key, value in UMAP_PARAMS.items():
    if key not in ['verbose', 'low_memory']:
        print(f"  {key:20s}: {value}")
print()
print(f"Processing {n_samples:,} samples with all CPU cores...")
print("Expected time: 1-3 minutes")
print()

start_umap = datetime.now()

# Create UMAP reducer
reducer = umap.UMAP(**UMAP_PARAMS)

# Fit and transform
umap_embeddings = reducer.fit_transform(latent_embeddings)

umap_time = (datetime.now() - start_umap).total_seconds()
samples_per_sec = n_samples / umap_time

print(f"\n✓ UMAP complete in {umap_time:.1f}s ({umap_time/60:.1f} min)")
print(f"  Processing speed: {samples_per_sec:.0f} samples/second")
print(f"  Output shape: {umap_embeddings.shape}")
print(f"  Output range:")
print(f"    X: [{umap_embeddings[:, 0].min():.2f}, {umap_embeddings[:, 0].max():.2f}]")
print(f"    Y: [{umap_embeddings[:, 1].min():.2f}, {umap_embeddings[:, 1].max():.2f}]")
print()

# ============================================================================
# STEP 3: Compute t-SNE embeddings
# ============================================================================
print("STEP 3: Computing t-SNE embeddings...")
print("-" * 80)
print(f"\nt-SNE Parameters (optimized for speed):")
for key, value in TSNE_PARAMS.items():
    if key not in ['verbose']:
        print(f"  {key:20s}: {value}")
print()
print(f"Processing {n_samples:,} samples with Barnes-Hut approximation...")
print("Expected time: 5-10 minutes")
print("Progress will be printed every 50 iterations...")
print()

start_tsne = datetime.now()

# Create t-SNE reducer
tsne_reducer = TSNE(**TSNE_PARAMS)

# Fit and transform
tsne_embeddings = tsne_reducer.fit_transform(latent_embeddings)

tsne_time = (datetime.now() - start_tsne).total_seconds()
samples_per_sec_tsne = n_samples / tsne_time

print(f"\n✓ t-SNE complete in {tsne_time:.1f}s ({tsne_time/60:.1f} min)")
print(f"  Processing speed: {samples_per_sec_tsne:.0f} samples/second")
print(f"  Output shape: {tsne_embeddings.shape}")
print(f"  Output range:")
print(f"    X: [{tsne_embeddings[:, 0].min():.2f}, {tsne_embeddings[:, 0].max():.2f}]")
print(f"    Y: [{tsne_embeddings[:, 1].min():.2f}, {tsne_embeddings[:, 1].max():.2f}]")
print()

# ============================================================================
# STEP 4: Save combined results
# ============================================================================
print("STEP 4: Saving combined UMAP + t-SNE results...")
print("-" * 80)

output_path = os.path.join(DATASET_DIR, OUTPUT_FILE)

# Create output dictionary with both embeddings
output_data = {
    # Original data
    'latent_embeddings': latent_embeddings,
    
    # UMAP results
    'umap_embeddings': umap_embeddings,
    'umap_n_neighbors': UMAP_PARAMS['n_neighbors'],
    'umap_min_dist': UMAP_PARAMS['min_dist'],
    'umap_metric': UMAP_PARAMS['metric'],
    'umap_n_epochs': UMAP_PARAMS['n_epochs'],
    'umap_computation_time': umap_time,
    
    # t-SNE results
    'tsne_embeddings': tsne_embeddings,
    'tsne_perplexity': TSNE_PARAMS['perplexity'],
    'tsne_learning_rate': TSNE_PARAMS['learning_rate'],
    'tsne_max_iter': TSNE_PARAMS['max_iter'],
    'tsne_method': TSNE_PARAMS['method'],
    'tsne_computation_time': tsne_time,
    
    # Metadata
    'n_samples': n_samples,
    'n_latent_dims': n_dims,
    'dataset_name': 'v07_AutoWithAirguns_50K',
    'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save to file
start_save = datetime.now()
savemat(output_path, output_data)
save_time = (datetime.now() - start_save).total_seconds()

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

print(f"\n✓ Saved to: {output_path}")
print(f"  File size: {file_size_mb:.1f} MB")
print(f"  Save time: {save_time:.1f}s")
print()

# ============================================================================
# SUMMARY
# ============================================================================
total_time = load_time + umap_time + tsne_time + save_time

print("="*80)
print("✓ COMPLETED SUCCESSFULLY!")
print("="*80)
print()
print(f"Dataset: {os.path.basename(DATASET_DIR)}")
print(f"  Samples:          {n_samples:,}")
print(f"  Latent dims:      {n_dims}D")
print()
print(f"Results:")
print(f"  UMAP output:      {umap_embeddings.shape}")
print(f"  t-SNE output:     {tsne_embeddings.shape}")
print()
print(f"Timing breakdown:")
print(f"  Load time:        {load_time:.1f}s")
print(f"  UMAP time:        {umap_time:.1f}s ({umap_time/60:.1f} min)")
print(f"  t-SNE time:       {tsne_time:.1f}s ({tsne_time/60:.1f} min)")
print(f"  Save time:        {save_time:.1f}s")
print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
print()
print(f"Output file: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
print()
print("File contents:")
print(f"  ✓ latent_embeddings:       {n_dims}D vectors ({n_samples:,} samples)")
print(f"  ✓ umap_embeddings:         2D projection")
print(f"  ✓ tsne_embeddings:         2D projection")
print(f"  ✓ umap_* parameters:       UMAP configuration")
print(f"  ✓ tsne_* parameters:       t-SNE configuration")
print(f"  ✓ computation times:       Performance metrics")
print(f"  ✓ metadata:                Dataset info & timestamp")
print()
print("="*80)
print()
print("Next steps:")
print("  • Use this file for visualization and analysis")
print("  • Compare UMAP vs t-SNE for your data")
print("  • Adjust parameters if needed and re-run")
print()
print("="*80)
