#!/usr/bin/env python3
"""
Generate Reconstructed Spectrograms for All Samples

This script:
1. Loads the trained autoencoder model
2. Processes all input spectrograms through the network
3. Saves reconstructed spectrograms as .mat files
4. Filenames: original_name_reconstr.mat

Output Format Options:
- MATLAB .mat format (default): Compatible with MATLAB, includes metadata
- Python .npy format: Fast, native NumPy format
- Both formats: Save in both .mat and .npy

Usage:
    # For CombinedDatasets (100K samples)
    python generate_all_reconstructions.py --model v08 --format mat
    
    # For MostlyManual (50K samples)
    python generate_all_reconstructions.py --model v06 --format mat
    
    # For AutoWithAirguns (50K samples)
    python generate_all_reconstructions.py --model v07 --format mat
    
    # Save both .mat and .npy
    python generate_all_reconstructions.py --model v08 --format both
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
import os
import glob
import argparse
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

# ============================================================================
# AUTOENCODER ARCHITECTURE (must match training)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Autoencoder architecture - must match training exactly"""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
                 base_channels=64, extra_conv=False):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        
        if extra_conv:
            nrow_reduced = nrow // 16
            ncol_reduced = ncol // 16
        else:
            nrow_reduced = nrow // 8
            ncol_reduced = ncol // 8
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c3, c4, 3, padding=1),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            flat_size = c4 * nrow_reduced * ncol_reduced
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            flat_size = c3 * nrow_reduced * ncol_reduced
        
        self.to_latent = nn.Sequential(
            nn.Linear(flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, flat_size),
            nn.ReLU(inplace=True)
        )
        
        if extra_conv:
            pad_h = (nrow - nrow_reduced * 16) % 2
            pad_w = (ncol - ncol_reduced * 16) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, stride=2),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        else:
            pad_h = (nrow - nrow_reduced * 8) % 2
            pad_w = (ncol - ncol_reduced * 8) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


def match_shape_center(recon: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    """Center-crop or pad reconstruction to match target dimensions."""
    _, _, rH, rW = recon.shape
    tH, tW = target_hw
    if rH > tH:
        dh = (rH - tH) // 2
        recon = recon[:, :, dh:dh + tH, :]
        rH = tH
    if rW > tW:
        dw = (rW - tW) // 2
        recon = recon[:, :, :, dw:dw + tW]
        rW = tW
    padH = tH - rH
    padW = tW - rW
    if padH > 0 or padW > 0:
        pad_left = padW // 2 if padW > 0 else 0
        pad_right = padW - pad_left if padW > 0 else 0
        pad_top = padH // 2 if padH > 0 else 0
        pad_bottom = padH - pad_top if padH > 0 else 0
        recon = F.pad(recon, (pad_left, pad_right, pad_top, pad_bottom))
    return recon


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODELS = {
    'v06': {
        'name': 'MostlyManual_50K',
        'results_dir': 'Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    },
    'v07': {
        'name': 'AutoWithAirguns_50K',
        'results_dir': 'Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    },
    'v08': {
        'name': 'CombinedDatasets_100K',
        'results_dir': 'Autoencoder_v08_100E_32LD_CombinedDatasets_100K_Date20251125-171340.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir',
                    '/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    }
}


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================================

def generate_reconstructions(model_version, output_format='mat', batch_size=32):
    """
    Generate reconstructed spectrograms for all samples
    
    Args:
        model_version: 'v06', 'v07', or 'v08'
        output_format: 'mat', 'npy', or 'both'
        batch_size: Batch size for processing
    """
    
    if model_version not in MODELS:
        raise ValueError(f"Unknown model version: {model_version}. Choose from: {list(MODELS.keys())}")
    
    config = MODELS[model_version]
    
    print("="*70)
    print(f"GENERATING RECONSTRUCTIONS: {config['name']}")
    print("="*70)
    print(f"Output format: {output_format}")
    print(f"Batch size: {batch_size}")
    print()
    
    # ========================================================================
    # STEP 1: Load the trained model
    # ========================================================================
    print("STEP 1: Loading trained model...")
    
    base_dir = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB"
    results_dir = os.path.join(base_dir, "results", config['results_dir'])
    model_path = os.path.join(results_dir, config['model_file'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedAutoencoder(
        nrow=121, ncol=104,
        latent_dim=config['latent_dim'],
        base_channels=config['base_channels'],
        extra_conv=config['extra_conv']
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"  ✓ Model loaded from: {model_path}")
    print(f"  ✓ Device: {device}")
    print()
    
    # ========================================================================
    # STEP 2: Load latent embeddings and reconstruct file mapping
    # ========================================================================
    print("STEP 2: Loading latent embeddings and file mapping...")
    
    embeddings_path = os.path.join(results_dir, 'latent_embeddings.mat')
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Latent embeddings not found: {embeddings_path}")
    
    embeddings_data = loadmat(embeddings_path)
    latent_embeddings = embeddings_data['latent_embeddings']
    n_samples = latent_embeddings.shape[0]
    
    print(f"  ✓ Loaded {n_samples:,} latent vectors")
    print(f"  ✓ Latent dimension: {latent_embeddings.shape[1]}")
    
    # Check if filenames are already stored in the embeddings file
    if 'original_filenames' in embeddings_data:
        print(f"\n  ✓ Using filenames from latent_embeddings.mat")
        original_filenames = embeddings_data['original_filenames'].flatten()
        if 'reconstruction_filenames' in embeddings_data:
            reconstruction_filenames = embeddings_data['reconstruction_filenames'].flatten()
        else:
            reconstruction_filenames = [f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in original_filenames]
        all_files = original_filenames
        print(f"    Found {len(all_files):,} filenames in embeddings file")
    else:
        # Fallback: Reconstruct file mapping from training datasets
        print(f"\n  ⚠ No filenames in embeddings file, reconstructing from datasets...")
        all_files = []
        
        for dataset_path in config['datasets']:
            if os.path.exists(dataset_path):
                # Look for .mat files (spectrogram files)
                mat_files = sorted(glob.glob(os.path.join(dataset_path, "*.mat")))
                
                if mat_files:
                    all_files.extend(mat_files)
                    print(f"    Found {len(mat_files):,} .mat files in {os.path.basename(dataset_path)}")
                else:
                    # Try jpg/png if no .mat files
                    jpg_files = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
                    png_files = sorted(glob.glob(os.path.join(dataset_path, "*.png")))
                    
                    if jpg_files:
                        all_files.extend(jpg_files)
                        print(f"    Found {len(jpg_files):,} .jpg files in {os.path.basename(dataset_path)}")
                    elif png_files:
                        all_files.extend(png_files)
                        print(f"    Found {len(png_files):,} .png files in {os.path.basename(dataset_path)}")
                    else:
                        print(f"    ⚠ No data files found in {os.path.basename(dataset_path)}")
        
        if len(all_files) == 0:
            print(f"\n  ⚠ No original files found, will use index-based naming")
            all_files = [f"sample_{i:06d}.mat" for i in range(n_samples)]
        elif len(all_files) > n_samples:
            print(f"\n  ⚠ File count mismatch: {len(all_files)} files vs {n_samples} embeddings")
            print(f"    Likely {len(all_files) - n_samples} files were skipped during training (corrupted/filtered)")
            print(f"    Using first {n_samples:,} files from sorted list to match training order")
            all_files = all_files[:n_samples]
            print(f"  ✓ File mapping adjusted: {len(all_files):,} files matched to embeddings")
        elif len(all_files) < n_samples:
            print(f"\n  ⚠ File count mismatch: {len(all_files)} files vs {n_samples} embeddings")
            print(f"    Missing files - using index-based naming for safety")
            all_files = [f"sample_{i:06d}.mat" for i in range(n_samples)]
        else:
            print(f"\n  ✓ File mapping complete: {len(all_files):,} files matched to embeddings")
        
        # Extract basenames and create reconstruction filenames
        all_files = [os.path.basename(f) for f in all_files]
        reconstruction_filenames = [f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in all_files]
    
    print()
    
    # ========================================================================
    # STEP 3: Create output directory
    # ========================================================================
    output_dir = os.path.join(results_dir, "reconstructed_spectrograms")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"STEP 3: Output directory created")
    print(f"  {output_dir}")
    print()
    
    # ========================================================================
    # STEP 4: Process latent vectors and generate reconstructions
    # ========================================================================
    print("STEP 4: Generating reconstructions from latent vectors...")
    print(f"  Processing {n_samples:,} samples in batches of {batch_size}...")
    print()
    
    start_time = datetime.now()
    processed_count = 0
    
    # Convert latent embeddings to tensor
    latent_tensor = torch.from_numpy(latent_embeddings).float().to(device)
    
    with torch.no_grad():
        # Process in batches
        for batch_start in tqdm(range(0, n_samples, batch_size), desc="Processing", unit="batch"):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_latent = latent_tensor[batch_start:batch_end]
            
            # Decode latent vectors to reconstructions
            # Go through decoder part of model
            x_recon = model.from_latent(batch_latent)
            x_recon = x_recon.view(x_recon.size(0), model.c_out, model.nrow_reduced, model.ncol_reduced)
            recon_batch = model.decoder(x_recon)
            
            # Match shape to original
            recon_batch = match_shape_center(recon_batch, (121, 104))
            
            # Save each reconstruction
            for i in range(batch_latent.size(0)):
                sample_idx = batch_start + i
                recon = recon_batch[i]
                
                # Convert to numpy
                recon_np = recon.squeeze().cpu().numpy()
                
                # Get filenames (use pre-generated reconstruction filename if available)
                orig_basename = all_files[sample_idx]
                if isinstance(reconstruction_filenames, (list, np.ndarray)):
                    output_basename = os.path.splitext(reconstruction_filenames[sample_idx])[0]
                else:
                    orig_name = os.path.splitext(orig_basename)[0]
                    output_basename = f"{orig_name}_reconstr"
                
                # Save in requested format(s) - MINIMAL DATA ONLY
                if output_format in ['mat', 'both']:
                    output_path_mat = os.path.join(output_dir, f"{output_basename}.mat")
                    # Only save the reconstruction spectrogram, no metadata
                    savemat(output_path_mat, {'spec_sample': recon_np})
                
                if output_format in ['npy', 'both']:
                    output_path_npy = os.path.join(output_dir, f"{output_basename}.npy")
                    np.save(output_path_npy, recon_np)
                
                processed_count += 1
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("="*70)
    print("RECONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Total files processed:  {processed_count:,}")
    print(f"Time elapsed:           {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Processing rate:        {processed_count/elapsed_time:.1f} files/second")
    print()
    print(f"Output directory:")
    print(f"  {output_dir}")
    print()
    print(f"File naming convention:")
    if len(all_files) > 0 and not all_files[0].startswith('sample_'):
        print(f"  Original:       {os.path.basename(all_files[0])}")
        orig_name = os.path.splitext(os.path.basename(all_files[0]))[0]
        print(f"  Reconstructed:  {orig_name}_reconstr.{output_format}")
    else:
        print(f"  sample_000000_reconstr.{output_format}")
        print(f"  sample_000001_reconstr.{output_format}")
        print(f"  ...")
    print()
    print("You can now:")
    print("  1. Load reconstructed spectrograms in MATLAB using sample index")
    print("  2. Use t-SNE/UMAP indices to find corresponding reconstructions")
    print("  3. Compare reconstructions across different clusters")
    print("="*70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reconstructed spectrograms for all samples")
    parser.add_argument('--model', type=str, required=True, choices=['v06', 'v07', 'v08'],
                       help='Model version: v06 (MostlyManual 50K), v07 (AutoWithAirguns 50K), v08 (Combined 100K)')
    parser.add_argument('--format', type=str, default='mat', choices=['mat', 'npy', 'both'],
                       help='Output format: mat (MATLAB), npy (NumPy), or both')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    
    args = parser.parse_args()
    
    generate_reconstructions(
        model_version=args.model,
        output_format=args.format,
        batch_size=args.batch_size
    )
