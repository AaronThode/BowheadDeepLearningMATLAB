#!/usr/bin/env python3
"""
FAST CLEAN AUTOENCODER - Training from Scratch (v03)

GUARANTEED FRESH START:
- No model loading or transfer learning
- No checkpoint resuming
- Random initialization only (controlled by seed)
- Each run completely independent

PERFORMANCE OPTIMIZATIONS:
- Minimal output samples (30 instead of 5000) for faster JPEG generation
- Limited t-SNE samples (30 instead of 1000) for instant visualization
- Non-blocking plots (plt.close() instead of plt.show())
- Efficient DataLoader streaming from disk
- Reduced default epochs (10 instead of 50) for quick iterations

USAGE:
    python Autoencoder_FastClean_v03.py [--epochs 10] [--lr 1e-3] [--seed 42]

All outputs saved to results/Autoencoder_v03_Date<TIMESTAMP>.dir/
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster plotting
import matplotlib.pyplot as plt
import os
import glob
import math
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import time
import gc  # For explicit garbage collection
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    TSNE = None
    KMeans = None
    silhouette_score = None
import argparse

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Architecture parameters
CHANNELS_DEFAULT = 64
LATENT_DIM_DEFAULT = 32
EXTRA_CONV_DEFAULT = False

# Training parameters (optimized for speed)
EPOCHS_DEFAULT = 100             # Reduced for faster iterations
LR_DEFAULT = 1e-3
SEED_DEFAULT = 42

# Output parameters (minimized for speed)
NUMBER_OUTPUT_IMAGE_SAMPLES = 30   # Reduced from 5000 for faster JPEG generation
PANEL_GROUP_SIZE = 3
SHOW_ERROR_PLOTS = False
DEFAULT_VERSION_TAG = "05_100E_32LD_MostlyManual"
TSNE_MAX_SAMPLES = None  # None => use all samples in dataset


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_global_seed(seed: int):
    """Set random seeds for reproducible initialization from scratch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directory(version_tag: str | None = None) -> str:
    """Create unique timestamped output directory."""
    tag = (version_tag or DEFAULT_VERSION_TAG).strip().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"Autoencoder_v{tag}_Date{timestamp}.dir"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_dir = os.path.join(results_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _minmax_norm(im: np.ndarray, auto_skip_if_unit: bool = True) -> np.ndarray:
    """Min-max normalize image to [0, 1] range."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    if auto_skip_if_unit and (-1e-4 <= im_min <= 1.0 + 1e-4) and (-1e-4 <= im_max <= 1.0 + 1e-4):
        return im
    return (im - im_min) / rng


# ============================================================================
# DATASET CLASS
# ============================================================================

class SNRDataset(Dataset):
    """Memory-efficient Dataset loading .mat files on-demand."""
    
    def __init__(self, directory: str, normalize: bool = True, 
                 seed: int | None = None, show_summary: bool = False):
        self.normalize = normalize
        self.file_paths: list[str] = []
        
        target_shape = None
        mat_files = sorted(glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True))
        
        for fp in mat_files:
            try:
                m = loadmat(fp)
                im = m.get('SNR_gram', None)
                if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                    continue
                if target_shape is None:
                    target_shape = im.shape
                if im.shape == target_shape:
                    self.file_paths.append(fp)
            except Exception:
                continue
        
        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files found in {directory}")
        
        self.target_shape = target_shape
        
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.file_paths))
            self.file_paths = [self.file_paths[i] for i in indices]
        
        if show_summary:
            print(f"SNRDataset: {len(self)} files with shape {target_shape}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        try:
            m = loadmat(fp)
            im = m['SNR_gram']
            if self.normalize:
                im = _minmax_norm(im)
            else:
                im = im.astype(np.float32)
            tensor = torch.from_numpy(im).unsqueeze(0)
            return tensor, 0
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
            h, w = self.target_shape
            return torch.zeros((1, h, w), dtype=torch.float32), 0


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """
    Autoencoder with batch normalization and no sigmoid constraint.
    ALWAYS INITIALIZED FROM SCRATCH - NO PRETRAINED WEIGHTS.
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, 
                 base_channels=CHANNELS_DEFAULT, extra_conv=EXTRA_CONV_DEFAULT):
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


def match_shape_center(recon: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
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


def select_samples_for_outputs(dataset: Dataset, n_samples: int, seed: int | None) -> tuple[torch.Tensor, list[str]]:
    """Select random samples for JPEG panel generation."""
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("No data available for output sampling")
    rng = np.random.default_rng(seed)
    total = len(dataset)
    k = min(n_samples, total)
    indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
    samples = []
    filenames = []
    for idx in indices:
        sample, _ = dataset[int(idx)]
        samples.append(sample.unsqueeze(0))
        if hasattr(dataset, 'file_paths'):
            filenames.append(os.path.basename(dataset.file_paths[int(idx)]))
        else:
            filenames.append("")
    return torch.cat(samples, dim=0).float(), filenames


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "", filenames: list[str] = None, 
                               show_error: bool = SHOW_ERROR_PLOTS) -> int:
    """Save JPEG panels showing reconstructions."""
    if samples is None or samples.shape[0] == 0:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    num_samples = samples.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    n_rows = 3 if show_error else 2
    
    for group_idx in range(group_count):
        start = group_idx * PANEL_GROUP_SIZE
        end = min(start + PANEL_GROUP_SIZE, num_samples)
        batch = samples[start:end]
        with torch.no_grad():
            recon, _ = model(batch)
            recon = match_shape_center(recon, target_hw)
        orig_np = batch.squeeze(1).cpu().numpy()
        recon_np = recon.squeeze(1).cpu().numpy()
        
        cols = recon_np.shape[0]
        fig, axes = plt.subplots(n_rows, cols, figsize=(3 * cols, 3 * n_rows))
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)
        
        for col in range(cols):
            sample_idx = start + col
            if filenames and sample_idx < len(filenames) and filenames[sample_idx]:
                title = filenames[sample_idx].replace('.mat', '')
                if len(title) > 25:
                    title = title[:22] + '...'
            else:
                title = f'Input {sample_idx + 1}'
            
            axes[0, col].imshow(orig_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[0, col].set_title(title, fontsize=6)
            axes[0, col].axis('off')
            axes[1, col].imshow(recon_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[1, col].axis('off')
            
            if show_error:
                diff_np = np.abs(orig_np - recon_np)
                axes[2, col].imshow(diff_np[col], cmap='hot', origin='lower', aspect='auto')
                axes[2, col].axis('off')
        
        if dataset_label:
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        panel_path = os.path.join(output_dir, f"{base_name}_{group_idx + 1:03d}.jpg")
        plt.tight_layout()
        plt.savefig(panel_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        panels_written += 1
    return panels_written


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_autoencoder_from_scratch(data_dir: str, n_samples: int = 15, latent_dim: int = LATENT_DIM_DEFAULT,
                                   channels: int = CHANNELS_DEFAULT, seed: int | None = SEED_DEFAULT,
                                   epochs: int = EPOCHS_DEFAULT, lr: float = LR_DEFAULT,
                                   tsne_samples: int | None = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                                   batch_size: int = 32, output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                                   version_tag: str = DEFAULT_VERSION_TAG, show_error: bool = SHOW_ERROR_PLOTS,
                                   k_clusters: int = 2, tsne_perplexity: float | None = None):
    """
    Train autoencoder from scratch with guaranteed fresh start and optimal performance.
    
    FRESH START GUARANTEES:
    - New model initialized with random weights (controlled by seed)
    - No checkpoint loading or transfer learning
    - Independent of any previous runs
    - Deterministic if seed is set
    
    PERFORMANCE OPTIMIZATIONS:
    - Non-blocking plots (no plt.show())
    - Limited output samples for fast JPEG generation
    - Reduced t-SNE samples for instant visualization
    - Efficient DataLoader streaming
    - Explicit garbage collection after major operations
    """
    # Start timing
    script_start_time = time.time()
    
    # STEP 1: Initialize random seed for reproducible FROM-SCRATCH initialization
    if seed is not None:
        set_global_seed(int(seed))
        print(f"Random seed set to {seed} for reproducible initialization")
    
    output_dir = create_output_directory(version_tag)
    print(f"="*70)
    print(f"FAST CLEAN AUTOENCODER - Training from Scratch")
    print(f"="*70)
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: epochs={epochs}, lr={lr}, latent_dim={latent_dim}, channels={channels}")
    print(f"="*70)
    
    # STEP 2: Load dataset
    print(f"\nLoading data from: {data_dir}")
    dataset_label = os.path.basename(data_dir.rstrip('/'))
    dataset = SNRDataset(data_dir, normalize=True, seed=seed, show_summary=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load visualization samples
    if tsne_samples is None:
        tsne_samples = n_samples
    viz_samples = min(tsne_samples, len(dataset))
    print(f"Loading {viz_samples} samples for visualization...")
    data_list = []
    for i in range(viz_samples):
        sample, _ = dataset[i]
        data_list.append(sample.unsqueeze(0))
    data_tensor = torch.cat(data_list, dim=0) if data_list else None
    
    # STEP 3: Initialize model FROM SCRATCH (no pretrained weights)
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    print(f"\nInitializing NEW model from random weights...")
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, 
                                base_channels=channels, extra_conv=extra_conv)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {4 if extra_conv else 3} conv layers, latent_dim={latent_dim}")
    
    # STEP 4: Train from scratch
    print(f"\nTraining model from scratch for {epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    losses = []
    epoch_times = []
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_data, _ in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch_data)
            output = match_shape_center(output, (nrow, ncol))
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        with torch.no_grad():
            o_min = float(output.min().cpu())
            o_max = float(output.max().cpu())
            o_mean = float(output.mean().cpu())
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: Loss={avg_loss:.4f} | out[{o_min:.3f}, {o_max:.3f}] | {epoch_time:.1f}s")
    
    training_elapsed = time.time() - training_start_time
    print(f"\nTraining complete: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
    print(f"  Avg per epoch: {training_elapsed/epochs:.1f}s")
    
    # Save model weights
    model_path = os.path.join(output_dir, 'autoencoder_clean.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    
    # Clear training memory
    del train_loader
    gc.collect()
    
    # STEP 5: Extract latent embeddings
    model.eval()
    tsne_sample_count = min(TSNE_MAX_SAMPLES, len(dataset)) if TSNE_MAX_SAMPLES else len(dataset)
    print(f"\nExtracting {tsne_sample_count} latent embeddings for t-SNE analysis...")
    all_latent = []
    with torch.no_grad():
        for i in range(tsne_sample_count):
            sample, _ = dataset[i]
            _, latent = model(sample.unsqueeze(0))
            all_latent.append(latent.cpu())
    improved_latent_full = torch.cat(all_latent, dim=0)
    
    # Compute reconstructions on viz subset
    with torch.no_grad():
        improved_recon, _ = model(data_tensor)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol))
    
    # STEP 6: Generate visualizations
    data_np = data_tensor.squeeze(1).numpy()
    print(f"\nGenerating visualizations...")
    
    # Plot 1: Reconstruction comparison
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    cols = min(10, data_np.shape[0])
    n_rows = 3 if show_error else 2
    fig, axes = plt.subplots(n_rows, cols, figsize=(15, 6 if n_rows == 2 else 9))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        imp_recon = improved_recon[i, 0].numpy()
        axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        if show_error:
            diff = np.abs(data_np[i] - imp_recon)
            axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title('Error')
            axes[2, i].axis('off')
    
    plt.suptitle(f'Autoencoder Reconstructions (epochs={epochs}, latent_dim={latent_dim})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training loss
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (epochs={epochs})')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
    plt.close()
    
    # Plot 3: t-SNE with clustering (auto-find optimal k)
    # Extract latent embeddings first (save even if t-SNE fails)
    imp_z = improved_latent_full.detach().cpu().numpy()
    dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    if TSNE is not None and improved_latent_full.shape[0] > 2:
        try:
            print(f"Computing t-SNE on {improved_latent_full.shape[0]} samples...")
            perplexity = min(30.0, (imp_z.shape[0] - 1) / 3.0) if tsne_perplexity is None else tsne_perplexity
            perplexity = max(2.0, min(perplexity, imp_z.shape[0] - 1))
            
            emb = TSNE(n_components=2, random_state=int(seed) if seed else 0, 
                      perplexity=perplexity, learning_rate='auto').fit_transform(imp_z)
            
            # Auto-find optimal k if k_clusters is None or 0
            optimal_k = k_clusters
            if KMeans is not None and silhouette_score is not None and (k_clusters is None or k_clusters == 0):
                print("Finding optimal number of clusters...")
                max_k = min(10, imp_z.shape[0] // 2)  # Test up to 10 clusters
                silhouette_scores = []
                k_range = range(2, max_k + 1)
                
                for k in k_range:
                    kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=int(seed) if seed else 0)
                    labels_temp = kmeans_temp.fit_predict(imp_z)
                    score = silhouette_score(imp_z, labels_temp)
                    silhouette_scores.append(score)
                    print(f"  k={k}: silhouette={score:.3f}")
                
                # Choose k with highest silhouette score
                optimal_k = k_range[np.argmax(silhouette_scores)]
                print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
                
                # Save elbow plot
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(list(k_range), silhouette_scores, 'bo-')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Optimal k Selection')
                plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'optimal_k_analysis.png'), dpi=150)
                plt.close()
            elif k_clusters is None or k_clusters == 0:
                optimal_k = 2  # Fallback if silhouette unavailable
            
            # Perform final clustering with optimal k
            if KMeans is not None:
                kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=int(seed) if seed else 0)
                clusters = kmeans.fit_predict(imp_z)
            else:
                clusters = (emb[:, 0] > np.median(emb[:, 0])).astype(int)
            
            # Generate color map for all clusters
            cmap = plt.cm.get_cmap('tab10', optimal_k)
            plt.figure(figsize=(7, 6))
            
            # Plot each cluster separately for legend
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                color = cmap(cluster_id)
                plt.scatter(emb[mask, 0], emb[mask, 1], 
                           c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            plt.title(f't-SNE Latent Space (k={optimal_k}, perplexity={perplexity:.1f})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
            
            # Add dataset label in bottom right (discrete/subtle)
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
                       ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tsne_latent.png'), dpi=160)
            plt.close()
            
            # Save latent embeddings and t-SNE results for re-plotting without retraining
            latent_data = {
                'latent_embeddings': imp_z,
                'tsne_embeddings': emb,
                'clusters': clusters,
                'optimal_k': optimal_k,
                'perplexity': perplexity,
                'dataset_label': dataset_label
            }
            savemat(os.path.join(output_dir, 'latent_embeddings.mat'), latent_data)
            print(f"Saved latent embeddings to latent_embeddings.mat (re-plot without retraining!)")
            
        except Exception as e:
            print(f"Warning: t-SNE visualization skipped: {e}")
            # Still save latent embeddings even if t-SNE failed
            emb = np.zeros((imp_z.shape[0], 2))  # Dummy t-SNE embeddings
            clusters = np.zeros(imp_z.shape[0], dtype=int)
            optimal_k = 2
            perplexity = 30.0
    else:
        print("Warning: TSNE not available or insufficient samples")
        emb = np.zeros((imp_z.shape[0], 2))
        clusters = np.zeros(imp_z.shape[0], dtype=int)
        optimal_k = 2
        perplexity = 30.0
    
    # ALWAYS save latent embeddings (even if t-SNE failed)
    print(f"Saving latent embeddings ({imp_z.shape[0]} samples, {imp_z.shape[1]}-dim)...")
    
    # Extract filenames from dataset (only basenames, matching embedding order)
    filenames = np.array([os.path.basename(dataset.file_paths[i]) for i in range(tsne_sample_count)], dtype=object)
    
    latent_data = {
        'latent_embeddings': imp_z,
        'tsne_embeddings': emb,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'perplexity': perplexity,
        'dataset_label': dataset_label,
        'filenames': filenames
    }
    embeddings_path = os.path.join(output_dir, 'latent_embeddings.mat')
    savemat(embeddings_path, latent_data)
    print(f"Saved latent embeddings to: {embeddings_path}")
    print(f"  -> Includes 'filenames' field mapping {len(filenames)} embeddings to source files")
    print("  -> Use replot_tsne_from_saved.py to re-plot with different k values!")
    
    # STEP 7: Save JPEG reconstruction panels
    try:
        panel_samples, filenames = select_samples_for_outputs(dataset, output_samples, seed)
        panels_written = save_reconstruction_panels(model, panel_samples, output_dir, (nrow, ncol),
                                                    dataset_label=dataset_label, filenames=filenames, 
                                                    show_error=show_error)
        print(f"Saved {panels_written} JPEG panel(s) ({panel_samples.shape[0]} samples)")
    except Exception as e:
        print(f"Warning: JPEG panels skipped: {e}")
    
    # STEP 8: Save data and timing log
    sample_data = {
        'spectrograms': data_np,
        'reconstructions': improved_recon.squeeze().numpy()
    }
    savemat(os.path.join(output_dir, 'reconstruction_data.mat'), sample_data)
    
    script_elapsed = time.time() - script_start_time
    timing_log = [
        f"FAST CLEAN AUTOENCODER - Training from Scratch",
        f"=" * 70,
        f"Start: {datetime.fromtimestamp(script_start_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {script_elapsed:.1f}s ({script_elapsed/60:.1f}min)",
        f"",
        f"Configuration:",
        f"  Dataset: {dataset_label}",
        f"  Files: {len(dataset)}",
        f"  Epochs: {epochs}",
        f"  Learning rate: {lr}",
        f"  Batch size: {batch_size}",
        f"  Latent dim: {latent_dim}",
        f"  Channels: {channels}",
        f"  Seed: {seed}",
        f"",
        f"Performance:",
        f"  Training: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)",
        f"  Per epoch: {training_elapsed/epochs:.1f}s",
        f"  Other ops: {script_elapsed - training_elapsed:.1f}s",
        f"",
        f"Results:",
        f"  Final loss: {losses[-1]:.6f}",
        f"  Model saved: autoencoder_clean.pth",
        f"  FRESH START: Model trained from random initialization",
    ]
    
    with open(os.path.join(output_dir, 'timing_log.txt'), 'w') as f:
        f.write('\n'.join(timing_log))
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Total: {script_elapsed:.1f}s ({script_elapsed/60:.1f}min)")
    print(f"  Training: {training_elapsed:.1f}s | Other: {script_elapsed - training_elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Clean Autoencoder - Train from Scratch")
    parser.add_argument("--data-dir", 
                       default="/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir",
                       help="Directory containing .mat files")
    parser.add_argument("--n-samples", type=int, default=15, help="Visualization samples")
    parser.add_argument("--tsne-samples", type=int, default=None, help="t-SNE samples (default: n-samples)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base channels")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help=f"Training epochs (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help=f"Learning rate (default: {LR_DEFAULT})")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help=f"Random seed (default: {SEED_DEFAULT})")
    parser.add_argument("--k-clusters", type=int, default=0, help="KMeans clusters for t-SNE (0=auto-detect optimal k)")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="t-SNE perplexity")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT, help="Use 4 conv layers")
    parser.add_argument("--no-extra-conv", dest='extra_conv', action='store_false', help="Use 3 conv layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-samples", type=int, default=NUMBER_OUTPUT_IMAGE_SAMPLES, help="JPEG panel samples")
    parser.add_argument("--show-error", action='store_true', default=SHOW_ERROR_PLOTS, help="Show error row")
    parser.add_argument("--no-error", dest='show_error', action='store_false', help="Hide error row")
    parser.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG, help="Version tag")
    args = parser.parse_args()
    
    train_autoencoder_from_scratch(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        latent_dim=args.latent_dim,
        channels=args.channels,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        tsne_samples=args.tsne_samples,
        extra_conv=args.extra_conv,
        batch_size=args.batch_size,
        output_samples=args.output_samples,
        version_tag=args.version_tag,
        show_error=args.show_error,
        k_clusters=args.k_clusters,
        tsne_perplexity=args.tsne_perplexity
    )
