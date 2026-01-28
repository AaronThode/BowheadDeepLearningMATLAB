#!/usr/bin/env python3
"""
Extract latent embeddings from a trained autoencoder model.

This script loads a saved model and extracts latent embeddings from your dataset,
then saves them for re-plotting t-SNE without retraining.

USAGE:
    python extract_latent_embeddings.py <path_to_model_dir>
    python extract_latent_embeddings.py results/Autoencoder_v03_Date20251117-111454.dir
    
    # With reconstruction panels:
    python extract_latent_embeddings.py results/... --generate-reconstructions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import glob
import math
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    TSNE = None
    KMeans = None
    silhouette_score = None

# Panel settings
PANEL_GROUP_SIZE = 8
SHOW_ERROR_PLOTS = True


# ============================================================================
# Copy classes from training script
# ============================================================================

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


class SNRDataset(Dataset):
    """Memory-efficient Dataset loading .mat files on-demand."""
    
    def __init__(self, directory, normalize: bool = True, 
                 seed: int | None = None, show_summary: bool = False):
        """
        Args:
            directory: Single directory path (str) or list of directory paths
            normalize: Whether to min-max normalize to [0,1]
            seed: Random seed for shuffling
            show_summary: Print dataset info
        """
        self.normalize = normalize
        self.file_paths: list[str] = []
        
        # Handle single directory or list of directories
        if isinstance(directory, str):
            directories = [directory]
        else:
            directories = list(directory)
        
        target_shape = None
        for data_dir in directories:
            mat_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True))
            
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
            raise RuntimeError(f"No valid .mat files found in {directories}")
        
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


class ImprovedAutoencoder(nn.Module):
    """Autoencoder with batch normalization and no sigmoid constraint."""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=64, 
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
                nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            flat_size = c4 * nrow_reduced * ncol_reduced
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
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
        
        # Decoder for reconstructions
        if extra_conv:
            pad_h = (nrow - nrow_reduced * 16) % 2
            pad_w = (ncol - ncol_reduced * 16) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, stride=2), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        else:
            pad_h = (nrow - nrow_reduced * 8) % 2
            pad_w = (ncol - ncol_reduced * 8) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
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
    
    def encode(self, x):
        """Encode input to latent space only."""
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        return self.to_latent(x_flat)


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


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "", filenames: list = None, 
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
            axes[1, col].set_title('Reconstruction', fontsize=6)
            axes[1, col].axis('off')
            
            if show_error:
                diff_np = np.abs(orig_np[col] - recon_np[col])
                axes[2, col].imshow(diff_np, cmap='hot', origin='lower', aspect='auto')
                axes[2, col].set_title('Error', fontsize=6)
                axes[2, col].axis('off')
        
        if dataset_label:
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        panel_path = os.path.join(output_dir, f"{base_name}_{group_idx + 1:03d}.jpg")
        plt.tight_layout()
        plt.savefig(panel_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        panels_written += 1
        print(f"  Saved: {panel_path}")
    return panels_written


def save_reconstructions_summary(model: nn.Module, samples: torch.Tensor, output_dir: str,
                                  target_hw: tuple[int, int], latent_dim: int, n_show: int = 5):
    """Save a summary PNG showing original vs reconstruction."""
    n_show = min(n_show, samples.shape[0])
    with torch.no_grad():
        recon, latent = model(samples[:n_show])
        recon = match_shape_center(recon, target_hw)
    
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_show):
        axes[0, i].imshow(samples[i].squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Autoencoder Reconstructions (latent_dim={latent_dim})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'reconstructions.png')}")


def extract_embeddings(model_dir, data_dir=None, n_samples=None, seed=42, 
                      latent_dim=64, channels=64, extra_conv=False, k_clusters=0,
                      generate_reconstructions=False, n_recon_samples=64):
    """Extract latent embeddings from trained model and save for re-plotting.
    
    Args:
        model_dir: Directory containing autoencoder_clean.pth
        data_dir: Single data directory (str) or list of directories. 
                  If None, uses default combined datasets.
        n_samples: Number of samples to extract. If None, extracts ALL samples.
        seed: Random seed
        latent_dim: Latent dimension of model
        channels: Base channels of model
        extra_conv: Whether model uses 4 conv layers
        k_clusters: Number of clusters (0=auto-find optimal)
        generate_reconstructions: If True, also generate reconstruction panels
        n_recon_samples: Number of samples for reconstruction panels
    """
    
    # Find model file
    model_path = os.path.join(model_dir, 'autoencoder_clean.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'improved_autoencoder.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    
    # Determine data directory - default to combined datasets
    if data_dir is None:
        data_dir = [
            "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir",
            "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir"
        ]
        print(f"Using default combined data directories:")
        for d in data_dir:
            print(f"  - {d}")
    
    # Load dataset (handles single dir or list of dirs)
    print(f"\nLoading dataset...")
    dataset = SNRDataset(data_dir, normalize=True, seed=seed, show_summary=True)
    
    # Get sample shape
    sample, _ = dataset[0]
    nrow, ncol = sample.shape[-2], sample.shape[-1]
    print(f"Data shape: {nrow} x {ncol}")
    
    # Initialize model
    print(f"Initializing model (latent_dim={latent_dim}, channels={channels}, extra_conv={extra_conv})...")
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim,
                                base_channels=channels, extra_conv=extra_conv)
    
    # Load weights (full model for reconstructions)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully!")
    
    # Extract latent embeddings - if n_samples is None, extract ALL
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))
    print(f"\nExtracting {n_samples} latent embeddings (dataset has {len(dataset)} total)...")
    all_latent = []
    all_samples = []  # Keep samples for reconstructions if needed
    all_filenames = []
    
    with torch.no_grad():
        for i in range(n_samples):
            sample, _ = dataset[i]
            latent = model.encode(sample.unsqueeze(0))
            all_latent.append(latent.cpu())
            
            # Store samples for reconstruction (only up to n_recon_samples)
            if generate_reconstructions and len(all_samples) < n_recon_samples:
                all_samples.append(sample)
                if hasattr(dataset, 'file_paths'):
                    all_filenames.append(os.path.basename(dataset.file_paths[i]))
                else:
                    all_filenames.append("")
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples}...")
    
    imp_z = torch.cat(all_latent, dim=0).numpy()
    print(f"Extracted {len(imp_z)} embeddings")
    
    # Run t-SNE
    if TSNE is not None:
        print("Computing t-SNE...")
        perplexity = min(30.0, (imp_z.shape[0] - 1) / 3.0)
        perplexity = max(2.0, perplexity)
        emb = TSNE(n_components=2, random_state=seed, perplexity=perplexity, 
                  learning_rate='auto').fit_transform(imp_z)
        print("t-SNE complete!")
    else:
        print("Warning: TSNE not available, skipping t-SNE projection")
        emb = np.zeros((imp_z.shape[0], 2))
        perplexity = 30.0
    
    # Find optimal k
    optimal_k = k_clusters if k_clusters > 0 else 2
    if KMeans is not None and silhouette_score is not None and k_clusters == 0:
        print("Finding optimal k...")
        max_k = min(10, imp_z.shape[0] // 2)
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=seed)
            labels_temp = kmeans_temp.fit_predict(imp_z)
            score = silhouette_score(imp_z, labels_temp)
            silhouette_scores.append(score)
            print(f"  k={k}: silhouette={score:.3f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal k={optimal_k}")
    
    # Cluster
    if KMeans is not None:
        kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=seed)
        clusters = kmeans.fit_predict(imp_z)
    else:
        clusters = np.zeros(imp_z.shape[0], dtype=int)
    
    # Save embeddings
    # Handle dataset label for single or multiple directories
    if isinstance(data_dir, list):
        dataset_label = "CombinedDatasets"
    else:
        dataset_label = os.path.basename(data_dir.rstrip('/'))
    latent_data = {
        'latent_embeddings': imp_z,
        'tsne_embeddings': emb,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'perplexity': perplexity,
        'dataset_label': dataset_label
    }
    
    output_path = os.path.join(model_dir, 'latent_embeddings.mat')
    savemat(output_path, latent_data)
    print(f"\nSaved embeddings to: {output_path}")
    
    # Generate reconstruction panels if requested
    if generate_reconstructions and all_samples:
        print(f"\nGenerating reconstruction panels for {len(all_samples)} samples...")
        samples_tensor = torch.stack(all_samples, dim=0)
        
        # Save summary reconstructions.png
        save_reconstructions_summary(model, samples_tensor, model_dir, (nrow, ncol), latent_dim)
        
        # Save individual panels
        panels_written = save_reconstruction_panels(
            model, samples_tensor, model_dir, (nrow, ncol),
            base_name="recon_panel", dataset_label=dataset_label, 
            filenames=all_filenames
        )
        print(f"  Generated {panels_written} reconstruction panels")
        
        # Save reconstruction data to .mat
        with torch.no_grad():
            recon, _ = model(samples_tensor)
            recon = match_shape_center(recon, (nrow, ncol))
        
        recon_data = {
            'originals': samples_tensor.squeeze(1).cpu().numpy(),
            'reconstructions': recon.squeeze(1).cpu().numpy(),
            'filenames': all_filenames
        }
        recon_path = os.path.join(model_dir, 'reconstruction_data.mat')
        savemat(recon_path, recon_data)
        print(f"  Saved reconstruction data to: {recon_path}")
    
    print("\nDone! You can use replot_tsne_from_saved.py to regenerate t-SNE plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent embeddings from trained model")
    parser.add_argument("model_dir", help="Directory containing autoencoder_clean.pth")
    parser.add_argument("--data-dir", nargs='+', default=None, 
                        help="Data directory/directories (uses combined datasets if not provided)")
    parser.add_argument("--n-samples", type=int, default=None, 
                        help="Number of samples to extract (default: ALL samples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension (default: 32)")
    parser.add_argument("--channels", type=int, default=64, help="Base channels")
    parser.add_argument("--extra-conv", action='store_true', help="Model uses 4 conv layers")
    parser.add_argument("--k-clusters", type=int, default=0, help="Number of clusters (0=auto)")
    parser.add_argument("--generate-reconstructions", "-r", action='store_true',
                        help="Generate reconstruction panels and plots")
    parser.add_argument("--n-recon-samples", type=int, default=64,
                        help="Number of samples for reconstruction panels (default: 64)")
    args = parser.parse_args()
    
    # Handle single or multiple data directories
    data_dir = args.data_dir[0] if args.data_dir and len(args.data_dir) == 1 else args.data_dir
    
    extract_embeddings(args.model_dir, data_dir, args.n_samples, args.seed,
                      args.latent_dim, args.channels, args.extra_conv, args.k_clusters,
                      args.generate_reconstructions, args.n_recon_samples)
