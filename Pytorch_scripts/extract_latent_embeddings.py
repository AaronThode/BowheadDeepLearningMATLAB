#!/usr/bin/env python3
"""
Extract latent embeddings from a trained autoencoder model.

This script loads a saved model and extracts latent embeddings from your dataset,
then saves them for re-plotting t-SNE without retraining.

USAGE:
    python extract_latent_embeddings.py <path_to_model_dir>
    python extract_latent_embeddings.py results/Autoencoder_v03_Date20251117-111454.dir
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
import argparse
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
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        return latent


def extract_embeddings(model_dir, data_dir=None, n_samples=30, seed=42, 
                      latent_dim=64, channels=64, extra_conv=False, k_clusters=0):
    """Extract latent embeddings from trained model and save for re-plotting."""
    
    # Find model file
    model_path = os.path.join(model_dir, 'autoencoder_clean.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'improved_autoencoder.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    
    # Determine data directory
    if data_dir is None:
        # Try to guess from directory name or use default
        data_dir = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_Balanced.dir"
        print(f"Using default data directory: {data_dir}")
    
    # Load dataset
    print(f"Loading dataset from: {data_dir}")
    dataset = SNRDataset(data_dir, normalize=True, seed=seed, show_summary=True)
    
    # Get sample shape
    sample, _ = dataset[0]
    nrow, ncol = sample.shape[-2], sample.shape[-1]
    print(f"Data shape: {nrow} x {ncol}")
    
    # Initialize model
    print(f"Initializing model (latent_dim={latent_dim}, channels={channels}, extra_conv={extra_conv})...")
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim,
                                base_channels=channels, extra_conv=extra_conv)
    
    # Load weights (only encoder and to_latent parts needed)
    state_dict = torch.load(model_path, map_location='cpu')
    # Filter to only keep encoder and to_latent weights
    encoder_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder.') or k.startswith('to_latent.')}
    model.load_state_dict(encoder_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")
    
    # Extract latent embeddings
    n_samples = min(n_samples, len(dataset))
    print(f"\nExtracting {n_samples} latent embeddings...")
    all_latent = []
    
    with torch.no_grad():
        for i in range(n_samples):
            sample, _ = dataset[i]
            latent = model(sample.unsqueeze(0))
            all_latent.append(latent.cpu())
            
            if (i + 1) % 100 == 0:
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
    print("Now you can use replot_tsne_from_saved.py!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent embeddings from trained model")
    parser.add_argument("model_dir", help="Directory containing autoencoder_clean.pth")
    parser.add_argument("--data-dir", default=None, help="Data directory (auto-detected if not provided)")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=64, help="Base channels")
    parser.add_argument("--extra-conv", action='store_true', help="Model uses 4 conv layers")
    parser.add_argument("--k-clusters", type=int, default=0, help="Number of clusters (0=auto)")
    args = parser.parse_args()
    
    extract_embeddings(args.model_dir, args.data_dir, args.n_samples, args.seed,
                      args.latent_dim, args.channels, args.extra_conv, args.k_clusters)
