#!/usr/bin/env python3
"""
Demonstrate a disentangled autoencoder strategy with:
  - Global log-compressed normalization (preserve relative amplitude across samples)
  - Seeded deterministic sample selection
  - Latent split into structure (z_struct) and context (z_context) components

This script parallels `demonstrate_reconstruction_issues.py` but adds factorization of the latent space
and global normalization to better align with downstream clustering goals (airgun vs whale vocalizations)
while distinguishing intrinsic signal morphology from broader time-series context.
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from typing import Tuple

# -------------------- Global Defaults --------------------
CHANNELS_DEFAULT = 64
LATENT_STRUCT_DEFAULT = 64
LATENT_CONTEXT_DEFAULT = 32
EPOCHS_DEFAULT = 60
LR_DEFAULT = 1e-3
NORMALIZATION_CACHE_NAME = "global_stats_cache.npz"

# -------------------- Utility: Global Log Normalization --------------------
def compute_global_stats(root: str, seed: int | None, max_files: int = 500) -> Tuple[float, float]:
    """Scan up to max_files .mat files (seeded order) to compute mean/std in log1p domain.
    Returns (mean, std)."""
    mat_files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files under {root}")

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if len(mat_files) > max_files:
        indices = rng.choice(len(mat_files), size=max_files, replace=False)
        mat_files = [mat_files[i] for i in indices]

    vals = []
    target_shape = None
    for fp in mat_files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram')
            if im is None or im.ndim != 2:
                continue
            if target_shape is None:
                target_shape = im.shape
            if im.shape != target_shape:
                continue
            im = im.astype(np.float32)
            im = np.log1p(np.maximum(im, 0.0))  # log compression
            vals.append(im)
        except Exception:
            continue
    if not vals:
        raise RuntimeError("No consistent-shape SNR_gram data for global stats.")

    stacked = np.stack(vals, axis=0)
    mean = float(stacked.mean())
    std = float(stacked.std() + 1e-6)  # prevent divide by zero
    return mean, std

def load_global_stats(root: str, seed: int | None, recompute: bool) -> Tuple[float, float]:
    cache_path = os.path.join(root, NORMALIZATION_CACHE_NAME)
    if recompute or not os.path.exists(cache_path):
        mean, std = compute_global_stats(root, seed)
        np.savez(cache_path, mean=mean, std=std)
    else:
        d = np.load(cache_path)
        mean = float(d['mean'])
        std = float(d['std'])
    return mean, std

def normalize_log_global(im: np.ndarray, mean: float, std: float) -> np.ndarray:
    im = im.astype(np.float32)
    im = np.log1p(np.maximum(im, 0.0))
    return (im - mean) / std

# -------------------- Data Loading with Seeded Selection --------------------

def select_snrgram_files(root: str, n_samples: int, seed: int | None) -> list[str]:
    mat_files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {root}")
    target_shape = None
    candidates: list[str] = []
    for fp in mat_files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram')
            if im is None or im.ndim != 2:
                continue
            if target_shape is None:
                target_shape = im.shape
            if im.shape != target_shape:
                continue
            candidates.append(fp)
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No consistent-shape SNR_gram candidates.")
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    k = min(n_samples, len(candidates))
    idx = rng.choice(len(candidates), size=k, replace=False)
    return [candidates[i] for i in np.sort(idx)]

def load_dataset(root: str, n_samples: int, seed: int | None, mean: float, std: float, show_files: bool) -> np.ndarray:
    files = select_snrgram_files(root, n_samples, seed)
    if show_files:
        print("Selected files:")
        for f in files:
            print(" -", os.path.relpath(f, root))
    data: list[np.ndarray] = []
    for fp in files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram')
            im = normalize_log_global(im, mean, std)
            data.append(im.astype(np.float32))
        except Exception:
            continue
    if not data:
        raise RuntimeError("Failed to load any normalized spectrograms.")
    return np.stack(data, axis=0)

# -------------------- Disentangled Autoencoder --------------------
class DisentangledAutoencoder(nn.Module):
    def __init__(self, nrow: int, ncol: int, base_channels: int,
                 latent_struct: int, latent_context: int):
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        # Structure branch encoder (focus on fine morphology)
        self.struct_encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Context branch (coarser pooling to capture surroundings)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, c1, 5, padding=2), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Compute shapes
        struct_r = nrow // 8
        struct_c = ncol // 8
        context_r = nrow // 8  # after 4 then 2 pooling
        context_c = ncol // 8
        self.struct_flat = c3 * struct_r * struct_c
        self.context_flat = c2 * context_r * context_c
        # Latent projections
        self.to_struct = nn.Sequential(
            nn.Linear(self.struct_flat, latent_struct * 2), nn.ReLU(), nn.Linear(latent_struct * 2, latent_struct)
        )
        self.to_context = nn.Sequential(
            nn.Linear(self.context_flat, latent_context * 2), nn.ReLU(), nn.Linear(latent_context * 2, latent_context)
        )
        # Decoder from combined latent
        combined = latent_struct + latent_context
        self.from_latent = nn.Sequential(
            nn.Linear(combined, (latent_struct + latent_context) * 2), nn.ReLU(),
            nn.Linear((latent_struct + latent_context) * 2, self.struct_flat), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8))
        )
        self.struct_r = struct_r
        self.struct_c = struct_c
        self.c3 = c3

    def forward(self, x: torch.Tensor):
        xs = self.struct_encoder(x)
        xc = self.context_encoder(x)
        xs_flat = xs.view(xs.size(0), -1)
        xc_flat = xc.view(xc.size(0), -1)
        z_struct = self.to_struct(xs_flat)
        z_context = self.to_context(xc_flat)
        z = torch.cat([z_struct, z_context], dim=1)
        recon_flat = self.from_latent(z)
        recon_feat = recon_flat.view(x.size(0), self.c3, self.struct_r, self.struct_c)
        out = self.decoder(recon_feat)
        return out, z_struct, z_context

# -------------------- Training --------------------
def match_shape_center(recon: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
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

def quick_train(model: nn.Module, data: torch.Tensor, epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        out, z_struct, z_context = model(data)
        out = match_shape_center(out, (data.shape[-2], data.shape[-1]))
        loss_recon = criterion(out, data)
        # Optional orthogonality penalty between z_struct and z_context
        z_struct_c = z_struct - z_struct.mean(0, keepdim=True)
        z_context_c = z_context - z_context.mean(0, keepdim=True)
        cov = (z_struct_c.T @ z_context_c) / (z_struct_c.size(0) - 1)
        ortho_penalty = cov.pow(2).mean()
        loss = loss_recon + 0.01 * ortho_penalty
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if ep % 10 == 0:
            print(f"Epoch {ep}: recon={loss_recon.item():.4f} ortho={ortho_penalty.item():.4f} total={loss.item():.4f}")
    return losses

# -------------------- Main Comparison --------------------
def run_disentangled_demo(data_dir: str, n_samples: int, seed: int | None, show_files: bool,
                          latent_struct: int, latent_context: int, channels: int,
                          epochs: int, lr: float, recompute_stats: bool):
    print(f"Data root: {data_dir}")
    mean, std = load_global_stats(data_dir, seed, recompute_stats)
    print(f"Global log-space stats: mean={mean:.4f} std={std:.4f}")
    data_np = load_dataset(data_dir, n_samples, seed, mean, std, show_files)
    data_t = torch.from_numpy(data_np).unsqueeze(1)
    print(f"Data tensor: {data_t.shape} range=[{data_t.min():.3f},{data_t.max():.3f}]")

    nrow, ncol = data_t.shape[-2], data_t.shape[-1]
    model = DisentangledAutoencoder(nrow, ncol, channels, latent_struct, latent_context)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent split: struct={latent_struct} context={latent_context}")

    losses = quick_train(model, data_t, epochs, lr)

    model.eval()
    with torch.no_grad():
        recon, z_struct, z_context = model(data_t)
        recon = match_shape_center(recon, (nrow, ncol))

    # Plot losses
    plt.figure(figsize=(8,4))
    plt.plot(losses)
    plt.yscale('log')
    plt.title(f'Training Loss (log scale) struct={latent_struct} context={latent_context} channels={channels}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('disentangled_training_loss.png', dpi=150)
    plt.show()

    # Reconstruction panel
    cols = min(5, data_t.size(0))
    fig, axes = plt.subplots(3, cols, figsize=(3*cols, 8))
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto')
        axes[0, i].set_title(f'Orig {i+1}')
        axes[0, i].axis('off')
        r = recon[i,0].numpy()
        axes[1, i].imshow(r, cmap='viridis', origin='lower', aspect='auto')
        mse = np.mean((data_np[i] - r)**2)
        axes[1, i].set_title(f'Recon MSE={mse:.3f}')
        axes[1, i].axis('off')
        diff = np.abs(data_np[i]-r)
        axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
        axes[2, i].set_title('Error')
        axes[2, i].axis('off')
    plt.suptitle(f'Disentangled AE Reconstructions (struct={latent_struct}, context={latent_context}, channels={channels})')
    plt.tight_layout()
    plt.savefig('disentangled_recon_panel.png', dpi=200)
    plt.show()

    # Save artifacts
    out = {
        'spectrograms': data_np,
        'recon': recon.squeeze().numpy(),
        'z_struct': z_struct.numpy(),
        'z_context': z_context.numpy(),
        'mean': mean,
        'std': std,
        'losses': np.array(losses)
    }
    savemat('disentangled_results.mat', out)
    print('Saved disentangled_results.mat')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentangled autoencoder demonstration')
    parser.add_argument('--data-dir', default='/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_With_Airguns.dir', help='Root folder with .mat files')
    parser.add_argument('--n-samples', type=int, default=12, help='Number of samples to select')
    parser.add_argument('--seed', type=int, default=42, help='Seed for file selection & global stats')
    parser.add_argument('--show-files', action='store_true', help='List selected files')
    parser.add_argument('--latent-struct', type=int, default=LATENT_STRUCT_DEFAULT, help='Structure latent dimension')
    parser.add_argument('--latent-context', type=int, default=LATENT_CONTEXT_DEFAULT, help='Context latent dimension')
    parser.add_argument('--channels', type=int, default=CHANNELS_DEFAULT, help='Base channel count')
    parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT, help='Training epochs')
    parser.add_argument('--lr', type=float, default=LR_DEFAULT, help='Learning rate')
    parser.add_argument('--recompute-stats', action='store_true', help='Force recompute global normalization stats')
    args = parser.parse_args()

    run_disentangled_demo(args.data_dir, args.n_samples, args.seed, args.show_files,
                          args.latent_struct, args.latent_context, args.channels,
                          args.epochs, args.lr, args.recompute_stats)
