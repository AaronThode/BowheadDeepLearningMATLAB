#!/usr/bin/env python3
"""
Demonstrate autoencoder reconstruction quality issues and solutions.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch.nn.functional as F
from scipy.io import loadmat, savemat
try:
    from sklearn.manifold import TSNE  # optional; fallback to PCA if unavailable
except Exception:  # pragma: no cover
    TSNE = None
import argparse

# ---- Global knobs (can be overridden via CLI) ----
# Base number of channels for convolutional blocks
CHANNELS_DEFAULT = 128
# Latent space dimensionality for both models
LATENT_DIM_DEFAULT = 128
# Default seed for deterministic file selection and initialization
SEED_DEFAULT = 42
# Default training epochs used by both models unless overridden via CLI
EPOCHS_DEFAULT = 100

def set_global_seed(seed: int):
    """Best-effort determinism across Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# Recreate the problematic original autoencoder
class OriginalAutoencoder(nn.Module):
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, n_channels=CHANNELS_DEFAULT):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = int(nrow / 8)
        ncol_reduced = int(ncol / 8) 
        nel_reduced = nrow_reduced * ncol_reduced * n_channels
        
        # Original architecture from your script
        c1 = max(4, n_channels // 4)
        c2 = max(8, n_channels // 2)
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, n_channels, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(n_channels, c2, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(c1, 1, [2, 2], stride=[2, 2], output_padding=(1, 0))
        self.fc1 = nn.Linear(nel_reduced, latent_dim)
        self.fc2 = nn.Linear(latent_dim, nel_reduced)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.nel_reduced = nel_reduced
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.n_channels = n_channels

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.nel_reduced)
        latent = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(latent))
        x = x.view(-1, self.n_channels, self.nrow_reduced, self.ncol_reduced)
        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        output = torch.sigmoid(self.t_conv3(x))  # PROBLEM: Sigmoid constraint!
        return output, latent

# Improved autoencoder without sigmoid
class ImprovedAutoencoder(nn.Module):
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, base_channels=CHANNELS_DEFAULT):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = nrow // 8
        ncol_reduced = ncol // 8
        
        # Encoder with batch norm
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
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
        
        # Decoder without final activation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 2, stride=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(c2, c1, 2, stride=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8)),
            # No sigmoid! Allow unbounded output
        )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.base_channels * 4, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent

def _minmax_norm(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    if im_max - im_min < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / (im_max - im_min)

def load_snrgrams_from_folder(root: str, n_samples: int = 8, normalize: bool = True,
                              seed: int | None = None, show_files: bool = False):
    """Load up to n_samples SNR_gram matrices with identical shape from a dataset folder.
    Searches recursively for .mat files, determines the first valid target shape encountered,
    collects all files of that shape, then picks n_samples using a seeded random selection.
    """
    mat_files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {root}")

    # First pass: identify target shape and collect candidate files of that shape
    target_shape = None
    candidates: list[str] = []
    for fp in mat_files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram', None)
            if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                continue
            if target_shape is None:
                target_shape = im.shape
            if im.shape != target_shape:
                continue
            candidates.append(fp)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(f"Found .mat files but none with consistent SNR_gram shapes under {root}")

    # Deterministic selection with seed
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    k = min(n_samples, len(candidates))
    selected_indices = rng.choice(len(candidates), size=k, replace=False)
    selected_files = [candidates[i] for i in np.sort(selected_indices)]

    if show_files:
        print("Selected files (seeded):")
        for fp in selected_files:
            try:
                print(" -", os.path.relpath(fp, root))
            except Exception:
                print(" -", fp)

    # Second pass: load only the selected files
    data: list[np.ndarray] = []
    for fp in selected_files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram', None)
            if normalize:
                im = _minmax_norm(im)
            data.append(im.astype(np.float32))
        except Exception:
            continue

    if not data:
        raise RuntimeError("Failed to load any SNR_gram data from the selected files.")
    if len(data) < n_samples:
        print(f"Loaded {len(data)} samples (requested {n_samples}); continuing with fewer.")

    arr = np.stack(data, axis=0)  # [N, H, W]
    return arr

def match_shape_center(recon: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Center-crop or pad recon to match (H,W)."""
    _, _, rH, rW = recon.shape
    tH, tW = target_hw
    # center crop
    if rH > tH:
        dh = (rH - tH) // 2
        recon = recon[:, :, dh:dh + tH, :]
        rH = tH
    if rW > tW:
        dw = (rW - tW) // 2
        recon = recon[:, :, :, dw:dw + tW]
        rW = tW
    # pad
    padH = tH - rH
    padW = tW - rW
    if padH > 0 or padW > 0:
        pad_left = padW // 2 if padW > 0 else 0
        pad_right = padW - pad_left if padW > 0 else 0
        pad_top = padH // 2 if padH > 0 else 0
        pad_bottom = padH - pad_top if padH > 0 else 0
        recon = F.pad(recon, (pad_left, pad_right, pad_top, pad_bottom))
    return recon

def compare_autoencoder_performance(data_dir: str, n_samples: int = 8, normalize: bool = True,
                                    latent_dim: int = LATENT_DIM_DEFAULT, channels: int = CHANNELS_DEFAULT,
                                    seed: int | None = SEED_DEFAULT, show_files: bool = False,
                                    improved_only: bool = False,
                                    epochs: int = EPOCHS_DEFAULT, lr: float = 1e-3, progress_interval: int | None = None):
    """Compare original vs improved autoencoder on real SNR_gram data from a folder."""
    # Ensure deterministic behavior for initialization and any stochastic ops
    if seed is not None:
        set_global_seed(int(seed))
    print(f"Loading SNR_gram data from: {data_dir}")
    data = load_snrgrams_from_folder(data_dir, n_samples=n_samples, normalize=normalize,
                                     seed=seed, show_files=show_files)
    data_tensor = torch.from_numpy(data).float().unsqueeze(1)  # Add channel dimension
    
    print(f"Data shape: {data_tensor.shape}")
    print(f"Data range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    print(f"Config -> latent_dim={latent_dim}, channels={channels}, seed={seed}")
    
    # Create models
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    improved_model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, base_channels=channels)
    if not improved_only:
        original_model = OriginalAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, n_channels=channels)
        print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"Improved model parameters: {sum(p.numel() for p in improved_model.parameters()):,}")
    
    # Quick training function
    def quick_train(model, data, epochs=50, lr=0.001, model_name="model", progress_interval: int | None = None):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = model(data)
            output = match_shape_center(output, (nrow, ncol))
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Output stats to detect collapse
            with torch.no_grad():
                o_min = float(output.min().cpu())
                o_max = float(output.max().cpu())
                o_mean = float(output.mean().cpu())
            if epoch % 10 == 0:
                print(f"  {model_name} Epoch {epoch}: Loss={loss.item():.4f} out[min={o_min:.3f}, max={o_max:.3f}, mean={o_mean:.3f}]")

            # Optional progress panels
            if progress_interval and progress_interval > 0 and (epoch % progress_interval == 0):
                try:
                    cols = min(5, data.shape[0])
                    fig, axes = plt.subplots(2, cols, figsize=(3*cols, 6))
                    for i in range(cols):
                        axes[0, i].imshow(data[i,0].cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
                        axes[0, i].set_title(f'Orig {i+1}')
                        axes[0, i].axis('off')
                        axes[1, i].imshow(output[i,0].detach().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
                        axes[1, i].set_title('Recon')
                        axes[1, i].axis('off')
                    plt.suptitle(f'{model_name} Recon Progress (epoch {epoch})')
                    plt.tight_layout()
                    plt.savefig(f'{model_name}_progress_epoch{epoch:03d}.png', dpi=120)
                    plt.close(fig)
                except Exception:
                    pass
        
        return losses
    
    # Train models per configuration
    if not improved_only:
        print("\nTraining original model...")
        original_losses = quick_train(original_model, data_tensor, epochs=epochs, lr=lr, model_name="original", progress_interval=progress_interval)
    
    print("\nTraining improved model...")
    improved_losses = quick_train(improved_model, data_tensor, epochs=epochs, lr=lr, model_name="improved", progress_interval=progress_interval)
    
    # Evaluate reconstructions
    improved_model.eval()
    if not improved_only:
        original_model.eval()
    
    with torch.no_grad():
        improved_recon, improved_latent = improved_model(data_tensor)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol))
        if not improved_only:
            original_recon, original_latent = original_model(data_tensor)
            original_recon = match_shape_center(original_recon, (nrow, ncol))
    
    # Create comparison figure (toggle original model rows/MSE)
    cols = min(10, data.shape[0])
    if improved_only:
        fig, axes = plt.subplots(3, cols, figsize=(15, 9))
        for i in range(cols):
            axes[0, i].imshow(data[i], cmap='viridis', origin='lower', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            imp_recon = improved_recon[i, 0].numpy()
            axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto')
            axes[1, i].set_title('Improved Model')  # no per-sample MSE shown
            axes[1, i].axis('off')
            diff = np.abs(data[i] - imp_recon)
            axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title('Error')
            axes[2, i].axis('off')
        plt.suptitle(f'Autoencoder: Improved Only (latent_dim={latent_dim}, channels={channels})\nTop: Originals, Middle: Improved Model, Bottom: Error')
        plt.tight_layout()
        plt.savefig('autoencoder_improved_only.png', dpi=200, bbox_inches='tight')
        plt.show()
    else:
        fig, axes = plt.subplots(4, cols, figsize=(15, 12))
        for i in range(cols):
            axes[0, i].imshow(data[i], cmap='viridis', origin='lower', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            orig_recon = original_recon[i, 0].numpy()
            axes[1, i].imshow(orig_recon, cmap='viridis', origin='lower', aspect='auto')
            mse_orig = np.mean((data[i] - orig_recon)**2)
            axes[1, i].set_title(f'Original Model\nMSE: {mse_orig:.3f}')
            axes[1, i].axis('off')
            imp_recon = improved_recon[i, 0].numpy()
            axes[2, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto')
            mse_imp = np.mean((data[i] - imp_recon)**2)
            axes[2, i].set_title(f'Improved Model\nMSE: {mse_imp:.3f}')
            axes[2, i].axis('off')
            diff = np.abs(data[i] - orig_recon)
            axes[3, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[3, i].set_title('Error (Original)')
            axes[3, i].axis('off')
        plt.suptitle(f'Autoencoder Comparison: Original vs Improved (latent_dim={latent_dim}, channels={channels})\nTop: Originals, 2nd: Original Model, 3rd: Improved Model, Bottom: Error')
        plt.tight_layout()
        plt.savefig('autoencoder_comparison.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    # Loss comparison plots (toggle original and MSE bar chart)
    if improved_only:
        plt.figure(figsize=(6, 4))
        plt.plot(improved_losses, label='Improved')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (Improved only) latent_dim={latent_dim}, channels={channels}')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('training_improved_only.png', dpi=150)
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(original_losses, label='Original (with sigmoid)')
        plt.plot(improved_losses, label='Improved (without sigmoid)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Training Loss Comparison (latent_dim={latent_dim}, channels={channels})')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # Latent t-SNE visualization instead of MSE bar
        orig_z = original_latent.detach().cpu().numpy()
        imp_z = improved_latent.detach().cpu().numpy()
        X = np.vstack([orig_z, imp_z])
        labels = np.array([0] * orig_z.shape[0] + [1] * imp_z.shape[0])
        # Choose a safe perplexity (< number of samples)
        n = X.shape[0]
        perplexity = max(2, min(30, (n - 1) // 3))
        if TSNE is not None and n > 2:
            try:
                emb = TSNE(n_components=2, random_state=int(seed) if seed is not None else 0,
                           init='pca', perplexity=perplexity, learning_rate='auto').fit_transform(X)
            except Exception:
                # Fallback to PCA if TSNE fails
                Xc = X - X.mean(0, keepdims=True)
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                emb = Xc @ Vt[:2].T
        else:
            # PCA fallback or degenerate small-N case
            Xc = X - X.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            emb = Xc @ Vt[:2].T
        # Plot scatter
        colors = np.where(labels == 0, '#d62728', '#2ca02c')  # red for original, green for improved
        plt.scatter(emb[labels == 0, 0], emb[labels == 0, 1], c='#d62728', label='Original latent', alpha=0.8, s=24)
        plt.scatter(emb[labels == 1, 0], emb[labels == 1, 1], c='#2ca02c', label='Improved latent', alpha=0.8, s=24)
        plt.title(f'Latent t-SNE (latent_dim={latent_dim}, channels={channels})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('training_comparison.png', dpi=150)
        plt.show()
    
    # Save sample data for further testing
    if improved_only:
        sample_data = {
            'spectrograms': data,
            'improved_recon': improved_recon.squeeze().numpy()
        }
    else:
        sample_data = {
            'spectrograms': data,
            'original_recon': original_recon.squeeze().numpy(),
            'improved_recon': improved_recon.squeeze().numpy()
        }
    savemat('autoencoder_comparison_data.mat', sample_data)
    print("Saved comparison data to autoencoder_comparison_data.mat")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AEs on real SNR_gram data")
    parser.add_argument("--data-dir", default="/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_With_Airguns.dir", help="Root folder containing .mat files")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to load")
    parser.add_argument("--no-normalize", action='store_true', help="Disable per-image min-max normalization")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent space size for both models")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base number of channels for conv blocks")
    parser.add_argument("--improved-only", action='store_true', help="Skip original model training and plots; hide MSE comparison")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Training epochs for each model (global default EPOCHS_DEFAULT)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--progress-interval", type=int, default=0, help="If >0, save interim recon panels every N epochs")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Seed for deterministic sample selection")
    parser.add_argument("--show-files", action='store_true', help="Print the list of selected files")
    args = parser.parse_args()
    compare_autoencoder_performance(args.data_dir,
                                    n_samples=args.n_samples,
                                    normalize=not args.no_normalize,
                                    latent_dim=args.latent_dim,
                                    channels=args.channels,
                                    seed=args.seed,
                                    show_files=args.show_files,
                                    improved_only=args.improved_only,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    progress_interval=(args.progress_interval if args.progress_interval > 0 else None))