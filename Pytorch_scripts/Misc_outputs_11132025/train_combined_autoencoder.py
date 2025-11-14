#!/usr/bin/env python3
"""
Train improved autoencoder fresh on combined ManyAirguns + ManyWhaleCalls datasets.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse
from pathlib import Path
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
except ImportError:
    TSNE = None
    KMeans = None

# ---- Global defaults ----
CHANNELS_DEFAULT = 128
LATENT_DIM_DEFAULT = 128
SEED_DEFAULT = 42
EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 32
LR_DEFAULT = 1e-3

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

# Improved autoencoder without sigmoid (copy from demonstrate_reconstruction_issues.py)
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
    """Per-image min-max normalization."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    if im_max - im_min < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / (im_max - im_min)

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

class CombinedSNRDataset(Dataset):
    """Dataset combining ManyAirguns and ManyWhaleCalls directories."""
    def __init__(self, data_dirs: list[str], normalize: bool = True, 
                 seed: int | None = None, max_samples: int | None = None):
        """
        Args:
            data_dirs: List of root directories to scan for .mat files
            normalize: Apply per-image min-max normalization
            seed: Random seed for shuffling file order
            max_samples: Optional limit on total samples (for testing)
        """
        self.normalize = normalize
        self.data_dirs = data_dirs  # Store for label determination
        
        # Collect all .mat files from all directories with source labels
        all_files = []
        all_labels = []
        for dir_idx, data_dir in enumerate(data_dirs):
            mat_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True))
            all_files.extend(mat_files)
            # Label: 0 if "Airgun" in path, 1 if "WhaleCalls" in path
            for fp in mat_files:
                if 'Airgun' in fp or 'airgun' in fp:
                    all_labels.append(0)
                elif 'Whale' in fp or 'whale' in fp:
                    all_labels.append(1)
                else:
                    all_labels.append(dir_idx)  # Fallback: use directory index
        
        print(f"Found {len(all_files)} total .mat files across {len(data_dirs)} directories")
        
        # First pass: determine target shape and filter consistent files
        target_shape = None
        candidates = []
        candidate_labels = []
        for fp, label in zip(all_files, all_labels):
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
                candidate_labels.append(label)
            except Exception:
                continue
        
        if not candidates:
            raise RuntimeError(f"No valid SNR_gram data found with consistent shapes")
        
        print(f"Filtered to {len(candidates)} files with consistent shape {target_shape}")
        
        # Shuffle and optionally limit (keep labels aligned)
        if seed is not None:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(candidates))
            candidates = [candidates[i] for i in perm]
            candidate_labels = [candidate_labels[i] for i in perm]
        
        if max_samples is not None:
            candidates = candidates[:max_samples]
            candidate_labels = candidate_labels[:max_samples]
            print(f"Limited to {len(candidates)} samples")
        
        self.file_paths = candidates
        self.labels = candidate_labels
        self.target_shape = target_shape
        
        # Print label distribution
        label_counts = {}
        for lbl in self.labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"Label distribution: {label_counts}")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram', None)
            if self.normalize:
                im = _minmax_norm(im)
            # Return as [1, H, W] tensor
            return torch.from_numpy(im.astype(np.float32)).unsqueeze(0)
        except Exception as e:
            # Fallback to zeros if loading fails
            print(f"Warning: failed to load {fp}: {e}")
            return torch.zeros(1, *self.target_shape, dtype=torch.float32)
    
    def get_with_label(self, idx):
        """Get item with its dataset label (0=airgun, 1=whale)."""
        img = self.__getitem__(idx)
        label = self.labels[idx]
        return img, label

def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None,
                     epochs: int, lr: float, device: str, save_dir: str, 
                     target_shape: tuple[int, int], checkpoint_interval: int = 10):
    """Train autoencoder with validation and checkpointing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            output, _ = model(data)
            output = match_shape_center(output, target_shape)
            
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output, _ = model(data)
                    output = match_shape_center(output, target_shape)
                    loss = criterion(output, data)
                    epoch_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = epoch_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # Periodic checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if val_loader else None,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch+1:03d}.pth'))
    
    # Save final model
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1] if val_loader else None,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    return train_losses, val_losses

def visualize_reconstructions(model: nn.Module, dataset: Dataset, device: str, 
                              save_path: str, n_samples: int = 8):
    """Generate reconstruction visualizations."""
    model.eval()
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    samples = []
    for idx in indices:
        samples.append(dataset[idx])
    
    data = torch.stack(samples).to(device)
    
    with torch.no_grad():
        recon, _ = model(data)
        target_shape = (data.shape[2], data.shape[3])
        recon = match_shape_center(recon, target_shape)
    
    data = data.cpu().numpy()
    recon = recon.cpu().numpy()
    
    cols = min(8, n_samples)
    fig, axes = plt.subplots(3, cols, figsize=(15, 9))
    
    for i in range(cols):
        axes[0, i].imshow(data[i, 0], cmap='viridis', origin='lower', aspect='auto')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon[i, 0], cmap='viridis', origin='lower', aspect='auto')
        mse = np.mean((data[i, 0] - recon[i, 0])**2)
        axes[1, i].set_title(f'Recon\nMSE: {mse:.4f}')
        axes[1, i].axis('off')
        
        diff = np.abs(data[i, 0] - recon[i, 0])
        axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
        axes[2, i].set_title('Error')
        axes[2, i].axis('off')
    
    plt.suptitle('Autoencoder Reconstructions (Combined Dataset)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction visualization to {save_path}")

def visualize_latent_space(model: nn.Module, dataset: CombinedSNRDataset, device: str,
                           save_path: str, n_samples: int = 100, k_clusters: int = 2,
                           tsne_perplexity: float = 30.0, seed: int = 42):
    """Generate t-SNE visualization of latent space, colored by dataset source."""
    model.eval()
    
    # Sample data points from the dataset
    n_samples = min(n_samples, len(dataset))
    indices = np.random.RandomState(seed).choice(len(dataset), n_samples, replace=False)
    
    samples = []
    labels = []  # 0 for airguns, 1 for whale calls
    for idx in indices:
        img, label = dataset.get_with_label(idx)
        samples.append(img)
        labels.append(label)
    
    data = torch.stack(samples).to(device)
    labels = np.array(labels)
    
    # Extract latent embeddings
    with torch.no_grad():
        _, latent = model(data)
    
    latent_np = latent.cpu().numpy()
    
    # Perform t-SNE
    print(f"Computing t-SNE for {n_samples} samples...")
    perplexity = min(tsne_perplexity, (n_samples - 1) / 3.0)
    perplexity = max(2.0, perplexity)
    
    if TSNE is not None and n_samples > 2:
        try:
            tsne = TSNE(n_components=2, random_state=seed, init='pca',
                       perplexity=perplexity, learning_rate='auto')
            embedding = tsne.fit_transform(latent_np)
        except Exception as e:
            print(f"t-SNE failed: {e}, falling back to PCA")
            # PCA fallback
            latent_centered = latent_np - latent_np.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(latent_centered, full_matrices=False)
            embedding = latent_centered @ Vt[:2].T
    else:
        # PCA fallback
        latent_centered = latent_np - latent_np.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(latent_centered, full_matrices=False)
        embedding = latent_centered @ Vt[:2].T
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by true dataset source
    colors_true = ['#1f77b4' if l == 0 else '#ff7f0e' for l in labels]
    ax1.scatter(embedding[:, 0], embedding[:, 1], c=colors_true, alpha=0.7, s=30)
    ax1.set_title(f'Latent Space by Dataset Source (n={n_samples})')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label=f'Airguns (n={np.sum(labels==0)})'),
        Patch(facecolor='#ff7f0e', label=f'Whale Calls (n={np.sum(labels==1)})')
    ]
    ax1.legend(handles=legend_elements, loc='best')
    
    # Plot 2: KMeans clustering (unsupervised)
    if KMeans is not None and n_samples >= k_clusters:
        kmeans = KMeans(n_clusters=k_clusters, n_init='auto', random_state=seed)
        clusters = kmeans.fit_predict(latent_np)
        
        # Use distinct colors for clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, k_clusters))
        colors_cluster = [cluster_colors[c] for c in clusters]
        
        ax2.scatter(embedding[:, 0], embedding[:, 1], c=colors_cluster, alpha=0.7, s=30)
        ax2.set_title(f'Latent Space by KMeans Clustering (k={k_clusters})')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        
        # Add cluster legend
        legend_elements_cluster = [
            Patch(facecolor=cluster_colors[i], label=f'Cluster {i} (n={np.sum(clusters==i)})')
            for i in range(k_clusters)
        ]
        ax2.legend(handles=legend_elements_cluster, loc='best')
    else:
        ax2.text(0.5, 0.5, 'KMeans unavailable or\ninsufficient samples',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Clustering (unavailable)')
    
    plt.suptitle(f'Latent Space Visualization (perplexity={perplexity:.1f})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space visualization to {save_path}")
    
    # Compute and print separation metrics
    if len(np.unique(labels)) == 2:
        # Compute silhouette-like separation
        airgun_emb = embedding[labels == 0]
        whale_emb = embedding[labels == 1]
        if len(airgun_emb) > 0 and len(whale_emb) > 0:
            airgun_center = airgun_emb.mean(0)
            whale_center = whale_emb.mean(0)
            between_dist = np.linalg.norm(airgun_center - whale_center)
            within_airgun = np.mean([np.linalg.norm(x - airgun_center) for x in airgun_emb])
            within_whale = np.mean([np.linalg.norm(x - whale_center) for x in whale_emb])
            separation_ratio = between_dist / (within_airgun + within_whale + 1e-10)
            print(f"  Separation ratio: {separation_ratio:.3f} (higher = better separation)")


def main():
    parser = argparse.ArgumentParser(
        description="Train improved autoencoder on combined ManyAirguns + ManyWhaleCalls datasets"
    )
    parser.add_argument("--airgun-dir", 
                       default="/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyAirguns.dir",
                       help="Path to ManyAirguns dataset")
    parser.add_argument("--whale-dir",
                       default="/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyWhaleCalls.dir", 
                       help="Path to ManyWhaleCalls dataset")
    parser.add_argument("--save-dir", default="./trained_models", help="Directory to save models and outputs")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base channel count")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed")
    parser.add_argument("--no-normalize", action='store_true', help="Disable per-image normalization")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total samples (for testing)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to train on")
    parser.add_argument("--tsne-samples", type=int, default=100, 
                       help="Number of samples for t-SNE visualization")
    parser.add_argument("--k-clusters", type=int, default=2, 
                       help="Number of clusters for KMeans in latent space")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                       help="Perplexity parameter for t-SNE")
    
    args = parser.parse_args()
    
    # Set seed
    set_global_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"combined_ae_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    
    # Save config
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    # Load combined dataset
    print("\nLoading combined dataset...")
    data_dirs = [args.airgun_dir, args.whale_dir]
    full_dataset = CombinedSNRDataset(
        data_dirs, 
        normalize=not args.no_normalize,
        seed=args.seed,
        max_samples=args.max_samples
    )
    
    # Split into train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=(args.device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(args.device == 'cuda')
    ) if n_val > 0 else None
    
    # Create model
    target_shape = full_dataset.target_shape
    print(f"\nTarget shape: {target_shape}")
    model = ImprovedAutoencoder(
        nrow=target_shape[0],
        ncol=target_shape[1],
        latent_dim=args.latent_dim,
        base_channels=args.channels
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=save_dir,
        target_shape=target_shape,
        checkpoint_interval=args.checkpoint_interval
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Generate visualizations
    print("\nGenerating reconstruction visualizations...")
    visualize_reconstructions(
        model, full_dataset, args.device,
        os.path.join(save_dir, 'reconstructions_final.png'),
        n_samples=8
    )
    
    print("\nGenerating latent space t-SNE visualization...")
    visualize_latent_space(
        model, full_dataset, args.device,
        os.path.join(save_dir, 'latent_tsne.png'),
        n_samples=args.tsne_samples,
        k_clusters=args.k_clusters,
        tsne_perplexity=args.tsne_perplexity,
        seed=args.seed
    )
    
    print(f"\nAll outputs saved to: {save_dir}")
    print(f"Best model: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"Final model: {os.path.join(save_dir, 'final_model.pth')}")

if __name__ == "__main__":
    main()
