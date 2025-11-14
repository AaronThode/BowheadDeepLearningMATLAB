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
import math
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
try:
    from sklearn.manifold import TSNE  # optional; fallback to PCA if unavailable
    from sklearn.cluster import KMeans  # optional; for clustering
except Exception:  # pragma: no cover
    TSNE = None
    KMeans = None
import argparse

# ---- Global knobs (can be overridden via CLI) ----
# Base number of channels for convolutional blocks
CHANNELS_DEFAULT = 128
# Latent space dimensionality for both models
LATENT_DIM_DEFAULT = 128
# Default seed for deterministic file selection and initialization
SEED_DEFAULT = 42
# Default training epochs used by both models unless overridden via CLI
EPOCHS_DEFAULT = 60
# Enable extra 4th convolutional layer for deeper feature extraction
EXTRA_CONV_DEFAULT = False
# Default number of samples to render into JPEG panels
NUMBER_OUTPUT_IMAGE_SAMPLES = 200
# Columns per JPEG panel (each column is one spectrogram)
PANEL_GROUP_SIZE = 10
DEFAULT_VERSION_TAG = "01"

def set_global_seed(seed: int):
    "Best-effort determinism across Python, NumPy, and PyTorch."
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


def create_output_directory(version_tag: str | None = None) -> str:
    """Create and return the unique output directory for this run."""
    tag = (version_tag or DEFAULT_VERSION_TAG).strip().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"Autoencoder_v{tag}_Date{timestamp}.dir"
    output_dir = os.path.join(os.getcwd(), dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def select_samples_for_outputs(dataset: Dataset | None, data_tensor: torch.Tensor | None,
                               n_samples: int, seed: int | None) -> torch.Tensor:
    """Select up to n_samples spectrograms for JPEG export."""
    if (dataset is None or len(dataset) == 0) and (data_tensor is None or data_tensor.shape[0] == 0):
        raise RuntimeError("No data available for output sampling")
    rng = np.random.default_rng(seed)
    if dataset is not None and len(dataset) > 0:
        total = len(dataset)
        k = min(n_samples, total)
        indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
        samples = []
        for idx in indices:
            sample, _ = dataset[int(idx)]
            samples.append(sample.unsqueeze(0))
        return torch.cat(samples, dim=0).float()
    # Fall back to in-memory tensor (already shaped [N, 1, H, W])
    total = data_tensor.shape[0]
    k = min(n_samples, total)
    if k <= 0:
        raise RuntimeError("Data tensor is empty; cannot sample outputs")
    indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
    stacked = torch.stack([data_tensor[int(i)] for i in indices], dim=0)
    return stacked.float()


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "") -> int:
    """Save JPEG panels for original, reconstruction, and error across the provided samples.

    Returns the number of panel files written.
    """
    if samples is None or samples.shape[0] == 0:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    num_samples = samples.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    for group_idx in range(group_count):
        start = group_idx * PANEL_GROUP_SIZE
        end = min(start + PANEL_GROUP_SIZE, num_samples)
        batch = samples[start:end]
        with torch.no_grad():
            recon, _ = model(batch)
            recon = match_shape_center(recon, target_hw)
        orig_np = batch.squeeze(1).cpu().numpy()
        recon_np = recon.squeeze(1).cpu().numpy()
        diff_np = np.abs(orig_np - recon_np)
        cols = recon_np.shape[0]
        fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 9))
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)
        for col in range(cols):
            axes[0, col].imshow(orig_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[0, col].set_title(f'Input {start + col + 1}')
            axes[0, col].axis('off')
            axes[1, col].imshow(recon_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[1, col].set_title('Recon')
            axes[1, col].axis('off')
            axes[2, col].imshow(diff_np[col], cmap='hot', origin='lower', aspect='auto')
            axes[2, col].set_title('Error')
            axes[2, col].axis('off')
        if dataset_label:
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        panel_path = os.path.join(output_dir, f"{base_name}_{group_idx + 1:03d}.jpg")
        plt.tight_layout()
        plt.savefig(panel_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        panels_written += 1
    return panels_written

# Improved autoencoder without sigmoid
class ImprovedAutoencoder(nn.Module):
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, base_channels=CHANNELS_DEFAULT,
                 extra_conv=EXTRA_CONV_DEFAULT):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        
        # Calculate reduced dimensions based on depth
        if extra_conv:
            # With 4 pooling layers: divide by 16
            nrow_reduced = nrow // 16
            ncol_reduced = ncol // 16
        else:
            # With 3 pooling layers: divide by 8
            nrow_reduced = nrow // 8
            ncol_reduced = ncol // 8
        
        # Encoder with batch norm
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8  # Only used if extra_conv=True
        
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
        
        # Decoder without final activation
        if extra_conv:
            # Calculate output padding for final layer (must be < stride)
            # After 4x upsampling with stride 2, we go from nrow_reduced to nrow_reduced*16
            # We need to add padding to reach exactly nrow x ncol
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
                # No sigmoid: Allow unbounded output
            )
        else:
            # Calculate output padding for 3-layer case
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
                # No sigmoid:  Allow unbounded output
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

def _minmax_norm(im: np.ndarray, auto_skip_if_unit: bool = True) -> np.ndarray:
    """Min-max normalize unless data already resides in [0, 1]."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    if auto_skip_if_unit and (-1e-4 <= im_min <= 1.0 + 1e-4) and (-1e-4 <= im_max <= 1.0 + 1e-4):
        return im
    return (im - im_min) / rng

class CombinedSNRDataset(Dataset):
    """
    PyTorch Dataset for loading SNR_gram .mat files on-demand.
    Stores file paths and labels, loads data in __getitem__ for memory efficiency.
    """
    def __init__(self, directories: list[str], normalize: bool = True, 
                 seed: int | None = None, show_summary: bool = False):
        """
        Args:
            directories: List of directory paths to scan for .mat files
            normalize: Whether to apply per-image min-max normalization
            seed: Random seed for shuffling file order
            show_summary: Print dataset statistics
        """
        self.normalize = normalize
        self.file_paths: list[str] = []
        self.labels: list[int] = []
        
        # Scan each directory and determine target shape from first valid file
        target_shape = None
        for dir_idx, root in enumerate(directories):
            mat_files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
            
            for fp in mat_files:
                try:
                    m = loadmat(fp)
                    im = m.get('SNR_gram', None)
                    if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                        continue
                    
                    # Set target shape from first valid file
                    if target_shape is None:
                        target_shape = im.shape
                    
                    # Only include files matching target shape
                    if im.shape == target_shape:
                        self.file_paths.append(fp)
                        self.labels.append(dir_idx)
                except Exception:
                    continue
        
        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files found in {directories}")
        
        self.target_shape = target_shape
        
        # Shuffle with seed for reproducibility
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.file_paths))
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
        if show_summary:
            print(f"CombinedSNRDataset: {len(self)} files with shape {target_shape}")
            label_counts = {}
            for label in self.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            for label, count in sorted(label_counts.items()):
                print(f"  Label {label}: {count} files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and return a single sample with its label."""
        fp = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            m = loadmat(fp)
            im = m['SNR_gram']
            
            if self.normalize:
                im = _minmax_norm(im)
            else:
                im = im.astype(np.float32)
            
            # Convert to tensor (C, H, W) format
            tensor = torch.from_numpy(im).unsqueeze(0)  # Add channel dimension
            return tensor, label
            
        except Exception as e:
            # If loading fails, return zeros (should be rare with pre-validated files)
            print(f"Warning: Failed to load {fp}: {e}")
            h, w = self.target_shape
            return torch.zeros((1, h, w), dtype=torch.float32), label
    
    def get_with_label(self, idx):
        """Alias for __getitem__ that explicitly returns (data, label)."""
        return self.__getitem__(idx)

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

def load_snrgrams_from_multiple_folders(roots: list[str], n_samples: int = 100, normalize: bool = True,
                                        seed: int | None = None, show_files: bool = False,
                                        load_all: bool = False):
    """Load SNR_gram matrices from multiple dataset folders with source labels.
    
    Args:
        roots: List of root directories to scan
        n_samples: Total number of samples to load across all directories (ignored if load_all=True)
        normalize: Apply per-image min-max normalization
        seed: Random seed for deterministic selection
        show_files: Print selected files
        load_all: If True, load ALL files instead of sampling (ignores n_samples)
    
    Returns:
        data: np.ndarray of shape [N, H, W]
        labels: np.ndarray of shape [N] with 0=first dir (airguns), 1=second dir (whales), etc.
    """
    # Collect all files from all directories with labels
    all_files = []
    all_labels = []
    
    for dir_idx, root in enumerate(roots):
        mat_files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
        if not mat_files:
            print(f"Warning: No .mat files found under {root}")
            continue
        all_files.extend(mat_files)
        # Label by directory index or heuristic
        for fp in mat_files:
            if 'Airgun' in fp or 'airgun' in fp:
                all_labels.append(0)
            elif 'Whale' in fp or 'whale' in fp:
                all_labels.append(1)
            else:
                all_labels.append(dir_idx)
    
    if not all_files:
        raise FileNotFoundError(f"No .mat files found under any of the provided directories")
    
    print(f"Found {len(all_files)} total .mat files across {len(roots)} directories")
    
    # First pass: determine target shape and filter
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
        raise RuntimeError(f"Found .mat files but none with consistent SNR_gram shapes")
    
    print(f"Filtered to {len(candidates)} files with consistent shape {target_shape}")
    
    # Deterministic selection with seed (keeping labels aligned)
    if load_all:
        print(f"Loading ALL {len(candidates)} files (load_all=True)")
        selected_files = candidates
        selected_labels = candidate_labels
        k = len(candidates)
    else:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        k = min(n_samples, len(candidates))
        selected_indices = rng.choice(len(candidates), size=k, replace=False)
        selected_indices = np.sort(selected_indices)
        
        selected_files = [candidates[i] for i in selected_indices]
        selected_labels = [candidate_labels[i] for i in selected_indices]
        print(f"Randomly selected {k} out of {len(candidates)} files")
    
    # Print label distribution
    label_counts = {}
    for lbl in selected_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"Label distribution: {label_counts}")
    
    if show_files:
        print("Selected files (seeded):")
        for fp, lbl in zip(selected_files, selected_labels):
            print(f" - [{lbl}] {os.path.basename(fp)}")
    
    # Second pass: load selected files
    data = []
    labels = []
    for fp, label in zip(selected_files, selected_labels):
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram', None)
            if normalize:
                im = _minmax_norm(im)
            data.append(im.astype(np.float32))
            labels.append(label)
        except Exception:
            continue
    
    if not data:
        raise RuntimeError("Failed to load any SNR_gram data from the selected files.")
    
    if len(data) < n_samples:
        print(f"Loaded {len(data)} samples (requested {n_samples}); continuing with fewer.")
    
    arr = np.stack(data, axis=0)  # [N, H, W]
    labels_arr = np.array(labels, dtype=np.int32)
    return arr, labels_arr

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

def compare_autoencoder_performance(data_dir: str | list[str], n_samples: int = 8, normalize: bool = True,
                                    latent_dim: int = LATENT_DIM_DEFAULT, channels: int = CHANNELS_DEFAULT,
                                    seed: int | None = SEED_DEFAULT, show_files: bool = False,
                                    epochs: int = EPOCHS_DEFAULT, lr: float = 1e-3, progress_interval: int | None = None,
                                    k_clusters: int = 2, tsne_perplexity: float | None = None,
                                    tsne_samples: int | None = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                                    load_all: bool = False, batch_size: int = 32,
                                    output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                                    version_tag: str = DEFAULT_VERSION_TAG):
    """Train and visualize the improved autoencoder on real SNR_gram data from folder(s).
    
    Args:
        data_dir: Single directory path (str) or list of directories for combined dataset
        n_samples: Number of samples for reconstruction visualization
        tsne_samples: Number of samples for t-SNE plot (if None, uses n_samples; only for multi-dir)
        load_all: If True, load ALL files from directories (WARNING: may use lots of memory!)
        batch_size: Batch size for DataLoader training (only used when load_all=False with multi-dir)
    """
    # Ensure deterministic behavior for initialization and any stochastic ops
    if seed is not None:
        set_global_seed(int(seed))
    output_dir = create_output_directory(version_tag)
    print(f"Run artifacts will be stored in: {output_dir}")
    
    # Handle single vs multiple directories
    use_dataloader = False
    data_tensor = None
    train_loader = None
    dataset = None
    
    # Derive dataset label for plot annotations
    single_dir_path: str | None = None
    if isinstance(data_dir, list):
        dataset_label = ", ".join([os.path.basename(d.rstrip('/')) for d in data_dir])
    else:
        dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    if isinstance(data_dir, list):
        print(f"Loading SNR_gram data from {len(data_dir)} directories:")
        for d in data_dir:
            print(f"  - {d}")
        
        if load_all:
            print("WARNING: --load-all enabled - loading ALL files into memory!")
            data, labels = load_snrgrams_from_multiple_folders(data_dir, n_samples=0,  # ignored when load_all=True
                                                               normalize=normalize, seed=seed, 
                                                               show_files=show_files, load_all=True)
            data_tensor = torch.from_numpy(data).float().unsqueeze(1)  # Add channel dimension
        else:
            # Use DataLoader for memory-efficient batch training
            print("Using DataLoader for batch-wise training on all files...")
            dataset = CombinedSNRDataset(data_dir, normalize=normalize, seed=seed, show_summary=True)
            
            # Create DataLoader with configurable batch size
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            use_dataloader = True
            print(f"DataLoader batch size: {batch_size}")
            
            # For visualization and t-SNE, load a small subset into memory
            if tsne_samples is None:
                tsne_samples = max(100, n_samples)
            viz_samples = min(tsne_samples, len(dataset))
            print(f"Loading {viz_samples} samples for visualization...")
            
            # Load visualization subset
            data_list = []
            labels_list = []
            for i in range(viz_samples):
                sample, label = dataset[i]
                data_list.append(sample.unsqueeze(0))  # Add batch dim
                labels_list.append(label)
            
            data_tensor = torch.cat(data_list, dim=0)
            labels = np.array(labels_list)
        
        has_labels = True
    else:
        print(f"Loading SNR_gram data from: {data_dir}")
        single_dir_path = data_dir
        data = load_snrgrams_from_folder(data_dir, n_samples=n_samples, normalize=normalize,
                                         seed=seed, show_files=show_files)
        data_tensor = torch.from_numpy(data).float().unsqueeze(1)  # Add channel dimension
        labels = None
        has_labels = False
    
    # Get dimensions from either source
    if data_tensor is not None:
        nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
        print(f"Data shape: {data_tensor.shape}")
        print(f"Data range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    else:
        # Get dimensions from first sample in dataset
        sample, _ = dataset[0]
        nrow, ncol = sample.shape[-2], sample.shape[-1]
        print(f"Sample shape: (1, {nrow}, {ncol})")
        print(f"Total files in DataLoader: {len(dataset)}")
    
    print(f"Config -> latent_dim={latent_dim}, channels={channels}, extra_conv={extra_conv}, seed={seed}")
    
    # Create model
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    improved_model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, 
                                         base_channels=channels, extra_conv=extra_conv)
    print(f"Improved model parameters: {sum(p.numel() for p in improved_model.parameters()):,}")
    
    # Quick training function
    def quick_train(model, data, epochs=50, lr=0.001, model_name="model", progress_interval: int | None = None):
        """
        Train autoencoder on data.
        
        Args:
            model: Autoencoder model to train
            data: Either a torch.Tensor (pre-loaded) or a DataLoader (batch-wise)
            epochs: Number of training epochs
            lr: Learning rate
            model_name: Name for logging and output files
            progress_interval: If set, save reconstruction panels every N epochs
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        
        # Determine if using DataLoader or pre-loaded tensor
        is_dataloader = isinstance(data, DataLoader)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            if is_dataloader:
                # Batch-wise training
                for batch_data, _ in data:  # batch_data shape: (batch, 1, H, W)
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
                
                # Output stats to detect collapse (sample from last batch)
                with torch.no_grad():
                    o_min = float(output.min().cpu())
                    o_max = float(output.max().cpu())
                    o_mean = float(output.mean().cpu())
                if epoch % 10 == 0:
                    print(f"  {model_name} Epoch {epoch}: Loss={avg_loss:.4f} out[min={o_min:.3f}, max={o_max:.3f}, mean={o_mean:.3f}]")
                
            else:
                # Single-batch training (original behavior)
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

            # Optional progress panels (only for pre-loaded tensor mode)
            if progress_interval and progress_interval > 0 and (epoch % progress_interval == 0) and not is_dataloader:
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
                    plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
                    plt.tight_layout()
                    progress_path = os.path.join(output_dir, f'{model_name}_progress_epoch{epoch:03d}.png')
                    plt.savefig(progress_path, dpi=120)
                    plt.close(fig)
                except Exception:
                    pass
        
        return losses
    
    # Train models per configuration
    train_data = train_loader if use_dataloader else data_tensor
    
    print("\nTraining improved model...")
    improved_losses = quick_train(improved_model, train_data, epochs=epochs, lr=lr, model_name="improved", progress_interval=progress_interval)
    model_path = os.path.join(output_dir, 'improved_autoencoder.pth')
    torch.save(improved_model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    
    # Extract latent embeddings from ALL data for t-SNE
    improved_model.eval()
    
    if use_dataloader:
        # Extract latent embeddings from entire dataset
        print(f"\nExtracting latent embeddings from all {len(dataset)} samples for t-SNE...")
        all_latent = []
        all_labels_list = []
        
        with torch.no_grad():
            for i, (sample, label) in enumerate(dataset):
                sample_batch = sample.unsqueeze(0)  # Add batch dimension
                _, latent = improved_model(sample_batch)
                all_latent.append(latent.cpu())
                all_labels_list.append(label)
                
                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples...")
        
        # Combine all latent vectors
        improved_latent_full = torch.cat(all_latent, dim=0)
        labels_full = np.array(all_labels_list)
        print(f"Extracted {len(improved_latent_full)} latent embeddings")
        
        # Also compute reconstructions on visualization subset
        with torch.no_grad():
            improved_recon, improved_latent = improved_model(data_tensor)
            improved_recon = match_shape_center(improved_recon, (nrow, ncol))
    else:
        # Pre-loaded tensor mode: use existing approach
        with torch.no_grad():
            improved_recon, improved_latent = improved_model(data_tensor)
            improved_recon = match_shape_center(improved_recon, (nrow, ncol))
        
        # Default to visualization subset, but attempt full-directory traversal for t-SNE
        improved_latent_full = improved_latent
        labels_full = labels if labels is not None else None

        if single_dir_path is not None:
            try:
                tsne_dataset = CombinedSNRDataset([single_dir_path], normalize=normalize, seed=None, show_summary=False)
                tsne_loader = DataLoader(tsne_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                print(f"Computing latent embeddings for all {len(tsne_dataset)} samples in {single_dir_path}...")
                tsne_latents = []
                tsne_labels = []
                with torch.no_grad():
                    for batch_data, lbl in tsne_loader:
                        _, latent_b = improved_model(batch_data)
                        tsne_latents.append(latent_b.cpu())
                        tsne_labels.extend(lbl.tolist())
                if tsne_latents:
                    improved_latent_full = torch.cat(tsne_latents, dim=0)
                    labels_full = np.array(tsne_labels)
                    print(f"t-SNE will use {len(improved_latent_full)} embeddings from the full dataset")
                else:
                    print("Warning: Failed to assemble any embeddings from the full dataset; using visualization subset instead.")
            except Exception as e:
                print(f"Warning: unable to build full-dataset embeddings for t-SNE ({e}); falling back to visualization subset.")
    
    # Convert tensor to numpy for visualization
    data_np = data_tensor.squeeze(1).numpy()  # Remove channel dimension for plotting
    
    # Print value ranges for debugging
    print(f"\nValue ranges:")
    print(f"  Input data: [{data_np.min():.4f}, {data_np.max():.4f}]")
    print(f"  Improved recon: [{improved_recon.min():.4f}, {improved_recon.max():.4f}]")
    
    # Create reconstruction figure for inputs vs improved outputs
    # Use consistent vmin/vmax across all plots for fair comparison
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    
    cols = min(10, data_np.shape[0])
    fig, axes = plt.subplots(3, cols, figsize=(15, 9))
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        imp_recon = improved_recon[i, 0].numpy()
        axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Improved Reconstruction')
        axes[1, i].axis('off')
        diff = np.abs(data_np[i] - imp_recon)
        axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
        axes[2, i].set_title('Error')
        axes[2, i].axis('off')
    
    # Determine number of conv layers in improved model
    n_conv_layers = 4 if extra_conv else 3
    plt.suptitle(f'Autoencoder Reconstructions (latent_dim={latent_dim}, channels={channels}, conv_layers={n_conv_layers})')
    plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
    plt.tight_layout()
    recon_fig_path = os.path.join(output_dir, 'autoencoder_improved.png')
    plt.savefig(recon_fig_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # Loss curve and latent visualization
    n_conv_layers = 4 if extra_conv else 3
    plt.figure(figsize=(6, 4))
    plt.plot(improved_losses, label='Improved')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss latent_dim={latent_dim}, channels={channels}, conv_layers={n_conv_layers}')
    plt.yscale('log')
    plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
    plt.tight_layout()
    training_fig_path = os.path.join(output_dir, 'training_improved.png')
    plt.savefig(training_fig_path, dpi=150)
    plt.show()
    # Latent t-SNE visualization with optional clustering
    try:
        # Use full dataset embeddings if available, otherwise visualization subset
        if 'improved_latent_full' in locals() and improved_latent_full is not None:
            imp_z = improved_latent_full.detach().cpu().numpy()
            labels_for_tsne = labels_full
            print(f"\nComputing t-SNE on {len(imp_z)} samples from full dataset...")
        else:
            imp_z = improved_latent.detach().cpu().numpy()
            labels_for_tsne = labels
            print(f"\nComputing t-SNE on {len(imp_z)} samples from visualization subset...")
        
        n = imp_z.shape[0]
        if n >= 2:
            # Choose a safe perplexity (< number of samples)
            if tsne_perplexity is None:
                perplexity = float(max(2.0, min(30.0, (n - 1) / 3.0)))
            else:
                perplexity = float(tsne_perplexity)
            # Ensure it's strictly less than n
            if perplexity >= n:
                print(f"Warning: tsne_perplexity={perplexity} >= n={n}; reducing to {n-1:.1f}")
                perplexity = max(2.0, n - 1.0)
            if TSNE is not None and n > 2:
                try:
                    emb = TSNE(n_components=2,
                               random_state=int(seed) if seed is not None else 0,
                               init='pca',
                               perplexity=perplexity,
                               learning_rate='auto').fit_transform(imp_z)
                except Exception:
                    # Fallback to PCA if TSNE fails
                    Xc = imp_z - imp_z.mean(0, keepdims=True)
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                    emb = Xc @ Vt[:2].T
            else:
                # PCA fallback or degenerate small-N case
                Xc = imp_z - imp_z.mean(0, keepdims=True)
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                emb = Xc @ Vt[:2].T

            # Create visualization: if we have multiple labels, show both true labels and clusters
            if labels_for_tsne is not None and len(np.unique(labels_for_tsne)) > 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: Colored by true dataset source
                colors_true = ['#1f77b4' if l == 0 else '#ff7f0e' for l in labels_for_tsne]
                ax1.scatter(emb[:, 0], emb[:, 1], c=colors_true, alpha=0.7, s=30)
                ax1.set_title(f'Latent Space by Dataset Source (n={n})')
                ax1.set_xlabel('t-SNE 1')
                ax1.set_ylabel('t-SNE 2')
                
                # Custom legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#1f77b4', label=f'Airguns (n={np.sum(labels_for_tsne==0)})'),
                    Patch(facecolor='#ff7f0e', label=f'Whale Calls (n={np.sum(labels_for_tsne==1)})')
                ]
                ax1.legend(handles=legend_elements, loc='best')
                
                # Plot 2: KMeans clustering
                if KMeans is not None and n >= k_clusters:
                    k = int(k_clusters) if isinstance(k_clusters, int) else 2
                    if k < 2:
                        print(f"Warning: k-clusters {k} < 2; using k=2")
                        k = 2
                    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=int(seed) if seed is not None else 0)
                    clusters = kmeans.fit_predict(imp_z)
                    
                    cluster_colors = plt.cm.tab10(np.linspace(0, 1, k))
                    colors_cluster = [cluster_colors[c] for c in clusters]
                    
                    ax2.scatter(emb[:, 0], emb[:, 1], c=colors_cluster, alpha=0.7, s=30)
                    ax2.set_title(f'Latent Space by KMeans (k={k})')
                    ax2.set_xlabel('t-SNE 1')
                    ax2.set_ylabel('t-SNE 2')
                    
                    legend_elements_cluster = [
                        Patch(facecolor=cluster_colors[i], label=f'Cluster {i} (n={np.sum(clusters==i)})')
                        for i in range(k)
                    ]
                    ax2.legend(handles=legend_elements_cluster, loc='best')
                else:
                    ax2.text(0.5, 0.5, 'Clustering unavailable', ha='center', va='center', transform=ax2.transAxes)
                
                plt.suptitle(f'Improved latent t-SNE\nlatent_dim={latent_dim}, channels={channels}, perplexity={perplexity:.1f}')
                plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
                plt.tight_layout()
                tsne_path = os.path.join(output_dir, 'improved_latent_tsne.png')
                plt.savefig(tsne_path, dpi=160)
                plt.show()
                
                # Compute separation metric for binary label sets
                if len(np.unique(labels_for_tsne)) == 2:
                    airgun_emb = emb[labels_for_tsne == 0]
                    whale_emb = emb[labels_for_tsne == 1]
                    if len(airgun_emb) > 0 and len(whale_emb) > 0:
                        airgun_center = airgun_emb.mean(0)
                        whale_center = whale_emb.mean(0)
                        between_dist = np.linalg.norm(airgun_center - whale_center)
                        within_airgun = np.mean([np.linalg.norm(x - airgun_center) for x in airgun_emb])
                        within_whale = np.mean([np.linalg.norm(x - whale_center) for x in whale_emb])
                        separation_ratio = between_dist / (within_airgun + within_whale + 1e-10)
                        print(f"  Separation ratio: {separation_ratio:.3f} (higher = better separation)")
            else:
                # Single-plot clustering visualization
                try:
                    if KMeans is None:
                        raise RuntimeError("KMeans not available")
                    k = int(k_clusters) if isinstance(k_clusters, int) else 2
                    if k < 2:
                        print(f"Warning: k-clusters {k} < 2; using k=2")
                        k = 2
                    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=int(seed) if seed is not None else 0)
                    clusters = kmeans.fit_predict(imp_z)
                except Exception:
                    # Fallback heuristic: split by sign along first embedding axis
                    clusters = (emb[:, 0] > np.median(emb[:, 0])).astype(int)

                colors = np.where(clusters == 0, '#1f77b4', '#ff7f0e')
                plt.figure(figsize=(6, 5))
                plt.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.85, s=28)
                plt.title(f'Improved latent t-SNE (k={k_clusters})\nlatent_dim={latent_dim}, channels={channels}, perplexity={perplexity:.1f}')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
                plt.tight_layout()
                tsne_path = os.path.join(output_dir, 'improved_latent_tsne.png')
                plt.savefig(tsne_path, dpi=160)
                plt.show()
    except Exception as e:
        print(f"Warning: latent t-SNE plotting skipped due to error: {e}")
    
    # Save JPEG panels for randomly selected samples
    try:
        panel_samples = select_samples_for_outputs(dataset, data_tensor, output_samples, seed)
        panels_written = save_reconstruction_panels(improved_model, panel_samples, output_dir, (nrow, ncol),
                                                     dataset_label=dataset_label)
        print(f"Saved {panels_written} JPEG panel(s) covering {panel_samples.shape[0]} samples to {output_dir}")
    except Exception as e:
        print(f"Warning: unable to create JPEG panels: {e}")

    # Save sample data for further testing
    # Use the in-memory visualization tensor if available (data_tensor -> data_np), otherwise fall back
    try:
        spectrograms = data_np
    except NameError:
        if 'data_tensor' in locals() and data_tensor is not None:
            spectrograms = data_tensor.squeeze(1).numpy()
        else:
            spectrograms = None

    sample_data = {
        'spectrograms': spectrograms,
        'improved_recon': improved_recon.squeeze().numpy()
    }
    mat_path = os.path.join(output_dir, 'autoencoder_reconstruction_data.mat')
    savemat(mat_path, sample_data)
    print(f"Saved reconstruction data to {mat_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AEs on real SNR_gram data")
    parser.add_argument("--data-dir", 
                       default="/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyWhaleCalls.dir", 
                       help="Root folder containing .mat files (or use --data-dirs for multiple)")
    parser.add_argument("--data-dirs", nargs='+', default=None,
                       help="Multiple directories for combined dataset (e.g., --data-dirs /path/airguns /path/whales)")
    parser.add_argument("--n-samples", type=int, default=15, help="Number of samples for reconstruction visualization")
    parser.add_argument("--tsne-samples", type=int, default=None, 
                       help="Number of samples for t-SNE plot (default: 100 for multi-dir, n-samples otherwise)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent space size for both models")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base number of channels for conv blocks")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Training epochs for each model (global default EPOCHS_DEFAULT)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--progress-interval", type=int, default=0, help="If >0, save interim recon panels every N epochs")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Seed for deterministic sample selection")
    parser.add_argument("--show-files", action='store_true', help="Print the list of selected files")
    parser.add_argument("--k-clusters", type=int, default=2, help="Number of clusters for KMeans on latent embeddings (improved-only t-SNE)")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="t-SNE perplexity; leave unset for auto based on sample size")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT, 
                       help=f"Enable extra convolutional layer (4 layers instead of 3) for deeper feature extraction (default: {EXTRA_CONV_DEFAULT})")
    parser.add_argument("--no-extra-conv", dest='extra_conv', action='store_false',
                       help="Disable extra convolutional layer (use 3 layers instead of 4)")
    parser.add_argument("--load-all", action='store_true', help="Load ALL files from directory(ies) instead of sampling (WARNING: high memory usage!)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader training (default: 32, only used without --load-all)")
    parser.add_argument("--output-samples", type=int, default=NUMBER_OUTPUT_IMAGE_SAMPLES,
                       help="Number of random spectrograms to render into JPEG panels (default: 200)")
    parser.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG,
                       help="Version identifier used in the Autoencoder_vXX_Date*.dir output folder name")
    args = parser.parse_args()
    
    # Use multiple directories if specified, otherwise single directory
    data_input = args.data_dirs if args.data_dirs is not None else args.data_dir
    
    compare_autoencoder_performance(data_input,
                                    n_samples=args.n_samples,
                                    normalize=True,
                                    latent_dim=args.latent_dim,
                                    channels=args.channels,
                                    seed=args.seed,
                                    show_files=args.show_files,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    progress_interval=(args.progress_interval if args.progress_interval > 0 else None),
                                    k_clusters=args.k_clusters,
                                    tsne_perplexity=args.tsne_perplexity,
                                    tsne_samples=args.tsne_samples,
                                    extra_conv=args.extra_conv,
                                    load_all=args.load_all,
                                    batch_size=args.batch_size,
                                    output_samples=args.output_samples,
                                    version_tag=args.version_tag)