#!/usr/bin/env python3
"""
Train and evaluate an improved autoencoder on SNR_gram spectrograms.

COMPUTATIONAL ORDER:
1. Parse command-line arguments (see main block at bottom)
2. Create versioned output directory (Autoencoder_vXX_DateYYYYMMDD-HHMMSS.dir)
3. Load dataset from specified directories (single or multiple)
4. Build and initialize autoencoder model architecture
5. Train the model using batch-wise gradient descent
6. Extract latent embeddings from trained model
7. Generate visualization plots (reconstructions, training curves, t-SNE, JPEG panels)
8. Save trained model weights and all output artifacts

SCRIPT PURPOSE:
- Trains an improved autoencoder WITHOUT sigmoid constraints (allows unbounded output)
- Supports both in-memory training (small datasets) and DataLoader-based training (large datasets)
- Generates reconstruction quality plots comparing input vs output
- Performs t-SNE visualization of latent space with optional KMeans clustering
- Saves trained model weights for later use with Apply_Autoencoder.py
- Creates JPEG panels showing reconstruction quality on random samples
- All outputs saved to timestamped directory for reproducibility
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

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================
# These defaults can be overridden via command-line arguments

# Architecture parameters
CHANNELS_DEFAULT = 64           # Base number of channels in convolutional layers
LATENT_DIM_DEFAULT = 128         # Dimensionality of latent space bottleneck
EXTRA_CONV_DEFAULT = False       # Enable 4th conv layer (deeper feature extraction)

# Training parameters
EPOCHS_DEFAULT = 100             # Default number of training epochs
LR_DEFAULT = 1e-3                # Default learning rate for Adam optimizer
SEED_DEFAULT = 42                # Random seed for reproducible results

# Output parameters
NUMBER_OUTPUT_IMAGE_SAMPLES = 1000  # Number of spectrograms for JPEG panel generation
PANEL_GROUP_SIZE = 10              # Spectrograms per JPEG panel (columns)
SHOW_ERROR_PLOTS = False           # Whether to include error row in reconstruction panels
DEFAULT_VERSION_TAG = "01"         # Version identifier for output directory naming


# ============================================================================
# UTILITY FUNCTIONS (Called throughout processing)
# ============================================================================

def set_global_seed(seed: int):
    """
    Set random seeds for reproducible results across all libraries.
    
    RUNS EARLY: Called at start of training to ensure deterministic behavior
    in random initialization, data sampling, and stochastic operations.
    """
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
    """
    Create and return the unique output directory for this training run.
    
    RUNS DURING STEP 2: Creates Autoencoder_vXX_DateYYYYMMDD-HHMMSS.dir folder
    where all outputs (model weights, plots, JPEG panels) will be saved.
    
    Args:
        version_tag: Version identifier (default: "01")
    
    Returns:
        Absolute path to created output directory
    """
    tag = (version_tag or DEFAULT_VERSION_TAG).strip().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"Autoencoder_v{tag}_Date{timestamp}.dir"
    output_dir = os.path.join(os.getcwd(), dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def select_samples_for_outputs(dataset: Dataset | None, data_tensor: torch.Tensor | None,
                               n_samples: int, seed: int | None) -> tuple[torch.Tensor, list[str]]:
    """
    Select up to n_samples spectrograms for JPEG reconstruction panel export.
    
    RUNS DURING STEP 7: After training completes, randomly samples images
    to visualize reconstruction quality in multi-panel JPEG files.
    
    Args:
        dataset: PyTorch Dataset (if using DataLoader mode)
        data_tensor: Pre-loaded tensor (if using in-memory mode)
        n_samples: Number of samples to select
        seed: Random seed for reproducible sampling
    
    Returns:
        Tuple of:
        - Tensor of shape (n_samples, 1, H, W) containing selected spectrograms
        - List of filenames (or empty strings if not available)
    """
    if (dataset is None or len(dataset) == 0) and (data_tensor is None or data_tensor.shape[0] == 0):
        raise RuntimeError("No data available for output sampling")
    rng = np.random.default_rng(seed)
    if dataset is not None and len(dataset) > 0:
        total = len(dataset)
        k = min(n_samples, total)
        indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
        samples = []
        filenames = []
        for idx in indices:
            sample, _ = dataset[int(idx)]
            samples.append(sample.unsqueeze(0))
            # Get filename if dataset has file_paths attribute
            if hasattr(dataset, 'file_paths'):
                filenames.append(os.path.basename(dataset.file_paths[int(idx)]))
            else:
                filenames.append("")
        return torch.cat(samples, dim=0).float(), filenames
    # Fall back to in-memory tensor (already shaped [N, 1, H, W])
    total = data_tensor.shape[0]
    k = min(n_samples, total)
    if k <= 0:
        raise RuntimeError("Data tensor is empty; cannot sample outputs")
    indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
    stacked = torch.stack([data_tensor[int(i)] for i in indices], dim=0)
    # No filenames available for in-memory tensors
    return stacked.float(), [""] * k


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "", filenames: list[str] = None, 
                               show_error: bool = SHOW_ERROR_PLOTS) -> int:
    """
    Save JPEG panels showing original, reconstruction, and optionally error for multiple samples.
    
    RUNS DURING STEP 7: Creates multiple JPEG files, each showing PANEL_GROUP_SIZE
    spectrograms (default 10) arranged in 2 or 3 rows: original input, reconstruction,
    and optionally absolute error map. This provides visual assessment of reconstruction quality.
    
    Args:
        model: Trained autoencoder model
        samples: Tensor of spectrograms to reconstruct (N, 1, H, W)
        output_dir: Directory to save JPEG panels
        target_hw: Target (height, width) for reconstruction matching
        base_name: Base filename for output JPEGs
        dataset_label: Dataset name to display on plots
        filenames: List of filenames to use as titles (optional)
        show_error: Whether to include error row in output
    
    Returns:
        Number of JPEG panel files written
    """
    if samples is None or samples.shape[0] == 0:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    num_samples = samples.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    
    # Determine number of rows
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
            # Get filename for this sample if available
            if filenames and sample_idx < len(filenames) and filenames[sample_idx]:
                # Remove .mat extension and truncate if too long
                title = filenames[sample_idx].replace('.mat', '')
                if len(title) > 25:
                    title = title[:22] + '...'
            else:
                title = f'Input {sample_idx + 1}'
            
            # Original image
            axes[0, col].imshow(orig_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[0, col].set_title(title, fontsize=6)
            axes[0, col].axis('off')
            
            # Reconstruction (no title)
            axes[1, col].imshow(recon_np[col], cmap='viridis', origin='lower', aspect='auto')
            axes[1, col].axis('off')
            
            # Error row (optional)
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
# MODEL ARCHITECTURE (Convolutional Autoencoder with Batch Normalization)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """
    Improved autoencoder architecture WITHOUT sigmoid activation in decoder.
    
    ARCHITECTURE:
    - Encoder: 3 or 4 convolutional blocks (depending on extra_conv flag)
      - Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d
      - Progressively increases channels: base → 2x → 4x → 8x (if extra_conv)
    - Latent bottleneck: Flattened features → FC layers → latent_dim vector
    - Decoder: Mirrors encoder with transposed convolutions
      - Each block: ConvTranspose2d → BatchNorm2d → ReLU
      - Final layer: NO sigmoid (allows unbounded output for better reconstruction)
    
    KEY IMPROVEMENT over original: Batch normalization stabilizes training,
    and removal of sigmoid allows model to match input dynamic range exactly.
    
    Args:
        nrow, ncol: Input spectrogram dimensions (height, width)
        latent_dim: Dimensionality of latent space bottleneck
        base_channels: Base number of channels in first conv layer
        extra_conv: If True, use 4 conv layers (deeper), else 3 layers
    """
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
        """
        Forward pass through encoder and decoder.
        
        TRAINING: This is called during each training iteration to compute
        reconstruction loss and update weights via backpropagation.
        
        Args:
            x: Input tensor of shape (batch, 1, H, W)
        
        Returns:
            output: Reconstructed spectrogram (batch, 1, H, W)
            latent: Latent space representation (batch, latent_dim)
        """
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent

# ============================================================================
# DATA PREPROCESSING (Normalization)
# ============================================================================

def _minmax_norm(im: np.ndarray, auto_skip_if_unit: bool = True) -> np.ndarray:
    """
    Min-max normalize image to [0, 1] range unless already normalized.
    
    RUNS DURING DATA LOADING: Applied to each spectrogram as it's loaded
    from .mat files to ensure consistent input scaling across dataset.
    
    Args:
        im: Input image array (2D spectrogram)
        auto_skip_if_unit: If True, skip normalization if data already in [0,1]
    
    Returns:
        Normalized image in [0, 1] range
    """
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
# DATASET CLASSES (PyTorch Dataset implementations for .mat file loading)
# ============================================================================

class CombinedSNRDataset(Dataset):
    """
    PyTorch Dataset for loading SNR_gram .mat files on-demand from disk.
    
    MEMORY EFFICIENT: Stores only file paths in memory, loads spectrograms
    on-demand in __getitem__(). This allows training on datasets that don't
    fit entirely in RAM.
    
    MULTI-DIRECTORY SUPPORT: Can combine spectrograms from multiple folders,
    assigning labels based on source directory (e.g., label 0 for airguns,
    label 1 for whale calls).
    
    SHAPE VALIDATION: Automatically filters files to ensure all spectrograms
    have consistent dimensions (required for batching).
    
    Args:
        directories: List of directory paths to scan for .mat files
        normalize: Whether to apply per-image min-max normalization
        seed: Random seed for shuffling file order
        show_summary: Print dataset statistics (file counts per label)
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
        """
        Load and return a single sample with its label.
        
        CALLED DURING TRAINING: PyTorch DataLoader calls this method for each
        sample in each batch during training iterations.
        
        Returns:
            tensor: Spectrogram as (1, H, W) tensor
            label: Integer label indicating source directory
        """
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

# ============================================================================
# DATA LOADING FUNCTIONS (Legacy functions for in-memory loading)
# ============================================================================
# NOTE: These functions load entire datasets into RAM. For large datasets,
# prefer using CombinedSNRDataset with DataLoader (--load-all flag disabled).

def load_snrgrams_from_folder(root: str, n_samples: int = 8, normalize: bool = True,
                              seed: int | None = None, show_files: bool = False):
    """
    Load up to n_samples SNR_gram matrices into memory from a single folder.
    
    LEGACY IN-MEMORY MODE: Loads spectrograms directly into a NumPy array.
    Used when dataset is small enough to fit in RAM for faster training.
    
    PROCESS:
    1. Recursively scans directory for .mat files
    2. Determines consistent shape from first valid file
    3. Filters files to match target shape
    4. Randomly selects n_samples files (seeded for reproducibility)
    5. Loads selected files into memory
    
    Args:
        root: Directory path to scan
        n_samples: Number of samples to load
        normalize: Apply min-max normalization
        seed: Random seed for deterministic selection
        show_files: Print list of selected files
    
    Returns:
        NumPy array of shape (n_samples, H, W)
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
    """
    Load SNR_gram matrices from multiple folders with source labels.
    
    MULTI-SOURCE IN-MEMORY MODE: Combines data from multiple directories,
    maintaining labels for each source (useful for airgun vs whale call
    separation in visualizations).
    
    Args:
        roots: List of root directories to scan
        n_samples: Total samples to load across all directories (ignored if load_all=True)
        normalize: Apply per-image min-max normalization
        seed: Random seed for deterministic selection
        show_files: Print selected files
        load_all: If True, load ALL files instead of sampling (WARNING: high memory usage!)
    
    Returns:
        data: np.ndarray of shape (N, H, W)
        labels: np.ndarray of shape (N,) with directory indices as labels
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

# ============================================================================
# SHAPE MATCHING UTILITY
# ============================================================================

def match_shape_center(recon: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Center-crop or pad reconstruction to match target dimensions.
    
    RUNS DURING TRAINING & EVALUATION: Ensures decoder output matches
    exact input dimensions (needed due to pooling/upsampling rounding).
    
    Args:
        recon: Reconstructed tensor (batch, 1, H, W)
        target_hw: Target (height, width) tuple
    
    Returns:
        Reshaped tensor matching target dimensions exactly
    """
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

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def compare_autoencoder_performance(data_dir: str, n_samples: int = 8, normalize: bool = True,
                                    latent_dim: int = LATENT_DIM_DEFAULT, channels: int = CHANNELS_DEFAULT,
                                    seed: int | None = SEED_DEFAULT, show_files: bool = False,
                                    epochs: int = EPOCHS_DEFAULT, lr: float = 1e-3, progress_interval: int | None = None,
                                    k_clusters: int = 2, tsne_perplexity: float | None = None,
                                    tsne_samples: int | None = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                                    batch_size: int = 32,
                                    output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                                    version_tag: str = DEFAULT_VERSION_TAG, show_error: bool = SHOW_ERROR_PLOTS):
    """
    Train and visualize the improved autoencoder on SNR_gram spectrograms.
    
    MAIN WORKFLOW (8 steps):
    1. Initialize: Set random seed, create output directory
    2. Load data: From single directory using DataLoader
    3. Build model: Initialize ImprovedAutoencoder architecture
    4. Train: Optimize reconstruction loss over specified epochs
    5. Extract: Compute latent embeddings for all/subset of data
    6. Visualize: Generate reconstruction comparison plots
    7. Analyze: Create t-SNE plots with optional KMeans clustering
    8. Export: Save model weights, plots, JPEG panels, and .mat files
    
    VISUALIZATION:
    - Reconstruction quality: Input vs output comparison (up to 10 samples)
    - Training curves: Loss over epochs (log scale)
    - t-SNE: 2D projection of latent space with KMeans clustering
    - JPEG panels: Grid of reconstructions with filenames (default: 5000 samples)
    
    Args:
        data_dir: Directory path containing .mat files
        n_samples: Number of samples for reconstruction visualization plots
        tsne_samples: Number of samples for t-SNE (if None, uses n_samples)
        batch_size: Batch size for DataLoader training
        output_samples: Number of samples for JPEG panel generation
        version_tag: Version identifier for output directory naming
        show_error: Whether to include error row in reconstruction panels
    """
    # ========================================================================
    # STEP 1: Initialize random seed and create output directory
    # ========================================================================
    # Ensure deterministic behavior for initialization and any stochastic ops
    if seed is not None:
        set_global_seed(int(seed))
    output_dir = create_output_directory(version_tag)
    print(f"Run artifacts will be stored in: {output_dir}")
    
    # ========================================================================
    # STEP 2: Load dataset from directory
    # ========================================================================
    print(f"Loading SNR_gram data from: {data_dir}")
    dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    # Use Dataset class to preserve filenames for panel generation
    dataset = CombinedSNRDataset([data_dir], normalize=normalize, seed=seed, show_summary=True)
    
    # Create DataLoader for training
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"DataLoader batch size: {batch_size}")
    
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
    
    # ========================================================================
    # STEP 3: Get dimensions and initialize model architecture
    # ========================================================================
    # Get dimensions from visualization data
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    print(f"Data shape: {data_tensor.shape}")
    print(f"Data range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    print(f"Total files in dataset: {len(dataset)}")
    print(f"Config -> latent_dim={latent_dim}, channels={channels}, extra_conv={extra_conv}, seed={seed}")
    
    # Create model with specified architecture
    improved_model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, 
                                         base_channels=channels, extra_conv=extra_conv)
    print(f"Improved model parameters: {sum(p.numel() for p in improved_model.parameters()):,}")
    
    # ========================================================================
    # STEP 4: Define training function
    # ========================================================================
    # Quick training function
    def quick_train(model, data, epochs=50, lr=0.001, model_name="model", progress_interval: int | None = None):
        """
        Train autoencoder using MSE reconstruction loss.
        
        TRAINING LOOP:
        1. Forward pass: Input → Encoder → Latent → Decoder → Reconstruction
        2. Compute loss: MSE between input and reconstruction
        3. Backward pass: Compute gradients via backpropagation
        4. Update weights: Adam optimizer adjusts parameters
        5. Repeat for all epochs
        
        Args:
            model: Autoencoder model to train
            data: DataLoader (batch-wise training)
            epochs: Number of training epochs
            lr: Learning rate for Adam optimizer
            model_name: Name for logging and output files
            progress_interval: If set, save reconstruction panels every N epochs
        
        Returns:
            losses: List of loss values per epoch (for plotting training curves)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
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
        
        return losses
    
    # ========================================================================
    # STEP 5: Train model and save weights
    # ========================================================================
    print("\nTraining improved model...")
    improved_losses = quick_train(improved_model, train_loader, epochs=epochs, lr=lr, model_name="improved", progress_interval=progress_interval)
    
    # Save trained model weights to output directory
    model_path = os.path.join(output_dir, 'improved_autoencoder.pth')
    torch.save(improved_model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    
    # ========================================================================
    # STEP 6: Extract latent embeddings from all data
    # ========================================================================
    # Extract latent embeddings from ALL data for t-SNE
    improved_model.eval()  # Set to evaluation mode
    
    print(f"\nExtracting latent embeddings from all {len(dataset)} samples for t-SNE...")
    all_latent = []
    
    with torch.no_grad():
        for i, (sample, label) in enumerate(dataset):
            sample_batch = sample.unsqueeze(0)  # Add batch dimension
            _, latent = improved_model(sample_batch)
            all_latent.append(latent.cpu())
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
    
    # Combine all latent vectors
    improved_latent_full = torch.cat(all_latent, dim=0)
    print(f"Extracted {len(improved_latent_full)} latent embeddings")
    
    # Also compute reconstructions on visualization subset
    with torch.no_grad():
        improved_recon, improved_latent = improved_model(data_tensor)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol))
    
    # ========================================================================
    # STEP 7: Generate visualization plots and export results
    # ========================================================================
    # Convert tensor to numpy for visualization
    data_np = data_tensor.squeeze(1).numpy()  # Remove channel dimension for plotting
    
    # Print value ranges for debugging
    print(f"\nValue ranges:")
    print(f"  Input data: [{data_np.min():.4f}, {data_np.max():.4f}]")
    print(f"  Improved recon: [{improved_recon.min():.4f}, {improved_recon.max():.4f}]")
    
    # ========================================================================
    # PLOT 1: Reconstruction quality comparison (input vs output vs error)
    # ========================================================================
    # Create reconstruction figure for inputs vs improved outputs
    # Use consistent vmin/vmax across all plots for fair comparison
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    
    cols = min(10, data_np.shape[0])
    n_rows = 3 if show_error else 2
    fig_height = 9 if show_error else 6
    fig, axes = plt.subplots(n_rows, cols, figsize=(15, fig_height))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        imp_recon = improved_recon[i, 0].numpy()
        axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Improved Reconstruction')
        axes[1, i].axis('off')
        if show_error:
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
    
    # ========================================================================
    # PLOT 2: Training loss curve
    # ========================================================================
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
    
    # ========================================================================
    # PLOT 3: t-SNE latent space visualization with clustering
    # ========================================================================
    # Latent t-SNE visualization with optional clustering
    try:
        # Use full dataset embeddings
        imp_z = improved_latent_full.detach().cpu().numpy()
        print(f"\nComputing t-SNE on {len(imp_z)} samples from full dataset...")
        
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

            # Simple clustering visualization
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
    
    # ========================================================================
    # EXPORT 1: Save JPEG reconstruction panels
    # ========================================================================
    # Save JPEG panels for randomly selected samples
    try:
        panel_samples, filenames = select_samples_for_outputs(dataset, data_tensor, output_samples, seed)
        panels_written = save_reconstruction_panels(improved_model, panel_samples, output_dir, (nrow, ncol),
                                                     dataset_label=dataset_label, filenames=filenames, 
                                                     show_error=show_error)
        print(f"Saved {panels_written} JPEG panel(s) covering {panel_samples.shape[0]} samples to {output_dir}")
    except Exception as e:
        print(f"Warning: unable to create JPEG panels: {e}")

    # ========================================================================
    # EXPORT 2: Save reconstruction data to .mat file
    # ========================================================================
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

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ========================================================================
    # Parse all command-line arguments for configuration
    parser = argparse.ArgumentParser(description="Compare AEs on real SNR_gram data")
    parser.add_argument("--data-dir", 
                       default="/Volumes/Bowhead2/11132025_Datasets/Unsupervised_database_Balanced.dir", 
                       help="Root folder containing .mat files")
    parser.add_argument("--n-samples", type=int, default=15, help="Number of samples for reconstruction visualization")
    parser.add_argument("--tsne-samples", type=int, default=None, 
                       help="Number of samples for t-SNE plot (default: uses n-samples)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent space size for both models")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base number of channels for conv blocks")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Training epochs for each model (global default EPOCHS_DEFAULT)")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help=f"Learning rate for Adam optimizer (global default LR_DEFAULT={LR_DEFAULT})")
    parser.add_argument("--progress-interval", type=int, default=0, help="If >0, save interim recon panels every N epochs")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Seed for deterministic sample selection")
    parser.add_argument("--show-files", action='store_true', help="Print the list of selected files")
    parser.add_argument("--k-clusters", type=int, default=2, help="Number of clusters for KMeans on latent embeddings (improved-only t-SNE)")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="t-SNE perplexity; leave unset for auto based on sample size")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT, 
                       help=f"Enable extra convolutional layer (4 layers instead of 3) for deeper feature extraction (default: {EXTRA_CONV_DEFAULT})")
    parser.add_argument("--no-extra-conv", dest='extra_conv', action='store_false',
                       help="Disable extra convolutional layer (use 3 layers instead of 4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader training (default: 32)")
    parser.add_argument("--output-samples", type=int, default=NUMBER_OUTPUT_IMAGE_SAMPLES,
                       help="Number of random spectrograms to render into JPEG panels (default: 5000)")
    parser.add_argument("--show-error", action='store_true', default=SHOW_ERROR_PLOTS,
                       help=f"Include error row in reconstruction panels (default: {SHOW_ERROR_PLOTS})")
    parser.add_argument("--no-error", dest='show_error', action='store_false',
                       help="Exclude error row from reconstruction panels")
    parser.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG,
                       help="Version identifier used in the Autoencoder_vXX_Date*.dir output folder name")
    args = parser.parse_args()
    
    # ========================================================================
    # LAUNCH TRAINING WORKFLOW
    # ========================================================================
    # Call main training function with all parsed arguments
    compare_autoencoder_performance(args.data_dir,
                                    n_samples=args.n_samples,
                                    normalize=True,  # Always normalize (auto-skip if already [0,1])
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
                                    batch_size=args.batch_size,
                                    output_samples=args.output_samples,
                                    version_tag=args.version_tag,
                                    show_error=args.show_error)