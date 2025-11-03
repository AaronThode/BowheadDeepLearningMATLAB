#!/usr/bin/env python3
"""
Fresh Bowhead Whale Autoencoder Training Script
Improved architecture without sigmoid bottleneck and better training strategies.

Global experiment knobs (edit-and-run):
- MODEL_CHANNELS: encoder/decoder channel widths
- MODEL_LATENT_DIM: latent space dimension
- PANEL_SEED: controls the fixed 15-image selection for the final panel
- USE_FILE_PICKER: if True, lets you pick specific .mat files to train/validate on
- USE_REFINEMENT_HEAD: add a small residual refinement block to sharpen reconstructions
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
import time

# ---------------------------
# Global experiment settings
# ---------------------------
MODEL_CHANNELS = [32, 64, 128]   # Edit to try wider/narrower models, e.g., [16,32,64] or [64,128,256]
MODEL_LATENT_DIM = 64            # Edit to change latent bottleneck size
EPOCHS = 5                      # Edit to set default training epochs
PANEL_SEED = 1337                # Controls the fixed 15-sample selection for the comparison panel
USE_FILE_PICKER = False          # If True, select .mat files via a dialog instead of scanning a folder
USE_REFINEMENT_HEAD = True       # If True, add a residual refinement head after decoder
USE_UNET_SKIPS = True           # If True, add U-Net style skip connections (encoder -> decoder)
USE_UPSAMPLE_CONV = True        # If True, use Upsample+Conv instead of ConvTranspose2d
USE_SE_BLOCKS = False            # If True, add Squeeze-and-Excitation blocks in encoder

# Resolve repo root regardless of current working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLI arguments
# Default to the Airguns dataset provided by the user
default_data_dir = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_With_Airguns.dir"
default_logdir = os.path.join(REPO_ROOT, "runs")
parser = argparse.ArgumentParser(description="Train improved bowhead spectrogram autoencoder")
parser.add_argument("--data-dir", default=default_data_dir, help="Path to dataset root (folder containing .mat files)")
parser.add_argument("--logdir", default=default_logdir, help="TensorBoard log directory")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (override default EPOCHS)")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--latent-dim", type=int, default=MODEL_LATENT_DIM, help="Latent space dimension")
parser.add_argument("--normalize", action='store_true', help="Normalize input data to [0,1]")
parser.add_argument("--l1-loss", action='store_true', help="Use L1 loss instead of MSE")
args = parser.parse_args()

# Dataset directories
savedir = [args.data_dir]
batch_size = args.batch_size
learning_rate = args.lr
validation_split = 0.2

# Define global hyperparameters

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for lightweight channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ImprovedAutoencoder(nn.Module):
    """Improved autoencoder with optional skips, SE blocks, and refine head.

    channels: list/tuple of three ints [c1, c2, c3] for encoder widths.
    use_refine: add residual refinement head on decoder output.
    use_unet_skips: enable U-Net style skip connections.
    use_upsample_conv: use Upsample+Conv instead of ConvTranspose2d in decoder.
    use_se_blocks: add SE blocks to encoder conv stages.
    """
    
    def __init__(self, nrow, ncol, latent_dim=64, channels=(32, 64, 128), use_refine=True,
                 use_unet_skips=False, use_upsample_conv=False, use_se_blocks=False):
        super(ImprovedAutoencoder, self).__init__()
        
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = nrow // 8
        ncol_reduced = ncol // 8
        assert len(channels) == 3, "channels must be a sequence of three ints, e.g., [32,64,128]"
        c1, c2, c3 = channels
        
        self.use_refine = use_refine
        self.use_skips = bool(use_unet_skips)
        self.use_upsample_conv = bool(use_upsample_conv)
        self.use_se = bool(use_se_blocks)

        # Encoder blocks (pre-pool), with optional SE
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            SEBlock(c1) if self.use_se else nn.Identity(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            SEBlock(c2) if self.use_se else nn.Identity(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            SEBlock(c3) if self.use_se else nn.Identity(),
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Calculate flattened size
        self.flat_size = c3 * nrow_reduced * ncol_reduced
        
        # Latent space with regularization
        self.to_latent = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, self.flat_size),
            nn.ReLU(inplace=True)
        )

        # Decoder layers (transpose convs or upsample+conv)
        if not self.use_upsample_conv:
            self.up1 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.up1_bn = nn.BatchNorm2d(c2)
            self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.up2_bn = nn.BatchNorm2d(c1)
            self.up3 = nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8))
        else:
            self.up1 = nn.Conv2d(c3, c2, 3, padding=1)
            self.up1_bn = nn.BatchNorm2d(c2)
            self.up2 = nn.Conv2d(c2, c1, 3, padding=1)
            self.up2_bn = nn.BatchNorm2d(c1)
            self.up3 = nn.Conv2d(c1, 1, 3, padding=1)

        # Skip fusion layers
        if self.use_skips:
            self.fuse1 = nn.Sequential(
                nn.Conv2d(c2 + c2, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )
            self.fuse2 = nn.Sequential(
                nn.Conv2d(c1 + c1, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse1 = None
            self.fuse2 = None

        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.latent_dim = latent_dim
        self.channels = (c1, c2, c3)

        # Optional residual refinement head to sharpen reconstructions
        if self.use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(1, max(8, c1 // 2), kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, c1 // 2), 1, kernel_size=3, padding=1),
            )
        else:
            self.refine = None
    
    def forward(self, x):
        # Encode (capture pre-pool activations for optional skips)
        e1 = self.enc1(x)
        x1 = self.pool1(e1)

        e2 = self.enc2(x1)
        x2 = self.pool2(e2)

        e3 = self.enc3(x2)
        x3 = self.pool3(e3)

        flattened = x3.view(x3.size(0), -1)
        latent = self.to_latent(flattened)
        
        # Decode
        decoded_flat = self.from_latent(latent)
        y = decoded_flat.view(decoded_flat.size(0), self.channels[2], self.nrow_reduced, self.ncol_reduced)

        # Step 1: up to c2 and optional skip with e2
        if not self.use_upsample_conv:
            y = F.relu(self.up1_bn(self.up1(y)), inplace=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
            y = F.relu(self.up1_bn(self.up1(y)), inplace=True)
        if self.use_skips:
            if e2.shape[-2:] != y.shape[-2:]:
                y = F.interpolate(y, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, e2], dim=1)
            y = self.fuse1(y)

        # Step 2: up to c1 and optional skip with e1
        if not self.use_upsample_conv:
            y = F.relu(self.up2_bn(self.up2(y)), inplace=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
            y = F.relu(self.up2_bn(self.up2(y)), inplace=True)
        if self.use_skips:
            if e1.shape[-2:] != y.shape[-2:]:
                y = F.interpolate(y, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, e1], dim=1)
            y = self.fuse2(y)

        # Step 3: to 1 channel at exact target size
        if not self.use_upsample_conv:
            output = self.up3(y)
        else:
            y = F.interpolate(y, size=(self.nrow, self.ncol), mode='bilinear', align_corners=False)
            output = self.up3(y)

        if self.refine is not None:
            output = output + self.refine(output)
        
        return output, latent

class SpectrogramDataset(Dataset):
    """Dataset class for loading .mat spectrogram files."""
    
    def __init__(self, file_list, transform=None, normalize=False):
        self.file_list = list(file_list)
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            mat = loadmat(file_path)
            image = mat.get('SNR_gram', None)
            if image is None or not isinstance(image, np.ndarray) or image.ndim != 2:
                raise ValueError("SNR_gram missing or invalid")
            
            # Optional normalization
            if self.normalize:
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image).float()
                if image.ndim == 2:
                    image = image.unsqueeze(0)
            return image
        except Exception as e:
            print(f"Warning: failed to load {file_path}: {e}")
            # Return zeros with expected shape (will be updated by first valid sample)
            return torch.zeros(1, 121, 104)

def _pick_mat_files(initial_dir):
    """Optional: open a file dialog to pick .mat files. Returns a tuple of paths or empty tuple."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select .mat files",
            filetypes=[("MAT files", "*.mat")],
            initialdir=initial_dir
        )
        root.destroy()
        return paths
    except Exception as e:
        print(f"File picker unavailable or failed: {e}")
        return tuple()

# Process each dataset folder
for folder in savedir:
    print(f"Using dataset root: {folder}")
    dataset_slug = os.path.basename(folder.rstrip('/')) if os.path.isdir(folder) else os.path.basename(folder.rstrip('/'))
    if not os.path.isdir(folder) and not USE_FILE_PICKER:
        print(f"Dataset folder does not exist: {folder}")
        continue
    
    # Collect all .mat files
    filelist = []
    if USE_FILE_PICKER:
        picked = _pick_mat_files(folder if os.path.isdir(folder) else REPO_ROOT)
        if picked:
            filelist = list(picked)
            # Name outputs after the parent directory of the first picked file
            dataset_slug = os.path.basename(os.path.dirname(filelist[0]))
            print(f"Using {len(filelist)} picked files from file dialog. Dataset tag: {dataset_slug}")
        else:
            print("No files picked; falling back to folder scan.")
    if not filelist:
        for root, dirs, files in os.walk(folder):
            for f in sorted(files):
                if f.endswith('.mat'):
                    filelist.append(os.path.join(root, f))
    
    if not filelist:
        print(f"No .mat files found in {folder}")
        continue
    
    print(f"Found {len(filelist)} .mat files in {folder}")
    
    # Get image dimensions from first valid file
    nrow, ncol = None, None
    for file_path in filelist[:10]:  # Check first 10 files
        try:
            image = loadmat(file_path)['SNR_gram']
            if isinstance(image, np.ndarray) and image.ndim == 2:
                nrow, ncol = image.shape
                print(f"Image dimensions: {nrow} x {ncol}")
                break
        except:
            continue
    
    if nrow is None:
        print("Could not determine image dimensions from valid files")
        continue
    
    # Filter valid files with consistent dimensions
    valid_files = []
    for fp in filelist:
        try:
            im = loadmat(fp).get('SNR_gram', None)
            if (isinstance(im, np.ndarray) and im.ndim == 2 and 
                im.shape == (nrow, ncol) and im.size > 0):
                valid_files.append(fp)
        except:
            continue
    
    print(f"Valid files with shape {nrow}x{ncol}: {len(valid_files)}")
    if len(valid_files) < 10:
        print("Too few valid files for training")
        continue
    
    # Setup TensorBoard logging
    run_name = f"improved_{dataset_slug}_" + time.strftime("%Y%m%d-%H%M%S")
    LOG_ROOT = args.logdir
    os.makedirs(LOG_ROOT, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(LOG_ROOT, run_name))
    print(f"TensorBoard logging -> {os.path.join(LOG_ROOT, run_name)}")
    
    # Log configuration (plain text)
    config_text = f"""
    Architecture: Improved Autoencoder
    Input size: {nrow} x {ncol}
    Latent dimension: {args.latent_dim}
    Channels: {MODEL_CHANNELS}
    Batch size: {batch_size}
    Learning rate: {learning_rate}
    Epochs: {args.epochs}
    Loss function: {'L1' if args.l1_loss else 'MSE'}
    Normalization: {args.normalize}
    Panel seed: {PANEL_SEED}
    Toggles: refine={USE_REFINEMENT_HEAD}, skips={USE_UNET_SKIPS}, upsample_conv={USE_UPSAMPLE_CONV}, se_blocks={USE_SE_BLOCKS}, file_picker={USE_FILE_PICKER}
    Run name: {run_name}
    Dataset tag: {dataset_slug}
    """
    writer.add_text('config', config_text, 0)

    # Also log a Markdown table for easy viewing in TensorBoard
    config_table = """
    | Key | Value |
    | --- | ----- |
    | Input size | {nrow} x {ncol} |
    | Latent dim | {args.latent_dim} |
    | Channels | {MODEL_CHANNELS} |
    | Batch size | {batch_size} |
    | Learning rate | {learning_rate} |
    | Epochs | {args.epochs} |
    | Loss | {'L1' if args.l1_loss else 'MSE'} |
    | Normalize | {args.normalize} |
    | Panel seed | {PANEL_SEED} |
    | Refine head | {USE_REFINEMENT_HEAD} |
    | U-Net skips | {USE_UNET_SKIPS} |
    | Upsample+Conv | {USE_UPSAMPLE_CONV} |
    | SE blocks | {USE_SE_BLOCKS} |
    | File picker | {USE_FILE_PICKER} |
    | Run name | {run_name} |
    | Dataset | {dataset_slug} |
    """
    writer.add_text('config/table', config_table, 0)

    # Persist config as JSON alongside the run for reproducibility
    run_dir = os.path.join(LOG_ROOT, run_name)
    config_json = {
        'input_size': [int(nrow), int(ncol)],
        'latent_dim': int(args.latent_dim),
        'channels': list(map(int, MODEL_CHANNELS)),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'epochs': int(args.epochs),
        'loss': 'L1' if args.l1_loss else 'MSE',
        'normalize': bool(args.normalize),
        'panel_seed': int(PANEL_SEED),
        'toggles': {
            'refine': bool(USE_REFINEMENT_HEAD),
            'unet_skips': bool(USE_UNET_SKIPS),
            'upsample_conv': bool(USE_UPSAMPLE_CONV),
            'se_blocks': bool(USE_SE_BLOCKS),
            'file_picker': bool(USE_FILE_PICKER),
        },
        'run_name': run_name,
        'dataset': dataset_slug,
        'data_dir': folder,
    }
    try:
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(config_json, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write run_config.json: {e}")
    
    # Create dataset and dataloaders
    transform = transforms.ToTensor() if not args.normalize else None
    dataset = SpectrogramDataset(valid_files, transform=transform, normalize=args.normalize)
    
    num_samples = len(dataset)
    num_train_samples = int((1 - validation_split) * num_samples)
    num_val_samples = num_samples - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])
    
    _workers = 0  # Avoid multiprocessing issues on macOS
    _pin_mem = torch.cuda.is_available()
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=_workers, pin_memory=_pin_mem)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=_workers, pin_memory=_pin_mem)
    
    print(f"Training samples: {num_train_samples}, Validation samples: {num_val_samples}")
    
    # Create improved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    autoencoder = ImprovedAutoencoder(
        nrow,
        ncol,
        latent_dim=args.latent_dim,
        channels=tuple(MODEL_CHANNELS),
        use_refine=USE_REFINEMENT_HEAD,
        use_unet_skips=USE_UNET_SKIPS,
        use_upsample_conv=USE_UPSAMPLE_CONV,
        use_se_blocks=USE_SE_BLOCKS,
    ).to(device)
    
    # Loss function
    criterion = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        autoencoder.train()
        train_loss_total = 0.0
        num_train_batches = 0
        
        warned_mismatch = False
        for batch_data in train_dataloader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            outputs, latent = autoencoder(batch_data.float())
            # Align output spatial dims to target if mismatched
            if outputs.shape[-2:] != batch_data.shape[-2:]:
                if not warned_mismatch:
                    print(f"[warn] train: resizing outputs from {tuple(outputs.shape[-2:])} to {tuple(batch_data.shape[-2:])}")
                    warned_mismatch = True
                outputs = torch.nn.functional.interpolate(outputs, size=batch_data.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, batch_data.float())
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            num_train_batches += 1
        
        # Validation phase
        autoencoder.eval()
        val_loss_total = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_data = batch_data.to(device)
                outputs, latent = autoencoder(batch_data.float())
                if outputs.shape[-2:] != batch_data.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(outputs, size=batch_data.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, batch_data.float())
                val_loss_total += loss.item()
                num_val_batches += 1
        
        # Calculate average losses
        train_loss_avg = train_loss_total / max(1, num_train_batches)
        val_loss_avg = val_loss_total / max(1, num_val_batches)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(autoencoder.state_dict(), f'improved_{dataset_slug}_best_model.pth')
        
        # Logging
        writer.add_scalar('Loss/train', train_loss_avg, epoch + 1)
        writer.add_scalar('Loss/val', val_loss_avg, epoch + 1)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        print(f'Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}')
        
        # Periodic reconstruction visualization
        if (epoch + 1) % 10 == 0:
            autoencoder.eval()
            with torch.no_grad():
                # Get a batch for visualization
                sample_batch = next(iter(val_dataloader))[:min(8, batch_size)].to(device)
                sample_recon, _ = autoencoder(sample_batch.float())
                if sample_recon.shape[-2:] != sample_batch.shape[-2:]:
                    sample_recon = torch.nn.functional.interpolate(sample_recon, size=sample_batch.shape[-2:], mode='bilinear', align_corners=False)
                
                # Create comparison grid
                comparison = torch.cat([sample_batch, sample_recon], dim=0)
                grid = make_grid(comparison.cpu(), nrow=sample_batch.size(0), normalize=True)
                writer.add_image(f'Reconstruction_Epoch_{epoch+1}', grid, epoch + 1)
    
    # Final model save
    final_model_name = f'improved_{dataset_slug}_final_model.pth'
    torch.save(autoencoder.state_dict(), final_model_name)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - Improved Autoencoder\\nFinal Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_name = f'improved_{dataset_slug}_training_plot.png'
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final reconstruction analysis
    autoencoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_dataloader))[:5].to(device)
        sample_recon, sample_latent = autoencoder(sample_batch.float())
        if sample_recon.shape[-2:] != sample_batch.shape[-2:]:
            sample_recon = torch.nn.functional.interpolate(sample_recon, size=sample_batch.shape[-2:], mode='bilinear', align_corners=False)
        
        # Reconstruction quality metrics
        mse_loss = nn.MSELoss()(sample_recon, sample_batch).item()
        mae_loss = nn.L1Loss()(sample_recon, sample_batch).item()
        
        print(f"Final reconstruction metrics:")
        print(f"  MSE: {mse_loss:.4f}")
        print(f"  MAE: {mae_loss:.4f}")
        
        # Visual comparison
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        
        for i in range(5):
            orig = sample_batch[i, 0].cpu().numpy()
            recon = sample_recon[i, 0].cpu().numpy()
            diff = np.abs(orig - recon)
            
            # Original
            im1 = axes[0, i].imshow(orig, cmap='viridis', origin='lower', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstruction
            im2 = axes[1, i].imshow(recon, cmap='viridis', origin='lower', aspect='auto')
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')
            
            # Difference
            im3 = axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title(f'Error Map')
            axes[2, i].axis('off')
        
        plt.suptitle('Final Reconstruction Quality - Improved Autoencoder', fontsize=14)
        plt.tight_layout()
        
        comparison_name = f'improved_{dataset_slug}_final_comparison.png'
        plt.savefig(comparison_name, dpi=200, bbox_inches='tight')
        plt.show()
    
    # ------------------------------------------------------------------
    # Random 15-sample panel: originals (top) and reconstructions (bottom)
    # with filename captions under each image (small font)
    # ------------------------------------------------------------------
    try:
        autoencoder.eval()
        with torch.no_grad():
            # Deterministically pick up to 15 files from the dataset using PANEL_SEED
            k = min(15, len(valid_files))
            vf_sorted = sorted(valid_files)
            rng = np.random.RandomState(PANEL_SEED)
            idxs = rng.choice(len(vf_sorted), size=k, replace=False)
            sample_files = [vf_sorted[i] for i in sorted(idxs)]

            # Load images and build batch
            imgs = []
            names = []
            for fp in sample_files:
                mat = loadmat(fp)
                im = mat.get('SNR_gram', None)
                if im is None:
                    continue
                if args.normalize:
                    im = (im - im.min()) / (im.max() - im.min() + 1e-8)
                t = torch.from_numpy(im).float().unsqueeze(0)  # [1,H,W]
                imgs.append(t)
                names.append(os.path.basename(fp))

            if imgs:
                batch = torch.stack(imgs, dim=0).to(device)  # [N,1,H,W]
                recon, _ = autoencoder(batch.float())        # [N,1,H,W]
                if recon.shape[-2:] != batch.shape[-2:]:
                    recon = torch.nn.functional.interpolate(recon, size=batch.shape[-2:], mode='bilinear', align_corners=False)

                cols = batch.size(0)
                fig, ax = plt.subplots(2, cols, figsize=(cols * 1.8, 4.0), dpi=200)
                if cols == 1:
                    # Ensure ax is 2D indexable when N=1
                    ax = np.array([[ax[0]], [ax[1]]])
                for i in range(cols):
                    orig = batch[i, 0].detach().cpu().numpy()
                    rec  = recon[i, 0].detach().cpu().numpy()
                    # Top: original
                    ax[0, i].imshow(orig, cmap='viridis', origin='lower', aspect='auto')
                    ax[0, i].axis('off')
                    # Place filename vertically along the right side of the image
                    ax[0, i].text(
                        1.01,
                        0.5,
                        names[i],
                        rotation=90,
                        va='center',
                        ha='left',
                        fontsize=6,
                        color='white',
                        transform=ax[0, i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1.0),
                        clip_on=False,
                    )
                    # Bottom: reconstructed
                    ax[1, i].imshow(rec, cmap='viridis', origin='lower', aspect='auto')
                    ax[1, i].axis('off')
                    # Place filename vertically along the right side of the image
                    ax[1, i].text(
                        1.01,
                        0.5,
                        names[i],
                        rotation=90,
                        va='center',
                        ha='left',
                        fontsize=6,
                        color='white',
                        transform=ax[1, i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1.0),
                        clip_on=False,
                    )
                plt.suptitle('Random 15 Samples: Originals (top) vs Reconstructions (bottom)')
                plt.tight_layout()
                panel_name = f'improved_{dataset_slug}_random15_panel.png'
                plt.savefig(panel_name, dpi=200, bbox_inches='tight')
                plt.show()
                print(f"Saved random sample panel: {panel_name}")
                # Persist the selected file list for traceability
                panel_list_name = f'improved_{dataset_slug}_random15_files.txt'
                try:
                    with open(panel_list_name, 'w') as fh:
                        for p in sample_files:
                            fh.write(p + "\n")
                    print(f"Saved list of panel files: {panel_list_name}")
                except Exception as e:
                    print(f"Failed to save panel file list: {e}")
            else:
                print("No images could be loaded for the random panel.")
    except Exception as e:
        print(f"Random 15-sample panel generation failed: {e}")

    writer.close()
    print(f"Training complete! Models saved:")
    print(f"  Best: improved_{dataset_slug}_best_model.pth")
    print(f"  Final: {final_model_name}")
    print(f"  Plots: {plot_name}, {comparison_name}{', ' + panel_name if 'panel_name' in locals() else ''}")

print("All datasets processed!")