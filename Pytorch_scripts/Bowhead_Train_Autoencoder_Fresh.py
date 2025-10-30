#!/usr/bin/env python3
"""
Fresh Bowhead Whale Autoencoder Training Script
Improved architecture without sigmoid bottleneck and better training strategies.
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
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time

# Resolve repo root regardless of current working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLI arguments
# Default to the Airguns dataset provided by the user
default_data_dir = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_With_Airguns.dir"
default_logdir = os.path.join(REPO_ROOT, "runs")
parser = argparse.ArgumentParser(description="Train improved bowhead spectrogram autoencoder")
parser.add_argument("--data-dir", default=default_data_dir, help="Path to dataset root (folder containing .mat files)")
parser.add_argument("--logdir", default=default_logdir, help="TensorBoard log directory")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--latent-dim", type=int, default=64, help="Latent space dimension")
parser.add_argument("--normalize", action='store_true', help="Normalize input data to [0,1]")
parser.add_argument("--l1-loss", action='store_true', help="Use L1 loss instead of MSE")
args = parser.parse_args()

# Dataset directories
savedir = [args.data_dir]
batch_size = args.batch_size
learning_rate = args.lr
validation_split = 0.2

class ImprovedAutoencoder(nn.Module):
    """Improved autoencoder architecture without sigmoid bottleneck."""
    
    def __init__(self, nrow, ncol, latent_dim=64):
        super(ImprovedAutoencoder, self).__init__()
        
        self.nrow, self.ncol = nrow, ncol
        nrow_reduced = nrow // 8
        ncol_reduced = ncol // 8
        
        # Encoder with progressive channel increase and batch normalization
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            
            # Block 2: 32 -> 64 channels  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        
        # Calculate flattened size
        self.flat_size = 128 * nrow_reduced * ncol_reduced
        
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
        
        # Decoder with exact dimension reconstruction
        self.decoder = nn.Sequential(
            # Block 1: 128 -> 64 channels
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 32 -> 1 channel (output)
            # CRITICAL: No sigmoid activation - allow unbounded outputs!
            nn.ConvTranspose2d(32, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8)),
        )
        
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        latent = self.to_latent(flattened)
        
        # Decode
        decoded_flat = self.from_latent(latent)
        decoded_reshaped = decoded_flat.view(decoded_flat.size(0), 128, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(decoded_reshaped)
        
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

# Process each dataset folder
for folder in savedir:
    print(f"Using dataset root: {folder}")
    if not os.path.isdir(folder):
        print(f"Dataset folder does not exist: {folder}")
        continue
    
    # Collect all .mat files
    filelist = []
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
    run_name = f"improved_{os.path.basename(folder.rstrip('/'))}_" + time.strftime("%Y%m%d-%H%M%S")
    LOG_ROOT = args.logdir
    os.makedirs(LOG_ROOT, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(LOG_ROOT, run_name))
    print(f"TensorBoard logging -> {os.path.join(LOG_ROOT, run_name)}")
    
    # Log configuration
    config_text = f"""
    Architecture: Improved Autoencoder
    Input size: {nrow} x {ncol}
    Latent dimension: {args.latent_dim}
    Batch size: {batch_size}
    Learning rate: {learning_rate}
    Epochs: {args.epochs}
    Loss function: {'L1' if args.l1_loss else 'MSE'}
    Normalization: {args.normalize}
    """
    writer.add_text('config', config_text, 0)
    
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
    
    autoencoder = ImprovedAutoencoder(nrow, ncol, latent_dim=args.latent_dim).to(device)
    
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
        
        for batch_data in train_dataloader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            outputs, latent = autoencoder(batch_data.float())
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
            torch.save(autoencoder.state_dict(), f'improved_{os.path.basename(folder.rstrip("/"))}_best_model.pth')
        
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
                
                # Create comparison grid
                comparison = torch.cat([sample_batch, sample_recon], dim=0)
                grid = make_grid(comparison.cpu(), nrow=sample_batch.size(0), normalize=True)
                writer.add_image(f'Reconstruction_Epoch_{epoch+1}', grid, epoch + 1)
    
    # Final model save
    final_model_name = f'improved_{os.path.basename(folder.rstrip("/"))}_final_model.pth'
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
    
    plot_name = f'improved_{os.path.basename(folder.rstrip("/"))}_training_plot.png'
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final reconstruction analysis
    autoencoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_dataloader))[:5].to(device)
        sample_recon, sample_latent = autoencoder(sample_batch.float())
        
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
            axes[2, i].set_title(f'|Error Map (for every pixel, the highlighted absolute difference between the original and the reconstructed)|')
            axes[2, i].axis('off')
        
        plt.suptitle('Final Reconstruction Quality - Improved Autoencoder', fontsize=14)
        plt.tight_layout()
        
        comparison_name = f'improved_{os.path.basename(folder.rstrip("/"))}_final_comparison.png'
        plt.savefig(comparison_name, dpi=200, bbox_inches='tight')
        plt.show()
    
    # ------------------------------------------------------------------
    # Random 15-sample panel: originals (top) and reconstructions (bottom)
    # with filename captions under each image (small font)
    # ------------------------------------------------------------------
    try:
        autoencoder.eval()
        with torch.no_grad():
            # Sample up to 15 random files from the dataset
            k = min(15, len(valid_files))
            sample_files = random.sample(valid_files, k)

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
                panel_name = f'improved_{os.path.basename(folder.rstrip("/"))}_random15_panel.png'
                plt.savefig(panel_name, dpi=200, bbox_inches='tight')
                plt.show()
                print(f"Saved random sample panel: {panel_name}")
            else:
                print("No images could be loaded for the random panel.")
    except Exception as e:
        print(f"Random 15-sample panel generation failed: {e}")

    writer.close()
    print(f"Training complete! Models saved:")
    print(f"  Best: improved_{os.path.basename(folder.rstrip('/'))}_best_model.pth")
    print(f"  Final: {final_model_name}")
    print(f"  Plots: {plot_name}, {comparison_name}{', ' + panel_name if 'panel_name' in locals() else ''}")

print("All datasets processed!")