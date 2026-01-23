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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time

# NOTE: Minimal edits policy
# CHANGE tags mark small, targeted modifications added for stability or configurability.
# The overall structure of the original script is preserved.

# Resolve repo root regardless of current working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# Global model toggles
# ---------------------------
# You can edit these defaults and/or override via CLI flags
MODEL_LATENT_DIM = 96          # Latent space size (was 64)
MODEL_CHANNELS = 128           # Deepest encoder/decoder channels (was 128)
ADD_EXTRA_CONV = False         # If True, add one more conv before last pooling

# ---------------------------
# Global training hyperparameters
# ---------------------------
# These control dataset batching, optimization, and the validation split ratio.
#
# BATCH_SIZE:
# - Number of samples processed together before the model updates its weights.
# - Larger batches improve throughput and provide smoother gradient estimates, but require more memory.
# - If you run out of GPU/CPU RAM or see slowdowns, reduce this value; if utilization is low, try increasing it.
# - Typical ranges: 16–256 depending on input size and hardware; 64 is a safe default for 2D spectrograms.
BATCH_SIZE = 64
#
# LEARNING_RATE:
# - Step size for the optimizer (Adam here). Higher values train faster but may become unstable; lower values are safer.
# - If loss is not decreasing or oscillates wildly, lower this; if training is very slow and stable, try increasing it slightly.
# - Typical ranges: 1e-3 to 1e-5; 1e-4 is a conservative starting point for autoencoders.
LEARNING_RATE = 1e-5
#
# VALIDATION_SPLIT:
# - Fraction of the dataset held out from training and used to report validation metrics each epoch.
# - Ensures we detect overfitting and track generalization; raising it gives more accurate validation at the cost of fewer training samples.
# - Typical ranges: 0.1–0.3; 0.2 (20%) is a common default.
VALIDATION_SPLIT = 0.2

# CLI arguments to make data/log locations and epochs configurable
# Defaults point to the BCB_Whale_Datasets root and the Airguns dataset
DEFAULT_ROOT_DIR = "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets"
DEFAULT_DATASET_NAME = "Unsupervised_database_With_Airguns.dir"  # CHANGE: default dataset updated per request
default_data_dir = os.path.join(DEFAULT_ROOT_DIR, DEFAULT_DATASET_NAME)
default_logdir = os.path.join(REPO_ROOT, "runs")
parser = argparse.ArgumentParser(description="Train bowhead spectrogram autoencoder with TensorBoard logging")
parser.add_argument("--data-dir", default=default_data_dir, help="Full path to dataset root (folder containing .mat files or subfolders)")
parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Root directory that contains multiple dataset folders")
parser.add_argument("--dataset", default=None, help="Dataset folder name under --root-dir to use instead of --data-dir (e.g., Unsupervised_database_No_Airguns.dir)")
parser.add_argument("--logdir", default=default_logdir, help="TensorBoard log directory (will be created if missing)")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--latent-dim", type=int, default=MODEL_LATENT_DIM, help="Latent bottleneck size")
parser.add_argument("--channels", type=int, default=MODEL_CHANNELS, help="Deepest encoder/decoder channel width")
parser.add_argument("--extra-conv", action='store_true', default=ADD_EXTRA_CONV, help="Add an extra Conv2d layer before the last pooling")
parser.add_argument("--normalize", action='store_true', help="Min-max normalize each spectrogram to [0,1] before feeding model (recommended if dynamic range varies)")  # CHANGE: enable per-image normalization
parser.add_argument("--debug-overfit", type=int, default=0, help="If >0, limit dataset to N samples and try to overfit (sanity check)")  # CHANGE: add overfit sanity mode
parser.add_argument("--l1-loss", action='store_true', help="CHANGE: use L1 loss instead of MSE (often sharper reconstructions)")
args = parser.parse_args()

# Resolve the effective dataset directory
data_dir = os.path.join(args.root_dir, args.dataset) if args.dataset else args.data_dir

# Dataset directories list (keep structure compatible with existing loop)
savedir = [data_dir]

# Use global training hyperparameters defined above
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
validation_split = VALIDATION_SPLIT

# Process each dataset folder separately
for folder in savedir:
    print(f"Using dataset root: {folder}")
    # Ensure folder exists
    if not os.path.isdir(folder):
        print(f"Dataset folder does not exist: {folder}")
        continue
    # Recursively collect all .mat files under the specified folder
    filelist = []
    for root, dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith('.mat'):
                filelist.append(os.path.join(root, f))
    if not filelist:
        print(f"No .mat files found in {folder}")
        continue

    file_path = filelist[0]
    image = loadmat(file_path)['SNR_gram']
    nrow, ncol = image.shape

    # Define channel and latent dimensions and derived shapes for the autoencoder (from globals/CLI)
    n_channels = int(args.channels)
    latent_dim = int(args.latent_dim)
    use_extra_conv = bool(args.extra_conv)
    nrow_reduced = int(nrow / 8)
    ncol_reduced = int(ncol / 8)
    nel_reduced = nrow_reduced * ncol_reduced * n_channels
    print("Images have dimensions of nrow,ncol =", nrow, ncol)
    print(f"Model config -> channels: {n_channels}, latent_dim: {latent_dim}, extra_conv: {use_extra_conv}")
    print("Reduced dims before latent:", nrow_reduced, ncol_reduced, nel_reduced)

    # Setup TensorBoard writer for this dataset
    run_name = f"{os.path.basename(folder.rstrip('/'))}_" + time.strftime("%Y%m%d-%H%M%S")
    LOG_ROOT = args.logdir
    os.makedirs(LOG_ROOT, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(LOG_ROOT, run_name))
    print(f"TensorBoard logging -> {os.path.join(LOG_ROOT, run_name)}")
    # Write small initial events so TB shows a run immediately
    try:
        writer.add_text('run/info', f'dataset_folder: {folder}', 0)
        writer.add_scalar('Run/started', 1, 0)
        writer.flush()
    except Exception:
        pass

    # Filter out invalid files: ensure SNR_gram exists, is 2D, and matches (nrow, ncol)
    valid_files = []
    dropped = 0
    for fp in filelist:
        try:
            im = loadmat(fp).get('SNR_gram', None)
            if im is None:
                dropped += 1
                continue
            if not isinstance(im, np.ndarray) or im.ndim != 2:
                dropped += 1
                continue
            if im.shape[0] == 0 or im.shape[1] == 0:
                dropped += 1
                continue
            if im.shape != (nrow, ncol):
                # Skip mismatched sizes to keep batches stackable
                dropped += 1
                continue
            valid_files.append(fp)
        except Exception:
            dropped += 1
            continue
    if dropped:
        print(f"Filtered out {dropped} invalid/mismatched files; {len(valid_files)} remain")
    else:
        print(f"All files valid: {len(valid_files)}")
    if not valid_files:
        print("No valid files remain after filtering; skipping this dataset")
        continue

    # CHANGE: small helper to optionally normalize each image
    def _normalize_image(im: np.ndarray) -> np.ndarray:
        if not args.normalize:
            return im
        im_min = im.min()
        im_max = im.max()
        if im_max - im_min < 1e-8:
            return np.zeros_like(im)
        return (im - im_min) / (im_max - im_min)

    class CustomDatasetFull(Dataset):
        def __init__(self, folder_paths, transform=None, shuffle=False):
            self.file_list = []
            for folder in folder_paths:
                files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.mat')]
                self.file_list.extend(files)
            if shuffle:
                random.shuffle(self.file_list)
            self.transform = transform

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            file_path = self.file_list[idx]
            image = loadmat(file_path)['SNR_gram']
            image = _normalize_image(image)  # CHANGE: apply optional per-image normalization
            my_debug = False
            if my_debug:
                print(file_path)
                fig = plt.figure(figsize=(15, 9))
                ax0 = fig.add_subplot(1, 2, 1)
                im0 = plt.imshow(image, cmap='gray', origin='lower')
                ax0.set_title('Input image')
                fig.colorbar(im0, ax=ax0)
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image).float()
                if image.ndim == 2:
                    image = image.unsqueeze(0)
            if my_debug:
                ax1 = fig.add_subplot(1, 2, 2)
                im1 = plt.imshow(image[0, :, :], cmap='gray', origin='lower')
                ax1.set_title('Converted image')
                fig.colorbar(im1, ax=ax1)
                plt.draw()
                plt.pause(5)
                plt.close('all')
            return image

    custom_transform = transforms.ToTensor()
    # Pass filtered valid_files directly to dataset to ensure consistent shapes
    class CustomDatasetFull(Dataset):
        def __init__(self, file_list, transform=None, shuffle=False):
            self.file_list = list(file_list)
            if shuffle:
                random.shuffle(self.file_list)
            self.transform = transform
        def __len__(self):
            return len(self.file_list)
        def __getitem__(self, idx):
            file_path = self.file_list[idx]
            # Robust load: handle missing files or bad contents by substituting zeros
            try:
                mat = loadmat(file_path)
                image = mat.get('SNR_gram', None)
                if image is None or not isinstance(image, np.ndarray) or image.ndim != 2:
                    raise ValueError("SNR_gram missing or invalid")
                image = _normalize_image(image)  # CHANGE: apply optional per-image normalization
            except Exception as e:
                print(f"Warning: failed to load {file_path}: {e}; substituting zeros of shape ({nrow},{ncol})")
                image = np.zeros((nrow, ncol), dtype=np.float32)
            my_debug = False
            if my_debug:
                print(file_path)
                fig = plt.figure(figsize=(15, 9))
                ax0 = fig.add_subplot(1, 2, 1)
                im0 = plt.imshow(image, cmap='gray', origin='lower')
                ax0.set_title('Input image')
                fig.colorbar(im0, ax=ax0)
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image).float()
                if image.ndim == 2:
                    image = image.unsqueeze(0)
            if my_debug:
                ax1 = fig.add_subplot(1, 2, 2)
                im1 = plt.imshow(image[0, :, :], cmap='gray', origin='lower')
                ax1.set_title('Converted image')
                fig.colorbar(im1, ax=ax1)
                plt.draw()
                plt.pause(5)
                plt.close('all')
            return image
    # CHANGE: Optional dataset limit for overfit-debug sanity check
    if args.debug_overfit and len(valid_files) > args.debug_overfit:
        valid_files = valid_files[:args.debug_overfit]
        print(f"Debug-overfit mode: limiting to {len(valid_files)} files")
    dataset = CustomDatasetFull(valid_files, transform=custom_transform, shuffle=False)
    print(dataset[0].size())
    image_dims = dataset[0].size()[1:]
    num_samples = len(dataset)
    num_train_samples = int((1 - validation_split) * num_samples)
    num_val_samples = num_samples - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])
    # Disable multiprocessing workers to avoid spawn errors on macOS; set workers=0
    _workers = 0
    # Only enable pinned memory if using CUDA to avoid warnings on CPU/MPS
    _pin_mem = torch.cuda.is_available()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=_workers, pin_memory=_pin_mem)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=_workers, pin_memory=_pin_mem)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=_workers, pin_memory=_pin_mem)

    # CHANGE: ensure decoder output matches input spatial size (crop/pad)
    def match_shape(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure recon spatial dims match target by center-cropping or padding as needed."""
        rB, rC, rH, rW = recon.shape
        tB, tC, tH, tW = target.shape
        # Only adjust H/W; assume batch/channel already aligned
        # Crop if too large
        if rH > tH:
            dh = (rH - tH) // 2
            recon = recon[:, :, dh:dh + tH, :]
        if rW > tW:
            dw = (rW - tW) // 2
            recon = recon[:, :, :, dw:dw + tW]
        # Pad if too small
        rH2, rW2 = recon.shape[2], recon.shape[3]
        padH = tH - rH2
        padW = tW - rW2
        if padH > 0 or padW > 0:
            pad_left = padW // 2 if padW > 0 else 0
            pad_right = padW - pad_left if padW > 0 else 0
            pad_top = padH // 2 if padH > 0 else 0
            pad_bottom = padH - pad_top if padH > 0 else 0
            recon = F.pad(recon, (pad_left, pad_right, pad_top, pad_bottom))
        return recon

    class Autoencoder(nn.Module):
        def __init__(self, latent_dim):
            super(Autoencoder, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.conv3 = nn.Conv2d(8, n_channels, 3, padding=1)
            # Optional extra conv at deepest stage (keeps H,W unchanged)
            if use_extra_conv:
                self.conv4 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
            self.t_conv1 = nn.ConvTranspose2d(n_channels, 8, 2, stride=2)
            self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
            self.t_conv3 = nn.ConvTranspose2d(4, 1, [2, 2], stride=[2, 2], output_padding=[1, 0])
            self.fc1 = nn.Linear(nel_reduced, latent_dim)
            self.fc2 = nn.Linear(latent_dim, nel_reduced)
            self.pool = nn.MaxPool2d(2, 2)
        def forward(self, x):
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            if use_extra_conv:
                x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)
            x = x.view(-1, nel_reduced)
            latent = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(latent))
            x = x.view(-1, n_channels, nrow_reduced, ncol_reduced)
            x = torch.nn.functional.relu(self.t_conv1(x))
            x = torch.nn.functional.relu(self.t_conv2(x))
            output = self.t_conv3(x)  # REMOVED SIGMOID - raw outputs allow broader reconstruction range
            return output, latent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss(reduction='mean')
    # CHANGE: allow switching between MSE (default) and L1
    criterion = nn.L1Loss(reduction='mean') if args.l1_loss else nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    autoencoder.to(device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    Losses = []
    ValLosses = []
    PSNRs = []  # CHANGE: track PSNR

    # CHANGE: Initial sanity logging before training loop
    try:
        autoencoder.eval()
        with torch.no_grad():
            sample_loader = val_dataloader if len(val_dataloader) > 0 else train_dataloader
            for batch in sample_loader:
                sample0 = batch.to(device)
                recon0, _ = autoencoder(sample0.float())
                recon0 = match_shape(recon0, sample0)
                break
        cols0 = min(6, sample0.size(0))
        grid_orig0 = make_grid(sample0[:cols0].detach().cpu(), nrow=cols0, normalize=True, scale_each=True)
        grid_recon0 = make_grid(recon0[:cols0].detach().cpu(), nrow=cols0, normalize=True, scale_each=True)
        writer.add_image('Sanity/Originals_BeforeTrain', grid_orig0, 0)
        writer.add_image('Sanity/Recons_BeforeTrain', grid_recon0, 0)
    except Exception as e:
        print(f"Sanity logging skipped: {e}")
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_loss_total = 0.0
        val_loss_total = 0.0
        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)
        autoencoder.train()
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs, latent = autoencoder(data.float())
            outputs = match_shape(outputs, data)  # CHANGE: enforce exact shape match before loss
            loss = criterion(outputs, data.float())
            loss.backward()
            optimizer.step()
        autoencoder.eval()
        with torch.no_grad():
            for data in train_dataloader:
                data = data.to(device)
                train_outputs, _ = autoencoder(data.float())
                train_outputs = match_shape(train_outputs, data)  # CHANGE
                train_loss = criterion(train_outputs, data.float())
                train_loss_total += train_loss.item()
        with torch.no_grad():
            # CHANGE: also compute PSNR from MSE on validation
            psnr_vals = []
            for val_data in val_dataloader:
                val_data = val_data.to(device)
                val_outputs, _ = autoencoder(val_data.float())
                val_outputs = match_shape(val_outputs, val_data)  # CHANGE
                val_loss = criterion(val_outputs, val_data.float())
                val_loss_total += val_loss.item()
                # Compute PSNR based on MSE
                mse_batch = torch.mean((val_outputs - val_data) ** 2).detach().item()
                if args.normalize:
                    max_val = 1.0
                else:
                    # Use dynamic max to estimate peak value; guard small values
                    max_val = max(1e-8, float(val_data.max().detach().cpu().item()))
                mse_eps = max(mse_batch, 1e-12)
                import math as _math
                psnr_vals.append(20.0 * _math.log10(max_val) - 10.0 * _math.log10(mse_eps))
        train_loss_avg = train_loss_total / max(1, num_train_batches)
        val_loss_avg = val_loss_total / max(1, num_val_batches)
        Losses.append(train_loss_avg)
        ValLosses.append(val_loss_avg)
        # CHANGE: epoch PSNR (average over val batches)
        if psnr_vals:
            psnr_epoch = float(sum(psnr_vals) / len(psnr_vals))
            PSNRs.append(psnr_epoch)
            print(f'  PSNR (val avg): {psnr_epoch:.2f} dB')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}')
        # CHANGE: Per-epoch reconstruction logging (first batch of val)
        try:
            autoencoder.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    sample = batch.to(device)
                    recon_epoch, _ = autoencoder(sample.float())
                    recon_epoch = match_shape(recon_epoch, sample)  # CHANGE
                    break
            cols_ep = min(6, sample.size(0))
            grid_orig_ep = make_grid(sample[:cols_ep].detach().cpu(), nrow=cols_ep, normalize=True, scale_each=True)
            grid_recon_ep = make_grid(recon_epoch[:cols_ep].detach().cpu(), nrow=cols_ep, normalize=True, scale_each=True)
            writer.add_image('EpochRecon/Originals', grid_orig_ep, epoch + 1)
            writer.add_image('EpochRecon/Recons', grid_recon_ep, epoch + 1)
        except Exception as e:
            print(f"Epoch reconstruction logging skipped: {e}")

    # TensorBoard: log per-epoch losses
    writer.add_scalar('Loss/train', train_loss_avg, epoch + 1)
    writer.add_scalar('Loss/val', val_loss_avg, epoch + 1)
    # CHANGE: Log PSNR if computed
    try:
        if len(PSNRs) >= (epoch + 1) and not (np.isnan(PSNRs[-1]) if PSNRs else True):
            writer.add_scalar('Metrics/PSNR_val', PSNRs[-1], epoch + 1)
    except Exception:
        pass
    writer.flush()

    # Save model and plot for this dataset
    model_name = os.path.basename(folder.rstrip('/')) + '_model.pth'
    torch.save(autoencoder.state_dict(), model_name)
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    plt.plot(Losses)
    plt.plot(ValLosses)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Losses for {os.path.basename(folder)}')
    plot_name = os.path.basename(folder.rstrip('/')) + '_loss_plot.png'
    plt.savefig(plot_name)
    plt.show()

    # Save another copy of the trained model following the requested naming
    # PSEUDOCODE: Save model weights to conv_autoencoder.pth
    # - serialize model parameters with state_dict
    # - write to file conv_autoencoder.pth in current working directory
    torch.save(autoencoder.state_dict(), 'conv_autoencoder.pth')

    # Batch visualization: original vs reconstructed in a 2 x N grid (similar to provided snippet)
    # PSEUDOCODE:
    # - switch model to eval mode (disable dropout/BN updates)
    # - with no gradients, take the first batch from val_dataloader as "data"
    # - run autoencoder to get reconstructions "recon"
    # - plot a 2-row grid with Originals on top and Reconstructions on bottom for up to 7 samples
    autoencoder.eval()
    with torch.no_grad():
        # Get one batch from validation loader
        for batch in val_dataloader:
            data = batch.to(device)
            recon, _ = autoencoder(data.float())
            break

    cols = min(7, data.size(0))  # number of samples to display
    fig, ax = plt.subplots(2, cols, figsize=(2 * cols, 4), dpi=250)
    for i in range(cols):
        # Move tensors to CPU numpy arrays
        orig = data[i].detach().cpu().numpy()     # shape [C,H,W] or [H,W]
        rec  = recon[i].detach().cpu().numpy()    # shape [C,H,W] or [H,W]

        # Handle grayscale (C=1) and color (C=3) cases
        if orig.ndim == 3:
            C, H, W = orig.shape
            if C == 1:
                ax[0, i].imshow(orig[0], cmap='gray', origin='lower')
            else:
                ax[0, i].imshow(np.transpose(orig, (1, 2, 0)), origin='lower')
        else:
            ax[0, i].imshow(orig, cmap='gray', origin='lower')
        ax[0, i].axis('off')

        if rec.ndim == 3:
            C, H, W = rec.shape
            if C == 1:
                ax[1, i].imshow(rec[0], cmap='gray', origin='lower')
            else:
                ax[1, i].imshow(np.transpose(rec, (1, 2, 0)), origin='lower')
        else:
            ax[1, i].imshow(rec, cmap='gray', origin='lower')
        ax[1, i].axis('off')

    # Titles
    ax[0, 0].set_title('Original')
    ax[1, 0].set_title('Reconstructed')
    plt.suptitle(f'Batch Reconstructions from {os.path.basename(folder)}')
    plt.tight_layout()
    plt.show()

    # TensorBoard: log image grids (originals and reconstructions)
    try:
        # Normalize to [0,1] for TB visualization
        grid_orig = make_grid(data[:cols].detach().cpu(), nrow=cols, normalize=True, scale_each=True)
        grid_recon = make_grid(recon[:cols].detach().cpu(), nrow=cols, normalize=True, scale_each=True)
        writer.add_image('Images/Originals', grid_orig, num_epochs)
        writer.add_image('Images/Reconstructed', grid_recon, num_epochs)
        writer.flush()
    except Exception as e:
        print(f"TensorBoard image logging skipped: {e}")

    # ------------------------------------------------------------------
    # Diagnostic 1: Histogram of pixel-wise reconstruction errors
    # ------------------------------------------------------------------
    # PSEUDOCODE:
    # - Initialize an empty list to collect errors
    # - For a limited number of validation batches (to keep it fast):
    #   - Move batch to device and run autoencoder to get reconstructions
    #   - Compute absolute pixel error |recon - input|
    #   - Flatten to 1D and move to CPU numpy, append to list
    # - Concatenate all errors and plot a histogram; save figure per dataset
    errors = []
    max_val_batches = 8  # limit for speed; increase for more coverage
    autoencoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            batch = batch.to(device)
            recon, _ = autoencoder(batch.float())
            batch_err = (recon - batch).abs().detach().cpu().numpy().ravel()
            errors.append(batch_err)
    if errors:
        import numpy as _np
        errors_all = _np.concatenate(errors)
        plt.figure(figsize=(6, 4))
        plt.hist(errors_all, bins=60, color='steelblue', alpha=0.85)
        plt.xlabel('Absolute pixel error')
        plt.ylabel('Count')
        plt.title(f'Reconstruction Error Histogram: {os.path.basename(folder)}')
        hist_name = os.path.basename(folder.rstrip('/')) + '_recon_error_hist.png'
        plt.tight_layout()
        plt.savefig(hist_name)
        plt.show()
        # TensorBoard: log histogram of reconstruction errors
        try:
            import torch as _torch
            writer.add_histogram('Hist/ReconstructionError', _torch.from_numpy(errors_all), num_epochs)
            writer.flush()
        except Exception as e:
            print(f"TensorBoard histogram logging skipped: {e}")

    # ------------------------------------------------------------------
    # Diagnostic 2: Latent space visualization (PCA and t-SNE)
    # ------------------------------------------------------------------
    # PSEUDOCODE:
    # - Initialize an empty list for latent vectors
    # - Iterate over a subset of the full dataloader to get latents
    #   - Move batch to device, forward pass, collect latent to CPU numpy
    # - Stack latents into an array [N, latent_dim]
    # - Apply PCA to 2D and scatter plot
    # - Apply t-SNE to 2D (on a sample if very large) and scatter plot
    latents_list = []
    max_total_samples = 3000  # cap for speed; adjust as needed
    collected = 0
    autoencoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, lat = autoencoder(batch.float())
            lat_np = lat.detach().cpu().numpy()
            latents_list.append(lat_np)
            collected += lat_np.shape[0]
            if collected >= max_total_samples:
                break
    if latents_list:
        import numpy as _np
        latents_np = _np.vstack(latents_list)
        # PCA 2D
        pca = PCA(n_components=2, random_state=0)
        lat_pca = pca.fit_transform(latents_np)
        plt.figure(figsize=(5, 4))
        plt.scatter(lat_pca[:, 0], lat_pca[:, 1], s=6, alpha=0.6, c='tab:blue')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title(f'Latent PCA: {os.path.basename(folder)} (n={lat_pca.shape[0]})')
        pca_name = os.path.basename(folder.rstrip('/')) + '_latent_pca.png'
        plt.tight_layout()
        plt.savefig(pca_name)
        plt.show()

        # t-SNE 2D (can be slow; run on a sample if large)
        sample_for_tsne = latents_np
        max_tsne = 2000
        if sample_for_tsne.shape[0] > max_tsne:
            idx = _np.random.RandomState(0).choice(sample_for_tsne.shape[0], size=max_tsne, replace=False)
            sample_for_tsne = sample_for_tsne[idx]
        try:
            tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=0, method='barnes_hut')
            lat_tsne = tsne.fit_transform(sample_for_tsne)
            plt.figure(figsize=(5, 4))
            plt.scatter(lat_tsne[:, 0], lat_tsne[:, 1], s=6, alpha=0.6, c='tab:green')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f'Latent t-SNE: {os.path.basename(folder)} (n={lat_tsne.shape[0]})')
            tsne_name = os.path.basename(folder.rstrip('/')) + '_latent_tsne.png'
            plt.tight_layout()
            plt.savefig(tsne_name)
            plt.show()
        except Exception as e:
            print(f"t-SNE failed or was skipped: {e}")

        # TensorBoard: add embeddings for projector (uses PCA/t-SNE in TB UI)
        try:
            # Sample up to 500 points for embedding to keep log size manageable
            import numpy as _np
            max_embed = 500
            idx = _np.random.RandomState(0).choice(latents_np.shape[0], size=min(max_embed, latents_np.shape[0]), replace=False)
            latents_sample = latents_np[idx]

            # Build matching label images from dataset - force all to same size
            label_imgs = []
            target_H, target_W = 64, 64  # Fixed small size for embedding thumbnails
            
            for j in idx[:min(len(idx), len(dataset))]:
                img = dataset[j]
                # Convert to tensor if needed
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(np.asarray(img)).float()
                else:
                    img = img.float()
                
                # Ensure CHW format
                if img.ndim == 2:
                    img = img.unsqueeze(0)  # Add channel dim -> [1,H,W]
                elif img.ndim == 3:
                    pass  # Already [C,H,W]
                else:
                    img = img.view(-1, img.size(-2), img.size(-1))  # Force to CHW
                
                # Force to 3 channels for RGB thumbnails
                if img.size(0) == 1:
                    img = img.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]
                elif img.size(0) > 3:
                    img = img[:3]  # Take first 3 channels
                elif img.size(0) == 2:
                    img = torch.cat([img, img[:1]], dim=0)  # Duplicate first channel to get 3
                
                # Resize to fixed target size using interpolation
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Normalize to [0,1] range for thumbnails
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                label_imgs.append(img.unsqueeze(0))  # Add batch dim -> [1,3,64,64]
                if len(label_imgs) >= len(idx):
                    break

            if label_imgs:
                # All images now guaranteed to be [1,3,64,64], so cat will work
                label_img_tensor = torch.cat(label_imgs[:len(idx)], dim=0)
                print(f"Embedding: {len(idx)} latents, label images shape: {label_img_tensor.shape}")
                writer.add_embedding(
                    torch.from_numpy(latents_sample).float(), 
                    label_img=label_img_tensor, 
                    global_step=num_epochs, 
                    tag='Embeddings/Latent'
                )
            else:
                writer.add_embedding(torch.from_numpy(latents_sample).float(), global_step=num_epochs, tag='Embeddings/Latent')
            writer.flush()
        except Exception as e:
            print(f"TensorBoard embedding logging skipped: {e}")

    # Close TensorBoard writer for this dataset
    writer.close()
