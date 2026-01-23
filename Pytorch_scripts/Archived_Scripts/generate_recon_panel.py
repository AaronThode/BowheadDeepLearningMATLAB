import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat


def build_autoencoder(nrow, ncol, n_channels=16, latent_dim=16):
    nrow_reduced = int(nrow / 8)
    ncol_reduced = int(ncol / 8)
    nel_reduced = nrow_reduced * ncol_reduced * n_channels

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.conv3 = nn.Conv2d(8, n_channels, 3, padding=1)
            self.t_conv1 = nn.ConvTranspose2d(n_channels, 8, 2, stride=2)
            self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
            self.t_conv3 = nn.ConvTranspose2d(4, 1, [2, 2], stride=[2, 2])
            self.fc1 = nn.Linear(nel_reduced, latent_dim)
            self.fc2 = nn.Linear(latent_dim, nel_reduced)
            self.pool = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, nel_reduced)
            latent = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(latent))
            x = x.view(-1, n_channels, nrow_reduced, ncol_reduced)
            x = torch.relu(self.t_conv1(x))
            x = torch.relu(self.t_conv2(x))
            out = torch.sigmoid(self.t_conv3(x))
            return out, latent

    return Autoencoder(), nel_reduced


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(repo_root, "Spectrogram_Image_Database.dir", "Unsupervised_images.dir")
    default_model = os.path.join(repo_root, "Unsupervised_images.dir_model.pth")

    p = argparse.ArgumentParser(description="Generate a 2xN panel of original vs reconstructed spectrograms")
    p.add_argument("--data-dir", default=default_data, help="Folder containing .mat files with SNR_gram variable")
    p.add_argument("--model", default=default_model, help="Path to trained model weights (.pth)")
    p.add_argument("--cols", type=int, default=7, help="Number of examples to show across")
    p.add_argument("--out", default=os.path.join(repo_root, "recon_panel.png"), help="Output PNG path")
    args = p.parse_args()

    # Collect .mat files
    if not os.path.isdir(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        print("Pass --data-dir to point to your spectrogram .mat folder.")
        return 2
    mat_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.mat"), recursive=True))
    if not mat_files:
        print(f"No .mat files found under {args.data_dir}")
        return 3

    # Load one file to infer shape
    sample_im = loadmat(mat_files[0]).get("SNR_gram", None)
    if sample_im is None or not isinstance(sample_im, np.ndarray) or sample_im.ndim != 2:
        print(f"SNR_gram missing/invalid in {mat_files[0]}")
        return 4
    nrow, ncol = sample_im.shape

    # Build model and load weights
    autoencoder, _ = build_autoencoder(nrow, ncol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device).float().eval()
    if not os.path.isfile(args.model):
        print(f"Model weights not found: {args.model}")
        print("Point --model to your *_model.pth or conv_autoencoder.pth file.")
        return 5
    state = torch.load(args.model, map_location=device)
    autoencoder.load_state_dict(state)

    # Pick first K valid images
    imgs = []
    for fp in mat_files:
        im = loadmat(fp).get("SNR_gram", None)
        if isinstance(im, np.ndarray) and im.ndim == 2 and im.shape == (nrow, ncol):
            imgs.append(torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float())  # [1,1,H,W]
        if len(imgs) >= args.cols:
            break
    if not imgs:
        print("Could not collect any valid images matching the inferred shape.")
        return 6
    batch = torch.cat(imgs, dim=0).to(device)

    with torch.no_grad():
        recon, _ = autoencoder(batch)

    cols = min(args.cols, batch.size(0))
    fig, ax = plt.subplots(2, cols, figsize=(2 * cols, 4), dpi=250)
    for i in range(cols):
        orig = batch[i].detach().cpu().numpy()
        rec = recon[i].detach().cpu().numpy()
        if orig.ndim == 3 and orig.shape[0] == 1:
            ax[0, i].imshow(orig[0], cmap='gray', origin='lower')
        else:
            ax[0, i].imshow(np.squeeze(orig), cmap='gray', origin='lower')
        ax[0, i].axis('off')
        if rec.ndim == 3 and rec.shape[0] == 1:
            ax[1, i].imshow(rec[0], cmap='gray', origin='lower')
        else:
            ax[1, i].imshow(np.squeeze(rec), cmap='gray', origin='lower')
        ax[1, i].axis('off')
    ax[0, 0].set_title('Original')
    ax[1, 0].set_title('Reconstructed')
    plt.suptitle('Originals (top) vs Autoencoder Reconstructions (bottom)')
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f"Saved panel -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
