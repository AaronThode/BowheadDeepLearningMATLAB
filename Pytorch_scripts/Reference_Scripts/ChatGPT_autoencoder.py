import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# 1. Data Preparation
# ------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((84, 128)),
    transforms.ToTensor(),  # scales to [0,1]
])

train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
val_dataset   = datasets.ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ------------------------------
# 2. Model Definition
# ------------------------------

class GrayAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(GrayAutoencoder, self).__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # (1,84,128) -> (32,42,64)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64,21,32)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # (128,11,16)
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(128 * 11 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 11 * 16)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # -> (64,22,32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> (32,44,64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # -> (1,88,128)
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        x = x.view(x.size(0), 128, 11, 16)
        x = self.decoder(x)
        # Crop from 88 to 84 in height (center)
        return x[:, :, 2:86, :]

# ------------------------------
# 3. Setup & Checkpoint
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GrayAutoencoder(latent_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

checkpoint_path = "gray_autoencoder_checkpoint.pth"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

start_epoch = 0
train_losses, val_losses = [], []

if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    print(f"✅ Resumed from epoch {start_epoch}")

# ------------------------------
# 4. Train & Evaluate Functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recons = model(imgs)
        loss = criterion(recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recons = model(imgs)
            loss = criterion(recons, imgs)
            running_loss += loss.item()
    return running_loss / len(loader)

# ------------------------------
# 5. Visualization Function
# ------------------------------

def save_reconstructions(model, loader, epoch, num_images=6):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recons = model(imgs)
    imgs = imgs.cpu().numpy()
    recons = recons.cpu().numpy()

    plt.figure(figsize=(12,4))
    for i in range(num_images):
        # Original
        plt.subplot(2, num_images, i+1)
        plt.imshow(np.squeeze(imgs[i]), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original")
        # Reconstruction
        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(np.squeeze(recons[i]), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed")
    plt.suptitle(f"Epoch {epoch+1} Reconstructions", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"recon_epoch_{epoch+1:02d}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Saved reconstruction sample to {save_path}")

# ------------------------------
# 6. Training Loop
# ------------------------------

num_epochs = 30

for epoch in range(start_epoch, num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.5f}  Val Loss: {val_loss:.5f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }, checkpoint_path)
    print("💾 Checkpoint saved")

    # Save sample reconstruction every epoch
    save_reconstructions(model, val_loader, epoch)

# ------------------------------
# 7. Plot Loss Curves
# ------------------------------

plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png"))
plt.show()

torch.save(model.state_dict(), "gray_autoencoder_final.pth")
print("✅ Training complete! Final model saved.")
