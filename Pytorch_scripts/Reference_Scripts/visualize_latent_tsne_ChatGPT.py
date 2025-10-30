import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import GrayAutoencoder  # assumes model is saved in model.py

# ------------------------------
# CONFIGURATION
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
model_path = "autoencoder_gray128x84.pth"

# ------------------------------
# LOAD MODEL
# ------------------------------
model = GrayAutoencoder(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ------------------------------
# DATA LOADING (adjust path)
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((84, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data/images", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# ------------------------------
# EXTRACT LATENT REPRESENTATIONS
# ------------------------------
latents = []
labels = []

with torch.no_grad():
    for imgs, lbls in dataloader:
        imgs = imgs.to(device)
        # encode only (forward through encoder)
        features = model.encoder(imgs)
        features = model.flatten(features)
        latent_vecs = model.fc_enc(features)
        latents.append(latent_vecs.cpu().numpy())
        labels.append(lbls.cpu().numpy())

latents = np.concatenate(latents, axis=0)
labels = np.concatenate(labels, axis=0)

print(f"Latent space shape: {latents.shape}")

# ------------------------------
# t-SNE VISUALIZATION
# ------------------------------
print("Running t-SNE... this may take a few minutes.")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
latents_2d = tsne.fit_transform(latents)

# Normalize for plotting
x_min, x_max = latents_2d.min(0), latents_2d.max(0)
latents_2d_norm = (latents_2d - x_min) / (x_max - x_min)

# ------------------------------
# PLOT RESULTS
# ------------------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents_2d_norm[:, 0], latents_2d_norm[:, 1], c=labels, cmap='tab10', s=10)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Autoencoder Latent Space")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.show()
