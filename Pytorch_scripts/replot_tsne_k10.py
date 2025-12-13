#!/usr/bin/env python3
"""
Replot t-SNE with k=10 clusters from existing latent embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import sys

# Path to your completed training results
RESULTS_DIR = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir"

# Load latent embeddings
print(f"Loading latent embeddings from: {RESULTS_DIR}")
data = loadmat(os.path.join(RESULTS_DIR, 'latent_embeddings.mat'))

# Check what's in the file
print(f"Keys in file: {list(data.keys())}")

# Load the t-SNE embeddings (should already be 2D from previous run)
if 'tsne_embeddings' in data:
    tsne_result = data['tsne_embeddings']
    print(f"Loaded t-SNE embeddings: {tsne_result.shape}")
elif 'latent_embeddings' in data:
    print("WARNING: Only latent embeddings found, need to run t-SNE...")
    from sklearn.manifold import TSNE
    latent = data['latent_embeddings']
    print(f"Running t-SNE on {latent.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    tsne_result = tsne.fit_transform(latent)
    print(f"t-SNE complete: {tsne_result.shape}")
else:
    print("ERROR: No embeddings found!")
    sys.exit(1)

# Get dataset name
dataset_name = "Unsupervised_database_MostlyManual"

# Cluster with k=10
print(f"\nClustering with k=10...")
k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(tsne_result)

# Compute silhouette score
silhouette = silhouette_score(tsne_result, labels)
print(f"Silhouette score for k=10: {silhouette:.3f}")

# Create the plot
plt.figure(figsize=(12, 10))
colors = plt.cm.tab10(np.linspace(0, 1, k))

for i in range(k):
    mask = labels == i
    count = np.sum(mask)
    plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                c=[colors[i]], label=f'Cluster {i} (n={count})',
                alpha=0.6, s=30, edgecolors='none')

plt.xlabel('t-SNE 1', fontsize=14)
plt.ylabel('t-SNE 2', fontsize=14)
plt.title(f't-SNE Latent Space (k={k}, silhouette={silhouette:.3f})', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

# Add dataset info
plt.text(0.02, 0.02, f'Dataset: {dataset_name}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', alpha=0.7)

# Save
output_path = os.path.join(RESULTS_DIR, 'tsne_k10_replot.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Show cluster statistics
print(f"\nCluster sizes:")
for i in range(k):
    count = np.sum(labels == i)
    pct = 100 * count / len(labels)
    print(f"  Cluster {i}: {count:5d} samples ({pct:5.2f}%)")

plt.show()
