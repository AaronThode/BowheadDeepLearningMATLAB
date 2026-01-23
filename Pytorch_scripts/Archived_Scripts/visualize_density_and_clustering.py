#!/usr/bin/env python3
"""
Create two complementary t-SNE visualizations:
1. Continuous density plot (heatmap showing data concentration)
2. DBSCAN clustering (finds natural groups without forcing a number)

WHAT IS BEING MEASURED:
- X and Y axes: 2D coordinates from t-SNE dimension reduction
- t-SNE compresses 32 numbers (latent features) down to 2 numbers for plotting
- Points close together = whale calls that sound similar
- Points far apart = whale calls that sound different

PLOT 1 (Density): Shows WHERE most whale calls are concentrated
PLOT 2 (DBSCAN): Shows IF there are natural groupings (islands vs continuum)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
import os

# ============================================================================
# LOAD DATA
# ============================================================================
RESULTS_DIR = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir"

print("="*70)
print("t-SNE Analysis: Continuous Density + Natural Clustering")
print("="*70)
print(f"\nLoading data from: {os.path.basename(RESULTS_DIR)}")

data = loadmat(os.path.join(RESULTS_DIR, 'latent_embeddings.mat'))
tsne_result = data['tsne_embeddings']
dataset_name = str(data['dataset_label'][0]) if 'dataset_label' in data else "Unknown"

print(f"Loaded {tsne_result.shape[0]:,} whale call spectrograms")
print(f"Dataset: {dataset_name}")
print(f"t-SNE coordinates: 2D (reduced from 32D latent space)")

# ============================================================================
# CREATE FIGURE WITH 2 SUBPLOTS
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# ============================================================================
# PLOT 1: CONTINUOUS DENSITY HEATMAP
# ============================================================================
print(f"\n{'='*70}")
print("PLOT 1: Continuous Density Heatmap")
print("="*70)
print("\nWHAT THIS SHOWS:")
print("  - Color intensity = how many whale calls are in that region")
print("  - Hot colors (yellow/red) = high concentration of similar calls")
print("  - Cool colors (blue/purple) = sparse regions, unusual calls")
print("  - Smooth gradients = continuous variation (not discrete types)")

print("\nX-AXIS: t-SNE dimension 1 (captures major acoustic variation)")
print("Y-AXIS: t-SNE dimension 2 (captures secondary acoustic variation)")
print("\nComputing density distribution...")

# Use Kernel Density Estimation for smooth density
xy = tsne_result.T
kde = gaussian_kde(xy)

# Create grid for density calculation
x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
density = np.reshape(kde(positions).T, xx.shape)

# Plot density heatmap
im = ax1.contourf(xx, yy, density, levels=20, cmap='viridis', alpha=0.8)
scatter1 = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                       c='white', s=1, alpha=0.3, edgecolors='none',
                       label=f'{len(tsne_result):,} whale calls')

ax1.set_xlabel('t-SNE Dimension 1 (Primary Acoustic Features)', fontsize=12, fontweight='bold')
ax1.set_ylabel('t-SNE Dimension 2 (Secondary Acoustic Features)', fontsize=12, fontweight='bold')
ax1.set_title('CONTINUOUS DENSITY MAP\nColor = Concentration of Similar Whale Calls', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=10)

# Add colorbar
cbar1 = plt.colorbar(im, ax=ax1, label='Density (calls per unit area)')
cbar1.set_label('Density (calls per unit area)', fontsize=11)

# Add interpretation text
ax1.text(0.02, 0.98, 
         'INTERPRETATION:\n' +
         '• Bright regions = many similar calls\n' +
         '• Dark regions = rare/unusual calls\n' +
         '• Smooth gradient = continuous variation',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

print("  ✓ Density plot complete")

# ============================================================================
# PLOT 2: DBSCAN NATURAL CLUSTERING
# ============================================================================
print(f"\n{'='*70}")
print("PLOT 2: DBSCAN Natural Clustering")
print("="*70)
print("\nWHAT THIS SHOWS:")
print("  - DBSCAN finds groups WITHOUT forcing a specific number")
print("  - Each color = a naturally occurring cluster (if any)")
print("  - Gray points = 'noise' (don't belong to any cluster)")
print("  - Reveals if calls form discrete types or continuous spectrum")

print("\nX-AXIS: t-SNE dimension 1 (same as Plot 1)")
print("Y-AXIS: t-SNE dimension 2 (same as Plot 1)")
print("\nRunning DBSCAN clustering...")

# DBSCAN parameters
eps = 3.0  # Maximum distance between points in same cluster
min_samples = 50  # Minimum points needed to form a cluster

dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
labels = dbscan.fit_predict(tsne_result)

# Count clusters (excluding noise = -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nRESULTS:")
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points: {n_noise:,} ({100*n_noise/len(labels):.1f}%)")

# Plot clusters
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise points in gray
        col = [0.5, 0.5, 0.5, 0.3]
        label = f'Noise (n={n_noise:,})'
        marker = '.'
        size = 5
    else:
        class_member_mask = (labels == k)
        n_in_cluster = np.sum(class_member_mask)
        label = f'Cluster {k} (n={n_in_cluster:,})'
        marker = 'o'
        size = 20
    
    class_member_mask = (labels == k)
    xy = tsne_result[class_member_mask]
    ax2.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                s=size, alpha=0.6, label=label, edgecolors='none')

ax2.set_xlabel('t-SNE Dimension 1 (Primary Acoustic Features)', fontsize=12, fontweight='bold')
ax2.set_ylabel('t-SNE Dimension 2 (Secondary Acoustic Features)', fontsize=12, fontweight='bold')
ax2.set_title(f'DBSCAN CLUSTERING\n{n_clusters} Natural Groups Found (eps={eps}, min_samples={min_samples})', 
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='best', fontsize=9, ncol=2)

# Add interpretation text
interpretation = 'INTERPRETATION:\n'
if n_clusters == 0:
    interpretation += '• NO distinct clusters found\n• Data forms continuous spectrum\n• Whale calls vary gradually'
elif n_clusters <= 3:
    interpretation += f'• {n_clusters} broad groups detected\n• May indicate major call categories\n• But high noise suggests overlap'
else:
    interpretation += f'• {n_clusters} groups detected\n• Multiple call types present\n• Check if biologically meaningful'

ax2.text(0.02, 0.98, interpretation,
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

print("  ✓ DBSCAN plot complete")

# Print cluster statistics
if n_clusters > 0:
    print(f"\nCLUSTER BREAKDOWN:")
    for k in range(n_clusters):
        count = np.sum(labels == k)
        pct = 100 * count / len(labels)
        print(f"  Cluster {k}: {count:6,} calls ({pct:5.2f}%)")
    print(f"  Noise:      {n_noise:6,} calls ({100*n_noise/len(labels):5.2f}%)")

# ============================================================================
# SAVE AND DISPLAY
# ============================================================================
plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, 'tsne_density_and_clustering.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ Saved combined plot to:")
print(f"  {output_path}")
print("="*70)

# ============================================================================
# SUMMARY INTERPRETATION
# ============================================================================
print(f"\n{'='*70}")
print("SUMMARY: What This Tells You About Your Whale Calls")
print("="*70)
print("\nPLOT 1 (Density):")
print("  Shows WHERE in acoustic space your whale calls cluster")
print("  Bright = common call types, Dark = rare variations")
print("\nPLOT 2 (DBSCAN):")
if n_clusters == 0:
    print("  NO distinct groupings found")
    print("  → Your whale calls form a CONTINUOUS SPECTRUM")
    print("  → Like a color gradient, not separate categories")
    print("  → Individual variation dominates over type differences")
elif n_clusters <= 3:
    print(f"  Found {n_clusters} broad groupings")
    print("  → May represent major call categories (e.g., song types)")
    print(f"  → But {100*n_noise/len(labels):.1f}% noise suggests significant overlap")
    print("  → Boundaries are fuzzy, not sharp")
else:
    print(f"  Found {n_clusters} distinct groups")
    print("  → Multiple call types detected")
    print("  → Could represent different behaviors, contexts, or individuals")
    print("  → Investigate what distinguishes each cluster acoustically")

print("\nX & Y AXES EXPLAINED:")
print("  • Not frequency or time - those are in the original spectrograms")
print("  • These are 'SUMMARY COORDINATES' from the autoencoder")
print("  • The autoencoder compressed each call to 32 numbers")
print("  • t-SNE further compressed to 2 numbers for visualization")
print("  • Distance = similarity (close = similar, far = different)")
print("  • Direction = type of difference (pitch vs duration vs noise, etc.)")
print("="*70)

plt.show()
