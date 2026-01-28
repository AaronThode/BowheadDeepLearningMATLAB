#!/usr/bin/env python3
"""
3D t-SNE and UMAP Comparison for Whale Calls

Creates side-by-side 3D visualizations comparing:
1. t-SNE 3D - focuses on local structure
2. UMAP 3D - preserves global structure

Both are rotatable and interactive!
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat, savemat
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import umap
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v08_100E_32LD_CombinedDatasets_100K_Date20251125-171340.dir"

print("="*70)
print("3D VISUALIZATION: t-SNE vs UMAP")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\nLoading latent embeddings...")
data = loadmat(os.path.join(RESULTS_DIR, 'latent_embeddings.mat'))
latent_embeddings = data['latent_embeddings']
dataset_name = str(data['dataset_label'][0]) if 'dataset_label' in data else "Unknown"

print(f"  ✓ Loaded {latent_embeddings.shape[0]:,} whale calls")
print(f"  ✓ Latent space: {latent_embeddings.shape[1]}D")
print(f"  ✓ Dataset: {dataset_name}")

# ============================================================================
# COMPUTE 3D t-SNE
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: Computing 3D t-SNE...")
print("="*70)
print("  t-SNE preserves LOCAL neighborhoods")
print("  Similar calls will cluster tightly")
print("  This may take 2-5 minutes...")

tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, 
               max_iter=1000, verbose=1)
tsne_result = tsne_3d.fit_transform(latent_embeddings)

print(f"\n  ✓ t-SNE complete: {tsne_result.shape}")

# ============================================================================
# COMPUTE 3D UMAP
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: Computing 3D UMAP...")
print("="*70)
print("  UMAP preserves BOTH local AND global structure")
print("  Faster and often clearer than t-SNE")
print("  This should take 1-2 minutes...")

reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, 
                    random_state=42, verbose=True)
umap_result = reducer.fit_transform(latent_embeddings)

print(f"\n  ✓ UMAP complete: {umap_result.shape}")

# ============================================================================
# FIND NATURAL CLUSTERS
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: Finding natural clusters...")
print("="*70)

# Cluster t-SNE
print("  Clustering t-SNE...")
dbscan_tsne = DBSCAN(eps=3.0, min_samples=50)
labels_tsne = dbscan_tsne.fit_predict(tsne_result)
n_clusters_tsne = len(set(labels_tsne)) - (1 if -1 in labels_tsne else 0)
n_noise_tsne = list(labels_tsne).count(-1)

print(f"    t-SNE: {n_clusters_tsne} clusters, {n_noise_tsne:,} noise points")

# Cluster UMAP  
print("  Clustering UMAP...")
dbscan_umap = DBSCAN(eps=1.0, min_samples=50)
labels_umap = dbscan_umap.fit_predict(umap_result)
n_clusters_umap = len(set(labels_umap)) - (1 if -1 in labels_umap else 0)
n_noise_umap = list(labels_umap).count(-1)

print(f"    UMAP:  {n_clusters_umap} clusters, {n_noise_umap:,} noise points")

# ============================================================================
# CREATE 3D PLOTS
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: Creating 3D plots...")
print("="*70)

fig = plt.figure(figsize=(20, 9))

# --- PLOT 1: t-SNE 3D ---
ax1 = fig.add_subplot(121, projection='3d')

# Sample points for faster rendering (optional)
n_plot = min(10000, len(tsne_result))
indices = np.random.choice(len(tsne_result), n_plot, replace=False)

# Color by cluster
colors_tsne = plt.cm.Spectral(np.linspace(0, 1, n_clusters_tsne + 1))

for k in set(labels_tsne):
    if k == -1:
        col = [0.7, 0.7, 0.7, 0.2]  # Gray for noise
        size = 3
    else:
        col = colors_tsne[k]
        size = 8
    
    class_mask = (labels_tsne == k)
    plot_mask = class_mask & np.isin(np.arange(len(labels_tsne)), indices)
    xyz = tsne_result[plot_mask]
    
    if len(xyz) > 0:
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                    c=[col], s=size, alpha=0.6, edgecolors='none')

ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
ax1.set_zlabel('t-SNE Dimension 3', fontsize=12, fontweight='bold')
ax1.set_title(f't-SNE 3D: {n_clusters_tsne} Natural Clusters\n' + 
              f'Showing {n_plot:,} of {len(tsne_result):,} whale calls',
              fontsize=14, fontweight='bold', pad=20)

# Rotate to nice viewing angle
ax1.view_init(elev=20, azim=45)

# --- PLOT 2: UMAP 3D ---
ax2 = fig.add_subplot(122, projection='3d')

# Color by cluster
colors_umap = plt.cm.Spectral(np.linspace(0, 1, n_clusters_umap + 1))

for k in set(labels_umap):
    if k == -1:
        col = [0.7, 0.7, 0.7, 0.2]  # Gray for noise
        size = 3
    else:
        col = colors_umap[k]
        size = 8
    
    class_mask = (labels_umap == k)
    plot_mask = class_mask & np.isin(np.arange(len(labels_umap)), indices)
    xyz = umap_result[plot_mask]
    
    if len(xyz) > 0:
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                    c=[col], s=size, alpha=0.6, edgecolors='none')

ax2.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
ax2.set_zlabel('UMAP Dimension 3', fontsize=12, fontweight='bold')
ax2.set_title(f'UMAP 3D: {n_clusters_umap} Natural Clusters\n' + 
              f'Showing {n_plot:,} of {len(umap_result):,} whale calls',
              fontsize=14, fontweight='bold', pad=20)

# Rotate to nice viewing angle
ax2.view_init(elev=20, azim=45)

plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================
output_plot = os.path.join(RESULTS_DIR, 'tsne_umap_3d_comparison.png')
plt.savefig(output_plot, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved plot: {output_plot}")

# Save 3D embeddings
output_data = os.path.join(RESULTS_DIR, 'embeddings_3d.mat')
savemat(output_data, {
    'tsne_3d': tsne_result,
    'umap_3d': umap_result,
    'labels_tsne_3d': labels_tsne,
    'labels_umap_3d': labels_umap,
    'n_clusters_tsne': n_clusters_tsne,
    'n_clusters_umap': n_clusters_umap,
})
print(f"✓ Saved 3D data: {output_data}")

# ============================================================================
# STATISTICS
# ============================================================================
print(f"\n{'='*70}")
print("CLUSTER STATISTICS")
print("="*70)

print(f"\nt-SNE 3D ({n_clusters_tsne} clusters):")
for k in range(n_clusters_tsne):
    count = np.sum(labels_tsne == k)
    pct = 100 * count / len(labels_tsne)
    print(f"  Cluster {k}: {count:7,} ({pct:5.2f}%)")
if n_noise_tsne > 0:
    print(f"  Noise:      {n_noise_tsne:7,} ({100*n_noise_tsne/len(labels_tsne):5.2f}%)")

print(f"\nUMAP 3D ({n_clusters_umap} clusters):")
for k in range(n_clusters_umap):
    count = np.sum(labels_umap == k)
    pct = 100 * count / len(labels_umap)
    print(f"  Cluster {k}: {count:7,} ({pct:5.2f}%)")
if n_noise_umap > 0:
    print(f"  Noise:      {n_noise_umap:7,} ({100*n_noise_umap/len(labels_umap):5.2f}%)")

# ============================================================================
# INTERPRETATION
# ============================================================================
print(f"\n{'='*70}")
print("WHAT THIS SHOWS")
print("="*70)

print("\nAXES (X, Y, Z):")
print("  • 3 dimensions of acoustic variation")
print("  • NOT frequency/time (that's in the spectrograms)")
print("  • Distance = how different whale calls sound")
print("  • Direction = type of acoustic difference")

print("\nt-SNE vs UMAP:")
print("  t-SNE 3D:")
print("    • Excellent for local clustering")
print("    • Similar calls very close together")
print("    • May distort overall structure")
print(f"    • Found {n_clusters_tsne} natural groups")

print("\n  UMAP 3D:")
print("    • Preserves local AND global structure")  
print("    • Better overall topology")
print("    • Faster computation")
print(f"    • Found {n_clusters_umap} natural groups")

print("\nINTERACTION:")
print("  • In the plot window, click and DRAG to rotate")
print("  • Zoom with scroll wheel / pinch")
print("  • Explore from different angles!")

print("\n" + "="*70)
print("✓ COMPLETE! Rotate the plot to explore your data in 3D!")
print("="*70)

plt.show()
