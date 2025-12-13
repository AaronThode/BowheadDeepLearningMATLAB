#!/usr/bin/env python3
"""
Create Interactive Web-Accessible 3D Visualization of Whale Call Embeddings

This script:
1. Loads the 100K latent embeddings
2. Computes 3D t-SNE and UMAP projections
3. Finds natural clusters
4. Creates an interactive HTML file with Plotly
5. The HTML file can be hosted anywhere (GitHub Pages, personal server, etc.)

The resulting visualization is:
- Fully interactive (rotate, zoom, pan)
- Self-contained (no external dependencies)
- Accessible from any web browser
- Small file size (~5-10 MB)
"""
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v08_100E_32LD_CombinedDatasets_100K_Date20251125-171340.dir"
OUTPUT_HTML = os.path.join(RESULTS_DIR, "interactive_whale_calls_3d.html")

# Subsample for faster computation and smaller file size (optional)
USE_SUBSAMPLE = True
SUBSAMPLE_SIZE = 10000  # Use 10K points for web visualization

print("="*70)
print("INTERACTIVE WEB VISUALIZATION: Whale Call Embeddings")
print("="*70)
print(f"Output: {OUTPUT_HTML}")
print()

# ============================================================================
# LOAD DATA AND FILE MAPPINGS
# ============================================================================
print("Loading latent embeddings...")
data = loadmat(os.path.join(RESULTS_DIR, 'latent_embeddings.mat'))
latent_embeddings = data['latent_embeddings']
dataset_name = str(data['dataset_label'][0]) if 'dataset_label' in data else "CombinedDatasets"

print(f"  ✓ Loaded {latent_embeddings.shape[0]:,} whale calls")
print(f"  ✓ Latent space: {latent_embeddings.shape[1]}D")
print(f"  ✓ Dataset: {dataset_name}")

# ============================================================================
# RECONSTRUCT FILE PATHS
# ============================================================================
print("\nReconstructing file paths from dataset order...")
import glob

# Determine which dataset(s) were used
if dataset_name == "CombinedDatasets":
    dataset_paths = [
        "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/models/Unsupervised_database_With_Airguns.dir",
        "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/models/Unsupervised_database_No_Airguns.dir"
    ]
else:
    # Single dataset
    dataset_paths = [f"/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/models/{dataset_name}.dir"]

# Collect all files in dataset order (try multiple extensions)
all_files = []
for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):
        # Try different file patterns
        for pattern in ["*.mat", "*.jpg", "*.png", "*"]:
            files = sorted(glob.glob(os.path.join(dataset_path, pattern)))
            if files:
                all_files.extend(files)
                print(f"  Found {len(files):,} {pattern} files in {os.path.basename(dataset_path)}")
                break

# If no files found, create generic IDs
if len(all_files) < latent_embeddings.shape[0]:
    print(f"  ⚠ Only found {len(all_files)} files, generating IDs for {latent_embeddings.shape[0]:,} embeddings")
    # Determine source dataset for each embedding
    n_total = latent_embeddings.shape[0]
    # Assume roughly equal split for combined datasets
    if dataset_name == "CombinedDatasets":
        n_half = n_total // 2
        file_names = [f"WithAirguns_sample_{i:06d}.mat" for i in range(n_half)]
        file_names += [f"NoAirguns_sample_{i:06d}.mat" for i in range(n_total - n_half)]
    else:
        file_names = [f"{dataset_name}_sample_{i:06d}.mat" for i in range(n_total)]
else:
    # Extract just the filenames for cleaner display
    file_names = [os.path.basename(f) for f in all_files[:len(latent_embeddings)]]
    
print(f"  ✓ Mapped {len(file_names):,} file identifiers to embeddings")

# Subsample if requested
if USE_SUBSAMPLE and latent_embeddings.shape[0] > SUBSAMPLE_SIZE:
    print(f"\n  Subsampling to {SUBSAMPLE_SIZE:,} points for web visualization...")
    np.random.seed(42)
    indices = np.random.choice(latent_embeddings.shape[0], SUBSAMPLE_SIZE, replace=False)
    latent_embeddings = latent_embeddings[indices]
    file_names = [file_names[i] for i in indices]
    print(f"  ✓ Using {latent_embeddings.shape[0]:,} points")

# ============================================================================
# COMPUTE 3D t-SNE
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: Computing 3D t-SNE...")
print("="*70)
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
# CREATE INTERACTIVE PLOTLY VISUALIZATION
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: Creating interactive web visualization...")
print("="*70)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    horizontal_spacing=0.05
)

# Color palette for clusters
def get_colors(labels):
    """Generate colors for clusters, gray for noise"""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Use a nice color palette
    colors = []
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    for label in labels:
        if label == -1:  # Noise
            colors.append('#cccccc')
        else:
            colors.append(color_palette[label % len(color_palette)])
    
    return colors, n_clusters

# Get colors for both methods
colors_tsne, n_clusters_tsne = get_colors(labels_tsne)
colors_umap, n_clusters_umap = get_colors(labels_umap)

# Add t-SNE plot with file name information
fig.add_trace(
    go.Scatter3d(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        z=tsne_result[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_tsne,
            line=dict(width=0),
            opacity=0.7
        ),
        customdata=[[file_names[i], labels_tsne[i]] for i in range(len(labels_tsne))],
        hovertemplate='<b>🐋 Whale Call</b><br>' +
                      '<b>File:</b> %{customdata[0]}<br>' +
                      '<b>Cluster:</b> %{customdata[1]}<br>' +
                      '<b>t-SNE Coords:</b><br>' +
                      '  Dim 1: %{x:.2f}<br>' +
                      '  Dim 2: %{y:.2f}<br>' +
                      '  Dim 3: %{z:.2f}' +
                      '<extra></extra>',
        name='t-SNE'
    ),
    row=1, col=1
)

# Add UMAP plot with file name information
fig.add_trace(
    go.Scatter3d(
        x=umap_result[:, 0],
        y=umap_result[:, 1],
        z=umap_result[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_umap,
            line=dict(width=0),
            opacity=0.7
        ),
        customdata=[[file_names[i], labels_umap[i]] for i in range(len(labels_umap))],
        hovertemplate='<b>🐋 Whale Call</b><br>' +
                      '<b>File:</b> %{customdata[0]}<br>' +
                      '<b>Cluster:</b> %{customdata[1]}<br>' +
                      '<b>UMAP Coords:</b><br>' +
                      '  Dim 1: %{x:.2f}<br>' +
                      '  Dim 2: %{y:.2f}<br>' +
                      '  Dim 3: %{z:.2f}' +
                      '<extra></extra>',
        name='UMAP'
    ),
    row=1, col=2
)

# Update layout with proper axis labels and descriptions
fig.update_layout(
    title=dict(
        text=f'<b>Bowhead Whale Call Embeddings - 3D Interactive Visualization</b><br>' +
             f'<sub>Dataset: {dataset_name} | {len(latent_embeddings):,} calls | ' +
             f't-SNE: {n_clusters_tsne} clusters | UMAP: {n_clusters_umap} clusters | ' +
             f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    height=950,
    showlegend=False,
    margin=dict(t=180, b=50, l=50, r=50),
    scene=dict(
        domain=dict(x=[0.0, 0.475], y=[0.0, 0.85]),
        xaxis=dict(title='t-SNE Dimension 1', backgroundcolor="rgb(230, 230,230)"),
        yaxis=dict(title='t-SNE Dimension 2', backgroundcolor="rgb(230, 230,230)"),
        zaxis=dict(title='t-SNE Dimension 3', backgroundcolor="rgb(230, 230,230)"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    scene2=dict(
        domain=dict(x=[0.525, 1.0], y=[0.0, 0.85]),
        xaxis=dict(title='UMAP Dimension 1', backgroundcolor="rgb(230, 230,230)"),
        yaxis=dict(title='UMAP Dimension 2', backgroundcolor="rgb(230, 230,230)"),
        zaxis=dict(title='UMAP Dimension 3', backgroundcolor="rgb(230, 230,230)"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    annotations=[
        # t-SNE title
        dict(
            text='<b style="font-size:16px">3D t-SNE (Local Structure)</b>',
            x=0.2375,
            y=0.96,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            showarrow=False,
            font=dict(size=16)
        ),
        # t-SNE description
        dict(
            text='<span style="font-size:11px">t-SNE preserves local neighborhoods, grouping similar whale calls together.<br>' +
                 'Points close together represent acoustically similar vocalizations.</span>',
            x=0.2375,
            y=0.91,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            showarrow=False,
            font=dict(size=11, color='#666666')
        ),
        # UMAP title
        dict(
            text='<b style="font-size:16px">3D UMAP (Global Structure)</b>',
            x=0.7625,
            y=0.96,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            showarrow=False,
            font=dict(size=16)
        ),
        # UMAP description
        dict(
            text='<span style="font-size:11px">UMAP preserves both local and global structure, revealing call type distributions.<br>' +
                 'Distinct clusters indicate different categories of whale vocalizations.</span>',
            x=0.7625,
            y=0.91,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            showarrow=False,
            font=dict(size=11, color='#666666')
        )
    ]
)

# Save as standalone HTML
print(f"\n  Saving HTML file...")
fig.write_html(
    OUTPUT_HTML,
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'whale_calls_3d',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
)

file_size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)
print(f"  ✓ HTML file created: {file_size_mb:.1f} MB")
print(f"\n{'='*70}")
print("SUCCESS!")
print("="*70)
print(f"\nInteractive visualization saved to:")
print(f"  {OUTPUT_HTML}")
print(f"\nTo make this accessible on the web, you have several options:")
print(f"\n1. GitHub Pages (FREE):")
print(f"   - Create a new repo or use an existing one")
print(f"   - Push the HTML file to the repo")
print(f"   - Enable GitHub Pages in Settings")
print(f"   - Access at: https://username.github.io/repo/interactive_whale_calls_3d.html")
print(f"\n2. Netlify Drop (FREE):")
print(f"   - Go to https://app.netlify.com/drop")
print(f"   - Drag and drop the HTML file")
print(f"   - Get instant URL like: https://random-name.netlify.app/interactive_whale_calls_3d.html")
print(f"\n3. Google Drive (FREE):")
print(f"   - Upload to Google Drive")
print(f"   - Make it publicly accessible")
print(f"   - Share the link")
print(f"\n4. Your own web server:")
print(f"   - Upload via FTP/SCP to your server")
print(f"   - Access at: https://yourserver.com/interactive_whale_calls_3d.html")
print(f"\n{'='*70}")

# Save clustering results
output_data = {
    'tsne_3d': tsne_result,
    'umap_3d': umap_result,
    'labels_tsne': labels_tsne,
    'labels_umap': labels_umap,
    'n_clusters_tsne': n_clusters_tsne,
    'n_clusters_umap': n_clusters_umap,
    'dataset_name': dataset_name,
    'n_samples': len(latent_embeddings)
}

mat_output = os.path.join(RESULTS_DIR, 'interactive_3d_results.mat')
savemat(mat_output, output_data)
print(f"\nClustering results also saved to: {mat_output}")
print("="*70)
