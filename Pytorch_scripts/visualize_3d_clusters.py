#!/usr/bin/env python3
"""
Interactive 3D Visualization of Latent Space Clusters

Uses Plotly to create interactive 3D scatter plots that you can spin, zoom, and rotate.
Works with saved latent_embeddings.mat from trained autoencoder runs.

USAGE:
    python visualize_3d_clusters.py <output_dir> [--method tsne|umap|pca] [--k 5]

REQUIREMENTS:
    pip install plotly umap-learn

FEATURES:
    - Interactive 3D rotation with mouse
    - Hover over points to see sample details
    - Toggle clusters on/off by clicking legend
    - Export as HTML for sharing
    - Multiple dimensionality reduction methods (t-SNE 3D, UMAP, PCA)
"""
import argparse
import os
import numpy as np
from scipy.io import loadmat, savemat
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    import umap
except ImportError:
    umap = None


def load_latent_embeddings(output_dir: str):
    """Load latent embeddings from saved .mat file."""
    mat_path = os.path.join(output_dir, 'latent_embeddings.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"latent_embeddings.mat not found in {output_dir}")
    
    data = loadmat(mat_path)
    latent = data['latent_embeddings']
    dataset_label = str(data.get('dataset_label', ['Unknown'])[0]) if 'dataset_label' in data else 'Unknown'
    
    print(f"Loaded {latent.shape[0]} samples with {latent.shape[1]}-dim embeddings")
    print(f"Dataset: {dataset_label}")
    
    return latent, data, dataset_label


def compute_3d_embedding(latent: np.ndarray, method: str = 'tsne', 
                         random_state: int = 42, perplexity: float = 30.0):
    """
    Compute 3D embedding using specified dimensionality reduction method.
    
    Methods:
        - 'tsne': t-SNE in 3D (slower but preserves local structure)
        - 'umap': UMAP in 3D (faster, preserves both local and global structure)
        - 'pca': PCA to 3D (fastest, linear projection)
    """
    print(f"\n{'='*60}")
    print(f"Computing 3D embedding using {method.upper()}...")
    print(f"Processing {latent.shape[0]:,} samples with {latent.shape[1]}-dim latent vectors")
    print(f"{'='*60}")
    
    if method == 'tsne':
        if TSNE is None:
            raise ImportError("sklearn not available. Install with: pip install scikit-learn")
        perplexity = min(perplexity, (latent.shape[0] - 1) / 3.0)
        perplexity = max(2.0, min(perplexity, latent.shape[0] - 1))
        print(f"t-SNE parameters: perplexity={perplexity:.1f}")
        print(f"This may take 5-15 minutes for large datasets...")
        import time
        start = time.time()
        emb_3d = TSNE(n_components=3, random_state=random_state, 
                      perplexity=perplexity, learning_rate='auto').fit_transform(latent)
        elapsed = time.time() - start
        print(f"✓ t-SNE completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    elif method == 'umap':
        if umap is None:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        n_neighbors = min(15, latent.shape[0] - 1)
        print(f"UMAP parameters: n_neighbors={n_neighbors}")
        if latent.shape[0] > 50000:
            print(f"⚠ Large dataset - this may take 2-5 minutes...")
        else:
            print(f"Computing... (typically 10-30 seconds)")
        import time
        import sys
        start = time.time()
        sys.stdout.flush()  # Force output to appear
        emb_3d = umap.UMAP(n_components=3, random_state=random_state, 
                          n_neighbors=n_neighbors, verbose=True).fit_transform(latent)
        elapsed = time.time() - start
        print(f"✓ UMAP completed in {elapsed:.1f}s")
    
    elif method == 'pca':
        print(f"Computing PCA... (instant)")
        import time
        start = time.time()
        pca = PCA(n_components=3, random_state=random_state)
        emb_3d = pca.fit_transform(latent)
        elapsed = time.time() - start
        explained_var = pca.explained_variance_ratio_
        print(f"✓ PCA completed in {elapsed:.2f}s")
        print(f"  Explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}, PC3={explained_var[2]:.1%}")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'umap', or 'pca'")
    
    print(f"✓ 3D embedding shape: {emb_3d.shape}")
    return emb_3d


def find_optimal_k(latent: np.ndarray, max_k: int = 10, random_state: int = 42):
    """Find optimal number of clusters using silhouette score."""
    from sklearn.metrics import silhouette_score
    
    max_k = min(max_k, latent.shape[0] // 2)
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print(f"\nFinding optimal k (testing k=2 to k={max_k})...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        labels = kmeans.fit_predict(latent)
        score = silhouette_score(latent, labels)
        silhouette_scores.append(score)
        if k <= 5 or k % 2 == 0:
            print(f"  k={k}: silhouette={score:.3f}")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
    
    return optimal_k


def cluster_embeddings(latent: np.ndarray, k: int, random_state: int = 42):
    """Perform k-means clustering on latent embeddings."""
    print(f"\nClustering into k={k} groups...")
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
    clusters = kmeans.fit_predict(latent)
    
    for cluster_id in range(k):
        count = np.sum(clusters == cluster_id)
        print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(clusters):.1f}%)")
    
    return clusters


def create_interactive_3d_plot(emb_3d: np.ndarray, clusters: np.ndarray, 
                               dataset_label: str, method: str, k: int,
                               output_path: str = None):
    """
    Create interactive 3D scatter plot with Plotly.
    
    Features:
        - Mouse drag to rotate
        - Scroll to zoom
        - Click legend to toggle clusters
        - Hover for sample details
    """
    print(f"\nCreating interactive 3D plot...")
    
    # Create color palette
    colors = px.colors.qualitative.Plotly
    if k > len(colors):
        colors = px.colors.sample_colorscale("turbo", [n/(k-1) for n in range(k)])
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each cluster
    for cluster_id in range(k):
        mask = clusters == cluster_id
        fig.add_trace(go.Scatter3d(
            x=emb_3d[mask, 0],
            y=emb_3d[mask, 1],
            z=emb_3d[mask, 2],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(
                size=4,
                color=colors[cluster_id % len(colors)],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=(
                f'<b>Cluster {cluster_id}</b><br>' +
                'X: %{x:.2f}<br>' +
                'Y: %{y:.2f}<br>' +
                'Z: %{z:.2f}<br>' +
                '<extra></extra>'
            )
        ))
    
    # Update layout
    method_name = method.upper()
    fig.update_layout(
        title=dict(
            text=f'3D {method_name} Latent Space (k={k})<br><sub>Dataset: {dataset_label}</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=f'{method_name} 1',
            yaxis_title=f'{method_name} 2',
            zaxis_title=f'{method_name} 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Save as HTML
    if output_path:
        print(f"Saving HTML file...")
        fig.write_html(output_path)
        print(f"✓ Saved interactive plot to: {output_path}")
        print(f"  → Open this file in your browser to spin/zoom/interact!")
        print(f"  → File path: {output_path}")
    
    # Don't auto-show (can hang) - user opens HTML manually
    # fig.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D visualization of autoencoder latent space clusters"
    )
    parser.add_argument("output_dir", help="Directory containing latent_embeddings.mat")
    parser.add_argument("--method", choices=['tsne', 'umap', 'pca'], default='umap',
                       help="3D embedding method (default: umap - fastest and most informative)")
    parser.add_argument("--k", type=int, default=0,
                       help="Number of clusters (0=auto-detect optimal)")
    parser.add_argument("--perplexity", type=float, default=30.0,
                       help="t-SNE perplexity (only for --method tsne)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--save-mat", action='store_true',
                       help="Save 3D embeddings to .mat file for future use")
    
    args = parser.parse_args()
    
    # Load latent embeddings
    latent, data, dataset_label = load_latent_embeddings(args.output_dir)
    
    # Check if 3D embeddings already computed
    mat_path_3d = os.path.join(args.output_dir, f'latent_embeddings_3d_{args.method}.mat')
    if os.path.exists(mat_path_3d):
        print(f"\nFound existing 3D {args.method.upper()} embeddings, loading...")
        data_3d = loadmat(mat_path_3d)
        emb_3d = data_3d['embedding_3d']
    else:
        # Compute 3D embedding
        emb_3d = compute_3d_embedding(latent, method=args.method, 
                                     random_state=args.seed,
                                     perplexity=args.perplexity)
        
        # Optionally save 3D embeddings
        if args.save_mat:
            data_3d = {
                'embedding_3d': emb_3d,
                'method': args.method,
                'latent_embeddings': latent
            }
            savemat(mat_path_3d, data_3d)
            print(f"Saved 3D embeddings to: {mat_path_3d}")
    
    # Determine k
    if args.k == 0:
        k = find_optimal_k(latent, max_k=10, random_state=args.seed)
    else:
        k = args.k
    
    # Cluster
    clusters = cluster_embeddings(latent, k, random_state=args.seed)
    
    # Create interactive plot
    output_html = os.path.join(args.output_dir, f'interactive_3d_{args.method}_k{k}.html')
    create_interactive_3d_plot(emb_3d, clusters, dataset_label, args.method, k, output_html)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"  Method: {args.method.upper()}")
    print(f"  Clusters: {k}")
    print(f"  HTML: {output_html}")
    print(f"  -> Open the HTML file in a browser to spin/zoom/interact!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
