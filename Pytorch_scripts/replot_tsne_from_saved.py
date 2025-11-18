#!/usr/bin/env python3
"""
Re-plot t-SNE with different k values WITHOUT retraining the autoencoder.

Loads saved latent embeddings from latent_embeddings.mat and regenerates
t-SNE plots with different clustering options.

USAGE:
    python replot_tsne_from_saved.py <path_to_output_dir> --k 5
    python replot_tsne_from_saved.py results/Autoencoder_v03_Date20251117-123456.dir --k 8
    python replot_tsne_from_saved.py results/Autoencoder_v03_Date20251117-123456.dir --auto
    
    python replot_tsne_from_saved.py /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v03_Date20251117-111454.dir --auto
    python replot_tsne_from_saved.py /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v04_32LD_Balanced_Date20251117-143437.dir --auto

    
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
from scipy.io import loadmat, savemat

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    KMeans = None
    silhouette_score = None


def replot_tsne(output_dir, k_clusters=None, auto_k=False, seed=42):
    """
    Re-plot t-SNE from saved latent embeddings.
    
    Args:
        output_dir: Directory containing latent_embeddings.mat
        k_clusters: Number of clusters (if None, uses saved optimal_k)
        auto_k: If True, re-run optimal k analysis
        seed: Random seed for reproducible clustering
    """
    # Load saved embeddings
    latent_path = os.path.join(output_dir, 'latent_embeddings.mat')
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"No latent_embeddings.mat found in {output_dir}\n"
                               f"Run the training script first to generate embeddings.")
    
    print(f"Loading embeddings from: {latent_path}")
    data = loadmat(latent_path)
    imp_z = data['latent_embeddings']
    emb = data['tsne_embeddings']
    saved_clusters = data['clusters'].flatten()
    saved_k = int(data['optimal_k'][0, 0]) if data['optimal_k'].size > 0 else 2
    perplexity = float(data['perplexity'][0, 0]) if data['perplexity'].size > 0 else 30.0
    dataset_label = str(data['dataset_label'][0]) if 'dataset_label' in data else 'Unknown'
    
    print(f"Loaded {imp_z.shape[0]} samples with {imp_z.shape[1]}-dim latent space")
    print(f"Original clustering: k={saved_k}")
    
    # Determine final k
    if auto_k:
        # Re-run optimal k analysis
        if KMeans is None or silhouette_score is None:
            print("Warning: scikit-learn not available, using saved k")
            optimal_k = saved_k
        else:
            print("Finding optimal number of clusters...")
            max_k = min(10, imp_z.shape[0] // 2)
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=seed)
                labels_temp = kmeans_temp.fit_predict(imp_z)
                score = silhouette_score(imp_z, labels_temp)
                silhouette_scores.append(score)
                print(f"  k={k}: silhouette={score:.3f}")
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
            
            # Save elbow plot
            plt.figure(figsize=(8, 4))
            plt.plot(list(k_range), silhouette_scores, 'bo-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Optimal k Selection')
            plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'replot_optimal_k_analysis.png'), dpi=150)
            plt.close()
            print(f"Saved elbow plot to: replot_optimal_k_analysis.png")
    elif k_clusters is not None:
        optimal_k = k_clusters
        print(f"Using specified k={optimal_k}")
    else:
        optimal_k = saved_k
        print(f"Using saved optimal k={optimal_k}")
    
    # Re-cluster with new k
    if KMeans is not None:
        print(f"Clustering with k={optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=seed)
        clusters = kmeans.fit_predict(imp_z)
        
        # Calculate silhouette score
        if silhouette_score is not None and optimal_k > 1:
            sil_score = silhouette_score(imp_z, clusters)
            print(f"Silhouette score: {sil_score:.3f}")
    else:
        print("Warning: KMeans not available, using saved clusters")
        clusters = saved_clusters
    
    # Generate color map for all clusters
    cmap = plt.colormaps.get_cmap('tab10').resampled(optimal_k)
    plt.figure(figsize=(7, 6))
    
    # Plot each cluster separately for legend
    for cluster_id in range(optimal_k):
        mask = clusters == cluster_id
        count = np.sum(mask)
        color = cmap(cluster_id)
        plt.scatter(emb[mask, 0], emb[mask, 1], 
                   c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id} (n={count})')
    
    plt.title(f't-SNE Latent Space (k={optimal_k}, perplexity={perplexity:.1f})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
    
    # Add dataset label
    plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
               ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
    
    plt.tight_layout()
    
    # Save with descriptive filename
    output_name = f'tsne_latent_k{optimal_k}.png'
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=160)
    plt.close()
    
    print(f"Saved re-plotted t-SNE to: {output_name}")
    
    # Save updated clustering
    updated_data = {
        'latent_embeddings': imp_z,
        'tsne_embeddings': emb,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'perplexity': perplexity,
        'dataset_label': dataset_label
    }
    updated_path = os.path.join(output_dir, f'latent_embeddings_k{optimal_k}.mat')
    savemat(updated_path, updated_data)
    print(f"Saved updated embeddings to: latent_embeddings_k{optimal_k}.mat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-plot t-SNE from saved embeddings")
    parser.add_argument("output_dir", help="Directory containing latent_embeddings.mat")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters (overrides saved value)")
    parser.add_argument("--auto", action='store_true', help="Auto-detect optimal k using silhouette scores")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clustering")
    args = parser.parse_args()
    
    replot_tsne(args.output_dir, k_clusters=args.k, auto_k=args.auto, seed=args.seed)
    print("\nDone! No retraining required.")
