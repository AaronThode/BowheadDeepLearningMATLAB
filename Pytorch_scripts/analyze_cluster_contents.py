#!/usr/bin/env python3
"""
Analyze what each cluster represents by examining the filenames within each cluster.

This script:
1. Loads latent embeddings and cluster assignments
2. Maps each sample to its original filename
3. Analyzes filename patterns within each cluster to understand what they contain

USAGE:
    python analyze_cluster_contents.py <results_dir>
    python analyze_cluster_contents.py <results_dir> --save-cluster-files
    python analyze_cluster_contents.py <results_dir> --k-clusters 8
    
EXAMPLES:
    # Analyze existing clusters from .mat file
    python analyze_cluster_contents.py ../results/Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir
    
    # Re-cluster with different k and analyze
    python analyze_cluster_contents.py ../results/... --k-clusters 10
    
    # Save list of files in each cluster
    python analyze_cluster_contents.py ../results/... --save-cluster-files

OUTPUT:
    For each cluster, this shows:
    - Number of samples
    - Breakdown by call type (extracted from filename)
    - Breakdown by year, site, and DASAR
    - Sample filenames
"""
import numpy as np
import os
import sys
import argparse
import re
from collections import Counter, defaultdict
from scipy.io import loadmat, savemat

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None
    silhouette_score = None


def parse_filename(filename: str) -> dict:
    """
    Parse a spectrogram filename to extract metadata.
    
    Expected format: S510G0T20100815T000017_Type4.mat
    - S: Prefix
    - 5: Site number
    - 10: Year (20XX)
    - G: DASAR letter (A-G)
    - 0: DASAR subindex
    - T: Time marker
    - 20100815: Date (YYYYMMDD)
    - T000017: Time (THHMMSS)
    - Type4: Call type
    
    Returns dict with parsed fields
    """
    info = {
        'filename': filename,
        'site': None,
        'year': None,
        'dasar': None,
        'date': None,
        'time': None,
        'call_type': None,
        'call_type_name': None,
    }
    
    # Extract call type from end (e.g., "Type4" -> 4)
    type_match = re.search(r'Type(\d+)', filename)
    if type_match:
        call_type = int(type_match.group(1))
        info['call_type'] = call_type
        
        # Map type numbers to names (based on Bowhead whale call classification)
        type_names = {
            0: 'Unknown/Other',
            1: 'Upcall',
            2: 'Downcall', 
            3: 'Constant',
            4: 'U-shaped',
            5: 'N-shaped',
            6: 'Other FM',
            7: 'Complex',
            8: 'Bearded Seal',
            9: 'Walrus',
        }
        info['call_type_name'] = type_names.get(call_type, f'Type{call_type}')
    
    # Extract site, year, DASAR from start (e.g., "S510G0" -> site=5, year=10, dasar=G0)
    start_match = re.match(r'S(\d)(\d{2})([A-Z])(\d)', filename)
    if start_match:
        info['site'] = start_match.group(1)
        info['year'] = '20' + start_match.group(2)  # Convert 10 -> 2010
        info['dasar'] = start_match.group(3) + start_match.group(4)  # e.g., "G0"
    
    # Extract date and time (e.g., "T20100815T000017")
    datetime_match = re.search(r'T(\d{8})T(\d{6})', filename)
    if datetime_match:
        info['date'] = datetime_match.group(1)  # YYYYMMDD
        info['time'] = datetime_match.group(2)  # HHMMSS
    
    return info


def analyze_cluster(cluster_filenames: list, cluster_id: int) -> dict:
    """Analyze the contents of a single cluster."""
    n_samples = len(cluster_filenames)
    
    # Parse all filenames
    parsed = [parse_filename(f) for f in cluster_filenames]
    
    # Count call types
    call_types = Counter(p['call_type_name'] for p in parsed if p['call_type_name'])
    
    # Count by year
    years = Counter(p['year'] for p in parsed if p['year'])
    
    # Count by site
    sites = Counter(p['site'] for p in parsed if p['site'])
    
    # Count by DASAR
    dasars = Counter(p['dasar'] for p in parsed if p['dasar'])
    
    return {
        'cluster_id': cluster_id,
        'n_samples': n_samples,
        'call_types': call_types,
        'years': years,
        'sites': sites,
        'dasars': dasars,
        'sample_filenames': cluster_filenames[:10],  # Keep first 10 as examples
    }


def print_cluster_analysis(analysis: dict, show_samples: bool = True):
    """Print formatted analysis for a cluster."""
    print(f"\n{'='*70}")
    print(f"CLUSTER {analysis['cluster_id']} ({analysis['n_samples']:,} samples)")
    print('='*70)
    
    # Call types
    print("\n📊 CALL TYPE BREAKDOWN:")
    for call_type, count in analysis['call_types'].most_common():
        pct = 100 * count / analysis['n_samples']
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"   {call_type:20s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Dominant call type
    if analysis['call_types']:
        dominant = analysis['call_types'].most_common(1)[0]
        print(f"\n   ➤ DOMINANT: {dominant[0]} ({100*dominant[1]/analysis['n_samples']:.1f}%)")
    
    # Years
    if analysis['years']:
        print("\n📅 YEAR BREAKDOWN:")
        for year, count in sorted(analysis['years'].items()):
            pct = 100 * count / analysis['n_samples']
            print(f"   {year}: {count:6,} ({pct:5.1f}%)")
    
    # Sites
    if analysis['sites']:
        print("\n📍 SITE BREAKDOWN:")
        for site, count in sorted(analysis['sites'].items()):
            pct = 100 * count / analysis['n_samples']
            print(f"   Site {site}: {count:6,} ({pct:5.1f}%)")
    
    # DASARs
    if len(analysis['dasars']) <= 10:
        print("\n🎤 DASAR BREAKDOWN:")
        for dasar, count in analysis['dasars'].most_common():
            pct = 100 * count / analysis['n_samples']
            print(f"   {dasar}: {count:6,} ({pct:5.1f}%)")
    else:
        print(f"\n🎤 DASARs: {len(analysis['dasars'])} different DASARs")
    
    # Sample filenames
    if show_samples and analysis['sample_filenames']:
        print("\n📁 SAMPLE FILENAMES:")
        for f in analysis['sample_filenames'][:5]:
            print(f"   • {f}")


def analyze_clusters(results_dir: str, k_clusters: int = None, save_files: bool = False):
    """Main analysis function."""
    
    # Load latent embeddings
    latent_path = os.path.join(results_dir, 'latent_embeddings.mat')
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"No latent_embeddings.mat found in {results_dir}")
    
    print(f"Loading data from: {results_dir}")
    data = loadmat(latent_path)
    
    latent = data['latent_embeddings']
    tsne = data.get('tsne_embeddings', None)
    
    # Check if filenames are in the .mat file
    if 'filenames' in data:
        filenames = [str(f[0]) if isinstance(f, np.ndarray) else str(f) 
                    for f in data['filenames'].flatten()]
        print(f"Loaded {len(filenames)} filenames from .mat file")
    else:
        print("WARNING: No filenames in .mat file!")
        print("Run: python map_embeddings_to_files.py --save-mapping")
        return
    
    print(f"Dataset size: {latent.shape[0]:,} samples, {latent.shape[1]} latent dimensions")
    
    # Get or compute clusters
    if k_clusters is not None and KMeans is not None:
        print(f"\nRe-clustering with k={k_clusters}...")
        kmeans = KMeans(n_clusters=k_clusters, n_init='auto', random_state=42)
        clusters = kmeans.fit_predict(latent)
        
        # Compute silhouette score
        if silhouette_score is not None:
            score = silhouette_score(latent, clusters)
            print(f"Silhouette score: {score:.3f}")
    elif 'clusters' in data:
        clusters = data['clusters'].flatten().astype(int)
        print(f"\nUsing existing clusters from .mat file")
    else:
        print("No clusters found. Use --k-clusters to specify number of clusters")
        return
    
    n_clusters = len(set(clusters))
    print(f"Number of clusters: {n_clusters}")
    
    # Analyze each cluster
    print("\n" + "="*70)
    print("CLUSTER-BY-CLUSTER ANALYSIS")
    print("="*70)
    
    all_analyses = []
    
    for cluster_id in sorted(set(clusters)):
        # Get filenames for this cluster
        mask = clusters == cluster_id
        cluster_filenames = [filenames[i] for i in range(len(filenames)) if mask[i]]
        
        # Analyze
        analysis = analyze_cluster(cluster_filenames, cluster_id)
        all_analyses.append(analysis)
        
        # Print
        print_cluster_analysis(analysis)
        
        # Save file list if requested
        if save_files:
            output_file = os.path.join(results_dir, f'cluster_{cluster_id}_files.txt')
            with open(output_file, 'w') as f:
                for fname in cluster_filenames:
                    f.write(fname + '\n')
            print(f"\n   📁 Saved file list: {output_file}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE: Dominant Call Type per Cluster")
    print("="*70)
    print(f"{'Cluster':>8} {'N Samples':>12} {'Dominant Type':>20} {'Percentage':>12}")
    print("-"*70)
    
    for analysis in all_analyses:
        if analysis['call_types']:
            dominant = analysis['call_types'].most_common(1)[0]
            dom_name = dominant[0]
            dom_pct = 100 * dominant[1] / analysis['n_samples']
        else:
            dom_name = 'Unknown'
            dom_pct = 0
        
        print(f"{analysis['cluster_id']:>8} {analysis['n_samples']:>12,} {dom_name:>20} {dom_pct:>11.1f}%")
    
    # Save summary to .mat
    summary_data = {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'cluster_sizes': [a['n_samples'] for a in all_analyses],
    }
    
    # Build call type matrix
    all_call_types = sorted(set(
        ct for a in all_analyses for ct in a['call_types'].keys()
    ))
    call_type_matrix = np.zeros((n_clusters, len(all_call_types)))
    for i, analysis in enumerate(all_analyses):
        for j, ct in enumerate(all_call_types):
            call_type_matrix[i, j] = analysis['call_types'].get(ct, 0)
    
    summary_data['call_type_names'] = np.array(all_call_types, dtype=object)
    summary_data['call_type_counts'] = call_type_matrix
    
    summary_path = os.path.join(results_dir, 'cluster_analysis.mat')
    savemat(summary_path, summary_data)
    print(f"\n✓ Saved cluster analysis to: {summary_path}")
    
    return all_analyses


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cluster contents by examining filenames"
    )
    parser.add_argument("results_dir",
                       help="Directory containing latent_embeddings.mat")
    parser.add_argument("--k-clusters", type=int, default=None,
                       help="Number of clusters (re-cluster if provided)")
    parser.add_argument("--save-cluster-files", action='store_true',
                       help="Save list of files in each cluster")
    
    args = parser.parse_args()
    
    analyze_clusters(args.results_dir, args.k_clusters, args.save_cluster_files)


if __name__ == "__main__":
    main()
