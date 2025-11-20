#!/usr/bin/env python3
"""
Map latent embedding indices to their corresponding image files.

This script recreates the exact file ordering used during training to map
each row in latent_embeddings.mat back to its original image file.

USAGE:
    python map_embeddings_to_files.py <results_dir> [--index 100]
    python map_embeddings_to_files.py <results_dir> --save-mapping
    
EXAMPLES:
    # Find which file corresponds to embedding index 100
    python map_embeddings_to_files.py ../results/Autoencoder_v04_32LD_Balanced_Date20251117-185303.dir --index 100
    
    # Save complete mapping to CSV file
    python map_embeddings_to_files.py ../results/Autoencoder_v04_32LD_Balanced_Date20251117-185303.dir --save-mapping
"""
import numpy as np
import glob
import os
import sys
import argparse
from scipy.io import loadmat


def get_dataset_path_and_seed(results_dir):
    """Extract dataset path and seed from timing_log.txt"""
    timing_log_path = os.path.join(results_dir, 'timing_log.txt')
    
    if not os.path.exists(timing_log_path):
        raise FileNotFoundError(f"No timing_log.txt found in {results_dir}")
    
    dataset_name = None
    seed = None
    
    with open(timing_log_path, 'r') as f:
        for line in f:
            if 'Dataset:' in line:
                dataset_name = line.split(':')[1].strip()
            elif 'Seed:' in line:
                seed_str = line.split(':')[1].strip()
                seed = int(seed_str) if seed_str != 'None' else None
    
    if dataset_name is None:
        raise ValueError("Could not find dataset name in timing_log.txt")
    
    return dataset_name, seed


def recreate_file_order(data_dir, seed=None):
    """
    Recreate the exact file ordering used during training.
    
    Steps:
    1. Find all .mat files recursively
    2. Sort alphabetically by full path
    3. Apply shuffle with seed if provided
    
    Returns:
        list of file paths in training order
    """
    # Step 1: Find all .mat files (same as SNRDataset.__init__)
    mat_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True))
    
    if not mat_files:
        raise RuntimeError(f"No .mat files found in {data_dir}")
    
    # Step 2: Already sorted alphabetically by glob
    print(f"Found {len(mat_files)} .mat files")
    
    # Step 3: Apply same shuffle as training if seed was used
    if seed is not None:
        print(f"Applying shuffle with seed={seed}")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(mat_files))
        mat_files = [mat_files[i] for i in indices]
    else:
        print("No seed used - files in alphabetical order")
    
    return mat_files


def update_mat_with_filenames(results_dir, file_list):
    """Update latent_embeddings.mat to include filenames array.
    
    Args:
        results_dir: Path to results directory containing latent_embeddings.mat
        file_list: Ordered list of full file paths matching embedding order
    
    Returns:
        True if successful, False otherwise
    """
    from scipy.io import savemat
    
    latent_path = os.path.join(results_dir, 'latent_embeddings.mat')
    
    if not os.path.exists(latent_path):
        print(f"Warning: {latent_path} not found, skipping .mat update")
        return False
    
    try:
        # Load existing .mat data
        data = loadmat(latent_path)
        
        # Create array of basenames (remove paths)
        filenames = np.array([os.path.basename(f) for f in file_list], dtype=object)
        
        # Add filenames to the data dictionary
        data['filenames'] = filenames
        
        # Save updated .mat file
        savemat(latent_path, data)
        
        print(f"Updated {latent_path} with {len(filenames)} filenames")
        return True
        
    except Exception as e:
        print(f"Error updating .mat file: {e}")
        return False


def map_index_to_file(results_dir, embedding_index):
    """Map a specific embedding index to its image file."""
    
    # Get dataset info from training log
    dataset_name, seed = get_dataset_path_and_seed(results_dir)
    print(f"\nDataset: {dataset_name}")
    print(f"Seed: {seed}")
    
    # Find dataset directory (search common locations)
    possible_dirs = [
        f"/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/{dataset_name}",
        f"../BCB_Whale_Datasets/{dataset_name}",
        f"../../BCB_Whale_Datasets/{dataset_name}",
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        raise FileNotFoundError(f"Could not find dataset directory: {dataset_name}")
    
    print(f"Dataset path: {data_dir}")
    
    # Recreate file order
    file_list = recreate_file_order(data_dir, seed)
    
    # Load latent embeddings to verify size
    latent_path = os.path.join(results_dir, 'latent_embeddings.mat')
    if os.path.exists(latent_path):
        data = loadmat(latent_path)
        num_embeddings = data['latent_embeddings'].shape[0]
        print(f"Latent embeddings: {num_embeddings} samples")
        
        if embedding_index >= num_embeddings:
            raise ValueError(f"Index {embedding_index} out of range (max: {num_embeddings-1})")
    
    # Get the file
    if embedding_index >= len(file_list):
        raise ValueError(f"Index {embedding_index} out of range (max: {len(file_list)-1})")
    
    matched_file = file_list[embedding_index]
    
    print(f"\n{'='*70}")
    print(f"MAPPING RESULT:")
    print(f"{'='*70}")
    print(f"Embedding index: {embedding_index}")
    print(f"Corresponds to:  {matched_file}")
    print(f"Filename:        {os.path.basename(matched_file)}")
    print(f"{'='*70}")
    
    # Always update .mat file with filenames
    update_mat_with_filenames(results_dir, file_list)
    
    return matched_file


def save_complete_mapping(results_dir, output_csv=None):
    """Save complete index-to-filename mapping as CSV and update .mat file."""
    
    # Get dataset info
    dataset_name, seed = get_dataset_path_and_seed(results_dir)
    print(f"\nDataset: {dataset_name}")
    print(f"Seed: {seed}")
    
    # Find dataset directory
    possible_dirs = [
        f"/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/{dataset_name}",
        f"../BCB_Whale_Datasets/{dataset_name}",
        f"../../BCB_Whale_Datasets/{dataset_name}",
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        raise FileNotFoundError(f"Could not find dataset directory: {dataset_name}")
    
    print(f"Dataset path: {data_dir}")
    
    # Recreate file order
    file_list = recreate_file_order(data_dir, seed)
    
    # Default output path
    if output_csv is None:
        output_csv = os.path.join(results_dir, 'embedding_to_file_mapping.csv')
    
    # Write CSV
    print(f"\nSaving CSV mapping to: {output_csv}")
    with open(output_csv, 'w') as f:
        f.write("embedding_index,filename\n")
        for idx, filepath in enumerate(file_list):
            filename = os.path.basename(filepath)
            f.write(f"{idx},{filename}\n")
    
    print(f"✓ Saved {len(file_list)} mappings to CSV")
    
    # Always update latent_embeddings.mat with filenames
    print(f"\nUpdating latent_embeddings.mat with filenames...")
    success = update_mat_with_filenames(results_dir, file_list)
    
    if success:
        print(f"✓ Added 'filenames' field to latent_embeddings.mat")
        print(f"  Each row in 'latent_embeddings' now has corresponding filename in 'filenames'")
    
    print(f"\nYou can now look up any embedding index in the CSV or load 'filenames' from the .mat file!")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Map latent embedding indices to original image files"
    )
    parser.add_argument("results_dir", 
                       help="Directory containing latent_embeddings.mat and timing_log.txt")
    parser.add_argument("--index", type=int,
                       help="Look up which file corresponds to this embedding index")
    parser.add_argument("--save-mapping", action='store_true',
                       help="Save complete index-to-filename mapping as CSV")
    parser.add_argument("--output", type=str,
                       help="Output CSV path (default: <results_dir>/embedding_to_file_mapping.csv)")
    
    args = parser.parse_args()
    
    if args.save_mapping:
        save_complete_mapping(args.results_dir, args.output)
    elif args.index is not None:
        map_index_to_file(args.results_dir, args.index)
    else:
        print("Error: Must specify either --index or --save-mapping")
        print("Examples:")
        print(f"  python {sys.argv[0]} <results_dir> --index 100")
        print(f"  python {sys.argv[0]} <results_dir> --save-mapping")
        sys.exit(1)


if __name__ == "__main__":
    main()
