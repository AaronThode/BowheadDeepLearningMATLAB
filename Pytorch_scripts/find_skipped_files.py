#!/usr/bin/env python3
"""
Identify which .mat files fail to load from the training datasets.
This helps explain the mismatch between file count (99,935) and embeddings (99,933).
"""

import glob
import os
from scipy.io import loadmat
from tqdm import tqdm

# Dataset paths (same order as training)
DATASETS = [
    '/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir',
    '/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'
]

def check_file(filepath):
    """Try to load a .mat file and check if it contains valid data."""
    try:
        m = loadmat(filepath)
        if 'SNR_gram' not in m:
            return False, "Missing 'SNR_gram' field"
        
        im = m['SNR_gram']
        if im.shape != (121, 104):
            return False, f"Wrong shape: {im.shape}"
        
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("IDENTIFYING FILES THAT FAIL TO LOAD")
    print("=" * 70)
    
    all_files = []
    failed_files = []
    
    for dataset_path in DATASETS:
        print(f"\nScanning: {os.path.basename(dataset_path)}")
        mat_files = sorted(glob.glob(os.path.join(dataset_path, "*.mat")))
        print(f"  Found {len(mat_files):,} .mat files")
        all_files.extend(mat_files)
    
    print(f"\n{'-' * 70}")
    print(f"Total files to check: {len(all_files):,}")
    print(f"Checking each file for loading errors...\n")
    
    for i, filepath in enumerate(tqdm(all_files, desc="Testing files")):
        success, error_msg = check_file(filepath)
        if not success:
            failed_files.append((i, os.path.basename(filepath), error_msg))
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total files:        {len(all_files):,}")
    print(f"Successfully loaded: {len(all_files) - len(failed_files):,}")
    print(f"Failed to load:     {len(failed_files)}")
    
    if failed_files:
        print(f"\n{'-' * 70}")
        print("FILES THAT FAILED TO LOAD:")
        print(f"{'-' * 70}")
        for idx, filename, error in failed_files:
            print(f"  Index {idx:6d}: {filename}")
            print(f"               Error: {error}")
    else:
        print("\nAll files loaded successfully - no errors found!")
        print("The 2-file mismatch might be due to:")
        print("  - Files filtered during dataset creation")
        print("  - Duplicates removed")
        print("  - Different sorting order")
    
    print("\n" + "=" * 70)
    
    # Save results to file
    output_file = "skipped_files_report.txt"
    with open(output_file, 'w') as f:
        f.write("SKIPPED FILES REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total files:        {len(all_files):,}\n")
        f.write(f"Successfully loaded: {len(all_files) - len(failed_files):,}\n")
        f.write(f"Failed to load:     {len(failed_files)}\n\n")
        
        if failed_files:
            f.write("FILES THAT FAILED TO LOAD:\n")
            f.write("-" * 70 + "\n")
            for idx, filename, error in failed_files:
                f.write(f"Index {idx:6d}: {filename}\n")
                f.write(f"             Error: {error}\n")
    
    print(f"Report saved to: {output_file}")

if __name__ == "__main__":
    main()
