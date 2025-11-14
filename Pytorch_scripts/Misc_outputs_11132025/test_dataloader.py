"""
Quick test to verify CombinedSNRDataset and DataLoader work correctly
"""
import sys
sys.path.insert(0, '/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts')

from Autooencoder_11092025 import CombinedSNRDataset
from torch.utils.data import DataLoader

# Test with the two directories
dirs = [
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyAirguns.dir",
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_ManyWhaleCalls.dir"
]

print("Creating CombinedSNRDataset...")
dataset = CombinedSNRDataset(dirs, normalize=True, seed=42, show_summary=True)

print(f"\nDataset length: {len(dataset)}")
print(f"Target shape: {dataset.target_shape}")

# Test loading a few samples
print("\nTesting individual sample loading:")
for i in range(3):
    sample, label = dataset[i]
    print(f"  Sample {i}: shape={sample.shape}, label={label}, dtype={sample.dtype}")

# Test DataLoader
print("\nTesting DataLoader with batch_size=16:")
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

for batch_idx, (batch_data, batch_labels) in enumerate(loader):
    print(f"  Batch {batch_idx}: data shape={batch_data.shape}, labels shape={batch_labels.shape}")
    if batch_idx >= 2:  # Just test first 3 batches
        break

print("\n✓ DataLoader test completed successfully!")
