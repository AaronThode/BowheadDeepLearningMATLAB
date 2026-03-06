#!/usr/bin/env python3
"""
Resume v10 training from checkpoint epoch 60
Completes epochs 61-100 and generates all final outputs
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datetime import datetime
import time
import gc

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    TSNE = None
    KMeans = None
    silhouette_score = None

# Import the model class and utility functions from the original script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Autoencoder_v02_20251118 import (
    ImprovedAutoencoder, SNRDataset, match_shape_center,
    select_samples_for_outputs, save_reconstruction_panels,
    _minmax_norm, set_global_seed
)

# Configuration
CHECKPOINT_DIR = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v10_100E_32LD__32C_CombinedDatasets_100K_Date20251211-163729.dir"
DATA_DIRS = [
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir",
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir"
]

# Training parameters (matching v10)
LATENT_DIM = 32
CHANNELS = 32
EPOCHS_TOTAL = 100
LR = 1e-3
SEED = 42
BATCH_SIZE = 32
EXTRA_CONV = False

# Output parameters
NUMBER_OUTPUT_IMAGE_SAMPLES = 30
PANEL_GROUP_SIZE = 3
SHOW_ERROR_PLOTS = False


def resume_training():
    """Resume training from the last checkpoint."""
    print("="*70)
    print("RESUMING v10 TRAINING FROM CHECKPOINT")
    print("="*70)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seed
    set_global_seed(SEED)
    
    # Load datasets
    print(f"\nLoading data from {len(DATA_DIRS)} directories...")
    datasets = []
    for i, dir_path in enumerate(DATA_DIRS, 1):
        print(f"  [{i}] {dir_path}")
        ds = SNRDataset(dir_path, normalize=True, seed=SEED, show_summary=True)
        datasets.append(ds)
    dataset = ConcatDataset(datasets)
    print(f"Total combined samples: {len(dataset)}")
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Get image dimensions
    sample, _ = dataset[0]
    nrow, ncol = sample.shape[-2], sample.shape[-1]
    print(f"Image dimensions: {nrow} x {ncol}")
    
    # Initialize model
    print(f"\nInitializing model (latent_dim={LATENT_DIM}, channels={CHANNELS})...")
    model = ImprovedAutoencoder(
        nrow=nrow, ncol=ncol,
        latent_dim=LATENT_DIM,
        base_channels=CHANNELS,
        extra_conv=EXTRA_CONV
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch60.pth')
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', [])
    
    print(f"Resuming from epoch {start_epoch}/{EPOCHS_TOTAL}")
    print(f"Previous loss: {checkpoint['loss']:.6f}")
    print(f"Previous losses history: {len(losses)} epochs")
    
    # Continue training
    criterion = nn.MSELoss()
    model.train()
    
    print(f"\nContinuing training for epochs {start_epoch+1}-{EPOCHS_TOTAL}...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS_TOTAL):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_data)
            output = match_shape_center(output, (nrow, ncol))
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        
        with torch.no_grad():
            o_min = float(output.min().cpu())
            o_max = float(output.max().cpu())
        
        elapsed = time.time() - training_start_time
        eta_seconds = (elapsed / (epoch - start_epoch + 1)) * (EPOCHS_TOTAL - epoch - 1)
        eta_minutes = eta_seconds / 60
        
        print(f"  Epoch {epoch+1:3d}/{EPOCHS_TOTAL}: Loss={avg_loss:.4f} | out[{o_min:.3f}, {o_max:.3f}] | {epoch_time:.1f}s | ETA: {eta_minutes:.1f}min")
        sys.stdout.flush()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, checkpoint_path)
            print(f"  >> Checkpoint saved: {checkpoint_path}")
            sys.stdout.flush()
    
    training_elapsed = time.time() - training_start_time
    print(f"\nTraining complete: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
    print(f"  Avg per epoch: {training_elapsed/(EPOCHS_TOTAL-start_epoch):.1f}s")
    
    # Save final model
    model_path = os.path.join(CHECKPOINT_DIR, 'autoencoder_clean.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to: {model_path}")
    
    # Generate all final outputs
    print("\n" + "="*70)
    print("GENERATING FINAL OUTPUTS")
    print("="*70)
    
    model.eval()
    
    # Extract latent embeddings for all samples
    print(f"\nExtracting latent embeddings for {len(dataset)} samples...")
    all_latent = []
    with torch.no_grad():
        for i in range(len(dataset)):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{len(dataset)}")
            sample, _ = dataset[i]
            sample = sample.unsqueeze(0).to(device)
            _, latent = model(sample)
            all_latent.append(latent.cpu())
    improved_latent_full = torch.cat(all_latent, dim=0)
    
    # Load visualization samples
    print("\nLoading visualization samples...")
    viz_samples = min(30, len(dataset))
    data_list = []
    for i in range(viz_samples):
        sample, _ = dataset[i]
        data_list.append(sample.unsqueeze(0))
    data_tensor = torch.cat(data_list, dim=0)
    
    # Compute reconstructions
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        improved_recon, _ = model(data_tensor)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol))
    
    data_np = data_tensor.squeeze(1).cpu().numpy()
    
    # Plot 1: Reconstruction comparison
    print("\nGenerating reconstruction comparison plot...")
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    cols = min(10, data_np.shape[0])
    fig, axes = plt.subplots(2, cols, figsize=(15, 6))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        imp_recon = improved_recon[i, 0].cpu().numpy()
        axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Autoencoder Reconstructions (epochs={EPOCHS_TOTAL}, latent_dim={LATENT_DIM})')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reconstructions.png")
    
    # Plot 2: Training loss
    print("\nGenerating training loss plot...")
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (epochs={EPOCHS_TOTAL})')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved: training_loss.png")
    
    # Plot 3: t-SNE
    imp_z = improved_latent_full.detach().cpu().numpy()
    dataset_label = "CombinedDatasets"
    
    if TSNE is not None:
        print(f"\nComputing t-SNE on {imp_z.shape[0]} samples...")
        perplexity = min(30.0, (imp_z.shape[0] - 1) / 3.0)
        perplexity = max(2.0, min(perplexity, imp_z.shape[0] - 1))
        
        emb = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, learning_rate='auto').fit_transform(imp_z)
        
        # Auto-find optimal k
        if KMeans is not None and silhouette_score is not None:
            print("Finding optimal number of clusters...")
            max_k = min(10, imp_z.shape[0] // 2)
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=SEED)
                labels_temp = kmeans_temp.fit_predict(imp_z)
                score = silhouette_score(imp_z, labels_temp)
                silhouette_scores.append(score)
                print(f"  k={k}: silhouette={score:.3f}")
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
            
            kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=SEED)
            clusters = kmeans.fit_predict(imp_z)
        else:
            optimal_k = 2
            clusters = (emb[:, 0] > np.median(emb[:, 0])).astype(int)
        
        # Plot t-SNE
        cmap = plt.cm.get_cmap('tab10', optimal_k)
        plt.figure(figsize=(7, 6))
        
        for cluster_id in range(optimal_k):
            mask = clusters == cluster_id
            color = cmap(cluster_id)
            plt.scatter(emb[mask, 0], emb[mask, 1], c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
        
        plt.title(f't-SNE Latent Space (k={optimal_k}, perplexity={perplexity:.1f})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
        plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'tsne_latent.png'), dpi=160)
        plt.close()
        print(f"  Saved: tsne_latent.png")
    else:
        emb = np.zeros((imp_z.shape[0], 2))
        clusters = np.zeros(imp_z.shape[0], dtype=int)
        optimal_k = 2
        perplexity = 30.0
    
    # Save latent embeddings
    print("\nSaving latent embeddings...")
    all_file_paths = []
    for ds in dataset.datasets:
        all_file_paths.extend(ds.file_paths)
    filenames = np.array([os.path.basename(all_file_paths[i]) for i in range(len(dataset))], dtype=object)
    reconstruction_filenames = np.array([f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in filenames], dtype=object)
    
    latent_data = {
        'latent_embeddings': imp_z,
        'tsne_embeddings': emb,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'perplexity': perplexity,
        'dataset_label': dataset_label,
        'original_filenames': filenames,
        'reconstruction_filenames': reconstruction_filenames
    }
    savemat(os.path.join(CHECKPOINT_DIR, 'latent_embeddings.mat'), latent_data)
    print(f"  Saved: latent_embeddings.mat")
    
    # Save JPEG reconstruction panels
    print("\nGenerating JPEG reconstruction panels...")
    panel_samples, panel_filenames = select_samples_for_outputs(dataset, NUMBER_OUTPUT_IMAGE_SAMPLES, SEED)
    panels_written = save_reconstruction_panels(
        model, panel_samples, CHECKPOINT_DIR, (nrow, ncol),
        dataset_label=dataset_label, filenames=panel_filenames,
        show_error=SHOW_ERROR_PLOTS, epochs=EPOCHS_TOTAL,
        latent_dim=LATENT_DIM, channels=CHANNELS
    )
    print(f"  Saved {panels_written} JPEG panel(s)")
    
    # Save reconstruction data
    sample_data = {
        'originals': data_np,
        'reconstructions': improved_recon.squeeze().cpu().numpy(),
        'filenames': panel_filenames
    }
    savemat(os.path.join(CHECKPOINT_DIR, 'reconstruction_data.mat'), sample_data)
    print(f"  Saved: reconstruction_data.mat")
    
    # Save timing log
    timing_log = [
        f"v10 TRAINING RESUMED AND COMPLETED",
        f"=" * 70,
        f"Resumed from: epoch 60",
        f"Completed: epochs 61-100",
        f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Configuration:",
        f"  Dataset: {dataset_label}",
        f"  Files: {len(dataset)}",
        f"  Epochs: {EPOCHS_TOTAL}",
        f"  Learning rate: {LR}",
        f"  Batch size: {BATCH_SIZE}",
        f"  Latent dim: {LATENT_DIM}",
        f"  Channels: {CHANNELS}",
        f"  Seed: {SEED}",
        f"",
        f"Results:",
        f"  Final loss: {losses[-1]:.6f}",
        f"  Model saved: autoencoder_clean.pth",
    ]
    
    with open(os.path.join(CHECKPOINT_DIR, 'timing_log.txt'), 'w') as f:
        f.write('\n'.join(timing_log))
    print(f"  Saved: timing_log.txt")
    
    print("\n" + "="*70)
    print("TRAINING RESUMED AND COMPLETED!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Output directory: {CHECKPOINT_DIR}")
    print("="*70)


if __name__ == "__main__":
    resume_training()
