
Worklog for preprocessing GSI data for deep learning.


TensorBoard diagnostics (autoencoder training)
----------------------------------------------
The script `Pytorch_scripts/Bowhead_Train_Autoencoder.py` now logs training diagnostics to TensorBoard.

What gets logged per dataset run
- Scalars: per-epoch training and validation loss
- Images: a grid of original vs reconstructed spectrograms (top row: originals, bottom row: reconstructions)
- Histograms: distribution of absolute pixel-wise reconstruction errors
- Embeddings: latent vectors for interactive visualization in TensorBoard Projector (PCA/t-SNE in the UI)

Where logs are written
- A timestamped run directory under `runs/`, e.g. `runs/Unsupervised_images.dir_YYYYMMDD-HHMMSS/`

Quick start (macOS, zsh)
1) Activate your Python environment
	- If you created the venv in the repo root:
	  source .venv_py31018/bin/activate
	- Or if you created it under Pytorch_scripts:
	  source Pytorch_scripts/.venv_py31018/bin/activate

2) Ensure TensorBoard is installed (one time)
	pip install tensorboard

3) Train to produce logs and artifacts
	cd Pytorch_scripts
	python Bowhead_Train_Autoencoder.py

4) Launch TensorBoard from the repo root (or anywhere above `runs/`)
	tensorboard --logdir runs

5) Open the printed URL in your browser (e.g., http://localhost:6006)

What to explore in TensorBoard
- Scalars tab: curves for `Loss/train` and `Loss/val` across epochs
- Images tab: side-by-side grids of original and reconstructed spectrograms
- Histograms tab: distribution of reconstruction errors for a sample of validation batches
- Projector tab: interactive 2D embedding of latent vectors; choose PCA or t-SNE in the UI

Notes
- If the default port is busy, add `--port 6007` (or any free port).
- On CPU or Apple MPS, `pin_memory` is disabled to avoid warnings; on CUDA it is enabled automatically.
- Large datasets can make training slow on CPU; reduce `batch_size` or `num_epochs` in the script to iterate faster.

