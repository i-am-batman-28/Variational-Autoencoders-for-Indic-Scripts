# VAE Experiment Plan for Letter a (dataset_Rohit)

## Objective
Build and evaluate a Variational Autoencoder (VAE) using only:
- dataset_Rohit/lohit_telugu/a
- dataset_Rohit/pothana2000/a

Target outcomes:
1. Dataset quality audit and feature summary.
2. Train a robust convolutional VAE for glyph generation.
3. Generate high-quality samples by decoding points from latent space.
4. Plot latent space in 2D for the letter a.
5. Produce an evaluation metrics matrix (interpreting the user request for "elution matrix").

## Scope and Constraints
- Two fonts only: lohit_telugu and pothana2000.
- One class only: letter a.
- Small-data regime (expected 160 samples total), so augmentation and regularization are necessary.
- Reproducibility required (fixed random seeds, saved config, saved outputs).

## Technical Plan

### Phase 1: Data Quality Audit
- Walk all images under the two target subsets.
- Validate image readability and count corrupt files.
- Compute dataset features:
  - Per-source and per-size counts.
  - Resolution and mode distribution.
  - Mean and standard deviation of pixel intensity.
  - Ink coverage ratio.
  - Sharpness estimate (Laplacian variance).
  - Duplicate detection via content hash.
  - Glyph center of mass (alignment consistency).
- Save JSON report and markdown summary.

### Phase 2: Modeling Strategy
- Use a convolutional VAE with:
  - Encoder: strided conv blocks.
  - Latent bottleneck: 2 dimensions (for direct 2D plotting).
  - Decoder: transposed conv blocks with sigmoid output.
- Loss:
  - Reconstruction loss (binary cross-entropy).
  - KL divergence with warm-up schedule (beta annealing).

### Phase 3: Training Protocol
- Stratified split by source and point size: 70/15/15 train/val/test.
- Lightweight augmentation on training split:
  - Small rotation.
  - Small translation.
  - Contrast jitter.
- Early stopping on validation ELBO.
- Save best checkpoint.

### Phase 4: Evaluation and Visualization
- Reconstruction quality:
  - BCE, MSE, PSNR, SSIM.
- Generation quality:
  - Decode a 2D grid in latent space and export image grid.
  - Decode random latent samples and export samples.
- Latent analysis:
  - Scatter plot of posterior means (mu) in 2D.
  - Color by source font and marker by point size.

### Phase 5: Evaluation Metrics Matrix ("Elution Matrix" Interpretation)
- Provide a compact matrix with categories:
  - Reconstruction fidelity: MSE, PSNR, SSIM.
  - Probabilistic quality: KL divergence, ELBO components.
  - Distribution quality (optional extension for larger data): FID/KID.
  - Latent separability signal: source-wise latent centroid distance.

## Deliverables
- scripts/vae_letter_a_pipeline.py
- docs/vae_research_sources.md
- results/vae_letter_a/dataset_quality_report.json
- results/vae_letter_a/dataset_quality_summary.md
- results/vae_letter_a/metrics_summary.json
- results/vae_letter_a/evaluation_matrix.md
- results/vae_letter_a/loss_curve.png
- results/vae_letter_a/latent_space_2d.png
- results/vae_letter_a/latent_traversal_grid.png
- results/vae_letter_a/random_samples.png
- results/vae_letter_a/reconstructions.png

## Risk Handling
- Overfitting due to small dataset:
  - Use augmentation, KL warm-up, and early stopping.
- Unstable training:
  - Gradient clipping, moderate learning rate, checkpoint best model.
- Ambiguous term "elution matrix":
  - Deliver an evaluation metrics matrix and clearly label it as interpretation.
