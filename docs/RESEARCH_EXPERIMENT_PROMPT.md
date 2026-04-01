# Research Experiment Prompt: VAE & Conditional VAE for Indic Scripts

## Context

This is a Bachelor's Thesis Project (BTP) on **Variational Autoencoders for Indic Scripts** (Telugu & Hindi). The goal is to produce publishable results by systematically training, evaluating, and scaling VAE and Conditional VAE models across fonts and characters.

**Existing codebase:** `scripts/vae_letter_a_pipeline.py` (1,159 lines) — a complete ConvVAE pipeline with dataset loading, training (beta-VAE + KL warmup), evaluation (MSE, PSNR, SSIM), and visualization (latent space, traversals, reconstructions, random samples).

**Datasets available:**
- `dataset_Rohit/` — Telugu: fonts `lohit_telugu`, `pothana2000` | 13 vowels | 4 sizes (10pt, 14pt, 18pt, 22pt) | 20 samples each | 128x128 PNG
- `dataset/` — Hindi: fonts `akshar`, `mangal` | 13 vowels | 4 sizes | 20 samples each | 128x128 PNG

---

## Instructions

Read the complete codebase (`scripts/`, root-level `.py` files, `docs/`, `results/`) and all documentation before proceeding. Understand the existing architecture, training loop, evaluation metrics, and output artifacts.

---

## Part A: Standard VAE Experiments

### Experiment 1 — Single Font, Single Letter (Baseline)

**Data:** `lohit_telugu` font, letter `a` only, 4 point-size classes (10pt, 14pt, 18pt, 22pt).

**Pipeline steps (execute in order, save every artifact):**

1. **Dataset audit** — Run quality checks. Log image counts per class, pixel statistics, sharpness, duplicates. Save as `dataset_quality_report.json`.

2. **Train VAE** — Train a ConvVAE with 2D latent space. Use beta-VAE with KL warmup. Apply batch normalization throughout encoder and decoder. Log per-epoch metrics (train loss, val loss, reconstruction loss, KL divergence, beta schedule) to `training_history.csv`.

3. **Plot all training curves** — Loss curves (train/val ELBO, reconstruction, KLD separately), beta schedule, learning rate schedule. Save as individual PNGs.

4. **2D latent space visualization** — Scatter plot of all samples colored and shaped by point-size class (10pt, 14pt, 18pt, 22pt). Verify visual class separation. Save as `latent_space_2d.png`.

5. **Reconstruction quality** — Plot original vs. reconstructed grids. Compute per-class metrics: MSE, PSNR (dB), SSIM. Save evaluation matrix as `evaluation_matrix.md` and `metrics_summary.json`.

6. **Generate samples from latent space** — Sample from the prior N(0, I). Generate a grid of samples. Visually and quantitatively assess quality. Save as `random_samples.png`.

7. **Latent traversal** — Sweep each latent dimension independently (z ∈ [-3, +3]) while holding others fixed. Plot grid. Save as `latent_traversal_grid.png`.

8. **Gaussian continuity check** — Fit a Gaussian to the encoded latent distribution. Plot the fitted Gaussian contour overlaid on the latent scatter. Compute a normality test (Shapiro-Wilk or KS test per dimension). Log p-values. Save plot as `latent_gaussian_fit.png`.

9. **Detach encoder, sample from learned distribution** — Freeze the encoder. Sample vectors from the fitted Gaussian parameters (not just standard normal). Decode them. Compare quality vs. standard prior samples. Save as `gaussian_samples.png`.

10. **Auxiliary classifier on latent space** — Train a small MLP classifier that takes a latent vector z and predicts the point-size class (10pt/14pt/18pt/22pt). Report classification accuracy, confusion matrix, per-class precision/recall. This validates whether the latent space encodes meaningful class structure. Save as `auxiliary_classifier_report.json` and `confusion_matrix.png`.

11. **Progressive depth scaling** — Incrementally increase encoder/decoder depth (e.g., 4 → 5 → 6 conv blocks). For each depth:
    - Retrain from scratch with the same hyperparameters.
    - Log all metrics and compare against the baseline.
    - Save a comparison table (`depth_scaling_comparison.md`).
    - Stop increasing depth when validation loss stops improving or auxiliary classifier accuracy saturates.

12. **Experiment summary** — Compile all observations, metrics, plots, and conclusions into `experiment_1_summary.md`.

**Output directory:** `results/exp1_single_font_single_letter/`

---

### Experiment 2 — Single Font, Multiple Letters

**Data:** `lohit_telugu` font, all 13 vowels (a, aa, i, ii, u, uu, ri, e, ai, o, au, am, aha), 4 point sizes each.

**Repeat the full pipeline from Experiment 1, with these additions:**

- Latent space should now show separation by **both letter identity and point size**. Use color for letter, marker shape/size for point size.
- Auxiliary classifier should predict **both letter class (13-way) and point-size class (4-way)**. Report both accuracies.
- Increase latent dimensionality as needed (start with 8D, use t-SNE/UMAP for visualization if dim > 2).
- Analyze whether visually similar characters (e.g., similar stroke patterns) cluster together in latent space. Document observations.

**Output directory:** `results/exp2_single_font_multi_letter/`

---

### Experiment 3 — Multiple Fonts, Single Letter

**Data:** Both `lohit_telugu` and `pothana2000` fonts, letter `a` only, 4 point sizes.

**Repeat the full pipeline from Experiment 1, with these additions:**

- Latent space visualization should show separation by **font** and **point size**. Use color for font, marker for size.
- Auxiliary classifier should predict **font (2-way) and point-size class (4-way)**.
- Analyze font-specific latent clusters: do the two fonts occupy distinct regions? Is there meaningful structure (e.g., interpolation between fonts produces plausible glyphs)?
- Generate interpolation samples between font clusters in latent space. Save as `font_interpolation.png`.

**Output directory:** `results/exp3_multi_font_single_letter/`

---

### Experiment 4 — Multiple Fonts, Multiple Letters (Full Scale)

**Data:** Both `lohit_telugu` and `pothana2000`, all 13 vowels, 4 point sizes.

**Repeat the full pipeline, with these additions:**

- Latent dimensionality: start with 16D or higher. Use t-SNE/UMAP for 2D projections.
- Auxiliary classifier predicts **font (2-way), letter (13-way), and point size (4-way)** — multi-head output.
- Analyze hierarchical latent structure: does the model learn a font axis vs. a character axis vs. a size axis?
- Compute inter-class and intra-class distances in latent space. Save as `latent_distance_analysis.json`.
- Full progressive depth scaling with comparison.

**Output directory:** `results/exp4_multi_font_multi_letter/`

---

## Part B: Conditional VAE (CVAE) Experiments

**After completing Part A**, implement a Conditional VAE that conditions on class labels (font, letter, point size) via label embedding concatenated to both encoder input and decoder input.

### Repeat all four experiments (1-4) using CVAE:

- **CVAE Experiment 1:** Single font, single letter, conditioned on point size.
- **CVAE Experiment 2:** Single font, multiple letters, conditioned on letter + point size.
- **CVAE Experiment 3:** Multiple fonts, single letter, conditioned on font + point size.
- **CVAE Experiment 4:** Multiple fonts, multiple letters, conditioned on font + letter + point size.

**For each CVAE experiment, additionally:**

- Generate class-conditional samples (e.g., "generate letter 'a' at 18pt in lohit_telugu style"). Validate via auxiliary classifier.
- Compare reconstruction quality and latent space structure against the corresponding standard VAE experiment.
- Save a side-by-side comparison document: `vae_vs_cvae_comparison.md`.

**Output directories:** `results/cvae_exp1/`, `results/cvae_exp2/`, `results/cvae_exp3/`, `results/cvae_exp4/`

---

## Global Requirements

### Normalization
- Apply **batch normalization** in all encoder and decoder conv layers.
- Normalize input images to [0, 1].
- Experiment with **layer normalization** or **instance normalization** if batch norm causes instability with small batches. Log which normalization works best.

### Logging (Non-Negotiable)
Every experiment must save:
- `training_history.csv` — per-epoch: train_loss, val_loss, recon_loss, kl_loss, beta, lr
- `metrics_summary.json` — final test metrics: MSE, PSNR, SSIM, BCE, KLD, ELBO
- `dataset_quality_report.json` — dataset audit results
- `evaluation_matrix.md` — human-readable quality assessment
- `experiment_config.json` — all hyperparameters, architecture details, data splits
- All plots as individual high-resolution PNGs (300 DPI)
- Model checkpoints: `best_model.pt` (best val loss) and `final_model.pt`
- Console/training logs: `training.log` (text file with timestamps)

### Directory Structure
```
results/
├── exp1_single_font_single_letter/
│   ├── model/
│   ├── plots/
│   ├── logs/
│   └── reports/
├── exp2_single_font_multi_letter/
│   ├── ...
├── exp3_multi_font_single_letter/
│   ├── ...
├── exp4_multi_font_multi_letter/
│   ├── ...
├── cvae_exp1/
│   ├── ...
├── cvae_exp2/
│   ├── ...
├── cvae_exp3/
│   ├── ...
├── cvae_exp4/
│   ├── ...
└── final_comparison/
    ├── vae_vs_cvae_comparison.md
    ├── cross_experiment_metrics.csv
    └── publication_figures/
```

### Research Rigor
- **Reproducibility:** Fix random seeds. Log all seeds. Save exact configs.
- **Statistical validity:** Run each key experiment with 3 different seeds if time permits. Report mean ± std.
- **Ablation tracking:** When changing one variable (depth, latent dim, normalization), keep everything else constant. Document in a comparison table.
- **Publication readiness:** Generate clean, labeled, publication-quality figures (proper axis labels, legends, titles, consistent color schemes across experiments).

### Execution Order
1. Plan the full pipeline architecture before writing any code.
2. Implement Experiment 1 end-to-end first. Validate every output.
3. Generalize the pipeline to support Experiments 2-4 via configuration.
4. Run Experiments 2, 3, 4 sequentially.
5. Implement CVAE architecture. Run CVAE Experiments 1-4.
6. Compile cross-experiment comparison and final analysis.

### Resources to Consult
- Existing literature analyses in project root: `PAPER_1_ANALYSIS_*.txt` through `PAPER_4_ANALYSIS_*.txt`
- `LITERATURE_REVIEW.txt` for methodology context
- `docs/BTP_Project_Documentation.md` for project scope
- `docs/vae_letter_a_experiment_plan.md` for prior experiment design
- Kingma & Welling (2013) — VAE fundamentals
- Higgins et al. (2017) — beta-VAE and disentangled representations
- Sohn et al. (2015) — Conditional VAE (CVAE) formulation
