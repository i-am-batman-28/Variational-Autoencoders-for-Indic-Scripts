# VAE Research Sources for This Experiment

This list focuses on VAE theory, practical training stability, latent-space analysis, and generation evaluation.

## Core VAE Theory

1. Auto-Encoding Variational Bayes (Kingma, Welling, 2013)
- Why it matters: Original VAE formulation, ELBO objective, reparameterization trick.
- URL: https://arxiv.org/abs/1312.6114

2. An Introduction to Variational Autoencoders (Kingma, Welling, 2019)
- Why it matters: Comprehensive modern tutorial and conceptual grounding.
- URL: https://arxiv.org/abs/1906.02691

3. Tutorial on Variational Autoencoders (Doersch, 2016)
- Why it matters: Intuitive derivation and implementation-friendly explanation.
- URL: https://arxiv.org/abs/1606.05908

## Better Latent Structure and KL Control

4. Understanding disentangling in beta-VAE (Burgess et al., 2018)
- Why it matters: Capacity scheduling and practical KL annealing guidance.
- URL: https://arxiv.org/abs/1804.03599

5. Isolating Sources of Disentanglement in VAEs (beta-TCVAE) (Chen et al., 2018)
- Why it matters: ELBO decomposition and latent-factor quality analysis.
- URL: https://arxiv.org/abs/1802.04942

6. A Framework for the Quantitative Evaluation of Disentangled Representations (Eastwood, Williams, 2018)
- Why it matters: Disentanglement, completeness, informativeness metrics.
- URL: https://openreview.net/forum?id=By-7dz-AZ

## Image Generation Quality Metrics

7. GANs Trained by a Two Time-Scale Update Rule (Heusel et al., 2017)
- Why it matters: Introduces Fréchet Inception Distance (FID).
- URL: https://arxiv.org/abs/1706.08500

8. Demystifying MMD GANs (Bińkowski et al., 2018)
- Why it matters: Kernel Inception Distance (KID), often robust for limited samples.
- URL: https://arxiv.org/abs/1801.01401

9. Precision and Recall for Distributions (Sajjadi et al., 2018)
- Why it matters: Separates fidelity from coverage in generative evaluation.
- URL: https://arxiv.org/abs/1806.00035

## Reconstruction and Perceptual Similarity

10. Image Quality Assessment: From Error Visibility to Structural Similarity (Wang et al., 2004)
- Why it matters: SSIM definition used for reconstruction quality.
- URL: https://ieeexplore.ieee.org/document/1284395

11. Peak Signal-to-Noise Ratio (PSNR) reference usage in image reconstruction literature
- Why it matters: Standard reconstruction quality scalar used alongside MSE/SSIM.
- URL: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

## Latent Space Visualization

12. Visualizing Data using t-SNE (van der Maaten, Hinton, 2008)
- Why it matters: Standard 2D embedding approach when latent dimension > 2.
- URL: https://jmlr.org/papers/v9/vandermaaten08a.html

## Interpreting "Elution Matrix"

The term "elution matrix" is not standard in VAE evaluation for image generation. In this project, it is interpreted as an Evaluation Metrics Matrix, summarizing quality dimensions in a single table:
- Reconstruction: MSE, PSNR, SSIM.
- Probabilistic objective: BCE, KL, ELBO.
- Distribution quality (extension): FID, KID, precision-recall.
- Latent geometry: centroid distance, overlap indicators, traversal smoothness.

Alternative likely terms in related workflows:
- Confusion matrix (for supervised classification tasks).
- Distance matrix (pairwise latent distances).
- Correlation matrix (latent-feature relationships).
