# Evaluation Metrics Matrix (Elution Matrix Interpretation)

This table interprets the request for an "elution matrix" as a compact evaluation metrics matrix for VAE quality.

| Quality Dimension | Metric | Value | Interpretation Goal |
|---|---:|---:|---|
| Reconstruction fidelity | MSE | 0.003055 | Lower is better |
| Reconstruction fidelity | PSNR (dB) | 26.0296 | Higher is better |
| Structural similarity | SSIM | 0.9803 | Higher is better |
| Probabilistic fit | BCE | 227.0696 | Lower is better |
| Latent regularization | KL divergence | 7.0931 | Balanced with BCE |
| Total objective | ELBO surrogate | 234.1627 | Lower is better |

## Extensions for larger datasets
- Distribution-level metrics: FID, KID
- Fidelity-coverage tradeoff: precision and recall for generative models