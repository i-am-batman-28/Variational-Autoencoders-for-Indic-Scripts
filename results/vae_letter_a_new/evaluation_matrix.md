# Evaluation Metrics Matrix (Elution Matrix Interpretation)

This table interprets the request for an "elution matrix" as a compact evaluation metrics matrix for VAE quality.

| Quality Dimension | Metric | Value | Interpretation Goal |
|---|---:|---:|---|
| Reconstruction fidelity | MSE | 0.002584 | Lower is better |
| Reconstruction fidelity | PSNR (dB) | 26.9188 | Higher is better |
| Structural similarity | SSIM | 0.9834 | Higher is better |
| Probabilistic fit | BCE | 223.6607 | Lower is better |
| Latent regularization | KL divergence | 6.1210 | Balanced with BCE |
| Total objective | ELBO surrogate | 229.7817 | Lower is better |

## Extensions for larger datasets
- Distribution-level metrics: FID, KID
- Fidelity-coverage tradeoff: precision and recall for generative models