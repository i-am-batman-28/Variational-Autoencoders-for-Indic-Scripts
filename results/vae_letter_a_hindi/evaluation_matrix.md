# Evaluation Metrics Matrix (Elution Matrix Interpretation)

This table interprets the request for an "elution matrix" as a compact evaluation metrics matrix for VAE quality.

| Quality Dimension | Metric | Value | Interpretation Goal |
|---|---:|---:|---|
| Reconstruction fidelity | MSE | 0.000737 | Lower is better |
| Reconstruction fidelity | PSNR (dB) | 32.1737 | Higher is better |
| Structural similarity | SSIM | 0.9966 | Higher is better |
| Probabilistic fit | BCE | 502.4212 | Lower is better |
| Latent regularization | KL divergence | 33.4085 | Balanced with BCE |
| Total objective | ELBO surrogate | 519.1255 | Lower is better |

## Extensions for larger datasets
- Distribution-level metrics: FID, KID
- Fidelity-coverage tradeoff: precision and recall for generative models