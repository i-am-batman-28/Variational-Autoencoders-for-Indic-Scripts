# Evaluation Matrix — Experiment 1 (Single Font, Single Letter)

## Global Metrics

| Metric | Value | Interpretation |
|--------|------:|----------------|
| MSE | 0.004107 | Lower is better |
| PSNR (dB) | 24.77 | Higher is better (>20 good, >30 excellent) |
| SSIM | 0.9751 | Higher is better (>0.8 good, >0.9 excellent) |
| BCE | 644.8932 | Lower is better |
| KL Divergence | 42.5707 | Balanced with BCE |
| ELBO | 687.4639 | Lower is better |

## Per-Class Metrics

| Point Size | MSE | PSNR (dB) | SSIM | Count |
|-----------|------:|----------:|-----:|------:|
| 10pt | 0.003694 | 24.61 | 0.9772 | 3 |
| 14pt | 0.007218 | 22.93 | 0.9568 | 3 |
| 18pt | 0.002264 | 26.47 | 0.9862 | 3 |
| 22pt | 0.003250 | 25.07 | 0.9803 | 3 |