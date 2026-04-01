# Experiment 1 Summary — Single Font, Single Letter

**Date:** 2026-04-01 22:40:57
**Font:** lohit_telugu
**Letter:** a
**Classes:** 10pt, 14pt, 18pt, 22pt

## 1. Dataset
- Total samples: 80
- Corrupt files: 0
- Duplicate groups: 0

## 2. Training Configuration
- Image size: 128
- Latent dim: 2
- Epochs (max): 150
- Beta max: 1.0
- KL warmup: 25 epochs
- Early stopping patience: 25

## 3. Baseline Model (depth=4) — Test Metrics
- MSE: 0.004107
- PSNR: 24.77 dB
- SSIM: 0.9751

## 4. Gaussian Continuity Analysis

- dim_0: Shapiro p=0.0029 → Deviates from normal
- dim_1: Shapiro p=0.0000 → Deviates from normal

## 5. Auxiliary Classifier (Point-Size Prediction)
- Test accuracy: 0.2500
- Best val accuracy: 0.2500

  - 10pt: P=0.25 R=1.00 F1=0.40
  - 14pt: P=0.00 R=0.00 F1=0.00
  - 18pt: P=0.00 R=0.00 F1=0.00
  - 22pt: P=0.00 R=0.00 F1=0.00

## 6. Progressive Depth Scaling

| Depth | MSE | PSNR | SSIM | Aux Accuracy |
|------:|------:|-----:|-----:|-------------:|
| 4 | 0.004107 | 24.77 | 0.9751 | 0.2500 |
| 5 | 0.003379 | 26.03 | 0.9787 | 0.2500 |
| 6 | 0.004798 | 24.09 | 0.9702 | 0.2500 |

## 7. Observations

_To be filled based on visual inspection of plots and numerical results._

---
*Generated automatically by exp1_single_font_single_letter.py*