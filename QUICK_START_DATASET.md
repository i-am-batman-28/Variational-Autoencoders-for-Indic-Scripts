# Quick Start: Production-Grade Dataset Generation

## Summary of Plan

### Sample Count
- **50 samples per alphabet per font** (recommended production-grade)
- **12 alphabets × 2 fonts = 1,200 total images**
- Can adjust: 30 (minimum) to 100 (optimal)

### Variations Strategy
1. **Geometric**: Rotation (±3°), Translation (±2px), Scaling (95-105%)
2. **Noise**: Light Gaussian noise (σ=0.015)
3. **Quality**: Slight blur, contrast adjustments
4. **Rendering**: Different DPI (72, 96, 150)

### Font Sizes
- **12pt**: 60% (30 samples) - Primary requirement
- **10pt**: 20% (10 samples) - Smaller variation
- **14pt**: 20% (10 samples) - Larger variation

### Output
- **128x128 pixels** (final size)
- **PNG format** (high quality)
- **Rendered at 300 DPI**, then resized

## Quick Commands

```bash
# Install dependencies
venv/bin/pip install -r requirements.txt

# Test fonts first
venv/bin/python generate_printed_dataset.py

# Generate full dataset (will prompt for confirmation)
venv/bin/python generate_printed_dataset.py
```

## What You'll Get

```
dataset/
├── akshar_unicode/
│   ├── a/ (50 PNG images)
│   ├── aa/ (50 PNG images)
│   └── ... (12 folders)
├── mangal/
│   └── ... (same structure)
└── metadata.json (complete dataset info)
```

## Adjustments

To change sample count, edit `CONFIG['samples_per_alphabet']` in the script:
- 30 = minimum viable
- 50 = recommended (production-grade)
- 75-100 = optimal (if you want more)

## Quality Features

✅ All characters clearly readable
✅ Well-centered in images
✅ Consistent quality
✅ Diverse variations
✅ Production-grade standards
