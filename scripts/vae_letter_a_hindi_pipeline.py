#!/usr/bin/env python3
"""
Wrapper to run the existing VAE letter-'a' pipeline on the HINDI dataset.

Uses production-level Hindi-specific settings for sharper reconstructions:
  - SOURCE_FONTS: akshar, mangal
  - dataset_root: dataset
  - output_dir: results/vae_letter_a_hindi
  - image_size: 128 (higher resolution for heavier Devanagari glyph)
  - latent_dim: 8 (more capacity than 2-D)
  - beta_max: 0.5, kl_warmup_epochs: 40, patience: 40, epochs: 160

Produces same artifacts as Telugu: quality report, checkpoint, curves,
latent plot, reconstructions, traversal, random samples, metrics, evaluation matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import vae_letter_a_pipeline as base


def main() -> None:
    # Override source fonts and keep letter/sizes the same
    # Restrict to Hindi fonts
    base.SOURCE_FONTS = ("akshar", "mangal")
    base.TARGET_LETTER = "a"
    base.TARGET_POINT_SIZES = ("10pt", "14pt", "18pt", "22pt")

    # Hindi-specific: higher resolution, more latent capacity, gentler KL, longer training
    argv = [
        sys.argv[0],
        "--dataset-root",
        str(Path("dataset")),
        "--output-dir",
        str(Path("results/vae_letter_a_hindi")),
        "--image-size",
        "128",
        "--batch-size",
        "16",
        "--latent-dim",          # higher latent capacity for heavier Hindi glyph
        "8",
        "--epochs",              # allow more training epochs
        "160",
        "--lr",
        "0.001",
        "--beta-max",            # slightly lower KL weight to reduce blur
        "0.5",
        "--kl-warmup-epochs",    # longer warmup so recon learns first
        "40",
        "--patience",            # more patience before early stopping
        "40",
        "--seed",
        "42",
    ]

    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()

