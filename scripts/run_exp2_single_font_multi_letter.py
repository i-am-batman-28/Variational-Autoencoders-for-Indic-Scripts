#!/usr/bin/env python3
"""
Experiment 2: Single Font, Multiple Letters
============================================
Data:  lohit_telugu / all 13 vowels / 4 point sizes
Goal:  Test if VAE can learn representations for multiple characters from one font.
"""
from pathlib import Path
from vae_experiment_core import ExperimentConfig, run_experiment

ALL_VOWELS = ("a", "aa", "i", "ii", "u", "uu", "ri", "e", "ai", "o", "au", "am", "aha")

cfg = ExperimentConfig(
    experiment_name="Exp2: Single Font, Multi Letter (lohit_telugu, 13 vowels)",
    dataset_root=Path("dataset_Rohit"),
    font_names=("lohit_telugu",),
    letter_names=ALL_VOWELS,
    point_sizes=("10pt", "14pt", "18pt", "22pt"),
    image_size=128,
    batch_size=16,
    latent_dim=8,          # higher dim for 13 chars
    lr=1e-3,
    epochs=200,
    early_stopping_patience=30,
    beta_max=0.5,          # lighter KL for multi-class
    kl_warmup_epochs=30,
    base_depth=4,
    depth_variants=(5,),   # skip depth 6 for speed
    classification_targets=[("letter", 13), ("point_size", 4)],
    output_dir=Path("results/exp2_single_font_multi_letter"),
    seed=42,
)

if __name__ == "__main__":
    run_experiment(cfg)
