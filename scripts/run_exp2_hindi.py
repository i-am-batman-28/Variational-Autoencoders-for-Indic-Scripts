#!/usr/bin/env python3
"""
Experiment 2 Hindi: Single Font (akshar), All 13 Vowels, 4 Point Sizes
"""
from pathlib import Path
from vae_experiment_core import ExperimentConfig, run_experiment

ALL_VOWELS = ("a", "aa", "i", "ii", "u", "uu", "ri", "e", "ai", "o", "au", "am", "aha")

cfg = ExperimentConfig(
    experiment_name="Exp2 Hindi: Single Font, Multi Letter (akshar, 13 vowels)",
    dataset_root=Path("dataset"),
    font_names=("akshar",),
    letter_names=ALL_VOWELS,
    point_sizes=("10pt", "14pt", "18pt", "22pt"),
    image_size=128,
    batch_size=16,
    latent_dim=8,
    lr=1e-3,
    epochs=200,
    early_stopping_patience=30,
    beta_max=0.5,
    kl_warmup_epochs=30,
    base_depth=4,
    depth_variants=(5,),
    classification_targets=[("letter", 13), ("point_size", 4)],
    output_dir=Path("results/exp2_hindi_single_font_multi_letter"),
    seed=42,
)

if __name__ == "__main__":
    run_experiment(cfg)
