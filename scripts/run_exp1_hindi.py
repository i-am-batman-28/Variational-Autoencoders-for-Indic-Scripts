#!/usr/bin/env python3
"""
Experiment 1 Hindi: Single Font (akshar), Single Letter (a), 4 Point Sizes
"""
from pathlib import Path
from vae_experiment_core import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="Exp1 Hindi: Single Font, Single Letter (akshar, letter a)",
    dataset_root=Path("dataset"),
    font_names=("akshar",),
    letter_names=("a",),
    point_sizes=("10pt", "14pt", "18pt", "22pt"),
    image_size=128,
    batch_size=16,
    latent_dim=2,
    lr=1e-3,
    epochs=150,
    early_stopping_patience=25,
    beta_max=1.0,
    kl_warmup_epochs=25,
    base_depth=4,
    depth_variants=(5, 6),
    classification_targets=[("point_size", 4)],
    output_dir=Path("results/exp1_hindi_single_font_single_letter"),
    seed=42,
)

if __name__ == "__main__":
    run_experiment(cfg)
