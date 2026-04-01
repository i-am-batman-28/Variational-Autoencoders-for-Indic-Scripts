#!/usr/bin/env python3
"""
Experiment 3: Multiple Fonts, Single Letter
============================================
Data:  lohit_telugu + pothana2000 / letter 'a' / 4 point sizes
Goal:  Test if VAE learns font-specific representations for the same character.
"""
from pathlib import Path
from vae_experiment_core import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="Exp3: Multi Font, Single Letter (lohit+pothana, letter a)",
    dataset_root=Path("dataset_Rohit"),
    font_names=("lohit_telugu", "pothana2000"),
    letter_names=("a",),
    point_sizes=("10pt", "14pt", "18pt", "22pt"),
    image_size=128,
    batch_size=16,
    latent_dim=2,          # 2D for clear font separation visualisation
    lr=1e-3,
    epochs=150,
    early_stopping_patience=25,
    beta_max=1.0,
    kl_warmup_epochs=25,
    base_depth=4,
    depth_variants=(5,),
    classification_targets=[("font", 2), ("point_size", 4)],
    output_dir=Path("results/exp3_multi_font_single_letter"),
    seed=42,
)

if __name__ == "__main__":
    run_experiment(cfg)
