#!/usr/bin/env python3
"""
Experiment 4 Telugu: Multi Font (lohit+pothana), All 13 Vowels, 4 Point Sizes (Full Scale)
"""
from pathlib import Path
from vae_experiment_core import ExperimentConfig, run_experiment

ALL_VOWELS = ("a", "aa", "i", "ii", "u", "uu", "ri", "e", "ai", "o", "au", "am", "aha")

cfg = ExperimentConfig(
    experiment_name="Exp4 Telugu: Multi Font, Multi Letter (full scale)",
    dataset_root=Path("dataset_Rohit"),
    font_names=("lohit_telugu", "pothana2000"),
    letter_names=ALL_VOWELS,
    point_sizes=("10pt", "14pt", "18pt", "22pt"),
    image_size=128,
    batch_size=32,
    latent_dim=16,
    lr=1e-3,
    epochs=250,
    early_stopping_patience=35,
    beta_max=0.5,
    kl_warmup_epochs=40,
    base_depth=4,
    depth_variants=(5,),
    classification_targets=[("font", 2), ("letter", 13), ("point_size", 4)],
    output_dir=Path("results/exp4_telugu_multi_font_multi_letter"),
    seed=42,
)

if __name__ == "__main__":
    run_experiment(cfg)
