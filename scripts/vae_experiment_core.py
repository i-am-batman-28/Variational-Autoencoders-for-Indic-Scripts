#!/usr/bin/env python3
"""
VAE Experiment Core — Shared infrastructure for Experiments 1-4.
================================================================
This module contains all reusable components: ConvVAE model, training loop,
evaluation metrics, latent analysis, auxiliary classifier, visualization,
and report generation.

Individual experiment scripts (exp1, exp2, exp3, exp4) import from here
and configure the specific data scope and hyper-parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy import stats as sp_stats
from scipy.ndimage import laplace
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────
# COLOUR PALETTE (colour-blind safe — Wong 2011)
# ──────────────────────────────────────────────────────────────────────
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#000000",
    "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">", "p", "H"]

# Matplotlib publication defaults
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


# ──────────────────────────────────────────────────────────────────────
# §1  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    experiment_name: str = "experiment"
    dataset_root: Path = Path("dataset_Rohit")
    font_names: tuple[str, ...] = ("lohit_telugu",)
    letter_names: tuple[str, ...] = ("a",)
    point_sizes: tuple[str, ...] = ("10pt", "14pt", "18pt", "22pt")
    image_size: int = 128

    # Training
    batch_size: int = 16
    latent_dim: int = 2
    lr: float = 1e-3
    epochs: int = 150
    early_stopping_patience: int = 25
    beta_max: float = 1.0
    kl_warmup_epochs: int = 25
    grad_clip_norm: float = 5.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42

    # Architecture
    base_depth: int = 4
    depth_variants: tuple[int, ...] = (5, 6)

    # Auxiliary classifier
    aux_hidden: int = 64
    aux_lr: float = 1e-3
    aux_epochs: int = 200
    aux_patience: int = 30
    aux_dropout: float = 0.3

    # Classification targets:  list of (label_name, extractor_func_name)
    # e.g. [("point_size", 4)] means classify by point_size with 4 classes
    # Concrete class names are derived from the data
    classification_targets: list[tuple[str, int]] = field(default_factory=lambda: [("point_size", 4)])

    # Output
    output_dir: Path = Path("results/experiment")
    dpi: int = 300

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset_root"] = str(self.dataset_root)
        d["output_dir"] = str(self.output_dir)
        return d


@dataclass
class ImageRecord:
    path: Path
    font: str
    letter: str
    point_size: str


# ──────────────────────────────────────────────────────────────────────
# §2  LOGGING
# ──────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, name: str = "vae_exp") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # clear existing handlers
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────
# §3  DATA LOADING & AUGMENTATION
# ──────────────────────────────────────────────────────────────────────

def load_image_as_grayscale(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.copy()
    if "A" in img.getbands():
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("L")
    else:
        img = img.convert("L")
    return img.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def apply_augmentation(image: Image.Image) -> Image.Image:
    out = image
    if random.random() < 0.85:
        out = out.rotate(random.uniform(-8, 8), resample=Image.Resampling.BICUBIC, fillcolor=255)
    if random.random() < 0.85:
        tx, ty = random.uniform(-3, 3), random.uniform(-3, 3)
        out = out.transform(out.size, Image.Transform.AFFINE, (1, 0, tx, 0, 1, ty),
                            resample=Image.Resampling.BICUBIC, fillcolor=255)
    if random.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.85, 1.20))
    return out


def collect_records(cfg: ExperimentConfig) -> list[ImageRecord]:
    records = []
    for font in cfg.font_names:
        for letter in cfg.letter_names:
            for pt in cfg.point_sizes:
                d = cfg.dataset_root / font / letter / pt
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if f.is_file() and f.suffix.lower() == ".png":
                        records.append(ImageRecord(path=f, font=font, letter=letter, point_size=pt))
    return records


def stratified_split(
    records: list[ImageRecord], train_ratio: float, val_ratio: float, seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    groups: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[f"{rec.font}:{rec.letter}:{rec.point_size}"].append(rec)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for group in groups.values():
        g = group[:]; rng.shuffle(g)
        n = len(g)
        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2); n_val = 1
        train.extend(g[:n_train])
        val.extend(g[n_train:n_train + n_val])
        test.extend(g[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


class MultiLabelGlyphDataset(Dataset):
    """Dataset that returns image + dict of label indices for multiple targets."""

    def __init__(
        self, records: list[ImageRecord], image_size: int,
        label_maps: dict[str, dict[str, int]],  # e.g. {"point_size": {"10pt":0,...}, "font": {...}}
        augment: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.label_maps = label_maps

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        rec = self.records[idx]
        image = load_image_as_grayscale(rec.path, self.image_size)
        if self.augment:
            image = apply_augmentation(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        labels = {}
        for target_name, mapping in self.label_maps.items():
            value = getattr(rec, target_name)
            labels[target_name] = mapping[value]
        return tensor, labels


def _collate_multi(batch):
    """Custom collate for MultiLabelGlyphDataset."""
    images = torch.stack([b[0] for b in batch])
    label_keys = batch[0][1].keys()
    labels = {k: torch.tensor([b[1][k] for b in batch], dtype=torch.long) for k in label_keys}
    return images, labels


def build_label_maps(records: list[ImageRecord], targets: list[str]) -> dict[str, dict[str, int]]:
    """Build {target_name: {value: index}} from records."""
    maps = {}
    for t in targets:
        values = sorted(set(getattr(r, t) for r in records))
        maps[t] = {v: i for i, v in enumerate(values)}
    return maps


# ──────────────────────────────────────────────────────────────────────
# §4  CONV-VAE MODEL
# ──────────────────────────────────────────────────────────────────────

def _channel_sequence(depth: int) -> list[int]:
    if depth <= 4:
        return [32, 64, 128, 256]
    elif depth == 5:
        return [32, 64, 128, 256, 512]
    elif depth == 6:
        return [16, 32, 64, 128, 256, 512]
    else:
        base = [16, 32] + [64 * (2 ** i) for i in range(depth - 2)]
        return [min(c, 512) for c in base]


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 2, image_size: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.depth = depth
        self.encoder_spatial = image_size // (2 ** depth)
        channels = _channel_sequence(depth)
        self.last_ch = channels[-1]

        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        self.encoder_out_dim = self.last_ch * self.encoder_spatial ** 2
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.encoder_out_dim)
        dec_layers: list[nn.Module] = []
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            dec_layers += [
                nn.ConvTranspose2d(rev[i], rev[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(rev[i+1]), nn.ReLU(inplace=True),
            ]
        dec_layers += [nn.ConvTranspose2d(rev[-1], 1, 4, stride=2, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.fc_decode(z).view(-1, self.last_ch, self.encoder_spatial, self.encoder_spatial)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ──────────────────────────────────────────────────────────────────────
# §5  MULTI-HEAD AUXILIARY CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

class MultiHeadClassifier(nn.Module):
    """MLP with shared trunk and separate classification heads."""

    def __init__(self, latent_dim: int, head_sizes: dict[str, int],
                 hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, n_classes)
            for name, n_classes in head_sizes.items()
        })

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {name: head(h) for name, head in self.heads.items()}


# ──────────────────────────────────────────────────────────────────────
# §6  LOSS & TRAINING
# ──────────────────────────────────────────────────────────────────────

def vae_loss(recon, x, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = bce + beta * kld
    return total, {"bce": float(bce.detach()), "kld": float(kld.detach()), "total": float(total.detach())}


def get_beta(epoch, warmup, beta_max):
    return min(beta_max, beta_max * (epoch + 1) / warmup) if warmup > 0 else beta_max


def evaluate_epoch(model, loader, device, beta):
    model.eval()
    bce_v, kld_v, tot_v = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            _, s = vae_loss(recon, x, mu, logvar, beta)
            bce_v.append(s["bce"]); kld_v.append(s["kld"]); tot_v.append(s["total"])
    return {k: float(np.mean(v)) if v else math.nan
            for k, v in zip(("bce", "kld", "total"), (bce_v, kld_v, tot_v))}


def compute_ssim(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x, sigma_y = x.var(), y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2*mu_x*mu_y + c1) * (2*sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(num / den) if den != 0 else 0.0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_vae(model, train_loader, val_loader, cfg, device, model_dir, logger, log_prefix=""):
    model_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    history = {k: [] for k in
               ("train_bce", "train_kld", "train_total", "val_bce", "val_kld", "val_total", "beta", "lr")}
    best_val, best_epoch, bad = float("inf"), -1, 0

    for epoch in range(cfg.epochs):
        model.train()
        beta = get_beta(epoch, cfg.kl_warmup_epochs, cfg.beta_max)
        t_bce, t_kld, t_tot = [], [], []

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, s = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            t_bce.append(s["bce"]); t_kld.append(s["kld"]); t_tot.append(s["total"])

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        tr = {k: np.mean(v) for k, v in zip(("bce","kld","total"), (t_bce,t_kld,t_tot))}
        vl = evaluate_epoch(model, val_loader, device, beta)

        for k in ("bce", "kld", "total"):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(vl[k]))
        history["beta"].append(beta); history["lr"].append(cur_lr)

        logger.info(f"{log_prefix}Epoch {epoch+1:03d}/{cfg.epochs} | β={beta:.3f} | "
                     f"lr={cur_lr:.2e} | train={tr['total']:.2f} | val={vl['total']:.2f}")

        if vl["total"] < best_val:
            best_val, best_epoch, bad = vl["total"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                        "epoch": epoch, "best_val_total": best_val, "depth": model.depth},
                       model_dir / "best_model.pt")
        else:
            bad += 1

        if bad >= cfg.early_stopping_patience:
            logger.info(f"{log_prefix}Early stopping at epoch {epoch+1}, best={best_epoch+1}")
            break

    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                "epoch": epoch, "depth": model.depth}, model_dir / "final_model.pt")
    logger.info(f"{log_prefix}Training complete. Best epoch: {best_epoch+1}, val: {best_val:.4f}")
    return history


# ──────────────────────────────────────────────────────────────────────
# §7  EVALUATION
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device, beta, class_names_by_target, label_maps):
    """Full eval with per-class breakdown for each classification target."""
    model.eval()
    bce_v, kld_v, tot_v = [], [], []
    global_mse, global_psnr, global_ssim = [], [], []

    # per-target per-class metrics
    per_target: dict[str, dict[str, dict[str, list]]] = {}
    for target_name, names in class_names_by_target.items():
        per_target[target_name] = {n: {"mse": [], "psnr": [], "ssim": []} for n in names}

    # reverse maps: index -> name
    idx_to_name = {}
    for target_name, mapping in label_maps.items():
        idx_to_name[target_name] = {v: k for k, v in mapping.items()}

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            labels = batch[1]
            recon, mu, logvar = model(x)
            _, s = vae_loss(recon, x, mu, logvar, beta)
            bce_v.append(s["bce"]); kld_v.append(s["kld"]); tot_v.append(s["total"])

            x_np, r_np = x.cpu().numpy(), recon.cpu().numpy()
            for i in range(x_np.shape[0]):
                xi, ri = x_np[i, 0], r_np[i, 0]
                mse = float(np.mean((xi - ri) ** 2))
                psnr = 20.0 * math.log10(1.0 / math.sqrt(mse + 1e-12))
                ssim = compute_ssim(xi, ri)
                global_mse.append(mse); global_psnr.append(psnr); global_ssim.append(ssim)

                for target_name in class_names_by_target:
                    idx = int(labels[target_name][i])
                    name = idx_to_name[target_name][idx]
                    per_target[target_name][name]["mse"].append(mse)
                    per_target[target_name][name]["psnr"].append(psnr)
                    per_target[target_name][name]["ssim"].append(ssim)

    result = {
        "global": {
            "bce": float(np.mean(bce_v)), "kld": float(np.mean(kld_v)),
            "total": float(np.mean(tot_v)), "mse": float(np.mean(global_mse)),
            "psnr": float(np.mean(global_psnr)), "ssim": float(np.mean(global_ssim)),
        },
        "per_target": {},
    }
    for target_name, classes in per_target.items():
        result["per_target"][target_name] = {}
        for cls_name, vals in classes.items():
            result["per_target"][target_name][cls_name] = {
                "mse": float(np.mean(vals["mse"])) if vals["mse"] else math.nan,
                "psnr": float(np.mean(vals["psnr"])) if vals["psnr"] else math.nan,
                "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else math.nan,
                "count": len(vals["mse"]),
            }
    return result


# ──────────────────────────────────────────────────────────────────────
# §8  DATASET QUALITY AUDIT
# ──────────────────────────────────────────────────────────────────────

def dataset_quality_audit(records, image_size, cfg):
    corrupt, modes, resolutions = [], Counter(), Counter()
    hashes: Counter = Counter()
    hash_to_files: defaultdict[str, list[str]] = defaultdict(list)
    int_means, int_stds, nw_ratios, sharp_vals = [], [], [], []

    by_group_counts: Counter[str] = Counter()
    for rec in records:
        key = f"{rec.font}/{rec.letter}/{rec.point_size}"
        by_group_counts[key] += 1
        try:
            with Image.open(rec.path) as raw:
                modes[raw.mode] += 1; resolutions[f"{raw.width}x{raw.height}"] += 1
            img = load_image_as_grayscale(rec.path, image_size)
            arr = np.asarray(img, dtype=np.float32)
        except Exception:
            corrupt.append(str(rec.path)); continue
        h = hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()
        hashes[h] += 1; hash_to_files[h].append(str(rec.path))
        int_means.append(float(arr.mean()/255)); int_stds.append(float(arr.std()/255))
        nw_ratios.append(float((arr < 245).mean()))
        sharp_vals.append(float(np.var(laplace(arr))))

    def _s(v):
        if not v: return {"count": 0}
        a = np.array(v, dtype=np.float64)
        return {"count": len(v), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    dup_groups = [v for v in hash_to_files.values() if len(v) > 1]
    return {
        "scope": {"fonts": list(cfg.font_names), "letters": list(cfg.letter_names),
                  "point_sizes": list(cfg.point_sizes), "size": f"{image_size}x{image_size}"},
        "integrity": {"total": len(records), "corrupt": len(corrupt),
                      "modes": dict(modes), "resolutions": dict(resolutions),
                      "duplicates": len(dup_groups), "unique": sum(1 for c in hashes.values() if c==1)},
        "counts_by_group": dict(sorted(by_group_counts.items())),
        "pixel_stats": {"intensity_mean": _s(int_means), "non_white": _s(nw_ratios), "sharpness": _s(sharp_vals)},
        "duplicate_groups": dup_groups,
    }


# ──────────────────────────────────────────────────────────────────────
# §9  LATENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def collect_latents(model, loader, device):
    """Returns (mu [N, D], labels_dict {target: array[N]})."""
    model.eval()
    mus, all_labels = [], defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu().numpy())
            labels = batch[1]
            for k, v in labels.items():
                all_labels[k].append(v.numpy())
    return np.concatenate(mus), {k: np.concatenate(v) for k, v in all_labels.items()}


def fit_gaussian(latents):
    mean = latents.mean(axis=0)
    cov = np.cov(latents, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    normality = {}
    for d in range(latents.shape[1]):
        col = latents[:, d]
        if len(col) >= 3:
            stat, p = sp_stats.shapiro(col)
            normality[f"dim_{d}"] = {"shapiro_stat": float(stat), "p_value": float(p)}
    ks = {}
    for d in range(latents.shape[1]):
        col = latents[:, d]
        std = float(np.sqrt(cov[d, d])) if cov[d, d] > 0 else 1e-8
        stat, p = sp_stats.kstest(col, "norm", args=(float(mean[d]), std))
        ks[f"dim_{d}"] = {"ks_stat": float(stat), "p_value": float(p)}
    return {"mean": mean.tolist(), "covariance": cov.tolist(),
            "shapiro_wilk": normality, "ks_test": ks, "n_samples": int(latents.shape[0])}


# ──────────────────────────────────────────────────────────────────────
# §10  AUXILIARY CLASSIFIER TRAINING
# ──────────────────────────────────────────────────────────────────────

def train_auxiliary_classifier(
    train_z, train_labels, val_z, val_labels, test_z, test_labels,
    cfg, head_sizes, label_maps, device, logger,
):
    """Train multi-head classifier. Returns per-head metrics."""
    model = MultiHeadClassifier(
        latent_dim=cfg.latent_dim, head_sizes=head_sizes,
        hidden=cfg.aux_hidden, dropout=cfg.aux_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.aux_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def _to_tensors(z, labels):
        zt = torch.tensor(z, dtype=torch.float32, device=device)
        lt = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in labels.items()}
        return zt, lt

    tz, tl = _to_tensors(train_z, train_labels)
    vz, vl = _to_tensors(val_z, val_labels)
    xz, xl = _to_tensors(test_z, test_labels)

    best_val_acc, best_state, patience_ctr = 0.0, None, 0
    idx_to_name = {tn: {v: k for k, v in m.items()} for tn, m in label_maps.items()}

    for ep in range(cfg.aux_epochs):
        model.train(); optimizer.zero_grad()
        logits = model(tz)
        loss = sum(criterion(logits[k], tl[k]) for k in logits)
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            vpred = model(vz)
            val_acc = np.mean([float((vpred[k].argmax(1) == vl[k]).float().mean()) for k in vpred])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.aux_patience:
            logger.info(f"Aux classifier early stop at epoch {ep+1}"); break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    results = {"best_val_accuracy": float(best_val_acc), "per_head": {}}
    with torch.no_grad():
        test_logits = model(xz)

    for target_name in head_sizes:
        pred = test_logits[target_name].argmax(1).cpu().numpy()
        true = xl[target_name].cpu().numpy()
        n_cls = head_sizes[target_name]
        names = [idx_to_name[target_name].get(i, str(i)) for i in range(n_cls)]

        acc = accuracy_score(true, pred)
        prec, rec, f1, sup = precision_recall_fscore_support(true, pred, average=None, zero_division=0)
        cm = confusion_matrix(true, pred, labels=list(range(n_cls)))
        cls_rpt = classification_report(true, pred, target_names=names, zero_division=0)

        logger.info(f"Aux [{target_name}] accuracy: {acc:.4f}")
        logger.info(f"\n{cls_rpt}")

        results["per_head"][target_name] = {
            "accuracy": float(acc), "class_names": names,
            "per_class": {names[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                      "f1": float(f1[i]), "support": int(sup[i])}
                          for i in range(len(names))},
            "confusion_matrix": cm.tolist(),
            "report": cls_rpt,
        }

    return results, model


# ──────────────────────────────────────────────────────────────────────
# §11  VISUALISATION
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history, plot_dir, dpi=300):
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    def _save(fname, tk, vk, ylabel, title):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, history[tk], label="Train", lw=2, color="#0072B2")
        if vk: ax.plot(epochs, history[vk], label="Val", lw=2, color="#D55E00")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=dpi); plt.close(fig)

    _save("loss_total.png", "train_total", "val_total", "ELBO", "Total Loss")
    _save("loss_recon.png", "train_bce", "val_bce", "BCE", "Reconstruction Loss")
    _save("loss_kl.png", "train_kld", "val_kld", "KL Divergence", "KL Divergence")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["beta"], lw=2, color="#009E73")
    ax.set_xlabel("Epoch"); ax.set_ylabel("β"); ax.set_title("β Schedule")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(plot_dir / "beta_schedule.png", dpi=dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["lr"], lw=2, color="#CC79A7")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.set_yscale("log"); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(plot_dir / "lr_schedule.png", dpi=dpi); plt.close(fig)


def plot_latent_2d(latents, labels, class_names, out_path, title="Latent Space", dpi=300):
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        mask = labels == c
        if not np.any(mask): continue
        ax.scatter(latents[mask, 0], latents[mask, 1], s=50, alpha=0.75,
                   c=PALETTE[c % len(PALETTE)], marker=MARKERS[c % len(MARKERS)],
                   label=name, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title(title)
    ax.legend(loc="best", ncol=max(1, len(class_names)//8))
    ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_latent_with_gaussian(latents, labels, class_names, gauss, out_path, dpi=300):
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        mask = labels == c
        if not np.any(mask): continue
        ax.scatter(latents[mask, 0], latents[mask, 1], s=50, alpha=0.7,
                   c=PALETTE[c % len(PALETTE)], marker=MARKERS[c % len(MARKERS)],
                   label=name, edgecolors="white", linewidths=0.3)
    mean, cov = np.array(gauss["mean"]), np.array(gauss["covariance"])
    rv = sp_stats.multivariate_normal(mean=mean, cov=cov)
    xr = np.linspace(latents[:,0].min()-1, latents[:,0].max()+1, 200)
    yr = np.linspace(latents[:,1].min()-1, latents[:,1].max()+1, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = rv.pdf(np.dstack((X, Y)))
    ax.contour(X, Y, Z, levels=3, colors="black", linewidths=1.0, alpha=0.6, linestyles="--")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("Latent Space + Gaussian Contours")
    ax.legend(loc="best"); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_latent_tsne(latents, labels, class_names, out_path, title="t-SNE", dpi=300):
    """t-SNE for high-dim latents → 2D."""
    from sklearn.manifold import TSNE
    if latents.shape[1] <= 2:
        plot_latent_2d(latents, labels, class_names, out_path, title, dpi)
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(latents)//4)))
    emb = tsne.fit_transform(latents)
    plot_latent_2d(emb, labels, class_names, out_path, title, dpi)


def save_reconstructions(model, loader, device, out_path, label_maps, max_items=10, dpi=300):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x = batch[0].to(device); labels = batch[1]
        recon, _, _ = model(x)
    x_np = x.cpu().numpy()[:max_items, 0]
    r_np = recon.cpu().numpy()[:max_items, 0]
    n = x_np.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.8*n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        # build label string
        parts = []
        for tn, mapping in label_maps.items():
            idx = int(labels[tn][i])
            inv = {v: k for k, v in mapping.items()}
            parts.append(inv.get(idx, str(idx)))
        lbl_str = "/".join(parts)
        axes[0, i].imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Orig\n{lbl_str}", fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(r_np[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("Recon", fontsize=7); axes[1, i].axis("off")
    fig.suptitle("Reconstructions"); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_random_samples(model, device, latent_dim, out_path, n_rows=4, n_cols=4, dpi=300):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_rows*n_cols, latent_dim, device=device)
        imgs = model.decode(z).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 1.8*n_rows))
    for idx in range(n_rows*n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1); axes[r, c].axis("off")
    fig.suptitle("Random Samples from N(0, I)"); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_gaussian_samples(model, device, gauss, out_path, n_rows=4, n_cols=4, dpi=300):
    model.eval()
    n = n_rows * n_cols
    z_np = np.random.multivariate_normal(
        np.array(gauss["mean"]), np.array(gauss["covariance"]), size=n).astype(np.float32)
    with torch.no_grad():
        imgs = model.decode(torch.from_numpy(z_np).to(device)).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 1.8*n_rows))
    for idx in range(n):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1); axes[r, c].axis("off")
    fig.suptitle("Samples from Fitted Gaussian"); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_latent_traversal(model, device, latent_dim, out_path, grid_size=15, zrange=3.0, dpi=300):
    model.eval()
    sz = model.image_size
    vals = np.linspace(-zrange, zrange, grid_size, dtype=np.float32)
    canvas = np.zeros((grid_size*sz, grid_size*sz), dtype=np.float32)
    with torch.no_grad():
        for iy, zy in enumerate(vals):
            for ix, zx in enumerate(vals):
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = float(zx)
                if latent_dim > 1: z[0, 1] = float(zy)
                dec = model.decode(z).cpu().numpy()[0, 0]
                canvas[iy*sz:(iy+1)*sz, ix*sz:(ix+1)*sz] = dec
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    tick_pos = np.arange(grid_size) * sz + sz/2
    ax.set_xticks(tick_pos); ax.set_xticklabels([f"{v:.1f}" for v in vals], fontsize=7, rotation=45)
    ax.set_yticks(tick_pos); ax.set_yticklabels([f"{v:.1f}" for v in vals], fontsize=7)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("Latent Traversal"); fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix", dpi=300):
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)*0.8+1), max(4, len(class_names)*0.7+1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# §12  REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────

def save_history_csv(history, out_path):
    keys = list(history.keys())
    n = max(len(v) for v in history.values()) if history else 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(n):
            w.writerow([i+1] + [history[k][i] if i < len(history[k]) else "" for k in keys])


def write_evaluation_matrix(metrics, out_path, experiment_name=""):
    g = metrics["global"]
    lines = [
        f"# Evaluation Matrix — {experiment_name}", "",
        "## Global Metrics", "",
        "| Metric | Value | Interpretation |",
        "|--------|------:|----------------|",
        f"| MSE | {g['mse']:.6f} | Lower is better |",
        f"| PSNR (dB) | {g['psnr']:.2f} | >20 good, >30 excellent |",
        f"| SSIM | {g['ssim']:.4f} | >0.8 good, >0.9 excellent |",
        f"| BCE | {g['bce']:.4f} | Lower is better |",
        f"| KL Divergence | {g['kld']:.4f} | Balanced with BCE |",
        f"| ELBO | {g['total']:.4f} | Lower is better |", "",
    ]
    for target_name, classes in metrics.get("per_target", {}).items():
        lines += [f"## Per-Class: {target_name}", "",
                  f"| {target_name} | MSE | PSNR | SSIM | Count |",
                  f"|{'---'*1}|------:|-----:|-----:|------:|"]
        for cn, v in classes.items():
            lines.append(f"| {cn} | {v['mse']:.6f} | {v['psnr']:.2f} | {v['ssim']:.4f} | {v['count']} |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_depth_comparison(depth_data, out_path):
    lines = [
        "# Depth Scaling Comparison", "",
        "| Depth | MSE | PSNR | SSIM | KLD | ELBO | Params |",
        "|------:|------:|-----:|-----:|----:|-----:|-------:|",
    ]
    for d, dm in sorted(depth_data.items()):
        gm = dm.get("metrics", {}).get("global", {})
        lines.append(f"| {d} | {gm.get('mse',0):.6f} | {gm.get('psnr',0):.2f} | "
                     f"{gm.get('ssim',0):.4f} | {gm.get('kld',0):.4f} | {gm.get('total',0):.2f} | {dm.get('param_count','?')} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# §13  ORCHESTRATOR (used by exp scripts)
# ──────────────────────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig):
    """Full experiment pipeline: audit → train → evaluate → analyse → report."""
    # Directories
    report_dir = cfg.output_dir / "reports"
    model_dir = cfg.output_dir / "model"
    plot_dir = cfg.output_dir / "plots"
    log_dir = cfg.output_dir / "logs"
    for d in (report_dir, model_dir, plot_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, cfg.experiment_name)
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {cfg.experiment_name}")
    logger.info("=" * 70)

    cfg_path = cfg.output_dir / "experiment_config.json"
    cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    set_seed(cfg.seed)

    # ── Step 1: Data ──
    logger.info("Step 1: Collecting data and quality audit")
    records = collect_records(cfg)
    if not records:
        logger.error(f"No images found under {cfg.dataset_root}!"); sys.exit(1)
    logger.info(f"Collected {len(records)} images")

    audit = dataset_quality_audit(records, cfg.image_size, cfg)
    (report_dir / "dataset_quality_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # ── Step 2: Split ──
    logger.info("Step 2: Splitting data")
    train_recs, val_recs, test_recs = stratified_split(records, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    logger.info(f"Split: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # ── Build label maps ──
    target_names = [t[0] for t in cfg.classification_targets]
    label_maps = build_label_maps(records, target_names)
    head_sizes = {t[0]: len(label_maps[t[0]]) for t in cfg.classification_targets}
    class_names_by_target = {tn: sorted(m.keys(), key=lambda x: m[x]) for tn, m in label_maps.items()}

    logger.info(f"Classification targets: {head_sizes}")
    for tn, names in class_names_by_target.items():
        logger.info(f"  {tn}: {names}")

    # ── Dataloaders ──
    ds_kw = dict(image_size=cfg.image_size, label_maps=label_maps)
    train_ds = MultiLabelGlyphDataset(train_recs, augment=True, **ds_kw)
    val_ds = MultiLabelGlyphDataset(val_recs, augment=False, **ds_kw)
    test_ds = MultiLabelGlyphDataset(test_recs, augment=False, **ds_kw)
    all_ds = MultiLabelGlyphDataset(records, augment=False, **ds_kw)

    dl_kw = dict(batch_size=cfg.batch_size, num_workers=0, collate_fn=_collate_multi)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kw)
    all_loader = DataLoader(all_ds, shuffle=False, **dl_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Baseline training (base_depth) ──
    depth = cfg.base_depth
    tag = f"[depth={depth}] "
    logger.info(f"Step 3: Training baseline model (depth={depth})")

    set_seed(cfg.seed)
    model = ConvVAE(latent_dim=cfg.latent_dim, image_size=cfg.image_size, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{tag}Parameters: {n_params:,}, spatial: {model.encoder_spatial}x{model.encoder_spatial}")

    depth_model_dir = model_dir / f"depth{depth}"
    history = train_vae(model, train_loader, val_loader, cfg, device, depth_model_dir, logger, tag)
    save_history_csv(history, log_dir / "training_history.csv")

    ckpt = torch.load(depth_model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Training curves
    if history["train_total"]:
        plot_training_curves(history, plot_dir, cfg.dpi)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, cfg.beta_max, class_names_by_target, label_maps)
    logger.info(f"{tag}Test: MSE={metrics['global']['mse']:.6f}, PSNR={metrics['global']['psnr']:.2f}, SSIM={metrics['global']['ssim']:.4f}")

    (report_dir / "metrics_summary.json").write_text(json.dumps({
        "config": cfg.to_dict(), "best_epoch": ckpt.get("epoch", -1)+1,
        "param_count": n_params, "test_metrics": metrics,
    }, indent=2), encoding="utf-8")
    write_evaluation_matrix(metrics, report_dir / "evaluation_matrix.md", cfg.experiment_name)

    # Latent analysis
    latents, all_labels = collect_latents(model, all_loader, device)

    # Choose a primary target for scatter plot colouring
    primary_target = target_names[0]
    primary_labels = all_labels[primary_target]
    primary_names = class_names_by_target[primary_target]

    if cfg.latent_dim == 2:
        plot_latent_2d(latents, primary_labels, primary_names,
                       plot_dir / "latent_space_2d.png",
                       title=f"Latent Space (coloured by {primary_target})", dpi=cfg.dpi)
    else:
        plot_latent_tsne(latents, primary_labels, primary_names,
                         plot_dir / "latent_space_tsne.png",
                         title=f"t-SNE (coloured by {primary_target})", dpi=cfg.dpi)

    # If multiple targets, also plot by secondary
    if len(target_names) > 1:
        for tn in target_names[1:]:
            fname = f"latent_space_by_{tn}.png"
            if cfg.latent_dim == 2:
                plot_latent_2d(latents, all_labels[tn], class_names_by_target[tn],
                               plot_dir / fname, title=f"Latent (by {tn})", dpi=cfg.dpi)
            else:
                plot_latent_tsne(latents, all_labels[tn], class_names_by_target[tn],
                                 plot_dir / fname, title=f"t-SNE (by {tn})", dpi=cfg.dpi)

    # Reconstructions
    save_reconstructions(model, test_loader, device, plot_dir / "reconstructions.png", label_maps, dpi=cfg.dpi)

    # Random samples
    save_random_samples(model, device, cfg.latent_dim, plot_dir / "random_samples.png", dpi=cfg.dpi)

    # Latent traversal (only for 2D)
    if cfg.latent_dim == 2:
        save_latent_traversal(model, device, cfg.latent_dim, plot_dir / "latent_traversal_grid.png", dpi=cfg.dpi)

    # Gaussian fit
    gauss = fit_gaussian(latents)
    (report_dir / "gaussian_analysis.json").write_text(json.dumps(gauss, indent=2), encoding="utf-8")
    for dim, info in gauss["shapiro_wilk"].items():
        logger.info(f"Shapiro {dim}: p={info['p_value']:.4f}")

    if cfg.latent_dim == 2:
        plot_latent_with_gaussian(latents, primary_labels, primary_names, gauss,
                                  plot_dir / "latent_gaussian_fit.png", cfg.dpi)
        save_gaussian_samples(model, device, gauss, plot_dir / "gaussian_samples.png", dpi=cfg.dpi)

    # Auxiliary classifier
    tr_z, tr_l = collect_latents(model, train_loader, device)
    vl_z, vl_l = collect_latents(model, val_loader, device)
    te_z, te_l = collect_latents(model, test_loader, device)

    aux_results, _ = train_auxiliary_classifier(
        tr_z, tr_l, vl_z, vl_l, te_z, te_l, cfg, head_sizes, label_maps, device, logger)
    (report_dir / "auxiliary_classifier_report.json").write_text(
        json.dumps(aux_results, indent=2, default=str), encoding="utf-8")

    # Confusion matrices
    for tn, head_data in aux_results["per_head"].items():
        cm = np.array(head_data["confusion_matrix"])
        plot_confusion_matrix(cm, head_data["class_names"],
                              plot_dir / f"confusion_matrix_{tn}.png",
                              title=f"Aux Classifier: {tn}", dpi=cfg.dpi)

    # ── Depth variants ──
    depth_comp = {str(depth): {"metrics": metrics, "param_count": n_params}}
    for var_depth in cfg.depth_variants:
        logger.info(f"Depth variant: {var_depth}")
        set_seed(cfg.seed)
        var_model = ConvVAE(latent_dim=cfg.latent_dim, image_size=cfg.image_size, depth=var_depth).to(device)
        vp = sum(p.numel() for p in var_model.parameters())
        logger.info(f"[depth={var_depth}] Params: {vp:,}")

        var_model_dir = model_dir / f"depth{var_depth}"
        var_hist = train_vae(var_model, train_loader, val_loader, cfg, device,
                             var_model_dir, logger, f"[depth={var_depth}] ")
        save_history_csv(var_hist, log_dir / f"training_history_depth{var_depth}.csv")

        var_ckpt = torch.load(var_model_dir / "best_model.pt", map_location=device, weights_only=False)
        var_model.load_state_dict(var_ckpt["model_state"])
        var_metrics = evaluate_model(var_model, test_loader, device, cfg.beta_max,
                                     class_names_by_target, label_maps)
        logger.info(f"[depth={var_depth}] MSE={var_metrics['global']['mse']:.6f}, "
                     f"PSNR={var_metrics['global']['psnr']:.2f}, SSIM={var_metrics['global']['ssim']:.4f}")

        depth_comp[str(var_depth)] = {"metrics": var_metrics, "param_count": vp}

        # Variant plots
        vdir = plot_dir / f"depth{var_depth}"
        vdir.mkdir(parents=True, exist_ok=True)
        if var_hist["train_total"]:
            plot_training_curves(var_hist, vdir, cfg.dpi)
        vlatents, vlabels = collect_latents(var_model, all_loader, device)
        if cfg.latent_dim == 2:
            plot_latent_2d(vlatents, vlabels[primary_target], primary_names,
                           vdir / "latent_space_2d.png", f"Latent (depth={var_depth})", cfg.dpi)
        save_reconstructions(var_model, test_loader, device, vdir / "reconstructions.png", label_maps, dpi=cfg.dpi)

    write_depth_comparison(depth_comp, report_dir / "depth_scaling_comparison.md")

    # ── Final manifest ──
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"{cfg.experiment_name} COMPLETE — Artifacts:")
    logger.info("=" * 70)
    for d in (report_dir, model_dir, plot_dir, log_dir):
        for f in sorted(d.rglob("*")):
            if f.is_file():
                logger.info(f"  {f.relative_to(cfg.output_dir)}")
    logger.info("=" * 70)
