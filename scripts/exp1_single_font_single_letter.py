#!/usr/bin/env python3
"""
Experiment 1: Single Font, Single Letter — Full VAE Pipeline
=============================================================
Data:  lohit_telugu / letter 'a' / 4 point-size classes (10pt, 14pt, 18pt, 22pt)
Goal:  Train a ConvVAE with 2-D latent space, evaluate reconstruction quality,
       analyse latent structure (Gaussian fit, auxiliary classifier), and
       perform progressive depth scaling.

Outputs (all under results/exp1_single_font_single_letter/):
  reports/   – JSON & Markdown reports
  model/     – checkpoints per depth variant
  plots/     – all visualisation PNGs (300 DPI, publication-quality)
  logs/      – CSV training history + timestamped training.log
  experiment_config.json
"""

from __future__ import annotations

import argparse
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
matplotlib.use("Agg")           # non-interactive backend
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
# §1  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

FONT_NAMES: tuple[str, ...] = ("lohit_telugu",)
LETTER_NAMES: tuple[str, ...] = ("a",)
POINT_SIZES: tuple[str, ...] = ("10pt", "14pt", "18pt", "22pt")

# Colour-blind-safe palette (Wong, 2011)
CLASS_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
CLASS_MARKERS = ["o", "s", "^", "D"]

# Matplotlib publication defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


@dataclass
class ExperimentConfig:
    """All hyper-parameters and paths for one experiment run."""
    # ── data ──
    dataset_root: Path = Path("dataset_Rohit")
    font_names: tuple[str, ...] = FONT_NAMES
    letter_names: tuple[str, ...] = LETTER_NAMES
    point_sizes: tuple[str, ...] = POINT_SIZES
    image_size: int = 128

    # ── training ──
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

    # ── architecture ──
    base_depth: int = 4
    depth_variants: tuple[int, ...] = (5, 6)

    # ── auxiliary classifier ──
    aux_hidden: int = 64
    aux_lr: float = 1e-3
    aux_epochs: int = 200
    aux_patience: int = 30
    aux_dropout: float = 0.3

    # ── output ──
    output_dir: Path = Path("results/exp1_single_font_single_letter")
    dpi: int = 300
    recon_log_interval: int = 10

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

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exp1")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
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
    img = img.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    return img


def apply_augmentation(image: Image.Image) -> Image.Image:
    out = image
    if random.random() < 0.85:
        angle = random.uniform(-8.0, 8.0)
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if random.random() < 0.85:
        tx = random.uniform(-3.0, 3.0)
        ty = random.uniform(-3.0, 3.0)
        out = out.transform(
            out.size, Image.Transform.AFFINE,
            (1.0, 0.0, tx, 0.0, 1.0, ty),
            resample=Image.Resampling.BICUBIC, fillcolor=255,
        )
    if random.random() < 0.5:
        c = random.uniform(0.85, 1.20)
        out = ImageEnhance.Contrast(out).enhance(c)
    return out


def collect_records(cfg: ExperimentConfig) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for font in cfg.font_names:
        for letter in cfg.letter_names:
            for pt in cfg.point_sizes:
                d = cfg.dataset_root / font / letter / pt
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if f.is_file() and f.suffix.lower() == ".png":
                        records.append(ImageRecord(
                            path=f, font=font, letter=letter, point_size=pt,
                        ))
    return records


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    groups: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[f"{rec.font}:{rec.letter}:{rec.point_size}"].append(rec)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for group in groups.values():
        g = group[:]
        rng.shuffle(g)
        n = len(g)
        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(g[:n_train])
        val.extend(g[n_train:n_train + n_val])
        test.extend(g[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class GlyphDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int,
        point_sizes: tuple[str, ...],
        augment: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.size_to_idx = {s: i for i, s in enumerate(point_sizes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        image = load_image_as_grayscale(rec.path, self.image_size)
        if self.augment:
            image = apply_augmentation(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)        # [1, H, W]
        label = self.size_to_idx[rec.point_size]
        return tensor, label


# ──────────────────────────────────────────────────────────────────────
# §4  CONV-VAE MODEL (dynamic depth)
# ──────────────────────────────────────────────────────────────────────

def _channel_sequence(depth: int) -> list[int]:
    """Return channel counts for each conv block."""
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
    """Convolutional VAE with configurable depth and batch normalisation."""

    def __init__(self, latent_dim: int = 2, image_size: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.depth = depth
        self.encoder_spatial = image_size // (2 ** depth)

        channels = _channel_sequence(depth)
        self.last_ch = channels[-1]

        # Encoder
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        self.encoder_out_dim = self.last_ch * self.encoder_spatial * self.encoder_spatial
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoder_out_dim)
        dec_layers: list[nn.Module] = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            dec_layers += [
                nn.ConvTranspose2d(rev_channels[i], rev_channels[i + 1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(rev_channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        # final layer → 1 channel
        dec_layers += [
            nn.ConvTranspose2d(rev_channels[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.last_ch, self.encoder_spatial, self.encoder_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ──────────────────────────────────────────────────────────────────────
# §5  AUXILIARY LATENT CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 64,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ──────────────────────────────────────────────────────────────────────
# §6  LOSS & TRAINING
# ──────────────────────────────────────────────────────────────────────

def vae_loss(
    recon: torch.Tensor, x: torch.Tensor,
    mu: torch.Tensor, logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = bce + beta * kld
    return total, {
        "bce": float(bce.detach()),
        "kld": float(kld.detach()),
        "total": float(total.detach()),
    }


def get_beta(epoch: int, warmup: int, beta_max: float) -> float:
    if warmup <= 0:
        return beta_max
    return min(beta_max, beta_max * (epoch + 1) / warmup)


def evaluate_epoch(
    model: ConvVAE, loader: DataLoader, device: torch.device, beta: float,
) -> dict[str, float]:
    model.eval()
    bce_v, kld_v, tot_v = [], [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            _, s = vae_loss(recon, x, mu, logvar, beta)
            bce_v.append(s["bce"])
            kld_v.append(s["kld"])
            tot_v.append(s["total"])
    return {
        "bce": float(np.mean(bce_v)) if bce_v else math.nan,
        "kld": float(np.mean(kld_v)) if kld_v else math.nan,
        "total": float(np.mean(tot_v)) if tot_v else math.nan,
    }


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Global SSIM between two [0,1] float images."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x, sigma_y = x.var(), y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(num / den) if den != 0 else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_vae(
    model: ConvVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    model_dir: Path,
    logger: logging.Logger,
    log_prefix: str = "",
) -> dict[str, list[float]]:
    """Train VAE. Returns history dict. Saves best_model.pt & final_model.pt."""
    model_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-5,
    )

    history: dict[str, list[float]] = {
        "train_bce": [], "train_kld": [], "train_total": [],
        "val_bce": [], "val_kld": [], "val_total": [],
        "beta": [], "lr": [],
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        beta = get_beta(epoch, cfg.kl_warmup_epochs, cfg.beta_max)

        t_bce, t_kld, t_tot = [], [], []
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, s = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()
            t_bce.append(s["bce"]); t_kld.append(s["kld"]); t_tot.append(s["total"])

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        tr = {"bce": np.mean(t_bce), "kld": np.mean(t_kld), "total": np.mean(t_tot)}
        vl = evaluate_epoch(model, val_loader, device, beta)

        for k in ("bce", "kld", "total"):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(vl[k]))
        history["beta"].append(beta)
        history["lr"].append(current_lr)

        logger.info(
            f"{log_prefix}Epoch {epoch+1:03d}/{cfg.epochs} | β={beta:.3f} | "
            f"lr={current_lr:.2e} | train={tr['total']:.2f} | val={vl['total']:.2f}"
        )

        if vl["total"] < best_val:
            best_val = vl["total"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "best_val_total": best_val,
                "depth": model.depth,
            }, model_dir / "best_model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.early_stopping_patience:
            logger.info(
                f"{log_prefix}Early stopping at epoch {epoch+1}, "
                f"best was epoch {best_epoch+1}"
            )
            break

    # save final model
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "depth": model.depth,
    }, model_dir / "final_model.pt")

    logger.info(f"{log_prefix}Training complete. Best epoch: {best_epoch+1}, best val: {best_val:.4f}")
    return history


# ──────────────────────────────────────────────────────────────────────
# §7  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    num_classes: int,
) -> dict[str, Any]:
    """Full evaluation: global + per-class MSE, PSNR, SSIM."""
    model.eval()
    bce_v, kld_v, tot_v = [], [], []
    per_class: dict[int, dict[str, list[float]]] = {
        c: {"mse": [], "psnr": [], "ssim": []} for c in range(num_classes)
    }
    global_mse, global_psnr, global_ssim = [], [], []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            _, s = vae_loss(recon, x, mu, logvar, beta)
            bce_v.append(s["bce"]); kld_v.append(s["kld"]); tot_v.append(s["total"])

            x_np = x.cpu().numpy()
            r_np = recon.cpu().numpy()
            labels_np = labels.numpy()

            for i in range(x_np.shape[0]):
                xi, ri = x_np[i, 0], r_np[i, 0]
                mse = float(np.mean((xi - ri) ** 2))
                psnr = 20.0 * math.log10(1.0 / math.sqrt(mse + 1e-12))
                ssim = compute_ssim(xi, ri)

                global_mse.append(mse)
                global_psnr.append(psnr)
                global_ssim.append(ssim)

                cls = int(labels_np[i])
                per_class[cls]["mse"].append(mse)
                per_class[cls]["psnr"].append(psnr)
                per_class[cls]["ssim"].append(ssim)

    result: dict[str, Any] = {
        "global": {
            "bce": float(np.mean(bce_v)),
            "kld": float(np.mean(kld_v)),
            "total": float(np.mean(tot_v)),
            "mse": float(np.mean(global_mse)),
            "psnr": float(np.mean(global_psnr)),
            "ssim": float(np.mean(global_ssim)),
        },
        "per_class": {},
    }
    for c in range(num_classes):
        name = POINT_SIZES[c] if c < len(POINT_SIZES) else str(c)
        vals = per_class[c]
        result["per_class"][name] = {
            "mse": float(np.mean(vals["mse"])) if vals["mse"] else math.nan,
            "psnr": float(np.mean(vals["psnr"])) if vals["psnr"] else math.nan,
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else math.nan,
            "count": len(vals["mse"]),
        }
    return result


# ──────────────────────────────────────────────────────────────────────
# §8  DATASET QUALITY AUDIT
# ──────────────────────────────────────────────────────────────────────

def dataset_quality_audit(
    records: list[ImageRecord], image_size: int,
) -> dict[str, Any]:
    corrupt: list[str] = []
    modes: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    hash_to_files: defaultdict[str, list[str]] = defaultdict(list)

    int_means, int_stds, nw_ratios, sharp_vals = [], [], [], []
    cx_vals, cy_vals = [], []
    by_size_counts: Counter[str] = Counter()
    by_size_nw: defaultdict[str, list[float]] = defaultdict(list)
    by_size_sharp: defaultdict[str, list[float]] = defaultdict(list)

    for rec in records:
        by_size_counts[rec.point_size] += 1
        try:
            with Image.open(rec.path) as raw:
                modes[raw.mode] += 1
                resolutions[f"{raw.width}x{raw.height}"] += 1
            img = load_image_as_grayscale(rec.path, image_size)
            arr = np.asarray(img, dtype=np.float32)
        except Exception:
            corrupt.append(str(rec.path))
            continue

        h = hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()
        hashes[h] += 1
        hash_to_files[h].append(str(rec.path))

        int_means.append(float(arr.mean() / 255.0))
        int_stds.append(float(arr.std() / 255.0))
        nw = float((arr < 245.0).mean())
        nw_ratios.append(nw)
        shp = float(np.var(laplace(arr)))
        sharp_vals.append(shp)
        by_size_nw[rec.point_size].append(nw)
        by_size_sharp[rec.point_size].append(shp)

        ink = np.clip(255.0 - arr, 0.0, None)
        mass = float(ink.sum())
        if mass > 0:
            ys, xs = np.indices(arr.shape)
            cx_vals.append(float((xs * ink).sum() / mass))
            cy_vals.append(float((ys * ink).sum() / mass))

    def _stat(v: list[float]) -> dict:
        if not v:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        a = np.array(v, dtype=np.float64)
        return {"count": len(v), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    dup_groups = [v for v in hash_to_files.values() if len(v) > 1]

    report = {
        "scope": {
            "fonts": list(FONT_NAMES), "letters": list(LETTER_NAMES),
            "point_sizes": list(POINT_SIZES), "resized_to": f"{image_size}x{image_size}",
        },
        "integrity": {
            "total": len(records), "corrupt": len(corrupt), "corrupt_files": corrupt,
            "mode_distribution": dict(modes), "resolution_distribution": dict(resolutions),
            "duplicate_groups": len(dup_groups), "unique_hashes": sum(1 for c in hashes.values() if c == 1),
        },
        "counts_by_size": dict(by_size_counts),
        "pixel_statistics": {
            "intensity_mean": _stat(int_means), "intensity_std": _stat(int_stds),
            "non_white_ratio": _stat(nw_ratios), "sharpness": _stat(sharp_vals),
            "center_x": _stat(cx_vals), "center_y": _stat(cy_vals),
        },
        "by_size_features": {
            sz: {"non_white_ratio": _stat(by_size_nw[sz]),
                 "sharpness": _stat(by_size_sharp[sz])}
            for sz in POINT_SIZES
        },
        "duplicate_groups": dup_groups,
    }
    return report


# ──────────────────────────────────────────────────────────────────────
# §9  LATENT SPACE ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def collect_latents(
    model: ConvVAE, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu_array [N, latent_dim], labels [N])."""
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for x, lbl in loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(mus), np.concatenate(labels)


def fit_gaussian(latents: np.ndarray) -> dict[str, Any]:
    """Fit multivariate Gaussian + per-dim normality tests."""
    mean = latents.mean(axis=0)
    cov = np.cov(latents, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    normality = {}
    for d in range(latents.shape[1]):
        stat, p = sp_stats.shapiro(latents[:, d])
        normality[f"dim_{d}"] = {"shapiro_stat": float(stat), "p_value": float(p)}

    # KS test against standard normal per dimension
    ks_tests = {}
    for d in range(latents.shape[1]):
        stat, p = sp_stats.kstest(latents[:, d], "norm",
                                  args=(float(mean[d]), float(np.sqrt(cov[d, d]))))
        ks_tests[f"dim_{d}"] = {"ks_stat": float(stat), "p_value": float(p)}

    return {
        "mean": mean.tolist(),
        "covariance": cov.tolist(),
        "shapiro_wilk": normality,
        "ks_test": ks_tests,
        "n_samples": int(latents.shape[0]),
    }


# ──────────────────────────────────────────────────────────────────────
# §10  AUXILIARY CLASSIFIER TRAINING
# ──────────────────────────────────────────────────────────────────────

def train_auxiliary_classifier(
    train_z: np.ndarray, train_y: np.ndarray,
    val_z: np.ndarray, val_y: np.ndarray,
    test_z: np.ndarray, test_y: np.ndarray,
    cfg: ExperimentConfig,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Train MLP on latent mu vectors. Returns metrics + trained model."""
    model = LatentClassifier(
        latent_dim=cfg.latent_dim, num_classes=num_classes,
        hidden=cfg.aux_hidden, dropout=cfg.aux_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.aux_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    def _to_tensor(z, y):
        return (torch.tensor(z, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.long, device=device))

    tz, ty = _to_tensor(train_z, train_y)
    vz, vy = _to_tensor(val_z, val_y)
    xz, xy = _to_tensor(test_z, test_y)

    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0

    for ep in range(cfg.aux_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(tz)
        loss = criterion(logits, ty)
        loss.backward()
        optimizer.step()

        # Val accuracy
        model.eval()
        with torch.no_grad():
            val_pred = model(vz).argmax(dim=1)
            val_acc = float((val_pred == vy).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= cfg.aux_patience:
            logger.info(f"Aux classifier early stop at epoch {ep+1}")
            break

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(xz)
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    test_y_np = xy.cpu().numpy()
    acc = accuracy_score(test_y_np, test_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        test_y_np, test_pred, average=None, zero_division=0,
    )
    cm = confusion_matrix(test_y_np, test_pred)
    class_names = [POINT_SIZES[i] for i in range(num_classes)]
    cls_report = classification_report(
        test_y_np, test_pred, target_names=class_names, zero_division=0,
    )

    logger.info(f"Auxiliary classifier test accuracy: {acc:.4f}")
    logger.info(f"\n{cls_report}")

    return {
        "test_accuracy": float(acc),
        "best_val_accuracy": float(best_val_acc),
        "per_class": {
            class_names[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                             "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "model": model,
    }


# ──────────────────────────────────────────────────────────────────────
# §11  VISUALISATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict[str, list[float]], plot_dir: Path, dpi: int = 300,
) -> None:
    """Save 5 separate training curve PNGs."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    def _save(fname, train_key, val_key, ylabel, title):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, history[train_key], label="Train", linewidth=2, color="#0072B2")
        if val_key:
            ax.plot(epochs, history[val_key], label="Val", linewidth=2, color="#D55E00")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=dpi)
        plt.close(fig)

    _save("loss_total.png", "train_total", "val_total", "ELBO", "Total Loss (ELBO)")
    _save("loss_recon.png", "train_bce", "val_bce", "BCE", "Reconstruction Loss (BCE)")
    _save("loss_kl.png", "train_kld", "val_kld", "KL Divergence", "KL Divergence")

    # Beta schedule
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["beta"], linewidth=2, color="#009E73")
    ax.set_xlabel("Epoch"); ax.set_ylabel("β"); ax.set_title("KL Weight (β) Schedule")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(plot_dir / "beta_schedule.png", dpi=dpi); plt.close(fig)

    # LR schedule
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["lr"], linewidth=2, color="#CC79A7")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log"); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(plot_dir / "lr_schedule.png", dpi=dpi); plt.close(fig)


def plot_latent_2d(
    latents: np.ndarray, labels: np.ndarray, class_names: list[str],
    out_path: Path, title: str = "2D Latent Space", dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(latents[mask, 0], latents[mask, 1], s=50, alpha=0.8,
                   c=CLASS_COLOURS[c % len(CLASS_COLOURS)],
                   marker=CLASS_MARKERS[c % len(CLASS_MARKERS)],
                   label=name, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="gray", lw=0.7, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.7, alpha=0.4)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_latent_with_gaussian(
    latents: np.ndarray, labels: np.ndarray, class_names: list[str],
    gauss_info: dict, out_path: Path, dpi: int = 300,
) -> None:
    """Latent scatter + fitted Gaussian contour overlay."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # scatter
    for c, name in enumerate(class_names):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(latents[mask, 0], latents[mask, 1], s=50, alpha=0.7,
                   c=CLASS_COLOURS[c % len(CLASS_COLOURS)],
                   marker=CLASS_MARKERS[c % len(CLASS_MARKERS)],
                   label=name, edgecolors="white", linewidths=0.3)

    # Gaussian contour
    mean = np.array(gauss_info["mean"])
    cov = np.array(gauss_info["covariance"])
    rv = sp_stats.multivariate_normal(mean=mean, cov=cov)

    x_range = np.linspace(latents[:, 0].min() - 1, latents[:, 0].max() + 1, 200)
    y_range = np.linspace(latents[:, 1].min() - 1, latents[:, 1].max() + 1, 200)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)

    # 1σ, 2σ, 3σ contours
    ax.contour(X, Y, Z, levels=3, colors="black", linewidths=1.0, alpha=0.6, linestyles="--")

    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title("Latent Space with Fitted Gaussian Contours")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_reconstructions(
    model: ConvVAE, loader: DataLoader, device: torch.device,
    out_path: Path, max_items: int = 10, dpi: int = 300,
) -> None:
    model.eval()
    with torch.no_grad():
        x, labels = next(iter(loader))
        x = x.to(device)
        recon, _, _ = model(x)
    x_np = x.cpu().numpy()[:max_items, 0]
    r_np = recon.cpu().numpy()[:max_items, 0]
    lbl = labels.numpy()[:max_items]
    n = x_np.shape[0]

    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Orig ({POINT_SIZES[lbl[i]]})", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(r_np[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("Recon", fontsize=8)
        axes[1, i].axis("off")
    fig.suptitle("Reconstructions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_random_samples(
    model: ConvVAE, device: torch.device, latent_dim: int,
    out_path: Path, n_rows: int = 4, n_cols: int = 4, dpi: int = 300,
) -> None:
    model.eval()
    n = n_rows * n_cols
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        imgs = model.decode(z).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    for idx in range(n):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
    fig.suptitle("Random Samples from N(0, I)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_gaussian_samples(
    model: ConvVAE, device: torch.device, gauss_info: dict,
    out_path: Path, n_rows: int = 4, n_cols: int = 4, dpi: int = 300,
) -> None:
    """Sample from the fitted Gaussian (not standard normal) and decode."""
    model.eval()
    n = n_rows * n_cols
    mean = np.array(gauss_info["mean"])
    cov = np.array(gauss_info["covariance"])
    z_np = np.random.multivariate_normal(mean, cov, size=n).astype(np.float32)
    z = torch.from_numpy(z_np).to(device)
    with torch.no_grad():
        imgs = model.decode(z).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    for idx in range(n):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
    fig.suptitle("Samples from Fitted Gaussian")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_latent_traversal(
    model: ConvVAE, device: torch.device, latent_dim: int,
    out_path: Path, grid_size: int = 15, zrange: float = 3.0, dpi: int = 300,
) -> None:
    model.eval()
    out_sz = model.image_size
    vals = np.linspace(-zrange, zrange, grid_size, dtype=np.float32)
    canvas = np.zeros((grid_size * out_sz, grid_size * out_sz), dtype=np.float32)
    with torch.no_grad():
        for iy, zy in enumerate(vals):
            for ix, zx in enumerate(vals):
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = float(zx)
                if latent_dim > 1:
                    z[0, 1] = float(zy)
                dec = model.decode(z).cpu().numpy()[0, 0]
                canvas[iy*out_sz:(iy+1)*out_sz, ix*out_sz:(ix+1)*out_sz] = dec
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title(f"Latent Traversal Grid ($z \\in [{-zrange}, {zrange}]$)")
    # set ticks at grid centers
    tick_pos = np.arange(grid_size) * out_sz + out_sz / 2
    tick_labels = [f"{v:.1f}" for v in vals]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], out_path: Path, dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted",
           title="Auxiliary Classifier — Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# §12  REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────

def save_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
    keys = list(history.keys())
    n = max(len(v) for v in history.values()) if history else 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(n):
            row = [i + 1] + [history[k][i] if i < len(history[k]) else "" for k in keys]
            w.writerow(row)


def write_evaluation_matrix(metrics: dict, out_path: Path) -> None:
    g = metrics["global"]
    lines = [
        "# Evaluation Matrix — Experiment 1 (Single Font, Single Letter)",
        "",
        "## Global Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|------:|----------------|",
        f"| MSE | {g['mse']:.6f} | Lower is better |",
        f"| PSNR (dB) | {g['psnr']:.2f} | Higher is better (>20 good, >30 excellent) |",
        f"| SSIM | {g['ssim']:.4f} | Higher is better (>0.8 good, >0.9 excellent) |",
        f"| BCE | {g['bce']:.4f} | Lower is better |",
        f"| KL Divergence | {g['kld']:.4f} | Balanced with BCE |",
        f"| ELBO | {g['total']:.4f} | Lower is better |",
        "",
        "## Per-Class Metrics",
        "",
        "| Point Size | MSE | PSNR (dB) | SSIM | Count |",
        "|-----------|------:|----------:|-----:|------:|",
    ]
    for name, vals in metrics["per_class"].items():
        lines.append(
            f"| {name} | {vals['mse']:.6f} | {vals['psnr']:.2f} | "
            f"{vals['ssim']:.4f} | {vals['count']} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_summary(
    all_data: dict[str, Any], out_path: Path,
) -> None:
    """Compile the final experiment_1_summary.md."""
    lines = [
        "# Experiment 1 Summary — Single Font, Single Letter",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Font:** {', '.join(FONT_NAMES)}",
        f"**Letter:** {', '.join(LETTER_NAMES)}",
        f"**Classes:** {', '.join(POINT_SIZES)}",
        "",
    ]

    # Dataset
    ds = all_data.get("dataset_audit", {})
    lines += [
        "## 1. Dataset",
        f"- Total samples: {ds.get('integrity', {}).get('total', '?')}",
        f"- Corrupt files: {ds.get('integrity', {}).get('corrupt', '?')}",
        f"- Duplicate groups: {ds.get('integrity', {}).get('duplicate_groups', '?')}",
        "",
    ]

    # Training
    cfg = all_data.get("config", {})
    lines += [
        "## 2. Training Configuration",
        f"- Image size: {cfg.get('image_size', '?')}",
        f"- Latent dim: {cfg.get('latent_dim', '?')}",
        f"- Epochs (max): {cfg.get('epochs', '?')}",
        f"- Beta max: {cfg.get('beta_max', '?')}",
        f"- KL warmup: {cfg.get('kl_warmup_epochs', '?')} epochs",
        f"- Early stopping patience: {cfg.get('early_stopping_patience', '?')}",
        "",
    ]

    # Baseline metrics
    base_m = all_data.get("baseline_metrics", {}).get("global", {})
    if base_m:
        lines += [
            "## 3. Baseline Model (depth=4) — Test Metrics",
            f"- MSE: {base_m.get('mse', '?'):.6f}" if isinstance(base_m.get('mse'), float) else f"- MSE: {base_m.get('mse', '?')}",
            f"- PSNR: {base_m.get('psnr', '?'):.2f} dB" if isinstance(base_m.get('psnr'), float) else f"- PSNR: {base_m.get('psnr', '?')}",
            f"- SSIM: {base_m.get('ssim', '?'):.4f}" if isinstance(base_m.get('ssim'), float) else f"- SSIM: {base_m.get('ssim', '?')}",
            "",
        ]

    # Gaussian analysis
    gauss = all_data.get("gaussian_analysis", {})
    if gauss:
        lines += ["## 4. Gaussian Continuity Analysis", ""]
        sw = gauss.get("shapiro_wilk", {})
        for dim, vals in sw.items():
            p = vals.get("p_value", "?")
            interp = "Cannot reject normality" if isinstance(p, float) and p > 0.05 else "Deviates from normal"
            lines.append(f"- {dim}: Shapiro p={p:.4f} → {interp}" if isinstance(p, float) else f"- {dim}: p={p}")
        lines.append("")

    # Auxiliary classifier
    aux = all_data.get("auxiliary_classifier", {})
    if aux:
        lines += [
            "## 5. Auxiliary Classifier (Point-Size Prediction)",
            f"- Test accuracy: {aux.get('test_accuracy', '?'):.4f}" if isinstance(aux.get('test_accuracy'), float) else f"- Test accuracy: {aux.get('test_accuracy', '?')}",
            f"- Best val accuracy: {aux.get('best_val_accuracy', '?'):.4f}" if isinstance(aux.get('best_val_accuracy'), float) else f"- Best val accuracy: {aux.get('best_val_accuracy', '?')}",
            "",
        ]
        for cls_name, cls_vals in aux.get("per_class", {}).items():
            lines.append(f"  - {cls_name}: P={cls_vals['precision']:.2f} R={cls_vals['recall']:.2f} F1={cls_vals['f1']:.2f}")
        lines.append("")

    # Depth scaling
    depth_comp = all_data.get("depth_comparison", {})
    if depth_comp:
        lines += [
            "## 6. Progressive Depth Scaling",
            "",
            "| Depth | MSE | PSNR | SSIM | Aux Accuracy |",
            "|------:|------:|-----:|-----:|-------------:|",
        ]
        for depth_str, dm in sorted(depth_comp.items()):
            gm = dm.get("metrics", {}).get("global", {})
            aa = dm.get("aux_accuracy", "?")
            lines.append(
                f"| {depth_str} | {gm.get('mse', 0):.6f} | {gm.get('psnr', 0):.2f} | "
                f"{gm.get('ssim', 0):.4f} | {aa if isinstance(aa, str) else f'{aa:.4f}'} |"
            )
        lines.append("")

    lines += [
        "## 7. Observations",
        "",
        "_To be filled based on visual inspection of plots and numerical results._",
        "",
        "---",
        f"*Generated automatically by exp1_single_font_single_letter.py*",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_depth_comparison(depth_data: dict, out_path: Path) -> None:
    lines = [
        "# Depth Scaling Comparison",
        "",
        "| Depth | Conv Blocks | MSE | PSNR (dB) | SSIM | KLD | ELBO | Aux Acc | Params |",
        "|------:|----------:|------:|----------:|-----:|----:|-----:|--------:|-------:|",
    ]
    for depth_str, dm in sorted(depth_data.items()):
        gm = dm.get("metrics", {}).get("global", {})
        aa = dm.get("aux_accuracy", "?")
        params = dm.get("param_count", "?")
        lines.append(
            f"| {depth_str} | {depth_str} | {gm.get('mse', 0):.6f} | {gm.get('psnr', 0):.2f} | "
            f"{gm.get('ssim', 0):.4f} | {gm.get('kld', 0):.4f} | {gm.get('total', 0):.2f} | "
            f"{aa if isinstance(aa, str) else f'{aa:.4f}'} | {params} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# §13  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run_single_depth(
    depth: int,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    all_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    plot_dir: Path,
    log_dir: Path,
    model_dir: Path,
    report_dir: Path,
    is_baseline: bool = False,
) -> dict[str, Any]:
    """Train + evaluate a single depth variant. Returns all metrics."""
    tag = f"[depth={depth}] "
    logger.info(f"{tag}{'='*50}")
    logger.info(f"{tag}Starting {'BASELINE' if is_baseline else 'VARIANT'} training")

    set_seed(cfg.seed)
    model = ConvVAE(latent_dim=cfg.latent_dim, image_size=cfg.image_size, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{tag}Model parameters: {n_params:,}")
    logger.info(f"{tag}Encoder spatial: {model.encoder_spatial}x{model.encoder_spatial}")

    depth_model_dir = model_dir / f"depth{depth}"
    history = train_vae(model, train_loader, val_loader, cfg, device,
                        depth_model_dir, logger, log_prefix=tag)

    # Save history
    suffix = "" if is_baseline else f"_depth{depth}"
    save_history_csv(history, log_dir / f"training_history{suffix}.csv")

    # Load best checkpoint
    ckpt = torch.load(depth_model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    best_epoch = ckpt.get("epoch", -1) + 1

    # Evaluate
    num_classes = len(cfg.point_sizes)
    metrics = evaluate_model(model, test_loader, device, cfg.beta_max, num_classes)
    logger.info(f"{tag}Test metrics: MSE={metrics['global']['mse']:.6f}, "
                f"PSNR={metrics['global']['psnr']:.2f}, SSIM={metrics['global']['ssim']:.4f}")

    # Plot training curves
    if len(history["train_total"]) > 0:
        if is_baseline:
            plot_training_curves(history, plot_dir, cfg.dpi)
        else:
            # single combined loss plot for variants
            depth_plot_dir = plot_dir / f"depth{depth}"
            depth_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_training_curves(history, depth_plot_dir, cfg.dpi)

    # Collect latents
    latents, labels = collect_latents(model, all_loader, device)

    # Latent space plot
    class_names = list(cfg.point_sizes)
    suffix_str = "" if is_baseline else f"_depth{depth}"
    plot_latent_2d(latents, labels, class_names,
                   plot_dir / f"latent_space_2d{suffix_str}.png",
                   title=f"2D Latent Space (depth={depth})", dpi=cfg.dpi)

    # Reconstructions
    save_reconstructions(model, test_loader, device,
                         plot_dir / f"reconstructions{suffix_str}.png",
                         max_items=10, dpi=cfg.dpi)

    result = {
        "depth": depth,
        "param_count": n_params,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "history": history,
        "model": model,
        "latents": latents,
        "labels": labels,
    }

    if is_baseline:
        # ── Steps only for baseline ──

        # Random samples from prior
        save_random_samples(model, device, cfg.latent_dim,
                            plot_dir / "random_samples.png", dpi=cfg.dpi)

        # Latent traversal
        save_latent_traversal(model, device, cfg.latent_dim,
                              plot_dir / "latent_traversal_grid.png",
                              grid_size=15, zrange=3.0, dpi=cfg.dpi)

        # Gaussian fit + normality
        gauss = fit_gaussian(latents)
        logger.info(f"{tag}Gaussian mean: {gauss['mean']}")
        for dim, info in gauss["shapiro_wilk"].items():
            logger.info(f"{tag}Shapiro-Wilk {dim}: stat={info['shapiro_stat']:.4f}, p={info['p_value']:.4f}")
        for dim, info in gauss["ks_test"].items():
            logger.info(f"{tag}KS test {dim}: stat={info['ks_stat']:.4f}, p={info['p_value']:.4f}")

        plot_latent_with_gaussian(latents, labels, class_names, gauss,
                                  plot_dir / "latent_gaussian_fit.png", cfg.dpi)

        # Samples from fitted Gaussian
        save_gaussian_samples(model, device, gauss,
                              plot_dir / "gaussian_samples.png", dpi=cfg.dpi)

        result["gaussian_analysis"] = gauss

        # Save Gaussian analysis
        gauss_path = report_dir / "gaussian_analysis.json"
        gauss_path.write_text(json.dumps(gauss, indent=2), encoding="utf-8")

    # Auxiliary classifier
    # Split latents according to dataset splits
    # We need to re-collect per-split latents
    train_latents, train_labels = collect_latents(model, train_loader, device)
    val_latents, val_labels = collect_latents(model, val_loader, device)
    test_latents, test_labels = collect_latents(model, test_loader, device)

    aux_result = train_auxiliary_classifier(
        train_latents, train_labels, val_latents, val_labels,
        test_latents, test_labels, cfg, num_classes, device, logger,
    )
    result["aux_accuracy"] = aux_result["test_accuracy"]
    result["aux_result"] = aux_result

    if is_baseline:
        # Save full aux report
        aux_save = {k: v for k, v in aux_result.items() if k != "model"}
        (report_dir / "auxiliary_classifier_report.json").write_text(
            json.dumps(aux_save, indent=2, default=str), encoding="utf-8")
        plot_confusion_matrix(
            np.array(aux_result["confusion_matrix"]), class_names,
            plot_dir / "confusion_matrix.png", cfg.dpi)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Single Font Single Letter VAE")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_Rohit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp1_single_font_single_letter"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--kl-warmup", type=int, default=25)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta_max=args.beta_max,
        kl_warmup_epochs=args.kl_warmup,
        early_stopping_patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Create directory hierarchy
    report_dir = cfg.output_dir / "reports"
    model_dir = cfg.output_dir / "model"
    plot_dir = cfg.output_dir / "plots"
    log_dir = cfg.output_dir / "logs"
    for d in (report_dir, model_dir, plot_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Single Font, Single Letter — Full VAE Pipeline")
    logger.info("=" * 70)

    # Save config
    cfg_path = cfg.output_dir / "experiment_config.json"
    cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"Config saved to {cfg_path}")

    set_seed(cfg.seed)

    # ── Step 1: Collect & audit dataset ──
    logger.info("Step 1: Dataset collection & quality audit")
    records = collect_records(cfg)
    if not records:
        logger.error(f"No images found under {cfg.dataset_root}!")
        sys.exit(1)
    logger.info(f"Collected {len(records)} images")

    audit = dataset_quality_audit(records, cfg.image_size)
    (report_dir / "dataset_quality_report.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Dataset audit saved")

    # ── Step 2: Split ──
    logger.info("Step 2: Stratified train/val/test split")
    train_recs, val_recs, test_recs = stratified_split(
        records, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    logger.info(f"Split: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # ── Dataloaders ──
    ds_args = dict(image_size=cfg.image_size, point_sizes=cfg.point_sizes)
    train_ds = GlyphDataset(train_recs, augment=True, **ds_args)
    val_ds = GlyphDataset(val_recs, augment=False, **ds_args)
    test_ds = GlyphDataset(test_recs, augment=False, **ds_args)
    all_ds = GlyphDataset(records, augment=False, **ds_args)

    dl_args = dict(batch_size=cfg.batch_size, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_args)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_args)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_args)
    all_loader = DataLoader(all_ds, shuffle=False, **dl_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Steps 3-11: Baseline (depth=4) ──
    logger.info("Steps 3-11: Training baseline model (depth=4)")
    baseline = run_single_depth(
        depth=cfg.base_depth, cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, all_loader=all_loader,
        device=device, logger=logger,
        plot_dir=plot_dir, log_dir=log_dir,
        model_dir=model_dir, report_dir=report_dir,
        is_baseline=True,
    )

    # Save baseline metrics
    baseline_metrics_save = {
        "config": cfg.to_dict(),
        "split": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
        "best_epoch": baseline["best_epoch"],
        "param_count": baseline["param_count"],
        "test_metrics": baseline["metrics"],
    }
    (report_dir / "metrics_summary.json").write_text(
        json.dumps(baseline_metrics_save, indent=2), encoding="utf-8")
    write_evaluation_matrix(baseline["metrics"], report_dir / "evaluation_matrix.md")

    # ── Step 12: Progressive depth scaling ──
    logger.info("Step 12: Progressive depth scaling")
    depth_comparison: dict[str, Any] = {
        str(cfg.base_depth): {
            "metrics": baseline["metrics"],
            "aux_accuracy": baseline["aux_accuracy"],
            "param_count": baseline["param_count"],
        }
    }

    for depth in cfg.depth_variants:
        logger.info(f"Training depth variant: {depth}")
        variant = run_single_depth(
            depth=depth, cfg=cfg,
            train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, all_loader=all_loader,
            device=device, logger=logger,
            plot_dir=plot_dir, log_dir=log_dir,
            model_dir=model_dir, report_dir=report_dir,
            is_baseline=False,
        )
        depth_comparison[str(depth)] = {
            "metrics": variant["metrics"],
            "aux_accuracy": variant["aux_accuracy"],
            "param_count": variant["param_count"],
        }

    write_depth_comparison(depth_comparison, report_dir / "depth_scaling_comparison.md")

    # ── Final summary ──
    logger.info("Compiling experiment summary")
    summary_data = {
        "config": cfg.to_dict(),
        "dataset_audit": audit,
        "baseline_metrics": baseline["metrics"],
        "gaussian_analysis": baseline.get("gaussian_analysis", {}),
        "auxiliary_classifier": {
            k: v for k, v in baseline["aux_result"].items() if k != "model"
        },
        "depth_comparison": depth_comparison,
    }
    write_experiment_summary(summary_data, report_dir / "experiment_1_summary.md")

    # ── Artifact manifest ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1 COMPLETE — Artifact Manifest:")
    logger.info("=" * 70)
    for root_d in (report_dir, model_dir, plot_dir, log_dir):
        for f in sorted(root_d.rglob("*")):
            if f.is_file():
                logger.info(f"  {f.relative_to(cfg.output_dir)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
