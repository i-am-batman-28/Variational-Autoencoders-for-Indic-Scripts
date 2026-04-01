#!/usr/bin/env python3
"""End-to-end VAE pipeline for letter 'a' from lohit_telugu and pothana2000.

Outputs:
- Dataset quality reports
- Trained VAE checkpoint
- Training curves
- 2D latent-space plot
- Latent traversal and random generation samples
- Reconstruction quality metrics and evaluation matrix
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy.ndimage import laplace
from torch import nn
from torch.utils.data import DataLoader, Dataset


SOURCE_FONTS = ("lohit_telugu", "pothana2000")
TARGET_LETTER = "a"
TARGET_POINT_SIZES = ("10pt", "14pt", "18pt", "22pt")


@dataclass
class ImageRecord:
    path: Path
    source_font: str
    point_size: str


@dataclass
class ExperimentConfig:
    dataset_root: Path
    output_dir: Path
    image_size: int = 64
    batch_size: int = 16
    latent_dim: int = 2
    lr: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 20
    beta_max: float = 1.0
    kl_warmup_epochs: int = 20
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    recon_log_interval: int = 10  # save val recon grid every N epochs (0 = off)


class GlyphDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment

        self.source_to_idx = {name: i for i, name in enumerate(SOURCE_FONTS)}
        self.size_to_idx = {name: i for i, name in enumerate(TARGET_POINT_SIZES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        image = load_image_as_grayscale(rec.path, self.image_size)

        if self.augment:
            image = apply_augmentation(image)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        source_idx = torch.tensor(self.source_to_idx[rec.source_font], dtype=torch.long)
        size_idx = torch.tensor(self.size_to_idx[rec.point_size], dtype=torch.long)

        return tensor, source_idx, size_idx


class ConvVAE(nn.Module):
    """Conv VAE that supports 64x64 or 128x128 via encoder_spatial = image_size // 16."""

    def __init__(self, latent_dim: int = 2, image_size: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder_spatial = image_size // 16  # 4 for 64, 8 for 128

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_out_dim = 256 * self.encoder_spatial * self.encoder_spatial
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.encoder_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.encoder_spatial, self.encoder_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        out = out.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=255,
        )

    if random.random() < 0.85:
        tx = random.uniform(-3.0, 3.0)
        ty = random.uniform(-3.0, 3.0)
        out = out.transform(
            out.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, tx, 0.0, 1.0, ty),
            resample=Image.Resampling.BICUBIC,
            fillcolor=255,
        )

    if random.random() < 0.5:
        c = random.uniform(0.85, 1.20)
        out = ImageEnhance.Contrast(out).enhance(c)

    return out


def collect_records(dataset_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for source in SOURCE_FONTS:
        for pt in TARGET_POINT_SIZES:
            target_dir = dataset_root / source / TARGET_LETTER / pt
            if not target_dir.is_dir():
                continue

            for f in sorted(target_dir.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() != ".png":
                    continue
                records.append(ImageRecord(path=f, source_font=source, point_size=pt))

    return records


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def dataset_quality_audit(
    records: list[ImageRecord],
    image_size: int,
) -> dict[str, Any]:
    corrupt_files: list[str] = []
    mode_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()

    hashes: Counter[str] = Counter()
    hash_to_files: defaultdict[str, list[str]] = defaultdict(list)

    intensity_means: list[float] = []
    intensity_stds: list[float] = []
    non_white_ratios: list[float] = []
    sharpness_values: list[float] = []
    center_x_values: list[float] = []
    center_y_values: list[float] = []

    by_source_counts: Counter[str] = Counter()
    by_source_size_counts: Counter[str] = Counter()

    by_source_non_white: defaultdict[str, list[float]] = defaultdict(list)
    by_source_sharpness: defaultdict[str, list[float]] = defaultdict(list)

    for rec in records:
        by_source_counts[rec.source_font] += 1
        by_source_size_counts[f"{rec.source_font}:{rec.point_size}"] += 1

        try:
            with Image.open(rec.path) as img_raw:
                mode_counter[img_raw.mode] += 1
                resolution_counter[f"{img_raw.width}x{img_raw.height}"] += 1

            img = load_image_as_grayscale(rec.path, image_size)
            arr = np.asarray(img, dtype=np.float32)

        except Exception:
            corrupt_files.append(str(rec.path))
            continue

        h = hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()
        hashes[h] += 1
        hash_to_files[h].append(str(rec.path))

        intensity_means.append(float(arr.mean() / 255.0))
        intensity_stds.append(float(arr.std() / 255.0))

        non_white = float((arr < 245.0).mean())
        non_white_ratios.append(non_white)

        sharpness = float(np.var(laplace(arr)))
        sharpness_values.append(sharpness)

        by_source_non_white[rec.source_font].append(non_white)
        by_source_sharpness[rec.source_font].append(sharpness)

        ink = np.clip(255.0 - arr, a_min=0.0, a_max=None)
        mass = float(ink.sum())
        if mass > 0.0:
            ys, xs = np.indices(arr.shape)
            cx = float((xs * ink).sum() / mass)
            cy = float((ys * ink).sum() / mass)
            center_x_values.append(cx)
            center_y_values.append(cy)

    duplicate_groups = [v for v in hash_to_files.values() if len(v) > 1]

    report: dict[str, Any] = {
        "target_scope": {
            "fonts": list(SOURCE_FONTS),
            "letter": TARGET_LETTER,
            "point_sizes": list(TARGET_POINT_SIZES),
            "resized_to": f"{image_size}x{image_size}",
        },
        "overall": {
            "total_records_scanned": len(records),
            "corrupt_file_count": len(corrupt_files),
            "corrupt_files": corrupt_files,
            "mode_distribution": dict(mode_counter),
            "resolution_distribution": dict(resolution_counter),
            "duplicate_hash_groups": len(duplicate_groups),
            "duplicate_file_count": int(sum(len(g) for g in duplicate_groups)),
            "unique_hash_count": int(sum(1 for c in hashes.values() if c == 1)),
        },
        "counts": {
            "by_source": dict(by_source_counts),
            "by_source_and_size": dict(by_source_size_counts),
        },
        "pixel_statistics": {
            "intensity_mean": summarize_numeric(intensity_means),
            "intensity_std": summarize_numeric(intensity_stds),
            "non_white_ratio": summarize_numeric(non_white_ratios),
            "sharpness_laplacian_var": summarize_numeric(sharpness_values),
            "center_of_mass_x": summarize_numeric(center_x_values),
            "center_of_mass_y": summarize_numeric(center_y_values),
        },
        "by_source_features": {
            source: {
                "non_white_ratio": summarize_numeric(by_source_non_white[source]),
                "sharpness_laplacian_var": summarize_numeric(by_source_sharpness[source]),
            }
            for source in SOURCE_FONTS
        },
        "duplicate_groups": duplicate_groups,
    }

    return report


def write_dataset_quality_summary(report: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Dataset Quality Summary (Letter a)")
    lines.append("")

    lines.append("## Scope")
    lines.append(f"- Fonts: {', '.join(report['target_scope']['fonts'])}")
    lines.append(f"- Letter: {report['target_scope']['letter']}")
    lines.append(f"- Point sizes: {', '.join(report['target_scope']['point_sizes'])}")
    lines.append(f"- Analysis resize: {report['target_scope']['resized_to']}")
    lines.append("")

    overall = report["overall"]
    lines.append("## Integrity")
    lines.append(f"- Total records scanned: {overall['total_records_scanned']}")
    lines.append(f"- Corrupt files: {overall['corrupt_file_count']}")
    lines.append(f"- Duplicate hash groups: {overall['duplicate_hash_groups']}")
    lines.append(f"- Duplicate file count: {overall['duplicate_file_count']}")
    lines.append("")

    counts = report["counts"]
    lines.append("## Counts by Source")
    for k, v in counts["by_source"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Counts by Source and Size")
    for k, v in sorted(counts["by_source_and_size"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    pix = report["pixel_statistics"]
    lines.append("## Pixel Feature Summary")
    lines.append(
        "- Intensity mean: "
        f"mean={pix['intensity_mean']['mean']:.4f}, "
        f"std={pix['intensity_mean']['std']:.4f}"
    )
    lines.append(
        "- Non-white ratio: "
        f"mean={pix['non_white_ratio']['mean']:.4f}, "
        f"std={pix['non_white_ratio']['std']:.4f}"
    )
    lines.append(
        "- Sharpness (Laplacian variance): "
        f"mean={pix['sharpness_laplacian_var']['mean']:.4f}, "
        f"std={pix['sharpness_laplacian_var']['std']:.4f}"
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("- This report focuses only on lohit_telugu/a and pothana2000/a.")
    lines.append("- Duplicates are detected using md5 hash of resized grayscale tensors.")
    lines.append("- Sharpness is estimated via variance of image Laplacian.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    groups: defaultdict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.source_font, rec.point_size)].append(rec)

    rng = random.Random(seed)

    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    test: list[ImageRecord] = []

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
        val.extend(g[n_train : n_train + n_val])
        test.extend(g[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = bce + beta * kld

    stats = {
        "bce": float(bce.detach().item()),
        "kld": float(kld.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, stats


def evaluate_epoch(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()

    bce_vals: list[float] = []
    kld_vals: list[float] = []
    total_vals: list[float] = []

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            _, stats = vae_loss(recon, x, mu, logvar, beta)
            bce_vals.append(stats["bce"])
            kld_vals.append(stats["kld"])
            total_vals.append(stats["total"])

    return {
        "bce": float(np.mean(bce_vals)) if bce_vals else math.nan,
        "kld": float(np.mean(kld_vals)) if kld_vals else math.nan,
        "total": float(np.mean(total_vals)) if total_vals else math.nan,
    }


def train_vae(
    model: ConvVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    history: dict[str, list[float]] = {
        "train_bce": [],
        "train_kld": [],
        "train_total": [],
        "val_bce": [],
        "val_kld": [],
        "val_total": [],
        "beta": [],
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(config.epochs):
        model.train()

        beta = config.beta_max
        if config.kl_warmup_epochs > 0:
            warm = (epoch + 1) / float(config.kl_warmup_epochs)
            beta = min(config.beta_max, config.beta_max * warm)

        train_bce_vals: list[float] = []
        train_kld_vals: list[float] = []
        train_total_vals: list[float] = []

        for x, _, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, logvar = model(x)
            loss, stats = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_bce_vals.append(stats["bce"])
            train_kld_vals.append(stats["kld"])
            train_total_vals.append(stats["total"])

        train_epoch = {
            "bce": float(np.mean(train_bce_vals)),
            "kld": float(np.mean(train_kld_vals)),
            "total": float(np.mean(train_total_vals)),
        }
        val_epoch = evaluate_epoch(model, val_loader, device, beta)

        history["train_bce"].append(train_epoch["bce"])
        history["train_kld"].append(train_epoch["kld"])
        history["train_total"].append(train_epoch["total"])
        history["val_bce"].append(val_epoch["bce"])
        history["val_kld"].append(val_epoch["kld"])
        history["val_total"].append(val_epoch["total"])
        history["beta"].append(beta)

        print(
            f"Epoch {epoch + 1:03d}/{config.epochs} | "
            f"beta={beta:.3f} | "
            f"train_total={train_epoch['total']:.4f} | "
            f"val_total={val_epoch['total']:.4f}"
        )

        if val_epoch["total"] < best_val:
            best_val = val_epoch["total"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "best_val_total": best_val,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        if config.recon_log_interval > 0 and (epoch + 1) % config.recon_log_interval == 0:
            recon_path = config.output_dir / f"recon_epoch_{epoch + 1:03d}.png"
            model.eval()
            save_reconstructions(model, val_loader, device, recon_path, max_items=10)
            model.train()

        scheduler.step()

        if bad_epochs >= config.early_stopping_patience:
            print(
                "Early stopping triggered "
                f"at epoch {epoch + 1}, best epoch was {best_epoch + 1}."
            )
            break

    return history


def compute_ssim_global(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = 0.01**2
    c2 = 0.03**2

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

    return float(num / den) if den != 0 else 0.0


def evaluate_model(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()

    bce_vals: list[float] = []
    kld_vals: list[float] = []
    total_vals: list[float] = []

    mse_vals: list[float] = []
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            _, stats = vae_loss(recon, x, mu, logvar, beta)

            bce_vals.append(stats["bce"])
            kld_vals.append(stats["kld"])
            total_vals.append(stats["total"])

            x_np = x.detach().cpu().numpy()
            r_np = recon.detach().cpu().numpy()

            for i in range(x_np.shape[0]):
                xi = x_np[i, 0]
                ri = r_np[i, 0]

                mse = float(np.mean((xi - ri) ** 2))
                mse_vals.append(mse)

                psnr = 20.0 * math.log10(1.0 / math.sqrt(mse + 1e-12))
                psnr_vals.append(float(psnr))

                ssim_vals.append(compute_ssim_global(xi, ri))

    return {
        "bce": float(np.mean(bce_vals)) if bce_vals else math.nan,
        "kld": float(np.mean(kld_vals)) if kld_vals else math.nan,
        "total": float(np.mean(total_vals)) if total_vals else math.nan,
        "mse": float(np.mean(mse_vals)) if mse_vals else math.nan,
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else math.nan,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else math.nan,
    }


def save_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
    keys = list(history.keys())
    rows = max(len(v) for v in history.values()) if history else 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])

        for i in range(rows):
            row = [i + 1]
            for k in keys:
                row.append(history[k][i] if i < len(history[k]) else "")
            writer.writerow(row)


def plot_loss_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train ELBO", linewidth=2)
    plt.plot(epochs, history["val_total"], label="Val ELBO", linewidth=2)
    plt.plot(epochs, history["beta"], label="Beta (KL weight)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Progress (Letter a)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def collect_latents(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    latents: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    with torch.no_grad():
        for x, src, sz in loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            latents.append(mu.detach().cpu().numpy())
            sources.append(src.numpy())
            sizes.append(sz.numpy())

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(sources, axis=0),
        np.concatenate(sizes, axis=0),
    )


def plot_latent_space_2d(
    latents: np.ndarray,
    source_ids: np.ndarray,
    size_ids: np.ndarray,
    out_path: Path,
) -> None:
    if latents.shape[1] < 2:
        raise ValueError("Latent dimension must be at least 2 for 2D plotting.")

    colors = {0: "#1463A5", 1: "#F17300"}
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    plt.figure(figsize=(8, 6))
    for src_idx, src_name in enumerate(SOURCE_FONTS):
        for size_idx, size_name in enumerate(TARGET_POINT_SIZES):
            mask = (source_ids == src_idx) & (size_ids == size_idx)
            if not np.any(mask):
                continue

            plt.scatter(
                latents[mask, 0],
                latents[mask, 1],
                s=32,
                alpha=0.8,
                c=colors[src_idx],
                marker=markers[size_idx],
                label=f"{src_name}-{size_name}",
                edgecolors="none",
            )

    plt.axhline(0.0, color="gray", linewidth=0.7, alpha=0.5)
    plt.axvline(0.0, color="gray", linewidth=0.7, alpha=0.5)
    plt.title("2D Latent Space for Letter a")
    plt.xlabel("z1")
    plt.ylabel("z2")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_reconstructions(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    max_items: int = 10,
) -> None:
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))
        x, _, _ = batch
        x = x.to(device)
        recon, _, _ = model(x)

    x_np = x.detach().cpu().numpy()[:max_items, 0]
    r_np = recon.detach().cpu().numpy()[:max_items, 0]

    n = x_np.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.2))

    for i in range(n):
        axes[0, i].imshow(x_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        axes[0, i].set_title("Original", fontsize=8)

        axes[1, i].imshow(r_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstruction", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_latent_traversal_grid(
    model: ConvVAE,
    device: torch.device,
    latent_dim: int,
    out_path: Path,
    grid_size: int = 12,
    zmin: float = -3.0,
    zmax: float = 3.0,
) -> None:
    model.eval()
    out_size = getattr(model, "image_size", 64)

    values = np.linspace(zmin, zmax, grid_size, dtype=np.float32)
    canvas = np.zeros((grid_size * out_size, grid_size * out_size), dtype=np.float32)

    with torch.no_grad():
        for iy, zy in enumerate(values):
            for ix, zx in enumerate(values):
                z = torch.zeros((1, latent_dim), device=device)
                z[0, 0] = float(zx)
                if latent_dim > 1:
                    z[0, 1] = float(zy)

                decoded = model.decode(z).detach().cpu().numpy()[0, 0]
                y0 = iy * out_size
                x0 = ix * out_size
                canvas[y0 : y0 + out_size, x0 : x0 + out_size] = decoded

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Latent Traversal Grid (z1, z2)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_random_samples(
    model: ConvVAE,
    device: torch.device,
    latent_dim: int,
    out_path: Path,
    n_rows: int = 4,
    n_cols: int = 4,
) -> None:
    model.eval()

    n = n_rows * n_cols
    with torch.no_grad():
        z = torch.randn((n, latent_dim), device=device)
        imgs = model.decode(z).detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].imshow(imgs[idx], cmap="gray", vmin=0.0, vmax=1.0)
            axes[r, c].axis("off")
            idx += 1

    plt.suptitle("Random Samples from Latent Space")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_evaluation_matrix(metrics: dict[str, float], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Metrics Matrix (Elution Matrix Interpretation)")
    lines.append("")
    lines.append("This table interprets the request for an \"elution matrix\" as a compact evaluation metrics matrix for VAE quality.")
    lines.append("")
    lines.append("| Quality Dimension | Metric | Value | Interpretation Goal |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| Reconstruction fidelity | MSE | {metrics['mse']:.6f} | Lower is better |")
    lines.append(f"| Reconstruction fidelity | PSNR (dB) | {metrics['psnr']:.4f} | Higher is better |")
    lines.append(f"| Structural similarity | SSIM | {metrics['ssim']:.4f} | Higher is better |")
    lines.append(f"| Probabilistic fit | BCE | {metrics['bce']:.4f} | Lower is better |")
    lines.append(f"| Latent regularization | KL divergence | {metrics['kld']:.4f} | Balanced with BCE |")
    lines.append(f"| Total objective | ELBO surrogate | {metrics['total']:.4f} | Lower is better |")
    lines.append("")
    lines.append("## Extensions for larger datasets")
    lines.append("- Distribution-level metrics: FID, KID")
    lines.append("- Fidelity-coverage tradeoff: precision and recall for generative models")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAE experiment for letter a")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset_Rohit"),
        help="Path to dataset_Rohit directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vae_letter_a"),
        help="Directory for reports, model, and plots",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--recon-log-interval",
        type=int,
        default=10,
        help="Save validation recon grid every N epochs (0=off)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Skip training and reuse existing checkpoint if available",
    )
    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        beta_max=args.beta_max,
        kl_warmup_epochs=args.kl_warmup_epochs,
        seed=args.seed,
        recon_log_interval=args.recon_log_interval,
    )

    set_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = config.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(config.dataset_root)
    if not records:
        raise RuntimeError(
            f"No target images found under {config.dataset_root}. "
            "Expected folders for lohit_telugu/a and pothana2000/a."
        )

    report = dataset_quality_audit(records, config.image_size)
    quality_json_path = config.output_dir / "dataset_quality_report.json"
    quality_md_path = config.output_dir / "dataset_quality_summary.md"
    quality_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_dataset_quality_summary(report, quality_md_path)

    train_records, val_records, test_records = stratified_split(
        records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    print(
        "Split sizes: "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )

    train_ds = GlyphDataset(train_records, image_size=config.image_size, augment=True)
    val_ds = GlyphDataset(val_records, image_size=config.image_size, augment=False)
    test_ds = GlyphDataset(test_records, image_size=config.image_size, augment=False)
    all_ds = GlyphDataset(records, image_size=config.image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    all_loader = DataLoader(
        all_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvVAE(latent_dim=config.latent_dim, image_size=config.image_size).to(device)
    ckpt_path = model_dir / "best_vae_letter_a.pt"

    history: dict[str, list[float]] = {
        "train_bce": [],
        "train_kld": [],
        "train_total": [],
        "val_bce": [],
        "val_kld": [],
        "val_total": [],
        "beta": [],
    }

    if args.reuse_checkpoint and ckpt_path.is_file():
        print(f"Reusing checkpoint and skipping training: {ckpt_path}")
    else:
        history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_path=ckpt_path,
        )

    if not ckpt_path.is_file():
        raise RuntimeError("Training ended without producing a checkpoint.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    final_beta = config.beta_max
    metrics = evaluate_model(model, test_loader, device, final_beta)

    history_csv_path = config.output_dir / "training_history.csv"
    loss_plot = config.output_dir / "loss_curve.png"
    if len(history["train_total"]) > 0:
        save_history_csv(history, history_csv_path)
        plot_loss_curves(history, loss_plot)

    latents, source_ids, size_ids = collect_latents(model, all_loader, device)
    latent_plot = config.output_dir / "latent_space_2d.png"
    plot_latent_space_2d(latents, source_ids, size_ids, latent_plot)

    recon_plot = config.output_dir / "reconstructions.png"
    save_reconstructions(model, test_loader, device, recon_plot, max_items=10)

    traversal_plot = config.output_dir / "latent_traversal_grid.png"
    save_latent_traversal_grid(
        model,
        device,
        latent_dim=config.latent_dim,
        out_path=traversal_plot,
        grid_size=12,
        zmin=-3.0,
        zmax=3.0,
    )

    random_plot = config.output_dir / "random_samples.png"
    save_random_samples(
        model,
        device,
        latent_dim=config.latent_dim,
        out_path=random_plot,
        n_rows=4,
        n_cols=4,
    )

    metrics_summary_path = config.output_dir / "metrics_summary.json"
    metrics_payload = {
        "config": {
            "dataset_root": str(config.dataset_root),
            "output_dir": str(config.output_dir),
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "lr": config.lr,
            "beta_max": config.beta_max,
            "kl_warmup_epochs": config.kl_warmup_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "seed": config.seed,
            "recon_log_interval": config.recon_log_interval,
        },
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "test_metrics": metrics,
        "best_checkpoint_epoch": int(ckpt.get("epoch", -1)) + 1,
        "best_val_total": float(ckpt.get("best_val_total", math.nan)),
    }
    metrics_summary_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    eval_matrix_path = config.output_dir / "evaluation_matrix.md"
    write_evaluation_matrix(metrics, eval_matrix_path)

    print("\nExperiment complete. Artifacts:")
    print(f"- Dataset quality report: {quality_json_path}")
    print(f"- Dataset quality summary: {quality_md_path}")
    print(f"- Checkpoint: {ckpt_path}")
    print(f"- Loss curve: {loss_plot}")
    print(f"- Latent plot: {latent_plot}")
    print(f"- Reconstructions: {recon_plot}")
    print(f"- Latent traversal: {traversal_plot}")
    print(f"- Random samples: {random_plot}")
    print(f"- Metrics summary: {metrics_summary_path}")
    print(f"- Evaluation matrix: {eval_matrix_path}")


if __name__ == "__main__":
    main()
