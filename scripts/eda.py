"""Exploratory data analysis for the occlusion challenge.

Run from repo root:
    python scripts/eda.py

Saves figures to figures/ and prints key statistics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "occlusion_datasets"
IMG_DIR = REPO_ROOT / "crops"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from src.metric import GENDER_FEMALE, GENDER_MALE, sample_weight  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_students.csv")
    return train, test


def basic_stats(train: pd.DataFrame, test: pd.DataFrame) -> None:
    banner("BASIC STATS")
    print(f"train: {len(train)} rows, {train.isna().sum().to_dict()} NaN per col")
    print(f"test : {len(test)} rows, {test.isna().sum().to_dict()} NaN per col")
    print(f"\ntrain.FaceOcclusion describe:\n{train['FaceOcclusion'].describe()}")
    print(f"\ntrain.gender value_counts:\n{train['gender'].value_counts(dropna=False)}")

    db_train = train["filename"].str.split("/").str[0].value_counts()
    db_test = test["filename"].str.split("/").str[0].value_counts()
    print(f"\ndatabase split (train):\n{db_train}")
    print(f"\ndatabase split (test):\n{db_test}")


def occlusion_histograms(train: pd.DataFrame, test: pd.DataFrame) -> None:
    banner("OCCLUSION DISTRIBUTION")
    train_clean = train.dropna(subset=["FaceOcclusion"])
    bins = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    axes[0].hist(train_clean["FaceOcclusion"], bins=bins, color="steelblue")
    axes[0].set_title(f"Train FaceOcclusion (n={len(train_clean)})")
    axes[0].set_xlabel("Occlusion ratio")
    axes[0].set_ylabel("Count")
    if "FaceOcclusion" in test.columns:
        test_clean = test.dropna(subset=["FaceOcclusion"])
        axes[1].hist(test_clean["FaceOcclusion"], bins=bins, color="indianred")
        axes[1].set_title(f"Test FaceOcclusion (n={len(test_clean)})")
    else:
        axes[1].text(0.5, 0.5, "No FaceOcclusion column in test_students.csv\n(labels hidden by organizers)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Test (no labels available)")
    axes[1].set_xlabel("Occlusion ratio")
    fig.tight_layout()
    out = FIG_DIR / "01_occlusion_hist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    bins_coarse = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    labels = [f"[{a:.2f},{b:.2f})" for a, b in zip(bins_coarse[:-1], bins_coarse[1:])]
    train_clean = train_clean.assign(occ_bin=pd.cut(train_clean["FaceOcclusion"], bins=bins_coarse, labels=labels, include_lowest=True))
    print(f"\ntrain occlusion bins (coarse):\n{train_clean['occ_bin'].value_counts().sort_index()}")


def gender_breakdown(train: pd.DataFrame) -> None:
    banner("GENDER × OCCLUSION")
    df = train.dropna(subset=["FaceOcclusion", "gender"]).copy()
    bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    labels = [f"[{a:.2f},{b:.2f})" for a, b in zip(bins[:-1], bins[1:])]
    df["occ_bin"] = pd.cut(df["FaceOcclusion"], bins=bins, labels=labels, include_lowest=True)
    cross = pd.crosstab(df["occ_bin"], df["gender"])
    cross.columns = ["female (0)" if c == GENDER_FEMALE else "male (1)" for c in cross.columns]
    print(f"counts:\n{cross}")
    print(f"\nfemale mean occlusion: {df[df.gender == GENDER_FEMALE]['FaceOcclusion'].mean():.4f}")
    print(f"male   mean occlusion: {df[df.gender == GENDER_MALE]['FaceOcclusion'].mean():.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    bins_h = np.linspace(0, 0.6, 41)
    ax.hist(df[df.gender == GENDER_FEMALE]["FaceOcclusion"], bins=bins_h, alpha=0.5, label="female", color="palevioletred")
    ax.hist(df[df.gender == GENDER_MALE]["FaceOcclusion"], bins=bins_h, alpha=0.5, label="male", color="steelblue")
    ax.set_xlabel("Occlusion ratio")
    ax.set_ylabel("Count")
    ax.set_title("Train occlusion distribution by gender")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "02_gender_occlusion.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nsaved {out}")


def metric_weight_demo() -> None:
    banner("METRIC WEIGHT FUNCTION  w = 1/30 + GT")
    gts = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"GT     : {gts}")
    print(f"weight : {sample_weight(gts).round(4)}")
    print(f"ratio vs GT=0: {(sample_weight(gts) / sample_weight(np.array(0.0))).round(2)}")


def check_image_paths(train: pd.DataFrame, n: int = 20) -> None:
    banner(f"SANITY CHECK ON {n} RANDOM IMAGE PATHS")
    sample = train.dropna().sample(n, random_state=0)
    missing = []
    sizes = []
    for fn in sample["filename"]:
        p = IMG_DIR / fn
        if not p.exists():
            missing.append(fn)
            continue
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception as e:
            missing.append(f"{fn} (open failed: {e})")
    print(f"missing/unreadable: {len(missing)} / {n}")
    for m in missing[:5]:
        print(f"  - {m}")
    if sizes:
        sizes = np.array(sizes)
        print(f"image sizes (W, H): unique = {set(map(tuple, sizes))}")


def main() -> None:
    train, test = load_dataframes()
    basic_stats(train, test)
    occlusion_histograms(train, test)
    gender_breakdown(train)
    metric_weight_demo()
    check_image_paths(train)
    print(f"\nfigures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
