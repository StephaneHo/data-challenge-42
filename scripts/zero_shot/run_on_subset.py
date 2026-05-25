"""Sanity-check the zero-shot pipeline on a small training subset.

For each of N images:
  - Run face parsing
  - Extract features
  - Compute uncalibrated 'raw' predictions (ratio_hull)
  - Compare against the GT FaceOcclusion column
  - Output Pearson correlation, scatter plot, sample seg visualizations

Usage:
    python scripts/zero_shot/run_on_subset.py --n 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot.parser import FaceParser  # noqa: E402
from src.zero_shot.pipeline import extract_features_from_paths  # noqa: E402

OUT_DIR = REPO_ROOT / "figures" / "zero_shot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--n", type=int, default=100, help="number of train samples to evaluate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    sample = train_csv.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(sample)} train images")

    print("loading SegFormer face-parsing model (may download ~50MB at first run)...")
    parser = FaceParser()
    print(f"model on {parser.device}")

    feats = extract_features_from_paths(parser, Path(args.image_dir), sample["filename"].tolist(),
                                        batch_size=args.batch_size)
    feats["gt"] = sample["FaceOcclusion"].values
    feats["gender"] = sample["gender"].values

    # Print summary stats
    print("\nFeature stats (head):")
    print(feats.describe().T[["mean", "std", "min", "max"]].round(4).to_string())

    # Pearson correlations against GT
    print("\nCorrelation with GT FaceOcclusion:")
    corrs = {}
    for col in feats.columns:
        if col in ("filename", "gt", "gender"):
            continue
        r = np.corrcoef(feats[col], feats["gt"])[0, 1]
        corrs[col] = r
        print(f"  {col:<32s}  r = {r:+.4f}")

    # Scatter plot best feature vs GT
    best_feat = max(corrs, key=lambda k: corrs[k])
    print(f"\nbest feature: {best_feat} (r={corrs[best_feat]:.3f})")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(feats[best_feat], feats["gt"], s=8, alpha=0.5)
    ax.set_xlabel(f"raw zero-shot: {best_feat}")
    ax.set_ylabel("ground truth FaceOcclusion")
    ax.set_title(f"Zero-shot raw vs GT (n={len(feats)}, r={corrs[best_feat]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlim(0, max(0.5, feats[best_feat].max() * 1.05))
    ax.set_ylim(0, max(0.5, feats["gt"].max() * 1.05))
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "scatter_raw_vs_gt.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")

    # Save the features CSV for downstream calibration tuning
    out_csv = OUT_DIR / "subset_features.csv"
    feats.to_csv(out_csv, index=False)
    print(f"saved {out_csv}  ({len(feats)} rows)")


if __name__ == "__main__":
    main()
