"""Fit a calibrator on (raw features, GT occlusion) extracted from train images.

Workflow:
  1. Sample N images from train.csv
  2. Parse + extract features
  3. Fit calibrator (linear by default) on (features, GT)
  4. Save calibrator + a calibration report (R^2, coefficients)

Usage:
    python scripts/zero_shot/fit_calibration.py --n 1000 --mode linear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot.calibrator import OcclusionCalibrator  # noqa: E402
from src.zero_shot.parser import FaceParser  # noqa: E402
from src.zero_shot.pipeline import extract_features_from_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="linear", choices=["linear", "isotonic"])
    p.add_argument("--feature-for-isotonic", default="ratio_hull")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=str(REPO_ROOT / "src" / "zero_shot" / "calibrator.pkl"))
    p.add_argument("--features-out", default=str(REPO_ROOT / "eval" / "zero_shot_train_features.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    sample = train_csv.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(sample)} train rows")

    parser = FaceParser()
    feats = extract_features_from_paths(parser, Path(args.image_dir), sample["filename"].tolist(),
                                        batch_size=args.batch_size)
    feats["FaceOcclusion"] = sample["FaceOcclusion"].values
    feats["gender"] = sample["gender"].values

    features_out = Path(args.features_out)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(features_out, index=False)
    print(f"saved features to {features_out}")

    cal = OcclusionCalibrator(mode=args.mode, feature_for_isotonic=args.feature_for_isotonic,
                              ridge_alpha=args.ridge_alpha)
    cal.fit(feats, feats["FaceOcclusion"].values)
    print(f"calibrator R^2 on train sample: {cal.train_r2_:.4f}")
    if args.mode == "linear":
        print("coefficients:")
        for name, coef in zip(cal.feature_names_, cal.model.coef_):
            print(f"  {name:<32s}  {coef:+.4f}")
        print(f"intercept: {cal.model.intercept_:+.4f}")

    cal.save(Path(args.out))
    print(f"saved calibrator to {args.out}")


if __name__ == "__main__":
    main()
