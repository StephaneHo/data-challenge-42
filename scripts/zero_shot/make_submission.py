"""Generate a hfactory-format submission CSV from cached test features.

Reads the test feature cache produced by `cache_val_features.py --source test`,
applies a named heuristic (optionally with TTA flip averaging from a flipped
cache), and writes a CSV with the format hfactory expects:
    filename, FaceOcclusion, gender='x'

Usage:
    # Without TTA
    python scripts/zero_shot/make_submission.py \\
        --features eval/cache/test_segformer_features.csv \\
        --heuristic simple_hull_scaled \\
        --out results/zero_shot_tf/test_predictions.csv

    # With TTA flip averaging
    python scripts/zero_shot/make_submission.py \\
        --features eval/cache/test_segformer_features.csv \\
        --features-flipped eval/cache/test_segformer_features_flip.csv \\
        --heuristic simple_hull_scaled \\
        --out results/zero_shot_tf/test_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot.heuristics import HEURISTICS, apply_heuristic, apply_with_tta  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="Test feature cache CSV (produced by cache_val_features.py --source test)")
    p.add_argument("--features-flipped", default=None,
                   help="Optional flipped-test feature cache for TTA averaging")
    p.add_argument("--heuristic", required=True, choices=list(HEURISTICS))
    p.add_argument("--out", required=True, help="Output submission CSV path")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"),
                   help="To verify row count and filename order against test_students.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feats = pd.read_csv(args.features)
    print(f"loaded {len(feats)} test feature rows")

    if args.features_flipped:
        feats_flipped = pd.read_csv(args.features_flipped)
        feats_flipped = feats_flipped.set_index("filename").reindex(feats["filename"]).reset_index()
        pred = apply_with_tta(args.heuristic, feats, feats_flipped)
        print(f"applied {args.heuristic} + TTA flip averaging")
    else:
        pred = apply_heuristic(args.heuristic, feats)
        print(f"applied {args.heuristic}")

    sub = pd.DataFrame({
        "filename": feats["filename"],
        "FaceOcclusion": pred,
        "gender": "x",
    })

    # Sanity vs test_students.csv
    test_csv = pd.read_csv(Path(args.data_dir) / "test_students.csv")
    expected_n = len(test_csv)
    if len(sub) != expected_n:
        print(f"WARNING: {len(sub)} rows in features, expected {expected_n} in test_students.csv")
    missing = set(test_csv["filename"]) - set(sub["filename"])
    extra = set(sub["filename"]) - set(test_csv["filename"])
    if missing:
        print(f"WARNING: {len(missing)} expected filenames missing from features")
    if extra:
        print(f"WARNING: {len(extra)} extra filenames not in test_students.csv")

    print(f"\npredictions stats:")
    print(f"  min: {sub['FaceOcclusion'].min():.4f}")
    print(f"  max: {sub['FaceOcclusion'].max():.4f}")
    print(f"  mean: {sub['FaceOcclusion'].mean():.4f}")
    print(f"  std: {sub['FaceOcclusion'].std():.4f}")
    nan_count = sub['FaceOcclusion'].isna().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN predictions")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows)")
    print(f"submit this CSV to hfactory for the Training-Free track")


if __name__ == "__main__":
    main()
