"""Run zero-shot inference on a CSV of filenames, produce predictions.

Two modes:
  - --val : also computes the val score (loads target/gender from train.csv with the same
            stratified split as training)
  - default (test) : writes filename,FaceOcclusion,gender='x' for submission format

Usage:
    # On val
    python scripts/zero_shot/predict_csv.py --val --out eval/val_zero_shot.csv

    # On test
    python scripts/zero_shot/predict_csv.py --test --out results/zero_shot/test_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import stratified_split  # noqa: E402
from src.metric import score  # noqa: E402
from src.zero_shot.calibrator import OcclusionCalibrator  # noqa: E402
from src.zero_shot.parser import FaceParser  # noqa: E402
from src.zero_shot.pipeline import extract_features_from_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--calibrator", default=str(REPO_ROOT / "src" / "zero_shot" / "calibrator.pkl"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--val", action="store_true",
                      help="Predict on the local val split (with GT, computes the official score)")
    mode.add_argument("--test", action="store_true",
                      help="Predict on the test set (no GT, output is submission-formatted)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = all; useful for quick dev")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parser = FaceParser()
    cal = OcclusionCalibrator.load(Path(args.calibrator))
    print(f"loaded calibrator ({cal.mode}, train R^2 = {cal.train_r2_:.4f})")

    if args.val:
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, val_df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
        if args.limit > 0:
            val_df = val_df.head(args.limit)
        print(f"val rows: {len(val_df)}")
        feats = extract_features_from_paths(parser, Path(args.image_dir),
                                            val_df["filename"].tolist(), batch_size=args.batch_size)
        feats["pred"] = cal.predict(feats)
        feats["target"] = val_df["FaceOcclusion"].values
        feats["gender"] = val_df["gender"].values
        s = score(feats)
        print(f"\nval score (zero-shot): {s['score']:.5f}")
        print(f"  err_female={s['err_female']:.5f}  err_male={s['err_male']:.5f}  "
              f"gap={s['gap']:.5f}  n_f={s['n_female']}  n_m={s['n_male']}")
        out_cols = ["filename", "pred", "target", "gender"]
    else:
        test_csv = pd.read_csv(Path(args.data_dir) / "test_students.csv")
        if args.limit > 0:
            test_csv = test_csv.head(args.limit)
        print(f"test rows: {len(test_csv)}")
        feats = extract_features_from_paths(parser, Path(args.image_dir),
                                            test_csv["filename"].tolist(), batch_size=args.batch_size)
        feats["FaceOcclusion"] = cal.predict(feats)
        feats["gender"] = "x"
        out_cols = ["filename", "FaceOcclusion", "gender"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats[out_cols].to_csv(out, index=False)
    print(f"wrote {out} ({len(feats)} rows)")


if __name__ == "__main__":
    main()
