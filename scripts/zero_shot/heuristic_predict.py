"""Apply a named TF-compliant heuristic to cached SegFormer features.

Reads a feature cache CSV (produced by `cache_val_features.py`), applies the
heuristic to compute predictions, and writes a CSV in the format expected by
`scripts/eval_harness.py`: filename, pred, target, gender.

Usage:
    python scripts/zero_shot/heuristic_predict.py \\
        --features eval/cache/val_segformer_features.csv \\
        --heuristic multi_feature \\
        --out eval/val_zs_multi_feature.csv

For TTA (averaging with horizontally-flipped features):
    python scripts/zero_shot/heuristic_predict.py \\
        --features eval/cache/val_segformer_features.csv \\
        --features-flipped eval/cache/val_segformer_features_flip.csv \\
        --heuristic multi_feature \\
        --out eval/val_zs_multi_feature_tta.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import score  # noqa: E402
from src.zero_shot.heuristics import HEURISTICS, apply_heuristic, apply_with_tta  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="Feature cache CSV (filename, gender, target, + 11 feature columns)")
    p.add_argument("--features-flipped", default=None,
                   help="Optional: feature cache for horizontally-flipped images, for TTA")
    p.add_argument("--heuristic", required=True, choices=list(HEURISTICS))
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feats = pd.read_csv(args.features)
    print(f"loaded features: {len(feats)} rows from {args.features}")

    if args.features_flipped:
        feats_flipped = pd.read_csv(args.features_flipped)
        if len(feats) != len(feats_flipped):
            raise SystemExit(f"feature row count mismatch: "
                             f"{len(feats)} vs {len(feats_flipped)}")
        # Make sure the two are aligned by filename
        feats_flipped = feats_flipped.set_index("filename").reindex(feats["filename"]).reset_index()
        pred = apply_with_tta(args.heuristic, feats, feats_flipped)
        print(f"applied {args.heuristic} + TTA (flip averaging)")
    else:
        pred = apply_heuristic(args.heuristic, feats)
        print(f"applied {args.heuristic}")

    out_df = pd.DataFrame({
        "filename": feats["filename"],
        "pred": pred,
        "target": feats["target"],
        "gender": feats["gender"],
    })

    s = score(out_df)
    print(f"\nval score: {s['score']:.5f}")
    print(f"  err_female: {s['err_female']:.5f}  (n={s['n_female']})")
    print(f"  err_male:   {s['err_male']:.5f}  (n={s['n_male']})")
    print(f"  gap:        {s['gap']:.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
