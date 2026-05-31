"""Strict evaluation harness for the Training-Free track.

Why this exists (vs estimate_scores.py): the training-free rules prohibit
optimizing against the dataset. Iterating multiple times against the same
evaluation samples is implicit optimization. This harness enforces a strict
ablation protocol:

  - A FIXED eval subset of N samples (default 2000), derived deterministically
    from train.csv via the same stratified split used at training time.
  - Each variant of the pipeline is evaluated ONCE.
  - The harness writes its outputs under reports/<variant_name>/ and refuses
    to overwrite a previous report unless --force is set (so you don't lose
    track of how many times you've evaluated against the subset).
  - A meta.json records exactly which samples were used, so reviewers can
    verify reproducibility.

Output per variant:
  reports/<variant>/
    meta.json            — subset hash, samples used, timestamp
    score.json           — official score + per-distribution reweighted estimates
    per_bin.csv          — per-(gender × bin) breakdown
    predictions.csv      — the variant's predictions on the eval subset

Usage:
    python scripts/eval_harness.py \\
        --predictions eval/val_zero_shot.csv \\
        --variant zero_shot_v1 \\
        --notes "BiSeNet+SegFormer fusion, no calibration"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import stratified_split  # noqa: E402
from src.metric import (  # noqa: E402
    OCC_BIN_LABELS,
    empirical_bin_probs,
    per_bin_breakdown,
    reweighted_score,
    score,
)

# Same distributions as estimate_scores.py (kept in sync intentionally).
TARGET_DISTRIBUTIONS = {
    "test-like (from brief)":      [0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003],
    "test-like (more spread)":     [0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.02],
    "uniform [0, 0.50)":           [1, 1, 1, 1, 2, 4, 0],
}

EVAL_SUBSET_SIZE_DEFAULT = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="CSV with columns: filename, pred, target, gender (produced by eval_val.py "
                        "or scripts/zero_shot/predict_csv.py --val)")
    p.add_argument("--variant", required=True, help="Variant name (e.g., 'zero_shot_v1')")
    p.add_argument("--notes", default="", help="Free-text description of the variant")
    p.add_argument("--design-samples", default="",
                   help="Comma-separated filenames inspected manually to design the pipeline (training-free disclosure)")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--eval-subset-size", type=int, default=EVAL_SUBSET_SIZE_DEFAULT)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing report for this variant")
    return p.parse_args()


def derive_eval_subset(data_dir: Path, size: int, val_frac: float, seed: int) -> tuple[pd.DataFrame, str]:
    """Produce a deterministic fixed eval subset.

    Uses the same stratified val split as training (seed=42, val_frac=0.15), then
    takes the first `size` rows by filename ordering. Returns (df, subset_hash).
    """
    train_csv = pd.read_csv(data_dir / "train.csv")
    _, val_df = stratified_split(train_csv, val_frac=val_frac, seed=seed)
    val_df = val_df.sort_values("filename").reset_index(drop=True)
    subset = val_df.head(size).reset_index(drop=True)
    h = hashlib.sha1(",".join(subset["filename"].tolist()).encode()).hexdigest()[:12]
    return subset, h


def main() -> None:
    args = parse_args()
    reports_root = Path(args.reports_dir)
    out_dir = reports_root / args.variant
    if out_dir.exists() and not args.force:
        print(f"ERROR: {out_dir} already exists. Use --force to overwrite "
              "(but think first — repeated evaluation against the same subset "
              "is implicit dataset optimization).")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_subset, subset_hash = derive_eval_subset(
        Path(args.data_dir), args.eval_subset_size, args.val_frac, args.seed
    )
    print(f"eval subset: {len(eval_subset)} samples (hash {subset_hash})")

    pred_df = pd.read_csv(args.predictions)
    merged = eval_subset[["filename", "FaceOcclusion", "gender"]].merge(
        pred_df[["filename", "pred"]], on="filename", how="left"
    )
    missing = merged["pred"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} eval samples have no prediction in {args.predictions} — dropping them.")
        merged = merged.dropna(subset=["pred"]).reset_index(drop=True)
    merged = merged.rename(columns={"FaceOcclusion": "target"})

    print(f"\nevaluating {len(merged)} predictions against fixed eval subset")
    s = score(merged)
    print(f"  score:      {s['score']:.5f}")
    print(f"  err_female: {s['err_female']:.5f}  (n={s['n_female']})")
    print(f"  err_male:   {s['err_male']:.5f}  (n={s['n_male']})")
    print(f"  gap:        {s['gap']:.5f}")

    # Reweight to alternative distributions
    val_bin_probs = empirical_bin_probs(merged["target"].to_numpy())
    reweighted = {"val (subset native)": s["score"]}
    for name, target in TARGET_DISTRIBUTIONS.items():
        reweighted[name] = reweighted_score(merged, target)["score"]
    print("\nreweighted to alternative distributions:")
    for name, sc in reweighted.items():
        print(f"  {name:<32s} {sc:.5f}")

    # Per-bin breakdown
    bb = per_bin_breakdown(merged)
    print("\nper-bin breakdown (saved to per_bin.csv):")
    print(bb.pivot(index="bin", columns="gender", values="weighted_err").round(5).to_string())

    # Write outputs
    meta = {
        "variant": args.variant,
        "notes": args.notes,
        "predictions_source": str(args.predictions),
        "eval_subset_size": len(eval_subset),
        "eval_subset_hash": subset_hash,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "timestamp": time.time(),
        "design_samples": [s.strip() for s in args.design_samples.split(",") if s.strip()],
        "val_empirical_bin_probs": dict(zip(OCC_BIN_LABELS, val_bin_probs.tolist())),
    }
    score_report = {
        "official_score": s,
        "reweighted_scores": reweighted,
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "score.json", "w") as f:
        json.dump(score_report, f, indent=2)
    bb.to_csv(out_dir / "per_bin.csv", index=False)
    merged.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\nreport written to {out_dir}/")


if __name__ == "__main__":
    main()
