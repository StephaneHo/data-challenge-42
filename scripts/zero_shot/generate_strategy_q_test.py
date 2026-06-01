"""Generate Strategy Q test predictions using InsightFace predicted gender.

Strategy Q (per-gender):
  cal = 0.45 if pred_gender == F (0) else 0.80
  a_low = 0.60 if F else 0.50
  if pj > 0.65:   pred = 0.15*0.5*(pjc+ps) + 0.85*min(pjc, ps)
  else:           pred = a_low*pjc + (1-a_low)*ps

Inputs:
  - eval/test_julien_baseline.csv          (pj per filename)
  - eval/test_zs_simple_hull_scaled_power07_tta.csv  (ps per filename)
  - eval/cache/test_gender_pred.csv         (pred_gender per filename, from InsightFace)

Output: results/julien_v6_strategy_q/test_predictions.csv

Fallback for missing gender prediction: default to M (majority class).
Fallback for missing pj (Julien failed): use ps directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CAL_F, CAL_M = 0.45, 0.80
ALOW_F, ALOW_M = 0.60, 0.50
TAU = 0.65
A_HIGH = 0.15


def main():
    out_dir = REPO_ROOT / "results" / "julien_v6_strategy_q"
    out_dir.mkdir(parents=True, exist_ok=True)

    j = pd.read_csv(REPO_ROOT / "eval" / "test_julien_baseline.csv").rename(columns={"FaceOcclusion": "pj"})
    sf = pd.read_csv(REPO_ROOT / "eval" / "test_zs_simple_hull_scaled_power07_tta.csv").rename(columns={"pred_ps": "ps"})
    gen = pd.read_csv(REPO_ROOT / "eval" / "cache" / "test_gender_pred.csv")

    df = j.merge(sf, on="filename").merge(gen, on="filename")
    print(f"loaded test: {len(df)} rows")
    print(f"  Julien pj NaN: {df.pj.isna().sum()}")
    print(f"  pred_gender NaN: {df.pred_gender.isna().sum()} (using fallback=M)")

    # Fallbacks
    pj_valid = ~df.pj.isna()
    g_valid = ~df.pred_gender.isna()
    df["g"] = df.pred_gender.fillna(1.0)  # default M

    # Per-gender params
    cal = np.where(df.g == 0.0, CAL_F, CAL_M)
    a_low = np.where(df.g == 0.0, ALOW_F, ALOW_M)
    pjc = df.pj.fillna(0).values * cal
    ps = df.ps.values

    pred_hi = A_HIGH * 0.5 * (pjc + ps) + (1 - A_HIGH) * np.minimum(pjc, ps)
    pred_lo = a_low * pjc + (1 - a_low) * ps
    pred = np.where(df.pj.fillna(0).values > TAU, pred_hi, pred_lo)
    # Fallback for missing pj: pure ps
    pred = np.where(pj_valid, pred, ps)
    pred = np.clip(pred, 0, 1)

    out = pd.DataFrame({
        "filename": df["filename"],
        "FaceOcclusion": pred,
        "gender": "x",
    })

    out_path = out_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  rows: {len(out)}")
    print(f"  NaN: {out.FaceOcclusion.isna().sum()}")
    print(f"  pred mean: {out.FaceOcclusion.mean():.3f}")
    print(f"  pred max:  {out.FaceOcclusion.max():.3f}")
    print()
    print("Distribution by bin:")
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    for i in range(len(edges) - 1):
        mask = (out.FaceOcclusion >= edges[i]) & (out.FaceOcclusion < edges[i + 1])
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.2f}): {mask.sum():>6} ({100*mask.mean():.1f}%)")

    # Pred gender distribution for sanity
    g_dist = df.pred_gender.dropna().value_counts().to_dict()
    print(f"\npred_gender distribution: F={int(g_dist.get(0.0, 0))} M={int(g_dist.get(1.0, 0))}")


if __name__ == "__main__":
    main()
