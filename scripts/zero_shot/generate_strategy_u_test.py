"""Generate Strategy U test predictions = 0.5 * Strategy S + 0.5 * Strategy Q_pred.

Strategy S (single-cal, no gender needed):
    pjc = pj * 0.85
    if pj > 0.65: pred = 0.15*0.5*(pjc+ps) + 0.85*min(pjc, ps)
    else:         pred = 0.60*pjc + 0.40*ps

Strategy Q_pred (per-gender, re-tuned for noisy InsightFace gender):
    cal_F=0.75, cal_M=0.70, a_lo_F=0.70, a_lo_M=0.50, tau=0.65, a_hi=0.15
    pjc = pj * (cal_F if gender==F else cal_M)
    a_lo = (a_lo_F if gender==F else a_lo_M)
    if pj > tau:  pred = 0.15*0.5*(pjc+ps) + 0.85*min(pjc, ps)
    else:         pred = a_lo*pjc + (1-a_lo)*ps

U = 0.5 * pred_S + 0.5 * pred_Q_pred

Inputs:
  - eval/test_julien_baseline.csv          (pj)
  - eval/test_zs_simple_hull_scaled_power07_tta.csv  (ps)
  - eval/cache/test_gender_pred.csv         (InsightFace pred_gender)

Output: results/julien_v8_strategy_u/test_predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def strategy_S(pj, ps):
    pjc = pj * 0.85
    pred_hi = 0.15 * 0.5 * (pjc + ps) + 0.85 * np.minimum(pjc, ps)
    pred_lo = 0.60 * pjc + 0.40 * ps
    return np.where(pj > 0.65, pred_hi, pred_lo)


def strategy_Q_pred(pj, ps, gender_arr):
    cal = np.where(gender_arr == 0.0, 0.75, 0.70)
    a_lo = np.where(gender_arr == 0.0, 0.70, 0.50)
    pjc = pj * cal
    pred_hi = 0.15 * 0.5 * (pjc + ps) + 0.85 * np.minimum(pjc, ps)
    pred_lo = a_lo * pjc + (1 - a_lo) * ps
    return np.where(pj > 0.65, pred_hi, pred_lo)


def main():
    out_dir = REPO_ROOT / "results" / "julien_v8_strategy_u"
    out_dir.mkdir(parents=True, exist_ok=True)

    j = pd.read_csv(REPO_ROOT / "eval" / "test_julien_baseline.csv").rename(columns={"FaceOcclusion": "pj"})
    sf = pd.read_csv(REPO_ROOT / "eval" / "test_zs_simple_hull_scaled_power07_tta.csv").rename(columns={"pred_ps": "ps"})
    gen_path = REPO_ROOT / "eval" / "cache" / "test_gender_pred.csv"
    if not gen_path.exists():
        raise SystemExit(f"missing {gen_path} (test gender cache not ready)")
    gen = pd.read_csv(gen_path)

    df = j.merge(sf, on="filename").merge(gen, on="filename")
    print(f"loaded test: {len(df)} rows")
    pj_valid = ~df.pj.isna()
    g_valid = ~df.pred_gender.isna()
    print(f"  pj NaN (Julien failed): {(~pj_valid).sum()}")
    print(f"  gender NaN (InsightFace failed): {(~g_valid).sum()} (fallback=M)")

    pj = df.pj.fillna(0).values
    ps = df.ps.values
    g = df.pred_gender.fillna(1.0).values  # M majority fallback

    pred_S = strategy_S(pj, ps)
    pred_Q_pred = strategy_Q_pred(pj, ps, g)
    pred_U = 0.5 * pred_S + 0.5 * pred_Q_pred
    # Fallback: if pj is NaN, use pure ps
    pred_U = np.where(pj_valid, pred_U, ps)
    pred_U = np.clip(pred_U, 0, 1)

    out = pd.DataFrame({
        "filename": df["filename"],
        "FaceOcclusion": pred_U,
        "gender": "x",
    })
    out_path = out_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out)} rows, NaN={out.FaceOcclusion.isna().sum()})")
    print(f"  pred mean: {out.FaceOcclusion.mean():.3f}")
    print(f"  pred max:  {out.FaceOcclusion.max():.3f}")
    print()
    print("Distribution by bin:")
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    for i in range(len(edges) - 1):
        mask = (out.FaceOcclusion >= edges[i]) & (out.FaceOcclusion < edges[i + 1])
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.2f}): {mask.sum():>6} ({100 * mask.mean():.1f}%)")

    # Also save individual Q_pred for fallback
    out_q = REPO_ROOT / "results" / "julien_v6_strategy_q_pred"
    out_q.mkdir(parents=True, exist_ok=True)
    pred_Q_only = strategy_Q_pred(pj, ps, g)
    pred_Q_only = np.where(pj_valid, pred_Q_only, ps).clip(0, 1)
    pd.DataFrame({"filename": df["filename"], "FaceOcclusion": pred_Q_only, "gender": "x"}).to_csv(
        out_q / "test_predictions.csv", index=False
    )
    print(f"\nalso wrote {out_q}/test_predictions.csv (Q_pred alone)")


if __name__ == "__main__":
    main()
