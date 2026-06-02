"""Analyze the output of the 2x2 factorial cross-test of 4 pipelines.

For each of the 4 ratios:
  r_3D_Bi (Julien's), r_3D_Sf, r_Bi_Cv, r_Sf_Cv

We compute:
  1. Mean prediction per (target_bin × gender) — diagnostic of bias direction
  2. Correlation with target (signal strength)
  3. Best calibrated multiplicative scalar per pipeline
  4. Best ensemble strategy combining BiSeNet skin + SegFormer skin

Also: per-(F vs M) breakdown of which pipeline wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import score  # noqa: E402

PIPELINES = ["r_3D_Bi", "r_3D_Sf", "r_Bi_Cv", "r_Sf_Cv"]
LABELS = {
    "r_3D_Bi": "3DDFA mask + BiSeNet skin (= Julien's)",
    "r_3D_Sf": "3DDFA mask + SegFormer skin",
    "r_Bi_Cv": "BiSeNet hull + BiSeNet skin",
    "r_Sf_Cv": "SegFormer hull + SegFormer skin",
}


def main():
    csv = REPO_ROOT / "eval" / "cache" / "val_cross_4pipelines.csv"
    if not csv.exists():
        raise SystemExit(f"missing {csv} (cross-test still running?)")
    df = pd.read_csv(csv)
    print(f"loaded {len(df)} rows from {csv}")
    print(f"  gender F count: {(df.gender == 0.0).sum()}, M count: {(df.gender == 1.0).sum()}")
    print()

    # ============== 1. Per-bin × gender mean predictions ==============
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    df["bin"] = pd.cut(df.target, bins=bins, right=False, labels=False)
    print("=" * 110)
    print("PER-BIN × GENDER mean predictions (raw — before any calibration)")
    print("=" * 110)
    for g_val, g_lbl in [(0.0, "F"), (1.0, "M")]:
        sub = df[df.gender == g_val]
        print(f"\n--- {g_lbl} ---")
        cols = ["n", "mean_target"] + PIPELINES
        out = pd.DataFrame()
        for b in range(len(bins) - 1):
            mask = sub.bin == b
            if mask.sum() == 0:
                continue
            row = {
                "bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})",
                "n": int(mask.sum()),
                "mean_target": sub.loc[mask, "target"].mean(),
            }
            for p in PIPELINES:
                row[p] = sub.loc[mask, p].mean()
            out = pd.concat([out, pd.DataFrame([row])])
        print(out.round(3).to_string(index=False))

    # ============== 2. Correlation with target ==============
    print()
    print("=" * 110)
    print("CORRELATION WITH TARGET (overall + per gender)")
    print("=" * 110)
    rows = []
    for p in PIPELINES:
        rows.append({
            "pipeline": LABELS[p],
            "corr_all": df[p].corr(df.target),
            "corr_F": df[df.gender == 0.0][p].corr(df[df.gender == 0.0].target),
            "corr_M": df[df.gender == 1.0][p].corr(df[df.gender == 1.0].target),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ============== 3. Best calibrated scalar per pipeline ==============
    print()
    print("=" * 110)
    print("BEST CALIBRATION SCALAR per pipeline (minimize weighted-MSE on this sample)")
    print("=" * 110)
    from src.metric import weighted_err
    for p in PIPELINES:
        valid = df[p].notna()
        if valid.sum() == 0:
            print(f"  {LABELS[p]:<45} ALL NaN, skipped")
            continue
        best = (1e9, 1.0)
        for cal in np.arange(0.10, 2.50, 0.05):
            pred = (df.loc[valid, p] * cal).clip(0, 1).values
            err = weighted_err(pred, df.loc[valid, "target"].values)
            if err < best[0]:
                best = (err, cal)
        print(f"  {LABELS[p]:<45} best cal = {best[1]:.2f}, err = {best[0]:.5f}")

    # ============== 4. Combined skin: BiSeNet + SegFormer ensemble ==============
    print()
    print("=" * 110)
    print("ENSEMBLE: combining BiSeNet skin and SegFormer skin")
    print("=" * 110)
    df["r_3D_avg"] = (df.r_3D_Bi + df.r_3D_Sf) / 2  # avg of skin sources, 3D mask
    df["r_Cv_avg"] = (df.r_Bi_Cv + df.r_Sf_Cv) / 2  # avg of skin sources, convex hull mask
    df["r_skin_avg"] = (df.r_3D_avg + df.r_Cv_avg) / 2  # full avg

    for p in ["r_3D_avg", "r_Cv_avg", "r_skin_avg"]:
        valid = df[p].notna()
        if valid.sum() == 0: continue
        corr = df.loc[valid, p].corr(df.loc[valid, "target"])
        best = (1e9, 1.0)
        from src.metric import weighted_err
        for cal in np.arange(0.10, 2.50, 0.05):
            pred = (df.loc[valid, p] * cal).clip(0, 1).values
            err = weighted_err(pred, df.loc[valid, "target"].values)
            if err < best[0]:
                best = (err, cal)
        print(f"  {p}: corr={corr:+.3f}, best cal={best[1]:.2f}, err={best[0]:.5f}")

    # ============== 5. IoU diagnostics ==============
    print()
    print("=" * 110)
    print("IoU diagnostics — how much do the masks/skins agree?")
    print("=" * 110)
    for col in ["iou_mask_3d_bi", "iou_mask_3d_sf", "iou_skin_bi_sf"]:
        print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    print()
    print("Interpretation:")
    print("  iou_mask_3d_bi : 3DDFA mask vs BiSeNet-hull mask (how similar are the 'theoretical' faces)")
    print("  iou_mask_3d_sf : 3DDFA mask vs SegFormer-hull mask")
    print("  iou_skin_bi_sf : BiSeNet skin vs SegFormer skin (DOMINANT source of variance)")


if __name__ == "__main__":
    main()
