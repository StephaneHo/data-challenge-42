"""Analyze YOLO-World cache on the 823 extreme cases (where Julien predicts > 0.7).

Reads:
  - eval/cache/val_yoloworld_extreme.csv   (one row per extreme case, one set of cols per prompt)
  - eval/val_julien_baseline.csv           (filename, pred, target, gender)

For each prompt:
  - distribution of `<p>_detected` among TRUE extremes (target >= 0.7) vs FALSE extremes (target < 0.7)
  - Pearson corr of `<p>_total_area_frac` vs target
  - mean target for detected vs non-detected

Then tests simple discriminative rules on the extreme subset only, comparing to the
current best calibration (pj * 0.40), using the official src.metric.score().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import score  # noqa: E402


def main():
    yolo_csv = REPO_ROOT / "eval" / "cache" / "val_yoloworld_extreme.csv"
    jul_csv = REPO_ROOT / "eval" / "val_julien_baseline.csv"

    if not yolo_csv.exists():
        raise SystemExit(f"missing {yolo_csv} (still running?)")

    yolo = pd.read_csv(yolo_csv)
    jul = pd.read_csv(jul_csv).rename(columns={"pred": "pj"})

    df = yolo.merge(jul, on="filename")
    print(f"merged: {len(df)} rows (extreme cases, pj > 0.7)")
    print(f"  true extremes (target >= 0.7): {(df['target'] >= 0.7).sum()}")
    print(f"  false extremes (target < 0.7): {(df['target'] < 0.7).sum()}")
    print(f"  mean target: {df['target'].mean():.3f}")
    print(f"  mean pj:     {df['pj'].mean():.3f}")
    print()

    prompts = [c.replace("_detected", "") for c in df.columns if c.endswith("_detected")]
    print(f"prompts found: {prompts}")
    print()

    # === Per-prompt discriminative power ===
    print("=" * 100)
    print("DISCRIMINATIVE POWER per prompt (on the extreme subset)")
    print("=" * 100)
    print(f"{'prompt':<15} {'%det@TE':>9} {'%det@FE':>9} {'lift':>6} "
          f"{'corr_area':>10} {'mean_tgt@det':>13} {'mean_tgt@!det':>15}")
    true_ext = df["target"] >= 0.7
    for p in prompts:
        det = df[f"{p}_detected"] == 1
        pct_det_true = 100 * (det & true_ext).sum() / max(true_ext.sum(), 1)
        pct_det_false = 100 * (det & ~true_ext).sum() / max((~true_ext).sum(), 1)
        lift = pct_det_true / max(pct_det_false, 0.01)
        corr_area = float(df[f"{p}_total_area_frac"].corr(df["target"]))
        mean_tgt_det = df.loc[det, "target"].mean() if det.sum() else 0.0
        mean_tgt_ndet = df.loc[~det, "target"].mean() if (~det).sum() else 0.0
        print(f"{p:<15} {pct_det_true:>9.1f} {pct_det_false:>9.1f} {lift:>6.2f} "
              f"{corr_area:>10.3f} {mean_tgt_det:>13.3f} {mean_tgt_ndet:>15.3f}")

    print()
    print("=" * 100)
    print("CALIBRATION RULES (extreme subset, using official src.metric.score)")
    print("=" * 100)
    df["pred_baseline"] = df["pj"] * 0.40
    res = score(df, pred_col="pred_baseline", gt_col="target")
    print(f"  baseline (pj * 0.40)   score={res['score']:.5f}  "
          f"err_F={res['err_female']:.5f}  err_M={res['err_male']:.5f}  |F-M|={res['gap']:.5f}  "
          f"(n_F={res['n_female']}, n_M={res['n_male']})")

    # Rules using "any face occluder detected" as a confidence signal
    occluder_keys = [k for k in ["hat", "face_mask", "sunglasses", "hand"] if k in prompts]
    if not occluder_keys:
        print("(no canonical occluder prompts, skipping rule eval)")
        return
    print(f"  using occluder prompts: {occluder_keys}")
    any_det = df[[f"{k}_detected" for k in occluder_keys]].max(axis=1) == 1
    print(f"  any-occluder detected on {any_det.sum()}/{len(df)} extreme cases ({100*any_det.mean():.1f}%)")
    print()

    grid = [
        ("ruleA", 0.40, 0.20),
        ("ruleB", 0.50, 0.20),
        ("ruleC", 0.60, 0.20),
        ("ruleD", 0.50, 0.25),
        ("ruleE", 0.70, 0.30),
    ]
    for tag, k_det, k_ndet in grid:
        col = f"pred_{tag}"
        df[col] = np.where(any_det, df["pj"] * k_det, df["pj"] * k_ndet)
        res = score(df, pred_col=col, gt_col="target")
        print(f"  {tag}: det->*{k_det:.2f}, !det->*{k_ndet:.2f}   "
              f"score={res['score']:.5f}  err_F={res['err_female']:.5f}  "
              f"err_M={res['err_male']:.5f}  |F-M|={res['gap']:.5f}")

    # Save for downstream
    out = REPO_ROOT / "eval" / "yolo_world_rules_eval.csv"
    cols = ["filename", "gender", "target", "pj", "pred_baseline"]
    cols += [f"pred_{tag}" for tag, *_ in grid]
    cols += [f"{p}_detected" for p in prompts]
    cols += [f"{p}_total_area_frac" for p in prompts]
    df[cols].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
