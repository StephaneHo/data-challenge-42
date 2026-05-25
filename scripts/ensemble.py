"""Linearly ensemble two prediction CSVs and find the optimal blend weight.

Takes two val_predictions.csv files (filename, pred, target, gender) and:
  - Loops over alpha in [0, 1]:  pred_ensemble = alpha * pred_a + (1 - alpha) * pred_b
  - Reports the best alpha on the official score
  - Also shows per-gender breakdown at best alpha

Usage:
    python scripts/ensemble.py --a eval/val_resnet50_8ep.csv --b eval/val_zero_shot.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import per_bin_breakdown, score  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first prediction CSV (filename, pred, target, gender)")
    p.add_argument("--b", required=True, help="second prediction CSV (same format)")
    p.add_argument("--alphas", type=int, default=21, help="number of alphas to scan (in [0, 1])")
    p.add_argument("--out", default=None, help="optional path to save the best-alpha blended CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    # Sanity: both must have the same filenames (we'll merge on filename)
    df = df_a.merge(df_b[["filename", "pred"]], on="filename", suffixes=("_a", "_b"))
    if len(df) != len(df_a) or len(df) != len(df_b):
        print(f"WARNING: merge produced {len(df)} rows, but inputs had {len(df_a)} and {len(df_b)}.")
        print("         Continuing on the intersection — make sure the two files cover the same samples.")
    print(f"merged {len(df)} rows from {Path(args.a).name} and {Path(args.b).name}")

    pred_a = df["pred_a"].to_numpy(dtype=np.float64)
    pred_b = df["pred_b"].to_numpy(dtype=np.float64)
    target = df["target"].to_numpy(dtype=np.float64)
    gender = df["gender"].to_numpy(dtype=np.float64)

    # Pure A and pure B for reference
    df_a_only = pd.DataFrame({"pred": pred_a, "target": target, "gender": gender})
    df_b_only = pd.DataFrame({"pred": pred_b, "target": target, "gender": gender})
    s_a = score(df_a_only)
    s_b = score(df_b_only)
    print(f"\nA only ({Path(args.a).name}):  score={s_a['score']:.5f}  err_f={s_a['err_female']:.5f}  err_m={s_a['err_male']:.5f}  gap={s_a['gap']:.5f}")
    print(f"B only ({Path(args.b).name}):  score={s_b['score']:.5f}  err_f={s_b['err_female']:.5f}  err_m={s_b['err_male']:.5f}  gap={s_b['gap']:.5f}")

    # Alpha sweep
    alphas = np.linspace(0, 1, args.alphas)
    print(f"\nalpha sweep (pred = alpha * pred_a + (1 - alpha) * pred_b):")
    print(f"{'alpha':>6} {'score':>10} {'err_F':>10} {'err_M':>10} {'gap':>10}")
    results = []
    for a in alphas:
        pred_e = a * pred_a + (1 - a) * pred_b
        df_e = pd.DataFrame({"pred": pred_e, "target": target, "gender": gender})
        s = score(df_e)
        results.append((a, s))
        print(f"{a:>6.2f} {s['score']:>10.5f} {s['err_female']:>10.5f} {s['err_male']:>10.5f} {s['gap']:>10.5f}")

    best_alpha, best_s = min(results, key=lambda r: r[1]["score"])
    print(f"\nbest alpha: {best_alpha:.2f}")
    print(f"  score:      {best_s['score']:.5f}")
    print(f"  err_female: {best_s['err_female']:.5f}")
    print(f"  err_male:   {best_s['err_male']:.5f}")
    print(f"  gap:        {best_s['gap']:.5f}")
    delta_a = best_s["score"] - s_a["score"]
    delta_b = best_s["score"] - s_b["score"]
    print(f"  improvement vs A: {delta_a:+.5f}  ({'better' if delta_a < 0 else 'worse'})")
    print(f"  improvement vs B: {delta_b:+.5f}  ({'better' if delta_b < 0 else 'worse'})")

    # Per-bin breakdown at best alpha
    pred_best = best_alpha * pred_a + (1 - best_alpha) * pred_b
    df_best = pd.DataFrame({"filename": df["filename"], "pred": pred_best,
                            "target": target, "gender": gender})
    print("\nper-bin breakdown at best alpha:")
    bb = per_bin_breakdown(df_best)
    print(bb.pivot(index="bin", columns="gender", values="weighted_err").round(5).to_string())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_best.to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
