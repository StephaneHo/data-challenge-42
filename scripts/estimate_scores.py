"""Per-bin error analysis and reweighted score estimation.

Takes one or more val_predictions.csv files (produced by scripts/eval_val.py).
For each file:
  - Computes the overall score (sanity vs training log)
  - Decomposes error by (occlusion bin × gender)
  - Estimates the score under several hypothesized test distributions
For multiple files: shows side-by-side comparison.

Usage:
    python scripts/estimate_scores.py eval/val_resnet50_8ep.csv
    python scripts/estimate_scores.py eval/*.csv          # compare versions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import (  # noqa: E402
    OCC_BIN_LABELS,
    empirical_bin_probs,
    per_bin_breakdown,
    reweighted_score,
    score,
)

# Hypothesized target distributions per occlusion bin.
# These are guesses — refine when interim feedback arrives.
TARGET_DISTRIBUTIONS = {
    "val (train-like)": None,  # special: use the file's own empirical
    "uniform over bins": [1, 1, 1, 1, 1, 1, 1],
    "test-like (mild spread)": [0.20, 0.20, 0.20, 0.15, 0.15, 0.08, 0.02],
    "test-like (strong spread)": [0.10, 0.15, 0.20, 0.20, 0.20, 0.12, 0.03],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv_files", nargs="+", help="One or more val_predictions.csv files")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def banner(s: str, char: str = "=") -> None:
    print(f"\n{char * 70}\n{s}\n{char * 70}")


def analyze_one(path: Path) -> dict:
    df = pd.read_csv(path)
    banner(f"FILE: {path.name}  ({len(df)} rows)")

    s = score(df)
    print(f"overall score     : {s['score']:.5f}")
    print(f"  err_female      : {s['err_female']:.5f}  (n={s['n_female']})")
    print(f"  err_male        : {s['err_male']:.5f}  (n={s['n_male']})")
    print(f"  gap |F - M|     : {s['gap']:.5f}")
    print(f"  mean_err        : {s['mean_err']:.5f}")

    banner("PER-BIN BREAKDOWN", char="-")
    bb = per_bin_breakdown(df)
    # Wide format for readability
    pivot_n = bb.pivot(index="bin", columns="gender", values="n").fillna(0).astype(int)
    pivot_err = bb.pivot(index="bin", columns="gender", values="weighted_err")
    pivot_bias = bb.pivot(index="bin", columns="gender", values="bias")
    pivot_contrib = bb.pivot(index="bin", columns="gender", values="err_contrib")

    print("Sample counts:")
    print(pivot_n.to_string())
    print("\nLocal weighted error per bin (smaller = better):")
    print(pivot_err.round(5).to_string())
    print("\nBias = mean_pred - mean_gt (positive = over-predicts):")
    print(pivot_bias.round(4).to_string())
    print("\nContribution to gender-level Err (sum per gender = Err_g):")
    print(pivot_contrib.round(5).to_string())

    banner("REWEIGHTED SCORE UNDER DIFFERENT TEST DISTRIBUTION ASSUMPTIONS", char="-")
    val_bin_probs = empirical_bin_probs(df["target"].to_numpy())
    print(f"val empirical bin probs: {dict(zip(OCC_BIN_LABELS, val_bin_probs.round(4)))}")

    estimates = {}
    for name, target in TARGET_DISTRIBUTIONS.items():
        if target is None:
            estimates[name] = s["score"]
        else:
            est = reweighted_score(df, target)
            estimates[name] = est["score"]
        print(f"  {name:<32s} -> {estimates[name]:.5f}")

    return {
        "file": path.name,
        "overall": s,
        "estimates": estimates,
        "bin_breakdown": bb,
    }


def comparison_table(results: list[dict]) -> None:
    banner("COMPARISON ACROSS VERSIONS")
    rows = []
    for r in results:
        row = {"file": r["file"], "score (val)": r["overall"]["score"], "gap": r["overall"]["gap"]}
        for k, v in r["estimates"].items():
            row[k] = v
        rows.append(row)
    cmp_df = pd.DataFrame(rows).set_index("file")
    print(cmp_df.round(5).to_string())

    if len(results) >= 2:
        baseline = cmp_df.iloc[0]
        print("\nDelta vs first file (negative = improvement):")
        delta = cmp_df.subtract(baseline, axis=1)
        print(delta.round(5).to_string())


def main() -> None:
    args = parse_args()
    results = []
    for path_str in args.csv_files:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        try:
            results.append(analyze_one(path))
        except Exception as e:
            print(f"ERROR processing {path}: {e}")
    if len(results) >= 2:
        comparison_table(results)


if __name__ == "__main__":
    main()
