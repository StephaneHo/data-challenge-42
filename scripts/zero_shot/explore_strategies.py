"""Comprehensive strategy exploration after the val/test distribution discovery.

Tests:
  1. Multi-distribution robust strategy (avg of brief test-like + more-spread)
  2. Alternative SegFormer bases (power05, 06, 08)
  3. Per-gender tuned strategies
  4. 3-zone (low/mid/high) strategies
  5. Strategy ensemble (E + C blend)

Reports the best candidate under each robustness criterion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import reweighted_score, score  # noqa: E402

TEST_LIKE_BRIEF = np.array([0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003])
TEST_LIKE_SPREAD = np.array([0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.02])
TEST_UNIFORM_HEAVY = np.array([0.05, 0.10, 0.10, 0.15, 0.25, 0.30, 0.05])  # hypothesis: even more shift


def s_eval(df, col, gt="target"):
    """Compute val + reweighted scores for all known test distributions."""
    s_val = score(df, pred_col=col, gt_col=gt)["score"]
    s_brief = reweighted_score(df, TEST_LIKE_BRIEF, pred_col=col, gt_col=gt)["score"]
    s_spread = reweighted_score(df, TEST_LIKE_SPREAD, pred_col=col, gt_col=gt)["score"]
    s_heavy = reweighted_score(df, TEST_UNIFORM_HEAVY, pred_col=col, gt_col=gt)["score"]
    s_robust = (s_brief + s_spread + s_heavy) / 3
    s_worst = max(s_brief, s_spread, s_heavy)
    return s_val, s_brief, s_spread, s_heavy, s_robust, s_worst


def load_inputs(ps_csv: str):
    j = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv").rename(columns={"pred": "pj"})
    sf = pd.read_csv(REPO_ROOT / "eval" / ps_csv)[["filename", "pred"]].rename(columns={"pred": "ps"})
    return j.merge(sf, on="filename")


def strategy_2zone(df, cal, tau, a_low, a_high):
    pjc = df.pj * cal
    pred_hi = a_high * (0.5 * (pjc + df.ps)) + (1 - a_high) * np.minimum(pjc, df.ps)
    pred_lo = a_low * pjc + (1 - a_low) * df.ps
    return np.where(df.pj > tau, pred_hi, pred_lo).clip(0, 1)


def strategy_3zone(df, cal, tau1, tau2, a_lo, a_mid, a_hi):
    pjc = df.pj * cal
    p_lo = a_lo * pjc + (1 - a_lo) * df.ps
    p_mid = a_mid * pjc + (1 - a_mid) * df.ps
    p_hi = a_hi * (0.5 * (pjc + df.ps)) + (1 - a_hi) * np.minimum(pjc, df.ps)
    pred = np.where(df.pj > tau2, p_hi, np.where(df.pj > tau1, p_mid, p_lo))
    return np.clip(pred, 0, 1)


def main():
    print("=" * 110)
    print("EXPLORATION 1: Multi-distribution robust 2-zone (avg of brief + spread)")
    print("=" * 110)
    df = load_inputs("val_zs_simple_hull_scaled_power07_tta.csv")
    print(f"loaded {len(df)} val samples (ps = power07_tta)")
    print()

    results = []
    for cal in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for tau in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
            for a_low in [0.4, 0.5, 0.6, 0.7, 0.78]:
                for a_high in [0.0, 0.15, 0.3, 0.5]:
                    df["p"] = strategy_2zone(df, cal, tau, a_low, a_high)
                    sv, sb, ss, sh, sr, sw = s_eval(df, "p")
                    results.append((cal, tau, a_low, a_high, sv, sb, ss, sh, sr, sw))

    # Sort by robust (avg of 3 distributions)
    results.sort(key=lambda x: x[8])
    print(f"{'cal':>5} {'tau':>5} {'a_lo':>5} {'a_hi':>5} {'val':>8} {'brief':>8} {'spread':>8} {'heavy':>8} {'ROBUST':>9} {'worst':>8}")
    for r in results[:12]:
        print(f"{r[0]:>5.2f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>5.2f} {r[4]:>8.5f} {r[5]:>8.5f} {r[6]:>8.5f} {r[7]:>8.5f} {r[8]:>9.5f} {r[9]:>8.5f}")
    print()

    # Sort by worst-case (most pessimistic)
    print("Top 8 by worst-case minimization (most robust):")
    results.sort(key=lambda x: x[9])
    for r in results[:8]:
        print(f"{r[0]:>5.2f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>5.2f} {r[4]:>8.5f} {r[5]:>8.5f} {r[6]:>8.5f} {r[7]:>8.5f} {r[8]:>9.5f} {r[9]:>8.5f}")
    print()

    # Best on brief alone (the official one)
    print("Top 8 on TEST-LIKE BRIEF alone (official):")
    results.sort(key=lambda x: x[5])
    for r in results[:8]:
        print(f"{r[0]:>5.2f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>5.2f} {r[4]:>8.5f} {r[5]:>8.5f} {r[6]:>8.5f} {r[7]:>8.5f} {r[8]:>9.5f} {r[9]:>8.5f}")


if __name__ == "__main__":
    main()
