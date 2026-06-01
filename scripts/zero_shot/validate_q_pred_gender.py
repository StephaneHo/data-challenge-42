"""Validate Strategy Q with predicted gender vs true gender on full val.

Runs Q with:
  - true gender (oracle)
  - InsightFace predicted gender
  - majority class (all M, baseline)
  - E (no per-gender) for reference

Reports val + reweighted scores on the 3 test distributions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import reweighted_score, score  # noqa: E402

TEST_LIKE_BRIEF = [0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003]
TEST_LIKE_SPREAD = [0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.02]
TEST_HEAVY = [0.05, 0.10, 0.10, 0.15, 0.25, 0.30, 0.05]


def strat_Q(pj, ps, gender_arr,
            cal_F=0.45, cal_M=0.80, tau=0.65, a_lo_F=0.60, a_lo_M=0.50, a_hi=0.15):
    pjc = np.where(gender_arr == 0.0, pj * cal_F, pj * cal_M)
    a_lo = np.where(gender_arr == 0.0, a_lo_F, a_lo_M)
    return np.where(pj > tau,
                    a_hi * 0.5 * (pjc + ps) + (1 - a_hi) * np.minimum(pjc, ps),
                    a_lo * pjc + (1 - a_lo) * ps).clip(0, 1)


def strat_E(pj, ps, cal=0.65, tau=0.60, a_lo=0.60, a_hi=0.30):
    pjc = pj * cal
    return np.where(pj > tau,
                    a_hi * 0.5 * (pjc + ps) + (1 - a_hi) * np.minimum(pjc, ps),
                    a_lo * pjc + (1 - a_lo) * ps).clip(0, 1)


def report(df, pred_col, label):
    sv = score(df, pred_col=pred_col, gt_col="target")["score"]
    sb = reweighted_score(df, TEST_LIKE_BRIEF, pred_col=pred_col, gt_col="target")["score"]
    ss = reweighted_score(df, TEST_LIKE_SPREAD, pred_col=pred_col, gt_col="target")["score"]
    sh = reweighted_score(df, TEST_HEAVY, pred_col=pred_col, gt_col="target")["score"]
    print(f"  {label:<35} val={sv:.5f}  brief={sb:.5f}  spread={ss:.5f}  heavy={sh:.5f}  "
          f"robust={(sb+ss+sh)/3:.5f}  worst={max(sb,ss,sh):.5f}")


def main():
    j = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv").rename(columns={"pred": "pj"})
    sf = pd.read_csv(REPO_ROOT / "eval" / "val_zs_simple_hull_scaled_power07_tta.csv")[["filename", "pred"]].rename(columns={"pred": "ps"})
    pred_gender = pd.read_csv(REPO_ROOT / "eval" / "cache" / "val_gender_pred.csv")

    df = j.merge(sf, on="filename").merge(pred_gender, on="filename")
    df["pred_gender"] = df.pred_gender.fillna(1.0)  # default M for missing
    pj_valid = ~df.pj.isna()
    df["pj"] = df.pj.fillna(0)

    acc = (df.gender == df.pred_gender).mean()
    print(f"loaded {len(df)} val samples")
    print(f"gender accuracy: {acc:.3f}")
    cm = pd.crosstab(df.gender, df.pred_gender)
    print(f"confusion:\n{cm}")
    print()

    # Strategy Q variants
    df["pred_Q_oracle"] = strat_Q(df.pj.values, df.ps.values, df.gender.values)
    df["pred_Q_predicted"] = strat_Q(df.pj.values, df.ps.values, df.pred_gender.values)
    df["pred_Q_allM"] = strat_Q(df.pj.values, df.ps.values, np.ones(len(df)))

    # Strategy E (no per-gender)
    df["pred_E"] = strat_E(df.pj.values, df.ps.values)

    print("Comparison on full val (no test reweighting issue here):")
    report(df, "pred_Q_oracle", "Q with TRUE gender (oracle)")
    report(df, "pred_Q_predicted", "Q with InsightFace gender")
    report(df, "pred_Q_allM", "Q assuming all M")
    report(df, "pred_E", "E (no per-gender, reference)")
    print()

    # Decomposition: where do the errors come from
    errors_mask = df.gender != df.pred_gender
    n_err = errors_mask.sum()
    print(f"Gender errors: {n_err}/{len(df)} ({100*n_err/len(df):.1f}%)")
    print(f"  F predicted as M: {((df.gender==0.0) & (df.pred_gender==1.0)).sum()}")
    print(f"  M predicted as F: {((df.gender==1.0) & (df.pred_gender==0.0)).sum()}")

    # MSE impact per-row
    mse_oracle = (df.pred_Q_oracle - df.target) ** 2
    mse_pred = (df.pred_Q_predicted - df.target) ** 2
    impact = mse_pred - mse_oracle
    print()
    print(f"MSE impact (pred-oracle):")
    print(f"  mean per-row (all):      {impact.mean():+.5f}")
    print(f"  mean per-row (correct):  {impact[~errors_mask].mean():+.5f}")
    print(f"  mean per-row (wrong):    {impact[errors_mask].mean():+.5f}")


if __name__ == "__main__":
    main()
