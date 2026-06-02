"""Standalone script for full val reweighted scoring.

Computes the official challenge score AND the reweighted scores under multiple
test distribution hypotheses. Self-contained — no project imports needed.

Usage A (CLI):
    python scripts/evaluate_with_reweighting.py \\
        --predictions my_predictions.csv \\
        --labels eval/val_julien_baseline.csv

Usage B (in a notebook):
    from evaluate_with_reweighting import evaluate_predictions
    results = evaluate_predictions(df_with_columns=['filename', 'pred', 'target', 'gender'])

Expected input format:
    A CSV (or DataFrame) with columns:
      - filename : str
      - pred     : float in [0, 1] (your prediction)
      - target   : float in [0, 1] (ground truth, only for val)
      - gender   : 0.0 = F, 1.0 = M

Output:
    A dict with:
      - score_val       : native val score
      - score_brief     : score reweighted as if val followed the brief distribution
      - score_spread    : ... spread (variant 1)
      - score_heavy     : ... heavy (variant 2, worst-case)
      - per-gender errors and bin breakdowns for each distribution
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# =======================================================================
# Constants from the official challenge metric
# =======================================================================
GENDER_FEMALE = 0.0
GENDER_MALE = 1.0

# Bins used for per-bin diagnostics and reweighting.
# (these are the bins of the official scoring metric)
OCC_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]

# Three plausible test distributions:
#   - brief  = the one announced in the Telecom challenge PDF
#   - spread = a variant with slightly more tail
#   - heavy  = a worst-case variant with more mass on high occlusion
TEST_DISTRIBUTIONS = {
    "brief":  np.array([0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003]),
    "spread": np.array([0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.020]),
    "heavy":  np.array([0.05, 0.10, 0.10, 0.15, 0.25, 0.30, 0.050]),
}


# =======================================================================
# Official metric
# =======================================================================
def sample_weight(gt: np.ndarray) -> np.ndarray:
    """Per-sample weight w_i = 1/30 + GT_i (official metric formula)."""
    return 1.0 / 30.0 + gt


def weighted_err(pred: np.ndarray, gt: np.ndarray) -> float:
    """Weighted MSE = sum_i w_i (p_i - GT_i)^2 / sum_i w_i."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    w = sample_weight(gt)
    return float(np.sum(w * (pred - gt) ** 2) / np.sum(w))


def native_score(df: pd.DataFrame, pred_col="pred", gt_col="target",
                 gender_col="gender") -> dict:
    """Native challenge score on val distribution.

        Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|
    """
    F = df[df[gender_col] == GENDER_FEMALE]
    M = df[df[gender_col] == GENDER_MALE]
    err_F = weighted_err(F[pred_col].values, F[gt_col].values)
    err_M = weighted_err(M[pred_col].values, M[gt_col].values)
    gap = abs(err_F - err_M)
    return {
        "err_female": err_F,
        "err_male": err_M,
        "mean_err": (err_F + err_M) / 2,
        "gap": gap,
        "score": (err_F + err_M) / 2 + gap,
        "n_female": len(F),
        "n_male": len(M),
    }


def _bin_index(values: np.ndarray, bins: list[float]) -> np.ndarray:
    """Map values to integer bin indices."""
    idx = np.searchsorted(bins, values, side="right") - 1
    return np.clip(idx, 0, len(bins) - 2)


def reweighted_score(df: pd.DataFrame, target_bin_probs: np.ndarray,
                     pred_col="pred", gt_col="target", gender_col="gender",
                     bins: list[float] = OCC_BINS) -> dict:
    """Compute the score AS IF val followed the `target_bin_probs` distribution.

    Each sample in bin b gets multiplier m_b = q_b / p_b
      where p_b = empirical val fraction in bin b (per gender)
            q_b = target fraction in bin b (the test distribution)
    This simulates the score we'd get on a test set with the given bin distribution.
    """
    q = np.asarray(target_bin_probs, dtype=np.float64)
    q = q / q.sum() if q.sum() > 0 else q

    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = sample_weight(gt)
    sqerr = (pred - gt) ** 2
    bin_idx = _bin_index(gt, bins)

    err_per_gender = {}
    for gval, glbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gval
        if mask_g.sum() == 0:
            err_per_gender[glbl] = float("nan")
            continue
        # Empirical bin probs for this gender
        bins_present = np.bincount(bin_idx[mask_g], minlength=len(bins) - 1)
        p = bins_present / mask_g.sum()
        # Mask out bins where val has no sample for this gender
        q_eff = np.where(p > 0, q, 0.0)
        q_eff = q_eff / q_eff.sum() if q_eff.sum() > 0 else q_eff
        # Per-sample multiplier
        m_per_bin = np.where(p > 0, q_eff / np.where(p > 0, p, 1.0), 0.0)
        m = m_per_bin[bin_idx]
        num = (m * w * sqerr)[mask_g].sum()
        den = (m * w)[mask_g].sum()
        err_per_gender[glbl] = float(num / den) if den > 0 else float("nan")

    err_F = err_per_gender["F"]
    err_M = err_per_gender["M"]
    return {
        "err_female": err_F,
        "err_male": err_M,
        "mean_err": (err_F + err_M) / 2,
        "gap": abs(err_F - err_M),
        "score": (err_F + err_M) / 2 + abs(err_F - err_M),
    }


def per_bin_breakdown(df: pd.DataFrame, pred_col="pred", gt_col="target",
                      gender_col="gender", bins=OCC_BINS) -> pd.DataFrame:
    """Per (bin × gender) breakdown of error contribution."""
    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = sample_weight(gt)
    sqerr = (pred - gt) ** 2
    wsqerr = w * sqerr
    bin_idx = _bin_index(gt, bins)

    rows = []
    for gval, glbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gval
        if mask_g.sum() == 0:
            continue
        for b in range(len(bins) - 1):
            mask_b = mask_g & (bin_idx == b)
            n = int(mask_b.sum())
            if n == 0:
                continue
            rows.append({
                "gender": glbl,
                "bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})",
                "n": n,
                "mean_pred": float(pred[mask_b].mean()),
                "mean_gt": float(gt[mask_b].mean()),
                "bias": float(pred[mask_b].mean() - gt[mask_b].mean()),
                "mse": float(sqerr[mask_b].mean()),
                "weighted_err": float(wsqerr[mask_b].sum() / w[mask_b].sum()),
            })
    return pd.DataFrame(rows)


# =======================================================================
# Public entry point
# =======================================================================
def evaluate_predictions(df: pd.DataFrame, pred_col="pred", gt_col="target",
                          gender_col="gender", verbose=True) -> dict:
    """Full evaluation: native score + 3 reweighted scores + per-bin breakdown.

    Returns a dict with the 4 scores and the per-bin DataFrame.
    """
    out = {}
    # 1. Native val score
    out["native_val"] = native_score(df, pred_col, gt_col, gender_col)

    # 2. Reweighted scores under each plausible test distribution
    for name, dist in TEST_DISTRIBUTIONS.items():
        out[f"reweighted_{name}"] = reweighted_score(df, dist, pred_col, gt_col, gender_col)

    # 3. Per-bin breakdown
    out["per_bin"] = per_bin_breakdown(df, pred_col, gt_col, gender_col)

    if verbose:
        print(f"\n{'=' * 70}")
        print("FULL VAL REWEIGHTED EVALUATION")
        print(f"{'=' * 70}")
        print(f"\nSample size: {len(df)} images "
              f"(F={out['native_val']['n_female']}, M={out['native_val']['n_male']})")
        print()
        print(f"{'distribution':<20} {'err_F':>10} {'err_M':>10} {'gap':>10} {'score':>10}")
        print(f"{'-' * 65}")
        s = out["native_val"]
        print(f"{'val natif (empiric)':<20} {s['err_female']:>10.5f} {s['err_male']:>10.5f} "
              f"{s['gap']:>10.5f} {s['score']:>10.5f}")
        for name in ["brief", "spread", "heavy"]:
            s = out[f"reweighted_{name}"]
            print(f"{'reweighted ' + name:<20} {s['err_female']:>10.5f} {s['err_male']:>10.5f} "
                  f"{s['gap']:>10.5f} {s['score']:>10.5f}")
        print()
        print("Per-bin breakdown:")
        print(out["per_bin"].round(4).to_string(index=False))

    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions with reweighting")
    parser.add_argument("--predictions", required=True,
                        help="CSV with columns filename + pred (your model output)")
    parser.add_argument("--labels", required=True,
                        help="CSV with columns filename + target + gender (ground truth)")
    parser.add_argument("--pred-col", default="pred",
                        help="Column name in predictions CSV (default: 'pred')")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    label_df = pd.read_csv(args.labels)

    # Normalize the pred column name to 'pred'
    if args.pred_col != "pred":
        pred_df = pred_df.rename(columns={args.pred_col: "pred"})

    # Merge on filename
    df = pred_df.merge(label_df[["filename", "target", "gender"]], on="filename")
    if len(df) < len(pred_df):
        print(f"WARNING: only {len(df)}/{len(pred_df)} predictions matched a label.")

    evaluate_predictions(df, verbose=True)


if __name__ == "__main__":
    main()
