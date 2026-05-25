"""Official scoring metric, matching weighted loss, and per-bin analysis.

    Err_g = sum_i w_i (p_i - GT_i)^2 / sum_i w_i,   w_i = 1/30 + GT_i
    Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


GENDER_FEMALE = 0.0
GENDER_MALE = 1.0

# Coarse bins matching src/data.py — used for stratification and per-bin diagnostics.
OCC_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
OCC_BIN_LABELS = [f"[{a:.2f},{b:.2f})" for a, b in zip(OCC_BINS[:-1], OCC_BINS[1:])]


def sample_weight(gt):
    return 1.0 / 30.0 + gt


def weighted_err(pred, gt) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    w = sample_weight(gt)
    return float(np.sum(w * (pred - gt) ** 2) / np.sum(w))


def score(df: pd.DataFrame, pred_col: str = "pred", gt_col: str = "target", gender_col: str = "gender") -> dict:
    female = df[df[gender_col] == GENDER_FEMALE]
    male = df[df[gender_col] == GENDER_MALE]
    err_f = weighted_err(female[pred_col], female[gt_col])
    err_m = weighted_err(male[pred_col], male[gt_col])
    return {
        "err_female": err_f,
        "err_male": err_m,
        "mean_err": (err_f + err_m) / 2.0,
        "gap": abs(err_f - err_m),
        "score": (err_f + err_m) / 2.0 + abs(err_f - err_m),
        "n_female": len(female),
        "n_male": len(male),
    }


def _bin_index(values: np.ndarray, bins: list[float]) -> np.ndarray:
    """Map values to integer bin index in [0, len(bins)-2]. Out-of-range -> clipped."""
    idx = np.searchsorted(bins, values, side="right") - 1
    return np.clip(idx, 0, len(bins) - 2)


def per_bin_breakdown(
    df: pd.DataFrame,
    pred_col: str = "pred",
    gt_col: str = "target",
    gender_col: str = "gender",
    bins: list[float] = OCC_BINS,
) -> pd.DataFrame:
    """Per-(bin × gender) error decomposition.

    For each bin × gender, returns:
      - n        : number of samples
      - mean_pred, mean_gt, bias (mean_pred - mean_gt)
      - mse      : unweighted MSE on this slice
      - weighted_err : weighted MSE on this slice (the local Err if it were the whole set)
      - err_contrib  : this slice's contribution to the gender-level Err
                       = sum_{i in slice} w_i (p_i - GT_i)^2  /  sum_{i in gender} w_i
                       (sum across bins of err_contrib per gender = Err_g)
      - w_mass_frac  : fraction of total per-gender weight that lives in this bin
    """
    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = sample_weight(gt)
    sqerr = (pred - gt) ** 2
    wsqerr = w * sqerr
    bin_idx = _bin_index(gt, bins)

    rows = []
    for gender_val, gender_lbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gender_val
        total_w_g = w[mask_g].sum()
        if total_w_g == 0:
            continue
        for b in range(len(bins) - 1):
            mask_b = mask_g & (bin_idx == b)
            n = int(mask_b.sum())
            if n == 0:
                continue
            slice_w = w[mask_b].sum()
            slice_wsqerr = wsqerr[mask_b].sum()
            rows.append({
                "gender": gender_lbl,
                "bin": OCC_BIN_LABELS[b],
                "n": n,
                "mean_pred": float(pred[mask_b].mean()),
                "mean_gt": float(gt[mask_b].mean()),
                "bias": float(pred[mask_b].mean() - gt[mask_b].mean()),
                "mse": float(sqerr[mask_b].mean()),
                "weighted_err": float(slice_wsqerr / slice_w),
                "err_contrib": float(slice_wsqerr / total_w_g),
                "w_mass_frac": float(slice_w / total_w_g),
            })
    return pd.DataFrame(rows)


def reweighted_score(
    df: pd.DataFrame,
    target_bin_probs: np.ndarray | list[float] | dict,
    pred_col: str = "pred",
    gt_col: str = "target",
    gender_col: str = "gender",
    bins: list[float] = OCC_BINS,
) -> dict:
    """Compute the score as if val followed `target_bin_probs` distribution.

    target_bin_probs can be:
      - a list/array of len(bins)-1 floats (one prob per bin, will be normalized)
      - a dict {bin_label_str: prob}

    Each sample in bin b gets multiplier m_b = q_b / p_b
      where p_b = empirical val fraction in bin b (computed per gender),
            q_b = target fraction in bin b.
    Bins with p_b=0 get m_b=0; remaining q values are renormalized for that gender.
    """
    if isinstance(target_bin_probs, dict):
        q = np.array([target_bin_probs.get(lbl, 0.0) for lbl in OCC_BIN_LABELS], dtype=np.float64)
    else:
        q = np.asarray(target_bin_probs, dtype=np.float64)
    q = q / q.sum() if q.sum() > 0 else q

    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = sample_weight(gt)
    sqerr = (pred - gt) ** 2
    bin_idx = _bin_index(gt, bins)

    err_per_gender = {}
    for gender_val, gender_lbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gender_val
        if mask_g.sum() == 0:
            err_per_gender[gender_lbl] = float("nan")
            continue
        # empirical bin probs for this gender
        bins_present = np.bincount(bin_idx[mask_g], minlength=len(bins) - 1)
        p = bins_present / mask_g.sum()
        # mask out target bins where val has no sample for this gender
        q_eff = np.where(p > 0, q, 0.0)
        if q_eff.sum() == 0:
            err_per_gender[gender_lbl] = float("nan")
            continue
        q_eff = q_eff / q_eff.sum()
        # per-sample multiplier
        m_per_bin = np.where(p > 0, q_eff / np.where(p > 0, p, 1.0), 0.0)
        m = m_per_bin[bin_idx]
        num = (m * w * sqerr)[mask_g].sum()
        den = (m * w)[mask_g].sum()
        err_per_gender[gender_lbl] = float(num / den) if den > 0 else float("nan")

    err_f = err_per_gender["F"]
    err_m = err_per_gender["M"]
    return {
        "err_female": err_f,
        "err_male": err_m,
        "mean_err": (err_f + err_m) / 2.0,
        "gap": abs(err_f - err_m),
        "score": (err_f + err_m) / 2.0 + abs(err_f - err_m),
    }


def empirical_bin_probs(values, bins: list[float] = OCC_BINS) -> np.ndarray:
    """Return an array of bin probabilities (sums to 1) for the given values."""
    values = np.asarray(values, dtype=np.float64)
    idx = _bin_index(values, bins)
    counts = np.bincount(idx, minlength=len(bins) - 1)
    total = counts.sum()
    return counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)


class WeightedMSELoss(nn.Module):
    """MSE weighted per-sample by w = 1/30 + GT, matching the official metric.

    Optionally rescales female contributions to balance per-gender mass in the batch,
    which proxies the fairness penalty without requiring per-gender batches.
    """

    def __init__(self, balance_gender: bool = False, eps: float = 1e-8):
        super().__init__()
        self.balance_gender = balance_gender
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, gender: torch.Tensor | None = None) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        w = 1.0 / 30.0 + target
        if self.balance_gender and gender is not None:
            g = gender.view(-1)
            mass_f = w[g == GENDER_FEMALE].sum()
            mass_m = w[g == GENDER_MALE].sum()
            if mass_f > self.eps and mass_m > self.eps:
                scale_f = 0.5 / (mass_f / (mass_f + mass_m))
                scale_m = 0.5 / (mass_m / (mass_f + mass_m))
                w = torch.where(g == GENDER_FEMALE, w * scale_f, w * scale_m)
        return (w * (pred - target) ** 2).sum() / (w.sum() + self.eps)
