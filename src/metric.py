"""Official scoring metric and matching weighted loss.

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


def sample_weight(gt: np.ndarray | torch.Tensor):
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
