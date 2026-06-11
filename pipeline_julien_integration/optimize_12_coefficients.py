"""=============================================================================
Code utilise pour calculer les 12 coefficients de la formule v_features
=============================================================================

Ce script est une **copie standalone** de la procedure d'optimisation utilisee
pour trouver les valeurs de F_WEIGHTS et M_WEIGHTS du notebook :

    F_WEIGHTS = {"hair_bi": 0.376, "hat_bi": 0.425, "other_bi": 0.478,
                 "hair_sf": 0.619, "hat_sf": 0.902, "other_bg_sf": 1.087}
    M_WEIGHTS = {"hair_bi": 0.489, "hat_bi": 0.294, "other_bi": 0.210,
                 "hair_sf": 0.484, "hat_sf": 0.609, "other_bg_sf": 0.382}

Methode : Nelder-Mead (scipy.optimize.minimize, sans gradient) minimisant
directement le SCORE OFFICIEL brief reweighted sur le cache val_features.csv.

Pourquoi Nelder-Mead :
  - Pas de gradient analytique calculable (le score reweighted utilise des
    operations non-differentiables : binning, gender split, abs(err_F - err_M)).
  - 12 dimensions, espace de recherche relativement petit -> simplex converge bien.
  - Robustesse aux discontinuites (le mapping bin est piecewise constant).
  - Tous les coefficients sont libres (pas de contrainte, pas de borne).

Input  : val_features.csv (genere par 01_cache_features.py, ~15K rows)
Output : coefficients.json + reporting des 5 metriques sur 4 distributions

Pour relancer cette optim :
    python optimize_12_coefficients.py \\
        --val-cache eval/cache/val_features.csv \\
        --out pipeline_v_features/coefficients.json

Score officiel obtenu sur val : brief = 0.00594 (val 15K, sans fallback).
============================================================================="""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# CONSTANTES (distributions du brief Telecom + bins de target)
# =============================================================================
GENDER_FEMALE = 0.0
GENDER_MALE = 1.0

OCC_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]

# Distribution officielle Telecom (= ce que IDEMIA reweighte pour scoring brief)
TEST_DISTRIBUTIONS = {
    "brief":  np.array([0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003]),
    "spread": np.array([0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.020]),
    "heavy":  np.array([0.05, 0.10, 0.10, 0.15, 0.25, 0.30, 0.050]),
}


# =============================================================================
# FORMULE v_features : pred = somme ponderee des 6 features (per genre)
# =============================================================================
def predict_v_features(df: pd.DataFrame, F_weights: dict, M_weights: dict,
                       gender_col: str = "gender") -> np.ndarray:
    """Calcule la prediction v_features pour chaque ligne du DataFrame.

    Formule (per gender) :
        pred = w_hair_bi * hair_bi_in_mask
             + w_hat_bi  * hat_bi_in_mask
             + w_other_bi * other_bi_in_mask
             + w_hair_sf * hair_sf_in_mask
             + w_hat_sf  * hat_sf_in_mask
             + w_other_bg_sf * (other_sf_in_mask + bg_sf_in_mask)

    Note : on additionne bg_sf et other_sf en un seul terme "other_bg_sf"
    (asymetrie BiSeNet vs SegFormer : bg_bi traite comme visible, bg_sf comme occlusion).

    Returns:
        np.array de predictions clip dans [0, 1].
    """
    hair_bi = df.hair_bi_in_mask.values
    hat_bi = df.hat_bi_in_mask.values
    other_bi = df.other_bi_in_mask.values
    hair_sf = df.hair_sf_in_mask.values
    hat_sf = df.hat_sf_in_mask.values
    other_bg_sf = (df.other_sf_in_mask + df.bg_sf_in_mask).values
    is_F = (df[gender_col].values == GENDER_FEMALE)

    pred_F = (F_weights["hair_bi"] * hair_bi + F_weights["hat_bi"] * hat_bi +
              F_weights["other_bi"] * other_bi + F_weights["hair_sf"] * hair_sf +
              F_weights["hat_sf"] * hat_sf + F_weights["other_bg_sf"] * other_bg_sf)
    pred_M = (M_weights["hair_bi"] * hair_bi + M_weights["hat_bi"] * hat_bi +
              M_weights["other_bi"] * other_bi + M_weights["hair_sf"] * hair_sf +
              M_weights["hat_sf"] * hat_sf + M_weights["other_bg_sf"] * other_bg_sf)

    return np.clip(np.where(is_F, pred_F, pred_M), 0, 1)


# =============================================================================
# SCORE OFFICIEL : brief reweighted (= metrique optimisee)
# =============================================================================
def _sample_weight(gt: np.ndarray) -> np.ndarray:
    """Poids par echantillon : w_i = 1/30 + GT_i (cf. brief IDEMIA)."""
    return 1.0 / 30.0 + gt


def reweighted_score(df: pd.DataFrame, target_bin_probs: np.ndarray,
                     pred_col: str = "pred", gt_col: str = "target",
                     gender_col: str = "gender", bins: list = OCC_BINS) -> dict:
    """Score si val suivait la distribution target_bin_probs (= brief Telecom).

    Reweighte chaque sample par m_b = q_b / p_b ou :
        p_b = fraction empirique val dans bin b (per gender)
        q_b = fraction cible dans bin b (= TEST_DISTRIBUTIONS[name])

    Score = (err_F + err_M) / 2 + |err_F - err_M|
    """
    q = np.asarray(target_bin_probs, dtype=np.float64)
    q = q / q.sum() if q.sum() > 0 else q

    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = _sample_weight(gt)
    sqerr = (pred - gt) ** 2
    bin_idx = np.clip(np.searchsorted(bins, gt, side="right") - 1, 0, len(bins) - 2)

    err_per_gender = {}
    n_per_gender = {}
    for gval, glbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gval
        n_per_gender[glbl] = int(mask_g.sum())
        if mask_g.sum() == 0:
            err_per_gender[glbl] = float("nan")
            continue
        bins_present = np.bincount(bin_idx[mask_g], minlength=len(bins) - 1)
        p = bins_present / mask_g.sum()
        q_eff = np.where(p > 0, q, 0.0)
        if q_eff.sum() == 0:
            err_per_gender[glbl] = float("nan")
            continue
        q_eff = q_eff / q_eff.sum()
        m_per_bin = np.where(p > 0, q_eff / np.where(p > 0, p, 1.0), 0.0)
        m = m_per_bin[bin_idx]
        num = (m * w * sqerr)[mask_g].sum()
        den = (m * w)[mask_g].sum()
        err_per_gender[glbl] = float(num / den) if den > 0 else float("nan")

    err_F = err_per_gender["F"]
    err_M = err_per_gender["M"]
    n_F = n_per_gender["F"]
    n_M = n_per_gender["M"]
    if np.isnan(err_F) or np.isnan(err_M):
        return {"err_F": err_F, "err_M": err_M, "mean_err": float("nan"),
                "gap": float("nan"), "score": float("nan"), "n_F": n_F, "n_M": n_M}
    mean_err = (n_F * err_F + n_M * err_M) / (n_F + n_M)
    gap = abs(err_F - err_M)
    score = (err_F + err_M) / 2.0 + gap  # < METRIQUE OFFICIELLE >
    return {"err_F": err_F, "err_M": err_M, "mean_err": mean_err,
            "gap": gap, "score": score, "n_F": n_F, "n_M": n_M}


# =============================================================================
# FONCTION OBJECTIF (= ce qu'on minimise dans Nelder-Mead)
# =============================================================================
def objective_brief(params: np.ndarray, df: pd.DataFrame) -> float:
    """Fonction objectif passee a scipy.optimize.minimize.

    Unpack les 12 params en F_weights et M_weights, calcule la prediction
    v_features, calcule le score brief reweighted, retourne ce score.

    L'optimizer cherche le minimum -> il trouve les coefficients qui minimisent
    le score officiel sur val.

    Args:
        params: array de 12 floats. Layout :
                params[0:6]  = F_weights : hair_bi, hat_bi, other_bi, hair_sf, hat_sf, other_bg_sf
                params[6:12] = M_weights : idem
        df: DataFrame val_features.csv avec colonne 'pred' preallouee

    Returns:
        score brief reweighted (float, on cherche son minimum)
    """
    F_w = {"hair_bi": params[0], "hat_bi": params[1], "other_bi": params[2],
           "hair_sf": params[3], "hat_sf": params[4], "other_bg_sf": params[5]}
    M_w = {"hair_bi": params[6], "hat_bi": params[7], "other_bi": params[8],
           "hair_sf": params[9], "hat_sf": params[10], "other_bg_sf": params[11]}
    df.loc[:, "pred"] = predict_v_features(df, F_w, M_w)
    return reweighted_score(df, TEST_DISTRIBUTIONS["brief"])["score"]


# =============================================================================
# MAIN : run l'optimisation Nelder-Mead
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cache", required=True, help="Path to val_features.csv")
    p.add_argument("--out", required=True, help="Path to save coefficients.json")
    args = p.parse_args()

    # === Load val cache ===
    df = pd.read_csv(args.val_cache)
    print(f"loaded {len(df)} rows from {args.val_cache}")
    n_before = len(df)
    df = df.dropna(subset=["hair_bi_in_mask"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} rows with NaN features (face detection failed)")
    print(f"  working on {len(df)} rows (F={(df.gender == 0.0).sum()}, M={(df.gender == 1.0).sum()})")
    df["pred"] = 0.0

    # === Initial guess (= v10_hyb expanded into 12 params) ===
    # v10_hyb : 0.4 * cal_Bi_g * (hair + hat + other)_bi
    #         + 0.6 * cal_Sf_g * (bg + hair + hat + other)_sf
    # F : cal_Bi_F = 1.2, cal_Sf_F = 1.1 -> 0.48, 0.66
    # M : cal_Bi_M = 0.9, cal_Sf_M = 0.9 -> 0.36, 0.54
    x0 = np.array([
        0.48, 0.48, 0.48,   0.66, 0.66, 0.66,    # F
        0.36, 0.36, 0.36,   0.54, 0.54, 0.54,    # M
    ])

    # === Optimization Nelder-Mead ===
    print()
    print("Optimizing 12 coefficients (Nelder-Mead)...")
    print("Objective : reweighted brief score (official IDEMIA metric)")
    res = minimize(
        objective_brief, x0, args=(df,),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7, "disp": True},
    )
    print()
    print(f"Optimization done. Brief Score: {res.fun:.5f}")

    # === Save coefficients ===
    F_w = {"hair_bi": float(res.x[0]),  "hat_bi": float(res.x[1]),  "other_bi": float(res.x[2]),
           "hair_sf": float(res.x[3]),  "hat_sf": float(res.x[4]),  "other_bg_sf": float(res.x[5])}
    M_w = {"hair_bi": float(res.x[6]),  "hat_bi": float(res.x[7]),  "other_bi": float(res.x[8]),
           "hair_sf": float(res.x[9]),  "hat_sf": float(res.x[10]), "other_bg_sf": float(res.x[11])}
    coefficients = {"F": F_w, "M": M_w, "brief_score": float(res.fun)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coefficients, f, indent=2)
    print(f"saved coefficients to {out_path}")
    print()
    print("Coefficients found :")
    print(f"{'feature':<15} {'F':>10} {'M':>10}")
    for k in ["hair_bi", "hat_bi", "other_bi", "hair_sf", "hat_sf", "other_bg_sf"]:
        print(f"{k:<15} {F_w[k]:>+10.4f} {M_w[k]:>+10.4f}")


if __name__ == "__main__":
    main()
