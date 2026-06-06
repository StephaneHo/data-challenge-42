"""Etape 2 : Optimisation des 12 coefficients de la pipeline v_features.

Charge le cache val_features.csv (produit par 01_cache_features.py) et cherche
les 12 poids (6 features x 2 genres) qui minimisent le SCORE OFFICIEL :
    Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|
sous la distribution test reweighted "brief" (officielle Telecom).

Output : coefficients.json + rapport d'evaluation complet (5 colonnes).

Usage :
    python 02_optimize_coefficients.py \\
        --val-cache eval/cache/val_features.csv \\
        --out pipeline_v_features/coefficients.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Tous les utilities sont dans evaluate.py (self-contained)
from evaluate import (
    TEST_DISTRIBUTIONS,
    evaluate_predictions,
    predict_v_features,
    reweighted_score,
)


def objective_brief(params: np.ndarray, df: pd.DataFrame) -> float:
    """Fonction objectif : score brief reweighted pour les 12 params donnees."""
    F_w = {"hair_bi": params[0], "hat_bi": params[1], "other_bi": params[2],
           "hair_sf": params[3], "hat_sf": params[4], "other_bg_sf": params[5]}
    M_w = {"hair_bi": params[6], "hat_bi": params[7], "other_bi": params[8],
           "hair_sf": params[9], "hat_sf": params[10], "other_bg_sf": params[11]}
    # On modifie la colonne pred du df une fois pour eviter le copy
    df.loc[:, "pred"] = predict_v_features(df, F_w, M_w)
    return reweighted_score(df, TEST_DISTRIBUTIONS["brief"], pred_col="pred",
                            gt_col="target")["score"]


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
    print(f"  working on {len(df)} rows (F={(df.gender == 0.0).sum()}, "
          f"M={(df.gender == 1.0).sum()})")
    # Pre-allocate pred column to avoid copy in optimization loop
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

    # === Optimization ===
    print()
    print("Optimizing 12 coefficients (Nelder-Mead)...")
    print("Objective : reweighted_brief Score (official metric)")
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

    # === Full evaluation report (5 columns x 4 distributions) ===
    df["pred"] = predict_v_features(df, F_w, M_w)
    evaluate_predictions(df, pred_col="pred", verbose=True)


if __name__ == "__main__":
    main()
