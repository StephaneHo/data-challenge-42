"""=============================================================================
Calcul des 12 coefficients de la formule v_features par REGRESSION LINEAIRE
=============================================================================

VERSION SIMPLE et EXPLICABLE de optimize_12_coefficients.py.

Methode : Regression Lineaire Ponderee (WLS = Weighted Least Squares).

C'est la methode standard de cours de stats. On cherche les 12 coefficients
qui minimisent la somme des carres des erreurs ponderee :

    Loss(w) = Sum_i sample_weight_i * (pred_i - target_i)^2

ou pred_i est la formule lineaire de Julien :

    pred = w_hair_bi * hair_ratio_b + w_hat_bi * hat_ratio_b + w_other_bi * other_ratio_b
         + w_hair_sf * hair_ratio_s + w_hat_sf * hat_ratio_s + w_other_bg_sf * other_ratio_s

Et le poids par sample :
    sample_weight_i = 1/30 + target_i    (= le poids du score officiel IDEMIA)

On fait UNE regression par genre (F et M) pour avoir 6 coefficients chacun, soit 12 au total.

=============================================================================
POURQUOI CETTE METHODE EST SIMPLE
=============================================================================
- SOLUTION EN FORMULE FERMEE (pas d'algo iteratif comme Nelder-Mead) :
    w* = (X^T diag(w_sample) X)^-1 X^T diag(w_sample) y
- sklearn fait cette formule en UNE LIGNE :
    LinearRegression(fit_intercept=False).fit(X, y, sample_weight=...)
- Reproductibilite parfaite : solution UNIQUE et DETERMINISTE
- Aucune dependance a un initial guess ou un nombre d'iterations

=============================================================================
LIMITATIONS (par rapport a Nelder-Mead)
=============================================================================
- N'optimise PAS directement le score officiel IDEMIA :
    Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|
    -> le |Err_F - Err_M| (penalite de fairness entre genres) n'est pas pris en compte
- N'optimise PAS le reweighting par bin de target (bins du brief)
- Resultat : score brief un peu moins bon (~+10%)

Mais c'est BEAUCOUP plus simple a comprendre et expliquer.

=============================================================================
USAGE
=============================================================================
    python optimize_12_coefficients_simple.py \\
        --val-cache eval/cache/val_features.csv \\
        --out coefficients_lineaire.json

Pour utiliser sur un subset (par exemple 3000 images) :
    python optimize_12_coefficients_simple.py \\
        --val-cache eval/cache/val_features.csv \\
        --n-subset 3000 \\
        --out coefficients_lineaire_3000.json

============================================================================="""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le DataFrame des 6 features de la formule v_features.

    Returns :
        DataFrame avec 6 colonnes :
            hair_bi, hat_bi, other_bi   (BiSeNet)
            hair_sf, hat_sf, other_bg_sf (SegFormer ; other_bg_sf = other_sf + bg_sf)
    """
    X = pd.DataFrame({
        "hair_bi":     df.hair_bi_in_mask.values,
        "hat_bi":      df.hat_bi_in_mask.values,
        "other_bi":    df.other_bi_in_mask.values,
        "hair_sf":     df.hair_sf_in_mask.values,
        "hat_sf":      df.hat_sf_in_mask.values,
        "other_bg_sf": (df.other_sf_in_mask + df.bg_sf_in_mask).values,
    })
    return X


def fit_weights_per_gender(df_gender: pd.DataFrame) -> dict:
    """Fit une regression lineaire ponderee sur un sous-ensemble de meme genre.

    Args:
        df_gender: DataFrame filtre sur un seul genre (gender == 0 pour F, == 1 pour M)

    Returns:
        dict de 6 coefficients (cles : hair_bi, hat_bi, other_bi, hair_sf, hat_sf, other_bg_sf)
    """
    # Construction des features (X) et target (y)
    X = build_feature_matrix(df_gender)
    y = df_gender.target.values

    # Poids par sample (= poids du score officiel IDEMIA : 1/30 + target)
    sample_weights = 1.0 / 30.0 + y

    # === Regression lineaire ponderee (= WLS = Weighted Least Squares) ===
    # fit_intercept=False car la formule de Julien n'a pas de terme constant
    # (sinon pred serait pred = intercept + w*X, ce qu'on ne veut pas)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y, sample_weight=sample_weights)

    # Construire le dict {feature_name: coefficient}
    return dict(zip(X.columns, reg.coef_.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache", required=True, help="Path to val_features.csv")
    parser.add_argument("--out", required=True, help="Path to save coefficients JSON")
    parser.add_argument("--n-subset", type=int, default=None,
                        help="Si spécifié, utilise un sous-ensemble aléatoire de N images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # === Charge val cache ===
    df = pd.read_csv(args.val_cache)
    print(f"Loaded {len(df)} rows from {args.val_cache}")
    df = df.dropna(subset=["hair_bi_in_mask"]).reset_index(drop=True)
    print(f"  After dropping NaN : {len(df)} rows")

    # === Sub-sample si demande ===
    if args.n_subset is not None and args.n_subset < len(df):
        df = df.sample(n=args.n_subset, random_state=args.seed).reset_index(drop=True)
        print(f"  Subset random : {len(df)} rows (seed={args.seed})")

    n_F = (df.gender == 0.0).sum()
    n_M = (df.gender == 1.0).sum()
    print(f"  F = {n_F}, M = {n_M}")
    print()

    # === Regression separee par genre ===
    print("Fitting weighted linear regression for FEMALES (gender == 0)...")
    F_w = fit_weights_per_gender(df[df.gender == 0.0])
    print("Fitting weighted linear regression for MALES (gender == 1)...")
    M_w = fit_weights_per_gender(df[df.gender == 1.0])
    print()

    # === Affichage ===
    print("=" * 60)
    print("12 COEFFICIENTS CALCULES PAR REGRESSION LINEAIRE PONDEREE")
    print("=" * 60)
    print(f"{'feature':<15} {'F (gender=0)':>15} {'M (gender=1)':>15}")
    print("-" * 60)
    for k in ["hair_bi", "hat_bi", "other_bi", "hair_sf", "hat_sf", "other_bg_sf"]:
        print(f"{k:<15} {F_w[k]:>+15.4f} {M_w[k]:>+15.4f}")
    print()

    # === Save ===
    out = {"F": F_w, "M": M_w, "method": "weighted_linear_regression"}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved : {out_path}")
    print()
    print("Note : ces coefficients minimisent la somme ponderee des carres d'erreur,")
    print("pas le score brief reweighted IDEMIA exactement. Resultat attendu :")
    print("  brief ~ 0.0065 (vs 0.00594 avec Nelder-Mead sur le score brief direct).")


if __name__ == "__main__":
    main()
