"""=============================================================================
Calcul des 12 coefficients par REGRESSION LINEAIRE PAR FEATURE (independante)
=============================================================================

VERSION LA PLUS PEDAGOGIQUE POSSIBLE.

Methode : pour chaque feature, on fait une regression lineaire 1D independante
sur cette feature seule. Le coefficient associe = la pente de la droite qui
minimise la somme des carres des erreurs entre les points (feature, target).

C'est la formule la plus simple de cours de stats, demontrable en 3 lignes :

    On cherche : pred = slope * X
    On minimise : Σᵢ (slope * Xᵢ - yᵢ)²
    Derivee par rapport a slope : Σᵢ 2 * Xᵢ * (slope * Xᵢ - yᵢ) = 0
    --> slope * Σᵢ Xᵢ² = Σᵢ Xᵢ * yᵢ
    --> slope = Σᵢ (Xᵢ * yᵢ) / Σᵢ Xᵢ²

C'est tout. Pas de matrice, pas de bibliotheque magique, juste des sommes.

=============================================================================
INTERPRETATION DE CHAQUE COEFFICIENT
=============================================================================
Le coefficient pour la feature X_k repond a la question :
    "Si X_k augmente de 0.1 (10% de plus), de combien augmente le target ?"

Reponse : 0.1 × slope_k

C'est intuitif et chaque coefficient a un sens ISOLEMENT (sans dependre des autres).

=============================================================================
LIMITATION
=============================================================================
Cette methode IGNORE les correlations entre features. Si hair_bi et hair_sf
sont correles (les 2 modeles voient les memes cheveux), chaque slope va
sur-estimer la contribution de sa feature.

Resultat : coefficients un peu trop gros, score brief plus eleve qu'avec
sklearn LinearRegression (qui prend en compte les correlations) ou Nelder-Mead
(qui optimise directement le score brief).

Mais c'est INCROYABLEMENT PEDAGOGIQUE.

=============================================================================
USAGE
=============================================================================
    python optimize_12_coefficients_pedagogique.py \\
        --val-cache eval/cache/val_features.csv \\
        --out coefficients_pedagogique.json

============================================================================="""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# La methode pedagogique : formule de la pente, feature par feature
# =============================================================================
def slope_minimum_carres(X: np.ndarray, y: np.ndarray) -> float:
    """Calcule la pente de la droite y = slope * X qui minimise Σ (slope*X - y)².

    C'est la formule de la regression lineaire 1D sans intercept :
        slope = Σ(X * y) / Σ(X * X)

    Demonstration :
        Loss(slope) = Σᵢ (slope * Xᵢ - yᵢ)²
        Derivee par rapport a slope : 2 Σᵢ Xᵢ (slope * Xᵢ - yᵢ) = 0
        --> slope * Σᵢ Xᵢ² = Σᵢ (Xᵢ * yᵢ)
        --> slope = Σᵢ (Xᵢ * yᵢ) / Σᵢ Xᵢ²
    """
    numerateur = float(np.sum(X * y))
    denominateur = float(np.sum(X * X))
    if denominateur == 0:
        return 0.0
    return numerateur / denominateur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache", required=True, help="Path to val_features.csv")
    parser.add_argument("--out", required=True, help="Path to save coefficients JSON")
    args = parser.parse_args()

    # === Charge val cache ===
    df = pd.read_csv(args.val_cache)
    df = df.dropna(subset=["hair_bi_in_mask"]).reset_index(drop=True)
    print(f"Loaded {len(df)} val images")
    print()

    # Pre-calcul other_bg_sf (= other_sf + bg_sf, comme dans la formule de Julien)
    df["other_bg_sf_combined"] = df.other_sf_in_mask + df.bg_sf_in_mask

    # Mapping feature -> colonne du DataFrame
    features = {
        "hair_bi":     "hair_bi_in_mask",
        "hat_bi":      "hat_bi_in_mask",
        "other_bi":    "other_bi_in_mask",
        "hair_sf":     "hair_sf_in_mask",
        "hat_sf":      "hat_sf_in_mask",
        "other_bg_sf": "other_bg_sf_combined",
    }

    # === Pour chaque genre, calcule chaque coefficient INDEPENDAMMENT ===
    coefs = {}
    for gender, label in [(0.0, "F"), (1.0, "M")]:
        sub = df[df.gender == gender]
        print(f"=== Genre {label} ({len(sub)} images) ===")
        coefs[label] = {}
        for feature_name, column in features.items():
            X = sub[column].values
            y = sub.target.values
            slope = slope_minimum_carres(X, y)
            coefs[label][feature_name] = slope
            print(f"  {feature_name:<15} : slope = sum(X*y) / sum(X*X) = "
                  f"{np.sum(X*y):.2f} / {np.sum(X*X):.2f} = {slope:+.4f}")
        print()

    # === Affichage final ===
    print("=" * 70)
    print("12 COEFFICIENTS (regression lineaire par feature, INDEPENDANTE)")
    print("=" * 70)
    print(f"{'feature':<15} {'F (gender=0)':>15} {'M (gender=1)':>15}")
    print("-" * 70)
    for k in ["hair_bi", "hat_bi", "other_bi", "hair_sf", "hat_sf", "other_bg_sf"]:
        print(f"{k:<15} {coefs['F'][k]:>+15.4f} {coefs['M'][k]:>+15.4f}")
    print()

    # === Save ===
    out = {"F": coefs["F"], "M": coefs["M"],
           "method": "per_feature_independent_linear_regression"}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved : {out_path}")
    print()
    print("Note pedagogique :")
    print("  Chaque coefficient est calcule comme la pente d'une regression 1D")
    print("  independante : c'est la formule slope = sum(X*y) / sum(X*X).")
    print("  Cette methode ignore les correlations entre features, donc les")
    print("  coefficients sont surestimes par rapport a la 'vraie' regression")
    print("  multi-feature, mais ils restent INTERPRETABLES isolement.")


if __name__ == "__main__":
    main()
