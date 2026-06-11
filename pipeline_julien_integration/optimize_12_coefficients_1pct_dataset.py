"""=============================================================================
Petite etude statistique sur 1% du dataset pour estimer les 12 coefficients
de la formule v_features.
=============================================================================

CONTEXTE :
---------
On a construit un pipeline d'evaluation qui :
  1. Tire une image au hasard du dataset val
  2. Calcule les features par segmentation (BiSeNet, SegFormer, 3DDFA-V2)
  3. Predit l'occlusion par formule lineaire ponderee

En tirant successivement des images, on observe certains biais :
  - Sur-prediction sur les images avec beaucoup de cheveux ou chapeaux
  - Sous-prediction sur les images avec accessoires (lunettes, micros)
  - Distributions de prediction differentes selon le genre

Pour corriger ces biais, on fait une *petite etude statistique* sur ~1% du
dataset (150 images sur 15K val), en restant dans l'esprit "training-free" :
on n'entraine aucun modele, on calcule juste des moyennes ponderees.

METHODE :
---------
Pour chaque genre et chaque feature, on calcule la pente d'une regression
lineaire 1D simple :

    slope_k = Σᵢ (Xᵢ × yᵢ) / Σᵢ Xᵢ²
    pred = slope_hair_bi × hair_bi + slope_hat_bi × hat_bi + ... (6 termes par genre)

C'est la formule de la pente de la droite y = slope × X qui minimise les erreurs
au carre. Demonstrable en 3 lignes :
    Loss = Σᵢ (slope × Xᵢ - yᵢ)²
    Derivee = 2 Σᵢ Xᵢ × (slope × Xᵢ - yᵢ) = 0
    --> slope = Σᵢ (Xᵢ × yᵢ) / Σᵢ Xᵢ²

USAGE :
-------
    python optimize_12_coefficients_1pct_dataset.py
        --val-cache eval/cache/val_features.csv
        --out coefficients_1pct.json
============================================================================="""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Taille de l'echantillon = 1% du dataset val (~150 sur 15K)
SAMPLE_FRACTION = 0.01  # 1% du dataset
RANDOM_SEED = 42


def slope_1d(X: np.ndarray, y: np.ndarray) -> float:
    """Pente de la regression lineaire 1D : slope = sum(X*y) / sum(X*X)."""
    den = float(np.sum(X * X))
    if den == 0:
        return 0.0
    return float(np.sum(X * y)) / den


def estimate_coefficients_from_sample(sample: pd.DataFrame) -> dict:
    """Estime les 12 coefficients sur un sous-echantillon par regression 1D
    feature-par-feature, separement pour chaque genre.
    """
    features = {
        "hair_bi":     "hair_bi_in_mask",
        "hat_bi":      "hat_bi_in_mask",
        "other_bi":    "other_bi_in_mask",
        "hair_sf":     "hair_sf_in_mask",
        "hat_sf":      "hat_sf_in_mask",
        "other_bg_sf": None,   # combine apres
    }

    coefs = {}
    for gender, label in [(0.0, "F"), (1.0, "M")]:
        sub = sample[sample.gender == gender]
        coefs[label] = {}
        if len(sub) < 5:
            print(f"  ATTENTION : genre {label} a seulement {len(sub)} images dans l'echantillon !")
        for feature_name, col in features.items():
            if feature_name == "other_bg_sf":
                X = (sub.other_sf_in_mask + sub.bg_sf_in_mask).values
            else:
                X = sub[col].values
            y = sub.target.values
            coefs[label][feature_name] = slope_1d(X, y)
    return coefs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache", required=True, help="Path to val_features.csv")
    parser.add_argument("--out", required=True, help="Path to save coefficients JSON")
    parser.add_argument("--fraction", type=float, default=SAMPLE_FRACTION,
                        help=f"Fraction du dataset a echantillonner (default {SAMPLE_FRACTION})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # === Charge val cache ===
    df = pd.read_csv(args.val_cache)
    df = df.dropna(subset=["hair_bi_in_mask"]).reset_index(drop=True)
    n_full = len(df)
    print(f"Dataset complet : {n_full} images")

    # === Tirage au hasard de 1% du dataset ===
    n_sample = max(50, int(round(n_full * args.fraction)))
    sample = df.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    n_F = (sample.gender == 0.0).sum()
    n_M = (sample.gender == 1.0).sum()
    print(f"Echantillon : {n_sample} images ({args.fraction*100:.1f}% du dataset), "
          f"F={n_F}, M={n_M}, seed={args.seed}")
    print()

    # === Estimation des coefficients ===
    print(f"Estimation des 12 coefficients (regression 1D par feature, par genre)...")
    print(f"Methode : slope_k = sum(X_k * y) / sum(X_k * X_k) sur l'echantillon")
    print()
    coefs = estimate_coefficients_from_sample(sample)

    # === Affichage ===
    print("=" * 60)
    print(f"12 COEFFICIENTS ESTIMES sur {n_sample} images ({args.fraction*100:.1f}% du dataset)")
    print("=" * 60)
    print(f"{'feature':<15} {'F (gender=0)':>15} {'M (gender=1)':>15}")
    print("-" * 60)
    for k in ["hair_bi", "hat_bi", "other_bi", "hair_sf", "hat_sf", "other_bg_sf"]:
        print(f"{k:<15} {coefs['F'][k]:>+15.4f} {coefs['M'][k]:>+15.4f}")
    print()

    # === Stabilite : refaire avec plusieurs seeds pour voir la variance ===
    print("=" * 60)
    print(f"STABILITE : on refait l'estimation avec 5 seeds differentes")
    print("=" * 60)
    print(f"{'feature':<15} {'F mean':>10} {'F std':>10} {'M mean':>10} {'M std':>10}")
    print("-" * 60)
    seeds = [42, 123, 456, 789, 1000]
    all_coefs = {"F": {k: [] for k in coefs["F"]}, "M": {k: [] for k in coefs["M"]}}
    for s in seeds:
        sample_s = df.sample(n=n_sample, random_state=s).reset_index(drop=True)
        c_s = estimate_coefficients_from_sample(sample_s)
        for label in ["F", "M"]:
            for k in c_s[label]:
                all_coefs[label][k].append(c_s[label][k])
    for k in ["hair_bi", "hat_bi", "other_bi", "hair_sf", "hat_sf", "other_bg_sf"]:
        F_mean = np.mean(all_coefs["F"][k]); F_std = np.std(all_coefs["F"][k])
        M_mean = np.mean(all_coefs["M"][k]); M_std = np.std(all_coefs["M"][k])
        print(f"{k:<15} {F_mean:>+10.4f} {F_std:>10.4f} {M_mean:>+10.4f} {M_std:>10.4f}")
    print()

    # === Save ===
    out = {
        "F": coefs["F"],
        "M": coefs["M"],
        "method": "per_feature_1d_regression",
        "fraction_used": args.fraction,
        "n_sample": n_sample,
        "seed": args.seed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved : {out_path}")


if __name__ == "__main__":
    main()
