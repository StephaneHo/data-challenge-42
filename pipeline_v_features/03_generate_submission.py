"""Etape 3 : Genere le CSV de submission a partir du cache test et des coefficients.

Applique la formule v_features (12 poids per-gender) sur chaque image du test,
et sauve les predictions au format attendu par hfactory.

Usage :
    python 03_generate_submission.py \\
        --test-cache eval/cache/test_features.csv \\
        --coefficients pipeline_v_features/coefficients.json \\
        --gender-cache eval/cache/test_gender_pred.csv \\
        --out submission.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Formule v_features (predict_v_features) est dans evaluate.py
from evaluate import predict_v_features

# Fallback prediction pour les images sans features (face detection failed).
# = moyenne empirique de target sur val (15001 images).
FALLBACK_PRED = 0.082


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-cache", required=True, help="Path to test_features.csv")
    p.add_argument("--coefficients", required=True, help="Path to coefficients.json")
    p.add_argument("--gender-cache", required=True,
                   help="Path to test_gender_pred.csv (InsightFace gender on test)")
    p.add_argument("--out", required=True, help="Path to save submission.csv")
    args = p.parse_args()

    # === Load test features ===
    test_df = pd.read_csv(args.test_cache)
    print(f"loaded test features: {len(test_df)} rows")

    # === Load gender (test n'a pas de gender GT, on utilise InsightFace) ===
    gender_df = pd.read_csv(args.gender_cache)
    if "pred_gender" in gender_df.columns:
        gender_df = gender_df.rename(columns={"pred_gender": "gender"})
    print(f"loaded gender cache: {len(gender_df)} rows")

    # === Merge sur filename ===
    df = test_df.merge(gender_df[["filename", "gender"]], on="filename", how="left")
    n_missing_g = df.gender.isna().sum()
    if n_missing_g > 0:
        print(f"WARN: {n_missing_g} images sans gender, fallback = M (majoritaire)")
        df["gender"] = df.gender.fillna(1.0)

    # === Load coefficients ===
    with open(args.coefficients) as f:
        coeffs = json.load(f)
    F_w = coeffs["F"]
    M_w = coeffs["M"]
    print(f"loaded coefficients (val brief score : {coeffs['brief_score']:.5f})")

    # === Predict (avec fallback sur images sans features) ===
    df["FaceOcclusion"] = float("nan")
    valid_mask = df["hair_bi_in_mask"].notna()
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"WARN: {n_invalid} images sans features (face detection failed)")
        print(f"      Their prediction = {FALLBACK_PRED} (mean val target)")

    if valid_mask.any():
        df.loc[valid_mask, "FaceOcclusion"] = predict_v_features(df[valid_mask], F_w, M_w)
    if n_invalid > 0:
        df.loc[~valid_mask, "FaceOcclusion"] = FALLBACK_PRED

    # === Output : filename, FaceOcclusion, gender (= 'x' format hfactory) ===
    out = df[["filename", "FaceOcclusion"]].copy()
    out["gender"] = "x"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print()
    print(f"saved submission to {out_path} ({len(out)} rows, NaN={out.FaceOcclusion.isna().sum()})")
    print()
    print("Distribution of predictions :")
    print(f"  mean   : {out.FaceOcclusion.mean():.4f}")
    print(f"  median : {out.FaceOcclusion.median():.4f}")
    print(f"  std    : {out.FaceOcclusion.std():.4f}")
    print(f"  min    : {out.FaceOcclusion.min():.4f}")
    print(f"  max    : {out.FaceOcclusion.max():.4f}")
    print()
    print("Distribution par bin :")
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    for i in range(len(edges) - 1):
        mask = (out.FaceOcclusion >= edges[i]) & (out.FaceOcclusion < edges[i + 1])
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.2f}) : {mask.sum():>6} ({100*mask.mean():.1f}%)")


if __name__ == "__main__":
    main()
