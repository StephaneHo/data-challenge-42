"""Module standalone : scoring officiel + reweighting + formule v_features.

Self-contained (aucune dependance projet, juste numpy/pandas).
Julien peut copier ce fichier dans son notebook et l'utiliser directement.

5 metriques par distribution :
  err_F      : weighted MSE sur Femmes (w_i = 1/30 + GT_i)
  err_M      : weighted MSE sur Hommes
  mean_err   : (n_F * err_F + n_M * err_M) / (n_F + n_M)
               = moyenne ponderee par taille de population
  gap        : |err_F - err_M|
  score      : (err_F + err_M) / 2 + gap         <- METRIQUE OFFICIELLE

4 distributions :
  val natif        : distribution empirique du val
  reweighted brief : distribution officielle Telecom du brief
  reweighted spread: variante un peu plus large
  reweighted heavy : variante plus lourde sur cas hauts (worst case)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENDER_FEMALE = 0.0
GENDER_MALE = 1.0

OCC_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]

TEST_DISTRIBUTIONS = {
    "brief":  np.array([0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003]),
    "spread": np.array([0.10, 0.15, 0.15, 0.15, 0.25, 0.18, 0.020]),
    "heavy":  np.array([0.05, 0.10, 0.10, 0.15, 0.25, 0.30, 0.050]),
}


# =============================================================================
# Formules de base (utilise par tous les scoring methods)
# =============================================================================
def sample_weight(gt: np.ndarray) -> np.ndarray:
    """Poids par echantillon : w_i = 1/30 + GT_i."""
    return 1.0 / 30.0 + gt


def weighted_err(pred: np.ndarray, gt: np.ndarray) -> float:
    """Weighted MSE : sum_i w_i (p_i - GT_i)^2 / sum_i w_i.

    Retourne NaN si l'echantillon est vide.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(gt) == 0:
        return float("nan")
    w = sample_weight(gt)
    total_w = float(np.sum(w))
    if total_w == 0:
        return float("nan")
    return float(np.sum(w * (pred - gt) ** 2) / total_w)


def _result_dict(err_F: float, err_M: float, n_F: int, n_M: int) -> dict:
    """Construit le dict des 5 metriques + n par genre.

    Robuste a n_F=0 ou n_M=0 (mean_err et gap retournent NaN).
    """
    total_n = n_F + n_M
    if total_n == 0 or np.isnan(err_F) or np.isnan(err_M):
        return {"err_F": err_F, "err_M": err_M, "mean_err": float("nan"),
                "gap": float("nan"), "score": float("nan"), "n_F": n_F, "n_M": n_M}
    mean_err = (n_F * err_F + n_M * err_M) / total_n
    gap = abs(err_F - err_M)
    score = (err_F + err_M) / 2.0 + gap
    return {"err_F": err_F, "err_M": err_M, "mean_err": mean_err,
            "gap": gap, "score": score, "n_F": n_F, "n_M": n_M}


# =============================================================================
# Scoring sur val natif (distribution empirique du val)
# =============================================================================
def native_score(df: pd.DataFrame, pred_col: str = "pred", gt_col: str = "target",
                 gender_col: str = "gender") -> dict:
    F = df[df[gender_col] == GENDER_FEMALE]
    M = df[df[gender_col] == GENDER_MALE]
    err_F = weighted_err(F[pred_col].values, F[gt_col].values)
    err_M = weighted_err(M[pred_col].values, M[gt_col].values)
    return _result_dict(err_F, err_M, len(F), len(M))


# =============================================================================
# Scoring sous reweighted (simule une distribution test differente)
# =============================================================================
def _bin_index(values: np.ndarray, bins: list[float]) -> np.ndarray:
    idx = np.searchsorted(bins, values, side="right") - 1
    return np.clip(idx, 0, len(bins) - 2)


def reweighted_score(df: pd.DataFrame, target_bin_probs: np.ndarray,
                     pred_col: str = "pred", gt_col: str = "target",
                     gender_col: str = "gender", bins: list[float] = OCC_BINS) -> dict:
    """Score si val suivait la distribution `target_bin_probs`.

    Chaque sample est repondere par m_b = q_b / p_b, ou :
        p_b = fraction empirique val dans bin b (per gender)
        q_b = fraction cible dans bin b
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
    n_per_gender = {}
    for gval, glbl in ((GENDER_FEMALE, "F"), (GENDER_MALE, "M")):
        mask_g = gender == gval
        n_per_gender[glbl] = int(mask_g.sum())
        if mask_g.sum() == 0:
            err_per_gender[glbl] = float("nan")
            continue
        # Empirical bin probs for this gender
        bins_present = np.bincount(bin_idx[mask_g], minlength=len(bins) - 1)
        p = bins_present / mask_g.sum()
        # Mask out bins where val has no sample for this gender
        q_eff = np.where(p > 0, q, 0.0)
        if q_eff.sum() == 0:
            err_per_gender[glbl] = float("nan")
            continue
        q_eff = q_eff / q_eff.sum()
        # Per-sample multiplier
        m_per_bin = np.where(p > 0, q_eff / np.where(p > 0, p, 1.0), 0.0)
        m = m_per_bin[bin_idx]
        num = (m * w * sqerr)[mask_g].sum()
        den = (m * w)[mask_g].sum()
        err_per_gender[glbl] = float(num / den) if den > 0 else float("nan")

    return _result_dict(err_per_gender["F"], err_per_gender["M"],
                        n_per_gender["F"], n_per_gender["M"])


# =============================================================================
# Per-bin breakdown (diagnostique)
# =============================================================================
def per_bin_breakdown(df: pd.DataFrame, pred_col: str = "pred", gt_col: str = "target",
                      gender_col: str = "gender", bins: list[float] = OCC_BINS) -> pd.DataFrame:
    """Decomposition par (bin x genre) : n, mean_pred, mean_gt, bias, weighted_err."""
    pred = df[pred_col].to_numpy(dtype=np.float64)
    gt = df[gt_col].to_numpy(dtype=np.float64)
    gender = df[gender_col].to_numpy(dtype=np.float64)
    w = sample_weight(gt)
    sqerr = (pred - gt) ** 2
    wsqerr = w * sqerr
    bin_idx = _bin_index(gt, bins)
    labels = [f"[{a:.2f},{b:.2f})" for a, b in zip(bins[:-1], bins[1:])]

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
            slice_w = w[mask_b].sum()
            rows.append({
                "gender": glbl,
                "bin": labels[b],
                "n": n,
                "mean_pred": float(pred[mask_b].mean()),
                "mean_gt": float(gt[mask_b].mean()),
                "bias": float(pred[mask_b].mean() - gt[mask_b].mean()),
                "weighted_err": float(wsqerr[mask_b].sum() / slice_w) if slice_w > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Formule v_features (utilise par 02_optimize + 03_generate_submission)
# =============================================================================
def predict_v_features(df: pd.DataFrame, F_weights: dict, M_weights: dict,
                       gender_col: str = "gender") -> np.ndarray:
    """Calcule la prediction v_features pour chaque ligne d'un DataFrame.

    F_weights et M_weights doivent contenir les 6 cles :
      hair_bi, hat_bi, other_bi, hair_sf, hat_sf, other_bg_sf

    DataFrame doit contenir les colonnes :
      hair_bi_in_mask, hat_bi_in_mask, other_bi_in_mask,
      hair_sf_in_mask, hat_sf_in_mask, other_sf_in_mask,
      bg_sf_in_mask, gender

    Retourne un np.ndarray de predictions in [0, 1].
    """
    # Pre-cache les arrays
    hair_bi = df.hair_bi_in_mask.values
    hat_bi = df.hat_bi_in_mask.values
    other_bi = df.other_bi_in_mask.values
    hair_sf = df.hair_sf_in_mask.values
    hat_sf = df.hat_sf_in_mask.values
    other_bg_sf = (df.other_sf_in_mask + df.bg_sf_in_mask).values
    is_F = (df[gender_col].values == GENDER_FEMALE)

    # Prediction F
    pred_F = (F_weights["hair_bi"] * hair_bi + F_weights["hat_bi"] * hat_bi +
              F_weights["other_bi"] * other_bi + F_weights["hair_sf"] * hair_sf +
              F_weights["hat_sf"] * hat_sf + F_weights["other_bg_sf"] * other_bg_sf)
    # Prediction M
    pred_M = (M_weights["hair_bi"] * hair_bi + M_weights["hat_bi"] * hat_bi +
              M_weights["other_bi"] * other_bi + M_weights["hair_sf"] * hair_sf +
              M_weights["hat_sf"] * hat_sf + M_weights["other_bg_sf"] * other_bg_sf)

    return np.clip(np.where(is_F, pred_F, pred_M), 0, 1)


# =============================================================================
# Entry point : evaluation complete sur les 4 distributions
# =============================================================================
def evaluate_predictions(df: pd.DataFrame, pred_col: str = "pred",
                          gt_col: str = "target", gender_col: str = "gender",
                          verbose: bool = True) -> dict:
    """Evaluation complete : native + 3 reweighted + per-bin breakdown.

    Retourne :
      - native_val, reweighted_brief, reweighted_spread, reweighted_heavy
        (chacun avec err_F, err_M, mean_err, gap, score, n_F, n_M)
      - per_bin : DataFrame avec la decomposition par (bin x genre)
    """
    out = {}
    out["native_val"] = native_score(df, pred_col, gt_col, gender_col)
    for name, dist in TEST_DISTRIBUTIONS.items():
        out[f"reweighted_{name}"] = reweighted_score(df, dist, pred_col, gt_col, gender_col)
    out["per_bin"] = per_bin_breakdown(df, pred_col, gt_col, gender_col)

    if verbose:
        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nSample size: {len(df)} (F={out['native_val']['n_F']}, M={out['native_val']['n_M']})")
        print()
        print(f"{'distribution':<18} {'err_F':>10} {'err_M':>10} {'mean_err':>10} {'gap':>10} {'score':>10}")
        print(f"{'-' * 80}")
        for name in ["native_val", "reweighted_brief", "reweighted_spread", "reweighted_heavy"]:
            s = out[name]
            label = name.replace("native_val", "val natif").replace("reweighted_", "")
            print(f"{label:<18} {s['err_F']:>10.5f} {s['err_M']:>10.5f} "
                  f"{s['mean_err']:>10.5f} {s['gap']:>10.5f} {s['score']:>10.5f}")
        print()
        print("Per-bin breakdown :")
        print(out["per_bin"].round(4).to_string(index=False))

    return out


def main():
    """CLI : evaluation autonome a partir d'un CSV de predictions."""
    p = argparse.ArgumentParser(description="Evaluate predictions with reweighting (5 columns)")
    p.add_argument("--predictions", required=True,
                   help="CSV avec : filename + pred (predictions du modele)")
    p.add_argument("--labels", required=True,
                   help="CSV avec : filename + target + gender (ground truth)")
    p.add_argument("--pred-col", default="pred",
                   help="Nom de la colonne de prediction (default: 'pred')")
    args = p.parse_args()

    pred_df = pd.read_csv(args.predictions)
    label_df = pd.read_csv(args.labels)

    if args.pred_col != "pred":
        pred_df = pred_df.rename(columns={args.pred_col: "pred"})

    df = pred_df.merge(label_df[["filename", "target", "gender"]], on="filename")
    if len(df) < len(pred_df):
        print(f"WARN: only {len(df)}/{len(pred_df)} predictions matched a label.")
    evaluate_predictions(df, verbose=True)


if __name__ == "__main__":
    main()
