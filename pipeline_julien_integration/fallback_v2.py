"""V2 simple fallback pour la pipeline de Julien.

Aligne sur les conventions du notebook `notebooks/DataChallengeJulien_TF.ipynb` :
  - Variables : visible_pixels_b/s, hair_ratio_b/s, hat_ratio_b/s, other_ratio_b/s
  - Coefficients : M_WEIGHTS / F_WEIGHTS avec cles hair_bi, hat_bi, other_bi, hair_sf, hat_sf, other_bg_sf
  - Gender : 'M' / 'F' (sortie de face.sex InsightFace)
  - Indices :
      VISIBLE_FACE_CLASSES_B = [0,1,2,3,4,5,6,10,11,12,13]   (bg + skin + eye_g + ...)
      VISIBLE_FACE_CLASSES_S = [1,2,3,4,5,6,7,10,11,12]      (skin + eye_g + ... PAS bg)

PROBLEME RESOLU
---------------
Quand un des deux modeles plante (BiSeNet sature en bg, ou SegFormer voit aucune peau),
la formule hybride donne des predictions catastrophiques (target=0 -> pred=1.0).

Le fallback detecte ces cas et bascule sur le modele qui marche.

USAGE DANS LE NOTEBOOK
----------------------
3 changements legers dans `occlusion_computation()` :

  1. AJOUTER au debut du fichier (avec les autres VISIBLE_FACE_CLASSES) :
     SKIN_ONLY_B = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # skin sans bg sans eye_g
     SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 10, 11, 12]   # skin sans eye_g (bg deja absent)
     BG_B        = [0]

  2. APRES ton calcul de visible_pixels_b/s, AJOUTER :
     skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
     bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
     skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
     skin_only_b_ratio = np.sum(skin_only_b & mask_theoretical) / total_pixels
     bg_b_ratio        = np.sum(bg_b        & mask_theoretical) / total_pixels
     skin_only_s_ratio = np.sum(skin_only_s & mask_theoretical) / total_pixels

  3. REMPLACER ton bloc `if pred_gender == 'M' / elif 'F'` par UN SEUL APPEL :

     from pipeline_julien_integration.fallback_v2 import apply_v2_fallback
     occlusion_score = apply_v2_fallback(
         pred_gender=pred_gender,
         hair_ratio_b=hair_ratio_b, hat_ratio_b=hat_ratio_b, other_ratio_b=other_ratio_b,
         hair_ratio_s=hair_ratio_s, hat_ratio_s=hat_ratio_s, other_ratio_s=other_ratio_s,
         skin_only_b_ratio=skin_only_b_ratio,
         bg_b_ratio=bg_b_ratio,
         skin_only_s_ratio=skin_only_s_ratio,
         M_WEIGHTS=M_WEIGHTS, F_WEIGHTS=F_WEIGHTS,
     )

Voila. Aucun changement de coefficients, aucun changement de ta formule principale,
juste 3 nouvelles features (skin_only_b, bg_b, skin_only_s) pour detecter le plantage.
"""
from __future__ import annotations

import numpy as np

# Seuils de detection plantage (calibres sur val, robustes en CV 5/5 GAIN)
BI_PLANTE_THRESHOLD = 0.70   # bg_b_ratio > 0.70 -> BiSeNet sature en bg
SF_PLANTE_THRESHOLD = 0.30   # skin_only_s_ratio < 0.30 -> SegFormer ne voit pas la peau


def _formule_complete(hair_ratio_b, hat_ratio_b, other_ratio_b,
                       hair_ratio_s, hat_ratio_s, other_ratio_s,
                       W):
    """La formule habituelle de Julien (somme ponderee des 6 features)."""
    return (W["hair_bi"] * hair_ratio_b + W["hat_bi"] * hat_ratio_b +
            W["other_bi"] * other_ratio_b + W["hair_sf"] * hair_ratio_s +
            W["hat_sf"] * hat_ratio_s + W["other_bg_sf"] * other_ratio_s)


def _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, W):
    """Contribution BiSeNet seule (3 termes)."""
    return W["hair_bi"] * hair_ratio_b + W["hat_bi"] * hat_ratio_b + W["other_bi"] * other_ratio_b


def _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, W):
    """Contribution SegFormer seule (3 termes)."""
    return W["hair_sf"] * hair_ratio_s + W["hat_sf"] * hat_ratio_s + W["other_bg_sf"] * other_ratio_s


def apply_v2_fallback(pred_gender,
                      hair_ratio_b, hat_ratio_b, other_ratio_b,
                      hair_ratio_s, hat_ratio_s, other_ratio_s,
                      skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio,
                      M_WEIGHTS, F_WEIGHTS) -> float:
    """V2 simple fallback : prediction d'occlusion avec gestion des plantages.

    Logique :
      - Si SegFormer plante (skin_only_s_ratio < 0.30) et BiSeNet OK  -> contribution BiSeNet seule
      - Si BiSeNet plante  (bg_b_ratio > 0.70)        et SegFormer OK -> contribution SegFormer seule
      - Si les 2 plantent -> celui qui est "le moins pire" (saturation la plus basse)
      - Si aucun ne plante (98% des images) -> formule complete habituelle

    Args:
        pred_gender: 'M' ou 'F' (sortie InsightFace face.sex)
        hair_ratio_b/s, hat_ratio_b/s, other_ratio_b/s: tes ratios habituels
        skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio: NOUVELLES features pour detection plante
        M_WEIGHTS, F_WEIGHTS: tes dicts de coefficients existants

    Returns:
        occlusion_score (float clippe entre 0 et 1)
    """
    # Selection des poids selon le genre predit
    if pred_gender == 'M':
        W = M_WEIGHTS
    elif pred_gender == 'F':
        W = F_WEIGHTS
    else:
        # Fallback : on prend la moyenne si pred_gender est inattendu
        W = {k: (M_WEIGHTS[k] + F_WEIGHTS[k]) / 2.0 for k in M_WEIGHTS}

    # Detection plantage (independant du genre)
    bi_plante = bg_b_ratio > BI_PLANTE_THRESHOLD
    sf_plante = skin_only_s_ratio < SF_PLANTE_THRESHOLD

    if sf_plante and not bi_plante:
        # SegFormer plante seul -> BiSeNet uniquement
        score = _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, W)
    elif bi_plante and not sf_plante:
        # BiSeNet plante seul -> SegFormer uniquement
        score = _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, W)
    elif bi_plante and sf_plante:
        # Les 2 plantent : "le moins pire" = celui dont le score de plantage est le plus bas
        # bg_b_ratio = score plantage BiSeNet  (haut = plus plante)
        # 1 - skin_only_s_ratio = score plantage SegFormer (haut = plus plante)
        bi_plantage_score = bg_b_ratio
        sf_plantage_score = 1.0 - skin_only_s_ratio
        if bi_plantage_score < sf_plantage_score:
            score = _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, W)
        else:
            score = _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, W)
    else:
        # Cas normal (98% des images) : formule complete habituelle
        score = _formule_complete(hair_ratio_b, hat_ratio_b, other_ratio_b,
                                   hair_ratio_s, hat_ratio_s, other_ratio_s, W)

    return float(np.clip(score, 0.0, 1.0))


def detect_plante_regime(skin_only_b_ratio: float, bg_b_ratio: float,
                          skin_only_s_ratio: float) -> str:
    """Retourne le regime de plantage detecte. Utile pour le diagnostic / debug.

    Returns:
        "normal" / "sf_only" / "bi_only" / "both"
    """
    bi_plante = bg_b_ratio > BI_PLANTE_THRESHOLD
    sf_plante = skin_only_s_ratio < SF_PLANTE_THRESHOLD
    if sf_plante and not bi_plante:
        return "sf_only"
    if bi_plante and not sf_plante:
        return "bi_only"
    if bi_plante and sf_plante:
        return "both"
    return "normal"
