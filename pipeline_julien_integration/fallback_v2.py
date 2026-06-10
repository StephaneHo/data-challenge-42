"""
=============================================================================
V2 simple fallback pour la pipeline de Julien
=============================================================================

OBJECTIF
--------
Sur ~245 images du val (sur 15001, donc ~1.6%), un des deux modeles de face-parsing
PLANTE catastrophiquement :

  - BiSeNet plante  : sur images B&W / vintage / floues / basses resolution.
                      Le modele classe quasi tout en "background" (classe 0).
                      Signal : bg_b_ratio > 0.70 (le bg domine le mask).

  - SegFormer plante : sur images tres desaturees / atypiques.
                       Le modele voit quasi aucune peau.
                       Signal : skin_only_s_ratio < 0.30 (peu de skin pur).

Quand un modele plante, la formule hybride (somme des contributions des 2 modeles)
est dominee par les valeurs absurdes du modele en panne. Resultat : predictions
catastrophiques (target=0 mais pred=1.0).

CE QUE FAIT LE FALLBACK
-----------------------
Detecte ces cas et bascule sur le modele qui marche :
  - SF plante seul -> on n'utilise QUE la contribution BiSeNet (3 termes au lieu de 6)
  - BI plante seul -> on n'utilise QUE la contribution SegFormer
  - Les 2 plantent -> on prend celui qui sature le moins ("le moins pire")
  - Aucun ne plante (~98% des cas) -> formule complete habituelle (identique a Julien)

RESULTAT
--------
CV 5/5 folds GAIN sur val, gain attendu sur test brief ~-10%.
Aucun coefficient nouveau a apprendre, aucun risque overfit.

VOIR README.md POUR LE GUIDE D'INTEGRATION (4 etapes dans l'ordre).
=============================================================================
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# SEUILS DE DETECTION PLANTAGE
# =============================================================================
# Calibres sur notre cache val_features.csv (15K images) en CV 5/5 folds GAIN.
#
# bg_b_ratio > 0.70 :
#   bg_b_ratio = fraction du mask_theoretical classifiee comme bg (classe 0) par BiSeNet.
#   Sur images normales, bg_b_ratio < 0.10 (BiSeNet ne devrait pas classer le visage en bg).
#   Si bg_b_ratio depasse 0.70, c'est que BiSeNet a etiquette ~tout le visage en bg
#   -> il a totalement plante (typique sur images B&W ou floues).
#
# skin_only_s_ratio < 0.30 :
#   skin_only_s_ratio = fraction du mask classifiee comme skin pur par SegFormer
#   (sans bg, sans eye_g, avec ears).
#   Sur images normales, skin_only_s_ratio > 0.50.
#   Si descend en dessous de 0.30, SegFormer ne voit plus la peau du visage
#   -> il a plante (typique sur images desaturees ou atypiques).
BI_PLANTE_THRESHOLD = 0.70
SF_PLANTE_THRESHOLD = 0.30


# =============================================================================
# HELPERS : la formule de Julien decomposee en 3 morceaux
# =============================================================================
# La formule complete de Julien = somme de 6 termes :
#     pred = w_hair_bi * hair_ratio_b + w_hat_bi * hat_ratio_b + w_other_bi * other_ratio_b   (BiSeNet)
#          + w_hair_sf * hair_ratio_s + w_hat_sf * hat_ratio_s + w_other_bg_sf * other_ratio_s (SegFormer)
#
# Pour le fallback, on a besoin de pouvoir calculer separement :
#   - la contribution BiSeNet (3 premiers termes)
#   - la contribution SegFormer (3 derniers termes)
#   - la formule complete (les 6 termes ensemble)


def _formule_complete(hair_ratio_b, hat_ratio_b, other_ratio_b,
                       hair_ratio_s, hat_ratio_s, other_ratio_s,
                       weights):
    """La formule habituelle de Julien : somme des 6 termes (= ce qui est dans le notebook).

    Utilisee quand AUCUN des 2 modeles ne plante (= cas normal, ~98% des images).
    Le resultat est strictement identique a ce que Julien calcule actuellement.
    """
    return (weights["hair_bi"] * hair_ratio_b
          + weights["hat_bi"] * hat_ratio_b
          + weights["other_bi"] * other_ratio_b
          + weights["hair_sf"] * hair_ratio_s
          + weights["hat_sf"] * hat_ratio_s
          + weights["other_bg_sf"] * other_ratio_s)


def _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, weights):
    """Moitie BiSeNet de la formule de Julien : 3 termes seulement.

    Utilisee quand SegFormer plante : on ignore les 3 termes SegFormer parce qu'ils
    sont catastrophiquement faux (other_ratio_s ~ 0.95 sur un visage normal a target=0).
    """
    return (weights["hair_bi"] * hair_ratio_b
          + weights["hat_bi"] * hat_ratio_b
          + weights["other_bi"] * other_ratio_b)


def _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, weights):
    """Moitie SegFormer de la formule de Julien : 3 termes seulement.

    Utilisee quand BiSeNet plante : on ignore les 3 termes BiSeNet parce qu'ils sont
    catastrophiquement faux.
    """
    return (weights["hair_sf"] * hair_ratio_s
          + weights["hat_sf"] * hat_ratio_s
          + weights["other_bg_sf"] * other_ratio_s)


# =============================================================================
# FONCTION PRINCIPALE : a appeler depuis occlusion_computation()
# =============================================================================
def apply_v2_fallback(pred_gender,
                      hair_ratio_b, hat_ratio_b, other_ratio_b,
                      hair_ratio_s, hat_ratio_s, other_ratio_s,
                      skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio,
                      M_WEIGHTS, F_WEIGHTS):
    """Calcule le score d'occlusion avec gestion automatique des plantages.

    Cette fonction remplace le bloc `if pred_gender == 'M' / elif 'F'` du notebook.
    Elle se comporte EXACTEMENT comme la formule actuelle quand aucun modele ne
    plante (cas normal, ~98% des images). Quand un modele plante, elle bascule
    intelligemment sur l'autre.

    Args:
        pred_gender (str): 'M' ou 'F', sortie de face.sex (InsightFace).
        hair_ratio_b, hat_ratio_b, other_ratio_b (float): ratios BiSeNet de Julien.
        hair_ratio_s, hat_ratio_s, other_ratio_s (float): ratios SegFormer de Julien.
        skin_only_b_ratio (float): NOUVEAU. Fraction de skin pur (sans bg, sans eye_g,
                                   avec ears) detectee par BiSeNet dans le mask.
                                   Utilise pour diagnostic/debug, pas dans la decision.
        bg_b_ratio (float):        NOUVEAU. Fraction de background detectee par BiSeNet
                                   dans le mask. Si > 0.70 -> BiSeNet plante.
        skin_only_s_ratio (float): NOUVEAU. Fraction de skin pur (sans bg, sans eye_g,
                                   avec ears) detectee par SegFormer dans le mask.
                                   Si < 0.30 -> SegFormer plante.
        M_WEIGHTS, F_WEIGHTS (dict): tes coefficients du notebook (inchanges).

    Returns:
        float: score d'occlusion clip entre 0 et 1 (= ton ancien occlusion_score).
    """
    # ------ ETAPE 1 : selection des poids selon le genre ------
    # Identique a la logique actuelle de ton notebook.
    if pred_gender == 'M':
        weights = M_WEIGHTS
    elif pred_gender == 'F':
        weights = F_WEIGHTS
    else:
        # Cas tres improbable mais on est defensif : moyenne des 2 si genre inattendu
        weights = {k: (M_WEIGHTS[k] + F_WEIGHTS[k]) / 2.0 for k in M_WEIGHTS}

    # ------ ETAPE 2 : detection plantage (independante du genre) ------
    # bg_b_ratio > 0.70 : BiSeNet a etiquette ~tout le visage en background
    # skin_only_s_ratio < 0.30 : SegFormer ne voit quasi aucune peau
    bisenet_plante = bg_b_ratio > BI_PLANTE_THRESHOLD
    segformer_plante = skin_only_s_ratio < SF_PLANTE_THRESHOLD

    # ------ ETAPE 3 : choix de la formule ------
    if segformer_plante and not bisenet_plante:
        # CAS 1 : SegFormer plante mais BiSeNet OK
        # -> on calcule pred avec UNIQUEMENT la moitie BiSeNet (3 termes au lieu de 6)
        # -> on ignore les ratios SegFormer (ils sont absurdes : other_ratio_s ~ 0.95)
        score = _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, weights)

    elif bisenet_plante and not segformer_plante:
        # CAS 2 : BiSeNet plante mais SegFormer OK
        # -> on calcule pred avec UNIQUEMENT la moitie SegFormer
        # -> on ignore les ratios BiSeNet
        score = _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, weights)

    elif bisenet_plante and segformer_plante:
        # CAS 3 : les 2 plantent -> on prend "le moins pire"
        # On mesure le degre de plantage de chaque modele (haut = plus plante) :
        #   - BiSeNet : bg_b_ratio (plus c'est haut, plus BiSeNet a sature en bg)
        #   - SegFormer : 1 - skin_only_s_ratio (plus c'est haut, moins SegFormer voit de peau)
        # On bascule sur le modele dont le score de plantage est le plus bas.
        plantage_score_bisenet = bg_b_ratio
        plantage_score_segformer = 1.0 - skin_only_s_ratio
        if plantage_score_bisenet < plantage_score_segformer:
            # BiSeNet est "moins pire" -> sa contribution est plus fiable
            score = _contribution_bisenet(hair_ratio_b, hat_ratio_b, other_ratio_b, weights)
        else:
            # SegFormer est "moins pire" -> sa contribution est plus fiable
            score = _contribution_segformer(hair_ratio_s, hat_ratio_s, other_ratio_s, weights)

    else:
        # CAS 4 (le plus frequent, ~98% des images) : aucun modele ne plante
        # -> on utilise la formule complete habituelle, identique a ton notebook
        score = _formule_complete(hair_ratio_b, hat_ratio_b, other_ratio_b,
                                   hair_ratio_s, hat_ratio_s, other_ratio_s, weights)

    # ------ ETAPE 4 : clip final entre 0 et 1 (identique a ton notebook) ------
    return float(np.clip(score, 0.0, 1.0))


# =============================================================================
# UTILITAIRE DIAGNOSTIC (optionnel, pour debug / inspection)
# =============================================================================
def detect_plante_regime(skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio):
    """Retourne le regime de plantage detecte sur une image. Utile pour debug.

    Returns:
        str: "normal" (aucun plante) / "sf_only" (SF plante seul) /
             "bi_only" (BI plante seul) / "both" (les 2 plantent)

    Exemple :
        regime = detect_plante_regime(0.85, 0.05, 0.80)   # "normal"
        regime = detect_plante_regime(0.95, 0.05, 0.10)   # "sf_only"
        regime = detect_plante_regime(0.05, 0.85, 0.80)   # "bi_only"
        regime = detect_plante_regime(0.05, 0.85, 0.10)   # "both"

    Sur les 15K val, repartition attendue : ~98% "normal", ~1.5% "bi_only",
    ~0.2% "sf_only", ~0.1% "both".
    """
    bisenet_plante = bg_b_ratio > BI_PLANTE_THRESHOLD
    segformer_plante = skin_only_s_ratio < SF_PLANTE_THRESHOLD
    if segformer_plante and not bisenet_plante:
        return "sf_only"
    if bisenet_plante and not segformer_plante:
        return "bi_only"
    if bisenet_plante and segformer_plante:
        return "both"
    return "normal"
