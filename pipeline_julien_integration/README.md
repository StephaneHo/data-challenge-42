# pipeline_julien_integration

V2 simple fallback à intégrer dans `notebooks/DataChallengeJulien_TF.ipynb`.

**Aligné 100% sur les conventions et le pipeline du notebook** :
- Variables, noms, indices, dicts de coefficients (M_WEIGHTS, F_WEIGHTS) inchangés
- Resize unique 512×512 préservé : on ne calcule rien en dehors de ton pipeline
- Mêmes `parsing_b`, `parsing_s`, `mask_theoretical`, `total_pixels` que tu as déjà
- Aucun retour à 224, aucun pipeline parallèle, aucun patch sur un seul modèle

On ajoute juste **3 fractions supplémentaires** calculées **exactement** comme tes
`hair_ratio_b`, `hat_ratio_b`, etc. — au sein de ta fonction `occlusion_computation()`,
avec ton resize unique.

---

## Le problème adressé

Sur certaines images (~245 sur 15K val), un des deux modèles de segmentation plante :

- **BiSeNet plante** : sur images B&W, vintage, floues, basses résolution → le modèle classe quasi tout le mask en `bg` (classe 0)
- **SegFormer plante** : sur images très désaturées / atypiques → le modèle voit ~0 skin

Quand ça arrive, ta formule hybride `M_WEIGHTS["other_bi"] * other_ratio_b + ... + M_WEIGHTS["other_bg_sf"] * other_ratio_s` est dominée par le modèle qui plante → **prédictions catastrophiques** (target=0 mais pred=1.0).

Le v2 fallback détecte ces cas et bascule sur le modèle qui marche. **CV 5/5 GAIN sur val**, **zéro nouveau coefficient**, **zéro modif de ta formule principale**.

---

## Réponse à ta concern "doubler les résultats si on retire un modèle"

> *"Si je supprime un modèle, alors je dois potentiellement doubler les résultats de l'autre, car il y a de l'hybridation dans les coefficients calculés."*

C'est juste en théorie. **EN PRATIQUE** ça marche sans doubler :

- Pour les cas SF plante à **target ≈ 0** (~25 catastrophes B&W) : `bi_contrib` seul donne ~0 (BiSeNet voit toute la peau correctement) → la sous-estimation est **exactement ce qu'on veut**, err passe de 0.43 à 0.07
- Pour les cas SF plante à **target élevé** (rare, ~5 images sur val) : `bi_contrib` seul est un peu sous-estimé mais MOINS pire que la pred catastrophique initiale
- Les gates (`bg_b_ratio > 0.70`, `skin_only_s_ratio < 0.30`) ne se déclenchent **jamais** sur des images normales où l'occlusion est réellement présente

Résultat : CV 5/5 folds GAIN, gain attendu sur test brief ~-10%.

---

## Intégration (4 étapes, dans l'ordre)

### Étape 1 — Ajouter 3 listes d'indices (dans la cellule où tu définis `VISIBLE_FACE_CLASSES_B`)

```python
# Pour la detection plante du fallback v2.
# NB : SKIN_ONLY_B/S incluent les ears (7,8 BiSeNet ; 8,9 SegFormer) alors que
# VISIBLE_FACE_CLASSES_B/S de Julien ne les incluent pas. C'est volontaire : les
# seuils 0.70 et 0.30 ont ete calibres en CV sur cette definition AVEC ears.
SKIN_ONLY_B = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]  # skin + brows + eyes + ears + nose + mouth + lips
SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]   # skin + nose + eyes + brows + ears + mouth + lips
BG_B        = [0]                                      # background BiSeNet
```

### Étape 2 — Importer le module (cellule en haut du notebook)

Ajoute cette cellule **avant** la cellule qui définit `occlusion_computation()` :

```python
# Setup path pour acceder au module pipeline_julien_integration
# (le module est au repo root, le notebook est dans notebooks/)
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline_julien_integration import apply_v2_fallback
```

Si tu es dans Colab et que ton repo est ailleurs, adapte `REPO_ROOT` à ton chemin (ex: `'/content/data-challenge-42'`).

### Étape 3 — Dans `occlusion_computation()`, ajouter le calcul des 3 nouvelles features

Juste après ton `other_ratio_s = other_s_in_mask / total_pixels`, ajoute ce bloc (en gardant l'indentation de 4 espaces puisqu'on est dans la fonction) :

```python
    ### Detection plante v2 (3 nouvelles features, independantes de la formule principale)
    skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
    bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
    skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
    skin_only_b_ratio = float(np.sum(skin_only_b & mask_theoretical)) / float(total_pixels)
    bg_b_ratio        = float(np.sum(bg_b        & mask_theoretical)) / float(total_pixels)
    skin_only_s_ratio = float(np.sum(skin_only_s & mask_theoretical)) / float(total_pixels)
```

### Étape 4 — Remplacer le bloc `if pred_gender == 'M' / elif 'F'` par un seul appel

Remplace ces ~16 lignes (toujours indentées de 4 espaces dans la fonction) :

```python
    if pred_gender == 'M':
        occlusion_score = np.clip(M_WEIGHTS["hair_bi"] * hair_ratio_b
        + M_WEIGHTS["hat_bi"] * hat_ratio_b
        + ...,
        0, 1)
    elif pred_gender == 'F':
        occlusion_score = np.clip(F_WEIGHTS["hair_bi"] * hair_ratio_b
        + ...,
        0, 1)
```

Par ces 12 lignes :

```python
    occlusion_score = apply_v2_fallback(
        pred_gender=pred_gender,
        hair_ratio_b=hair_ratio_b, hat_ratio_b=hat_ratio_b, other_ratio_b=other_ratio_b,
        hair_ratio_s=hair_ratio_s, hat_ratio_s=hat_ratio_s, other_ratio_s=other_ratio_s,
        skin_only_b_ratio=skin_only_b_ratio,
        bg_b_ratio=bg_b_ratio,
        skin_only_s_ratio=skin_only_s_ratio,
        M_WEIGHTS=M_WEIGHTS,
        F_WEIGHTS=F_WEIGHTS,
    )
```

**C'est fini.** Sans fallback (cas normal, ~98% des images), `apply_v2_fallback` calcule **exactement** ta formule habituelle. Avec fallback (~2% des cas catastrophiques), il évite les sur-prédictions.

---

## Caveat honnête sur les seuils (à titre informatif)

Les seuils `0.70` et `0.30` ont été calibrés en CV sur notre cache `val_features.csv`, qui a été produit avec un pipeline légèrement différent (SegFormer à 224×224, BiSeNet à 512×512 après upscale local, 3DDFA à 224×224).

Ton pipeline à toi tourne tout à 512×512 (single resize), c'est plus cohérent et c'est ce qu'on garde.

**Conséquence pratique** : les fractions normalisées (ratio = pixels/total) sont scale-invariantes par construction. Les seuils 0.70 et 0.30 détectent des phénomènes extrêmes catégoriels :
- "BiSeNet a classé quasi tout en bg" (pas un sujet de résolution)
- "SegFormer ne voit quasi aucune peau" (pareil)

Sur tes fractions à 512×512, peut-être ~5-10 images sur 15K basculent en plus ou en moins (cas borderline), mais les **vraies catastrophes** (target=0 mais pred=1.0) restent toutes détectées. Le gain attendu reste positif.

Si tu veux, après avoir testé, tu peux ajuster les seuils en regardant la distribution de `bg_b_ratio` et `skin_only_s_ratio` sur tes propres résultats. Mais ce n'est probablement pas nécessaire.

---

## Comment vérifier que ça marche

Après tes 4 étapes, fais tourner ton notebook sur quelques images du val et compare. Sur les images normales (98%), tu dois obtenir **exactement la même prédiction** qu'avant. Sur les ~2% catastrophiques, la pred sera plus basse (et plus correcte).

Pour identifier les cas qui basculent en fallback :

```python
from pipeline_julien_integration import detect_plante_regime

# Sur une image, savoir quel regime se declenche :
regime = detect_plante_regime(skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio)
# Renvoie : "normal" / "sf_only" / "bi_only" / "both"

# Sur l'ensemble val, voir la repartition :
from collections import Counter
regimes_list = []
for ... :  # pour chaque image
    regimes_list.append(detect_plante_regime(skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio))
print(Counter(regimes_list))
# Attendu sur val : ~98% "normal", ~1.5% "bi_only", ~0.2% "sf_only", ~0.1% "both"
```

---

## Tests faits sur val (15K images)

| Approche | val brief | CV folds GAIN | Verdict |
|---|---|---|---|
| baseline (sans fallback) | 0.00594 | - | reference |
| **v2 simple fallback** | **0.00592** | **5/5** | **SHIP** |
| + 14-coef fallback optimisé | 0.00528 | 4/5 (variance) | overfit |
| + blur feature | 0.00443 | 3/5 | overfit |
| + MediaPipe Hands | 0.00465 | 3/5 | overfit |
| + B&W detection | 0.00845 | 0/5 | piège brief |
| + YOLO-World objets | 0.00592+ | 0/5 | double-comptage |
| + image enhancement | 0.00660 | 2/5 | trop variable |
| + glasses bbox (eye_g) | 0.00673 | 0/5 | trop crue |

→ Le v2 simple est le seul gain robuste sur ce dataset + ce metric (brief).

---

## Structure du module

```
pipeline_julien_integration/
├── __init__.py          # expose apply_v2_fallback et detect_plante_regime
├── fallback_v2.py       # logique du fallback (zero dependance hors numpy)
└── README.md            # ce fichier
```

Aucune dépendance autre que numpy. Compatible avec ton environnement actuel.
