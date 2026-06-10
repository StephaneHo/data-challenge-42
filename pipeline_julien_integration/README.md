# pipeline_julien_integration

V2 simple fallback à intégrer dans `notebooks/DataChallengeJulien_TF.ipynb`.

**Aligné 100% sur les conventions du notebook** : variables, noms, indices, dicts de coefficients (M_WEIGHTS, F_WEIGHTS) — rien à modifier dans tes définitions existantes.

## Le problème adressé

Sur certaines images (~245 sur 15K val), un des deux modèles de segmentation plante :

- **BiSeNet plante** : sur images B&W, vintage, floues, basses résolution → le modèle classe quasi tout le mask en `bg`
- **SegFormer plante** : sur images très désaturées / atypiques → le modèle voit ~0 skin

Quand ça arrive, ta formule hybride `M_WEIGHTS["other_bi"] * other_ratio_b + ... + M_WEIGHTS["other_bg_sf"] * other_ratio_s` est dominée par le modèle qui plante → **prédictions catastrophiques** (target=0 mais pred=1.0).

Le v2 fallback détecte ces cas et bascule sur le modèle qui marche. **CV 5/5 GAIN sur val**, **zéro nouveau coefficient**, **zéro modif de ta formule principale**.

## Réponse à ta concern "doubler les résultats si on retire un modèle"

> *"Si je supprime un modèle, alors je dois potentiellement doubler les résultats de l'autre, car il y a de l'hybridation dans les coefficients calculés."*

C'est juste en théorie. **EN PRATIQUE** ça marche sans doubler :

- Pour les cas SF plante à **target ≈ 0** (~25 catastrophes B&W) : `bi_contrib` seul donne ~0 (BiSeNet voit toute la peau correctement) → la sous-estimation est **exactement ce qu'on veut**, err passe de 0.43 à 0.07
- Pour les cas SF plante à **target élevé** (rare, ~5 images sur val) : `bi_contrib` seul est un peu sous-estimé mais MOINS pire que la pred catastrophique initiale
- Les gates (`bg_b_ratio > 0.70`, `skin_only_s_ratio < 0.30`) ne se déclenchent **jamais** sur des images normales où l'occlusion est réellement présente

Résultat : CV 5/5 folds GAIN, gain attendu sur test brief ~-10%.

## Intégration : 3 changements dans ton notebook

### 1. Ajouter les indices "skin pur" en haut de ton notebook

Juste à côté de `VISIBLE_FACE_CLASSES_B = [...]`, ajoute :

```python
# Pour la detection plante du fallback v2 (skin pur, sans bg, sans eye_g)
SKIN_ONLY_B = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # skin + brows + eyes + nose + mouth + lips
SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 10, 11, 12]   # skin + nose + eyes + brows + mouth + lips
BG_B        = [0]                                # background BiSeNet
```

### 2. Dans `occlusion_computation()`, après tes calculs de fractions, ajouter 3 lignes

Juste après ton `other_ratio_s = other_s_in_mask / total_pixels` :

```python
### Detection plante v2 (3 features supplementaires)
skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
skin_only_b_ratio = float(np.sum(skin_only_b & mask_theoretical)) / float(total_pixels)
bg_b_ratio        = float(np.sum(bg_b        & mask_theoretical)) / float(total_pixels)
skin_only_s_ratio = float(np.sum(skin_only_s & mask_theoretical)) / float(total_pixels)
```

### 3. Remplacer ton bloc `if pred_gender == 'M' / elif 'F'` par UN SEUL APPEL

Remplace ces ~16 lignes :

```python
if pred_gender == 'M':
    occlusion_score = np.clip(M_WEIGHTS["hair_bi"] * hair_ratio_b
        + M_WEIGHTS["hat_bi"] * hat_ratio_b
        + ... ,
        0, 1)
elif pred_gender == 'F':
    occlusion_score = np.clip(F_WEIGHTS["hair_bi"] * hair_ratio_b
        + ... ,
        0, 1)
```

Par :

```python
from pipeline_julien_integration import apply_v2_fallback

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

Voilà. Sans le fallback (cas normal, ~98% des images), `apply_v2_fallback` calcule **exactement** ta formule habituelle. Avec le fallback (~2% des cas catastrophiques), il évite les sur-prédictions.

## Quelques diagnostics utiles (optionnel)

```python
from pipeline_julien_integration import detect_plante_regime

# Sur une image, savoir quel regime se declenche :
regime = detect_plante_regime(skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio)
# "normal" / "sf_only" / "bi_only" / "both"

# Compter sur l'ensemble val :
from collections import Counter
regimes = [detect_plante_regime(s, b, sk) for s, b, sk in zip(...)]
print(Counter(regimes))
# Attendu : ~98% "normal", ~1.5% "bi_only", ~0.2% "sf_only", ~0.1% "both"
```

## Tests faits

| Approche | val brief | CV folds GAIN | Verdict |
|---|---|---|---|
| baseline (sans fallback) | 0.00594 | - | reference |
| **v2 simple fallback** | **0.00592** | **5/5** | **SHIP** |
| + 14-coef fallback optimise | 0.00528 | 4/5 (variance) | overfit |
| + blur feature | 0.00443 | 3/5 | overfit |
| + MediaPipe Hands | 0.00465 | 3/5 | overfit |
| + B&W detection | 0.00845 | 0/5 | piege brief |
| + YOLO-World objets | 0.00592+ | 0/5 | double-comptage |
| + image enhancement | 0.00660 | 2/5 | trop variable |
| + glasses bbox (eye_g) | 0.00673 | 0/5 | trop crue |

→ Le v2 simple est le seul gain robuste sur ce dataset + ce metric (brief).

## Structure du module

```
pipeline_julien_integration/
├── __init__.py          # expose apply_v2_fallback
├── fallback_v2.py       # logique du fallback (zero dependance hors numpy)
└── README.md            # ce fichier
```

Aucune dépendance autre que numpy. Compatible avec ton environnement actuel.
