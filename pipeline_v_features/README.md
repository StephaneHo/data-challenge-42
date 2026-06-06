# Pipeline v_features

Pipeline de prediction d'occlusion faciale qui **decompose** le mask 3DDFA en
composants (hair, hat, other) pour chaque modele de segmentation (BiSeNet et
SegFormer), avec calibration **per-feature x per-gender** (12 coefficients).

## Score val brief (sans correction Julien) : **0.00497**

vs v10_hyb 0.00668, soit **-25.6%** (bootstrap 100% confidence).

A re-calculer avec la correction de mask Julien integree.

## Contenu du dossier

| Fichier | Role |
|---|---|
| `evaluate.py` | **Module standalone** : scoring officiel + reweighting + formule v_features. **Self-contained** (juste numpy/pandas). Julien peut l'importer directement dans son notebook. |
| `01_cache_features.py` | Etape 1 : faire tourner les modeles sur val/test et sauver les 5 fractions par image. ~20h val, ~40h test. |
| `02_optimize_coefficients.py` | Etape 2 : optimiser les 12 coefficients par Nelder-Mead sur val. ~5 min. |
| `03_generate_submission.py` | Etape 3 : appliquer la formule sur test, generer le CSV submission. ~1 min. |
| `coefficients.json` | (Genere par etape 2) Les 12 coefficients optimaux. |

## Pipeline en bref

```
[image test]
     |
     +--------+--------+
     |        |        |
     v        v        v
RetinaFace  3DDFA-V2  InsightFace
   bbox     dense     gender
     |     mesh        |
     |        |        |
     |        v        |
     | convex hull     |
     |    |            |
     |    v            |
     |  ***            |
     | correction      |
     | Julien:         |
     | warpAffine      |
     |  ***            |
     |    |            |
     |    v            |
     | mask_theorique  |
     |    |            |
     +----+----+       |
          |            |
   +------+------+     |
   |             |     |
   v             v     |
BiSeNet      SegFormer |
  @512         @512    |
   |             |     |
   v             v     |
classes      classes   |
 0..18        0..18    |
   |             |     |
   v             v     |
5 fractions  5 fractions
{skin,bg,    {skin,bg,
 hair,hat,    hair,hat,
 other}       other}
   |             |     |
   +------+------+-----+
          |
          v
   FORMULE v_features
   (12 poids per-feature
    per-gender)
          |
          v
   FaceOcclusion
```

## La formule v_features

Pour chaque image (avec g = genre predit) :

```python
FaceOcclusion = clip(
    w_hair_bi_g   * hair_bi_in_mask                       +
    w_hat_bi_g    * hat_bi_in_mask                        +
    w_other_bi_g  * other_bi_in_mask                      +
    w_hair_sf_g   * hair_sf_in_mask                       +
    w_hat_sf_g    * hat_sf_in_mask                        +
    w_otherbgsf_g * (other_sf_in_mask + bg_sf_in_mask),
    0, 1
)
```

ou `*_in_mask` = (pixels de la classe INTER mask_3D_corrige) / aire_mask.

**Note importante** : on traite `bg` differemment selon le modele :
- **BiSeNet** : `bg` est inclus dans `r_3D_Bi_bg` (= 1 - skin - bg). On dit que
  bg = "visible" car BiSeNet confond souvent face skin avec bg.
- **SegFormer** : `bg` est ajoute a `other`. SegFormer est precis, peu de bg
  dans le mask.

Cette asymetrie est l'idee centrale de la pipeline (= "hybride v10").

## Correction de mask (Julien)

Le mask 3DDFA brut est decale vs le visage reel. Correction empirique :

```python
scale_x = 0.9
scale_y = 1.05
tx = 15
ty = -10
M = np.array([[scale_x, 0, tx], [0, scale_y, ty]], dtype=np.float32)
mask = cv2.warpAffine(mask, M, dsize=img.shape[:2])
```

**Cette correction est INTEGREE dans `01_cache_features.py`** (fonction
`apply_julien_mask_correction`).

## Evaluation : 5 colonnes par distribution

`evaluate.py` reporte ces 5 metriques pour chaque distribution test :

| Metrique | Formule |
|---|---|
| `err_F` | weighted MSE sur Femmes, w_i = 1/30 + GT_i |
| `err_M` | weighted MSE sur Hommes, idem |
| `mean_err` | (n_F * err_F + n_M * err_M) / (n_F + n_M) -- moyenne ponderee |
| `gap` | \|err_F - err_M\| |
| `score` | (err_F + err_M)/2 + gap -- **METRIQUE OFFICIELLE** |

4 distributions : `val natif`, `reweighted brief` (officielle), `reweighted spread`, `reweighted heavy`.

L'**optimisation se fait sur `score` (officiel) sous distribution `brief`**.

## Indices de classes (verifies via model.config)

| Classe | BiSeNet (CelebAMask-HQ) | SegFormer (jonathandinu) |
|---|---|---|
| bg | 0 | 0 |
| skin | 1 | 1 |
| nose | 10 | 2 |
| l_brow | 2 | 6 |
| r_brow | 3 | 7 |
| l_eye | 4 | 4 |
| r_eye | 5 | 5 |
| eye_g (glasses) | 6 | 3 |
| l_ear | 7 | 8 |
| r_ear | 8 | 9 |
| ear_r (earring) | 9 | 15 |
| mouth | 11 | 10 |
| u_lip | 12 | 11 |
| l_lip | 13 | 12 |
| neck | 14 | 17 |
| neck_l (necklace) | 15 | 16 |
| cloth | 16 | 18 |
| hair | 17 | 13 |
| hat | 18 | 14 |

**ATTENTION** : ces indices sont DIFFERENTS entre BiSeNet et SegFormer ! C'est la
source de plusieurs bugs trouves au cours du projet. Voir les constantes en haut
de `01_cache_features.py`.

## Workflow complet

```bash
# 1. Cache val (~20h) -- correction Julien incluse
python pipeline_v_features/01_cache_features.py \
    --source val --resume \
    --out eval/cache/val_features.csv

# 2. Optimisation des coefficients (5 min)
cd pipeline_v_features  # important : pour l'import de evaluate
python 02_optimize_coefficients.py \
    --val-cache ../eval/cache/val_features.csv \
    --out coefficients.json

# 3. Cache test (~40h) -- correction Julien incluse
cd ..
python pipeline_v_features/01_cache_features.py \
    --source test --resume \
    --out eval/cache/test_features.csv

# 4. Generation submission (1 min)
cd pipeline_v_features
python 03_generate_submission.py \
    --test-cache ../eval/cache/test_features.csv \
    --coefficients coefficients.json \
    --gender-cache ../eval/cache/test_gender_pred.csv \
    --out ../submission.csv
```

## Pour Julien : integrer dans son notebook

Julien peut copier-coller :

1. **`evaluate.py`** entier dans une cellule -> il a immediatement le scoring
   complet a 5 colonnes + reweighting.

2. **La fonction `compute_features_for_image()`** depuis `01_cache_features.py`
   -> il peut calculer les fractions sur n'importe quelle image avec ses propres
   modeles deja charges.

3. **La fonction `predict_v_features()`** depuis `evaluate.py` -> applique la
   formule v_features sur un DataFrame avec les coefficients.

Le minimum pour reproduire la pipeline : `evaluate.py` + la fonction
`compute_features_for_image()`.
