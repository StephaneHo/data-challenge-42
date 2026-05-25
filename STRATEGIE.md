# Stratégie — Data Challenge Face Occlusion (IDEMIA / Telecom)

Ce document trace les décisions techniques et stratégiques prises pour ce challenge, avec leurs raisons. Objectif : pouvoir reprendre le fil entre deux sessions et expliquer chaque choix à un examinateur.

## 1. Le problème

**Tâche** : régresser le pourcentage d'occlusion d'un visage (valeur dans `[0, 1]`) à partir d'une image 224×224.

**Données** :
- 100 000 images d'entraînement avec label `FaceOcclusion` et `gender`
- 29 980 images de test sans label — c'est pour celles-ci qu'on doit prédire

**Distributions très différentes train ↔ test** :
- Train : pic vers 0 (50% des images en `[0, 0.05]`), queue très fine
- Test : distribution plus étalée, pic vers `[0.10, 0.20]`, queue plus épaisse

C'est ce **shift train→test** qui rend le problème intéressant et qui conditionne toute la stratégie.

## 2. La métrique officielle

```
Err_g = Σ w_i · (p_i − GT_i)²  /  Σ w_i,   avec w_i = 1/30 + GT_i
Score = (Err_F + Err_M) / 2  +  |Err_F − Err_M|
```

Calculée séparément sur les sous-ensembles femme (`gender=0`) et homme (`gender=1`), puis combinée. **Plus petit = mieux.**

### Ce que ça impose

- **Sur-pondération des hautes occlusions** : `w` passe de 0.033 (à GT=0) à 0.533 (à GT=0.5). Une erreur sur un visage très occludé compte 16× plus qu'une erreur sur un visage peu occludé. Conséquence : une MSE simple **sous-entraîne le tail haute occlusion**.
- **Pénalité d'équité genre** : le terme `|Err_F − Err_M|` double la pénalité d'un écart entre genres. Un modèle "bon en moyenne mais biaisé" est durement pénalisé.

## 3. Les décisions d'architecture (Phase 1)

### Pourquoi un split stratifié

Le notebook starter coupe `df[:20000]` en val — pas de tirage aléatoire. Sa val est donc **biaisée par l'ordre du CSV**, et son score val ne reflète rien de stable.

On stratifie par **(gender × bin d'occlusion)** pour que la val ait la même mixité que le train. Sans ça, la val finirait majoritairement composée de "hommes peu occludés" (la majorité du train) et donnerait un score val trompeur.

### Pourquoi une loss pondérée alignée sur la métrique

`WeightedMSELoss(w = 1/30 + GT)` reproduit exactement la pondération de la métrique officielle. L'option `balance_gender=True` rescale les contributions F/M dans le batch pour que les deux genres pèsent de manière équilibrée.

L'alternative serait une `MSELoss` standard — c'est ce que fait le notebook starter. Mais on optimiserait alors une mauvaise fonction objective.

### Pourquoi un balanced sampler

Sans sampler équilibré, ~50% des batchs seraient des "hommes basse occlusion" (la classe ultra-majoritaire). Le modèle apprendrait surtout à prédire `~0` et à reconnaître le genre.

Notre `WeightedRandomSampler` équilibre les 2×7 buckets (gender × bin d'occlusion). Conséquence visible dans les résultats : le **gap F/M est divisé par 3** par rapport à un sampler uniforme.

### Pourquoi des augmentations "occlusion-safe"

Le label = ratio (aire occludée / aire visage). Toute augmentation qui change ce ratio **corrompt silencieusement le label** :

| Augmentation | OK ? | Raison |
|---|---|---|
| Horizontal flip | ✅ | Symétrique, ratio invariant |
| ColorJitter | ✅ | Aucun changement spatial |
| Rotation faible (±10°) | ✅ | Ratio quasi inchangé |
| RandomResizedCrop | ❌ | Change l'aire visible |
| RandomErasing | ❌ | **Ajoute de l'occlusion artificielle → label faux** |
| RandomPerspective | ❌ | Peut sortir la zone occludée du cadre |

C'est un piège classique : un pipeline générique d'augmentation cassera la supervision sans erreur visible.

### Pourquoi une sortie sigmoid

Le label est dans `[0, 1]`. Sans sigmoid, le modèle peut prédire `−0.05` ou `1.3`, ce qui n'a pas de sens et pénalise la loss inutilement. Le sigmoid est gratuit et borne proprement la sortie.

## 4. Les règles hfactory

- **10 soumissions maximum** pour tout le challenge
- **Seule la dernière soumission compte** comme score officiel (les précédentes sont écrasées)
- À chaque soumission, on reçoit un **score interim** calculé sur un sous-ensemble du test
- La **distribution de ce sous-ensemble est intermédiaire** entre train et test
- Le **score final** (post-challenge) est calculé sur l'intégralité du test, avec la vraie distribution test
- → Le score interim **n'est PAS le score final**

## 5. Le piège du score val

### Notre val locale = 0.00171

Calculée sur les 15 001 images mises de côté du train. Comme elle suit la distribution train (skewed vers 0), elle **surestime nos performances réelles**.

### Analogie de l'examen

- **Train** = 90% d'exos faciles, 10% difficiles
- **Val** = échantillon du train → même mixité (90/10)
- **Test** = examen plus équilibré (50/50)
- Notre score val 14/15 = sur des exos majoritairement faciles
- Notre score interim ≠ notre score final ≠ notre score val

### Ordres de grandeur estimés

| Distribution | Score estimé |
|---|---|
| val (train-like) | 0.00171 (mesuré) |
| interim (mixte) | ~0.0025 − 0.0040 (estimation) |
| final (test-like) | ~0.0030 − 0.0050 (estimation) |

À calibrer dès la soumission #1.

## 6. Quand faire confiance à val ?

### Améliorations "honnêtes" — confiance haute

Quand val baisse grâce à un changement **structurel** au modèle, l'interim suit avec une corrélation forte :

| Type de changement | Confiance val → interim |
|---|---|
| TTA (flip averaging) | Très haute |
| Backbone plus gros | Haute |
| Plus d'époques (si pas d'overfit) | Haute |
| Ensemble de modèles | Très haute |
| Plus de données / augmentation | Haute |

Pour ces changements-là, on peut itérer localement sans submit — la baisse val sera fiable.

### Améliorations "risquées" — confiance basse

Quand val baisse grâce à un **tuning ciblé sur les défauts de val**, ça peut être un overfit val pur :

| Type de changement | Confiance val → interim |
|---|---|
| Clipping sur la plage observée en val | Basse (overfit distribution) |
| Tuning du sampler pour réduire un écart F/M observé sur val | Moyenne à basse |
| Post-processing par bin | Basse |
| Entraînement très long (>20 époques) | Diminue avec les époques |

Pour ces changements, on vérifie via une soumission.

## 7. La stratégie de soumissions

L'idée : **utiliser le moins de soumissions possible**, en faisant confiance à la val pour les améliorations "honnêtes".

### Soumission #1 — calibration + meilleur baseline possible

On empile les améliorations safe avant la première soumission, pour que la calibration porte sur un modèle déjà fort :

- ✅ **Plus d'époques (12-15)** : la loss baissait encore à l'époque 8, on sous-entraînait
- ✅ **TTA flip averaging** : gain quasi gratuit
- ❌ Pas de changement de backbone (à garder pour plus tard, pour pouvoir attribuer les effets)
- ❌ Pas d'ensemble (à construire plus tard, après avoir entraîné plusieurs modèles)

**Objectif** : score interim attendu autour de 0.0020-0.0030.

### Soumissions #2-#9 — expériences ciblées

On ne soumet que **si on a une raison forte** de penser à une amélioration (val locale qui baisse significativement sur changement "honnête", ou test d'hypothèse risquée à vérifier).

Idées priorisées :
- Backbone plus gros (EfficientNet-B3 ou ConvNeXt-Tiny) + TTA
- Ensemble du backbone large + resnet50
- Rebalance sampler (notre `err_m > err_f` actuel suggère qu'on a sur-corrigé l'autre sens)
- Augmentation synthétique : ajouter de l'occlusion artificielle sur des visages clairs, avec aire calculée (résout le problème des 36 samples > 0.5 seulement)

### Soumission #10 — la sécurité

Doit être **au moins aussi bonne que notre meilleur connu** (puisque seule la #10 compte). Probablement un ensemble bien rodé.

## 8. Outils locaux (implémentés)

Pour pouvoir itérer sans submit, on a construit :

1. **`scripts/eval_val.py`** : charge un checkpoint, calcule les prédictions sur la val locale, sauve `eval/val_*.csv` (filename, pred, target, gender). Option `--tta {none,flip}`.
2. **`scripts/estimate_scores.py`** : prend un ou plusieurs `val_predictions.csv` :
   - Affiche le score global (sanity check vs training log)
   - **Décompose par (genre × bin d'occlusion)** : counts, erreur locale, biais, contribution à l'erreur de chaque genre
   - **Estime le score sous plusieurs distributions cibles** (val train-like, test-like réaliste, test-like plus étalé, uniforme [0, 0.5])
   - Compare plusieurs versions côte à côte si plusieurs CSV passés
3. **`--resume PATH` dans `scripts/train.py`** : reprend un entraînement depuis un checkpoint, pour étendre une run existante (8 → 15 époques) sans repartir de zéro. À combiner avec `--no-scheduler` et un `--lr` plus bas (e.g., 1e-4) pour un fine-tune doux.
4. **`--tta {none,flip}` dans `scripts/infer.py` et `scripts/eval_val.py`** : flip averaging. La prédiction est la moyenne de `model(X)` et `model(flip(X))`. Coût : ×2 sur le temps d'inférence (5 min au lieu de 3 sur le full test).

## 9. Analyse d'erreurs sur la baseline (resnet50, 8 époques, balanced)

Effectuée le 2026-05-25 avec `scripts/estimate_scores.py eval/val_resnet50_8ep.csv`. Les chiffres exacts sont à retrouver dans cet output ; les enseignements ci-dessous.

### Distribution des erreurs par (genre × bin)

| bin | err F | err M | constat |
|---|---|---|---|
| [0.00, 0.05) | 0.00072 | 0.00066 | OK partout |
| [0.05, 0.15) | ~0.0005 | ~0.0008 | OK |
| [0.15, 0.30) | ~0.0007 | 0.0009-0.0012 | Hommes commencent à dériver |
| [0.30, 0.50) | 0.00079 | **0.00383** | **M = 5× pire que F** |
| [0.50, 1.01) | 0.00732 | **0.17598** | **M = 24× pire que F** (3 samples) |

### Bias systématique (regression to the mean)

| bin | bias F | bias M |
|---|---|---|
| [0.00, 0.05) | +0.016 | +0.015 (sur-prédit ~+15%) |
| [0.15, 0.20) | ~0 | ~0 (bien calibré) |
| [0.30, 0.50) | −0.011 | −0.031 (sous-prédit) |
| [0.50, 1.01) | −0.082 | −0.280 (sous-prédit massivement) |

Pattern net : le modèle **tire toutes ses prédictions vers la zone [0.10, 0.20]**, sa "zone de confort". Il a peur de prédire 0% ou 50%+ même quand c'est la vérité. Classique avec un sampler balancé agressif qui force le modèle à voir des hautes occlusions, mais sans suffisamment d'exemples pour qu'il commit vraiment dessus.

### D'où vient l'erreur totale ?

Sur la val, 30% de l'erreur masculine (Err_M = 0.00135) provient des **3 samples à occlusion > 50%**. Ça veut dire :
- L'erreur globale est statistiquement instable (3 samples seulement)
- Mais structurellement, **on est mauvais sur le tail haute occlusion**
- Sur le test, s'il y a même seulement 1-2% d'images en [0.50+) avec des hommes, ça pourrait dégrader fortement le score

### Estimations interim/final avec ces erreurs

| Distribution cible | Score estimé |
|---|---|
| val (train-like) | 0.00171 |
| uniform [0, 0.50) | 0.00364 |
| test-like (realistic, estimé depuis brief histogram) | 0.01068 |
| test-like (more spread) | 0.01734 |

**Le score interim attendu sur hfactory est probablement entre 0.004 et 0.011.**

## 10. Décisions issues de cette analyse

Les améliorations à tester avant la soumission #1 (par ordre de priorité) :

### A. Plus d'époques (12-15 vs 8 actuelles)
- **Pourquoi** : la loss baissait encore à l'époque 8 (0.00017 vs val 0.00171, signe que le modèle pourrait apprendre encore)
- **Comment** : `python scripts/train.py --resume checkpoints/resnet50_best.pt --epochs 7 --lr 1e-4 --no-scheduler`
- **Coût** : ~1h Colab supplémentaire
- **Risque** : faible (le modèle peut overfitter si on pousse trop, à surveiller via val)

### B. TTA en inférence (flip averaging)
- **Pourquoi** : gain quasi gratuit, presque garanti, n'overfitte pas
- **Comment** : `python scripts/infer.py --checkpoint ... --tta flip`
- **Coût** : 5 min d'inférence en plus (vs 3 sans TTA)
- **Risque** : très faible

### C. Tester un sampler moins agressif (à itération suivante)
- **Pourquoi** : notre `err_m > err_f` suggère qu'on a sur-corrigé. Le sampler balanced a privilégié les femmes en hautes occlusions, mais a coupé l'exposition des hommes en faibles occlusions
- **Comment** : `python scripts/train.py --no-balanced-sampler --no-balanced-loss` (un puis l'autre puis les deux)
- **Risque** : moyen — on peut perdre la maîtrise du gap F/M

### D. Augmentation synthétique haute occlusion (plus tard)
- **Pourquoi** : 36 samples > 0.5 dans 100k de train, c'est minuscule. Si on génère 5000 samples synthétiques en superposant des formes opaques avec aire calculée, on peut combler le tail
- **Coût** : 3-4h de dev
- **Risque** : moyen — dépend de la qualité du synthétique

### Plan pour la #1

```
1. Reprendre l'entraînement à partir du checkpoint actuel pour 7 époques de plus (lr=1e-4)
2. Inférence sur test avec --tta flip
3. estimate_scores sur val avec TTA pour s'assurer du gain
4. Submit le test_predictions.csv
```

## 11. Piste parallèle : zero-shot par segmentation faciale

En complément de l'approche entraînée, on explore une approche **sans entraînement** (zero-shot), proposée par le collègue. L'idée : utiliser des modèles pré-entraînés sur des millions de visages pour mesurer **géométriquement** la zone occludée. Aucun apprentissage requis sur les 100k images IDEMIA.

### Pourquoi cette piste a de la valeur

| Trained (notre baseline) | Zero-shot |
|---|---|
| Apprend des 100k images IDEMIA | Modèles pré-entraînés sur millions d'images |
| Sensible au shift train→test | Probablement plus robuste au shift |
| Coût GPU significatif | Inference CPU seule, 0 entraînement |
| Difficile à expliquer | Très interprétable (mesure géométrique directe) |
| Compte 1 soumission par expérience | Idem, mais errurs décorrélées → bon pour ensemble |

Si la piste donne un signal correct, **un ensemble (trained + zero-shot) est souvent l'amélioration la plus rentable** en compétition, car les deux pipelines font des erreurs très différentes.

### Deux variantes en parallèle

**Pipeline du collègue (complet, 3D-aware)** :
```
Image → RetinaFace (detection + alignement)
      → 3DDFA-V2 (reconstruction 3D du visage)
      → Pose-aware visible face mesh (calcul de l'aire théorique du visage selon la pose)
      → Rasterized theoretical visible mask (projection 2D)
      → BiSeNet face parsing (segmentation observée)
      → Ratio (théorique − observé) / théorique
      → Heuristic corrections
      → Score final
```

**Variante 2D-only (la nôtre, plus légère)** :
```
Image (déjà cropée 224×224 par IDEMIA)
   → SegFormer face-parsing (jonathandinu/face-parsing, 19 classes CelebAMask-HQ)
   → Identification des classes "face" (skin, brows, eyes, nose, mouth, lips)
                     et "occluders" (eye_g, cloth, hair, hat)
   → Ratio occluder / (face + occluder) dans la zone du visage
   → Calibration linéaire/isotonique sur 1000 train images vs labels GT
   → Score calibré
```

**Différences clés** :
- Pas de 3DDFA-V2 → pas de gestion explicite de la pose (les crops IDEMIA sont en général frontaux)
- Pas de RetinaFace → on suppose le crop déjà aligné
- Tout en 2D dans le plan image

### Valeur de comparaison entre les deux variantes

| Si pipeline collègue gagne | Si variante 2D gagne | Si égales |
|---|---|---|
| Le 3D est essentiel pour ce dataset | 2D suffit, on peut simplifier | On ensemble les deux pour décorréler |

C'est une **étude d'ablation** : "est-ce que les étapes 3D apportent réellement quelque chose, ou est-ce que la version 2D suffit pour ces crops alignés ?"

### Choix techniques de la variante 2D

| Décision | Motivation |
|---|---|
| **SegFormer `jonathandinu/face-parsing`** | Plus précis que BiSeNet, distribution HuggingFace propre, mêmes 19 classes CelebAMask-HQ |
| **Pas de MediaPipe / RetinaFace** | Les crops IDEMIA sont déjà alignés ; un détecteur supplémentaire = dépendance lourde sans gain |
| **Calibration apprise sur 1000 samples train** | La segmentation seule a un biais systématique vs la définition IDEMIA ; un fit linéaire/isotone aligne les échelles |
| **Architecture modulaire dans `src/zero_shot/`** | Le collègue peut plugger son `3D-aware face mesh` dans le même squelette s'il veut |

### Coût CPU (local, sans GPU)

| Étape | Par image | 15k val | 30k test |
|---|---|---|---|
| Inférence SegFormer | ~200 ms | ~50 min | ~1h40 |
| Post-processing | ~5 ms | ~1 min | ~2 min |

Total : **~1h pour la val, ~2h pour le test**, en arrière-plan sur le CPU local.

### Plan d'exécution

1. **Coder l'architecture modulaire** `src/zero_shot/` (face_parser, occlusion_estimator, calibration, pipeline)
2. **CLI scripts** dans `scripts/zero_shot/` (predict, fit_calibration, run_on_subset)
3. **Sanity test sur 100 images** train → scatter plot raw_ratio vs GT. Si corrélation > 0.5, on continue.
4. **Fit calibration** sur 1000 train.
5. **Full val** (~1h CPU) → comparer avec notre baseline trained via `estimate_scores.py`.
6. **Si résultats prometteurs** → full test, candidat de soumission ou composant d'ensemble.



```
1. Idée d'amélioration (ex: plus d'époques)
2. Implémenter / entraîner sur Colab
3. python scripts/eval_val.py --checkpoint XYZ.pt --backbone X --out eval/val_xyz.csv
4. Télécharger val_xyz.csv en local
5. python scripts/estimate_scores.py eval/val_*.csv  (compare toutes les versions)
6. Si gain net vs précédent → submit
   Sinon → garde l'idée comme bouchon ou rejette
```

## 12. Référence : ce qu'on connaît déjà

| Repère | Score |
|---|---|
| Random / 0 constant | ~0.04 |
| Prédire la moyenne train (0.083) | ~0.018 |
| Baseline starter notebook (MSE, 1 epoch) | 0.00428 |
| **Notre resnet50, 8 époques, balanced, val (train-like)** | **0.00171** |
| Notre resnet50, 8 époques, balanced, estimé interim hfactory | **~0.004 - 0.011** |
| Très bon modèle sur ce type de tâche | ~0.001 |
| Score parfait (théorique) | 0 |

---

**Dernière mise à jour** : 2026-05-25, après analyse fine des erreurs, avant la soumission #1.
