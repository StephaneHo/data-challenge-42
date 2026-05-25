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

## 8. Outils locaux à construire

Pour pouvoir itérer sans submit, on a besoin de :

1. **`scripts/eval_val.py`** : charge un checkpoint, calcule les prédictions sur la val locale, sauve `val_predictions.csv` (filename, pred, target, gender).
2. **`scripts/estimate_scores.py`** : prend `val_predictions.csv`, calcule le score sous plusieurs hypothèses de distribution (train-like, mixte, test-like). Affiche aussi la décomposition par bin et par genre pour identifier d'où viennent les erreurs.
3. **`--resume` dans `train.py`** : reprend un entraînement depuis un checkpoint (pour étendre 8 → 15 époques sans repartir de zéro).
4. **Mode TTA dans `infer.py`** : option `--tta flip` qui fait deux passes (original + flip) et moyenne.

## 9. Workflow type entre deux soumissions

```
1. Idée d'amélioration (ex: plus d'époques)
2. Implémenter localement
3. Entraîner sur Colab
4. Télécharger le checkpoint et val_predictions.csv
5. Local : python scripts/estimate_scores.py val_predictions.csv
   → estimation interim/final
6. Si gain significatif vs précédent → submit
   Sinon → garde l'idée comme bouchon ou rejette
```

## 10. Référence : ce qu'on connaît déjà

| Repère | Score |
|---|---|
| Random / 0 constant | ~0.04 |
| Prédire la moyenne train (0.083) | ~0.018 |
| Baseline starter notebook (MSE, 1 epoch) | 0.00428 |
| **Notre resnet50, 8 époques, balanced, val** | **0.00171** |
| Très bon modèle sur ce type de tâche | ~0.001 |
| Score parfait (théorique) | 0 |

---

**Dernière mise à jour** : 2026-05-25, avant la soumission #1.
