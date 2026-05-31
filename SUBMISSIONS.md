# hfactory Submissions Log

10 submissions max total. Each new submission **overwrites the previous one** —
only the LAST counts for the final official score. Choose what to submit very
deliberately.

## Candidates ready (2026-05-31, test distribution from brief re-read)

All five CSVs are 29 980 rows, validated format. The choice depends on which
distribution we trust most for the FINAL score.

**Caveat sur le test-like estimate** : je lis le histogramme test du brief
visuellement, l'incertitude par bin est ~±3%. Les variants dont les test-like
diffèrent de moins de ~5% sont essentiellement à égalité dans le bruit.

| Candidate path | Composition | Test-like est. | Subset score | Gap | Notes |
|---|---|---|---|---|---|
| `results/zero_shot_tf_v6_power06_tta/test_predictions.csv` | `simple_hull_scaled_power06_tta` | **0.01712** | 0.03736 | 0.01717 | **NEW: best test-like after corrected distribution** |
| `results/zero_shot_tf_v2/test_predictions.csv` | `simple_hull_scaled_power07_tta` | 0.01777 | 0.02239 | 0.00877 | Within +4% of v6 (likely within noise) |
| `results/zero_shot_tf_v3_ens85_15/test_predictions.csv` | 0.85 power07 + 0.15 cw | ~0.019 | ~0.020 | ~0.0070 | Compromise |
| `results/zero_shot_tf_v4_ens70_30/test_predictions.csv` | 0.70 power07 + 0.30 cw | 0.01991 | 0.01766 | 0.00504 | Interim score balance |
| `results/zero_shot_tf_v5_ens50_50/test_predictions.csv` | 0.50 power07 + 0.50 cw | 0.02186 | **0.01496** | **0.00240** | Best subset + best fairness |

**Recommended: v6 (`power06_tta`)** because:
- Best test-like under the corrected distribution from the brief
- Predictions mean (0.245) closer to test distribution mean (~0.18 estimated)

**Safe alternative: v2 (`power07_tta`)** because:
- Test-like differs by only 4% from v6 (within distribution-reading noise)
- Lower gap (0.00877 vs 0.01717) → safer on fairness penalty
- Predictions mean (0.195) compromise between train (0.083) and test (~0.18)

**Risky alternative: v5 (50/50 ensemble)** if you really fear the gap term:
v5 has 7× smaller gap but ~28% worse test-like.

## The 10 cases

| # | Date | File submitted | Variant | Interim score | Gap (interim) | val-to-interim ratio | Decision rationale |
|---|------|----------------|---------|---------------|---------------|----------------------|--------------------|
| 1 |      |                |         |               |               |                      |                    |
| 2 |      |                |         |               |               |                      |                    |
| 3 |      |                |         |               |               |                      |                    |
| 4 |      |                |         |               |               |                      |                    |
| 5 |      |                |         |               |               |                      |                    |
| 6 |      |                |         |               |               |                      |                    |
| 7 |      |                |         |               |               |                      |                    |
| 8 |      |                |         |               |               |                      |                    |
| 9 |      |                |         |               |               |                      |                    |
| 10 |     |                |         |               |               |                      | **Final — best of everything** |

## Track

**Training-Free uniquement** (décision 2026-05-26).

Toutes les soumissions vont à cette track. Pipelines candidats :
- `zero_shot_v1_heuristic` — SegFormer face-parsing + heuristique fixe
- `julien_tf_*` — 3DDFA-V2 + BiSeNet + heuristiques (pipeline de Julien)
- Ensemble des deux

Track Model Training : abandonnée.

## Pre-submission checklist

Before clicking "Submit" on hfactory, verify:

- [ ] The CSV has exactly **29 980 rows**
- [ ] 3 columns: `filename`, `FaceOcclusion`, `gender`
- [ ] No NaN, no negatives, no values > 1
- [ ] `gender` column filled with `x` (or whatever the platform expects)
- [ ] The variant is recorded in `EXPERIMENTS.md` with `Submit candidate` decision
- [ ] We expect interim better than (or comparable to) our previous submission
- [ ] If we're about to submit a "risky" variant, the next one is planned to fall back to a known-good

## Once a submission is graded

1. Note the interim score in the table above
2. Compute the val-to-interim ratio = `interim / val_subset_score`. Watch how it drifts across submissions.
3. If the variant did better than expected → keep exploring that family
4. If it did worse → think about which population is now hurting (gap? high-occlusion?), then plan the next one
