# hfactory Submissions Log

10 submissions max total. Each new submission **overwrites the previous one** —
only the LAST counts for the final official score. Choose what to submit very
deliberately.

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
