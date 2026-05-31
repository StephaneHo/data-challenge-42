# Experiments Log

This file is the leaderboard of all pipeline variants we've evaluated.
Add a row whenever you run `scripts/eval_harness.py` on a new variant.

**Rules of the game** (training-free track + general hygiene):
- One row = one variant evaluated against the fixed eval subset (hash recorded in `reports/<variant>/meta.json`).
- The eval subset must stay the same across rows for absolute scores to be comparable.
- "Decision" column drives the workflow: `Reference` (baseline kept), `Iterate` (worth a follow-up), `Drop` (rejected), `Submit candidate` (queued for hfactory).
- Don't edit past rows — append. Past failures are useful.
- For the **training-free track**, the `Compliance` column must be `TF` (TF-compliant) or `Trained` (uses fit on dataset). Only TF variants can go to the TF track on Moodle.

For tactical comparison run `python scripts/compare_reports.py`.

---

## Variants

All variants below were evaluated against the same fixed eval subset (hash `1b1686fb`, 2000 samples from val).
Per-bin × per-gender breakdown is in each `reports/<variant>/per_bin.csv`.

### Historical reference (Model Training track, no longer pursued)

| ID | Date | Variant | Hypothesis | Subset score | Test-like est. (brief) | Gap F/M | Compliance | Decision |
|----|------|---------|-----------|--------------|------------------------|---------|------------|----------|
| 001 | 2026-05-25 | `trained_resnet50_8ep_balanced` | ResNet50 8 epochs, balanced sampler + loss | 0.00128 | 0.00212 | 0.00051 | Trained | Reference (track abandoned 2026-05-26) |
| 002 | 2026-05-26 | `zero_shot_v0_calibrated` | SegFormer + 11 features + Ridge calibration on 500 train | 0.01035 | 0.01704 | 0.00311 | Trained (Ridge) | Reference (NOT TF-compliant — Ridge prohibited) |
| 003 | 2026-05-26 | (no report) `ensemble alpha=0.98` | Linear blend trained 8ep + zero-shot v0 on full 15k val | 0.00171 | n/a | 0.00071 | Trained | Dropped (optimum alpha = pure trained, no gain) |

### TF-compliant variants — SegFormer face-parsing pipeline (our side, 2026-05-30/31)

All TF variants use the **same eval subset** (`1b1686fb`) so deltas are directly comparable.
Reference for delta columns: `zs_simple_hull_scaled` (the current best).

| ID | Date | Variant | Hypothesis (what changed vs the previous) | Subset score | Δ vs ref | Test-like est. | Δ vs ref | Gap F/M | Δ gap | Decision |
|----|------|---------|------|---|---|---|---|---|---|---|
| 004 | 2026-05-31 | `zs_simple_hull` | Baseline TF: `pred = clip(ratio_hull, 0, 1)`. Single feature, no scaling, no fit. | 0.03249 | +0.00972 | 0.03737 | +0.00780 | 0.01351 | +0.00584 | Reference (worst) |
| 005 | 2026-05-31 | `zs_simple_hull_scaled` | Same as 004 but × 1.5 multiplier (chosen because raw under-shoots GT). | 0.02277 | **(ref) best** | **0.02957** | (ref) | **0.00767** | (ref) | **Best so far** |
| 006 | 2026-05-31 | `zs_multi_feature` | 0.5·ratio_hull + 0.3·occluder_in_hull + 0.2·bg_in_hull. Combine 3 signals. | 0.04381 | +0.02104 | 0.04507 | +0.01550 | 0.01924 | +0.01157 | Drop (worse than 004) |
| 007 | 2026-05-31 | `zs_multi_feature_scaled` | Same as 006 × 1.5 multiplier. | 0.03748 | +0.01471 | 0.03867 | +0.00910 | 0.01628 | +0.00861 | Drop (worse than 005) |
| 008 | 2026-05-31 | `zs_pose_aware` | multi_feature + bonus `0.5 × max(0, 0.3 − face_area_frac)` for very small face area. | 0.04366 | +0.02089 | 0.04434 | +0.01477 | 0.01914 | +0.01147 | Drop |
| 009 | 2026-05-31 | `zs_hair_aware` | multi_feature + `0.2 × hair_in_hull_frac` (hair counts a bit toward occlusion). | 0.03951 | +0.01674 | 0.04326 | +0.01369 | 0.01699 | +0.00932 | Drop |

### TTA flip variants (2026-05-31)

Each heuristic above was also evaluated with horizontal-flip TTA (run on cached
`val_segformer_features_flip.csv`). The variant suffix is `_tta`.

| Variant | Subset score | Δ vs same-no-TTA | Test-like est. | Δ vs same-no-TTA |
|---|---|---|---|---|
| `zs_simple_hull_scaled_tta` | **0.02248** | −0.00029 | **0.02905** | −0.00052 |
| `zs_simple_hull_tta` | 0.03229 | −0.00020 | 0.03707 | −0.00030 |
| `zs_multi_feature_scaled_tta` | 0.03734 | −0.00014 | 0.03833 | −0.00034 |
| `zs_hair_aware_tta` | 0.03938 | −0.00013 | 0.04305 | −0.00021 |
| `zs_pose_aware_tta` | 0.04356 | −0.00010 | 0.04393 | −0.00041 |
| `zs_multi_feature_tta` | 0.04371 | −0.00010 | 0.04481 | −0.00026 |

**TTA gain on test-like: -0.5% to -1.8% depending on the heuristic.** Smaller than the typical 3-5% from literature because our heuristics are very simple (low prediction variance), so there's little noise to average out. Still: TTA helps in every case, never hurts.

**Current best TF candidate: `zs_simple_hull_scaled_tta`** (test-like estimate **0.02905**).

### Phase A2 — Power transformation + class-weighted variants (2026-05-31)

After observing that the heuristics systematically under-predict the
high-occlusion tail, we tried fixed concave power transformations (`x^k` with
`k < 1`) to compress low values less than high ones and boost the tail.

| Variant | Hypothesis | Subset score | Δ vs ref | Test-like est. | Δ vs ref | Gap F/M | Decision |
|---|---|---|---|---|---|---|---|
| `zs_simple_hull_scaled_power07_tta` | `simple_hull_scaled` ^ 0.7 + TTA | 0.02239 | **(best)** | **0.01658** | (best) | 0.00877 | **NEW BEST candidate** |
| `zs_simple_hull_scaled_power07` | same, no TTA | 0.02307 | +0.00068 | 0.01721 | +0.00063 | 0.00904 | |
| `zs_simple_hull_scaled_power06_tta` | ^ 0.6 (more aggressive boost) | 0.03736 | | 0.01728 | +0.00070 | 0.01717 | Drop |
| `zs_simple_hull_scaled_power08_tta` | ^ 0.8 (milder boost) | 0.01394 | best subset | 0.01952 | +0.00294 | **0.00201** | Drop (gap minimum, but tail under-predicted) |
| `zs_simple_hull_scaled_power05_tta` | ^ 0.5 = sqrt (very aggressive) | 0.06323 | | 0.02518 | | 0.02754 | Drop (too aggressive) |
| `zs_class_weighted_scaled_tta` | new feature mix × 1.5 (1·ratio_hull + 0.5·bg_in_hull + 0.3·occluder + 0.1·hair) | 0.01930 | -0.00309 | 0.02537 | | 0.00476 | Reference for ensembles |
| `zs_class_weighted_scaled_power07_tta` | class_weighted_scaled ^ 0.7 | 0.03300 | | 0.01826 | +0.00168 | 0.01528 | Drop (worse than simple variant) |
| `zs_adaptive_floor_tta` | simple_hull_scaled + floor 0.15 if occluder > 5% | 0.02244 | | 0.02901 | | 0.00768 | Drop (no improvement) |
| `zs_piecewise_boost_tta` | piecewise linear boost in [0.1, 0.5] | 0.02190 | | 0.02927 | | 0.00693 | Drop |

**Key insight:** `simple_hull_scaled_power07` divides our best test-like estimate by **−43%** vs the previous best (`simple_hull_scaled_tta`: 0.02905 → 0.01658). A simple concave power transform was a much bigger win than any feature engineering.

### Phase B — Face Mesh signal (2026-05-31)

Cached MediaPipe Face Mesh on full val (98.2% detection rate, 1.8% failures or face_mesh_area < 0.15).
Added fixed bonuses to `power07_tta` when MediaPipe fails or detects a tiny face.

| Variant | Hypothesis | Subset | Test-like | Gap | Decision |
|---|---|---|---|---|---|
| `zs_mesh_no_face_floor_tta` | +0.10 bonus when MediaPipe fails (1.8% images) | 0.02306 | 0.01732 | 0.00893 | Drop (slightly worse) |
| `zs_mesh_tiny_face_floor_tta` | +0.05 bonus when mesh area < 0.15 (1.8%) | 0.02261 | 0.01691 | 0.00871 | Drop |
| `zs_mesh_combined_tta` | both bonuses | 0.02311 | 0.01732 | 0.00897 | Drop |

**Conclusion:** Face Mesh signals don't help. The 1.8% of affected images don't carry enough signal even with reasonable fixed bonuses.

### Phase C2 — CLIP zero-shot (SKIPPED)

Tested CLIP-vit-base on 50 val samples with discrete occlusion prompts ("a clear face", ..., "a face mostly hidden").
**Pearson correlation with GT = −0.16** (essentially noise). Skipped full val run (~2h saved).

### Ensemble experiments (2026-05-31)

Linear blends of the best per-metric variants:

| Variant | Composition | Subset | Test-like (subset) | Gap | Notes |
|---|---|---|---|---|---|
| `zs_simple_hull_scaled_power07_tta` | 1.0 × power07 + 0.0 × cw | 0.02239 | 0.01658 | 0.00877 | Best test-like |
| `zs_ens_power07_power08_70` | 0.7 power07 + 0.3 power08 | 0.01953 | 0.01712 | 0.00675 | |
| `zs_ens_power07_power08_50` | 0.5 power07 + 0.5 power08 | 0.01778 | 0.01765 | 0.00540 | |
| `zs_ens_power07_cw_70` | 0.7 power07 + 0.3 class_weighted_scaled | 0.01766 | 0.01813 | 0.00504 | |
| `zs_ens_power07_cw_50` | 0.5 power07 + 0.5 class_weighted_scaled | 0.01496 | 0.01968 | **0.00240** | Best subset + best gap |

**Pattern:** ensembles improve subset score and reduce gap BUT degrade test-like estimate. The reason: power07_tta amplifies the tail (matching the test distribution from the brief), while class_weighted_scaled is more conservative. Mixing pulls toward conservative.

**Submission strategy:**
- If we trust the brief test histogram → `power07_tta` alone (best test-like 0.01658)
- If we trust the train-like subset → 50/50 ensemble (best subset 0.01496, best gap 0.00240)
- Middle ground (interim score balances both): 85/15 split

Four submission CSVs generated, all from the same test cache:
| Path | Composition | Predicted mean |
|---|---|---|
| `results/zero_shot_tf_v2/test_predictions.csv` | power07_tta alone | 0.195 |
| `results/zero_shot_tf_v3_ens85_15/test_predictions.csv` | 0.85 power07 + 0.15 cw | 0.183 |
| `results/zero_shot_tf_v4_ens70_30/test_predictions.csv` | 0.70 power07 + 0.30 cw | 0.172 |
| `results/zero_shot_tf_v5_ens50_50/test_predictions.csv` | 0.50 power07 + 0.50 cw | 0.157 |

For comparison the brief's test distribution has mean ~0.15.

### ⚠️ Correction de la distribution test (2026-05-31)

In a careful re-reading of the test histogram in `task_brief.pdf` (29 980 images), I corrected the bin-probability estimate from
`[0.13, 0.17, 0.18, 0.17, 0.22, 0.12, 0.01]` to
`[0.18, 0.16, 0.14, 0.15, 0.26, 0.15, 0.003]`.

Main changes:
- [0.00, 0.05): underestimated by 5 points (peak at 0 is taller than I thought)
- [0.10, 0.15): overestimated by 4 points
- [0.20, 0.30): underestimated by 4 points

All reports' `score.json` and the `test-like (from brief)` column have been re-computed with the corrected distribution.

**Resulting ranking shake-up**:

| Variant | Old test-like | NEW test-like | Δ |
|---|---|---|---|
| `zs_simple_hull_scaled_power06_tta` | 0.01728 | **0.01712** | **NEW BEST** |
| `zs_simple_hull_scaled_power07_tta` | 0.01658 | 0.01777 | was best, now #3 (+7%) |
| `zs_mesh_tiny_face_floor_tta` | 0.01691 | 0.01810 | (+7%) |
| `zs_ens_power07_power08_70` | 0.01712 | 0.01863 | (+9%) |

Power 0.6 amplifies the tail more aggressively than power 0.7, which now pays
off because the corrected distribution puts more weight on the tail bins.

**Uncertainty disclosure**: my visual reading of the histogram could be off by
±3% per bin. Variants whose test-like estimates differ by less than ~5% are
essentially tied within this noise.

### Per-bin × per-gender breakdown of the top 5 (subset hash 1b1686fb, 2000 samples)

This is the decomposition Julien specifically asked for: where does each variant gain or lose?

**Female weighted error per bin (smaller = better)**

| bin | power07_tta | ens 70/30 | ens 50/50 | power08_tta | simple_hull_scaled_tta | trained (historical) |
|---|---|---|---|---|---|---|
| [0.00, 0.05) | 0.02350 | 0.02034 | 0.01445 | 0.01396 | **0.00534** | 0.00068 |
| [0.05, 0.10) | 0.00950 | 0.00749 | 0.00374 | 0.00379 | **0.00058** | 0.00042 |
| [0.10, 0.15) | 0.00408 | 0.00294 | 0.00147 | **0.00133** | 0.00237 | 0.00041 |
| [0.15, 0.20) | **0.00284** | 0.00269 | 0.00416 | 0.00339 | 0.00900 | 0.00041 |
| [0.20, 0.30) | **0.00360** | 0.00483 | 0.00929 | 0.00885 | 0.02195 | 0.00062 |
| [0.30, 0.50) | **0.02378** | 0.02763 | 0.03877 | 0.03785 | 0.06413 | 0.00067 |

**Male weighted error per bin (smaller = better)**

| bin | power07_tta | ens 70/30 | ens 50/50 | power08_tta | simple_hull_scaled_tta | trained (historical) |
|---|---|---|---|---|---|---|
| [0.00, 0.05) | 0.02812 | 0.02450 | 0.01845 | 0.01708 | **0.00639** | 0.00061 |
| [0.05, 0.10) | 0.01803 | 0.01525 | 0.01076 | 0.00984 | **0.00333** | 0.00092 |
| [0.10, 0.15) | 0.00850 | 0.00673 | 0.00436 | **0.00371** | 0.00222 | 0.00085 |
| [0.15, 0.20) | 0.00402 | **0.00337** | 0.00341 | 0.00300 | 0.00689 | 0.00070 |
| [0.20, 0.30) | **0.00616** | 0.00724 | 0.01093 | 0.01091 | 0.02320 | 0.00333 |
| [0.30, 0.50) | **0.03237** | 0.03665 | 0.04768 | 0.04776 | 0.07504 | 0.00086 |

**Sample counts in the eval subset (same for all variants)**

| bin | n_F | n_M |
|---|---|---|
| [0.00, 0.05) | 142 | 828 |
| [0.05, 0.10) | 172 | 199 |
| [0.10, 0.15) | 156 | 111 |
| [0.15, 0.20) | 96 | 73 |
| [0.20, 0.30) | 108 | 46 |
| [0.30, 0.50) | 53 | 16 |

**Per-bin reading (critical for choosing what to submit):**

- **Low bins [0.00, 0.10)** : `simple_hull_scaled_tta` (no power transform) wins. The power 0.7 inflates predictions for samples with true GT ≈ 0, which hurts here.
- **Mid bins [0.10, 0.20)** : `power08_tta` or `power07_tta` win.
- **High bins [0.20, 0.50)** : `power07_tta` dominates. Designed for this.

The trade-off is **systematic**: power07 sacrifices low-occlusion accuracy to gain tail accuracy.

The brief's test distribution puts ~13% mass in [0, 0.05) but ~12% in [0.30, 0.50). Our eval subset has 49% in [0, 0.05) (train-like). So `power07_tta` looks slightly worse on the subset but much better on the test-like estimate, and the test-like estimate is closer to what the FINAL score will reflect.

**Implication for Julien**: when comparing your 3DDFA pipeline variants, look at the per-bin breakdown. A variant that improves the mean might be losing accuracy on a specific occluded population.

### Key findings from the TF ablation (for Julien)

1. **Simple beats complex.** A single-feature heuristic (`ratio_hull`) with a fixed 1.5× scaling beats every multi-feature combination we tried. The 0.5 weight in `multi_feature` dilutes the dominant geometric signal too much.

2. **Scaling matters a lot.** `simple_hull` → `simple_hull_scaled` saved 0.008 on test-like estimate (−21%). The raw `ratio_hull` systematically under-shoots GT because the SegFormer parser over-segments hair as face in some images.

3. **Adding more features HURTS without learned combination.** When we can't fit weights (TF rule), arbitrary 0.5/0.3/0.2-style weights make things worse than just using the dominant signal alone. **Implication for Julien**: stick with one well-chosen geometric signal + fixed scaling, rather than trying to blend many auxiliary signals with arbitrary coefficients.

4. **Pose-aware and hair-aware bonuses didn't help.** Adding small bonuses for "small face area" (proxy for extreme pose) or "hair in face area" worsened the score. The signals are noisy at the per-sample level even though they correlate with GT in aggregate.

5. **Female error is 2× male error** across all variants. The zero-shot inherently struggles more on women (likely because they have higher mean occlusion in train and the heuristic under-predicts the high-occlusion tail).

6. **TF inherently costs ~10× vs Trained on this dataset.** Best TF (0.030) vs trained baseline (0.002) on the same test-like distribution. The training-free constraint is a hard ceiling.

### Pipelines to evaluate next (still TF-only)

| Planned variant | Why | Status |
|---|---|---|
| `zs_simple_hull_scaled_tta` | Best TF variant + horizontal flip averaging | **Done** — test-like 0.02905 |
| `julien_tf_baseline` | 3DDFA-V2 + BiSeNet (real geometric mesh, not SegFormer convex hull) | Pending Julien |
| `julien_tf_tta` | Pipeline Julien + flip averaging | Pending Julien |
| `julien_tf_hands` | Pipeline Julien + MediaPipe Hands bonus | Pending Julien |
| `ensemble_zs_julien` | Blend best `zs_*` + best `julien_*` if both have reasonable scores | Pending both |

### Next step toward submission

For our best variant (`zs_simple_hull_scaled_tta`) we still need to produce a `test_predictions.csv` for hfactory submission. That requires:
1. Cache SegFormer features on the test set (~3h CPU, 30k images)
2. Cache TTA flipped features on test (~3h CPU)
3. Apply `simple_hull_scaled` heuristic with TTA flip averaging → submission CSV

Until then, our submission-ready files are:
- `results/zero_shot/test_predictions.csv` — based on `zero_shot_v0_calibrated` (Ridge, NOT TF-compliant — for reference only, do NOT submit to TF track)

## ⚠️ Track decision (2026-05-26)

**Focus exclusif Training-Free.** Les variantes `trained_*` ne sont plus candidates pour la soumission — `trained_resnet50_8ep_balanced` est gardé comme référence de calibration uniquement.

## How to add a new row

```bash
# 1. Produce val predictions
python scripts/eval_val.py --checkpoint ... --backbone ... --tta flip --out eval/val_<variant>.csv

# 2. Run the harness (writes reports/<variant>/)
python scripts/eval_harness.py \
    --predictions eval/val_<variant>.csv \
    --variant <variant> \
    --notes "Short description"

# 3. Open this file, add a row to the "Variants" table
# 4. Commit the .md + the reports/<variant>/ together
```
