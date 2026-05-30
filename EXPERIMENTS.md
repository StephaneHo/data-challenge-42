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
| `zs_simple_hull_scaled_tta` | Best TF variant + horizontal flip averaging | Cache running (background) |
| `julien_tf_baseline` | 3DDFA-V2 + BiSeNet (real geometric mesh, not SegFormer convex hull) | Pending Julien |
| `julien_tf_tta` | Pipeline Julien + flip averaging | Pending Julien |
| `julien_tf_hands` | Pipeline Julien + MediaPipe Hands bonus | Pending Julien |
| `ensemble_zs_julien` | Blend best `zs_*` + best `julien_*` if both have reasonable scores | Pending both |

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
