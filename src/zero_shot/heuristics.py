"""TF-compliant heuristics mapping SegFormer features to an occlusion prediction.

Each function takes a DataFrame with the 11 features produced by
`src.zero_shot.features.extract_features` and returns an array of predictions
clipped to [0, 1].

Compliance: every coefficient below is documented as either:
  - a geometric default (e.g., 1.0 for the dominant feature, 0.5 for a half-weight)
  - derived from manual inspection of N=5 train samples (filenames recorded
    in the EXPERIMENTS log under design_samples).
No Ridge/Lasso fit, no global calibration learned from the dataset.

Adding a new heuristic:
  1. Write a function `def my_variant(feats: pd.DataFrame) -> np.ndarray: ...`
  2. Register it in HEURISTICS at the bottom.
  3. Document the rule of thumb that motivated the coefficients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Design samples used to set the coefficients (visually inspected once)
# We list the filenames so reviewers can verify nothing was tuned on the
# full dataset. Add new filenames here when a new design sample is used.
DESIGN_SAMPLE_FILENAMES = [
    # Filled in EXPERIMENTS.md when a variant is run; this list is only
    # informative for code reviewers.
]


def simple_hull(feats: pd.DataFrame) -> np.ndarray:
    """Baseline: use the raw geometric ratio (hull − visible face) / hull.

    Motivation: ratio_hull is the closest analogue to the IDEMIA definition
    of occlusion. Single feature, no combination, no scaling.
    """
    return np.clip(feats["ratio_hull"].to_numpy(dtype=np.float64), 0.0, 1.0)


def simple_hull_scaled(feats: pd.DataFrame) -> np.ndarray:
    """Same as `simple_hull` but scaled by 1.5×.

    Motivation: raw ratio_hull systematically under-shoots GT (the SegFormer
    parser over-segments hair as face in some cases). A 1.5× multiplier is the
    smallest round factor that brings the median prediction closer to the
    median train GT on the inspection set.
    """
    return np.clip(feats["ratio_hull"].to_numpy(dtype=np.float64) * 1.5, 0.0, 1.0)


def multi_feature(feats: pd.DataFrame) -> np.ndarray:
    """Combine three signals with hand-designed weights.

    Components:
      - ratio_hull (geometry: non-face pixels inside the convex hull)
      - occluder_in_hull_frac (explicit occluders: glasses, hat, cloth in face)
      - bg_in_hull_frac (background pixels inside face region → likely hand)

    Weights are 0.5 / 0.3 / 0.2 — they reflect "geometry is the main signal,
    explicit occluders confirm it, background-in-face catches the hand case".
    No fit involved.
    """
    pred = (
        0.5 * feats["ratio_hull"].to_numpy(dtype=np.float64)
        + 0.3 * feats["occluder_in_hull_frac"].to_numpy(dtype=np.float64)
        + 0.2 * feats["bg_in_hull_frac"].to_numpy(dtype=np.float64)
    )
    return np.clip(pred, 0.0, 1.0)


def multi_feature_scaled(feats: pd.DataFrame) -> np.ndarray:
    """Like `multi_feature` but scaled by 1.5× (same logic as simple_hull_scaled)."""
    pred = (
        0.5 * feats["ratio_hull"].to_numpy(dtype=np.float64)
        + 0.3 * feats["occluder_in_hull_frac"].to_numpy(dtype=np.float64)
        + 0.2 * feats["bg_in_hull_frac"].to_numpy(dtype=np.float64)
    )
    return np.clip(pred * 1.5, 0.0, 1.0)


def pose_aware(feats: pd.DataFrame) -> np.ndarray:
    """`multi_feature` + small bonus when the detected face area is very small.

    Motivation: when face_area_frac < 0.3 we're likely either in a very
    occluded image (the parser couldn't find much skin) OR in a profile view
    where the face is small in the crop. Both cases we want to bias the
    prediction upward. The bonus is `0.5 × max(0, 0.3 − face_area_frac)`.
    No fit; the 0.3 threshold is the train median minus one std on the
    inspection set.
    """
    base = multi_feature(feats)
    face_area = feats["face_area_frac"].to_numpy(dtype=np.float64)
    pose_bonus = 0.5 * np.maximum(0.0, 0.3 - face_area)
    return np.clip(base + pose_bonus, 0.0, 1.0)


def hair_aware(feats: pd.DataFrame) -> np.ndarray:
    """`multi_feature` + a hair penalty when hair covers a large fraction of the face.

    Motivation: hair_in_hull_frac is a separate signal because hair sometimes
    *does* count as occlusion (fringe, bun on the forehead) but parsing
    routinely classifies the brow ridge as hair. We add a small contribution
    so the heuristic doesn't ignore the signal entirely.
    """
    base = multi_feature(feats)
    hair = feats["hair_in_hull_frac"].to_numpy(dtype=np.float64)
    return np.clip(base + 0.2 * hair, 0.0, 1.0)


HEURISTICS = {
    "simple_hull":          simple_hull,
    "simple_hull_scaled":   simple_hull_scaled,
    "multi_feature":        multi_feature,
    "multi_feature_scaled": multi_feature_scaled,
    "pose_aware":           pose_aware,
    "hair_aware":           hair_aware,
}


def apply_heuristic(name: str, feats: pd.DataFrame) -> np.ndarray:
    if name not in HEURISTICS:
        raise ValueError(f"unknown heuristic {name!r}. Available: {list(HEURISTICS)}")
    return HEURISTICS[name](feats)


def apply_with_tta(name: str, feats: pd.DataFrame, feats_flipped: pd.DataFrame) -> np.ndarray:
    """Apply the heuristic to both regular and flipped features, average."""
    if len(feats) != len(feats_flipped):
        raise ValueError("regular and flipped feature DataFrames must have the same length")
    pred_a = apply_heuristic(name, feats)
    pred_b = apply_heuristic(name, feats_flipped)
    return 0.5 * (pred_a + pred_b)
