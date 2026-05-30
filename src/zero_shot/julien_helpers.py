"""Integration helpers for Julien's pipeline.

Drop-in convenience layer that wires our improvements (TTA, hand detection,
cross-check decomposition, parser fusion) into a single enhanced predictor.

Compliance: all combinations are fixed heuristics with no fit step.
The blending coefficients below are chosen from manual inspection on
3-5 train images (the "design samples" disclosed in the EXPERIMENTS log),
not optimized on the dataset.

Usage in Julien's notebook:

    from src.zero_shot.tta import tta_flip
    from src.zero_shot.hands import HandOcclusionDetector
    from src.zero_shot.cross_check import OcclusionDecomposition
    from src.zero_shot.julien_helpers import enhanced_occlusion

    hand_detector = HandOcclusionDetector()

    def occlusion_v1(app, img):
        # Julien's existing pipeline up to having mask_theoretical and parsing:
        # ... (his code) ...
        mask_theoretical, parsing, base_ratio = his_pipeline_returning_three_things(app, img)

        # Add the cross-check decomposition and the hand bonus
        return enhanced_occlusion(
            base_ratio=base_ratio,
            mask_theoretical=mask_theoretical,
            parsing=parsing,
            img_bgr=img,
            hand_detector=hand_detector,
        )

    # And wrap with TTA:
    occlusion_v2 = tta_flip(occlusion_v1)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.zero_shot.cross_check import OcclusionDecomposition


# Fixed weights derived from manual inspection of 5 train images
# (filename hashes recorded in the report's design_samples field).
# Do NOT optimize these on the dataset — that would violate the TF rule.
HEURISTIC_WEIGHTS = {
    "hard_occluders_bonus": 0.30,   # weight given to glasses/hat/cloth/bg above the base ratio
    "hand_bonus":           0.50,   # weight given to detected-hand overlap
    "cap":                  0.95,   # ceil for the final prediction
    "floor":                0.0,    # floor for the final prediction
}


def enhanced_occlusion(
    base_ratio: float,
    mask_theoretical: np.ndarray,
    parsing: np.ndarray,
    img_bgr: Optional[np.ndarray] = None,
    hand_detector=None,
    weights: dict = HEURISTIC_WEIGHTS,
) -> float:
    """Combine the base ratio with cross-check decomposition and hand detection.

    The combination is a fixed convex blend, NOT learned from the dataset.

    Parameters
    ----------
    base_ratio : Julien's pipeline output (1 - visible/expected)
    mask_theoretical : 3DDFA-derived binary face region
    parsing : BiSeNet (or fused) label map (must use BiSeNet taxonomy by default)
    img_bgr : original BGR image (only needed if hand_detector is provided)
    hand_detector : optional HandOcclusionDetector instance

    Returns
    -------
    A single float in [0, 1].
    """
    decomp = OcclusionDecomposition.from_masks(mask_theoretical, parsing)

    # The cross-check view: how much of the "expected face" is unambiguously
    # occluded by hard occluders (glasses, hat, cloth, background).
    hard = decomp.hard_occluders

    # Take the max of (Julien's base, hard occluders) — these are two different
    # estimators of the same thing, and the max is conservative.
    combined = max(base_ratio, hard * (1.0 + weights["hard_occluders_bonus"]))

    # Add the hand bonus if a detector is provided
    if hand_detector is not None and img_bgr is not None:
        hand_overlap = hand_detector.hand_in_face_ratio(img_bgr, mask_theoretical)
        combined = combined + weights["hand_bonus"] * hand_overlap

    # Clamp
    return float(min(weights["cap"], max(weights["floor"], combined)))


def make_enhanced_predictor(
    base_predict_fn,
    hand_detector=None,
    extract_mesh_and_parsing=None,
    weights: dict = HEURISTIC_WEIGHTS,
):
    """Wrap Julien's predict function so it returns the enhanced ratio.

    Requires:
      - `base_predict_fn(app, img)` returns the base ratio
      - `extract_mesh_and_parsing(app, img)` returns (mask_theoretical, parsing)
        if Julien doesn't already return them from `base_predict_fn`.

    Usage:
        predict_v1 = make_enhanced_predictor(
            base_predict_fn=julien_occlusion,
            hand_detector=HandOcclusionDetector(),
            extract_mesh_and_parsing=julien_intermediate_results,
        )
        ratio = predict_v1(app, img)
    """
    if extract_mesh_and_parsing is None:
        raise ValueError(
            "extract_mesh_and_parsing must be provided so we can compute the "
            "cross-check decomposition. Refactor Julien's pipeline so it can "
            "expose (mask_theoretical, parsing) alongside the base ratio."
        )

    def predict(app, img):
        base = float(base_predict_fn(app, img))
        mask_theoretical, parsing = extract_mesh_and_parsing(app, img)
        return enhanced_occlusion(
            base_ratio=base,
            mask_theoretical=mask_theoretical,
            parsing=parsing,
            img_bgr=img,
            hand_detector=hand_detector,
            weights=weights,
        )

    return predict
