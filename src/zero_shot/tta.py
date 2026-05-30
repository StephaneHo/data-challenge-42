"""Test-Time Augmentation wrappers.

Why: the face-parsing and 3DDFA models give slightly different predictions
on an image and its mirror, even though the occlusion ratio is invariant
to horizontal flip. Averaging the two reduces variance.

Compliance: this is purely deterministic post-processing, no fit involved.
Suitable for the Training-Free track.

Usage:
    from src.zero_shot.tta import tta_flip

    # Wrap any function (img_bgr) -> float
    predict_with_tta = tta_flip(occlusion_computation_julien)

    # Same call signature as the wrapped fn
    ratio = predict_with_tta(app, img_bgr)
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

import cv2
import numpy as np


def tta_flip(predict_fn: Callable) -> Callable:
    """Wrap a prediction function so it averages predictions on the image
    and its horizontal flip.

    The wrapped function must:
      - Take a BGR image (HxWx3 ndarray) as one of its positional or
        keyword arguments (we look for the first ndarray-like argument).
      - Return a single float (the occlusion ratio).

    If the function takes extra positional args (e.g., the InsightFace `app`),
    they pass through unchanged. The image is detected by type.
    """

    @wraps(predict_fn)
    def wrapped(*args, **kwargs):
        new_args = list(args)
        img_index = _find_image_arg(args)
        if img_index is None:
            raise ValueError("tta_flip: no ndarray-like image argument found in call")
        img = args[img_index]
        img_flipped = cv2.flip(img, 1)

        # First call with original image
        pred_orig = float(predict_fn(*args, **kwargs))

        # Second call with flipped image
        new_args[img_index] = img_flipped
        pred_flip = float(predict_fn(*new_args, **kwargs))

        return 0.5 * (pred_orig + pred_flip)

    return wrapped


def tta_average_predictions(*preds: float) -> float:
    """Convenience helper: average an arbitrary number of float predictions."""
    if not preds:
        raise ValueError("need at least one prediction to average")
    return float(np.mean(preds))


def _find_image_arg(args: tuple) -> int | None:
    """Find the index of the first ndarray (HxWx3) argument in args.

    This is heuristic: scans positional args left-to-right and returns the
    first one that looks like an image (3D ndarray with last dim in {3, 4}).
    """
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[-1] in (3, 4):
            return i
    return None
