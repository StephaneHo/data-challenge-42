"""Extract per-class areas and ratios from face-parsing label maps."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from src.zero_shot.parser import (
    CLASS_BACKGROUND,
    CLASS_CLOTH,
    CLASS_EYE_G,
    CLASS_HAIR,
    CLASS_HAT,
    FACE_PART_CLASSES,
    HAIR_CLASSES,
    OCCLUDER_CLASSES,
)


FEATURE_NAMES = (
    "face_area_frac",       # pixels in FACE_PARTS / total pixels
    "occluder_area_frac",   # pixels in OCCLUDERS (= eye_g + hat + cloth) / total pixels
    "hair_area_frac",       # pixels in HAIR / total pixels
    "background_frac",      # pixels in BACKGROUND / total pixels
    # Per-class subdivisions of OCCLUDERS (added 2026-06-01 for finer corrections):
    "hat_area_frac",        # pixels in HAT / total pixels
    "glasses_area_frac",    # pixels in EYE_G / total pixels
    "cloth_area_frac",      # pixels in CLOTH / total pixels
    "convex_hull_area_frac",  # area of convex hull of FACE_PARTS / total pixels
    "occluder_in_hull_frac",  # occluder pixels inside convex hull / hull area
    "hair_in_hull_frac",      # hair pixels inside convex hull / hull area
    "bg_in_hull_frac",        # background pixels inside hull / hull area (catches hands etc.)
    "hat_in_hull_frac",       # hat pixels inside hull / hull area
    "glasses_in_hull_frac",   # glasses pixels inside hull / hull area
    "cloth_in_hull_frac",     # cloth pixels inside hull / hull area
    "ratio_simple",         # occluder_area / (face_area + occluder_area)
    "ratio_hull",           # (hull − face) / hull (the "geometric" ratio)
    "ratio_hull_strict",    # (occluder_in_hull) / hull
)


def _filled_convex_hull_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Compute a binary mask filling the convex hull of `binary_mask`.

    Uses scipy.ndimage.binary_fill_holes after a coarse dilation to approximate
    the convex hull cheaply. For a strict convex hull, scipy.spatial.ConvexHull
    would be more accurate but adds polygon rasterization overhead.

    For our 224x224 face crops this is a good enough proxy.
    """
    if binary_mask.sum() < 4:
        return np.zeros_like(binary_mask, dtype=bool)
    # Dilate to bridge small gaps, then fill holes, then erode back.
    dilated = ndimage.binary_dilation(binary_mask, iterations=3)
    filled = ndimage.binary_fill_holes(dilated)
    return filled


def extract_features(seg_map: np.ndarray) -> dict[str, float]:
    """Compute a fixed-size feature vector from a single segmentation map.

    Parameters
    ----------
    seg_map : np.ndarray of shape (H, W), uint8 with class labels in [0, 18]

    Returns
    -------
    dict mapping FEATURE_NAMES to float values in [0, 1]
    """
    total = float(seg_map.size)
    face_mask = np.isin(seg_map, FACE_PART_CLASSES)
    occluder_mask = np.isin(seg_map, OCCLUDER_CLASSES)
    hair_mask = np.isin(seg_map, HAIR_CLASSES)
    bg_mask = seg_map == CLASS_BACKGROUND
    # Per-class occluder masks
    hat_mask = seg_map == CLASS_HAT
    glasses_mask = seg_map == CLASS_EYE_G
    cloth_mask = seg_map == CLASS_CLOTH

    face_area = float(face_mask.sum())
    occluder_area = float(occluder_mask.sum())
    hair_area = float(hair_mask.sum())
    bg_area = float(bg_mask.sum())
    hat_area = float(hat_mask.sum())
    glasses_area = float(glasses_mask.sum())
    cloth_area = float(cloth_mask.sum())

    hull_mask = _filled_convex_hull_mask(face_mask)
    hull_area = float(hull_mask.sum())

    if hull_area > 0:
        occluder_in_hull = float((occluder_mask & hull_mask).sum())
        hair_in_hull = float((hair_mask & hull_mask).sum())
        bg_in_hull = float((bg_mask & hull_mask).sum())
        hat_in_hull = float((hat_mask & hull_mask).sum())
        glasses_in_hull = float((glasses_mask & hull_mask).sum())
        cloth_in_hull = float((cloth_mask & hull_mask).sum())
        face_in_hull = float((face_mask & hull_mask).sum())
        ratio_hull = max(0.0, (hull_area - face_in_hull) / hull_area)
        ratio_hull_strict = occluder_in_hull / hull_area
    else:
        occluder_in_hull = 0.0
        hair_in_hull = 0.0
        bg_in_hull = 0.0
        hat_in_hull = 0.0
        glasses_in_hull = 0.0
        cloth_in_hull = 0.0
        ratio_hull = 0.0
        ratio_hull_strict = 0.0

    denom_simple = face_area + occluder_area
    ratio_simple = (occluder_area / denom_simple) if denom_simple > 0 else 0.0

    return {
        "face_area_frac": face_area / total,
        "occluder_area_frac": occluder_area / total,
        "hair_area_frac": hair_area / total,
        "background_frac": bg_area / total,
        "hat_area_frac": hat_area / total,
        "glasses_area_frac": glasses_area / total,
        "cloth_area_frac": cloth_area / total,
        "convex_hull_area_frac": hull_area / total,
        "occluder_in_hull_frac": (occluder_in_hull / hull_area) if hull_area > 0 else 0.0,
        "hat_in_hull_frac": (hat_in_hull / hull_area) if hull_area > 0 else 0.0,
        "glasses_in_hull_frac": (glasses_in_hull / hull_area) if hull_area > 0 else 0.0,
        "cloth_in_hull_frac": (cloth_in_hull / hull_area) if hull_area > 0 else 0.0,
        "hair_in_hull_frac": (hair_in_hull / hull_area) if hull_area > 0 else 0.0,
        "bg_in_hull_frac": (bg_in_hull / hull_area) if hull_area > 0 else 0.0,
        "ratio_simple": ratio_simple,
        "ratio_hull": ratio_hull,
        "ratio_hull_strict": ratio_hull_strict,
    }


def features_dataframe(seg_maps: np.ndarray, filenames: list[str]) -> pd.DataFrame:
    """Vectorize feature extraction over a stack of segmentation maps.

    Parameters
    ----------
    seg_maps : (N, H, W) uint8
    filenames : list of length N
    """
    rows = [extract_features(seg_maps[i]) for i in range(len(seg_maps))]
    df = pd.DataFrame(rows)
    df.insert(0, "filename", filenames)
    return df
