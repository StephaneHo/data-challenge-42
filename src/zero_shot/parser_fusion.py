"""Fuse two face-parsing models (BiSeNet + SegFormer) at the semantic-group level.

Why: BiSeNet (zllrunning/face-parsing.PyTorch) and SegFormer
(jonathandinu/face-parsing) were both trained on CelebAMask-HQ but use
different class index conventions and make different errors. Combining them
at the per-pixel "is this face / occluder / hair / background" level is more
robust than relying on one alone.

Compliance: no learned parameters. The fusion rule is a fixed majority vote
across both models.

Usage:
    # Run both parsers separately to get per-pixel label maps:
    parsing_bisenet  = bisenet_model_out.argmax(0).cpu().numpy()
    parsing_segformer = segformer_face_parser.parse_one(img_pil)

    # Map both to the unified taxonomy and combine:
    fused = fuse_to_semantic(
        parsing_bisenet, taxonomy="bisenet",
        parsing_other=parsing_segformer, other_taxonomy="segformer",
    )
    # fused is a (H, W) array with values in {FACE, BACKGROUND, GLASSES, ...}
"""
from __future__ import annotations

import numpy as np


# Unified semantic groups (the labels we care about for occlusion estimation).
SEM_BACKGROUND = 0
SEM_FACE = 1
SEM_GLASSES = 2
SEM_HAT = 3
SEM_CLOTH = 4
SEM_HAIR = 5
SEM_OTHER = 6

SEMANTIC_GROUPS = {
    SEM_BACKGROUND: "background",
    SEM_FACE: "face",
    SEM_GLASSES: "glasses",
    SEM_HAT: "hat",
    SEM_CLOTH: "cloth",
    SEM_HAIR: "hair",
    SEM_OTHER: "other",
}


# Maps from raw class index → semantic group.
# BiSeNet (zllrunning/face-parsing.PyTorch, CelebAMask-HQ standard order)
BISENET_TO_SEMANTIC = {
    0: SEM_BACKGROUND,
    1: SEM_FACE,   # skin
    2: SEM_FACE,   # l_brow
    3: SEM_FACE,   # r_brow
    4: SEM_FACE,   # l_eye
    5: SEM_FACE,   # r_eye
    6: SEM_GLASSES,  # eye_g
    7: SEM_OTHER,  # l_ear
    8: SEM_OTHER,  # r_ear
    9: SEM_OTHER,  # ear_r (earring)
    10: SEM_FACE,  # nose
    11: SEM_FACE,  # mouth
    12: SEM_FACE,  # u_lip
    13: SEM_FACE,  # l_lip
    14: SEM_OTHER, # neck
    15: SEM_OTHER, # neck_l (necklace)
    16: SEM_CLOTH, # cloth (includes mask)
    17: SEM_HAIR,  # hair
    18: SEM_HAT,   # hat
}

# SegFormer jonathandinu/face-parsing (different order — read from model card)
SEGFORMER_TO_SEMANTIC = {
    0: SEM_BACKGROUND,
    1: SEM_FACE,   # skin
    2: SEM_FACE,   # nose
    3: SEM_GLASSES,  # eye_g
    4: SEM_FACE,   # l_eye
    5: SEM_FACE,   # r_eye
    6: SEM_FACE,   # l_brow
    7: SEM_FACE,   # r_brow
    8: SEM_OTHER,  # l_ear
    9: SEM_OTHER,  # r_ear
    10: SEM_FACE,  # mouth
    11: SEM_FACE,  # u_lip
    12: SEM_FACE,  # l_lip
    13: SEM_HAIR,  # hair
    14: SEM_HAT,   # hat
    15: SEM_OTHER, # ear_r
    16: SEM_OTHER, # neck_l
    17: SEM_OTHER, # neck
    18: SEM_CLOTH, # cloth
}


TAXONOMY_MAPS = {
    "bisenet": BISENET_TO_SEMANTIC,
    "segformer": SEGFORMER_TO_SEMANTIC,
}


def to_semantic(parsing: np.ndarray, taxonomy: str) -> np.ndarray:
    """Translate a parser's class indices to the unified semantic taxonomy."""
    if taxonomy not in TAXONOMY_MAPS:
        raise ValueError(f"unknown taxonomy {taxonomy!r}, available: {list(TAXONOMY_MAPS)}")
    mapping = TAXONOMY_MAPS[taxonomy]
    out = np.full_like(parsing, fill_value=SEM_OTHER, dtype=np.uint8)
    for raw_id, sem_id in mapping.items():
        out[parsing == raw_id] = sem_id
    return out


def fuse_to_semantic(
    parsing_a: np.ndarray, taxonomy_a: str,
    parsing_b: np.ndarray, taxonomy_b: str,
    strategy: str = "vote_then_face_priority",
) -> np.ndarray:
    """Fuse two parsings into a single semantic-group label map.

    strategy options:
      - "vote_then_face_priority" : if A and B agree → that group. Otherwise:
            prefer FACE when one says FACE and the other says BACKGROUND/OTHER
            (face parsing models tend to under-segment skin); prefer the
            occluder otherwise (FACE losing to GLASSES/HAT/CLOTH).
      - "intersection_face"        : a pixel is FACE only if both agree on FACE;
            otherwise the higher-numbered occluder class wins.
      - "union_occluder"           : a pixel is an occluder if either model
            says so; FACE only if both agree.
    """
    if parsing_a.shape != parsing_b.shape:
        raise ValueError(f"shape mismatch: {parsing_a.shape} vs {parsing_b.shape}")

    sem_a = to_semantic(parsing_a, taxonomy_a)
    sem_b = to_semantic(parsing_b, taxonomy_b)

    if strategy == "vote_then_face_priority":
        out = sem_a.copy()
        disagree = sem_a != sem_b
        # On disagreement, prefer non-background non-OTHER over background/OTHER
        # (i.e., trust whichever model claims a specific class).
        a_specific = (sem_a != SEM_BACKGROUND) & (sem_a != SEM_OTHER)
        b_specific = (sem_b != SEM_BACKGROUND) & (sem_b != SEM_OTHER)
        # Where only B is specific, take B
        out = np.where(disagree & ~a_specific & b_specific, sem_b, out)
        # Where both specific and they disagree, prefer occluders over FACE
        both_specific = disagree & a_specific & b_specific
        occluder_a = np.isin(sem_a, [SEM_GLASSES, SEM_HAT, SEM_CLOTH])
        occluder_b = np.isin(sem_b, [SEM_GLASSES, SEM_HAT, SEM_CLOTH])
        # Prefer occluder when one is occluder and the other is FACE/HAIR
        out = np.where(both_specific & occluder_b & ~occluder_a, sem_b, out)
        return out

    if strategy == "intersection_face":
        agree_face = (sem_a == SEM_FACE) & (sem_b == SEM_FACE)
        # If either says occluder, take the higher index (occluders are 2-4)
        return np.where(agree_face, SEM_FACE, np.maximum(sem_a, sem_b))

    if strategy == "union_occluder":
        # FACE only when both say FACE
        out = np.full_like(sem_a, SEM_OTHER)
        out[(sem_a == SEM_FACE) & (sem_b == SEM_FACE)] = SEM_FACE
        out[(sem_a == SEM_BACKGROUND) & (sem_b == SEM_BACKGROUND)] = SEM_BACKGROUND
        # Occluder = max of both (since occluder IDs are 2-4)
        out = np.where((sem_a >= SEM_GLASSES) | (sem_b >= SEM_GLASSES),
                       np.maximum(sem_a, sem_b), out)
        return out

    raise ValueError(f"unknown strategy {strategy!r}")


def semantic_face_mask(semantic_map: np.ndarray) -> np.ndarray:
    """Convenience: binary mask of pixels classified as face in the semantic map."""
    return (semantic_map == SEM_FACE).astype(np.uint8)


def semantic_occluder_mask(semantic_map: np.ndarray) -> np.ndarray:
    """Convenience: binary mask of pixels classified as an unambiguous occluder
    (glasses, hat, cloth, or background)."""
    return np.isin(semantic_map,
                   [SEM_GLASSES, SEM_HAT, SEM_CLOTH, SEM_BACKGROUND]).astype(np.uint8)
