"""Cross-check between the 3D theoretical face mesh and the 2D face parsing.

Julien's pipeline already counts "non-face pixels inside the theoretical mesh"
implicitly via (1 - visible_ratio). This module makes the contribution of each
occluder type explicit and exposes auxiliary features that the heuristic
combination at the end of his pipeline can use.

Compliance: pure numpy bookkeeping over already-computed model outputs.
No fit, no learned parameters.

Usage:
    from src.zero_shot.cross_check import OcclusionDecomposition

    decomp = OcclusionDecomposition.from_masks(mask_theoretical, parsing)
    # decomp.glasses, decomp.hat, decomp.cloth, decomp.hair_in_face,
    # decomp.background_in_face, decomp.total_occluded, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# CelebAMask-HQ taxonomy used by both BiSeNet and SegFormer face-parsing.
# Note: BiSeNet and our SegFormer differ on a few indices — this module
# accepts either taxonomy through parameter overrides.
class BiSeNetClasses:
    BACKGROUND = 0
    SKIN = 1
    L_BROW = 2
    R_BROW = 3
    L_EYE = 4
    R_EYE = 5
    EYE_G = 6
    L_EAR = 7
    R_EAR = 8
    EAR_R = 9
    NOSE = 10
    MOUTH = 11
    U_LIP = 12
    L_LIP = 13
    NECK = 14
    NECK_L = 15
    CLOTH = 16
    HAIR = 17
    HAT = 18


# Default groups (BiSeNet taxonomy)
DEFAULT_FACE_CLASSES = (BiSeNetClasses.SKIN, BiSeNetClasses.L_BROW, BiSeNetClasses.R_BROW,
                       BiSeNetClasses.L_EYE, BiSeNetClasses.R_EYE, BiSeNetClasses.L_EAR,
                       BiSeNetClasses.R_EAR, BiSeNetClasses.NOSE, BiSeNetClasses.MOUTH,
                       BiSeNetClasses.U_LIP, BiSeNetClasses.L_LIP)
DEFAULT_BACKGROUND_CLASSES = (BiSeNetClasses.BACKGROUND,)
DEFAULT_GLASSES_CLASSES = (BiSeNetClasses.EYE_G,)
DEFAULT_HAT_CLASSES = (BiSeNetClasses.HAT,)
DEFAULT_CLOTH_CLASSES = (BiSeNetClasses.CLOTH,)
DEFAULT_HAIR_CLASSES = (BiSeNetClasses.HAIR,)


@dataclass
class OcclusionDecomposition:
    """Decomposed view of the occlusion inside the theoretical face mesh.

    All fractions are computed as `(class pixels in mesh) / (mesh pixels)`.
    """

    mesh_area: int
    face_in_mesh: float
    background_in_mesh: float  # likely hand / out-of-frame / very dark areas
    glasses_in_mesh: float
    hat_in_mesh: float
    cloth_in_mesh: float       # masks, scarves, garments crossing the face
    hair_in_mesh: float
    other_in_mesh: float       # ears, neck, etc.

    @property
    def total_occluded(self) -> float:
        """1 - face_in_mesh. Sum of all non-face fractions inside the mesh."""
        return float(max(0.0, min(1.0, 1.0 - self.face_in_mesh)))

    @property
    def hard_occluders(self) -> float:
        """Sum of glasses + hat + cloth + background — the unambiguous occluders.

        Hair is excluded because IDEMIA may or may not count it as occlusion
        depending on whether it actually covers face pixels.
        """
        return float(self.glasses_in_mesh + self.hat_in_mesh
                     + self.cloth_in_mesh + self.background_in_mesh)

    @classmethod
    def from_masks(
        cls,
        mask_theoretical: np.ndarray,
        parsing: np.ndarray,
        face_classes=DEFAULT_FACE_CLASSES,
        background_classes=DEFAULT_BACKGROUND_CLASSES,
        glasses_classes=DEFAULT_GLASSES_CLASSES,
        hat_classes=DEFAULT_HAT_CLASSES,
        cloth_classes=DEFAULT_CLOTH_CLASSES,
        hair_classes=DEFAULT_HAIR_CLASSES,
    ) -> "OcclusionDecomposition":
        """Build a decomposition from a (H, W) theoretical mask and parsing label map."""
        mesh = mask_theoretical > 0
        mesh_area = int(mesh.sum())
        if mesh_area == 0:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        def frac(classes):
            return float(np.isin(parsing, list(classes))[mesh].mean())

        face = frac(face_classes)
        bg = frac(background_classes)
        glasses = frac(glasses_classes)
        hat = frac(hat_classes)
        cloth = frac(cloth_classes)
        hair = frac(hair_classes)
        accounted = face + bg + glasses + hat + cloth + hair
        other = float(max(0.0, 1.0 - accounted))

        return cls(
            mesh_area=mesh_area,
            face_in_mesh=face,
            background_in_mesh=bg,
            glasses_in_mesh=glasses,
            hat_in_mesh=hat,
            cloth_in_mesh=cloth,
            hair_in_mesh=hair,
            other_in_mesh=other,
        )
