"""Hand occlusion detection via MediaPipe Hands.

Why: BiSeNet face-parsing doesn't have a "hand" class — a hand covering the
face gets labelled as `skin` (same texture) or `background`. Neither signal
flags it as occlusion. Running MediaPipe Hands separately gives us an
explicit hand mask we can intersect with the 3DDFA theoretical face mask.

Compliance: MediaPipe Hands is a pre-trained model with public weights. No
fit on the IDEMIA dataset.

Usage:
    from src.zero_shot.hands import HandOcclusionDetector

    detector = HandOcclusionDetector()   # auto-downloads model on first run

    # Returns the fraction of theoretical face area covered by detected hand(s)
    ratio = detector.hand_in_face_ratio(img_bgr, mask_theoretical)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:
    raise ImportError(
        "mediapipe is required for hand detection. Install with: "
        "pip install mediapipe"
    ) from e


# MediaPipe-published hand landmarker model (float16, full version)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "hand_landmarker.task"


def _download_model(target: Path) -> None:
    """Download the hand_landmarker task file from MediaPipe's CDN."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MediaPipe HandLandmarker model -> {target}")
    urllib.request.urlretrieve(HAND_MODEL_URL, target)
    print(f"downloaded {target.stat().st_size / 1e6:.1f} MB")


class HandOcclusionDetector:
    """Detect hands in an image and report their overlap with a face mask."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.3,
    ):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            _download_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def hand_pixel_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return a (H, W) uint8 mask: 1 where a detected hand's convex hull lies.

        Returns an all-zero mask if no hands are detected.
        """
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        mask = np.zeros((h, w), dtype=np.uint8)
        if not result.hand_landmarks:
            return mask
        for hand_lm in result.hand_landmarks:
            pts = np.array(
                [[int(lm.x * w), int(lm.y * h)] for lm in hand_lm],
                dtype=np.int32,
            )
            if len(pts) < 3:
                continue
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 1)
        return mask

    def hand_in_face_ratio(
        self,
        img_bgr: np.ndarray,
        face_mask: np.ndarray,
    ) -> float:
        """Fraction of `face_mask` pixels overlapping with detected hand(s).

        Parameters
        ----------
        img_bgr : (H, W, 3) BGR uint8 image
        face_mask : (H, W) uint8 mask of the theoretical face region
                    (e.g., from 3DDFA convex hull). Nonzero = face.

        Returns
        -------
        float in [0, 1]. 0 means no hand in face, 1 means hand covers entire face.
        """
        face_area = int((face_mask > 0).sum())
        if face_area == 0:
            return 0.0
        hand_mask = self.hand_pixel_mask(img_bgr)
        overlap = int(((hand_mask > 0) & (face_mask > 0)).sum())
        return overlap / face_area

    def close(self) -> None:
        """Free underlying MediaPipe resources. Safe to call multiple times."""
        if hasattr(self, "detector"):
            try:
                self.detector.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
