"""SegFormer face parsing wrapper.

Uses HuggingFace `jonathandinu/face-parsing` — a SegFormer trained on CelebAMask-HQ.
Outputs a per-pixel class label in [0, 18], following CelebAMask-HQ taxonomy:

  0  background       10 mouth
  1  skin             11 u_lip
  2  nose             12 l_lip
  3  eye_g (glasses)  13 hair
  4  l_eye            14 hat
  5  r_eye            15 ear_r (earring)
  6  l_brow           16 neck_l (necklace)
  7  r_brow           17 neck
  8  l_ear            18 cloth (clothes / masks)
  9  r_ear

Class groupings used by features.py:
  FACE_PARTS  = skin, nose, eyes, brows, mouth, lips, ears  (the unoccluded face)
  OCCLUDERS   = glasses, hat, cloth                          (typical occluders)
  HAIR        = hair                                         (sometimes occludes, sometimes frames)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


# Class IDs of the CelebAMask-HQ taxonomy (same order the model emits).
CLASS_BACKGROUND = 0
CLASS_SKIN = 1
CLASS_NOSE = 2
CLASS_EYE_G = 3
CLASS_L_EYE = 4
CLASS_R_EYE = 5
CLASS_L_BROW = 6
CLASS_R_BROW = 7
CLASS_L_EAR = 8
CLASS_R_EAR = 9
CLASS_MOUTH = 10
CLASS_U_LIP = 11
CLASS_L_LIP = 12
CLASS_HAIR = 13
CLASS_HAT = 14
CLASS_EAR_R = 15
CLASS_NECK_L = 16
CLASS_NECK = 17
CLASS_CLOTH = 18

NUM_CLASSES = 19

FACE_PART_CLASSES = (CLASS_SKIN, CLASS_NOSE, CLASS_L_EYE, CLASS_R_EYE,
                     CLASS_L_BROW, CLASS_R_BROW, CLASS_MOUTH, CLASS_U_LIP, CLASS_L_LIP,
                     CLASS_L_EAR, CLASS_R_EAR)
OCCLUDER_CLASSES = (CLASS_EYE_G, CLASS_HAT, CLASS_CLOTH)
HAIR_CLASSES = (CLASS_HAIR,)

DEFAULT_MODEL = "jonathandinu/face-parsing"


class FaceParser:
    """Wraps a SegFormer face-parsing model. Designed for batched CPU inference on 224x224 crops.

    The default processor resizes inputs to 512x512, which is ~5x slower than needed for our
    already-cropped 224x224 inputs. We override do_resize=False to process at native resolution.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | torch.device | None = None,
                 do_resize: bool = False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.processor = SegformerImageProcessor.from_pretrained(model_name, do_resize=do_resize)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def parse_batch(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Run face parsing on a list of PIL images.

        Returns
        -------
        np.ndarray of shape (N, H, W) with int class labels in [0, 18].
        H, W match the input image dimensions (the model output is upsampled if needed).
        """
        images = list(images)
        sizes = [img.size[::-1] for img in images]  # (H, W) for each image
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (N, num_classes, H_logits, W_logits) — usually 1/4 resolution

        # Upsample to original size per-image (sizes may differ in general, equal here)
        labels = []
        for i, (h, w) in enumerate(sizes):
            up = torch.nn.functional.interpolate(
                logits[i:i + 1], size=(h, w), mode="bilinear", align_corners=False
            )
            labels.append(up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8))
        return np.stack(labels, axis=0)

    def parse_one(self, image: Image.Image) -> np.ndarray:
        return self.parse_batch([image])[0]


def parse_paths(parser: FaceParser, paths: list[Path], batch_size: int = 8) -> np.ndarray:
    """Convenience: load images from disk and parse them in batches."""
    all_labels = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        labels = parser.parse_batch(batch_imgs)
        all_labels.append(labels)
    return np.concatenate(all_labels, axis=0)
