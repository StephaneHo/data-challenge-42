# Plug-in improvements for Julien's TF pipeline

This doc shows how to integrate four drop-in modules into your existing
`occlusion_computation` function. All four are **fully Training-Free
compliant** (no fit on the IDEMIA dataset, fixed-weight heuristics only).

## Setup

You already have most of what you need. Just make sure the venv is synced
and that the hand-landmarker model is cached locally (it auto-downloads on
first use):

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## What each module does

| Module | What it gives you |
|---|---|
| `src.zero_shot.tta` | Wraps your predict function to average prediction on image + horizontal flip. Reduces variance. |
| `src.zero_shot.hands` | MediaPipe Hands detector → reports fraction of theoretical face area covered by a detected hand. Catches the "hand in front of face" case which BiSeNet misses (hand = "skin"). |
| `src.zero_shot.cross_check` | Decomposes the occlusion inside the 3DDFA mesh by class (glasses, hat, cloth, hair, background). Makes the heuristic combination explicit. |
| `src.zero_shot.parser_fusion` | Combines BiSeNet output with our SegFormer (`jonathandinu/face-parsing`) output at the semantic-group level. Two parsers with decorrelated errors → more robust face/occluder masks. |
| `src.zero_shot.julien_helpers` | High-level `enhanced_occlusion(...)` that bundles the cross-check + hand bonus into one call with fixed weights. |

## Refactor needed on your side

Your current function:

```python
def occlusion_computation(app, img, display_results=False):
    # ... your code ...
    return occlusion_ratio
```

To use our cross-check + hand bonus you need access to `mask_theoretical`
and `parsing` outside the function. The cleanest refactor is to split into
two:

```python
def occlusion_intermediates(app, img):
    """Same pipeline but returns the intermediate masks too."""
    # 1. Face detection
    faces = app.get(img)
    face = faces[0]
    bbox = face.bbox

    # 2. 3DDFA
    boxes = [bbox]
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

    pts = ver_lst[0][:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask_theoretical = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_theoretical, hull, 1)

    # 3. BiSeNet parsing
    input_tensor = to_tensor(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = net(input_tensor)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)

    # 4. Base ratio
    visible_skin_mask = np.isin(parsing, VISIBLE_FACE_CLASSES).astype(np.uint8)
    visible_pixels = (visible_skin_mask & mask_theoretical).sum()
    total_pixels = mask_theoretical.sum()
    base_ratio = 1.0 - visible_pixels / max(total_pixels, 1)

    return base_ratio, mask_theoretical, parsing


def occlusion_computation(app, img, display_results=False):
    base_ratio, _, _ = occlusion_intermediates(app, img)
    return base_ratio
```

## Integration recipe

```python
import sys, cv2, numpy as np
from pathlib import Path

# Make sure the repo root is on sys.path so `src` resolves
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adapt depth

from src.zero_shot.tta import tta_flip
from src.zero_shot.hands import HandOcclusionDetector
from src.zero_shot.julien_helpers import enhanced_occlusion

# Initialize the hand detector once. It auto-downloads its model (~8 MB)
# to src/zero_shot/weights/hand_landmarker.task on first run.
hand_detector = HandOcclusionDetector()


def occlusion_v1(app, img):
    """Base ratio + cross-check decomposition + hand-detection bonus."""
    base_ratio, mask_theoretical, parsing = occlusion_intermediates(app, img)
    return enhanced_occlusion(
        base_ratio=base_ratio,
        mask_theoretical=mask_theoretical,
        parsing=parsing,
        img_bgr=img,
        hand_detector=hand_detector,
    )


# Add TTA on top (~2x inference cost)
occlusion_v2 = tta_flip(occlusion_v1)
```

In your val/test loops, swap `occlusion_computation(app, img)` for
`occlusion_v2(app, img)`. Everything else (CSV writing, score reporting)
stays the same.

## Optional: dual parser fusion

This doubles inference time. Worth trying once the rest of the pipeline is
stable.

```python
from src.zero_shot.parser import FaceParser
from src.zero_shot.parser_fusion import fuse_to_semantic, semantic_face_mask, SEM_FACE
from PIL import Image

segformer = FaceParser()  # auto-downloads jonathandinu/face-parsing

def occlusion_intermediates_fused(app, img_bgr):
    # ... face detection + 3DDFA as before, giving mask_theoretical ...

    # BiSeNet parsing (Julien's existing call):
    input_tensor = to_tensor(img_bgr).unsqueeze(0).to(device)
    with torch.inference_mode():
        parsing_bisenet = net(input_tensor)[0].squeeze(0).cpu().numpy().argmax(0)

    # SegFormer parsing on the same image:
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    parsing_segformer = segformer.parse_one(img_pil)

    # Fuse to semantic groups (FACE / GLASSES / HAT / CLOTH / HAIR / BG / OTHER)
    semantic = fuse_to_semantic(parsing_bisenet, "bisenet",
                                 parsing_segformer, "segformer",
                                 strategy="vote_then_face_priority")
    face_mask = semantic_face_mask(semantic)

    # New base ratio uses the fused face mask:
    base_ratio = 1.0 - ((face_mask & mask_theoretical).sum() / max(mask_theoretical.sum(), 1))

    # Note: the cross_check module expects a single-parser label map. For the
    # fused case, you can pass either `parsing_bisenet` (its taxonomy is the
    # default) or `semantic` (semantic-group taxonomy) but you'll need to adapt
    # the class IDs in OcclusionDecomposition.from_masks.

    return base_ratio, mask_theoretical, parsing_bisenet  # keep bisenet for cross-check
```

## Evaluation protocol

For each new variant Julien builds, follow the same protocol we use for
ours so all numbers go in the same `reports/` folder and `EXPERIMENTS.md`:

```bash
# 1. Run the pipeline on val, save CSV with columns filename, pred, target, gender
# 2. Pass it through our harness:
python scripts/eval_harness.py \
    --predictions eval/val_julien_v1.csv \
    --variant julien_tf_v1 \
    --notes "3DDFA + BiSeNet + cross-check + hand + TTA flip" \
    --design-samples "5 train rows manually inspected to set hand_bonus=0.5"

# 3. Compare to existing entries:
python scripts/compare_reports.py
```

## Fixed weights — disclosure

The blending coefficients in `julien_helpers.HEURISTIC_WEIGHTS`:

```python
HEURISTIC_WEIGHTS = {
    "hard_occluders_bonus": 0.30,   # rescale of (glasses + hat + cloth + bg) fraction
    "hand_bonus":           0.50,   # coefficient on detected-hand overlap
    "cap":                  0.95,
    "floor":                0.0,
}
```

These are chosen from manual inspection of 5 train samples. Document the
filenames in your `--design-samples` flag when running the harness so
reviewers can verify the TF compliance of this small inspection subset.
