"""Cache MediaPipe Hands occlusion signal on the full val split.

For each val image we record:
  - hand_pixel_count : number of pixels labelled "hand" by MediaPipe
  - hand_in_face_ratio : fraction of the 3DDFA-like face mask covered by hands

Because we don't have 3DDFA-V2 running locally, the "face mask" is approximated
here by the same convex hull of SegFormer face pixels that the cached features
already use (i.e., the `convex_hull` mask used internally by features.py).

Output: eval/cache/val_hands.csv
  columns: filename, hand_pixel_frac (whole-image), hand_in_hull_frac

Usage:
    python scripts/zero_shot/cache_val_hands.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import stratified_split  # noqa: E402
from src.zero_shot.features import _filled_convex_hull_mask  # noqa: E402
from src.zero_shot.hands import HandOcclusionDetector  # noqa: E402
from src.zero_shot.parser import FACE_PART_CLASSES, FaceParser  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--out", default=str(REPO_ROOT / "eval" / "cache" / "val_hands.csv"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = full val")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    _, val_df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    if args.limit > 0:
        val_df = val_df.head(args.limit)
    print(f"computing hand signals for {len(val_df)} val rows")

    detector = HandOcclusionDetector()
    parser = FaceParser()  # needed to derive a face-region proxy for hand_in_face

    rows = []
    image_dir = Path(args.image_dir)
    try:
        for fn in tqdm(val_df["filename"].tolist(), desc="hands"):
            img_pil = Image.open(image_dir / fn).convert("RGB")
            arr_rgb = np.asarray(img_pil)
            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

            hand_mask = detector.hand_pixel_mask(arr_bgr)
            hand_pixels = int((hand_mask > 0).sum())
            total_pixels = arr_bgr.shape[0] * arr_bgr.shape[1]
            hand_pixel_frac = hand_pixels / total_pixels

            seg_map = parser.parse_one(img_pil)
            face_mask = np.isin(seg_map, FACE_PART_CLASSES)
            hull_mask = _filled_convex_hull_mask(face_mask)
            hull_area = int(hull_mask.sum())
            if hull_area > 0:
                hand_in_hull = int(((hand_mask > 0) & hull_mask).sum())
                hand_in_hull_frac = hand_in_hull / hull_area
            else:
                hand_in_hull_frac = 0.0

            rows.append({
                "filename": fn,
                "hand_pixel_frac": hand_pixel_frac,
                "hand_in_hull_frac": hand_in_hull_frac,
            })
    finally:
        detector.close()

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out} ({len(out_df)} rows)")
    print(f"hand_pixel_frac: mean={out_df['hand_pixel_frac'].mean():.4f}, "
          f"max={out_df['hand_pixel_frac'].max():.4f}")
    print(f"hand_in_hull_frac: mean={out_df['hand_in_hull_frac'].mean():.4f}, "
          f"max={out_df['hand_in_hull_frac'].max():.4f}")
    nonzero = (out_df['hand_in_hull_frac'] > 0).sum()
    print(f"images with hand in face area: {nonzero}/{len(out_df)} ({100*nonzero/len(out_df):.1f}%)")


if __name__ == "__main__":
    main()
