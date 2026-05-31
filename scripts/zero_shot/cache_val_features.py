"""Cache SegFormer features on the full val split for fast ablation studies.

Why: each TF-compliant heuristic variant we want to try only changes how we
combine the same 11 features into a single prediction. Re-running SegFormer
15k times per variant is wasteful; we cache the features once and reuse them.

Output: eval/cache/val_segformer_features.csv
  columns: filename, gender, target (= FaceOcclusion), and the 11 feature columns
  produced by src/zero_shot/features.extract_features.

Run from repo root:
    python scripts/zero_shot/cache_val_features.py

To also cache the horizontally-flipped variant (needed for TTA):
    python scripts/zero_shot/cache_val_features.py --flip
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
from src.zero_shot.features import features_dataframe  # noqa: E402
from src.zero_shot.parser import FaceParser  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--out", default=str(REPO_ROOT / "eval" / "cache" / "val_segformer_features.csv"))
    p.add_argument("--source", choices=["val", "test"], default="val",
                   help="Which subset to cache features for. "
                        "'val' = stratified val split of train.csv (has GT labels). "
                        "'test' = test_students.csv (no GT labels — for submission inference).")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--flip", action="store_true",
                   help="Also flip each image horizontally before parsing (for TTA cache)")
    p.add_argument("--limit", type=int, default=0, help="0 = full")
    return p.parse_args()


def parse_val(parser: FaceParser, image_dir: Path, df: pd.DataFrame,
              batch_size: int, flip: bool) -> pd.DataFrame:
    rows = []
    fns = df["filename"].tolist()
    for i in tqdm(range(0, len(fns), batch_size),
                  desc="flip" if flip else "parse",
                  total=(len(fns) + batch_size - 1) // batch_size):
        batch_fns = fns[i:i + batch_size]
        imgs = []
        for fn in batch_fns:
            img_pil = Image.open(image_dir / fn).convert("RGB")
            if flip:
                arr = np.asarray(img_pil)
                arr = cv2.flip(arr, 1)
                img_pil = Image.fromarray(arr)
            imgs.append(img_pil)
        seg_maps = parser.parse_batch(imgs)
        rows.append(features_dataframe(seg_maps, batch_fns))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
        has_labels = True
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
        has_labels = False
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"caching SegFormer features on {len(df)} {args.source} rows (flip={args.flip})")

    parser = FaceParser()
    feats = parse_val(parser, Path(args.image_dir), df,
                      batch_size=args.batch_size, flip=args.flip)
    if has_labels:
        feats["target"] = df["FaceOcclusion"].values
        feats["gender"] = df["gender"].values

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    print(f"wrote {out} ({len(feats)} rows, {len(feats.columns)} cols)")


if __name__ == "__main__":
    main()
