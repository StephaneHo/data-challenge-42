"""Run Julien's pipeline twice (original + horizontal flip) on a sample.

Output adds a 'pred_flipped' column to enable TTA averaging analysis.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

from src.data import stratified_split  # noqa: E402
from scripts.zero_shot.run_julien_pipeline import compute_occlusion, setup_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    df = df.head(args.limit)
    print(f"running Julien's pipeline (original + flipped) on {len(df)} samples")

    app, tddfa, net, device, to_tensor = setup_models()

    rows = []
    image_dir = Path(args.image_dir)
    for fn in tqdm(df["filename"].tolist(), desc="orig+flip"):
        img = cv2.imread(str(image_dir / fn))
        if img is None:
            rows.append((fn, np.nan, np.nan))
            continue
        try:
            p_orig = compute_occlusion(img, app, tddfa, net, device, to_tensor)
            p_flip = compute_occlusion(cv2.flip(img, 1), app, tddfa, net, device, to_tensor)
        except Exception as e:
            print(f"\nfailed on {fn}: {e}")
            p_orig = p_flip = np.nan
        rows.append((fn, p_orig, p_flip))

    pred_df = pd.DataFrame(rows, columns=["filename", "pred", "pred_flipped"])
    merged = df[["filename", "FaceOcclusion", "gender"]].merge(pred_df, on="filename")
    merged = merged.rename(columns={"FaceOcclusion": "target"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
