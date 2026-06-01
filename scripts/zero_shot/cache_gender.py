"""Cache InsightFace gender prediction for test (or val) images.

InsightFace's app.get() returns faces with .gender (0=female, 1=male, matching our convention)
and .gender_confidence (when available). Output: filename, pred_gender, gender_score.

Used to enable per-gender strategies on the test set where gender isn't provided.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        from src.data import stratified_split
        _, df = stratified_split(train_csv, val_frac=0.15, seed=42)
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")

    if args.limit > 0:
        df = df.head(args.limit)
    print(f"will process {len(df)} {args.source} images")

    print("loading InsightFace buffalo_l (only detection + genderage)...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "genderage"],
    )
    app.prepare(ctx_id=0, det_size=(224, 224))

    image_dir = Path(args.image_dir)
    rows = []
    t0 = time.time()
    n_no_face = 0
    for i, fn in enumerate(tqdm(df["filename"].tolist(), desc="gender")):
        img_path = image_dir / fn
        img = cv2.imread(str(img_path))
        if img is None:
            rows.append({"filename": fn, "pred_gender": np.nan, "n_faces": 0})
            continue
        faces = app.get(img)
        if not faces:
            rows.append({"filename": fn, "pred_gender": np.nan, "n_faces": 0})
            n_no_face += 1
            continue
        face = faces[0]
        # InsightFace convention: gender 0=female, 1=male (sex attribute = 'F' or 'M')
        # face.sex is 'F' or 'M' (newer API); face.gender is 0/1 (older API)
        if hasattr(face, "sex") and face.sex is not None:
            g = 0.0 if face.sex == "F" else 1.0
        else:
            g = float(face.gender) if hasattr(face, "gender") else np.nan
        rows.append({
            "filename": fn,
            "pred_gender": g,
            "n_faces": len(faces),
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  [{i+1}/{len(df)}] {rate:.1f} img/s, ETA {eta/60:.1f} min, no-face={n_no_face}")

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\nwrote {out} ({len(out_df)} rows in {elapsed:.0f}s)")
    print(f"  no face detected: {n_no_face} ({100*n_no_face/len(out_df):.1f}%)")
    pred_gender_valid = out_df["pred_gender"].dropna()
    print(f"  gender distribution: F={int((pred_gender_valid == 0).sum())} "
          f"M={int((pred_gender_valid == 1).sum())}")


if __name__ == "__main__":
    main()
