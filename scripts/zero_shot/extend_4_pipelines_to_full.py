"""After the 2x2 diagnostic identifies the winning combination, extend it to
the full val (15k) and test (30k) sets.

This script re-runs the 4 pipelines on every val/test image. Cost: ~13h val,
~26h test on CPU. Run only if the diagnostic justifies it.

Reuses the inference function from cross_test_4_pipelines.py.
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
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

from src.data import stratified_split  # noqa: E402
from scripts.zero_shot.cross_test_4_pipelines import (  # noqa: E402
    setup_models, compute_4_pipelines,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="If out exists, append to it (skip already-processed)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=0.15, seed=42)
        df = df.rename(columns={"FaceOcclusion": "target"})
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
        df["target"] = float("nan")
        df["gender"] = float("nan")

    if args.limit > 0:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    existing = []
    if args.resume and out_path.exists():
        existing_df = pd.read_csv(out_path)
        seen = set(existing_df.filename)
        existing = existing_df.to_dict("records")
        print(f"resume: {len(seen)} already done")
        df = df[~df.filename.isin(seen)]

    print(f"will process {len(df)} {args.source} images")
    app, tddfa, net_bi, sf_model, sf_proc, to_tensor = setup_models()

    rows = list(existing)
    image_dir = Path(args.image_dir)
    t0 = time.time()
    n_err = 0
    save_every = 500

    for i, (fn, target, gender) in enumerate(
        tqdm(
            list(zip(df["filename"], df["target"], df["gender"])),
            desc="4-pipelines-full",
        )
    ):
        img = cv2.imread(str(image_dir / fn))
        if img is None:
            n_err += 1
            continue
        try:
            d = compute_4_pipelines(img, app, tddfa, net_bi, sf_model, sf_proc, to_tensor)
        except Exception as e:
            n_err += 1
            print(f"\nfailed {fn}: {e}")
            continue
        d["filename"] = fn
        d["target"] = float(target) if not pd.isna(target) else float("nan")
        d["gender"] = float(gender) if not pd.isna(gender) else float("nan")
        rows.append(d)

        # Save checkpoint every 500 images so we can resume
        if (i + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"\n  [{i+1}/{len(df)}] saved checkpoint, {rate:.2f} img/s ETA {eta/3600:.1f}h errs={n_err}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nfinal write {out_path} ({len(rows)} rows, errs={n_err})")


if __name__ == "__main__":
    main()
