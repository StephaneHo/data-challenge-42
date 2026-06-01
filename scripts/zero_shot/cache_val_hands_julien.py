"""Cache MediaPipe Hands aligned with Julien's pipeline (3DDFA mesh).

For each val image:
  - Run InsightFace + 3DDFA-V2 to get the mask_theoretical (same as Julien)
  - Run MediaPipe Hands to get the hand mask
  - Compute hand_in_julien_mesh = (hand ∩ mesh) / mesh_area

This is the hand-occlusion signal aligned with Julien's pipeline, NOT with our
SegFormer-based pipeline. Use this when integrating with Julien's predictions.

Output columns: filename, face_detected, mesh_area_frac,
                hand_pixel_frac, hand_in_julien_mesh

Usage:
    python scripts/zero_shot/cache_val_hands_julien.py \\
        --source val --out eval/cache/val_hands_julien.csv
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

from src.data import stratified_split  # noqa: E402
from src.zero_shot.hands import HandOcclusionDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_julien_face_models():
    """Load InsightFace and TDDFA (no BiSeNet — we only need the 3D mesh)."""
    print("loading InsightFace...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    print("loading 3DDFA-V2...")
    import yaml
    from TDDFA import TDDFA
    cfg = yaml.load(open(REPO_ROOT / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO_ROOT / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)
    return app, tddfa


def compute_julien_mesh(img_bgr, app, tddfa):
    """Replicates Julien's mesh_theoretical computation only.

    Returns a (H, W) uint8 mask of the projected 3DDFA convex hull, or None if face not detected.
    """
    faces = app.get(img_bgr)
    if not faces:
        return None
    bbox = faces[0].bbox
    param_lst, roi_box_lst = tddfa(img_bgr, [bbox])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    pts = ver_lst[0][:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def main():
    args = parse_args()
    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"running Julien-aligned hands signal on {len(df)} {args.source} rows")

    app, tddfa = setup_julien_face_models()
    hand_detector = HandOcclusionDetector()

    rows = []
    image_dir = Path(args.image_dir)
    try:
        for fn in tqdm(df["filename"].tolist(), desc="hands+mesh"):
            img = cv2.imread(str(image_dir / fn))
            if img is None:
                rows.append({"filename": fn, "face_detected": 0, "mesh_area_frac": 0.0,
                             "hand_pixel_frac": 0.0, "hand_in_julien_mesh": 0.0})
                continue
            try:
                mesh = compute_julien_mesh(img, app, tddfa)
                hand_mask = hand_detector.hand_pixel_mask(img)
                hand_pixels = int((hand_mask > 0).sum())
                hand_pixel_frac = hand_pixels / (img.shape[0] * img.shape[1])

                if mesh is None:
                    rows.append({"filename": fn, "face_detected": 0, "mesh_area_frac": 0.0,
                                 "hand_pixel_frac": hand_pixel_frac, "hand_in_julien_mesh": 0.0})
                    continue

                mesh_area = int(mesh.sum())
                hand_in_mesh = int(((hand_mask > 0) & (mesh > 0)).sum())
                rows.append({
                    "filename": fn,
                    "face_detected": 1,
                    "mesh_area_frac": mesh_area / (img.shape[0] * img.shape[1]),
                    "hand_pixel_frac": hand_pixel_frac,
                    "hand_in_julien_mesh": hand_in_mesh / mesh_area if mesh_area > 0 else 0.0,
                })
            except Exception as e:
                print(f"\nfailed on {fn}: {e}")
                rows.append({"filename": fn, "face_detected": 0, "mesh_area_frac": 0.0,
                             "hand_pixel_frac": 0.0, "hand_in_julien_mesh": 0.0})
    finally:
        hand_detector.close()

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(out_df)} rows)")
    print(f"  face_detected: {out_df.face_detected.sum()}/{len(out_df)}")
    print(f"  hand_pixel_frac mean: {out_df.hand_pixel_frac.mean():.4f}")
    print(f"  hand_in_julien_mesh mean: {out_df.hand_in_julien_mesh.mean():.4f}")
    nz = (out_df.hand_in_julien_mesh > 0).sum()
    print(f"  images with hand-in-mesh > 0: {nz}/{len(out_df)} ({100*nz/len(out_df):.1f}%)")


if __name__ == "__main__":
    main()
