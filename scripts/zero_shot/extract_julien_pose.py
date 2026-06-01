"""Extract yaw/pitch/roll from 3DDFA-V2 for each val image, save to CSV.

Joins with val_julien_baseline.csv to enable per-pose analysis:
  - Is the over-prediction ×3.4 worse on profile images than frontal?
  - Does the × 0.40 calibration work uniformly across poses or does it fail on profiles?

Faster than full Julien pipeline (skips BiSeNet) — about 0.6 s/image.

Usage:
    python scripts/zero_shot/extract_julien_pose.py --limit 500 --out eval/val_julien_pose_sample.csv
    python scripts/zero_shot/extract_julien_pose.py --out eval/val_julien_pose.csv
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--baseline-csv", default=str(REPO_ROOT / "eval" / "val_julien_baseline.csv"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll (in degrees) from 3x3 rotation matrix.

    Convention : yaw = around Y axis (left/right head turn), pitch = around X
    axis (up/down nod), roll = around Z axis (head tilt).
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def main() -> None:
    args = parse_args()

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"extracting pose from 3DDFA on {len(df)} val rows")

    from insightface.app import FaceAnalysis
    print("loading InsightFace...")
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

    rows = []
    image_dir = Path(args.image_dir)
    for fn in tqdm(df["filename"].tolist(), desc="pose"):
        img = cv2.imread(str(image_dir / fn))
        if img is None:
            rows.append({"filename": fn, "yaw": np.nan, "pitch": np.nan, "roll": np.nan,
                         "face_detected": 0})
            continue
        try:
            faces = app.get(img)
            if not faces:
                rows.append({"filename": fn, "yaw": np.nan, "pitch": np.nan, "roll": np.nan,
                             "face_detected": 0})
                continue
            bbox = faces[0].bbox
            param_lst, roi_box_lst = tddfa(img, [bbox])
            # First 12 numbers = 3x4 transform = R | t.
            # R is the 3x3 rotation part (in columns 0..2).
            pose = param_lst[0][:12].reshape(3, 4)
            R = pose[:, :3]
            yaw, pitch, roll = rotation_matrix_to_euler(R)
            rows.append({"filename": fn, "yaw": yaw, "pitch": pitch, "roll": roll,
                         "face_detected": 1})
        except Exception as e:
            print(f"\nfailed on {fn}: {e}")
            rows.append({"filename": fn, "yaw": np.nan, "pitch": np.nan, "roll": np.nan,
                         "face_detected": 0})

    pose_df = pd.DataFrame(rows)
    # Join with the existing val_julien_baseline if available
    baseline_path = Path(args.baseline_csv)
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        merged = baseline.merge(pose_df, on="filename", how="left")
    else:
        merged = pose_df

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(merged)} rows)")
    print(f"yaw: mean={pose_df.yaw.mean():.1f}°, std={pose_df.yaw.std():.1f}°, "
          f"|yaw|>30°: {(pose_df.yaw.abs() > 30).sum()}/{len(pose_df)}")
    print(f"pitch: mean={pose_df.pitch.mean():.1f}°, std={pose_df.pitch.std():.1f}°")
    print(f"roll: mean={pose_df.roll.mean():.1f}°, std={pose_df.roll.std():.1f}°")


if __name__ == "__main__":
    main()
