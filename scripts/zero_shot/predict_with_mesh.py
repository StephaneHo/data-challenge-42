"""Heuristics that combine SegFormer features with MediaPipe Face Mesh signals.

Reads:
    - eval/cache/val_segformer_features.csv (and optionally _flip.csv for TTA)
    - eval/cache/val_face_mesh.csv

Applies named heuristics that use Face Mesh signals as a modifier on the base
SegFormer-based prediction. Writes predictions in the harness format.

Heuristics defined here (all TF-compliant — fixed weights from inspection):
  - mesh_no_face_floor    : pred = simple_hull_scaled_power07 + bonus if face NOT detected
  - mesh_tiny_face_floor  : pred = simple_hull_scaled_power07 + bonus if face_mesh_area_frac < 0.15
  - mesh_combined         : both bonuses, capped

Usage:
    python scripts/zero_shot/predict_with_mesh.py \\
        --features eval/cache/val_segformer_features.csv \\
        --mesh eval/cache/val_face_mesh.csv \\
        --features-flipped eval/cache/val_segformer_features_flip.csv \\
        --heuristic mesh_combined \\
        --out eval/val_zs_mesh_combined_tta.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metric import score  # noqa: E402
from src.zero_shot.heuristics import simple_hull_scaled_power07  # noqa: E402


def _base_with_tta(feats: pd.DataFrame, feats_flipped: pd.DataFrame | None) -> np.ndarray:
    """Apply the current-best heuristic (simple_hull_scaled_power07), with TTA if provided."""
    base = simple_hull_scaled_power07(feats)
    if feats_flipped is not None:
        feats_flipped = feats_flipped.set_index("filename").reindex(feats["filename"]).reset_index()
        base_flipped = simple_hull_scaled_power07(feats_flipped)
        base = 0.5 * (base + base_flipped)
    return base


def mesh_no_face_floor(base: np.ndarray, mesh_df: pd.DataFrame) -> np.ndarray:
    """If MediaPipe didn't detect a face → add bonus +0.1 (capped at 1).

    Reasoning: face detector failures correlate with heavy occlusion or extreme pose.
    Fixed bonus of 0.1 inspected from 5 samples where MediaPipe failed.
    """
    no_face = (mesh_df["face_detected"].to_numpy() == 0)
    out = base.copy()
    out[no_face] = np.minimum(1.0, base[no_face] + 0.1)
    return out


def mesh_tiny_face_floor(base: np.ndarray, mesh_df: pd.DataFrame) -> np.ndarray:
    """If face mesh area is very small (< 0.15 of image) → add bonus +0.05.

    Reasoning: small detected face often means extreme pose where mesh
    underestimates the actual visible face, or heavy occlusion compressed the
    detection. Fixed 0.05 bonus from inspection.
    """
    tiny = mesh_df["face_mesh_area_frac"].to_numpy() < 0.15
    out = base.copy()
    out[tiny] = np.minimum(1.0, base[tiny] + 0.05)
    return out


def mesh_combined(base: np.ndarray, mesh_df: pd.DataFrame) -> np.ndarray:
    """Both bonuses combined (no-face bonus dominates if both apply, no double-bonus)."""
    no_face = (mesh_df["face_detected"].to_numpy() == 0)
    tiny = mesh_df["face_mesh_area_frac"].to_numpy() < 0.15
    bonus = np.where(no_face, 0.1, np.where(tiny, 0.05, 0.0))
    return np.minimum(1.0, base + bonus)


HEURISTICS = {
    "mesh_no_face_floor":   mesh_no_face_floor,
    "mesh_tiny_face_floor": mesh_tiny_face_floor,
    "mesh_combined":        mesh_combined,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--features-flipped", default=None)
    p.add_argument("--mesh", required=True, help="Face Mesh cache CSV")
    p.add_argument("--heuristic", required=True, choices=list(HEURISTICS))
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feats = pd.read_csv(args.features)
    mesh = pd.read_csv(args.mesh).set_index("filename").reindex(feats["filename"]).reset_index()
    flipped = pd.read_csv(args.features_flipped) if args.features_flipped else None
    print(f"loaded features: {len(feats)}, mesh: {len(mesh)} (missing mesh: {mesh['face_detected'].isna().sum()})")

    base = _base_with_tta(feats, flipped)
    print(f"base pred (power07{'+tta' if flipped is not None else ''}): "
          f"mean={base.mean():.4f}, std={base.std():.4f}")

    fn = HEURISTICS[args.heuristic]
    pred = fn(base, mesh)
    print(f"after {args.heuristic}: mean={pred.mean():.4f}, std={pred.std():.4f}")

    out_df = pd.DataFrame({
        "filename": feats["filename"],
        "pred": pred,
        "target": feats["target"],
        "gender": feats["gender"],
    })
    s = score(out_df)
    print(f"val score: {s['score']:.5f}  err_F: {s['err_female']:.5f}  err_M: {s['err_male']:.5f}  gap: {s['gap']:.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
