"""Load a checkpoint and generate test_predictions.csv for submission.

Usage:
    python scripts/infer.py --checkpoint checkpoints/resnet50_best.pt --backbone resnet50

Output: test_predictions.csv at repo root with columns: filename, FaceOcclusion, gender
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FaceOcclusionDataset, get_eval_transforms  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", default=str(REPO_ROOT / "test_predictions.csv"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tta", default="none", choices=["none", "flip"],
                   help="Test-time augmentation: 'flip' averages prediction on image and its horizontal flip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    test_csv = pd.read_csv(Path(args.data_dir) / "test_students.csv")
    print(f"test rows: {len(test_csv)}")

    model = build_model(args.backbone, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    print(f"loaded checkpoint from epoch {state.get('epoch', '?')} (val score {state.get('score', '?'):.5f})")

    ds = FaceOcclusionDataset(test_csv, args.image_dir, transform=get_eval_transforms(), training=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    pred_df = predict(model, loader, device, tta=args.tta)
    if args.tta != "none":
        print(f"used TTA mode: {args.tta}")
    pred_df["gender"] = "x"
    pred_df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(pred_df)} rows)")
    print(pred_df.head())


if __name__ == "__main__":
    main()
