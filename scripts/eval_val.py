"""Load a checkpoint and produce val_predictions.csv on the local val split.

Run on Colab (where the GPU + full data are) or locally if you've got a checkpoint.
Output CSV columns: filename, pred, target, gender — same format estimate_scores.py expects.

Usage:
    python scripts/eval_val.py \\
        --checkpoint checkpoints/resnet50_best.pt \\
        --backbone resnet50 \\
        --out eval/val_resnet50_8ep.csv
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

from src.data import FaceOcclusionDataset, get_eval_transforms, stratified_split  # noqa: E402
from src.metric import score  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="MUST match training-time value to recover the same val split")
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match training-time value to recover the same val split")
    p.add_argument("--out", default=str(REPO_ROOT / "eval" / "val_predictions.csv"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    _, val_df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    print(f"val split size: {len(val_df)}")

    model = build_model(args.backbone, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    print(f"loaded checkpoint from epoch {state.get('epoch', '?')} "
          f"(val score at save: {state.get('score', float('nan')):.5f})")

    ds = FaceOcclusionDataset(val_df, args.image_dir, transform=get_eval_transforms(), training=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    pred_df = evaluate(model, loader, device)

    s = score(pred_df)
    print(f"\nval score (sanity): {s['score']:.5f}")
    print(f"  err_female={s['err_female']:.5f}  err_male={s['err_male']:.5f}  "
          f"gap={s['gap']:.5f}  n_f={s['n_female']}  n_m={s['n_male']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(pred_df)} rows)")


if __name__ == "__main__":
    main()
