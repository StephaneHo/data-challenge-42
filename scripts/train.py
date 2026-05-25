"""Full training script. Designed to run on Colab/Kaggle GPU.

Usage:
    python scripts/train.py --backbone resnet50 --epochs 10 --batch-size 128 --lr 3e-4

Outputs:
    - checkpoints/<backbone>_best.pt     (best model by val score)
    - logs/<backbone>_history.csv        (per-epoch metrics)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    FaceOcclusionDataset,
    build_balanced_sampler,
    get_eval_transforms,
    get_train_transforms,
    stratified_split,
)
from src.metric import WeightedMSELoss  # noqa: E402
from src.model import build_model, count_params  # noqa: E402
from src.train import fit  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--out-dir", default=str(REPO_ROOT))
    p.add_argument("--backbone", default="resnet50",
                   choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=0, help="0 = use all")
    p.add_argument("--no-balanced-sampler", action="store_true")
    p.add_argument("--no-balanced-loss", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt to load weights from before training")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Disable OneCycleLR — recommended when resuming a fine-tune")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    print(f"loaded {len(train_csv)} train rows")

    train_df, val_df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    if args.max_train > 0:
        train_df = train_df.sample(n=min(args.max_train, len(train_df)), random_state=args.seed).reset_index(drop=True)
    print(f"split: train={len(train_df)}, val={len(val_df)}")

    train_ds = FaceOcclusionDataset(train_df, args.image_dir, transform=get_train_transforms())
    val_ds = FaceOcclusionDataset(val_df, args.image_dir, transform=get_eval_transforms())

    if args.no_balanced_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    else:
        sampler = build_balanced_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    print(f"backbone={args.backbone}  params={count_params(model):,}")

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"resumed from {args.resume} (saved at epoch {state.get('epoch', '?')}, "
              f"val score {state.get('score', float('nan')):.5f})")

    loss_fn = WeightedMSELoss(balance_gender=not args.no_balanced_loss)
    ckpt_path = ckpt_dir / f"{args.backbone}_best.pt"
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        loss_fn=loss_fn,
        weight_decay=args.weight_decay,
        checkpoint_path=ckpt_path,
        use_scheduler=not args.no_scheduler,
    )

    hist_path = log_dir / f"{args.backbone}_history.csv"
    pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list)}).to_csv(hist_path, index=False)
    with open(log_dir / f"{args.backbone}_run.json", "w") as f:
        json.dump({"args": vars(args), "best_score": history["best_score"]}, f, indent=2)
    print(f"\nbest val score: {history['best_score']:.5f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"history:    {hist_path}")


if __name__ == "__main__":
    main()
