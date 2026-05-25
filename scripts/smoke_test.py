"""End-to-end smoke test on a small subset — validates the whole pipeline on CPU.

Run from repo root:
    python scripts/smoke_test.py

What it does:
  - loads train.csv
  - takes a small stratified subset (default 5000 train + 1000 val)
  - trains mobilenet_v3_small for 1 epoch on CPU with the official metric
  - reports val score and saves a tiny test_predictions.csv

Goal is NOT to get a competitive score — only to prove the pipeline works.
Real training belongs on Colab/Kaggle.
"""
from __future__ import annotations

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
from src.metric import WeightedMSELoss, score  # noqa: E402
from src.model import build_model, count_params  # noqa: E402
from src.train import evaluate, fit, predict  # noqa: E402

DATA_DIR = REPO_ROOT / "occlusion_datasets"
IMG_DIR = REPO_ROOT / "crops"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

SUBSET_TRAIN = 5000
SUBSET_VAL = 1000
SUBSET_TEST = 500
EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-3


def main() -> None:
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    test_csv = pd.read_csv(DATA_DIR / "test_students.csv")
    print(f"loaded {len(train_csv)} train rows, {len(test_csv)} test rows")

    train_full, val_full = stratified_split(train_csv, val_frac=0.15, seed=42)
    print(f"stratified split: train={len(train_full)}, val={len(val_full)}")

    train_df = train_full.sample(n=min(SUBSET_TRAIN, len(train_full)), random_state=0).reset_index(drop=True)
    val_df = val_full.sample(n=min(SUBSET_VAL, len(val_full)), random_state=0).reset_index(drop=True)
    test_df = test_csv.sample(n=min(SUBSET_TEST, len(test_csv)), random_state=0).reset_index(drop=True)
    print(f"smoke subsets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  train gender mix: {train_df['gender'].value_counts().to_dict()}")
    print(f"  val   gender mix: {val_df['gender'].value_counts().to_dict()}")

    train_ds = FaceOcclusionDataset(train_df, IMG_DIR, transform=get_train_transforms())
    val_ds = FaceOcclusionDataset(val_df, IMG_DIR, transform=get_eval_transforms())
    test_ds = FaceOcclusionDataset(test_df, IMG_DIR, transform=get_eval_transforms(), training=False)

    sampler = build_balanced_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model("mobilenet_v3_small", pretrained=True).to(device)
    print(f"model params: {count_params(model):,}")

    print("\n--- baseline (untrained head) ---")
    baseline_df = evaluate(model, val_loader, device)
    print(score(baseline_df))

    print("\n--- training ---")
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        loss_fn=WeightedMSELoss(balance_gender=True),
        checkpoint_path=CHECKPOINT_DIR / "smoke_best.pt",
        use_scheduler=True,
    )
    print(f"history: {history}")

    print("\n--- predicting on test subset ---")
    pred_df = predict(model, test_loader, device)
    pred_df["gender"] = "x"
    out = REPO_ROOT / "test_predictions_smoke.csv"
    pred_df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(pred_df)} rows)")
    print(pred_df.head())


if __name__ == "__main__":
    main()
