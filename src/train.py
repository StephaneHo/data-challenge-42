"""Training loop, evaluation, and inference utilities."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metric import WeightedMSELoss, score


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: WeightedMSELoss,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    losses: list[float] = []
    pbar = tqdm(loader, desc="train", leave=False)
    for step, (X, y, gender, _filename) in enumerate(pbar):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)
        pred = model(X)
        loss = loss_fn(pred, y, gender)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{np.mean(losses[-log_every:]):.5f}")
    return float(np.mean(losses))


def _forward_with_tta(model: nn.Module, X: torch.Tensor, tta: str) -> torch.Tensor:
    """TTA-aware forward pass.

    tta = "none" : single forward
    tta = "flip" : average of forward(X) and forward(horizontal_flip(X))
    """
    pred = model(X)
    if tta == "flip":
        pred_flip = model(torch.flip(X, dims=[3]))
        pred = (pred + pred_flip) / 2.0
    elif tta != "none":
        raise ValueError(f"Unknown tta mode: {tta}")
    return pred


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, tta: str = "none") -> pd.DataFrame:
    model.eval()
    rows = []
    for X, y, gender, filename in tqdm(loader, desc="val", leave=False):
        X = X.to(device, non_blocking=True)
        pred = _forward_with_tta(model, X, tta).cpu().numpy()
        y_np = y.numpy()
        g_np = gender.numpy()
        for i in range(len(pred)):
            rows.append({
                "filename": filename[i],
                "pred": float(pred[i]),
                "target": float(y_np[i]),
                "gender": float(g_np[i]),
            })
    return pd.DataFrame(rows)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta: str = "none") -> pd.DataFrame:
    model.eval()
    rows = []
    for X, filename in tqdm(loader, desc="test", leave=False):
        X = X.to(device, non_blocking=True)
        pred = _forward_with_tta(model, X, tta).cpu().numpy()
        for i in range(len(pred)):
            rows.append({"filename": filename[i], "FaceOcclusion": float(pred[i])})
    return pd.DataFrame(rows)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    loss_fn: WeightedMSELoss | None = None,
    weight_decay: float = 1e-4,
    checkpoint_path: str | Path | None = None,
    use_scheduler: bool = True,
) -> dict:
    if loss_fn is None:
        loss_fn = WeightedMSELoss(balance_gender=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        total_steps = max(1, epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1,
        )

    history = {"train_loss": [], "val_score": [], "val_err_f": [], "val_err_m": [], "val_gap": []}
    best_score = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_df = evaluate(model, val_loader, device)
        s = score(val_df)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:2d}/{epochs}  loss={train_loss:.5f}  "
            f"score={s['score']:.5f}  err_f={s['err_female']:.5f}  "
            f"err_m={s['err_male']:.5f}  gap={s['gap']:.5f}  ({elapsed:.0f}s)"
        )
        history["train_loss"].append(train_loss)
        history["val_score"].append(s["score"])
        history["val_err_f"].append(s["err_female"])
        history["val_err_m"].append(s["err_male"])
        history["val_gap"].append(s["gap"])
        if s["score"] < best_score:
            best_score = s["score"]
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "score": best_score, "epoch": epoch}, checkpoint_path)
                print(f"  -> new best, saved to {checkpoint_path}")
    history["best_score"] = best_score
    return history
