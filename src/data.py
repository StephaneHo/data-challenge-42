"""Dataset, stratified split, and image transforms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.metric import GENDER_FEMALE, GENDER_MALE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Coarse bins matching the EDA — used for stratification and for diagnostic per-bin scores.
OCC_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 42,
    occ_bins: list[float] = OCC_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify by (gender, occlusion bin) so val mirrors the joint distribution."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True).copy()
    df["_bin"] = pd.cut(df["FaceOcclusion"], bins=occ_bins, include_lowest=True, right=False)
    val_idx: list[int] = []
    for _, group in df.groupby(["gender", "_bin"], observed=True):
        n_val = max(1, int(round(len(group) * val_frac)))
        n_val = min(n_val, len(group))
        picked = rng.choice(group.index.values, size=n_val, replace=False)
        val_idx.extend(picked.tolist())
    val_mask = df.index.isin(val_idx)
    train_df = df.loc[~val_mask].drop(columns=["_bin"]).reset_index(drop=True)
    val_df = df.loc[val_mask].drop(columns=["_bin"]).reset_index(drop=True)
    return train_df, val_df


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentations that do NOT alter the occluded fraction of the face.

    Avoid: RandomResizedCrop (changes face area), RandomErasing (adds fake occlusion → wrong label).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomRotation(degrees=10, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FaceOcclusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str | Path, transform=None, training: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["filename"]
        img = Image.open(self.image_dir / filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.training:
            y = np.float32(row["FaceOcclusion"])
            gender = np.float32(row["gender"])
            return img, y, gender, filename
        return img, filename


def build_balanced_sampler(df: pd.DataFrame, occ_bins: list[float] = OCC_BINS) -> torch.utils.data.WeightedRandomSampler:
    """Sampler that equalizes (gender × occlusion-bin) buckets to break the gender↔occlusion correlation.

    Each of the 2×B buckets gets equal expected mass; within a bucket samples are uniform.
    """
    df = df.reset_index(drop=True)
    bins = pd.cut(df["FaceOcclusion"], bins=occ_bins, include_lowest=True, right=False)
    key = list(zip(df["gender"], bins.astype(str)))
    counts: dict = {}
    for k in key:
        counts[k] = counts.get(k, 0) + 1
    n_buckets = len(counts)
    weights = np.array([1.0 / (counts[k] * n_buckets) for k in key], dtype=np.float64)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(df),
        replacement=True,
    )
