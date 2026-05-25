"""End-to-end zero-shot occlusion prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.zero_shot.calibrator import OcclusionCalibrator
from src.zero_shot.features import features_dataframe
from src.zero_shot.parser import FaceParser


def extract_features_from_paths(
    parser: FaceParser,
    image_dir: Path,
    filenames: list[str],
    batch_size: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Run parsing + feature extraction over a list of image filenames (relative to image_dir).

    Returns a DataFrame with one row per filename and columns:
      filename, <FEATURE_NAMES>
    """
    rows = []
    iterator = range(0, len(filenames), batch_size)
    if progress:
        iterator = tqdm(iterator, desc="parse", total=(len(filenames) + batch_size - 1) // batch_size)
    for i in iterator:
        batch_fns = filenames[i:i + batch_size]
        batch_imgs = [Image.open(Path(image_dir) / fn).convert("RGB") for fn in batch_fns]
        seg = parser.parse_batch(batch_imgs)
        rows.append(features_dataframe(seg, batch_fns))
    return pd.concat(rows, ignore_index=True)


def predict(
    parser: FaceParser,
    calibrator: OcclusionCalibrator,
    image_dir: Path,
    filenames: list[str],
    batch_size: int = 8,
) -> pd.DataFrame:
    """Return DataFrame: filename, FaceOcclusion (calibrated)."""
    feats = extract_features_from_paths(parser, image_dir, filenames, batch_size=batch_size)
    feats["FaceOcclusion"] = calibrator.predict(feats)
    return feats[["filename", "FaceOcclusion"]]
