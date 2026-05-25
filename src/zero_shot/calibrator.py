"""Calibration: map raw segmentation features to IDEMIA's occlusion definition.

We fit a small regressor on (raw features, ground-truth occlusion) pairs from
the training set. This corrects systematic biases between what the face-parsing
model considers an occluder and what IDEMIA labelers measured.

Two regressors are supported:
  - "linear"   : multi-feature linear regression (ridge with small L2)
  - "isotonic" : isotonic regression on a single chosen feature (monotone, no extrapolation)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.zero_shot.features import FEATURE_NAMES


class OcclusionCalibrator:
    """Wraps a regressor mapping segmentation features → predicted occlusion ratio."""

    def __init__(self, mode: str = "linear", feature_for_isotonic: str = "ratio_hull",
                 ridge_alpha: float = 1.0):
        if mode not in ("linear", "isotonic"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.feature_for_isotonic = feature_for_isotonic
        self.ridge_alpha = ridge_alpha
        self.model = None
        self.feature_names_: list[str] | None = None
        self.train_r2_: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OcclusionCalibrator":
        """X holds at least the columns in FEATURE_NAMES; y is the GT occlusion ratio."""
        y = np.asarray(y, dtype=np.float64)
        if self.mode == "linear":
            self.feature_names_ = list(FEATURE_NAMES)
            features = X[self.feature_names_].to_numpy(dtype=np.float64)
            self.model = Ridge(alpha=self.ridge_alpha)
            self.model.fit(features, y)
            y_hat = np.clip(self.model.predict(features), 0.0, 1.0)
        else:
            self.feature_names_ = [self.feature_for_isotonic]
            x = X[self.feature_for_isotonic].to_numpy(dtype=np.float64)
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(x, y)
            y_hat = self.model.predict(x)
        self.train_r2_ = float(r2_score(y, y_hat))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("calibrator not fit yet")
        if self.mode == "linear":
            features = X[self.feature_names_].to_numpy(dtype=np.float64)
            return np.clip(self.model.predict(features), 0.0, 1.0)
        else:
            x = X[self.feature_names_[0]].to_numpy(dtype=np.float64)
            return self.model.predict(x)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "mode": self.mode,
                "feature_for_isotonic": self.feature_for_isotonic,
                "ridge_alpha": self.ridge_alpha,
                "model": self.model,
                "feature_names": self.feature_names_,
                "train_r2": self.train_r2_,
            }, f)
        meta_path = path.with_suffix(".json")
        meta = {
            "mode": self.mode,
            "feature_names": self.feature_names_,
            "train_r2": self.train_r2_,
        }
        if self.mode == "linear" and hasattr(self.model, "coef_"):
            meta["coef"] = dict(zip(self.feature_names_, self.model.coef_.tolist()))
            meta["intercept"] = float(self.model.intercept_)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "OcclusionCalibrator":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cal = cls(mode=d["mode"], feature_for_isotonic=d["feature_for_isotonic"],
                  ridge_alpha=d["ridge_alpha"])
        cal.model = d["model"]
        cal.feature_names_ = d["feature_names"]
        cal.train_r2_ = d["train_r2"]
        return cal
