"""Turn a raw dataset into a compact set of ML-relevant meta-features."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    is_classification_target,
    split_feature_types,
    safe_skew,
    outlier_fraction,
)


@dataclass
class DatasetProfile:
    """A structured, JSON-serialisable summary of a dataset."""

    # Shape
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int

    # Task
    task: str  # "classification" | "regression"
    n_classes: int | None
    imbalance_ratio: float | None  # majority / minority (>=1); None for regression
    minority_fraction: float | None

    # Data quality
    missing_fraction: float
    duplicate_row_fraction: float
    constant_feature_fraction: float

    # Feature structure
    dim_ratio: float  # n_features / n_samples
    max_correlation: float  # strongest abs pairwise corr among numeric features
    mean_abs_skew: float
    mean_outlier_fraction: float
    max_categorical_cardinality: int
    sparsity: float  # fraction of numeric entries equal to zero

    # Signal
    target_signal: float  # 0..1 rough strength of feature->target relationship

    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _target_signal(X_num: pd.DataFrame, y: pd.Series, task: str) -> float:
    """Rough 0..1 signal strength via normalised mutual information."""
    if X_num.shape[1] == 0 or len(X_num) < 10:
        return 0.0
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        Xf = X_num.fillna(X_num.median(numeric_only=True)).to_numpy()
        # Subsample for speed on large data.
        if len(Xf) > 5000:
            idx = np.random.RandomState(0).choice(len(Xf), 5000, replace=False)
            Xf, yy = Xf[idx], y.iloc[idx]
        else:
            yy = y
        if task == "classification":
            mi = mutual_info_classif(Xf, yy, random_state=0)
        else:
            mi = mutual_info_regression(Xf, yy, random_state=0)
        # Squash: MI is unbounded, map through a saturating function.
        return float(np.clip(np.tanh(np.max(mi)), 0.0, 1.0))
    except Exception:
        return 0.0


def profile(X: pd.DataFrame, y: pd.Series, task: str | None = None) -> DatasetProfile:
    """Compute a :class:`DatasetProfile` from features ``X`` and target ``y``."""
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)

    if task is None:
        task = "classification" if is_classification_target(y) else "regression"

    n_samples, n_features = X.shape
    numeric_cols, categorical_cols = split_feature_types(X)
    X_num = X[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=X.index)

    # ---- task-specific target stats ----
    n_classes = imbalance_ratio = minority_fraction = None
    if task == "classification":
        counts = y.value_counts()
        n_classes = int(counts.size)
        if counts.size:
            minority_fraction = float(counts.min() / counts.sum())
            imbalance_ratio = float(counts.max() / max(counts.min(), 1))

    # ---- data quality ----
    missing_fraction = float(X.isna().mean().mean()) if n_features else 0.0
    duplicate_row_fraction = float(X.duplicated().mean()) if n_samples else 0.0
    constant_feature_fraction = (
        float(np.mean([X[c].nunique(dropna=True) <= 1 for c in X.columns]))
        if n_features
        else 0.0
    )

    # ---- feature structure ----
    dim_ratio = n_features / max(n_samples, 1)

    max_correlation = 0.0
    if X_num.shape[1] >= 2:
        corr = np.array(X_num.corr().abs().to_numpy(), dtype=float, copy=True)
        np.fill_diagonal(corr, np.nan)
        if np.isfinite(corr).any():
            max_correlation = float(np.nanmax(corr))

    if X_num.shape[1]:
        mean_abs_skew = float(np.mean([abs(safe_skew(X_num[c].to_numpy())) for c in X_num.columns]))
        mean_outlier_fraction = float(np.mean([outlier_fraction(X_num[c].to_numpy()) for c in X_num.columns]))
        num_vals = X_num.to_numpy(dtype=float)
        finite = num_vals[np.isfinite(num_vals)]
        sparsity = float(np.mean(finite == 0)) if finite.size else 0.0
    else:
        mean_abs_skew = mean_outlier_fraction = sparsity = 0.0

    max_categorical_cardinality = (
        int(max((X[c].nunique(dropna=True) for c in categorical_cols), default=0))
    )

    target_signal = _target_signal(X_num, y, task)

    return DatasetProfile(
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        task=task,
        n_classes=n_classes,
        imbalance_ratio=imbalance_ratio,
        minority_fraction=minority_fraction,
        missing_fraction=missing_fraction,
        duplicate_row_fraction=duplicate_row_fraction,
        constant_feature_fraction=constant_feature_fraction,
        dim_ratio=dim_ratio,
        max_correlation=max_correlation,
        mean_abs_skew=mean_abs_skew,
        mean_outlier_fraction=mean_outlier_fraction,
        max_categorical_cardinality=max_categorical_cardinality,
        sparsity=sparsity,
        target_signal=target_signal,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
