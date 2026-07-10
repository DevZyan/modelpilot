"""Internal helpers. Not part of the public API."""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_classification_target(y: pd.Series, threshold: int = 20) -> bool:
    """Heuristically decide whether a target is a classification task.

    Rules of thumb:
      * object / category / bool dtypes  -> classification
      * numeric with few unique integer-like values -> classification
      * everything else -> regression
    """
    y = pd.Series(y)
    if y.dtype == bool or str(y.dtype) in ("object", "category"):
        return True

    n_unique = y.nunique(dropna=True)
    if n_unique <= 2:
        return True

    if pd.api.types.is_float_dtype(y):
        # Floats that are really whole numbers with few levels look categorical.
        non_null = y.dropna()
        all_integral = np.all(np.equal(np.mod(non_null.to_numpy(), 1), 0))
        return bool(all_integral and n_unique <= threshold)

    if pd.api.types.is_integer_dtype(y):
        return n_unique <= threshold

    return False


def split_feature_types(X: pd.DataFrame, max_numeric_cardinality: int = 15):
    """Return (numeric_cols, categorical_cols).

    Low-cardinality integer columns are treated as categorical because that is
    almost always what they encode (e.g. an ordinal 1-5 rating).
    """
    numeric, categorical = [], []
    n_rows = max(len(X), 1)
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nunique = s.nunique(dropna=True)
            # Small distinct integer set relative to rows -> categorical.
            if (
                pd.api.types.is_integer_dtype(s)
                and nunique <= max_numeric_cardinality
                and nunique < n_rows * 0.5
            ):
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def safe_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 3 or np.allclose(values.std(), 0.0):
        return 0.0
    return float(pd.Series(values).skew())


def outlier_fraction(values: np.ndarray) -> float:
    """Fraction of points outside 1.5*IQR whiskers."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 4:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(np.mean((values < lo) | (values > hi)))
