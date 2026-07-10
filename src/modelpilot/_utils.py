"""Internal helpers. Not part of the public API."""
from __future__ import annotations

from typing import Any

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


def _is_stringy(s: pd.Series) -> bool:
    """True for object / string / str dtypes (covers pandas 3.0's ``str`` dtype)."""
    return str(s.dtype) in ("object", "string", "str") or pd.api.types.is_string_dtype(s)


def _looks_datetime(s: pd.Series, sample: int = 200, min_success: float = 0.9) -> bool:
    """True if ``s`` is a datetime dtype or an object column that parses as dates."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not _is_stringy(s):
        return False
    non_null = s.dropna()
    if non_null.empty:
        return False
    # Pure numbers ("42", 3.14) parse as datetimes but almost never are dates.
    head = non_null.head(sample)
    if all(_is_number_like(v) for v in head):
        return False
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(head, errors="coerce")
    return float(parsed.notna().mean()) >= min_success


def _is_number_like(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _mean_word_count(s: pd.Series, sample: int = 500) -> float:
    non_null = s.dropna().astype(str).head(sample)
    if non_null.empty:
        return 0.0
    return float(np.mean([len(v.split()) for v in non_null]))


def _looks_text(s: pd.Series, n_rows: int, mean_words_min: float = 3.0) -> bool:
    """Free-text heuristic: many distinct values *and* multi-word content.

    Distinguishes prose/comments from high-cardinality categoricals (IDs, codes),
    which are single tokens even when unique.
    """
    if not _is_stringy(s):
        return False
    nunique = s.nunique(dropna=True)
    high_card = nunique > max(20, 0.5 * n_rows)
    return high_card and _mean_word_count(s) >= mean_words_min


def split_feature_types(X: pd.DataFrame, max_numeric_cardinality: int = 15):
    """Classify columns into (numeric, categorical, datetime, text).

    * Low-cardinality integer columns -> categorical (e.g. an ordinal 1-5 rating).
    * Datetime dtypes or date-parseable object columns -> datetime.
    * High-cardinality multi-word object columns -> text (free prose).
    * Everything else object/category/bool -> categorical.
    """
    numeric, categorical, datetime, text = [], [], [], []
    n_rows = max(len(X), 1)
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            if (
                pd.api.types.is_integer_dtype(s)
                and nunique <= max_numeric_cardinality
                and nunique < n_rows * 0.5
            ):
                categorical.append(col)
            else:
                numeric.append(col)
        elif _looks_datetime(s):
            datetime.append(col)
        elif _looks_text(s, n_rows):
            text.append(col)
        else:
            categorical.append(col)
    return numeric, categorical, datetime, text


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
