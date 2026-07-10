"""Meta-learn the scoring rule weights from empirical benchmark outcomes.

The heuristic scorer in :mod:`modelpilot.recommend` emits, for every model
family, a set of ``rule_id -> delta`` contributions. This module fits a
multiplier per ``rule_id`` so that the weighted contributions best predict how
each family *actually* ranks when cross-validated. Feed the learned multipliers
back via ``analyze(..., weights=...)`` and the advisor is calibrated on data
instead of hand-tuned constants — while staying fully transparent (each rule
keeps its reason string; only its magnitude changes).

Nothing here runs at import time and no network access is required: point
:func:`fit` at any list of ``(X, y)`` datasets you trust (see
:func:`builtin_datasets` for an offline starter set).
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import profiling
from .patterns import detect as _detect
from .recommend import recommend as _recommend
from .benchmark import benchmark as _benchmark


def _dataset_rows(X: pd.DataFrame, y: pd.Series, task: str | None, cv: int):
    """Yield ``(contributions, empirical_z_score)`` for each family on one dataset."""
    prof = profiling.profile(X, y, task=task)
    pats = _detect(prof, X, y, probe_nonlinearity=True)
    recs = _recommend(prof, pats)                                 # default weights
    keys = [r.key for r in recs]
    results = _benchmark(X, y, prof, keys, cv=cv)

    score_by_key = {
        b.key: b.score for b in results
        if b.error is None and b.score == b.score            # drop failures / NaN
    }
    if len(score_by_key) < 2:
        return

    scores = np.array(list(score_by_key.values()), dtype=float)
    mean, std = float(scores.mean()), float(scores.std())
    if std == 0.0:
        return

    contrib_by_key = {r.key: r.contributions for r in recs}
    for key, raw in score_by_key.items():
        z = (raw - mean) / std
        yield contrib_by_key.get(key, {}), z


def fit(
    datasets: Iterable[tuple],
    *,
    cv: int = 3,
    ridge_alpha: float = 1.0,
    clip: tuple[float, float] = (-1.0, 3.0),
) -> dict[str, float]:
    """Learn a ``rule_id -> multiplier`` map from cross-validated outcomes.

    Parameters
    ----------
    datasets:
        Iterable of ``(X, y)`` or ``(X, y, task)`` tuples to learn from. More and
        more varied datasets give better-calibrated weights.
    cv:
        Cross-validation folds used per dataset while benchmarking.
    ridge_alpha:
        L2 strength of the ridge fit that maps rule contributions to score.
    clip:
        Clamp learned multipliers to this range so no single rule can dominate.

    Returns
    -------
    dict
        Multipliers keyed by ``rule_id``. Pass to ``analyze(..., weights=...)``.
    """
    from sklearn.linear_model import Ridge

    rows: list[tuple[dict[str, float], float]] = []
    for item in datasets:
        if len(item) == 3:
            X, y, task = item
        else:
            (X, y), task = item, None
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        rows.extend(_dataset_rows(X, y, task, cv))

    if not rows:
        raise ValueError("no usable (dataset, family) observations were collected")

    rule_ids = sorted({rid for contrib, _ in rows for rid in contrib})
    if not rule_ids:
        raise ValueError("no scoring rules fired across the given datasets")
    index = {rid: i for i, rid in enumerate(rule_ids)}

    design = np.zeros((len(rows), len(rule_ids)), dtype=float)
    target = np.zeros(len(rows), dtype=float)
    for r, (contrib, z) in enumerate(rows):
        for rid, delta in contrib.items():
            design[r, index[rid]] = delta
        target[r] = z

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(design, target)

    lo, hi = clip
    return {rid: float(np.clip(model.coef_[index[rid]], lo, hi)) for rid in rule_ids}


def save(weights: dict[str, float], path: str) -> str:
    """Write learned weights to ``path`` as JSON; returns the path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, indent=2, sort_keys=True)
    return path


def load(path: str) -> dict[str, float]:
    """Load weights previously written by :func:`save`."""
    with open(path, encoding="utf-8") as fh:
        return {str(k): float(v) for k, v in json.load(fh).items()}


def builtin_datasets() -> list[tuple[Any, Any]]:
    """A small offline starter set (from scikit-learn) for demos and tests."""
    from sklearn.datasets import (
        load_iris, load_wine, load_breast_cancer, load_diabetes,
    )
    out = []
    for loader in (load_iris, load_wine, load_breast_cancer, load_diabetes):
        frame = loader(as_frame=True).frame
        out.append((frame.drop(columns="target"), frame["target"]))
    return out
