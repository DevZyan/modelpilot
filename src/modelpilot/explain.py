"""Fit the recommended model and surface which features drive it.

Recommendations tell you *which family* to use and why; this closes the loop by
fitting the pick and reporting the features it actually leans on — impurity
importances for trees, coefficient magnitudes for linear models, and a
permutation-importance fallback for everything else.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .benchmark import build_pipeline
from .profiling import DatasetProfile


@dataclass
class Explanation:
    key: str
    name: str
    method: str                      # "impurity" | "coefficient" | "permutation"
    features: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"  Top features for {self.name}  (importance: {self.method})",
                 "  " + "-" * 48]
        if not self.features:
            lines.append("    (no importances available for this model)")
        else:
            top = max((abs(v) for _, v in self.features), default=1.0) or 1.0
            for feat, val in self.features:
                bar = "#" * int(round(abs(val) / top * 20))
                lines.append(f"    {feat[:28]:28} {val:8.4f} {bar}")
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = ", ".join(f"{f}={v:.3f}" for f, v in self.features[:3])
        return f"<Explanation {self.name!r} via {self.method}: {head}>"


def _feature_names(pipe, X: pd.DataFrame) -> list[str]:
    pre = pipe.named_steps.get("pre")
    if pre is not None:
        try:
            # Drop the ColumnTransformer's "num__"/"cat__"/"dt__" prefixes.
            return [str(n).split("__", 1)[-1] for n in pre.get_feature_names_out()]
        except Exception:
            pass
    return [str(c) for c in X.columns]


def _model_importances(model) -> tuple[np.ndarray | None, str]:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float), "impurity"
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:                       # multiclass: average |coef| over classes
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        return coef, "coefficient"
    return None, ""


def explain(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DatasetProfile,
    key: str,
    top: int = 15,
) -> Explanation:
    """Fit family ``key`` on ``(X, y)`` and rank its most influential features."""
    from .catalog import get_family

    pipe = build_pipeline(profile, key)
    name = (get_family(key).name if get_family(key) else key)
    if pipe is None:
        return Explanation(key, name, "", [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
        model = pipe.named_steps.get("model", pipe.steps[-1][1])

        values, method = _model_importances(model)
        if values is not None:
            names = _feature_names(pipe, X)
        else:
            # Model isn't natively interpretable: permute raw input columns.
            from sklearn.inspection import permutation_importance
            method = "permutation"
            names = [str(c) for c in X.columns]
            result = permutation_importance(pipe, X, y, n_repeats=5, random_state=0)
            values = np.asarray(result.importances_mean, dtype=float)

    n = min(len(names), len(values))
    pairs = sorted(zip(names[:n], values[:n]), key=lambda kv: abs(kv[1]), reverse=True)
    return Explanation(key, name, method, pairs[:top])
