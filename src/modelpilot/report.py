"""The high-level object users interact with."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import profiling, patterns as patterns_mod, recommend as recommend_mod
from .benchmark import benchmark as _benchmark, BenchmarkResult
from .patterns import Pattern
from .profiling import DatasetProfile
from .recommend import Recommendation


@dataclass
class DatasetReport:
    """Bundles the profile, detected patterns and model recommendations."""

    profile: DatasetProfile
    patterns: list[Pattern]
    recommendations: list[Recommendation]
    _X: pd.DataFrame
    _y: pd.Series

    # ---- convenience accessors ----
    @property
    def task(self) -> str:
        return self.profile.task

    @property
    def best(self) -> Recommendation:
        return self.recommendations[0]

    def recommend(self, top_k: int | None = None) -> list[Recommendation]:
        return self.recommendations[:top_k] if top_k else self.recommendations

    def benchmark(self, top_k: int = 5, cv: int = 5) -> list[BenchmarkResult]:
        """Empirically cross-validate the top recommendations."""
        keys = [r.key for r in self.recommendations[:top_k]]
        return _benchmark(self._X, self._y, self.profile, keys, cv=cv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "patterns": [p.__dict__ for p in self.patterns],
            "recommendations": [r.__dict__ for r in self.recommendations],
        }

    # ---- pretty printing ----
    def summary(self) -> str:
        p = self.profile
        lines = []
        lines.append("=" * 62)
        lines.append("  modelpilot report")
        lines.append("=" * 62)
        lines.append(f"  Task            : {p.task}")
        lines.append(f"  Shape           : {p.n_samples} rows x {p.n_features} features "
                     f"({p.n_numeric} numeric, {p.n_categorical} categorical)")
        if p.task == "classification":
            lines.append(f"  Classes         : {p.n_classes} "
                         f"(imbalance {p.imbalance_ratio:.1f}x)" if p.imbalance_ratio
                         else f"  Classes         : {p.n_classes}")
        lines.append(f"  Missing         : {p.missing_fraction:.1%}")
        lines.append(f"  Target signal   : {p.target_signal:.2f}")
        lines.append("")
        lines.append("  Patterns detected:")
        if self.patterns:
            for pat in self.patterns:
                mark = {"info": "·", "warn": "!", "high": "!!"}.get(pat.severity, "·")
                lines.append(f"    {mark:>2} {pat.label} — {pat.detail}")
        else:
            lines.append("    (none — clean, well-behaved dataset)")
        lines.append("")
        lines.append("  Recommended models:")
        for i, r in enumerate(self.recommendations[:5], 1):
            lines.append(f"    {i}. {r.name}  (score {r.score:+.1f})")
            if r.reasons:
                top = [x for x in r.reasons if x.startswith("+")][:2]
                for reason in top:
                    lines.append(f"         {reason}")
        lines.append("=" * 62)
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<DatasetReport task={self.profile.task} "
                f"best={self.recommendations[0].name!r} "
                f"patterns={len(self.patterns)}>")


def analyze(
    data: pd.DataFrame,
    y: Any = None,
    *,
    target: str | None = None,
    task: str | None = None,
    probe_nonlinearity: bool = True,
) -> DatasetReport:
    """Analyze a dataset and return a :class:`DatasetReport`.

    Parameters
    ----------
    data:
        A pandas DataFrame of features (optionally including the target column).
    y:
        The target as a Series/array, if not part of ``data``.
    target:
        Name of the target column inside ``data`` (alternative to ``y``).
    task:
        Force ``"classification"`` or ``"regression"``; inferred if omitted.
    probe_nonlinearity:
        Run the quick linear-vs-tree probe (adds a couple of seconds).

    Examples
    --------
    >>> import modelpilot as mp
    >>> report = mp.analyze(df, target="price")
    >>> report.summary()
    >>> report.benchmark()
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if target is not None:
        if target not in data.columns:
            raise KeyError(f"target column {target!r} not found in data")
        y_series = data[target]
        X = data.drop(columns=[target])
    elif y is not None:
        y_series = pd.Series(y).reset_index(drop=True)
        X = data.reset_index(drop=True)
    else:
        raise ValueError("Provide either `target=` (column name) or `y=` (values).")

    if len(X) != len(y_series):
        raise ValueError(f"X has {len(X)} rows but y has {len(y_series)}.")

    prof = profiling.profile(X, y_series, task=task)
    pats = patterns_mod.detect(prof, X, y_series, probe_nonlinearity=probe_nonlinearity)
    recs = recommend_mod.recommend(prof, pats)
    return DatasetReport(prof, pats, recs, X, y_series)
