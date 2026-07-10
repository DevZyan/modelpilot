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
class LeaderboardRow:
    """One model family with both its heuristic rank and its empirical score."""

    key: str
    name: str
    heuristic_rank: int              # 1-based position in the recommendation ranking
    heuristic_score: float
    benchmark_score: float           # NaN if it could not be evaluated
    metric: str
    std: float
    fit_seconds: float
    error: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.error:
            return f"{self.name}: FAILED ({self.error})"
        return (f"{self.name}: {self.metric}={self.benchmark_score:.4f} "
                f"(heuristic #{self.heuristic_rank})")


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

    def leaderboard(self, top_k: int = 5, cv: int = 5) -> list[LeaderboardRow]:
        """Combine the heuristic ranking with real cross-validated scores.

        Cross-validates the top ``top_k`` recommended families, then returns
        them sorted by empirical score (best-first, failures last), each row
        carrying its original heuristic rank so you can see where the advisor
        and the evidence agree — or don't.
        """
        shortlist = self.recommendations[:top_k]
        rank_of = {r.key: i + 1 for i, r in enumerate(shortlist)}
        score_of = {r.key: r.score for r in shortlist}
        results = _benchmark(self._X, self._y, self.profile,
                             [r.key for r in shortlist], cv=cv)
        rows = [
            LeaderboardRow(
                key=b.key,
                name=b.name,
                heuristic_rank=rank_of.get(b.key, 0),
                heuristic_score=score_of.get(b.key, 0.0),
                benchmark_score=b.score,
                metric=b.metric,
                std=b.std,
                fit_seconds=b.fit_seconds,
                error=b.error,
            )
            for b in results
        ]
        rows.sort(key=lambda r: (r.error is not None,
                                 -(r.benchmark_score if r.benchmark_score == r.benchmark_score else -1e9)))
        return rows

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
        types = f"{p.n_numeric} numeric, {p.n_categorical} categorical"
        if p.n_datetime:
            types += f", {p.n_datetime} datetime"
        if p.n_text:
            types += f", {p.n_text} text"
        lines.append(f"  Shape           : {p.n_samples} rows x {p.n_features} features "
                     f"({types})")
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

    def to_html(self, *, include_leaderboard: bool = False,
                top_k: int = 5, cv: int = 5) -> str:
        """Render this report as a self-contained HTML string.

        Set ``include_leaderboard=True`` to cross-validate the top ``top_k``
        picks and embed the measured-score table (slower).
        """
        from .html_report import render_html
        return render_html(self, include_leaderboard=include_leaderboard,
                           top_k=top_k, cv=cv)

    def save_html(self, path: str, *, include_leaderboard: bool = False,
                  top_k: int = 5, cv: int = 5) -> str:
        """Write the HTML report to ``path`` and return the path."""
        html = self.to_html(include_leaderboard=include_leaderboard, top_k=top_k, cv=cv)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path

    def explain(self, key: str | None = None, top: int = 15):
        """Fit a recommended model and rank the features it relies on.

        Defaults to the top recommendation; pass ``key`` to explain another
        family. Returns an :class:`~modelpilot.explain.Explanation`.
        """
        from .explain import explain as _explain
        key = key or self.recommendations[0].key
        return _explain(self._X, self._y, self.profile, key, top=top)

    def leaderboard_summary(self, top_k: int = 5, cv: int = 5) -> str:
        """Pretty-print the combined heuristic + benchmark leaderboard."""
        rows = self.leaderboard(top_k=top_k, cv=cv)
        lines = ["=" * 70,
                 "  modelpilot leaderboard  (heuristic rank vs. measured score)",
                 "=" * 70,
                 f"  {'#':>2}  {'model':<34}{'heur':>6}{'score':>10}{'±std':>9}",
                 "  " + "-" * 66]
        for i, r in enumerate(rows, 1):
            if r.error:
                lines.append(f"  {i:>2}  {r.name[:34]:<34}{('#'+str(r.heuristic_rank)):>6}"
                             f"{'  FAILED':>19}")
            else:
                lines.append(f"  {i:>2}  {r.name[:34]:<34}{('#'+str(r.heuristic_rank)):>6}"
                             f"{r.benchmark_score:>10.4f}{r.std:>9.3f}")
        lines.append("=" * 70)
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
    weights: dict[str, float] | bool | None = None,
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
    weights:
        Meta-learned rule multipliers that rescale the scoring rules. ``None``
        (default) applies the calibrated weights bundled with the package;
        ``False`` disables them for the raw heuristic; a dict uses your own (see
        :func:`modelpilot.meta.fit` / :func:`modelpilot.meta.load`).

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

    if weights is None:
        from . import meta
        weights = meta.default_weights() or None
    elif weights is False:
        weights = None

    prof = profiling.profile(X, y_series, task=task)
    pats = patterns_mod.detect(prof, X, y_series, probe_nonlinearity=probe_nonlinearity)
    recs = recommend_mod.recommend(prof, pats, weights=weights)
    return DatasetReport(prof, pats, recs, X, y_series)
