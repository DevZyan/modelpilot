"""Score candidate model families against the profile + detected patterns.

The scorer is transparent by design: every adjustment carries a human-readable
reason, so a recommendation is never a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import CATALOG, ModelFamily
from .patterns import Pattern
from .profiling import DatasetProfile


@dataclass
class Recommendation:
    key: str
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} (score {self.score:+.1f})"


def _base_score(m: ModelFamily) -> float:
    """Prior: gradient boosting and forests are the strongest tabular defaults."""
    return {
        "hist_gbm": 3.0,
        "random_forest": 2.0,
        "regularized_linear": 1.0,
        "logreg": 0.5,
    }.get(m.key, 0.0)


def recommend(
    profile: DatasetProfile,
    patterns: list[Pattern],
    top_k: int | None = None,
) -> list[Recommendation]:
    """Return model families ranked best-first for this dataset."""
    codes = {p.code for p in patterns}
    task = profile.task
    recs: list[Recommendation] = []

    for m in CATALOG:
        if task not in m.tasks:
            continue

        score = _base_score(m)
        reasons: list[str] = []

        def bump(delta, why):
            nonlocal score
            score += delta
            reasons.append(f"{'+' if delta >= 0 else ''}{delta:.1f}  {why}")

        # ---- size effects ----
        big = profile.n_samples > 50_000
        small = "SMALL_DATA" in codes
        if big and "scales_to_big_n" not in m.tags:
            bump(-2.0, "scales poorly to large row counts")
        if small:
            if m.key in ("logreg", "regularized_linear", "naive_bayes", "knn"):
                bump(+1.0, "simple model suits small data")
            if m.key in ("mlp", "hist_gbm"):
                bump(-1.0, "risks overfitting on small data")

        # ---- dimensionality ----
        if "HIGH_DIM" in codes:
            if m.key == "regularized_linear":
                bump(+2.5, "regularization excels when features > samples")
            if m.key == "naive_bayes":
                bump(+1.0, "robust in high dimensions")
            if m.key in ("knn", "svm_rbf"):
                bump(-1.5, "distance/kernel methods suffer the curse of dimensionality")

        # ---- nonlinearity ----
        if "NONLINEAR" in codes:
            if "tree" in m.tags or m.key == "svm_rbf":
                bump(+1.5, "captures the detected nonlinear structure")
            if "linear" in m.tags:
                bump(-1.5, "linear model misses nonlinear structure")
        if "LINEAR_FRIENDLY" in codes and "linear" in m.tags:
            bump(+1.5, "data looks linearly separable")

        # ---- categorical / cardinality ----
        if "HIGH_CARD_CAT" in codes:
            if "handles_categorical" in m.tags:
                bump(+1.0, "tree ensembles handle high-cardinality categoricals well")
            if "linear" in m.tags or m.key in ("knn", "svm_rbf"):
                bump(-1.0, "one-hot encoding blows up dimensionality")

        # ---- data quality ----
        if "MISSING_HEAVY" in codes and "handles_missing" in m.tags:
            bump(+0.5, "handles missing values natively")
        if "OUTLIER_HEAVY" in codes:
            if "tree" in m.tags:
                bump(+0.5, "trees are robust to outliers")
            if m.key in ("knn", "logreg", "regularized_linear"):
                bump(-0.5, "sensitive to outliers")
        if "MULTICOLLINEAR" in codes:
            if m.key == "regularized_linear":
                bump(+0.5, "penalty stabilizes correlated features")
            if m.key == "logreg":
                bump(-0.5, "unstable coefficients under multicollinearity")
        if "SPARSE" in codes and m.key in ("naive_bayes", "regularized_linear"):
            bump(+1.0, "handles sparse features efficiently")

        # ---- imbalance ----
        if "IMBALANCED" in codes:
            if "ensemble" in m.tags:
                bump(+0.5, "class weighting / boosting helps on imbalance")
            if m.key == "knn":
                bump(-0.5, "kNN biased toward majority class")

        recs.append(Recommendation(m.key, m.name, round(score, 2), reasons))

    recs.sort(key=lambda r: r.score, reverse=True)
    if top_k is not None:
        recs = recs[:top_k]
    return recs
