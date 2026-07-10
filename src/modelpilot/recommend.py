"""Score candidate model families against the profile + detected patterns.

The scorer is transparent by design: every adjustment carries a stable
``rule_id`` and a human-readable reason, so a recommendation is never a black
box. Those rule ids also let a meta-learner (see :mod:`modelpilot.meta`) rescale
each rule's weight from empirical data without changing any code here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import list_model_families, custom_scorer
from .patterns import Pattern
from .profiling import DatasetProfile


@dataclass
class Recommendation:
    key: str
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    # rule_id -> signed contribution actually added to the score (post-weight).
    contributions: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} (score {self.score:+.1f})"


_BASE_PRIOR = {
    "hist_gbm": 3.0,
    "random_forest": 2.0,
    "regularized_linear": 1.0,
    "logreg": 0.5,
}


def recommend(
    profile: DatasetProfile,
    patterns: list[Pattern],
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[Recommendation]:
    """Return model families ranked best-first for this dataset.

    ``weights`` optionally maps a ``rule_id`` to a multiplier applied to that
    rule's default delta (see :func:`modelpilot.meta.load`). Unlisted rules keep
    a multiplier of 1.0, so partial weight sets are fine.
    """
    codes = {p.code for p in patterns}
    task = profile.task
    w = weights or {}
    recs: list[Recommendation] = []

    for m in list_model_families():
        if task not in m.tasks:
            continue

        score = 0.0
        reasons: list[str] = []
        contributions: dict[str, float] = {}

        def bump(rule_id, delta, why):
            nonlocal score
            eff = delta * w.get(rule_id, 1.0)
            if eff == 0.0:
                return
            score += eff
            contributions[rule_id] = contributions.get(rule_id, 0.0) + eff
            reasons.append(f"{'+' if eff >= 0 else ''}{eff:.1f}  {why}")

        # ---- prior ----
        prior = _BASE_PRIOR.get(m.key, 0.0)
        if prior:
            bump(f"base:{m.key}", prior, "strong tabular default")

        # ---- size effects ----
        big = profile.n_samples > 50_000
        small = "SMALL_DATA" in codes
        if big and "scales_to_big_n" not in m.tags:
            bump("big:no_scale", -2.0, "scales poorly to large row counts")
        if small:
            if m.key in ("logreg", "regularized_linear", "naive_bayes", "knn"):
                bump("small:simple", +1.0, "simple model suits small data")
            if m.key in ("mlp", "hist_gbm"):
                bump("small:overfit", -1.0, "risks overfitting on small data")

        # ---- dimensionality ----
        if "HIGH_DIM" in codes:
            if m.key == "regularized_linear":
                bump("highdim:reg", +2.5, "regularization excels when features > samples")
            if m.key == "naive_bayes":
                bump("highdim:nb", +1.0, "robust in high dimensions")
            if m.key in ("knn", "svm_rbf"):
                bump("highdim:curse", -1.5,
                     "distance/kernel methods suffer the curse of dimensionality")

        # ---- nonlinearity ----
        if "NONLINEAR" in codes:
            if "tree" in m.tags or m.key == "svm_rbf":
                bump("nonlinear:capture", +1.5, "captures the detected nonlinear structure")
            if "linear" in m.tags:
                bump("nonlinear:miss", -1.5, "linear model misses nonlinear structure")
        if "LINEAR_FRIENDLY" in codes and "linear" in m.tags:
            bump("linear:friendly", +1.5, "data looks linearly separable")

        # ---- categorical / cardinality ----
        if "HIGH_CARD_CAT" in codes:
            if "handles_categorical" in m.tags:
                bump("highcard:trees", +1.0,
                     "tree ensembles handle high-cardinality categoricals well")
            if "linear" in m.tags or m.key in ("knn", "svm_rbf"):
                bump("highcard:onehot", -1.0, "one-hot encoding blows up dimensionality")

        # ---- data quality ----
        if "MISSING_HEAVY" in codes and "handles_missing" in m.tags:
            bump("missing:native", +0.5, "handles missing values natively")
        if "OUTLIER_HEAVY" in codes:
            if "tree" in m.tags:
                bump("outlier:trees", +0.5, "trees are robust to outliers")
            if m.key in ("knn", "logreg", "regularized_linear"):
                bump("outlier:sensitive", -0.5, "sensitive to outliers")
        if "MULTICOLLINEAR" in codes:
            if m.key == "regularized_linear":
                bump("collinear:penalty", +0.5, "penalty stabilizes correlated features")
            if m.key == "logreg":
                bump("collinear:unstable", -0.5, "unstable coefficients under multicollinearity")
        if "SPARSE" in codes and m.key in ("naive_bayes", "regularized_linear"):
            bump("sparse:efficient", +1.0, "handles sparse features efficiently")

        # ---- imbalance ----
        if "IMBALANCED" in codes:
            if "ensemble" in m.tags:
                bump("imbalance:ensemble", +0.5, "class weighting / boosting helps on imbalance")
            if m.key == "knn":
                bump("imbalance:knn", -0.5, "kNN biased toward majority class")

        # ---- user-supplied scoring rules for custom families ----
        scorer = custom_scorer(m.key)
        if scorer is not None:
            for delta, why in scorer(m, profile, codes):
                bump(f"custom:{m.key}", float(delta), why)

        recs.append(Recommendation(m.key, m.name, round(score, 2), reasons, contributions))

    recs.sort(key=lambda r: r.score, reverse=True)
    if top_k is not None:
        recs = recs[:top_k]
    return recs
