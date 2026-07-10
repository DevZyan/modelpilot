"""Detect named, human-readable patterns from a :class:`DatasetProfile`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .profiling import DatasetProfile


@dataclass
class Pattern:
    code: str          # machine-readable, e.g. "IMBALANCED"
    label: str         # short human label
    detail: str        # one-line explanation with the evidence
    severity: str      # "info" | "warn" | "high"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.upper()}] {self.code}: {self.detail}"


# Thresholds are deliberately explicit and tweakable.
SMALL_DATA = 1000
WIDE_RATIO = 0.2           # n_features / n_samples above this = high dimensional
IMBALANCE = 3.0            # majority/minority above this = imbalanced
HIGH_CARDINALITY = 50
MULTICOLLINEAR = 0.9
MISSING = 0.05
OUTLIER = 0.05
SPARSE = 0.6
SKEW = 2.0
LOW_SIGNAL = 0.05


def _nonlinearity_gap(X: pd.DataFrame, y: pd.Series, profile: DatasetProfile) -> float | None:
    """Estimate how much nonlinear structure exists.

    Compares quick cross-validated scores of a linear model vs a shallow tree
    ensemble on a numeric subsample. Returns ``tree - linear`` (higher = more
    nonlinear signal the linear model misses). ``None`` if it cannot be run.
    """
    num = profile.numeric_cols
    if len(num) == 0 or profile.n_samples < 60:
        return None
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.linear_model import LogisticRegression, Ridge

        Xn = X[num].apply(pd.to_numeric, errors="coerce")
        # Subsample for speed.
        if len(Xn) > 3000:
            idx = np.random.RandomState(0).choice(len(Xn), 3000, replace=False)
            Xn, yy = Xn.iloc[idx], y.iloc[idx]
        else:
            yy = y

        if profile.task == "classification":
            linear = make_pipeline(SimpleImputer(), StandardScaler(),
                                   LogisticRegression(max_iter=500))
            tree = make_pipeline(SimpleImputer(),
                                 DecisionTreeClassifier(max_depth=6, random_state=0))
            scoring = "accuracy"
        else:
            linear = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
            tree = make_pipeline(SimpleImputer(),
                                 DecisionTreeRegressor(max_depth=6, random_state=0))
            scoring = "r2"

        cv = min(3, int(yy.value_counts().min())) if profile.task == "classification" else 3
        cv = max(cv, 2)
        lin_score = float(np.mean(cross_val_score(linear, Xn, yy, cv=cv, scoring=scoring)))
        tree_score = float(np.mean(cross_val_score(tree, Xn, yy, cv=cv, scoring=scoring)))
        return tree_score - lin_score
    except Exception:
        return None


def detect(
    profile: DatasetProfile,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    probe_nonlinearity: bool = True,
) -> list[Pattern]:
    """Return the list of :class:`Pattern` objects that apply to ``profile``."""
    p = profile
    found: list[Pattern] = []

    def add(code, label, detail, severity="info"):
        found.append(Pattern(code, label, detail, severity))

    if p.n_samples < SMALL_DATA:
        add("SMALL_DATA", "Small dataset",
            f"Only {p.n_samples} rows — complex models risk overfitting.", "warn")

    if p.dim_ratio > WIDE_RATIO:
        add("HIGH_DIM", "High dimensional",
            f"{p.n_features} features vs {p.n_samples} rows "
            f"(ratio {p.dim_ratio:.2f}) — regularisation matters.", "warn")

    if p.task == "classification" and p.imbalance_ratio and p.imbalance_ratio >= IMBALANCE:
        add("IMBALANCED", "Class imbalance",
            f"Majority/minority ratio {p.imbalance_ratio:.1f} "
            f"(minority {p.minority_fraction:.1%}) — accuracy is misleading.", "high")

    if p.max_categorical_cardinality >= HIGH_CARDINALITY:
        add("HIGH_CARD_CAT", "High-cardinality categoricals",
            f"A categorical has {p.max_categorical_cardinality} levels — "
            "one-hot explodes; prefer target/ordinal encoding + trees.", "warn")

    if p.max_correlation >= MULTICOLLINEAR:
        add("MULTICOLLINEAR", "Multicollinearity",
            f"Two features correlate at {p.max_correlation:.2f} — "
            "hurts linear coefficient stability.", "warn")

    if p.missing_fraction >= MISSING:
        add("MISSING_HEAVY", "Substantial missing data",
            f"{p.missing_fraction:.1%} of cells are missing — "
            "imputation quality will matter.", "warn")

    if p.mean_outlier_fraction >= OUTLIER:
        add("OUTLIER_HEAVY", "Outliers present",
            f"~{p.mean_outlier_fraction:.1%} of numeric values are outliers — "
            "trees are more robust than linear/kNN.", "info")

    if p.sparsity >= SPARSE:
        add("SPARSE", "Sparse features",
            f"{p.sparsity:.1%} of numeric entries are zero — "
            "favours linear / naive Bayes models.", "info")

    if p.mean_abs_skew >= SKEW:
        add("SKEWED", "Skewed features",
            f"Mean |skew| {p.mean_abs_skew:.1f} — consider log/power transforms "
            "for scale-sensitive models.", "info")

    if p.target_signal < LOW_SIGNAL:
        add("LOW_SIGNAL", "Weak feature-target signal",
            f"Estimated signal {p.target_signal:.2f} — even good models "
            "may struggle; revisit feature engineering.", "high")

    if p.duplicate_row_fraction > 0.02:
        add("DUPLICATES", "Duplicate rows",
            f"{p.duplicate_row_fraction:.1%} of rows are duplicated — "
            "may leak between train/test.", "warn")

    if p.constant_feature_fraction > 0:
        add("CONSTANT_FEATURES", "Constant features",
            f"{p.constant_feature_fraction:.1%} of features are constant — drop them.", "info")

    if p.n_datetime:
        cols = ", ".join(p.datetime_cols[:3]) + ("…" if p.n_datetime > 3 else "")
        add("HAS_DATETIME", "Datetime features",
            f"{p.n_datetime} datetime column(s) ({cols}) — extract year/month/day/"
            "weekday/hour parts; raw timestamps aren't model-ready.", "info")

    if p.n_text:
        cols = ", ".join(p.text_cols[:3]) + ("…" if p.n_text > 3 else "")
        add("HAS_TEXT", "Free-text features",
            f"{p.n_text} free-text column(s) ({cols}) — vectorise (TF-IDF / embeddings) "
            "before modeling; they're dropped by the default benchmark.", "info")

    # Optional empirical nonlinearity probe.
    if probe_nonlinearity and X is not None and y is not None:
        gap = _nonlinearity_gap(X, y, profile)
        if gap is not None:
            if gap > 0.05:
                add("NONLINEAR", "Nonlinear structure",
                    f"A shallow tree beats a linear model by {gap:+.2f} — "
                    "nonlinear models should win.", "info")
            elif gap < -0.02:
                add("LINEAR_FRIENDLY", "Linear-friendly",
                    f"A linear model matches/beats a tree ({gap:+.2f}) — "
                    "linear models are a strong, simple choice.", "info")

    return found
