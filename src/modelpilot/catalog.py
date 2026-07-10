"""Catalog of candidate model families and their characteristics.

Each entry stays framework-light: estimator builders are resolved lazily in
``benchmark.py`` so importing the package never requires xgboost/lightgbm.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelFamily:
    key: str
    name: str
    tasks: tuple[str, ...]          # which of classification/regression it supports
    blurb: str
    tags: frozenset[str] = field(default_factory=frozenset)


# Tags used by the scorer:
#   linear, tree, ensemble, kernel, distance, probabilistic, neural,
#   handles_missing, handles_categorical, scales_to_big_n, needs_scaling
CATALOG: tuple[ModelFamily, ...] = (
    ModelFamily(
        "logreg", "Logistic / Linear Regression", ("classification", "regression"),
        "Fast, interpretable linear baseline. Hard to beat when the signal is linear.",
        frozenset({"linear", "needs_scaling"}),
    ),
    ModelFamily(
        "regularized_linear", "Regularized Linear (Ridge/Lasso/ElasticNet)",
        ("classification", "regression"),
        "Linear model with L1/L2 penalties. Shines in high-dimensional / p>n / sparse settings.",
        frozenset({"linear", "needs_scaling"}),
    ),
    ModelFamily(
        "random_forest", "Random Forest", ("classification", "regression"),
        "Robust bagged trees. Great default: handles nonlinearity, outliers and mixed types.",
        frozenset({"tree", "ensemble", "handles_missing", "handles_categorical", "scales_to_big_n"}),
    ),
    ModelFamily(
        "hist_gbm", "Gradient Boosting (HistGradientBoosting / XGBoost / LightGBM)",
        ("classification", "regression"),
        "Usually the top performer on tabular data. Handles nonlinearity and interactions.",
        frozenset({"tree", "ensemble", "handles_missing", "handles_categorical", "scales_to_big_n"}),
    ),
    ModelFamily(
        "svm_rbf", "SVM (RBF kernel)", ("classification", "regression"),
        "Powerful on small/medium nonlinear data, but scales poorly past ~10k rows.",
        frozenset({"kernel", "needs_scaling"}),
    ),
    ModelFamily(
        "knn", "k-Nearest Neighbours", ("classification", "regression"),
        "Simple, nonparametric. Good on small low-dimensional data; degrades in high dimensions.",
        frozenset({"distance", "needs_scaling"}),
    ),
    ModelFamily(
        "naive_bayes", "Naive Bayes", ("classification",),
        "Extremely fast probabilistic baseline; strong on sparse / high-dimensional data.",
        frozenset({"probabilistic", "scales_to_big_n"}),
    ),
    ModelFamily(
        "mlp", "Neural Network (MLP)", ("classification", "regression"),
        "Flexible function approximator; needs lots of data and tuning to pay off on tabular.",
        frozenset({"neural", "needs_scaling", "scales_to_big_n"}),
    ),
)

BY_KEY = {m.key: m for m in CATALOG}
