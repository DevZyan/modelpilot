"""Catalog of candidate model families and their characteristics.

Each entry stays framework-light: estimator builders are resolved lazily in
``benchmark.py`` so importing the package never requires xgboost/lightgbm.

Users can extend the catalog at runtime with :func:`register_model_family`,
attaching their own estimator and (optionally) scoring rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


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


# ---------------------------------------------------------------------------
# Runtime-extensible registry
# ---------------------------------------------------------------------------
#
# A "builder" is a callable ``(task: str) -> estimator | None`` invoked lazily at
# benchmark time (so heavy imports stay optional). For convenience an estimator
# class or a ready instance may be passed instead and is wrapped automatically.
#
# A "scorer" is an optional callable ``(family, profile, pattern_codes) ->
# Iterable[(delta: float, reason: str)]`` that nudges the family's score with
# transparent reasons, exactly like the built-in rules.

Builder = Callable[[str], Any]
Scorer = Callable[["ModelFamily", Any, "set[str]"], "Iterable[tuple[float, str]]"]

_CUSTOM_FAMILIES: dict[str, ModelFamily] = {}
_CUSTOM_BUILDERS: dict[str, Builder] = {}
_CUSTOM_SCORERS: dict[str, Scorer] = {}


def _as_builder(estimator: Any) -> Builder:
    if isinstance(estimator, type):                 # an estimator class
        return lambda task, cls=estimator: cls()
    if hasattr(estimator, "fit"):                   # a ready estimator instance
        return lambda task, est=estimator: est
    if callable(estimator):                         # a (task) -> estimator builder
        return estimator
    raise TypeError(
        "estimator must be a builder callable (task) -> estimator, an estimator "
        "class, or an estimator instance with a .fit method"
    )


def register_model_family(
    family: ModelFamily,
    estimator: Any,
    *,
    scorer: Scorer | None = None,
    overwrite: bool = False,
) -> None:
    """Register a custom model family so it joins the recommendation ranking.

    Parameters
    ----------
    family:
        A :class:`ModelFamily` describing the model (its ``key`` must be unique).
    estimator:
        How to build the estimator for a task: a callable ``(task) -> estimator``,
        an estimator class, or a ready estimator instance.
    scorer:
        Optional ``(family, profile, pattern_codes) -> [(delta, reason), ...]``
        used to nudge this family's score with auditable reasons.
    overwrite:
        Allow replacing an existing custom family with the same key.

    Example
    -------
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> import modelpilot as mp
    >>> fam = mp.ModelFamily("extra_trees", "Extra Trees",
    ...                       ("classification",), "Randomised trees.",
    ...                       frozenset({"tree", "ensemble"}))
    >>> def score(f, prof, codes):
    ...     return [(1.0, "extra randomisation helps on noisy data")] if "OUTLIER_HEAVY" in codes else []
    >>> mp.register_model_family(fam, ExtraTreesClassifier, scorer=score)
    """
    if not isinstance(family, ModelFamily):
        raise TypeError("family must be a ModelFamily instance")
    key = family.key
    if key in BY_KEY:
        raise ValueError(f"{key!r} is a built-in family and cannot be overridden")
    if key in _CUSTOM_FAMILIES and not overwrite:
        raise ValueError(f"{key!r} already registered; pass overwrite=True to replace")
    _CUSTOM_FAMILIES[key] = family
    _CUSTOM_BUILDERS[key] = _as_builder(estimator)
    if scorer is not None:
        _CUSTOM_SCORERS[key] = scorer
    else:
        _CUSTOM_SCORERS.pop(key, None)


def unregister_model_family(key: str) -> None:
    """Remove a previously registered custom family. Built-ins can't be removed."""
    if key in BY_KEY:
        raise ValueError(f"{key!r} is a built-in family and cannot be removed")
    if key not in _CUSTOM_FAMILIES:
        raise KeyError(f"no custom family registered under {key!r}")
    _CUSTOM_FAMILIES.pop(key, None)
    _CUSTOM_BUILDERS.pop(key, None)
    _CUSTOM_SCORERS.pop(key, None)


def list_model_families() -> tuple[ModelFamily, ...]:
    """All families the recommender will consider: built-ins plus custom ones."""
    return CATALOG + tuple(_CUSTOM_FAMILIES.values())


def get_family(key: str) -> ModelFamily | None:
    return BY_KEY.get(key) or _CUSTOM_FAMILIES.get(key)


def custom_builder(key: str) -> Builder | None:
    return _CUSTOM_BUILDERS.get(key)


def custom_scorer(key: str) -> Scorer | None:
    return _CUSTOM_SCORERS.get(key)
