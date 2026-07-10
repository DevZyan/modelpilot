"""Build real sklearn pipelines and cross-validate a shortlist of models.

This is what turns a *recommendation* into *evidence*: it runs the candidate
families with sensible preprocessing and reports scores + fit time.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .profiling import DatasetProfile


@dataclass
class BenchmarkResult:
    key: str
    name: str
    score: float          # primary metric (higher is better)
    metric: str
    std: float
    fit_seconds: float
    error: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.error:
            return f"{self.name}: FAILED ({self.error})"
        return f"{self.name}: {self.metric}={self.score:.4f} (±{self.std:.3f})"


def _datetime_parts(df) -> np.ndarray:
    """Expand each datetime column into [year, month, day, weekday, hour]."""
    df = pd.DataFrame(df)
    cols = []
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        cols.extend([s.dt.year, s.dt.month, s.dt.day, s.dt.dayofweek, s.dt.hour])
    return pd.concat(cols, axis=1).to_numpy(dtype=float)


def _fill_text(s) -> list[str]:
    """Coerce a text column to a clean list of strings for the vectorizer."""
    return pd.Series(s).fillna("").astype(str).tolist()


def _to_dense(x):
    """Densify a (possibly sparse) matrix so every estimator can consume it."""
    return x.toarray() if hasattr(x, "toarray") else x


_DT_PARTS = ("year", "month", "day", "weekday", "hour")


def _dt_names(transformer, input_features):
    """Feature names for the datetime-parts expansion (5 per source column)."""
    return np.asarray([f"{c}_{part}" for c in input_features for part in _DT_PARTS])


def _preprocessor(profile: DatasetProfile):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer

    num = profile.numeric_cols
    cat = profile.categorical_cols
    dts = profile.datetime_cols
    txt = profile.text_cols

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if num:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            num,
        ))
    if cat:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("ohe", ohe)]),
            cat,
        ))
    if dts:
        transformers.append((
            "dt",
            Pipeline([("parts", FunctionTransformer(_datetime_parts, validate=False,
                                                    feature_names_out=_dt_names)),
                      ("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            dts,
        ))
    for i, col in enumerate(txt):
        # A scalar column selector hands the vectorizer a 1-D column of strings.
        transformers.append((
            f"txt{i}",
            Pipeline([("fill", FunctionTransformer(_fill_text, validate=False,
                                                   feature_names_out="one-to-one")),
                      ("tfidf", TfidfVectorizer(max_features=200)),
                      ("dense", FunctionTransformer(_to_dense, validate=False,
                                                    feature_names_out="one-to-one"))]),
            col,
        ))
    if not transformers:
        return "passthrough"
    return ColumnTransformer(transformers, remainder="drop")


def _estimator(key: str, task: str):
    """Return an estimator instance for a catalog key, or None if unavailable."""
    from .catalog import custom_builder

    builder = custom_builder(key)
    if builder is not None:
        return builder(task)

    from sklearn.linear_model import (
        LogisticRegression, Ridge, ElasticNet,
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    clf = task == "classification"
    table = {
        "logreg": (LogisticRegression(max_iter=1000) if clf else Ridge()),
        "regularized_linear": (
            LogisticRegression(penalty="l2", C=0.5, max_iter=1000) if clf
            else ElasticNet(alpha=0.1)
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1) if clf
            else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        ),
        "hist_gbm": (
            HistGradientBoostingClassifier(random_state=0) if clf
            else HistGradientBoostingRegressor(random_state=0)
        ),
        "svm_rbf": (SVC(probability=False) if clf else SVR()),
        "knn": (KNeighborsClassifier() if clf else KNeighborsRegressor()),
        "naive_bayes": (GaussianNB() if clf else None),
        "mlp": (
            MLPClassifier(max_iter=300, random_state=0) if clf
            else MLPRegressor(max_iter=300, random_state=0)
        ),
    }
    return table.get(key)


def build_pipeline(profile: DatasetProfile, key: str):
    """Return an *unfitted* preprocessing + estimator pipeline, or None."""
    from sklearn.pipeline import Pipeline

    est = _estimator(key, profile.task)
    if est is None:
        return None
    pre = _preprocessor(profile)
    if pre == "passthrough":
        return Pipeline([("model", est)])
    return Pipeline([("pre", pre), ("model", est)])


def _scoring(profile: DatasetProfile) -> tuple[str, str]:
    if profile.task == "classification":
        if profile.n_classes and profile.n_classes > 2:
            return "f1_macro", "f1_macro"
        return "roc_auc", "roc_auc"
    return "r2", "r2"


def benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DatasetProfile,
    keys: list[str],
    cv: int = 5,
    max_rows: int = 20_000,
) -> list[BenchmarkResult]:
    """Cross-validate the model families named in ``keys``. Best-first output."""
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    # Subsample very large datasets so the benchmark stays fast.
    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

    scoring, metric = _scoring(profile)
    if profile.task == "classification":
        min_class = int(pd.Series(y).value_counts().min())
        cv = max(2, min(cv, min_class))

    pre = _preprocessor(profile)
    results: list[BenchmarkResult] = []

    for key in keys:
        est = _estimator(key, profile.task)
        name = _display_name(key)
        if est is None:
            results.append(BenchmarkResult(key, name, float("nan"), metric, 0.0, 0.0,
                                           error="not available for this task"))
            continue
        try:
            pipe = Pipeline([("pre", pre), ("model", est)]) if pre != "passthrough" \
                else Pipeline([("model", est)])
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            dt = time.perf_counter() - t0
            results.append(BenchmarkResult(
                key, name, float(np.mean(scores)), metric,
                float(np.std(scores)), dt,
            ))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            results.append(BenchmarkResult(key, name, float("nan"), metric, 0.0, 0.0,
                                           error=str(exc)[:120]))

    results.sort(key=lambda r: (r.error is not None, -(r.score if r.score == r.score else -1e9)))
    return results


def _display_name(key: str) -> str:
    from .catalog import get_family
    m = get_family(key)
    return m.name if m else key
