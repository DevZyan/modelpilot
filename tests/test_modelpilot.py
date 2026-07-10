import warnings

import numpy as np
import pandas as pd
import pytest

import modelpilot as mp

warnings.filterwarnings("ignore")


@pytest.fixture
def clf_df():
    from sklearn.datasets import load_iris
    return load_iris(as_frame=True).frame


@pytest.fixture
def reg_df():
    from sklearn.datasets import load_diabetes
    return load_diabetes(as_frame=True).frame


# ---------------- profiling ----------------

def test_profile_infers_classification(clf_df):
    prof = mp.profile(clf_df.drop(columns="target"), clf_df["target"])
    assert prof.task == "classification"
    assert prof.n_classes == 3
    assert prof.n_samples == 150
    assert prof.n_features == 4


def test_profile_infers_regression(reg_df):
    prof = mp.profile(reg_df.drop(columns="target"), reg_df["target"])
    assert prof.task == "regression"
    assert prof.n_classes is None


def test_profile_is_serialisable(clf_df):
    prof = mp.profile(clf_df.drop(columns="target"), clf_df["target"])
    d = prof.to_dict()
    import json
    json.dumps(d)  # must not raise
    assert d["task"] == "classification"


def test_missing_and_constant_detected():
    X = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "const": [7, 7, 7, 7, 7]})
    y = pd.Series([0, 1, 0, 1, 0])
    prof = mp.profile(X, y)
    assert prof.missing_fraction > 0
    assert prof.constant_feature_fraction > 0


# ---------------- patterns ----------------

def test_imbalance_pattern():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, weights=[0.95, 0.05], random_state=0)
    report = mp.analyze(pd.DataFrame(X), y=y, probe_nonlinearity=False)
    assert "IMBALANCED" in {p.code for p in report.patterns}


def test_high_dim_pattern():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 200))
    y = pd.Series(rng.randint(0, 2, 40))
    report = mp.analyze(X, y=y, probe_nonlinearity=False)
    assert "HIGH_DIM" in {p.code for p in report.patterns}


# ---------------- recommend ----------------

def test_recommendations_ranked_and_reasoned(reg_df):
    report = mp.analyze(reg_df, target="target", probe_nonlinearity=False)
    recs = report.recommend()
    assert len(recs) >= 3
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)  # best first
    assert all(hasattr(r, "reasons") for r in recs)


def test_high_dim_favours_regularized_linear():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(50, 150))
    y = pd.Series(rng.randint(0, 2, 50))
    report = mp.analyze(X, y=y, probe_nonlinearity=False)
    top_keys = [r.key for r in report.recommend(3)]
    assert "regularized_linear" in top_keys


# ---------------- benchmark ----------------

def test_benchmark_runs(clf_df):
    report = mp.analyze(clf_df, target="target", probe_nonlinearity=False)
    results = report.benchmark(top_k=3, cv=3)
    assert len(results) == 3
    ok = [r for r in results if r.error is None]
    assert ok and all(r.score == r.score for r in ok)  # not NaN


# ---------------- API / errors ----------------

def test_analyze_accepts_X_y(clf_df):
    report = mp.analyze(clf_df.drop(columns="target"), y=clf_df["target"])
    assert report.task == "classification"


def test_missing_target_raises(clf_df):
    with pytest.raises(KeyError):
        mp.analyze(clf_df, target="does_not_exist")


def test_no_target_raises(clf_df):
    with pytest.raises(ValueError):
        mp.analyze(clf_df)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mp.analyze(pd.DataFrame({"a": [1, 2, 3]}), y=[0, 1])


def test_report_to_dict_serialisable(clf_df):
    import json
    report = mp.analyze(clf_df, target="target", probe_nonlinearity=False)
    json.dumps(report.to_dict())
