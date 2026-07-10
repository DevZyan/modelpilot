"""modelpilot — profile a dataset, spot its patterns, and get told which model will win.

Quickstart
----------
>>> import modelpilot as mp
>>> report = mp.analyze(df, target="target")
>>> report.summary()          # EDA + patterns + ranked recommendations
>>> report.benchmark()        # cross-validate the shortlist for real numbers
"""
from __future__ import annotations

from .report import analyze, DatasetReport, LeaderboardRow
from .profiling import profile, DatasetProfile
from .patterns import detect, Pattern
from .recommend import recommend, Recommendation
from .benchmark import benchmark, BenchmarkResult
from .explain import explain, Explanation
from . import meta
from .catalog import (
    CATALOG,
    ModelFamily,
    register_model_family,
    unregister_model_family,
    list_model_families,
)

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "DatasetReport",
    "LeaderboardRow",
    "profile",
    "DatasetProfile",
    "detect",
    "Pattern",
    "recommend",
    "Recommendation",
    "benchmark",
    "BenchmarkResult",
    "explain",
    "Explanation",
    "meta",
    "CATALOG",
    "ModelFamily",
    "register_model_family",
    "unregister_model_family",
    "list_model_families",
    "__version__",
]
