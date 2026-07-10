"""modelpilot — profile a dataset, spot its patterns, and get told which model will win.

Quickstart
----------
>>> import modelpilot as mp
>>> report = mp.analyze(df, target="target")
>>> report.summary()          # EDA + patterns + ranked recommendations
>>> report.benchmark()        # cross-validate the shortlist for real numbers
"""
from __future__ import annotations

from .report import analyze, DatasetReport
from .profiling import profile, DatasetProfile
from .patterns import detect, Pattern
from .recommend import recommend, Recommendation
from .benchmark import benchmark, BenchmarkResult
from .catalog import CATALOG, ModelFamily

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "DatasetReport",
    "profile",
    "DatasetProfile",
    "detect",
    "Pattern",
    "recommend",
    "Recommendation",
    "benchmark",
    "BenchmarkResult",
    "CATALOG",
    "ModelFamily",
    "__version__",
]
