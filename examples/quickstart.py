"""Minimal end-to-end example."""
import modelpilot as mp
from sklearn.datasets import load_wine

df = load_wine(as_frame=True).frame
report = mp.analyze(df, target="target")

report.summary()

print("\nCross-validated leaderboard:")
for r in report.benchmark(top_k=4):
    print("  ", r)
