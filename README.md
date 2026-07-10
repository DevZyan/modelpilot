# modelpilot

**Point it at a dataset. It profiles the data, spots the patterns, and tells you which model will perform best — with reasons, then proves it.**

Most tooling makes you choose: either a *heuristic* advisor you can't trust, or a
*brute-force* sweep (like `lazypredict`) that runs everything with zero guidance.
`modelpilot` does both, in order:

1. **Profile** — extract ML-relevant meta-features (shape, imbalance, missingness,
   dimensionality, collinearity, feature→target signal).
2. **Detect patterns** — turn those into named, explainable flags:
   `IMBALANCED`, `HIGH_DIM`, `NONLINEAR`, `MULTICOLLINEAR`, `HIGH_CARD_CAT`, …
3. **Recommend** — rank model families with a transparent, per-reason score.
4. **Benchmark** *(optional)* — cross-validate the shortlist with proper
   preprocessing and hand back real numbers.

No black box: every recommendation shows *why*.

## Install

```bash
pip install modelpilot            # core (numpy, pandas, scikit-learn)
pip install "modelpilot[boost]"   # + xgboost / lightgbm
```

## Quickstart

```python
import modelpilot as mp
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True).frame
report = mp.analyze(df, target="target")

report.summary()          # EDA + patterns + ranked recommendations
report.benchmark()        # cross-validate the top picks for real scores
```

```
==============================================================
  modelpilot report
==============================================================
  Task            : regression
  Shape           : 442 rows x 10 features (10 numeric, 0 categorical)
  Missing         : 0.0%
  Target signal   : 0.41

  Patterns detected:
     · Linear-friendly — a linear model matches a tree; simple is strong.

  Recommended models:
    1. Regularized Linear (Ridge/Lasso/ElasticNet)  (score +2.5)
         +1.5  data looks linearly separable
    2. Gradient Boosting (HistGradientBoosting/XGBoost/LightGBM)
==============================================================
```

## API

```python
report = mp.analyze(data, target="col")     # or: mp.analyze(X, y=y)
report.profile          # DatasetProfile (all meta-features, .to_dict())
report.patterns         # list[Pattern] with codes, severities, explanations
report.recommend(3)     # top-3 recommendations, each with .reasons
report.benchmark(5)     # cross-validated leaderboard
report.to_dict()        # everything, JSON-serialisable
```

Each layer is usable on its own: `mp.profile(X, y)`, `mp.detect(profile)`,
`mp.recommend(profile, patterns)`, `mp.benchmark(...)`.

## How the recommendation works

Recommendations start from tabular-ML priors (gradient boosting and random
forests are strong defaults) and are then nudged up or down by the detected
patterns — e.g. `HIGH_DIM` boosts regularized linear models and penalizes
distance/kernel methods; `NONLINEAR` boosts trees and penalizes plain linear
models. Every nudge is recorded as a reason string, so the score is fully
auditable. `benchmark()` then validates the shortlist empirically so you never
have to take the heuristic on faith.

## Roadmap

- [ ] HTML report export
- [ ] Pluggable custom model families
- [ ] Meta-learned scoring (train the nudge weights on OpenML datasets)
- [ ] Text / datetime feature detection
- [ ] `report.leaderboard()` combining heuristic rank + benchmark score

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
