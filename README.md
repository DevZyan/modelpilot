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
report.benchmark(5)     # cross-validated scores for the shortlist
report.leaderboard(5)   # heuristic rank + measured score, side by side
report.to_html()        # self-contained HTML report (also .save_html(path))
report.to_dict()        # everything, JSON-serialisable
```

Each layer is usable on its own: `mp.profile(X, y)`, `mp.detect(profile)`,
`mp.recommend(profile, patterns)`, `mp.benchmark(...)`.

### Plug in your own model families

```python
from sklearn.ensemble import ExtraTreesClassifier

fam = mp.ModelFamily("extra_trees", "Extra Trees", ("classification",),
                     "Randomised trees.", frozenset({"tree", "ensemble"}))

def score(family, profile, pattern_codes):
    return [(1.0, "extra randomisation helps on noisy data")] if "OUTLIER_HEAVY" in pattern_codes else []

mp.register_model_family(fam, ExtraTreesClassifier, scorer=score)
# now it competes in analyze()/recommend()/benchmark() like any built-in
```

### Calibrate the scorer on data (meta-learning)

```python
weights = mp.meta.fit(mp.meta.builtin_datasets())   # or your own [(X, y), ...]
mp.meta.save(weights, "weights.json")

report = mp.analyze(df, target="target", weights=weights)  # data-tuned rules
```

`meta.fit` learns a multiplier per scoring rule from cross-validated outcomes,
so the advisor is calibrated empirically while every rule keeps its reason.

## How the recommendation works

Recommendations start from tabular-ML priors (gradient boosting and random
forests are strong defaults) and are then nudged up or down by the detected
patterns — e.g. `HIGH_DIM` boosts regularized linear models and penalizes
distance/kernel methods; `NONLINEAR` boosts trees and penalizes plain linear
models. Every nudge is recorded as a reason string, so the score is fully
auditable. `benchmark()` then validates the shortlist empirically so you never
have to take the heuristic on faith.

## Roadmap

- [x] HTML report export — `report.to_html()` / `report.save_html(path)`
- [x] Pluggable custom model families — `mp.register_model_family(...)`
- [x] Meta-learned scoring — `mp.meta.fit(...)` learns per-rule weights
- [x] Text / datetime feature detection — `HAS_TEXT` / `HAS_DATETIME` patterns
- [x] `report.leaderboard()` combining heuristic rank + benchmark score
- [ ] Native categorical/text handling inside `benchmark()` (TF-IDF, target enc.)
- [ ] Ship a pre-trained default weight set fit on OpenML

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
