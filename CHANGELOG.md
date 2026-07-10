# Changelog

## [Unreleased]
### Added
- `report.leaderboard()` / `leaderboard_summary()` — combine the heuristic
  ranking with real cross-validated scores in one table.
- `report.to_html()` / `save_html()` — self-contained, theme-aware HTML report.
- Text and datetime feature detection (`HAS_TEXT`, `HAS_DATETIME` patterns;
  `profile.text_cols`, `profile.datetime_cols`).
- Pluggable model families via `register_model_family()` /
  `unregister_model_family()` / `list_model_families()`, with optional custom
  scoring rules.
- `modelpilot.meta` — meta-learn per-rule scoring weights from cross-validated
  outcomes (`meta.fit`/`save`/`load`); apply via `analyze(..., weights=...)`.
- `benchmark()` now feeds datetime (year/month/day/weekday/hour) and free-text
  (TF-IDF) columns into the models instead of dropping them.
- `report.explain()` — fit the recommended model and rank the features driving
  it (impurity / coefficient / permutation), with real feature names.
- Bundled, calibrated default rule weights applied automatically by `analyze()`
  (`meta.default_weights()`; opt out with `analyze(weights=False)`).
- `modelpilot` command-line interface: `modelpilot data.csv --target col`
  with `--benchmark`, `--explain`, `--html`, `--json` options.
### Changed
- Scoring rules now carry stable `rule_id`s and per-rule `contributions`,
  enabling weight calibration without changing rule code.
- Meta weight multipliers are clamped non-negative so calibration only scales a
  rule's magnitude, never flips its designed direction.

## [0.1.0] - 2026-07-06
### Added
- `analyze()` one-call pipeline: profile -> patterns -> recommendations.
- `DatasetProfile` meta-feature extraction (shape, imbalance, missingness,
  dimensionality, collinearity, outliers, sparsity, target signal).
- Pattern detection with a light linear-vs-tree nonlinearity probe.
- Transparent, per-reason model recommendation scorer over 8 model families.
- `benchmark()` for cross-validated empirical validation of the shortlist.
- Test suite, CI, packaging (src layout, hatchling).
