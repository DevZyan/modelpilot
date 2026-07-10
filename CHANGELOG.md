# Changelog

## [0.1.0] - 2026-07-06
### Added
- `analyze()` one-call pipeline: profile -> patterns -> recommendations.
- `DatasetProfile` meta-feature extraction (shape, imbalance, missingness,
  dimensionality, collinearity, outliers, sparsity, target signal).
- Pattern detection with a light linear-vs-tree nonlinearity probe.
- Transparent, per-reason model recommendation scorer over 8 model families.
- `benchmark()` for cross-validated empirical validation of the shortlist.
- Test suite, CI, packaging (src layout, hatchling).
