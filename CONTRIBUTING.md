# Contributing

Thanks for helping improve modelpilot!

## Dev setup
```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Guidelines
- Keep the core dependency-light (numpy/pandas/scikit-learn). Heavy libraries go
  behind optional extras.
- Every recommendation adjustment must carry a human-readable reason string.
- Add or update tests for any behavior change.
- Run `ruff` and `pytest` before opening a PR.

## Good first issues
- HTML report export (`report.to_html()`).
- Datetime / text feature detection in profiling.
- Additional model families behind the `[boost]` extra.
