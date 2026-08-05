# KrypX

KrypX is a reproducible research pipeline for testing long-or-cash cryptocurrency
trading signals without look-ahead leakage. Phase 1 targets BTC/USDT hourly candles,
next-open execution, fixed holding periods, and an XGBoost classifier.

The implementation is intentionally milestone-driven. The current code contains the
Milestone 0 repository bootstrap only; market-data fetching, feature engineering,
training, and backtesting are not implemented yet.

## Development setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

Run the required checks:

```bash
.venv/bin/black --check .
.venv/bin/ruff check .
.venv/bin/pytest
```

Project behavior and research invariants are defined in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).


TEst