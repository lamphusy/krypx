# KrypX

KrypX is a reproducible research pipeline for testing long-or-cash cryptocurrency
trading signals without look-ahead leakage. Phase 1 targets BTC/USDT hourly candles,
next-open execution, fixed holding periods, and an XGBoost classifier.

The implementation is intentionally milestone-driven. Milestone 1 provides the market-data
foundation: paginated public CCXT downloads, transient-network retries, strict candle
validation, incremental updates, atomic latest-file replacement, and immutable
content-addressed raw snapshots. Feature engineering, training, and backtesting remain future
milestones.

## Fetch market data

Fetch the default BTC/USDT hourly history with either entry point:

```bash
krypx fetch
python scripts/run_pipeline.py fetch
```

Override the defaults when needed:

```bash
krypx fetch --symbol ETH/USDT --timeframe 4h --lookback-days 1095
```

The command loads any valid local history, resumes at the next expected candle, excludes the
current incomplete candle, and rejects duplicate, missing, off-grid, non-UTC, or invalid OHLCV
records. A successful update writes:

```text
data/raw/{symbol_slug}_{timeframe}.csv
data/raw/snapshots/{symbol_slug}_{timeframe}/{sha256}.csv
```

The first file is the atomically replaced latest-data convenience copy. The second is the exact
immutable input that downstream research must reference. The command prints both paths and the
snapshot SHA-256 digest.

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
