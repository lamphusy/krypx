# KrypX

KrypX is a reproducible research pipeline for testing long-or-cash cryptocurrency
trading signals without look-ahead leakage. Phase 1 targets BTC/USDT hourly candles,
next-open execution, fixed holding periods, and an XGBoost classifier.

The implementation is intentionally milestone-driven. Milestones 1 through 3 provide the data
and evaluation-boundary foundation: reproducible market-data snapshots, trailing-only technical
features, executable next-open labels, multiplicative cost thresholds, isolated holdout data,
and purged walk-forward folds. Model training and backtesting remain future milestones.

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

## Prepare features and labels

After fetching market data, build the inference-ready and labeled datasets:

```bash
krypx prepare
python scripts/run_pipeline.py prepare
```

Symbol and timeframe overrides must identify an existing fetched dataset:

```bash
krypx prepare --symbol ETH/USDT --timeframe 4h
```

Preparation resolves the mutable latest file to its matching immutable SHA-256 snapshot and
verifies those exact bytes before computing anything. It then:

1. Computes 24 ordered, trailing-only trend, momentum, volatility, volume, and return features.
2. Removes only the leading indicator warm-up rows from the inference dataset.
3. Labels decisions using entry at `t + 1` open and exit at `t + horizon + 1` open.
4. Removes the final `horizon + 1` unrealizable rows only from the labeled training dataset.
5. Atomically writes:

```text
data/interim/{symbol_slug}_{timeframe}_features.csv
data/processed/{symbol_slug}_{timeframe}_labeled.csv
```

The inference file retains the latest complete feature row. Label files also retain entry and
exit timestamps and prices so future split and leakage checks can audit exactly which market
observations each target used.

## Plan chronological evaluation splits

Milestone 3 exposes a reusable split API for the later `validate` command:

```python
from pathlib import Path

from crypto_ai.features.dataset import load_labeled_dataset
from crypto_ai.modeling.splits import create_split_plan, save_split_metadata

labeled = load_labeled_dataset(Path("data/processed/btc_usdt_1h_labeled.csv"))
plan = create_split_plan(labeled)
save_split_metadata(plan, Path("data/processed/btc_usdt_1h_split_metadata.json"))
```

The holdout is ceiling-rounded to 20% before removing a separate `horizon + 1` boundary purge.
Walk-forward validation then operates only on development rows, with expanding training windows,
contiguous validation blocks, and another complete label-lookahead gap before every validation
block. Both holdout and fold boundaries are checked against the stored `exit_timestamp` values,
not merely their DataFrame positions.

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
