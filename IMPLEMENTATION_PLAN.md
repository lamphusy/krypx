# Crypto AI Trading Signal System

## Implementation Plan and Technical Specification

**Version:** 2.1.0
**Last updated:** 2026-08-02
**Current implementation scope:** Phase 1 — Baseline Research Pipeline
**Primary implementation assistant:** Codex
**Author:** Sy Lam

**Version 2.1 corrections:** Added holdout-boundary purging and label timestamp
provenance; separated decision rows from execution-price context; fixed full-equity
position sizing, state-machine ordering, and metric definitions; aligned label and
backtest cost math; moved configuration inside the package; and required immutable
content-addressed data snapshots.

---

# 0. Instructions for Codex

This document is the single source of truth for the project.

Before implementing or modifying any part of the project:

1. Read this document completely.
2. Read the tests related to the module being changed.
3. Follow the data contracts, execution assumptions, naming conventions, and module boundaries defined here.
4. Do not introduce new libraries, architectural patterns, model types, or abstractions unless explicitly requested by the human developer.
5. Do not weaken, delete, or bypass tests to make the code pass.
6. Never use future information when computing features, creating validation splits, or generating trading signals.
7. Never evaluate the final holdout set during routine development.
8. Prefer correctness and clarity over premature optimization.
9. After every implementation task:

   * Run formatting.
   * Run linting.
   * Run the relevant tests.
   * Report files changed.
   * Report commands executed.
   * Report remaining limitations.

If this document conflicts with an implementation detail in the existing repository, this document takes precedence unless the human developer explicitly says otherwise.

---

# 1. Project Overview

## 1.1 Goal

Build a modular and reproducible Crypto AI Trading Signal System that:

1. Fetches historical OHLCV market data from Binance through CCXT.
2. Stores and incrementally updates clean market data.
3. Computes technical features using only information available at each decision timestamp.
4. Creates an executable, cost-aware classification target.
5. Trains an XGBoost classifier.
6. Evaluates the model through purged walk-forward validation.
7. Performs one final backtest on an untouched chronological holdout period.
8. Compares the model against simple trading and machine-learning baselines.
9. Saves versioned models, feature schemas, metrics, predictions, and run metadata.
10. Later serves live signals through an API.

The system produces trading signals only. It does not automatically execute orders.

---

## 1.2 Phase 1 Objective

Phase 1 must produce a statistically valid research pipeline for one market configuration:

```text
Symbol: BTC/USDT
Timeframe: 1h
Direction: Long or cash
Model: XGBoost binary classifier
Signal values: BUY or STAY_OUT
Execution: Enter at the next candle open
Holding period: Fixed number of candles
Default prediction horizon: 4 candles
```

Phase 1 is intended to answer:

> Does the model provide repeatable out-of-sample value after realistic transaction costs compared with simple baselines?

Completing the pipeline is not evidence that the strategy is profitable.

---

## 1.3 Future Phases

### Phase 2 — Alternative Features

Possible additions:

* Point-in-time news sentiment.
* On-chain data.
* Funding rates.
* Open interest.
* Liquidation data.
* Cross-asset features.
* Market-regime features.

These features must not be added until the Phase 1 baseline has been evaluated and frozen.

### Phase 3 — Signal API

* FastAPI service.
* Scheduled market-data refresh.
* Scheduled signal generation.
* Separate model-training worker.
* Model registry and controlled model activation.

### Phase 4 — Dashboard

* Current signal.
* Signal history.
* Model metadata.
* Equity curve.
* Drawdown chart.
* Strategy metrics.
* Data freshness and health indicators.

---

## 1.4 Non-Goals for Phase 1

Phase 1 does not include:

* Automated order execution.
* Portfolio allocation across multiple assets.
* Leverage or margin trading.
* Short selling.
* Futures or perpetual contracts.
* High-frequency or sub-minute trading.
* Multi-exchange arbitrage.
* Deep-learning models.
* LLM sentiment.
* Hyperparameter optimization.
* Real-time streaming infrastructure.
* Mobile or web dashboards.
* Claims of guaranteed profitability.

---

# 2. Core Research Principles

## 2.1 Chronological Integrity

All training, validation, and testing operations must preserve chronological order.

Data must never be shuffled.

For every split:

```text
Training data occurs first.
A purge gap follows training data.
Validation or test data occurs strictly afterward.
```

---

## 2.2 Point-in-Time Correctness

A feature for decision row $t$ may use only information available at or before the close of candle $t$.

It may not use:

* The open, high, low, close, or volume of candle (t+1).
* A centered rolling window.
* A backward fill from future values.
* A future news article.
* A scaler fitted on future observations.
* A label-derived value.
* A statistic calculated from the complete dataset.

---

## 2.3 Executable Assumptions

The system observes the completed candle at time $t$.

It cannot use the close of candle $t$ and assume execution at that same close.

The Phase 1 execution policy is:

```text
Decision time:
Immediately after candle t closes.

Signal:
Generated using features from candle t and earlier.

Entry:
Open of candle t+1.

Holding period:
Exactly H complete candles.

Exit:
Open of candle t+H+1.
```

For the default horizon (H=4):

```text
Decision after candle t closes.
Enter at open t+1.
Hold candles t+1, t+2, t+3, and t+4.
Exit at open t+5.
```

---

## 2.4 Untouched Final Holdout

The final chronological holdout must be selected before model evaluation.

The holdout may not be used for:

* Feature selection.
* Hyperparameter selection.
* Signal-threshold selection.
* Debugging model behavior.
* Repeated backtests.
* Choosing transaction-cost assumptions.
* Choosing the prediction horizon.
* Comparing alternative model configurations.

The final holdout should be evaluated only after the development pipeline and configuration are frozen.

---

## 2.5 Evaluation and Production Models Are Different

Two separate final models must exist.

### Evaluation model

Trained only on the development period.

Used to produce predictions for the untouched holdout.

### Production model

Trained after the evaluation report is finalized.

May use all available labeled data.

Used for later live inference.

The production model must never be used to claim holdout performance because it has been trained on holdout labels.

---

# 3. Precise Problem Definition

## 3.1 Decision Row

A row at timestamp $t$ represents information available after candle $t$ has fully closed.

The timestamp column refers to the candle open time, following the exchange OHLCV convention.

---

## 3.2 Model Input

The input vector is:

$$
\mathbf{x}_t \in \mathbb{R}^{d}
$$

where $d$ is the number of selected feature columns.

Every feature in $\mathbf{x}_t$ must be computable from rows whose timestamps are less than or equal to $t$.

---

## 3.3 Executable Forward Return

Let:

* $O_{t+1}$ is the open price of the next candle.
* $O_{t+H+1}$ is the open price after holding for $H$ candles.

The gross executable forward return is:

$$
r_t^{(H)} = \frac{O_{t+H+1}}{O_{t+1}} - 1
$$

This return definition must be shared by:

* Label generation.
* Backtesting.
* Tests.
* Documentation.
* Metric interpretation.

---

## 3.4 Cost-Aware Binary Label

The label is:

$$
y_t =
\begin{cases}
1, & r_t^{(H)} > \tau \\
0, & r_t^{(H)} \leq \tau
\end{cases}
$$

where:

* $y_t=1$ means BUY.
* $y_t=0$ means STAY_OUT.
* $\tau$ is the minimum required gross return.

The minimum required return accounts for estimated trading costs and a configurable safety margin.

Use the shared cost helper defined in Section 6.3:

```python
MIN_REQUIRED_RETURN = minimum_gross_return_for_net_edge(
    fee_rate=TAKER_FEE_RATE,
    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    half_spread_bps_per_side=HALF_SPREAD_BPS_PER_SIDE,
    minimum_net_edge_bps=MIN_EDGE_BPS,
)
```

The exact values are assumptions and must be stored in the run metadata.

The label distribution is not required to be 50/50.

---

## 3.5 Signal Rule

The model produces:

$$
p_t = P(y_t=1 \mid \mathbf{x}_t)
$$

The Phase 1 signal rule is:

$$
\text{signal}_t =
\begin{cases}
\text{BUY}, & p_t \geq \theta \\
\text{STAY_OUT}, & p_t < \theta
\end{cases}
$$

The initial Phase 1 threshold is fixed in configuration:

```python
SIGNAL_THRESHOLD = 0.50
```

It must not be optimized using the final holdout.

Raw XGBoost probability output must be described as a `probability_score`, not guaranteed statistical confidence.

Probability calibration is outside Phase 1.

---

## 3.6 Position Policy

Phase 1 supports only one open position at a time.

Rules:

1. Start in cash.
2. Generate a signal after every completed candle.
3. When the signal is BUY and no position is open:

   * Enter at the next candle open.
4. Hold for exactly `PREDICTION_HORIZON` candles.
5. Exit at the scheduled exit open.
6. Ignore new BUY signals while a position is already open.
7. Do not open overlapping trades.
8. Do not short when the signal is STAY_OUT.
9. Accept a trade only when its scheduled exit is within the evaluated price context.
10. Do not evaluate decision rows without a valid future exit price.
11. Invest 100% of currently available equity on each entry.
12. Permit fractional asset quantities.
13. Do not use leverage, borrowed capital, or partial position sizing.
14. Cash earns no yield during Phase 1.

The quantity purchased at entry is:

```python
investable_capital = equity_before_entry * (1.0 - fee_rate)
position_quantity = investable_capital / entry_fill_price
```

The implementation may use an algebraically equivalent cash-flow representation,
but it must reconcile exactly with the net-growth-factor formula in Section 13.3.

This stateful logic may use a clear chronological loop. Vectorization is not required.

Correctness is more important than avoiding loops.

---

# 4. System Architecture

```text
┌───────────────────────────────────────────────────────────┐
│                     Market Data Layer                     │
│                                                           │
│ Binance public API through CCXT                           │
│        │                                                  │
│        ▼                                                  │
│ Incremental OHLCV update                                  │
│        │                                                  │
│        ▼                                                  │
│ Raw-data validation                                       │
│        │                                                  │
│        ▼                                                  │
│ Immutable raw snapshot plus latest-data convenience file  │
└──────────────────────────────┬────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────┐
│                    Feature and Label Layer                │
│                                                           │
│ compute_features()                                        │
│        │                                                  │
│        ├── Training path → add_labels()                   │
│        │                                                  │
│        └── Inference path → latest complete feature row   │
└──────────────────────────────┬────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────┐
│                       Research Layer                      │
│                                                           │
│ Chronological development/holdout split                   │
│        │                                                  │
│ Boundary purge covering complete label lookahead          │
│        │                                                  │
│ Purged walk-forward validation on development data        │
│        │                                                  │
│ Out-of-fold predictions and classification metrics        │
│        │                                                  │
│ Train evaluation model on boundary-purged development     │
└──────────────────────────────┬────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────┐
│                       Backtest Layer                      │
│                                                           │
│ Evaluation-model predictions on untouched holdout         │
│        │                                                  │
│ Next-open execution                                       │
│        │                                                  │
│ Fixed holding period                                      │
│        │                                                  │
│ Fees, spread, and slippage                                │
│        │                                                  │
│ Strategy and baseline metrics                             │
└──────────────────────────────┬────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────┐
│                       Artifact Layer                      │
│                                                           │
│ Evaluation model                                          │
│ Production model                                          │
│ Feature schema                                            │
│ Predictions                                               │
│ Trade ledger                                              │
│ Equity curve                                              │
│ Metrics                                                   │
│ Run manifest                                              │
└───────────────────────────────────────────────────────────┘
```

---

# 5. Repository Structure

Use a Python `src` package layout.

```text
crypto-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── artifacts/
│   ├── evaluations/
│   ├── production/
│   └── runs/
│
├── src/
│   └── crypto_ai/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetch.py
│       │   ├── storage.py
│       │   └── validation.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── build.py
│       │   └── labels.py
│       │
│       ├── modeling/
│       │   ├── __init__.py
│       │   ├── splits.py
│       │   ├── baselines.py
│       │   ├── train.py
│       │   └── metrics.py
│       │
│       ├── backtesting/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── metrics.py
│       │
│       ├── artifacts/
│       │   ├── __init__.py
│       │   ├── manifest.py
│       │   └── registry.py
│       │
│       ├── costs.py
│       ├── cli.py
│       └── logging_config.py
│
├── scripts/
│   └── run_pipeline.py
│
├── tests/
│   ├── conftest.py
│   │
│   ├── data/
│   │   ├── test_fetch.py
│   │   ├── test_storage.py
│   │   └── test_validation.py
│   │
│   ├── features/
│   │   ├── test_build.py
│   │   └── test_labels.py
│   │
│   ├── modeling/
│   │   ├── test_splits.py
│   │   ├── test_baselines.py
│   │   └── test_train.py
│   │
│   ├── backtesting/
│   │   ├── test_engine.py
│   │   └── test_metrics.py
│   │
│   ├── test_costs.py
│   │
│   └── integration/
│       └── test_pipeline.py
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── .env.example
├── .gitignore
├── README.md
└── IMPLEMENTATION_PLAN.md
```

---

# 6. Configuration

## 6.1 General Rule

All configurable values must be defined in `src/crypto_ai/config/settings.py`.

Functions must accept settings or explicit parameters where practical. Business logic must not contain hidden magic numbers.

CLI arguments may override configuration values for a specific run.

---

## 6.2 Settings Specification

```python
# src/crypto_ai/config/settings.py

from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_SNAPSHOTS_DIR = DATA_RAW_DIR / "snapshots"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVALUATIONS_DIR = ARTIFACTS_DIR / "evaluations"
PRODUCTION_DIR = ARTIFACTS_DIR / "production"
RUNS_DIR = ARTIFACTS_DIR / "runs"


# ==========================================================
# Market data
# ==========================================================

EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

LOOKBACK_DAYS = 3 * 365
CANDLES_PER_REQUEST = 1000

MAX_FETCH_RETRIES = 3
RETRY_BASE_SECONDS = 2.0

DROP_INCOMPLETE_LAST_CANDLE = True


# ==========================================================
# Feature engineering
# ==========================================================

PREDICTION_HORIZON = 4

EMA_SHORT = 9
EMA_LONG = 21

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14
STOCH_RSI_PERIOD = 14

BB_PERIOD = 20
BB_STD_DEV = 2.0

ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20

RETURN_PERIODS = [1, 2, 3, 6, 12, 24]


# ==========================================================
# Label
# ==========================================================

TAKER_FEE_RATE = 0.001
SLIPPAGE_BPS_PER_SIDE = 2.0
HALF_SPREAD_BPS_PER_SIDE = 1.0
MIN_EDGE_BPS = 5.0


# ==========================================================
# Dataset splitting
# ==========================================================

FINAL_HOLDOUT_RATIO = 0.20

N_WALK_FORWARD_SPLITS = 5
WALK_FORWARD_TEST_RATIO = 0.10

# Label uses open at t+1 and open at t+H+1.
PURGE_GAP_ROWS = PREDICTION_HORIZON + 1


# ==========================================================
# Reproducibility
# ==========================================================

RANDOM_SEED = 42


# ==========================================================
# Model
# ==========================================================

SIGNAL_THRESHOLD = 0.50

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}


# ==========================================================
# Backtesting
# ==========================================================

INITIAL_CAPITAL = 10_000.0
# Annual rate used only for risk-adjusted reporting. Cash earns no yield.
ANNUAL_RISK_FREE_RATE = 0.0

RANDOM_BASELINE_SIMULATIONS = 1_000

COST_SCENARIOS = {
    "low": {
        "fee_rate": TAKER_FEE_RATE,
        "slippage_bps_per_side": 1.0,
        "half_spread_bps_per_side": 0.5,
    },
    "base": {
        "fee_rate": TAKER_FEE_RATE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "half_spread_bps_per_side": HALF_SPREAD_BPS_PER_SIDE,
    },
    "high": {
        "fee_rate": TAKER_FEE_RATE,
        "slippage_bps_per_side": 5.0,
        "half_spread_bps_per_side": 2.0,
    },
}

# ==========================================================
# Columns
# ==========================================================

RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

LABEL_COLUMNS = [
    "entry_timestamp",
    "exit_timestamp",
    "entry_open",
    "exit_open",
    "gross_forward_return",
    "label",
]

NON_FEATURE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    *LABEL_COLUMNS,
]
```

---

## 6.3 Cost Calculations

Use helper functions instead of duplicating cost formulas.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """Per-side fee and adverse execution assumptions."""

    fee_rate: float
    slippage_bps_per_side: float
    half_spread_bps_per_side: float


def bps_to_rate(bps: float) -> float:
    """Convert basis points to a decimal rate."""
    return bps / 10_000.0


def minimum_gross_return_for_net_edge(
    fee_rate: float,
    slippage_bps_per_side: float,
    half_spread_bps_per_side: float,
    minimum_net_edge_bps: float,
) -> float:
    """Return the gross market return required to retain a net edge."""
    one_side_execution_rate = bps_to_rate(
        slippage_bps_per_side + half_spread_bps_per_side
    )
    minimum_net_edge_rate = bps_to_rate(minimum_net_edge_bps)
    return (
        (1.0 + minimum_net_edge_rate)
        * (1.0 + one_side_execution_rate)
        / (
            (1.0 - one_side_execution_rate)
            * (1.0 - fee_rate) ** 2
        )
        - 1.0
    )
```

Validate that fee and execution rates are non-negative and strictly below `1.0`.
`CostConfig` and these helpers belong in `crypto_ai/costs.py` and are shared by
label generation and backtesting.

Estimated one-side non-fee execution cost:

```python
one_side_execution_rate = (
    bps_to_rate(SLIPPAGE_BPS_PER_SIDE)
    + bps_to_rate(HALF_SPREAD_BPS_PER_SIDE)
)
```

Estimated one-side total cost:

```python
one_side_total_cost = (
    TAKER_FEE_RATE
    + one_side_execution_rate
)
```

Estimated round-trip cost:

```python
round_trip_cost_rate = 2.0 * one_side_total_cost
```

`round_trip_cost_rate` is an additive reporting approximation only. It must not be
used to construct the label threshold.

Minimum required gross market return must use the same multiplicative execution
model as the backtest:

```python
minimum_net_edge_rate = bps_to_rate(MIN_EDGE_BPS)

minimum_required_return = (
    (1.0 + minimum_net_edge_rate)
    * (1.0 + one_side_execution_rate)
    / (
        (1.0 - one_side_execution_rate)
        * (1.0 - TAKER_FEE_RATE) ** 2
    )
    - 1.0
)
```

This is the minimum raw open-to-open market return expected to leave the configured
net edge after adverse entry and exit fills and both fees. These are modeling
assumptions, not guaranteed real transaction costs.

---

# 7. Data Layer

# 7.1 `data/fetch.py`

## Responsibility

Fetch historical OHLCV candles from Binance through CCXT.

Support:

* Initial historical download.
* Incremental updates.
* Pagination.
* Retry handling.
* Rate limiting.
* Exclusion of incomplete candles.
* Duplicate removal.
* Chronological sorting.

---

## Main Function

```python
def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = CANDLES_PER_REQUEST,
) -> pd.DataFrame:
    """Fetch closed OHLCV candles from the configured exchange.

    Args:
        symbol: CCXT trading pair, such as "BTC/USDT".
        timeframe: Candle timeframe, such as "1h".
        since_ms: Inclusive start timestamp in milliseconds.
        until_ms: Optional exclusive end timestamp in milliseconds.
        limit: Maximum records requested per API call.

    Returns:
        Chronologically sorted OHLCV DataFrame with UTC timestamps.

    Raises:
        MarketDataNetworkError: If network retries are exhausted.
        MarketDataExchangeError: If the exchange rejects the request.
        MarketDataValidationError: If returned data is invalid.
    """
```

---

## Exchange Configuration

Use:

```python
exchange_class = getattr(ccxt, EXCHANGE_ID)

exchange = exchange_class({
    "enableRateLimit": True,
})
```

No API key is required for public Binance OHLCV data.

Do not manually sleep after every request when CCXT rate limiting is enabled, except as part of retry backoff.

---

## Pagination Behavior

For every request:

1. Request data beginning at `since_ms`.
2. Convert returned rows to the required schema.
3. Append rows.
4. Set the next `since_ms` to:

```python
last_returned_timestamp_ms + timeframe_duration_ms
```

5. Stop when:

   * No rows are returned.
   * `until_ms` has been reached.
   * The exchange returns no newer timestamps.
   * The requested historical range is complete.

Protect against infinite pagination loops.

---

## Retry Behavior

Retry only transient network-related errors.

Use exponential backoff:

```text
Attempt 1: wait 2 seconds
Attempt 2: wait 4 seconds
Attempt 3: wait 8 seconds
```

Do not silently return partial data after retries are exhausted.

Raise a project-specific exception.

---

## Incomplete Candle Removal

The most recent exchange candle may still be open.

Calculate the expected close time:

```python
candle_close_time = candle_open_time + timeframe_duration
```

Keep only candles satisfying:

```python
candle_close_time <= current_utc_time
```

All training and inference data must use completed candles only.

Pass an explicit timezone-aware `current_utc_time` into the candle-filtering helper.
Production callers supply the real current time; tests supply a fixed time. Do not
read the clock separately for every row.

---

# 7.2 `data/storage.py`

## Responsibilities

* Generate safe filenames.
* Load existing market data.
* Merge incremental data.
* Remove duplicates.
* Save files atomically.
* Persist immutable content-addressed snapshots.
* Return the exact snapshot identity used by downstream steps.
* Preserve UTC timestamps.

---

## Symbol Slug

```python
def symbol_to_slug(symbol: str) -> str:
    """Convert an exchange symbol to a safe filename component."""
    return (
        symbol.lower()
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
```

Example:

```text
BTC/USDT → btc_usdt
```

---

## Raw Data Path

```python
def get_raw_data_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """Return the mutable latest-data convenience path."""
    slug = symbol_to_slug(symbol)
    return DATA_RAW_DIR / f"{slug}_{timeframe}.csv"
```

This path is a convenience pointer containing the latest successfully validated
dataset. It is not the immutable research input.

Every successful update must also store the exact finalized CSV bytes under a
content-addressed snapshot path:

```text
data/raw/snapshots/{symbol_slug}_{timeframe}/{sha256}.csv
```

If a snapshot with that hash already exists, reuse it. Never modify or overwrite an
existing snapshot. Every training or evaluation manifest must reference the snapshot
path and SHA-256 hash actually used. Keeping only a hash of the mutable latest-data
file is insufficient for reproducibility.

---

## Incremental Update Function

```python
@dataclass(frozen=True)
class MarketDataResult:
    """Validated market data and its persisted snapshot identity."""

    data: pd.DataFrame
    latest_path: Path
    snapshot_path: Path
    sha256: str


def load_or_update_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_days: int,
) -> MarketDataResult:
    """Update OHLCV and return data with its immutable snapshot identity."""
```

Behavior:

### When no local file exists

1. Determine `since_ms` using `lookback_days`.
2. Fetch the complete available range.
3. Validate it.
4. Save it atomically.
5. Return the validated data and snapshot identity.

### When a local file exists

1. Load the existing file.
2. Parse timestamps as timezone-aware UTC.
3. Find the latest stored timestamp.
4. Fetch beginning at the next expected candle.
5. Merge old and new rows.
6. Remove duplicates by timestamp.
7. Sort chronologically.
8. Remove incomplete candles.
9. Validate the result.
10. Save atomically.
11. Return the validated data and snapshot identity.

Do not skip fetching merely because the local file is “recent.”

---

## Atomic Writes

Write to a temporary path first:

```text
btc_usdt_1h.csv.tmp
```

Then replace the final file only after a successful write.

A failed write must not corrupt the previous valid dataset.

Hash the finalized temporary file, persist its immutable snapshot, and only then
replace the latest-data convenience file. Tests must verify that an earlier snapshot
remains byte-for-byte unchanged after an incremental update.

---

# 7.3 `data/validation.py`

## Required Raw-Data Checks

Validation must verify:

1. Required columns are present.
2. The DataFrame is not empty.
3. Timestamps are timezone-aware UTC.
4. Timestamps are strictly increasing after sorting.
5. Duplicate timestamps do not exist.
6. All prices and volumes are finite.
7. Prices are positive.
8. Volume is non-negative.
9. `high >= low`.
10. `low <= open <= high`.
11. `low <= close <= high`.
12. Candle intervals match the configured timeframe.
13. Candle timestamps align to the exchange timeframe grid; for `1h`, timestamps
    must fall exactly on the UTC hour.
14. Missing intervals are reported.
15. The final candle is closed.
16. Numeric price columns use `float64`.

---

## Missing Candle Policy

For Phase 1:

* Missing candles must produce a clear validation error.
* Do not automatically forward-fill OHLCV values.
* Do not fabricate missing candles.
* Do not silently continue with irregular intervals.

A later phase may introduce an explicit exchange-gap policy.

---

# 8. Feature Engineering

# 8.1 Separation of Features and Labels

Feature computation and label creation must be separate.

```text
Training:
Raw OHLCV
    ↓
compute_features()
    ↓
add_labels()
    ↓
Labeled model dataset

Inference:
Raw OHLCV
    ↓
compute_features()
    ↓
Latest complete feature row
    ↓
Model prediction
```

`compute_features()` must not create or reference future columns.

`add_labels()` must not modify feature values.

---

# 8.2 `features/build.py`

## Main Function

```python
def compute_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute point-in-time technical features.

    Args:
        df: Valid chronological OHLCV data.

    Returns:
        DataFrame containing original market columns and feature columns.

    Raises:
        FeatureEngineeringError: If required input data is missing or invalid.
    """
```

---

## Required Features

### Trend Features

```text
ema_short
ema_long
ema_ratio
close_to_ema_short
close_to_ema_long
```

Definitions:

$$
\text{ema_ratio}_t =
\frac{\text{EMA}_{short,t}}{\text{EMA}_{long,t}} - 1
$$

$$
\text{close_to_ema_short}_t =
\frac{C_t}{\text{EMA}_{short,t}} - 1
$$

$$
\text{close_to_ema_long}_t =
\frac{C_t}{\text{EMA}_{long,t}} - 1
$$

---

### MACD Features

```text
macd
macd_signal
macd_diff
```

---

### Momentum Features

```text
rsi
stoch_rsi
```

---

### Volatility Features

```text
bb_width
bb_pct
atr
atr_pct
candle_range_pct
body_return
```

Definitions:

$$
\text{atr_pct}_t = \frac{\text{ATR}_t}{C_t}
$$

$$
\text{candle_range_pct}_t = \frac{H_t-L_t}{C_t}
$$

$$
\text{body_return}_t = \frac{C_t}{O_t}-1
$$

---

### Volume Features

```text
volume_change
volume_ma_ratio
```

Definitions:

$$
\text{volume_change}_t = \frac{V_t}{V_{t-1}}-1
$$

$$
\text{volume_ma_ratio}_t =
\frac{V_t}{\operatorname{MA}_{volume,t}} - 1
$$

---

### Lagged Return Features

For each configured period (n):

```text
return_1
return_2
return_3
return_6
return_12
return_24
```

Definition:

$$
\text{return}_{n,t} = \frac{C_t}{C_{t-n}}-1
$$

---

## Exact Indicator Contract

Use the pinned `ta` package version from `requirements-lock.txt` with
`fillna=False`. Use `EMAIndicator`, `MACD`, `RSIIndicator`,
`StochRSIIndicator`, `BollingerBands`, and `AverageTrueRange` with the configured
windows. Do not substitute pandas defaults or another indicator implementation
without changing the specification and re-freezing the development configuration.

`stoch_rsi` means the raw `StochRSIIndicator.stochrsi()` output, not its smoothed
`%K` or `%D` output. Define Bollinger features explicitly so library percentage
scaling cannot change their meaning:

```python
bb_width = (bb_upper - bb_lower) / bb_middle
bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
```

Use trailing rolling means with `min_periods` equal to the configured window. Use
explicit shift-based ratios for lagged returns and volume change; do not rely on an
implicit `pct_change()` fill policy.

---

## Feature Rules

1. Use trailing calculations only.
2. Never use centered windows.
3. Never use backward filling.
4. Do not fit scalers on the full dataset.
5. Do not use raw future prices.
6. Do not overwrite original OHLCV columns.
7. Replace positive and negative infinity with `NaN`.
8. Preserve timestamps.
9. Preserve chronological order.
10. Log the number of warm-up rows removed.

XGBoost does not require feature scaling for Phase 1.

---

## Missing Values

During training preparation:

* Remove only rows whose required features are incomplete.
* Removal is expected at the beginning due to rolling windows.
* Do not remove rows because future labels are missing inside `compute_features()`.

During live inference:

* The latest closed candle must have all required feature values.
* If it does not, raise a clear insufficient-history error.

---

# 8.3 `features/labels.py`

## Main Function

```python
def add_labels(
    feature_df: pd.DataFrame,
    horizon: int,
    minimum_required_return: float,
) -> pd.DataFrame:
    """Add executable forward-return columns and binary labels."""
```

---

## Label Columns

```python
result["entry_timestamp"] = result["timestamp"].shift(-1)

result["exit_timestamp"] = result["timestamp"].shift(
    -(horizon + 1)
)

result["entry_open"] = result["open"].shift(-1)

result["exit_open"] = result["open"].shift(
    -(horizon + 1)
)

result["gross_forward_return"] = (
    result["exit_open"]
    / result["entry_open"]
    - 1.0
)

result["label"] = (
    result["gross_forward_return"]
    > minimum_required_return
).astype("int8")
```

Rows without both entry and exit timestamps and prices must be removed from the
labeled training dataset. The timestamp provenance columns are required for leakage
audits and must never be model features.

Removing these rows from the labeled decision dataset must not remove them from the
raw or inference-ready feature dataset. The backtest needs those retained OHLCV rows
as execution-price context for otherwise valid tail decisions.

---

## Important Label Constraint

The final label lookahead is:

```python
LABEL_LOOKAHEAD_ROWS = horizon + 1
```

This value determines the minimum purge gap used between training and validation periods.

---

# 9. Dataset Splitting

# 9.1 Development and Final Holdout Split

The full labeled decision-row dataset must first be divided chronologically. The
holdout size is calculated before removing the boundary-purge rows:

```text
Earlier rows: Development candidates
Next H+1 rows: Boundary purge; never fitted
Last 20%: Final holdout decision rows
```

The boundary purge is mandatory because a development label at row `t` reads the
open at `t + H + 1`. Without it, the final development labels would contain prices
from the final holdout even though the fitted feature indexes appear disjoint.

Function:

```python
def split_development_holdout(
    df: pd.DataFrame,
    holdout_ratio: float,
    label_lookahead_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return development, boundary-purge, and untouched holdout rows."""
```

Required boundary calculation:

```python
holdout_size = max(1, math.ceil(len(df) * holdout_ratio))
holdout_start = len(df) - holdout_size
purge_start = holdout_start - label_lookahead_rows

if purge_start <= 0:
    raise DatasetSplitError("Insufficient rows for development, purge, and holdout")

development = df.iloc[:purge_start]
boundary_purge = df.iloc[purge_start:holdout_start]
holdout = df.iloc[holdout_start:]
```

Required invariants:

```python
assert (
    development["timestamp"].max()
    < boundary_purge["timestamp"].min()
    <= boundary_purge["timestamp"].max()
    < holdout["timestamp"].min()
)

assert len(boundary_purge) == label_lookahead_rows

assert (
    development["exit_timestamp"].max()
    < holdout["timestamp"].min()
)
```

The boundary calculation is positional and does not depend on DataFrame index-label
arithmetic. Preserve the input indexes in each returned partition and always verify
the actual `exit_timestamp` invariant directly.

The holdout start, boundary-purge range, row counts, and timestamps must be saved in
the run manifest.

---

## 9.2 Holdout Isolation

After the holdout is created:

* Walk-forward validation uses only the development period.
* Baseline selection uses only the development period.
* Model configuration uses only the development period.
* Debug plots use only the development period.
* Boundary-purge rows are not used for fitting or development metric calculation.
* The holdout file should not be inspected during routine model development.

The leakage test must inspect label provenance, not only the indexes passed to
`model.fit()`. For every fitted development row, `exit_timestamp` must be strictly
earlier than the first holdout timestamp.

---

# 9.3 Purged Walk-Forward Validation

## Main Function

```python
def walk_forward_splits(
    development_df: pd.DataFrame,
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window chronological splits with a purge gap."""
```

---

## Split Shape

```text
Fold 1:
[ TRAIN ][ GAP ][ TEST ]

Fold 2:
[       TRAIN       ][ GAP ][ TEST ]

Fold 3:
[             TRAIN             ][ GAP ][ TEST ]
```

Training expands over time.

The validation set always occurs in the future.

Successive validation blocks must be non-overlapping and contiguous so concatenated
out-of-fold predictions define one continuous development backtest window:

```python
assert max(test_indices_for_fold_k) + 1 == min(
    test_indices_for_fold_k_plus_1
)
```

Rows validated in earlier folds may enter the expanding training window of later
folds; they must never reappear as validation rows.

---

## Required Split Invariant

For every fold:

```python
last_train_position + gap_rows < first_test_position
```

More directly, labels from training rows must not use prices from the validation period:

```python
assert (
    max(train_indices)
    + PREDICTION_HORIZON
    + 1
    < min(test_indices)
)
```

---

## Test Size

Compute an integer validation size from the development dataset.

Example:

```python
test_size_rows = max(
    1,
    int(len(development_df) * WALK_FORWARD_TEST_RATIO),
)
```

The implementation must verify that enough observations exist for all folds, rolling-feature warm-up, and purge gaps.

---

# 10. Feature Schema

Feature columns must be derived at runtime immediately after `compute_features()`,
then checked against the exact expected schema. A denylist alone is insufficient
because an accidental helper, fold, or future-derived column could otherwise become
a model input silently.

```python
derived_feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

expected_feature_columns = get_expected_feature_columns(
    return_periods=RETURN_PERIODS,
)

if derived_feature_columns != expected_feature_columns:
    raise FeatureEngineeringError(
        "Derived feature columns do not match the ordered expected schema"
    )

feature_columns = expected_feature_columns
```

`get_expected_feature_columns()` belongs in `features/build.py` and returns the
ordered required feature names from Section 8, expanding `return_{n}` columns in
configured order. Training and inference both use it only for validation; inference
still loads the authoritative saved model schema.

Before training:

1. Verify that feature columns are not empty.
2. Verify that every feature is numeric.
3. Verify that feature names are unique.
4. Verify that no feature contains infinity.
5. Verify that the feature matrix contains no missing values.
6. Preserve the exact column order.

Save the ordered feature list with every model.

Inference must use exactly the saved feature order:

```python
inference_matrix = feature_df[
    saved_feature_columns
]
```

Do not rely on arbitrary DataFrame column order.

---

# 11. Model Training

# 11.1 Primary Model

Use:

```python
xgboost.XGBClassifier
```

Use only the fixed parameters defined in configuration during Phase 1.

Do not conduct hyperparameter optimization in Phase 1 unless the human developer explicitly changes the plan.

---

# 11.2 Walk-Forward Training

## Function

```python
def evaluate_walk_forward(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
) -> WalkForwardResult:
    """Evaluate a fixed XGBoost configuration using purged walk-forward splits."""
```

For every fold:

1. Create a new model instance.
2. Fit only on that fold’s training rows.
3. Predict probabilities for that fold’s validation rows.
4. Convert probabilities to signals using the fixed threshold.
5. Save fold predictions.
6. Compute classification metrics.
7. Save fold metadata.
8. Never reuse a fitted model across folds.

---

## Fold Output

For every validation row, save:

```text
timestamp
fold_number
actual_label
probability_score
predicted_label
signal
```

---

## Classification Metrics

Calculate per fold and across concatenated out-of-fold predictions:

* Accuracy.
* Balanced accuracy.
* Precision for class 1.
* Recall for class 1.
* F1 score for class 1.
* Precision for class 0.
* Recall for class 0.
* F1 score for class 0.
* Log loss.
* ROC-AUC, when both classes exist.
* Precision-recall AUC, when both classes exist.
* Brier score.
* Confusion matrix.
* Positive-label rate.
* Predicted-positive rate.

When a metric is undefined because a fold contains one class, return `None` or `NaN` with a warning. Do not crash or invent a value.

Accuracy must not be treated as the primary measure of strategy quality.

---

# 11.3 Evaluation Model

After completing walk-forward validation, train an evaluation model on every row of
the boundary-purged development dataset:

```python
def train_evaluation_model(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
) -> XGBClassifier:
    """Train the model used for untouched holdout evaluation."""
```

The evaluation model must not use any holdout or boundary-purge row. Its final label
exit timestamp must be strictly earlier than the holdout start timestamp.

Save it separately from the production model.

---

# 11.4 Production Model

Only after final evaluation is completed and the report is frozen:

```python
def train_production_model(
    full_labeled_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
) -> XGBClassifier:
    """Train a production model using all currently labeled history."""
```

Production training must be a separate explicit CLI command.

It must not automatically overwrite the evaluation model.

---

# 11.5 Feature Importance

For the evaluation model, save:

* Gain importance.
* Weight importance.
* Cover importance, when available.

Clearly label these as global model importances.

Do not use global feature importance as an explanation for an individual signal.

Local prediction explanations are outside Phase 1.

---

# 12. Baseline Models and Strategies

XGBoost must be compared with more than Buy & Hold.

# 12.1 Machine-Learning Baseline

Implement logistic regression using the same feature columns.

Because logistic regression requires feature scaling:

1. Build a scikit-learn pipeline.
2. Fit the scaler only on each training fold.
3. Apply the fitted scaler to that fold’s validation data.
4. Never scale using the full dataset before splitting.

Example:

```python
Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
    ),
])
```

---

# 12.2 Trading Baselines

Implement:

### Cash baseline

Never enters the market.

```text
Return: 0; Phase 1 cash earns no yield
Exposure: 0%
Trades: 0
```

### Buy & Hold

* Enter at the first open of the common performance window.
* Exit at the final open of the common performance window.
* Apply one entry cost and one exit cost.

### EMA crossover

Example rule:

```text
BUY when ema_short > ema_long.
STAY_OUT otherwise.
```

Use the same execution delay, costs, and no-overlap policy where applicable.

### Simple momentum

Example rule:

```text
BUY when return_24 > 0.
STAY_OUT otherwise.
```

Use the same execution assumptions.

### Random exposure baseline

Generate independent Bernoulli BUY signals whose probability equals the model's
predicted-positive rate on the same out-of-sample decision rows. This uses model
predictions, never actual labels or realized returns, and is a descriptive null
comparison rather than a fitted strategy.

Run exactly `RANDOM_BASELINE_SIMULATIONS` simulations. Simulation `i` uses a
deterministic seed derived from `RANDOM_SEED + i`. Apply the same execution delay,
position sizing, costs, fixed holding period, no-overlap state machine, and common
performance window as the model.

Report at least the median, 5th percentile, and 95th percentile for total return,
Sharpe ratio, and maximum drawdown, plus the fraction of simulations whose total
return equals or exceeds the model's total return.

The random-baseline definition and simulation count must be frozen using development
data. Apply that frozen procedure once to the final holdout; do not change it after
seeing holdout outcomes.

---

# 13. Backtesting

# 13.1 `backtesting/engine.py`

## Main Function

```python
def run_backtest(
    market_df: pd.DataFrame,
    probability_scores: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    signal_threshold: float,
    initial_capital: float,
    cost_config: CostConfig,
) -> BacktestResult:
    """Simulate the fixed-horizon long-or-cash strategy."""
```

Requirements:

* `market_df` contains the complete chronological OHLCV price context, including
  any rows needed to execute the final decision rows at `t+1` and `t+H+1`.
* `probability_scores.index` contains only decision rows and must be an ordered,
  unique subset of `market_df.index`.
* When classification context is requested, `actual_labels` must be non-null and
  have exactly the same ordered index as `probability_scores`. Live inference may
  pass `None`, in which case label-dependent metrics are omitted.
* The engine must never create signals on market-context rows that are absent from
  `probability_scores`.
* Probability scores must contain no missing values.
* The engine must not call the model.
* The engine receives already-generated out-of-sample predictions.
* Backtesting logic must be independent from training logic.
* Entry and exit positions are resolved against the positional order of
  `market_df`, not by adding integers to arbitrary index labels.

For a holdout backtest, pass the untouched holdout predictions together with the
original market frame extending through the last required exit open. Do not truncate
the market frame to the labeled decision rows.

---

# 13.2 Trading State Machine

Maintain:

```text
position_open
pending_entry_index
entry_index
scheduled_exit_index
entry_price
```

Process each market row chronologically in this order:

1. At the candle open, execute any scheduled exit.
2. At the candle open, execute any pending entry.
3. Mark the position through the current open-to-next-open interval.
4. After the candle closes, evaluate a signal only if the row index exists in
   `probability_scores`.

This ordering means a position that exited at the current open is flat when the
current candle's close becomes a decision point. A BUY at that close may schedule a
new entry for the following open. Same-open exit and re-entry are not allowed.

For each eligible decision row (t):

### When no position is open and no entry is pending

If:

```python
probability_score >= signal_threshold
```

then schedule:

```text
Entry index: t+1
Exit index: t+H+1
```

Only schedule the trade when both indexes exist inside the evaluable period.

Scheduling sets `pending_entry_index = t+1`. The position becomes open only when
that entry is executed; it must not receive market returns before the entry open.

### When a position is already open

Ignore new signals until the scheduled exit has executed. Because exits are
processed before close-time decisions, the exit candle itself may generate a new
signal for entry at the next candle open.

### At scheduled exit

Close the position and record the completed trade.

Every accepted trade must already have a valid scheduled exit. Never shorten the
fixed holding period merely to liquidate at the end of available data. Encountering
an unresolved final position is an error.

---

# 13.3 Execution Prices

For a long trade:

```python
entry_market_price = open_at_entry
exit_market_price = open_at_exit
```

Approximate adverse entry fill:

```python
entry_fill_price = entry_market_price * (
    1.0
    + slippage_rate
    + half_spread_rate
)
```

Approximate adverse exit fill:

```python
exit_fill_price = exit_market_price * (
    1.0
    - slippage_rate
    - half_spread_rate
)
```

Gross filled-price return:

```python
filled_gross_return = (
    exit_fill_price
    / entry_fill_price
    - 1.0
)
```

Net return after fees:

```python
net_growth_factor = (
    exit_fill_price
    / entry_fill_price
    * (1.0 - fee_rate)
    * (1.0 - fee_rate)
)

net_trade_return = net_growth_factor - 1.0
```

Do not subtract the fee only once.

---

# 13.4 Candle-Level Equity Curve

The backtest must produce a candle-level equity curve.

The common performance window for the model and all trading baselines begins at the
open immediately after the first decision row and ends at the latest exit open
required by the final eligible decision row. The market context may contain rows
outside this window, but they must not affect reported performance.

For each open-to-open interval:

$$
r_i^{open} = \frac{O_{i+1}}{O_i}-1
$$

When the strategy holds a position during interval (i):

$$
r_i^{strategy} = r_i^{open}
$$

Otherwise:

$$
r_i^{strategy}=0
$$

Entry and exit costs must be applied at their actual transaction timestamps.

Equity is marked at every open in the performance window after processing transactions
scheduled for that open. Entry equity reflects the adverse fill and entry fee. Exit
equity reflects the adverse fill and exit fee. The final equity must reconcile with
compounding the completed trade `net_return` values in ledger order.

Cumulative equity is:

$$
E_t = E_0 \prod_{i=1}^{t}(1+r_i^{net})
$$

The implementation must avoid index misalignment.

All result series must explicitly use the market-data index.

The equity output must include at least:

```text
timestamp
equity
period_return
position_open
market_exposure
```

---

# 13.5 Trade Ledger

Every completed trade must create one row:

```text
trade_id
signal_timestamp
entry_timestamp
exit_timestamp
entry_market_price
entry_fill_price
position_quantity
equity_before_entry
entry_fee
entry_execution_cost
exit_market_price
exit_fill_price
exit_fee
exit_execution_cost
equity_after_exit
holding_candles
gross_market_return
filled_gross_return
total_fee_rate
total_execution_cost_rate
net_return
pnl
probability_score
winning_trade
```

All fee and execution-cost amount columns use quote currency. Define adverse fill
costs against the contemporaneous market open for the executed quantity:

```python
entry_fee = equity_before_entry * fee_rate
entry_execution_cost = position_quantity * (
    entry_fill_price - entry_market_price
)

gross_exit_value = position_quantity * exit_fill_price
exit_fee = gross_exit_value * fee_rate
exit_execution_cost = position_quantity * (
    exit_market_price - exit_fill_price
)
equity_after_exit = gross_exit_value - exit_fee

total_fee_rate = 1.0 - (1.0 - fee_rate) ** 2
total_execution_cost_rate = 1.0 - (
    (1.0 - one_side_execution_rate)
    / (1.0 + one_side_execution_rate)
)
```

`pnl = equity_after_exit - equity_before_entry`. Ledger PnL and all four cost amount
columns must reconcile with the candle-level equity curve within floating-point
tolerance.

`num_trades` equals the number of completed trade-ledger rows.

Do not calculate trades as `position_changes / 2`.

---

# 13.6 Backtest Metrics

## Return Metrics

* Total return.
* Annualized return.
* Buy & Hold total return.
* Excess return versus Buy & Hold.
* Excess return versus cash.

## Risk Metrics

* Annualized volatility.
* Sharpe ratio.
* Sortino ratio.
* Maximum drawdown.
* Maximum drawdown duration.
* Calmar ratio.

## Trading Metrics

* Number of completed trades.
* Win rate by completed trade.
* Average trade return.
* Median trade return.
* Average winning return.
* Average losing return.
* Largest winning trade.
* Largest losing trade.
* Profit factor.
* Market exposure.
* Turnover.
* Average holding period.
* Total estimated costs.

## Classification Context

Also include:

* Positive prediction rate.
* Average BUY probability.
* Actual positive-label rate.

Use these exact Phase 1 definitions:

* `total_return = final_equity / initial_capital - 1`.
* Annualized return uses geometric compounding over the number of open-to-open
  intervals in the common performance window.
* Annualized volatility is the sample standard deviation (`ddof=1`) of candle-level
  net returns multiplied by `sqrt(periods_per_year)`.
* Maximum drawdown duration is the largest number of consecutive open-to-open
  intervals for which equity remains below its previous running peak. Include an
  unrecovered drawdown at the end of the evaluation.
* Calmar ratio is annualized return divided by the absolute maximum drawdown. Return
  `None` when maximum drawdown is zero.
* Market exposure is held open-to-open intervals divided by all open-to-open intervals
  in the common performance window.
* Turnover is the sum of absolute entry and exit market notionals divided by average
  candle-level equity over the common performance window. It is reported for the
  complete evaluation window, not annualized.
* Average holding period is the arithmetic mean of `holding_candles` across completed
  trades.
* Total estimated costs are the sum, in quote currency, of entry fees, exit fees,
  adverse entry-fill cost, and adverse exit-fill cost recorded by the trade ledger.

For empty-trade cases, return zero for total return, exposure, turnover, trade count,
and total estimated costs. Trade-distribution metrics that have no observations must
be `None`, not invented as zero.

---

# 13.7 Annualization

Calculate periods per year dynamically.

```python
PERIODS_PER_YEAR = {
    "1m": 60 * 24 * 365,
    "5m": 12 * 24 * 365,
    "15m": 4 * 24 * 365,
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}
```

Sharpe ratio:

$$
\text{Sharpe} = \frac{\bar r-r_f}{s_r}\sqrt{P}
$$

where:

* $\bar r$ is mean candle return.
* $r_f$ is the per-period risk-free rate.
* $s_r$ is return standard deviation.
* $P$ is periods per year.

If standard deviation is zero, return `None` or `NaN`.

Do not return infinity.

Convert the configured annual risk-free rate before calculating excess returns:

```python
risk_free_rate_per_period = (
    (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / periods_per_year)
    - 1.0
)
```

Annualized return is:

```python
annualized_return = (
    (1.0 + total_return) ** (periods_per_year / n_periods)
    - 1.0
)
```

Sortino uses all periods in the performance window, including zero-return cash
periods:

```python
excess_returns = period_returns - risk_free_rate_per_period
downside_deviation = np.sqrt(
    np.mean(np.minimum(excess_returns, 0.0) ** 2)
)
sortino = (
    excess_returns.mean()
    / downside_deviation
    * np.sqrt(periods_per_year)
)
```

Return `None` when there are too few observations or the downside deviation is zero.

---

# 13.8 Maximum Drawdown

Given equity $E_t$:

$$
\text{Peak}_t = \max_{u \leq t}E_u
$$

$$
\text{Drawdown}_t = \frac{E_t}{\text{Peak}_t}-1
$$

Maximum drawdown is:

$$
\min_t \text{Drawdown}_t
$$

It must be zero or negative.

---

# 13.9 Profit Factor

$$
\text{Profit Factor} =
\frac{\sum \text{positive trade PnL}}
{\left|\sum \text{negative trade PnL}\right|}
$$

If no losing trades exist, return `None` and add an explanatory warning rather than infinity.

---

# 13.10 Cost Sensitivity

The final evaluation report must use the exact frozen `COST_SCENARIOS` from
configuration:

```text
Low-cost scenario
Base-cost scenario
High-cost scenario
```

```text
Low:
Fee assumptions unchanged
Slippage: 1 bp per side
Half-spread: 0.5 bp per side

Base:
Slippage: 2 bps per side
Half-spread: 1 bp per side

High:
Slippage: 5 bps per side
Half-spread: 2 bps per side
```

The base scenario is the official Phase 1 result.

Cost scenarios must be defined and committed before inspecting holdout performance.
Sensitivity changes execution costs only; it does not regenerate labels or retrain
the frozen evaluation model.

---

# 14. Artifact Management

# 14.1 Run ID

Every execution that trains or evaluates a model must generate a unique run ID.

Recommended format:

```text
YYYYMMDDTHHMMSSZ_symbol_timeframe_short_commit
```

Example:

```text
20260802T160500Z_btc_usdt_1h_a1b2c3d
```

---

# 14.2 Run Directory

```text
artifacts/runs/{run_id}/
├── config.json
├── manifest.json
├── feature_columns.json
├── fold_metrics.json
├── oof_predictions.csv
├── classification_report.json
├── feature_importance.csv
├── logs.txt
└── holdout_evaluation_claim.json  # created only by evaluate-holdout
```

---

# 14.3 Evaluation Directory

```text
artifacts/evaluations/{run_id}/
├── input_data_snapshot.csv
├── evaluation_model.json
├── feature_columns.json
├── holdout_predictions.csv
├── trade_ledger.csv
├── equity_curve.csv
├── metrics.json
├── baseline_metrics.json
├── cost_sensitivity.json
└── evaluation_manifest.json
```

---

# 14.4 Production Directory

```text
artifacts/production/
├── versions/
│   └── {model_version}/
│       ├── model.json
│       ├── feature_columns.json
│       └── manifest.json
│
└── active_model.json
```

---

# 14.5 Run Manifest

Every run manifest must contain:

```text
run_id
created_at_utc
project_version
Git commit
Git branch
dirty_worktree flag
Python version
dependency versions
random seed
exchange
symbol
timeframe
data path
immutable snapshot path
data hash
data start timestamp
data end timestamp
row count
feature configuration
feature columns
label definition
prediction horizon
minimum required return
development boundary
boundary-purge start and end
holdout boundary
purge gap
walk-forward configuration
model parameters
signal threshold
fee assumptions
spread assumptions
slippage assumptions
classification metrics
strategy metrics
warnings
holdout-evaluation claim status
invalidation reason, when applicable
```

`input_data_snapshot.csv` must be a byte-for-byte copy or verified hard link of the
snapshot named by the manifest. Its SHA-256 must be checked after placement. This
makes a finalized evaluation self-contained even if convenience data files later
change.

---

# 14.6 Model Metadata

Every saved model must record:

```text
model_version
model_type
training_start
training_end
training_row_count
feature_columns
feature_schema_hash
model_parameters
prediction_horizon
label threshold
signal threshold
data hash
code commit
created_at_utc
```

---

# 15. Command-Line Interface

Use subcommands instead of only skip flags.

Primary entry:

```bash
python scripts/run_pipeline.py <command>
```

---

## 15.1 Fetch Data

```bash
python scripts/run_pipeline.py fetch
```

Optional overrides:

```bash
python scripts/run_pipeline.py fetch \
  --symbol ETH/USDT \
  --timeframe 4h \
  --lookback-days 1095
```

Behavior:

* Load existing data.
* Fetch missing closed candles.
* Validate.
* Save the latest-data file atomically.
* Persist or reuse its immutable content-addressed snapshot.
* Print the snapshot path and SHA-256 hash.

---

## 15.2 Build Dataset

```bash
python scripts/run_pipeline.py prepare
```

Behavior:

1. Load a valid immutable raw snapshot and record its hash.
2. Compute features.
3. Save inference-ready feature data.
4. Add labels.
5. Save labeled model data.
6. Print:

   * Row counts.
   * Date range.
   * Feature count.
   * Removed warm-up rows.
   * Removed unlabeled tail rows.

`prepare` must not print or log label distributions or return summaries from the full
real dataset because the final rows have not yet been isolated as holdout. Outcome
summaries begin only after `validate` creates the boundary and must cover development
rows only.

---

## 15.3 Walk-Forward Evaluation

```bash
python scripts/run_pipeline.py validate
```

Behavior:

1. Load labeled data.
2. Create development, boundary-purge, and holdout partitions.
3. Verify development label exit timestamps precede the holdout start.
4. Keep boundary-purge and holdout rows isolated.
5. Run purged walk-forward validation on development data.
6. Evaluate XGBoost.
7. Evaluate logistic regression.
8. Save out-of-fold predictions and metrics.
9. Train and save the evaluation XGBoost model using development data only.
10. Print the development-only label distribution and split boundaries, but no
    holdout label or return statistics.

This command must not backtest the final holdout.

---

## 15.4 Final Holdout Evaluation

```bash
python scripts/run_pipeline.py evaluate-holdout \
  --run-id <development-run-id>
```

Behavior:

1. Complete all preflight validation without loading holdout values.
2. Verify from frozen metadata that the evaluation model used no boundary-purge or
   holdout labels.
3. Atomically create a holdout-evaluation claim for the development run.
4. Load the frozen evaluation model, exact immutable data snapshot, and untouched
   holdout only after the claim succeeds.
5. Generate probabilities.
6. Run the strategy backtest using market context through the final required exit.
7. Run trading baselines over the identical performance window.
8. Run the frozen cost-sensitivity scenarios.
9. Save immutable evaluation artifacts and mark the claim completed.
10. Print a comparison table.

The claim is an exclusive file created with failure-if-exists semantics under the
development run directory. If any claim already exists, including a failed or
incomplete claim, the command must refuse to expose the holdout again. Recovery
requires an explicit human research decision and must be recorded as an invalidated
evaluation; it is not a normal retry path.

This command should display a warning:

```text
You are evaluating the final holdout.
Do not use these results for iterative model tuning.
```

---

## 15.5 Train Production Model

```bash
python scripts/run_pipeline.py train-production
```

Behavior:

1. Load all currently labeled data.
2. Train a production model.
3. Save it under a new version.
4. Do not overwrite previous versions.
5. Do not automatically activate it unless explicitly requested.

---

## 15.6 Full Development Pipeline

```bash
python scripts/run_pipeline.py run-development
```

Runs:

```text
fetch
prepare
validate
```

It must not automatically evaluate the final holdout.

---

# 16. Console Output

Use logging for module-level progress.

The CLI may print concise formatted summaries.

Example validation summary:

```text
Development period:
2023-08-01 00:00 UTC → 2025-12-10 14:00 UTC

Final holdout:
2025-12-10 15:00 UTC → 2026-08-01 22:00 UTC

Walk-forward results:
Model                  Accuracy   PR-AUC   Log Loss   Brier
XGBoost                  0.527     0.541      0.691   0.248
Logistic Regression      0.511     0.518      0.696   0.251
```

Example holdout summary:

```text
Holdout backtest — Base cost scenario

Metric                  XGBoost    EMA Rule    Momentum    Buy & Hold
Total Return              4.2%       1.1%         0.8%         6.3%
Sharpe                    0.61       0.22         0.15         0.74
Max Drawdown             -8.4%     -12.2%       -14.1%       -18.7%
Exposure                 31.5%      44.2%        49.1%       100.0%
Completed Trades            42         61           58            1
```

Example numbers above are formatting examples only and must never be hardcoded.

---

# 17. Testing Strategy

# 17.1 General Rules

* Use `pytest`.
* Use `pytest-mock`.
* Tests must never make real network calls.
* Test data must be deterministic.
* Use fixed random seeds.
* Each public function needs:

  * A normal-case test.
  * At least one edge-case test.
* Tests must validate behavior, not only output shape.
* Test failures must not be hidden by broad exception handling.

---

# 17.2 Synthetic OHLCV Fixture

Create deterministic synthetic data containing:

* Trend.
* Cyclic movement.
* Noise.
* Nonconstant volume.
* Valid OHLC relationships.
* UTC hourly timestamps.

Example design:

$$
C_t = 100 + 0.01t + 2\sin(t/24) + \epsilon_t
$$

Construct:

```text
open
high
low
close
volume
```

such that all OHLC consistency rules hold.

---

# 17.3 Data Tests

Required tests:

```text
test_fetch_paginates_until_end
test_fetch_retries_network_errors
test_fetch_raises_after_retry_exhaustion
test_fetch_does_not_retry_permanent_exchange_error
test_incomplete_last_candle_is_removed
test_incremental_fetch_starts_after_last_timestamp
test_duplicate_timestamps_are_removed
test_atomic_write_preserves_previous_file_on_failure
test_incremental_update_preserves_immutable_snapshot
test_market_data_result_identifies_exact_snapshot
test_manifest_references_exact_snapshot_hash
test_timestamp_is_utc
test_timestamp_aligns_to_timeframe_grid
test_missing_candle_is_detected
test_invalid_ohlc_relationship_is_rejected
test_negative_volume_is_rejected
test_symbol_slug_is_filesystem_safe
```

---

# 17.4 Feature Tests

Required tests:

```text
test_compute_features_preserves_timestamp
test_compute_features_preserves_ohlcv_columns
test_expected_feature_columns_exist
test_features_are_numeric
test_feature_rows_are_chronological
test_feature_output_has_no_infinity
test_latest_closed_row_is_available_for_inference
test_insufficient_history_raises_clear_error
```

---

# 17.5 Strong Look-Ahead Test

Implement:

```text
test_future_data_changes_do_not_modify_past_features
```

Procedure:

1. Compute features from the original dataset.
2. Select a cutoff timestamp (T).
3. Modify every OHLCV value after (T).
4. Recompute features.
5. Verify that all feature values at and before (T) are identical.

This is the primary feature look-ahead test.

---

# 17.6 Label Tests

Required tests:

```text
test_entry_uses_next_open
test_exit_uses_horizon_plus_one_open
test_label_records_entry_and_exit_timestamp_provenance
test_label_uses_executable_forward_return
test_label_uses_minimum_required_return
test_minimum_gross_return_reconciles_with_net_edge
test_unrealizable_tail_rows_are_removed
test_features_are_unchanged_when_labels_are_added
```

Use a small manually calculated price example.

Example:

```text
Decision at row 0
Entry at open row 1 = 100
Horizon = 2
Exit at open row 3 = 103

Gross return = 103 / 100 - 1 = 3%
```

---

# 17.7 Split Tests

Required tests:

```text
test_development_precedes_holdout
test_holdout_ratio_is_correct_with_rounding
test_walk_forward_training_precedes_validation
test_walk_forward_training_expands
test_walk_forward_validation_blocks_are_contiguous
test_purge_gap_is_applied
test_training_labels_do_not_touch_validation_prices
test_holdout_boundary_purge_covers_label_lookahead
test_development_label_exits_precede_holdout_start
test_holdout_is_not_present_in_any_walk_forward_fold
test_insufficient_rows_raises_error
```

Critical invariant:

```python
assert (
    max(train_indices)
    + PREDICTION_HORIZON
    + 1
    < min(validation_indices)
)
```

---

# 17.8 Training Tests

Required tests:

```text
test_feature_columns_are_derived_at_runtime
test_non_feature_columns_are_excluded
test_model_receives_saved_feature_order
test_new_model_is_created_for_each_fold
test_fold_predictions_cover_only_validation_rows
test_oof_predictions_are_chronological
test_model_serialization_preserves_predictions
test_evaluation_model_uses_development_data_only
test_production_model_is_saved_separately
test_fixed_random_seed_is_reproducible
```

---

# 17.9 Backtest Tests

Required tests:

```text
test_signal_enters_at_next_open
test_trade_holds_exact_horizon
test_backtest_uses_price_context_beyond_final_decision_row
test_market_context_rows_do_not_generate_signals
test_trade_invests_full_current_equity
test_new_signals_are_ignored_while_position_open
test_exit_is_processed_before_exit_candle_signal
test_entry_fee_is_charged
test_exit_fee_is_charged
test_slippage_is_adverse_on_entry
test_slippage_is_adverse_on_exit
test_spread_is_adverse_on_both_sides
test_no_fee_when_no_trade_occurs
test_no_overlapping_trades
test_final_open_position_is_not_left_unresolved
test_num_trades_matches_trade_ledger
test_trade_return_matches_manual_calculation
test_equity_curve_has_no_nan
test_equity_curve_has_no_infinity
test_equity_index_matches_market_index
test_zero_volatility_sharpe_is_not_infinite
test_max_drawdown_is_zero_or_negative
test_buy_hold_uses_entry_and_exit_costs
```

---

# 17.10 Leakage Test

Required:

```text
test_holdout_is_never_used_during_training
test_prepare_does_not_report_full_dataset_outcome_statistics
test_holdout_evaluation_claim_prevents_second_access
test_failed_holdout_claim_is_not_silently_removed
```

The test must patch or instrument model fitting and verify that no holdout or
boundary-purge index is passed into `fit()`. It must also verify that every fitted
row's `exit_timestamp` is earlier than the first holdout timestamp; checking fitted
indexes alone is not sufficient.

---

# 17.11 Integration Test

```text
test_phase1_pipeline_runs_end_to_end_on_synthetic_data
```

The integration test must:

1. Use mocked exchange responses.
2. Fetch or load synthetic OHLCV.
3. Validate data.
4. Compute features.
5. Add labels.
6. Split development, boundary-purge, and holdout rows.
7. Run walk-forward validation.
8. Train an evaluation model.
9. Generate holdout predictions.
10. Run the backtest with market-price context through the final required exit.
11. Save artifacts.
12. Verify expected artifact files exist.

---

# 18. Error Handling

Create project-specific exceptions.

```python
class CryptoAIError(Exception):
    """Base project exception."""


class MarketDataError(CryptoAIError):
    """Base market-data exception."""


class MarketDataNetworkError(MarketDataError):
    """Network request failed after retries."""


class MarketDataExchangeError(MarketDataError):
    """Exchange rejected the market-data request."""


class MarketDataValidationError(MarketDataError):
    """Market data violated required invariants."""


class FeatureEngineeringError(CryptoAIError):
    """Feature computation failed."""


class LabelGenerationError(CryptoAIError):
    """Label generation failed."""


class DatasetSplitError(CryptoAIError):
    """Chronological split construction failed."""


class ModelTrainingError(CryptoAIError):
    """Model training failed."""


class BacktestError(CryptoAIError):
    """Backtest execution failed."""


class ArtifactError(CryptoAIError):
    """Artifact loading or saving failed."""
```

Rules:

* Never use a bare `except`.
* Do not silently swallow errors.
* Log context before re-raising.
* Do not convert serious data errors into empty DataFrames.
* Do not continue training with invalid or incomplete data.
* Error messages should include symbol, timeframe, operation, and relevant timestamp where available.

---

# 19. Logging

Use Python’s `logging` module.

Module code must not rely on `print()`.

Recommended format:

```text
2026-08-02T16:20:13Z | INFO | crypto_ai.data.fetch | Fetching BTC/USDT 1h from ...
```

Log:

* Start and completion of pipeline steps.
* Data date range and row count.
* Number of fetched candles.
* Number of duplicates removed.
* Missing-candle validation.
* Number of feature warm-up rows removed.
* Development-only label distribution after holdout isolation.
* Split boundaries.
* Fold boundaries.
* Model training completion.
* Artifact paths.
* Backtest trade count.
* Warnings and undefined metrics.

Do not log secrets.

---

# 20. Coding Conventions

## 20.1 Python

* Python 3.11 or newer.
* Type hints on every public function.
* Docstrings on every public function and public class.
* Maximum line length: 100.
* Prefer small, testable functions.
* Avoid hidden global state.
* Avoid mutable default arguments.
* Use `pathlib.Path`.
* Use timezone-aware UTC datetimes.
* Use `float64` for market prices.
* Use `int8` for binary labels when practical.
* Use lowercase snake_case for columns.
* Keep timestamps as a plain DataFrame column.

---

## 20.2 Formatting and Linting

Use:

```text
black
ruff
```

Required commands:

```bash
black .
ruff check .
pytest
```

Codex must not report a task as complete until relevant checks pass or it clearly reports why they do not pass.

---

## 20.3 Imports

Use absolute project imports:

```python
from crypto_ai.config import settings
from crypto_ai.costs import CostConfig
from crypto_ai.features.build import compute_features
```

Avoid modifying `sys.path` inside production modules.

The package should be installed in editable mode during development:

```bash
pip install -e .
```

---

## 20.4 Pandas Indexing

Because index misalignment is a known risk:

* Preserve indexes intentionally.
* Reset indexes only when explicitly required.
* When creating a Series from NumPy predictions, pass the original DataFrame index.

Correct:

```python
probabilities = pd.Series(
    model.predict_proba(X)[:, 1],
    index=X.index,
    name="probability_score",
)
```

Incorrect:

```python
probabilities = pd.Series(
    model.predict_proba(X)[:, 1]
)
```

Before arithmetic between Series:

```python
assert left.index.equals(right.index)
```

---

## 20.5 File I/O

* Create directories with `mkdir(parents=True, exist_ok=True)`.
* Use atomic writes.
* Include CSV headers.
* Save dates as ISO 8601 UTC strings.
* Parse timestamp columns explicitly when loading.
* Never concatenate file paths with string operators.
* Do not overwrite evaluation artifacts.
* Store JSON with stable formatting.

---

# 21. Dependencies

## Runtime Dependencies

```text
ccxt
pandas
numpy
ta
xgboost
scikit-learn
python-dotenv
```

## Development Dependencies

```text
pytest
pytest-mock
black
ruff
```

Use compatible version ranges in human-edited requirement files.

Generate and commit a lockfile containing exact resolved versions for reproducibility.

Do not add FastAPI, APScheduler, React, Anthropic, or dashboard dependencies during Phase 1.

---

# 22. Git Conventions

## Branch

```text
phase-1-baseline
```

## Commit Format

```text
[module] concise description
```

Examples:

```text
[data] add incremental OHLCV fetching
[features] add point-in-time technical indicators
[labels] align target with next-open execution
[splits] add purged walk-forward validation
[backtest] add fixed-horizon state machine
[tests] verify holdout isolation
```

## Gitignored Files

```text
.env
data/
artifacts/
__pycache__/
.pytest_cache/
.ruff_cache/
.venv/
```

Do not ignore example configuration or test fixtures that are required for reproducibility.

---

# 23. Phase 1 Implementation Sequence

# Milestone 0 — Repository Bootstrap

## Deliverables

* Git repository initialization when the project is not already a worktree.
* `.gitignore` and `phase-1-baseline` working branch.
* Repository structure.
* `pyproject.toml`.
* Runtime and development requirements.
* Importable `crypto_ai` package.
* Logging configuration.
* Base exceptions.
* Initial configuration.
* Test configuration.
* CI workflow.

## Acceptance Criteria

```bash
git rev-parse --is-inside-work-tree
pip install -e .
black --check .
ruff check .
pytest
```

All commands complete successfully.

---

# Milestone 1 — Market Data

**Status:** Implemented with mocked-network coverage. The `fetch` command, strict OHLCV
validation, incremental atomic storage, and immutable SHA-256 snapshots are available.

## Deliverables

* Symbol slug helper.
* Initial OHLCV fetch.
* Pagination.
* Retry handling.
* Incomplete-candle filtering.
* Incremental update.
* Atomic CSV storage.
* Immutable content-addressed raw snapshots.
* Data validation.
* Unit tests.

## Acceptance Criteria

* No real network requests in tests.
* Duplicate timestamps are removed.
* Missing candles are detected.
* Current incomplete candle is excluded.
* Incremental fetching starts at the correct timestamp.
* Valid BTC/USDT hourly data can be downloaded and stored.
* Incremental updates never modify an existing snapshot.

---

# Milestone 2 — Features and Labels

## Deliverables

* `compute_features()`.
* Required technical indicators.
* Point-in-time validation test.
* `add_labels()`.
* Executable forward-return target.
* Cost-aware label threshold.
* Shared `CostConfig` and multiplicative cost helpers.
* Processed dataset storage.

## Acceptance Criteria

* Future-data perturbation does not change past features.
* Latest closed row remains available for inference.
* Labels use next-open entry and horizon-based next-open exit.
* Labels store entry and exit timestamp provenance.
* Training-tail rows without known outcomes are removed.
* No feature column contains missing or infinite values after training preparation.

---

# Milestone 3 — Dataset Splitting

## Deliverables

* Development/boundary-purge/holdout split.
* Purged expanding walk-forward splits.
* Split metadata.
* Split tests.

## Acceptance Criteria

* Holdout appears in no training or validation fold.
* Boundary-purge rows appear in no model fit.
* Every fold is chronological.
* Purge gap is at least `PREDICTION_HORIZON + 1`.
* Training labels do not reference validation-period prices.
* Every development label exit timestamp precedes the holdout start.

---

# Milestone 4 — Models and Validation

## Deliverables

* XGBoost training.
* Logistic-regression baseline.
* Per-fold predictions.
* Out-of-fold predictions.
* Classification metrics.
* Evaluation model trained on development only.
* Feature schema artifact.
* Feature-importance artifact.

## Acceptance Criteria

* Every fold creates a new model.
* Validation rows are never used for fitting that fold.
* Saved and reloaded models produce matching predictions.
* Evaluation model training ends before the holdout begins.
* Evaluation model training excludes all boundary-purge rows.
* Results are reproducible with the same seed and data.

---

# Milestone 5 — Backtesting

## Deliverables

* Fixed-horizon state machine.
* Next-open execution.
* No overlapping positions.
* Fee, spread, and slippage model.
* Trade ledger.
* Candle-level equity curve.
* Strategy metrics.
* Buy & Hold baseline.
* Cash baseline.
* EMA baseline.
* Momentum baseline.
* Random exposure baseline.
* Cost-sensitivity analysis.

## Acceptance Criteria

* Manual trade calculations match engine output.
* Entry and exit costs are both applied.
* All positions have valid exits.
* Tail decisions use the retained market-price context needed for valid exits.
* Every entry invests exactly 100% of current equity without leverage.
* No missing or infinite equity values.
* Trade count equals ledger length.
* Baselines use equivalent execution assumptions.
* Model and baseline metrics use the identical performance window.

---

# Milestone 6 — Development Report

## Deliverables

* Development-period walk-forward report.
* Fold comparison.
* XGBoost versus logistic regression.
* Trading-baseline comparison using out-of-fold predictions where applicable.
* Feature-importance summary.
* Limitations.
* Frozen configuration for final evaluation.

## Acceptance Criteria

Before evaluating the holdout, freeze:

```text
Feature list
Prediction horizon
Label threshold
Signal threshold
XGBoost parameters
Cost assumptions
Execution policy
Baseline definitions
Metric definitions
```

Commit the frozen configuration to Git.

---

# Milestone 7 — Final Holdout Evaluation

## Deliverables

* Holdout predictions.
* Trade ledger.
* Equity curve.
* Strategy metrics.
* Baseline metrics.
* Cost-sensitivity results.
* Immutable evaluation manifest.
* README summary.

## Acceptance Criteria

* Evaluation model uses development data only.
* An exclusive claim prevents routine repeat evaluation of the holdout under the
  development run.
* Holdout is evaluated once under the frozen configuration.
* All costs are included.
* Results are compared against all required baselines.
* Evaluation artifacts cannot be silently overwritten.
* The exact immutable input snapshot is included and hash-verified.
* Limitations and uncertainty are documented.

---

# Milestone 8 — Production Model

## Deliverables

* Production model trained on all labeled history.
* Production feature schema.
* Production manifest.
* Versioned storage.

## Acceptance Criteria

* Production model is stored separately from the evaluation model.
* Production model is not used to calculate historical holdout metrics.
* Previous production versions remain available.
* Inference uses the exact saved feature order.

---

# 24. Phase 1 Acceptance Criteria

Phase 1 is technically complete only when all conditions below are satisfied.

## Data Integrity

* Market data is chronological.
* Market data contains no duplicate timestamps.
* Missing candles are detected.
* Only closed candles are used.
* OHLCV relationships are valid.
* Incremental updates work.
* Timestamps align to the configured exchange timeframe grid.
* Every research run references an immutable, content-addressed raw snapshot.

## Feature Integrity

* Features use only current and past information.
* Future-data perturbation tests pass.
* Feature and label creation are separate.
* The latest closed candle can produce an inference feature row.
* Feature columns are deterministic and ordered.
* Derived columns exactly match the expected feature allowlist.

## Split Integrity

* The final holdout is selected chronologically.
* The final holdout is excluded from all development activity.
* Boundary-purge rows cover the full label lookahead and enter no fit or development
  metric.
* Every fitted development label exit timestamp precedes the holdout start.
* Walk-forward folds are chronological.
* The purge gap covers the complete label lookahead.

## Model Integrity

* XGBoost uses fixed documented parameters.
* A logistic-regression baseline is evaluated.
* Classification metrics are saved.
* Evaluation and production models are separate.
* Model metadata and feature schemas are saved.

## Backtest Integrity

* Signals execute no earlier than the next open.
* The holding period matches the label horizon.
* Overlapping positions are not allowed.
* Trades use 100% of current equity, fractional quantities, and no leverage.
* Entry and exit fees are included.
* Spread and slippage are included.
* The equity curve is index-aligned.
* The trade ledger reconciles with total returns.
* Decision rows and their extended execution-price context are handled separately.
* All strategies use an identical performance window and metric definitions.
* Results are compared against simple baselines.
* Cost-sensitivity results are reported.

## Reproducibility

* A dependency lockfile exists.
* Random seeds are stored.
* Data hashes and immutable snapshot paths are stored.
* Final evaluation artifacts include a verified copy of their exact input snapshot.
* Git commit information is stored.
* Run manifests are complete.
* Artifacts are versioned.
* Tests pass on a clean machine.

---

# 25. Research Success Criteria

Engineering completion and research success are separate.

## Engineering Completion

The pipeline is correct, reproducible, tested, and executable.

## Research Success

Evidence may justify continuing toward Phase 2 or Phase 3 when:

1. Results are not dependent on one short time window.
2. Performance remains reasonable across cost scenarios.
3. XGBoost provides value over simple baselines.
4. Performance is not produced by one or two extreme trades.
5. Drawdown is operationally acceptable.
6. Trade count is large enough for meaningful interpretation.
7. Results remain directionally similar across multiple chronological folds.
8. No leakage or accounting error is detected.

There is no required minimum accuracy.

A model with high accuracy may still be unprofitable.

A model with modest accuracy may be useful when:

* Winning trades are larger than losing trades.
* Costs are controlled.
* Drawdown is acceptable.
* Performance is stable.

---

# 26. Conditions That Block Phase 3

Do not begin a public signal API merely because Phase 1 code works.

Phase 3 should be delayed when:

* Holdout performance collapses after costs.
* Results depend on one market regime.
* The model does not beat simple baselines.
* Trade count is too small.
* Drawdowns are excessive.
* Data gaps remain unresolved.
* Probability scores are unstable.
* Retraining materially changes historical conclusions.
* Evaluation cannot be reproduced.
* Model and feature versions cannot be traced.

A failed or neutral Phase 1 result is still useful because it establishes a valid baseline for Phase 2 experiments.

---

# 27. Known Risks and Mitigations

| Risk                                           | Assessment  | Mitigation                                                              |
| ---------------------------------------------- | ----------- | ----------------------------------------------------------------------- |
| Model performance is close to random           |        High | Treat Phase 1 as a baseline; compare with simple models and strategies |
| Look-ahead leakage                             | High impact | Point-in-time tests, fold and holdout-boundary purges, label timestamps |
| Target and execution mismatch                  | High impact | Shared next-open target and backtest definition                        |
| Historical overfitting                         |        High | Fixed Phase 1 parameters, walk-forward validation, untouched holdout   |
| Optimistic transaction costs                   |        High | Fees, spread, slippage, and cost-sensitivity scenarios                 |
| Market-regime dependence                       |        High | Use multiple years and report fold-level stability                     |
| API data gaps                                  |      Medium | Strict interval validation and failed-run behavior                     |
| Incomplete current candle                      |      Medium | Explicit close-time filtering                                          |
| DataFrame index misalignment                   |      Medium | Preserve indexes and assert equality before arithmetic                 |
| Probability misinterpretation                  |      Medium | Use `probability_score`; do not claim calibrated confidence            |
| Artifact overwrite                             |      Medium | Unique run IDs, evaluation claims, and immutable directories            |
| Mutable source data prevents reproduction      |      Medium | Content-addressed snapshots copied into final evaluations               |
| Dependency drift                               |      Medium | Exact dependency lockfile                                              |
| Exchange behavior changes                      |      Medium | Isolate exchange adapter and preserve raw data                         |
| Very small number of trades                    |      Medium | Report trade count and avoid strong conclusions                        |
| Production model differs from evaluation model |    Expected | Maintain separate artifacts and purposes                               |

---

# 28. Phase 2 Preparation Requirements

Phase 2 sentiment work must not begin until Phase 1 is frozen.

When Phase 2 begins, historical sentiment must be point-in-time correct.

At minimum, every article will require:

```text
article_id
source
title
published_at
first_seen_at
asset
content_hash
sentiment_model_id
prompt_version
sentiment_score
scored_at
```

A candle-level sentiment feature may use only articles with:

```python
article["first_seen_at"] <= decision_timestamp
```

Caching sentiment only by coin and calendar date is prohibited for hourly modeling.

This section is informational only. No sentiment code belongs in Phase 1.

---

# 29. Phase 3 Architecture Requirement

When the API phase begins, signal serving, signal refresh, and model training must be separate responsibilities.

Recommended architecture:

```text
Scheduled data job:
Fetch newly closed candles
Compute latest features
Generate and store current signals

Training job:
Train candidate models
Evaluate candidates
Write versioned model artifacts

FastAPI service:
Load active model and signal artifacts
Serve read-only API requests
```

Do not run an uncoordinated scheduler inside every FastAPI worker.

This section is informational only. No API code belongs in Phase 1.

---

# 30. Glossary

| Term                      | Definition                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------- |
| OHLCV                     | Open, High, Low, Close, and Volume values for a candle                                |
| Candle                    | Aggregated market activity for one fixed time interval                                |
| Decision row              | Data available immediately after one candle closes                                    |
| Feature                   | Model input computed from present and historical information                          |
| Label                     | Target value the model learns to predict                                              |
| Horizon                   | Number of complete candles held after entry                                           |
| Entry open                | Opening price of the candle after the decision candle                                 |
| Exit open                 | Opening price after the configured holding period                                     |
| Point-in-time correctness | Ensuring historical rows use only information available at that time                  |
| Look-ahead bias           | Future information leaking into training or historical predictions                    |
| Purge gap                 | Rows removed between training and validation to prevent overlapping label information |
| Boundary purge            | Rows excluded before the final holdout so development labels cannot use holdout prices |
| Price context             | OHLCV rows retained for execution even when they are not eligible decision rows        |
| Walk-forward validation   | Chronological evaluation using expanding training periods                             |
| Development period        | Data used for model development and validation                                        |
| Final holdout             | Untouched future data used for one final evaluation                                   |
| Evaluation model          | Model trained only on development data for holdout prediction                         |
| Production model          | Model trained on all labeled data for later live use                                  |
| Probability score         | Model output for the positive class; not necessarily calibrated confidence            |
| Signal threshold          | Probability score required to produce BUY                                             |
| Slippage                  | Difference between expected and simulated execution price                             |
| Spread                    | Difference between bid and ask prices                                                 |
| Trade ledger              | Table containing every simulated completed trade                                      |
| Equity curve              | Portfolio value through time                                                          |
| Drawdown                  | Decline from a previous equity peak                                                   |
| Exposure                  | Fraction of evaluated time during which the strategy holds a position                 |
| Turnover                  | Amount of position entry and exit activity                                            |
| Profit factor             | Gross winning PnL divided by absolute gross losing PnL                                |
| Baseline                  | Simple reference method used to judge whether the model adds value                    |

---

# 31. Final Codex Checklist

Before completing any Phase 1 coding task, Codex must verify:

```text
[ ] The implementation follows this specification.
[ ] No future information enters a feature.
[ ] No final-holdout row enters model fitting.
[ ] No boundary-purge row enters fitting or development metrics.
[ ] Every fitted development label exits before the holdout begins.
[ ] Purge gap covers horizon plus next-open execution.
[ ] Label and backtest use the same entry and exit prices.
[ ] The backtest retains price context through every scheduled exit.
[ ] Position sizing and metric windows match the fixed Phase 1 definitions.
[ ] Current incomplete candles are excluded.
[ ] DataFrame indexes are intentionally preserved.
[ ] Entry and exit costs are both included.
[ ] Relevant tests were added or updated.
[ ] Formatting passes.
[ ] Linting passes.
[ ] Tests pass.
[ ] Artifacts are not silently overwritten.
[ ] The run references an immutable data snapshot and verified hash.
[ ] Files changed and limitations are reported.
```

---

# 32. Recommended First Codex Task

Use this initial implementation request:

```text
Read IMPLEMENTATION_PLAN.md completely.

Implement only Milestone 0: Repository Bootstrap.

Create the specified src-based repository structure, configuration module,
project-specific exception classes, logging configuration, pyproject.toml,
runtime and development requirement files, pytest configuration, and a minimal
CI workflow.

If the directory is not already a Git worktree, initialize it and create the
`phase-1-baseline` branch. Do not create a commit unless the human developer asks.

Do not implement market-data fetching, feature engineering, model training, or
backtesting yet.

Requirements:
1. Python 3.11+.
2. Absolute imports through the crypto_ai package.
3. Black line length 100.
4. Ruff configured consistently with Black.
5. Tests must verify that:
   - the package imports successfully,
   - required directories can be created,
   - settings paths resolve under the project root,
   - custom exceptions inherit from CryptoAIError.
6. Run:
   - black --check .
   - ruff check .
   - pytest
7. Report:
   - files created,
   - design decisions,
   - commands executed,
   - test results,
   - unresolved issues.

Do not add libraries or architecture not defined in IMPLEMENTATION_PLAN.md.
```
