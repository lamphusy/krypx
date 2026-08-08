# KrypX

KrypX is a reproducible Phase 1 research pipeline for testing long-or-cash
cryptocurrency signals without look-ahead leakage. The baseline configuration uses
BTC/USDT hourly candles, trailing technical features, an XGBoost classifier, next-open
execution, a fixed four-candle holding period, and explicit fees, spread, and slippage.

The Phase 1 pipeline is implemented end to end, but the real Phase 1 research evaluation
is a separate milestone. Code completion means the pipeline can produce auditable
out-of-sample results; it does not mean the strategy is profitable, useful, or suitable
for live trading. No profitability conclusion follows from passing the test suite.

## Pipeline

The normal workflow has five deliberately separate stages:

```bash
# 1. Fetch and validate closed candles; create an immutable raw snapshot.
krypx fetch

# 2. Build point-in-time features and executable next-open labels.
krypx prepare

# 3. Run development-only purged walk-forward validation.
krypx validate

# 4. Human-authorized action only: evaluate the frozen holdout once.
krypx evaluate-holdout --run-id <development-run-id>

# 5. After explicitly accepting that completed evaluation, train a production version.
krypx train-production --evaluation-run-id <development-run-id>
```

`validate` compares XGBoost with fold-local scaled logistic regression, saves continuous
out-of-fold predictions, runs development-period trading baselines, trains an evaluation
model only on the boundary-purged development rows, and leaves the final holdout
unevaluated.

`evaluate-holdout` is a separate, human-authorized research action and is intentionally
irreversible for a development run. It creates an exclusive claim before reading holdout
values; completed, failed, and interrupted claims all prevent routine repeat access. The
command recreates features from the verified raw snapshot and uses the run's frozen
schema, split, threshold, execution, cost, and random-baseline configuration. Running
`validate` never grants permission to evaluate the holdout automatically.

Train a distinct, versioned model on all labeled history only after a completed final
evaluation has been reviewed and explicitly accepted:

```bash
krypx train-production --evaluation-run-id <development-run-id>
```

The evaluation run ID is workflow authorization and provenance only: production fitting
does not use holdout metrics as inputs. Preflight checks require a completed claim and
compatible, intact evaluation artifacts. The command does not overwrite or activate an
older production model. To fetch, prepare, and validate in one development-only command,
use `krypx run-development`; it never evaluates the final holdout.

Every command is also available through:

```bash
python scripts/run_pipeline.py <command>
```

## What is saved

Market data is stored as both a latest-data convenience file and an immutable,
content-addressed snapshot:

```text
data/raw/{symbol_slug}_{timeframe}.csv
data/raw/snapshots/{symbol_slug}_{timeframe}/{sha256}.csv
```

Each prepared feature/label bundle has a completion manifest containing file hashes,
feature schema, frozen feature/label settings, and the exact immutable raw snapshot that
produced it. Development and production workflows reject incomplete or mismatched bundles
instead of silently following a newer latest-data file.

Development runs under `artifacts/runs/{run_id}/` contain the frozen configuration,
manifest, split metadata, feature schema, fold metrics, classification comparison,
out-of-fold predictions, development strategy metrics, global XGBoost feature importance,
and a readable development report.

The matching `artifacts/evaluations/{run_id}/` directory contains a verified copy of the
exact raw snapshot, the development-only evaluation model and metadata, and—only after the
one-time command—holdout predictions, trade ledger, candle-level equity curve, model and
baseline metrics, cost sensitivity, and an immutable evaluation manifest.

Production versions are stored separately under
`artifacts/production/versions/{model_version}/`, each with its own model, authoritative
feature order, schema hash, and training manifest.

## Execution and comparison contract

A decision is made after candle `t` closes, enters at open `t+1`, and exits at open
`t+H+1`. The backtest invests all current equity without leverage, supports fractional
quantity, forbids overlapping positions, processes exits before new close-time signals,
and applies adverse entry and exit fills plus fees on both sides.

The model is compared over an identical performance window with cash, cost-aware Buy &
Hold, EMA crossover, 24-period momentum, and deterministic random exposure. Low, base,
and high execution-cost scenarios are reported; the base scenario is the official Phase 1
result. Low- and high-cost scenarios are sensitivity checks and must not replace the base
scenario when stating the official result.

## Development setup

Python 3.11 or newer is required. XGBoost also needs a working OpenMP runtime; on macOS,
installing `libomp` through Homebrew may be necessary.

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

The complete data contracts, metric definitions, invariants, and research limitations are
defined in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
