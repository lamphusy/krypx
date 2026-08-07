"""Central configuration for the Phase 1 research pipeline."""

from pathlib import Path

PROJECT_VERSION = "0.1.0"

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

REQUIRED_DIRECTORIES = (
    DATA_RAW_DIR,
    DATA_RAW_SNAPSHOTS_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    EVALUATIONS_DIR,
    PRODUCTION_DIR,
    RUNS_DIR,
)

EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

LOOKBACK_DAYS = 3 * 365
CANDLES_PER_REQUEST = 1000

MAX_FETCH_RETRIES = 3
RETRY_BASE_SECONDS = 2.0

DROP_INCOMPLETE_LAST_CANDLE = True

PREDICTION_HORIZON = 4
LABEL_LOOKAHEAD_ROWS = PREDICTION_HORIZON + 1

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

TAKER_FEE_RATE = 0.001
SLIPPAGE_BPS_PER_SIDE = 2.0
HALF_SPREAD_BPS_PER_SIDE = 1.0
MIN_EDGE_BPS = 5.0

FINAL_HOLDOUT_RATIO = 0.20

N_WALK_FORWARD_SPLITS = 5
WALK_FORWARD_TEST_RATIO = 0.10

PURGE_GAP_ROWS = LABEL_LOOKAHEAD_ROWS

RANDOM_SEED = 42

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

INITIAL_CAPITAL = 10_000.0
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
