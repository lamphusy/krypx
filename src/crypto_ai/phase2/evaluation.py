"""Synthetic-only Milestone 9 one-time evaluation workflow.

No entry point in this module can read a real holdout or acquire a real claim.
Snapshot descriptors are hashed without parsing prices before the irreversible
Milestone 8 claim; the same pinned descriptors supply the post-claim bytes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import os
import re
import stat
import struct
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.backtesting.baselines import buy_and_hold_backtest
from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.build import compute_features
from crypto_ai.features.labels import add_labels
from crypto_ai.phase2 import backtests, dataset, experiments, holdout
from crypto_ai.phase2.evaluation_store import (
    EvaluationArtifact,
    EvaluationStore,
    _verify_development_files,
)
from crypto_ai.sentiment import aggregation as sentiment_aggregation
from crypto_ai.sentiment.aggregation import FEATURE_DTYPES, FeatureValues
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    ArticleRecord,
    ScoreRecord,
    derive_duplicate_group_id,
    format_utc_timestamp,
    validate_article_collection,
    validate_article_record,
    validate_score_record,
)
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GapAttempt,
    GroupAnchor,
    TerminalGapEvidence,
    _dedup_fingerprint,
    _expected_source_locator_at,
    _validate_terminal_gap_evidence,
)
from crypto_ai.sentiment.scoring import ScoreArtifact
from crypto_ai.sentiment.storage import (
    ContentAddressedStore,
    _fsync_directory_descriptor,
    _open_directory_path,
    _read_regular_file_at_once,
)

SPECIFICATION_ID = "phase2-milestone9-one-time-future-evaluation-v1"
DEVELOPMENT_SCHEMA = "phase2-synthetic-development-provenance-v1"
MODEL_SCHEMA = "phase2-synthetic-frozen-model-v2"
DEVELOPMENT_ROWS_SCHEMA = "phase2-synthetic-development-rows-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationError(CryptoAIError):
    """The synthetic one-time evaluation contract failed."""


class EvaluationInputError(EvaluationError):
    """An input is malformed or outside the authorized synthetic seam."""


class EvaluationPreflightError(EvaluationError):
    """Outcome-free preflight could not verify its frozen evidence."""


class EvaluationIntegrityError(EvaluationError):
    """Claimed evidence, arithmetic, or publication failed verification."""


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedNewsParent:
    """Local, replay-verified M4 holdout and M5 Development publications."""

    cas_root: Path
    aggregation_id: str
    development_dataset_id: str


@dataclass(frozen=True, slots=True)
class SyntheticEvaluationRequest:
    run_id: str
    development_run_dir: Path
    evaluation_root: Path
    boundary: holdout.BoundaryPurgePlan
    readiness: holdout.ReadinessReport
    protocol_sha256: str
    input_inventory_sha256: str
    code_commit: str
    dependency_lock_sha256: str
    market_snapshot: SnapshotRef
    article_snapshot: SnapshotRef
    score_snapshot: SnapshotRef
    feature_snapshot: SnapshotRef
    development_market: SnapshotRef | None = None
    development_rows: SnapshotRef | None = None
    aggregation_evidence: SnapshotRef | None = None
    verified_news_parent: VerifiedNewsParent | None = None
    random_simulations: int = backtests.RANDOM_SIMULATIONS
    reduced_fixture_mode: bool = False


@dataclass(slots=True)
class _PinnedSnapshot:
    reference: SnapshotRef
    descriptor: int
    initial_identity: tuple[int, int, int, int, int, int]
    raw: bytes

    def capture(self) -> bytes:
        """Return the single verified byte capture; never reread a pathname."""
        current = os.fstat(self.descriptor)
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity != self.initial_identity or not stat.S_ISREG(current.st_mode):
            raise EvaluationIntegrityError("pinned snapshot identity changed")
        return self.raw


@dataclass(slots=True)
class _PinnedDevelopmentRun:
    """Keep the verified source inode alive through claim and completion."""

    path: Path
    descriptor: int
    identity: tuple[int, int]
    files: dict[str, bytes]

    def assert_attached(self) -> None:
        current = os.fstat(self.descriptor)
        if (current.st_dev, current.st_ino) != self.identity or not stat.S_ISDIR(current.st_mode):
            raise EvaluationIntegrityError("pinned Development directory changed")
        try:
            attached = _open_directory_path(
                self.path,
                description="preflight-verified Development run",
                expected_identity=self.identity,
            )
        except (CryptoAIError, OSError) as exc:
            raise EvaluationIntegrityError(
                "Development directory was replaced after pinning"
            ) from exc
        os.close(attached)

    def verify(self) -> None:
        self.assert_attached()
        _verify_development_files(self.descriptor, self.files)
        self.assert_attached()


class _OpaqueMarketIndex:
    """Check only ordinal and UTC-close identity while price bytes stay opaque."""

    def __init__(
        self, readiness: holdout.ReadinessReport, boundary: holdout.BoundaryPurgePlan
    ) -> None:
        self.readiness = readiness
        self.boundary = boundary
        self.pending = b""
        self.next_ordinal = -1
        self.previous_open: datetime | None = None
        self.opened_at: list[datetime] = []
        self.market_candles: list[dict[str, Any]] = []

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        if len(self.pending) > 1024 * 1024 + 4096:
            raise EvaluationPreflightError("market snapshot has an overlong opaque row")
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            self._line(line)

    def _line(self, line: bytes) -> None:
        if line.endswith(b"\r"):
            raise EvaluationPreflightError("market index must use canonical LF rows")
        if self.next_ordinal == -1:
            if line != b"market_ordinal,timestamp,open,high,low,close,volume":
                raise EvaluationPreflightError("market index header differs from frozen schema")
            self.next_ordinal = 0
            return
        fields = line.split(b",", 2)
        if len(fields) != 3 or fields[0] != str(self.next_ordinal).encode("ascii"):
            raise EvaluationPreflightError("market index has an original-ordinal gap")
        try:
            opened_at = _parse_utc(fields[1].decode("ascii"))
        except UnicodeError as exc:
            raise EvaluationPreflightError("market timestamp is not ASCII UTC") from exc
        if self.previous_open is not None and opened_at - self.previous_open != timedelta(hours=1):
            raise EvaluationPreflightError("market index is not hourly-contiguous")
        if self.next_ordinal == self.boundary.first_holdout_ordinal and (
            opened_at + timedelta(hours=1) != self.boundary.first_holdout_decision_at
        ):
            raise EvaluationPreflightError("market index differs from first holdout decision")
        if self.next_ordinal >= self.readiness.market_first_ordinal:
            self.market_candles.append(
                {
                    "ordinal": self.next_ordinal,
                    "opened_at": format_utc_timestamp(opened_at),
                    "closed_at": format_utc_timestamp(opened_at + timedelta(hours=1)),
                }
            )
        self.opened_at.append(opened_at)
        self.previous_open = opened_at
        self.next_ordinal += 1

    def finish(self) -> None:
        if self.pending or self.next_ordinal - 1 != self.readiness.market_last_ordinal:
            raise EvaluationPreflightError("market index omits required closed hourly candles")


class _NewsSnapshotSchema:
    """Parse permitted news schemas during the same stream that hashes each file."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.pending = b""
        self.articles: list[ArticleRecord] = []
        self.scores: list[ScoreRecord] = []

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        if len(self.pending) > 128_000_000:
            raise EvaluationPreflightError("news snapshot record exceeds JSON bound")
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if not line:
                raise EvaluationPreflightError("news snapshot contains a blank record")
            try:
                parsed = dataset._json(line)
                if type(parsed) is not dict:
                    raise EvaluationPreflightError("news snapshot record is not an object")
                if self.kind == "article":
                    self.articles.append(validate_article_record(parsed))
                else:
                    self.scores.append(validate_score_record(parsed))
            except CryptoAIError as exc:
                raise EvaluationPreflightError(
                    "news snapshot schema or identity is invalid"
                ) from exc

    def finish(self) -> None:
        if self.pending:
            raise EvaluationPreflightError("news snapshot is missing its final LF")


class _FeatureHeader:
    """Verify the saved feature schema without reading decision values pre-claim."""

    def __init__(self) -> None:
        self.pending = b""
        self.verified = False

    def feed(self, chunk: bytes) -> None:
        if self.verified:
            return
        self.pending += chunk
        if b"\n" in self.pending:
            header, _ = self.pending.split(b"\n", 1)
            if len(header) > 4096:
                raise EvaluationPreflightError("feature header is overlong")
            expected = ("market_ordinal,decision_at," + ",".join(dataset.COMBINED_COLUMNS)).encode()
            if header != expected:
                raise EvaluationPreflightError("holdout feature column order changed")
            self.pending = b""
            self.verified = True
        elif len(self.pending) > 4096:
            raise EvaluationPreflightError("feature header is overlong or missing")

    def finish(self) -> None:
        if not self.verified:
            raise EvaluationPreflightError("holdout feature header is absent")


def _pin_snapshot(
    reference: SnapshotRef,
    *,
    market_index: _OpaqueMarketIndex | None = None,
    news_schema: _NewsSnapshotSchema | None = None,
    feature_header: _FeatureHeader | None = None,
    development: _PinnedDevelopmentRun | None = None,
) -> _PinnedSnapshot:
    if type(reference) is not SnapshotRef or not _sha256(reference.sha256):
        raise EvaluationInputError("a synthetic snapshot reference is malformed")
    _safe_temporary_path(reference.path, existing=True)
    if development is None:
        parent_descriptor = _open_directory_path(
            reference.path.parent, description="synthetic snapshot parent"
        )
    else:
        if reference.path.parent != development.path:
            raise EvaluationPreflightError("development evidence is outside the frozen run")
        development.assert_attached()
        parent_descriptor = os.dup(development.descriptor)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(reference.path.name, flags, dir_fd=parent_descriptor)
        identity_stat = os.fstat(descriptor)
        if not stat.S_ISREG(identity_stat.st_mode):
            raise EvaluationPreflightError("snapshot is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            if market_index is not None:
                market_index.feed(chunk)
            if news_schema is not None:
                news_schema.feed(chunk)
            if feature_header is not None:
                feature_header.feed(chunk)
        if digest.hexdigest() != reference.sha256:
            raise EvaluationPreflightError("snapshot SHA-256 differs from frozen reference")
        if market_index is not None:
            market_index.finish()
        if news_schema is not None:
            news_schema.finish()
        if feature_header is not None:
            feature_header.finish()
        identity = (
            identity_stat.st_dev,
            identity_stat.st_ino,
            identity_stat.st_mode,
            identity_stat.st_size,
            identity_stat.st_mtime_ns,
            identity_stat.st_ctime_ns,
        )
        result = _PinnedSnapshot(reference, descriptor, identity, b"".join(chunks))
        result.capture()
        descriptor = None
        return result
    except EvaluationError:
        raise
    except OSError as exc:
        raise EvaluationPreflightError("snapshot is unreadable or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _development_manifest(
    request: SyntheticEvaluationRequest, development: _PinnedDevelopmentRun
) -> tuple[dict, bytes]:
    path = request.development_run_dir / "development_manifest.json"
    development.assert_attached()
    _safe_temporary_path(path, existing=True)
    raw, _ = _read_regular_file_at_once(
        development.descriptor, path.name, description="frozen synthetic development manifest"
    )
    parsed = dataset._json(raw)
    fit = request.boundary.frozen_fit
    required = {
        "schema_version",
        "synthetic",
        "run_id",
        "development_cutoff_ordinal",
        "purge_ordinals",
        "first_holdout_ordinal",
        "selected_augmented_cell",
        "matched_control_cell",
        "shared_labeled_rows_sha256",
        "fit_manifest_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "augmented_fit_count",
        "control_fit_count",
        "fitted_max_decision_ordinal",
        "augmented_feature_columns",
        "control_feature_columns",
        "signal_threshold",
        "development_market_sha256",
        "development_rows_sha256",
        "model_family",
    }
    expected = {
        "schema_version": DEVELOPMENT_SCHEMA,
        "synthetic": True,
        "run_id": request.development_run_dir.name,
        "development_cutoff_ordinal": request.boundary.development_cutoff_ordinal,
        "purge_ordinals": list(request.boundary.purge_ordinals),
        "first_holdout_ordinal": request.boundary.first_holdout_ordinal,
        "selected_augmented_cell": fit.selected_augmented_cell,
        "matched_control_cell": fit.matched_control_cell,
        "shared_labeled_rows_sha256": fit.shared_labeled_rows_sha256,
        "fit_manifest_sha256": fit.fit_manifest_sha256,
        "augmented_model_sha256": fit.augmented_model_sha256,
        "control_model_sha256": fit.control_model_sha256,
        "augmented_fit_count": 1,
        "control_fit_count": 1,
        "fitted_max_decision_ordinal": request.boundary.development_cutoff_ordinal,
        "augmented_feature_columns": list(dataset.COMBINED_COLUMNS),
        "control_feature_columns": list(dataset.TECHNICAL_COLUMNS),
        "signal_threshold": 0.5,
        "development_market_sha256": (
            request.development_market.sha256
            if type(request.development_market) is SnapshotRef
            else None
        ),
        "development_rows_sha256": (
            request.development_rows.sha256
            if type(request.development_rows) is SnapshotRef
            else None
        ),
        "model_family": (
            "LogisticRegression" if fit.selected_augmented_cell == "C" else "XGBClassifier"
        ),
    }
    if type(parsed) is not dict or set(parsed) != required or parsed != expected:
        raise EvaluationPreflightError("development manifest does not prove a frozen single fit")
    return parsed, raw


def _development_rows_payload(market_raw: bytes, cutoff: int) -> dict[str, Any]:
    """Rebuild labeled development decisions from only pre-holdout market bytes."""
    if type(cutoff) is not int or cutoff < 0:
        raise EvaluationPreflightError("development cutoff is invalid")
    market = _market_frame(market_raw, cutoff + backtests.HORIZON + 1)
    technical = compute_features(market)
    threshold = minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0)
    labeled = add_labels(technical, horizon=backtests.HORIZON, minimum_required_return=threshold)
    if cutoff not in labeled.index or any(index > cutoff for index in labeled.index):
        raise EvaluationPreflightError("development labels extend beyond the frozen cutoff")
    no_news: dict[str, int | float] = {
        name: (
            24.0
            if name == "hours_since_latest_article"
            else 1 if name == "news_missing_24h" else 0.0 if dtype == "float64" else 0
        )
        for name, dtype in FEATURE_DTYPES
    }
    rows = []
    for ordinal, row in labeled.iterrows():
        decision_at = row["timestamp"].to_pydatetime() + timedelta(hours=1)
        features = [float(row[name]) for name in dataset.TECHNICAL_COLUMNS]
        features.extend(no_news[name] for name in dataset.SENTIMENT_COLUMNS)
        rows.append(
            {
                "market_ordinal": int(ordinal),
                "decision_at": format_utc_timestamp(decision_at),
                "exit_ordinal": int(ordinal) + backtests.HORIZON + 1,
                "exit_at": format_utc_timestamp(
                    market.iloc[int(ordinal) + backtests.HORIZON + 1]["timestamp"].to_pydatetime()
                ),
                "label": int(row["label"]),
                "features": features,
            }
        )
    if not rows or rows[-1]["market_ordinal"] != cutoff:
        raise EvaluationPreflightError("development prepared rows omit the cutoff")
    return {
        "schema_version": DEVELOPMENT_ROWS_SCHEMA,
        "synthetic": True,
        "market_sha256": sha256_bytes(market_raw),
        "feature_columns": list(dataset.COMBINED_COLUMNS),
        "rows": rows,
    }


def _development_dataset_manifest_bytes(market_raw: bytes, rows_raw: bytes, cutoff: int) -> bytes:
    """Derive a path-independent manifest from replay-verified Development bytes."""
    if type(market_raw) is not bytes or type(rows_raw) is not bytes or type(cutoff) is not int:
        raise EvaluationInputError("development dataset inputs are malformed")
    if cutoff < 0:
        raise EvaluationInputError("development cutoff is invalid")
    return canonicalize(
        {
            "schema_version": "phase2-synthetic-development-dataset-manifest-v1",
            "synthetic": True,
            "development_cutoff_ordinal": cutoff,
            "market_snapshot_sha256": sha256_bytes(market_raw),
            "labeled_dataset_sha256": sha256_bytes(rows_raw),
            "technical_columns": list(dataset.TECHNICAL_COLUMNS),
            "sentiment_columns": list(dataset.SENTIMENT_COLUMNS),
            "label_columns": ["label"],
            "combined_feature_columns": list(dataset.COMBINED_COLUMNS),
        }
    )


def _fitted_model_artifact(rows: list[dict[str, Any]], *, cell: str, rows_sha256: str) -> bytes:
    columns = dataset.COMBINED_COLUMNS if cell in ("C", "D") else dataset.TECHNICAL_COLUMNS
    matrix = np.asarray(
        [
            [row["features"][dataset.COMBINED_COLUMNS.index(name)] for name in columns]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    if not np.array_equal(np.unique(labels), [0, 1]) or not np.isfinite(matrix).all():
        raise EvaluationPreflightError("development fit needs finite rows with both labels")
    model = experiments._make_model(cell)
    try:
        model.fit(pd.DataFrame(matrix, columns=columns), labels)
    except Exception as exc:
        raise EvaluationPreflightError("frozen development model cannot be refitted") from exc
    family = "LogisticRegression" if cell in ("A", "C") else "XGBClassifier"
    common = {
        "schema_version": MODEL_SCHEMA,
        "synthetic": True,
        "model_family": family,
        "feature_columns": list(columns),
        "training_rows_sha256": rows_sha256,
        "hyperparameters": (
            experiments.LOGISTIC_PARAMS
            if family == "LogisticRegression"
            else experiments.XGBOOST_PARAMS
        ),
        "classes": [0, 1],
    }
    if family == "LogisticRegression":
        scaler, classifier = model.named_steps["scaler"], model.named_steps["classifier"]
        common.update(
            {
                "scaler_parameters": experiments.SCALER_PARAMS,
                "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
                "scaler_var": scaler.var_.astype(np.float64).tolist(),
                "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
                "coefficients": classifier.coef_[0].astype(np.float64).tolist(),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    else:
        common["booster_ubj_hex"] = bytes(model.get_booster().save_raw(raw_format="ubj")).hex()
    return canonicalize(common)


def build_synthetic_development_fit(
    development_market_raw: bytes,
    cutoff: int,
    pair: tuple[str, str] = ("C", "A"),
) -> tuple[bytes, bytes, bytes]:
    """Generate/replay synthetic prepared rows and both frozen fit artifacts."""
    if type(development_market_raw) is not bytes or pair not in {("C", "A"), ("D", "B")}:
        raise EvaluationInputError("only synthetic matched model pairs are supported")
    payload = _development_rows_payload(development_market_raw, cutoff)
    rows_raw = canonicalize(payload)
    rows_sha256 = sha256_bytes(rows_raw)
    return (
        rows_raw,
        _fitted_model_artifact(payload["rows"], cell=pair[0], rows_sha256=rows_sha256),
        _fitted_model_artifact(payload["rows"], cell=pair[1], rows_sha256=rows_sha256),
    )


def _sha256(value: object) -> bool:
    return type(value) is str and SHA256_PATTERN.fullmatch(value) is not None


def _safe_temporary_path(path: Path, *, existing: bool) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise EvaluationInputError("synthetic paths must be absolute Path objects")
    try:
        resolved = path.resolve(strict=existing)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvaluationInputError("synthetic path cannot be resolved") from exc
    if resolved == temp_root or not resolved.is_relative_to(temp_root):
        raise EvaluationInputError("real development and holdout paths are not authorized")
    return resolved


def _parse_utc(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise EvaluationInputError("timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EvaluationInputError("invalid UTC timestamp") from exc
    if parsed.tzinfo != UTC or format_utc_timestamp(parsed) != value:
        raise EvaluationInputError("noncanonical UTC timestamp")
    return parsed


def _finite_number(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        raise EvaluationInputError(f"{field} must be finite")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvaluationInputError(f"{field} must be representable as float64") from exc
    if not math.isfinite(converted):
        raise EvaluationInputError(f"{field} must be finite")
    return converted


def _ledger_exits(ledger: object) -> dict[datetime, float]:
    if type(ledger) not in (list, tuple):
        raise EvaluationInputError("trade ledger must be an ordered sequence")
    exits: dict[datetime, float] = {}
    previous: datetime | None = None
    for row in ledger:
        if type(row) is not dict or "exit_timestamp" not in row or "pnl" not in row:
            raise EvaluationInputError("trade ledger has an invalid exit row")
        exit_at = _parse_utc(row["exit_timestamp"])
        if previous is not None and exit_at <= previous:
            raise EvaluationInputError("trade exits must be unique and chronological")
        exits[exit_at] = _finite_number(row["pnl"], field="trade PnL")
        previous = exit_at
    return exits


def rolling_incremental_concentration(
    augmented_ledger: object, control_ledger: object
) -> float | None:
    """Maximum 30-day share of positive net-currency incremental trade PnL."""
    augmented = _ledger_exits(augmented_ledger)
    control = _ledger_exits(control_ledger)
    exits = sorted(set(augmented) | set(control))
    if not exits:
        return None
    increments = {}
    for at in exits:
        difference = augmented.get(at, 0.0) - control.get(at, 0.0)
        if not math.isfinite(difference):
            raise EvaluationInputError("incremental trade PnL exceeds finite range")
        increments[at] = max(0.0, difference)
    try:
        denominator = math.fsum(increments.values())
    except OverflowError as exc:
        raise EvaluationInputError("incremental trade PnL total exceeds finite range") from exc
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None
    day = exits[0].date()
    last = exits[-1].date()
    largest = 0.0
    while day <= last:
        window_start = datetime.combine(day, datetime.min.time(), UTC)
        try:
            window_end = window_start + timedelta(days=30)
        except OverflowError:
            window_end = None
        try:
            numerator = math.fsum(
                value
                for at, value in increments.items()
                if window_start <= at and (window_end is None or at < window_end)
            )
        except OverflowError as exc:
            raise EvaluationInputError(
                "rolling incremental trade PnL exceeds finite range"
            ) from exc
        largest = max(largest, numerator / denominator)
        if day == last:
            break
        day += timedelta(days=1)
    return largest


def evaluate_final_gates(
    *,
    augmented_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    cash_metrics: dict[str, Any],
    augmented_ledger: object,
    control_ledger: object,
    elapsed_days: float,
    planned_minimum_days: int,
) -> dict[str, Any]:
    """Evaluate only the exact frozen conjunctive base-cost research gates."""
    if any(type(value) is not dict for value in (augmented_metrics, control_metrics, cash_metrics)):
        raise EvaluationInputError("base-cost metrics must be dictionaries")
    if (
        type(elapsed_days) is not float
        or not math.isfinite(elapsed_days)
        or type(planned_minimum_days) is not int
        or planned_minimum_days < 180
    ):
        raise EvaluationInputError("invalid frozen readiness duration")
    keys = ("total_return", "sharpe_ratio", "profit_factor", "maximum_drawdown", "num_trades")
    if (
        any(key not in augmented_metrics for key in keys)
        or any(key not in control_metrics for key in ("total_return", "maximum_drawdown"))
        or "total_return" not in cash_metrics
    ):
        raise EvaluationInputError("required final-gate metric missing")
    for source in (augmented_metrics, control_metrics, cash_metrics):
        for key, value in source.items():
            if key in keys and value is not None:
                _finite_number(value, field=key)
    n = augmented_metrics["num_trades"]
    if type(n) is not int or n < 0:
        raise EvaluationInputError("completed trade count must be a nonnegative integer")
    if n != len(_ledger_exits(augmented_ledger)):
        raise EvaluationInputError("completed trade count differs from the retained trade ledger")
    concentration = rolling_incremental_concentration(augmented_ledger, control_ledger)

    def valid_positive(value: object, floor: float) -> bool:
        return (
            type(value) in (int, float) and _finite_number(value, field="final gate metric") > floor
        )

    aug_return = augmented_metrics["total_return"]
    ctl_return = control_metrics["total_return"]
    cash_return = cash_metrics["total_return"]
    aug_drawdown = augmented_metrics["maximum_drawdown"]
    ctl_drawdown = control_metrics["maximum_drawdown"]
    for name, value in (
        ("augmented return", aug_return),
        ("control return", ctl_return),
        ("cash return", cash_return),
    ):
        _finite_number(value, field=name)
    if type(aug_drawdown) not in (int, float) or type(ctl_drawdown) not in (int, float):
        raise EvaluationInputError("maximum drawdown must be defined")
    if aug_drawdown > 0.0 or ctl_drawdown > 0.0:
        raise EvaluationInputError("maximum drawdown has an invalid sign")
    gates = {
        "planned_duration": elapsed_days >= planned_minimum_days,
        "completed_trades": n >= 50,
        "positive_return": valid_positive(aug_return, 0.0),
        "beats_cash": aug_return > cash_return,
        "beats_control": aug_return > ctl_return,
        "positive_sharpe": valid_positive(augmented_metrics["sharpe_ratio"], 0.0),
        "profit_factor": valid_positive(augmented_metrics["profit_factor"], 1.05),
        "maximum_drawdown": abs(aug_drawdown) <= 0.20,
        "drawdown_vs_control": Decimal(str(abs(aug_drawdown)))
        <= Decimal(str(abs(ctl_drawdown))) + Decimal("0.02"),
        "rolling_incremental_concentration": concentration is not None and concentration <= 0.40,
    }
    operands = {
        "elapsed_days": elapsed_days,
        "planned_minimum_days": planned_minimum_days,
        "completed_trades": n,
        "augmented_total_return": aug_return,
        "control_total_return": ctl_return,
        "cash_total_return": cash_return,
        "augmented_sharpe": augmented_metrics["sharpe_ratio"],
        "augmented_profit_factor": augmented_metrics["profit_factor"],
        "augmented_drawdown_magnitude": abs(aug_drawdown),
        "control_drawdown_magnitude": abs(ctl_drawdown),
        "rolling_incremental_concentration": concentration,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "schema_version": "phase2-final-evidence-gates-v1",
        "synthetic": True,
        "official_scenario": "base",
        "operands": operands,
        "gates": gates,
        "research_verdict": verdict,
        "production_decision": "NO-GO",
    }
    canonicalize(result)
    return result


@dataclass(frozen=True, slots=True)
class _FrozenModel:
    family: str
    columns: tuple[str, ...]
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    coefficients: np.ndarray | None
    intercept: float | None
    booster_ubj: bytes | None


def _verified_model(raw: bytes, columns: tuple[str, ...]) -> _FrozenModel:
    parsed = dataset._json(raw)
    if (
        type(parsed) is not dict
        or parsed.get("schema_version") != MODEL_SCHEMA
        or parsed.get("synthetic") is not True
        or parsed.get("feature_columns") != list(columns)
        or not _sha256(parsed.get("training_rows_sha256"))
        or parsed.get("classes") != [0, 1]
        or parsed.get("model_family") not in {"LogisticRegression", "XGBClassifier"}
    ):
        raise EvaluationPreflightError("frozen synthetic model schema or order mismatch")
    family = parsed["model_family"]
    common = {
        "schema_version",
        "synthetic",
        "model_family",
        "feature_columns",
        "training_rows_sha256",
        "hyperparameters",
        "classes",
    }
    if family == "LogisticRegression":
        if (
            set(parsed)
            != common
            | {
                "scaler_parameters",
                "scaler_mean",
                "scaler_var",
                "scaler_scale",
                "coefficients",
                "intercept",
            }
            or parsed["hyperparameters"] != experiments.LOGISTIC_PARAMS
            or parsed["scaler_parameters"] != experiments.SCALER_PARAMS
            or any(
                type(parsed[key]) is not list or len(parsed[key]) != len(columns)
                for key in ("scaler_mean", "scaler_var", "scaler_scale", "coefficients")
            )
        ):
            raise EvaluationPreflightError("frozen logistic model/scaler contract changed")
        mean, variance, scale, coefficients = (
            np.asarray([_finite_number(x, field=key) for x in parsed[key]], dtype=np.float64)
            for key in ("scaler_mean", "scaler_var", "scaler_scale", "coefficients")
        )
        if (variance < 0).any() or (scale <= 0).any():
            raise EvaluationPreflightError("frozen logistic scaler has invalid variance/scale")
        return _FrozenModel(
            family,
            columns,
            mean,
            scale,
            coefficients,
            _finite_number(parsed["intercept"], field="frozen intercept"),
            None,
        )
    if (
        set(parsed) != common | {"booster_ubj_hex"}
        or parsed["hyperparameters"] != experiments.XGBOOST_PARAMS
        or type(parsed["booster_ubj_hex"]) is not str
        or len(parsed["booster_ubj_hex"]) > 100_000_000
    ):
        raise EvaluationPreflightError("frozen XGBoost model contract changed")
    try:
        booster_ubj = bytes.fromhex(parsed["booster_ubj_hex"])
    except ValueError as exc:
        raise EvaluationPreflightError("frozen XGBoost booster is not hex") from exc
    if not booster_ubj:
        raise EvaluationPreflightError("frozen XGBoost booster is empty")
    return _FrozenModel(family, columns, None, None, None, None, booster_ubj)


def _csv_rows(raw: bytes, expected_header: tuple[str, ...]) -> list[list[str]]:
    if type(raw) is not bytes:
        raise EvaluationIntegrityError("exact CSV bytes are required")
    try:
        stream = io.StringIO(raw.decode("utf-8"), newline="")
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if tuple(header) != expected_header:
            raise EvaluationIntegrityError("CSV column order differs from frozen schema")
        rows = list(reader)
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise EvaluationIntegrityError("CSV bytes are malformed") from exc
    if not rows or any(len(row) != len(expected_header) for row in rows):
        raise EvaluationIntegrityError("CSV has an empty or malformed row set")
    return rows


def _market_frame(raw: bytes, expected_last_ordinal: int | None) -> pd.DataFrame:
    rows = _csv_rows(raw, ("market_ordinal", "timestamp", "open", "high", "low", "close", "volume"))
    timestamps: list[pd.Timestamp] = []
    prices: dict[str, list[float]] = {
        name: [] for name in ("open", "high", "low", "close", "volume")
    }
    for expected_ordinal, row in enumerate(rows):
        if row[0] != str(expected_ordinal):
            raise EvaluationIntegrityError("market snapshot has an original-ordinal gap")
        opened_at = _parse_utc(row[1])
        if timestamps and opened_at - timestamps[-1].to_pydatetime() != timedelta(hours=1):
            raise EvaluationIntegrityError("market snapshot is not hourly-contiguous")
        try:
            values = [float(value) for value in row[2:]]
        except ValueError as exc:
            raise EvaluationIntegrityError("market OHLCV is malformed") from exc
        opened, high, low, close, volume = values
        if (
            not all(math.isfinite(value) for value in values)
            or min(opened, high, low, close) <= 0.0
            or volume < 0.0
            or high < low
            or not low <= opened <= high
            or not low <= close <= high
        ):
            raise EvaluationIntegrityError("market OHLCV violates authentic candle bounds")
        timestamps.append(pd.Timestamp(opened_at))
        for name, value in zip(prices, values, strict=True):
            prices[name].append(value)
    if expected_last_ordinal is not None and (
        type(expected_last_ordinal) is not int or len(rows) - 1 != expected_last_ordinal
    ):
        raise EvaluationIntegrityError("market snapshot differs from readiness market bounds")
    frame = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps),
            **{name: np.asarray(values, dtype=np.float64) for name, values in prices.items()},
        }
    )
    backtests._market_invariants(frame)
    return frame


@dataclass(frozen=True, slots=True)
class _SyntheticCoverageInterval:
    start_at: str
    end_at_exclusive: str
    outcome: str


def _news_parent_bytes(parent: VerifiedNewsParent) -> bytes:
    if (
        type(parent) is not VerifiedNewsParent
        or not isinstance(parent.cas_root, Path)
        or not _sha256(parent.aggregation_id)
        or not _sha256(parent.development_dataset_id)
    ):
        raise EvaluationPreflightError("verified news parent descriptor is malformed")
    _safe_temporary_path(parent.cas_root, existing=True)
    return canonicalize(
        {
            "schema_version": "phase2-verified-news-parents-v1",
            "synthetic": True,
            "cas_root": str(parent.cas_root),
            "aggregation_id": parent.aggregation_id,
            "development_dataset_id": parent.development_dataset_id,
        }
    )


def _parse_news_parent(raw: bytes) -> VerifiedNewsParent:
    try:
        value = dataset._json(raw)
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema_version",
                "synthetic",
                "cas_root",
                "aggregation_id",
                "development_dataset_id",
            }
            or value["schema_version"] != "phase2-verified-news-parents-v1"
            or value["synthetic"] is not True
            or type(value["cas_root"]) is not str
            or not Path(value["cas_root"]).is_absolute()
        ):
            raise EvaluationPreflightError("retained news parent descriptor is malformed")
        parent = VerifiedNewsParent(
            Path(value["cas_root"]), value["aggregation_id"], value["development_dataset_id"]
        )
        if _news_parent_bytes(parent) != raw:
            raise EvaluationPreflightError("retained news parent is noncanonical")
        return parent
    except EvaluationError:
        raise
    except (CryptoAIError, ValueError, TypeError, KeyError, OSError) as exc:
        raise EvaluationPreflightError("retained news parent cannot be parsed") from exc


def _feature_rows_bits(
    rows: dict[int, FeatureValues | None],
) -> tuple[tuple[int, tuple[tuple[str, bytes | int], ...] | None], ...]:
    """Compare causal M4 values at exact IEEE-754 binary64 precision."""
    result = []
    for ordinal, row in sorted(rows.items()):
        if row is None:
            result.append((ordinal, None))
            continue
        values = row.to_dict()
        result.append(
            (
                ordinal,
                tuple(
                    (
                        name,
                        (
                            struct.pack(">d", float(values[name]))
                            if dtype == "float64"
                            else int(values[name])
                        ),
                    )
                    for name, dtype in FEATURE_DTYPES
                ),
            )
        )
    return tuple(result)


def _verified_news_parent_expectations(
    parent: VerifiedNewsParent,
    *,
    evidence_raw: bytes,
    articles: tuple[ArticleRecord, ...],
    scores: tuple[ScoreRecord, ...],
    opened_at: list[datetime],
    first_ordinal: int,
    last_decision_ordinal: int,
    development_market_raw: bytes,
    development_rows_raw: bytes,
    protocol_sha256: str,
) -> dict[int, FeatureValues | None]:
    """Replay real immutable M4/M5 publications, including GSG and mock-score parents.

    Only the Development M5 publication is opened here; its labels are compared
    with the independently replayed Development prefix. Holdout prices stay opaque.
    """
    try:
        _news_parent_bytes(parent)
        cas = ContentAddressedStore(parent.cas_root)
        frozen_development_market = _market_frame(development_market_raw, None)

        def validate_development_metadata(value: object) -> None:
            if (
                type(value) is not dict
                or set(value) != {"schema_version", "dataset_id", "manifest_sha256"}
                or value["schema_version"] != dataset.SCHEMA
                or value["dataset_id"] != parent.development_dataset_id
                or value["manifest_sha256"] != parent.development_dataset_id
            ):
                raise EvaluationPreflightError("M5 publication metadata is not frozen")

        opaque_development = cas.read_publication(
            dataset.PUBLICATION_PREFIX + parent.development_dataset_id,
            metadata_prevalidator=validate_development_metadata,
        )
        opaque_input = dataset._json(opaque_development.files["input.json"])
        expected_generator = {
            "start_at": format_utc_timestamp(
                frozen_development_market.iloc[0].timestamp.to_pydatetime()
            ),
            "hours": len(frozen_development_market),
        }
        if (
            type(opaque_input) is not dict
            or type(opaque_input.get("market")) is not dict
            or any(
                opaque_input["market"].get(name) != value
                for name, value in expected_generator.items()
            )
        ):
            raise EvaluationPreflightError(
                "M5 parent is not the Development-only market generation"
            )
        aggregation = sentiment_aggregation.AggregationStore(cas).get(parent.aggregation_id)
        development = dataset.DatasetStore(cas).get(parent.development_dataset_id)
        if aggregation is None or development is None:
            raise EvaluationPreflightError("immutable news parent publication is missing")
        agg_input = dataset._json(dict(aggregation.files)["input.json"])
        if agg_input["protocol_config_sha256"] != protocol_sha256:
            raise EvaluationPreflightError("M4 protocol differs from frozen evaluation protocol")
        artifacts = tuple(
            ScoreArtifact(tuple(sorted((name, bytes.fromhex(raw)) for name, raw in entry.items())))
            for entry in agg_input["score_artifacts"]
        )
        prepared = sentiment_aggregation._prepare(
            cas,
            sentiment_aggregation.SyntheticAggregationInput(
                agg_input["state_publication_id"],
                artifacts,
                agg_input["coverage_as_of"],
                agg_input["protocol_config_sha256"],
            ),
        )
        stated_evidence = dataset._json(evidence_raw)
        state_files = dict(prepared.state_files)
        if (
            type(stated_evidence) is not dict
            or stated_evidence.get("terminal_gap_evidence")
            != dataset._json(state_files["gap-evidence.json"])
            or stated_evidence.get("coverage_start_at") != prepared.terminal_intervals[0].start_at
            or stated_evidence.get("coverage_end_at_exclusive")
            != prepared.terminal_intervals[-1].end_at_exclusive
        ):
            raise EvaluationPreflightError(
                "claimed gap/coverage evidence differs from immutable M4 chronology"
            )
        if (
            len({item.article_version_id for item in articles}) != len(articles)
            or len({item.content_hash for item in scores}) != len(scores)
            or len({item.article_version_id for item in prepared.articles})
            != len(prepared.articles)
            or len({item.content_hash for item in prepared.scores}) != len(prepared.scores)
        ):
            raise EvaluationPreflightError("M4 news identity or score identity competes")
        if {item.article_version_id: item.to_dict() for item in prepared.articles} != {
            item.article_version_id: item.to_dict() for item in articles
        } or {item.content_hash: item.to_dict() for item in prepared.scores} != {
            item.content_hash: item.to_dict() for item in scores
        }:
            raise EvaluationPreflightError(
                "article or score snapshot differs from replay-verified raw/response parents"
            )
        development_files = dict(development.files)
        development_market = dataset._frame(
            development_files["market-price-context.json"], dataset.RAW_COLUMNS
        )
        if (
            not development_market.equals(frozen_development_market)
            or development.manifest["market_snapshot_sha256"]
            != sha256_bytes(development_files["market.json"])
            or sha256_bytes(development_files["protocol.json"]) != protocol_sha256
        ):
            raise EvaluationPreflightError(
                "M5 Development parent is detached from the frozen Development market"
            )
        # Partial window overlap still shares immutable minute-level facts.
        # A different corpus cannot be hidden by moving the M4 decision grid
        # beyond Development or by omitting the first part of its 24h windows.
        development_chronology = dataset._json(
            development_files["parents/articles/chronology.json"]
        )["terminal_intervals"]
        development_intervals = {item["start_at"]: item for item in development_chronology}
        for interval in prepared.terminal_intervals:
            previous = development_intervals.get(interval.start_at)
            if previous is not None and previous != interval.to_dict():
                raise EvaluationIntegrityError(
                    "M4/M5 overlapping immutable news chronology disagrees"
                )
        # The independently valid parents must describe the same causal history
        # wherever M4 certifies the complete trailing window of an M5 decision.
        # Replay from M4 chronology, rather than intersecting published decision
        # grids: a holdout-only M4 grid can still certify Development decisions.
        coverage_start = _parse_utc(prepared.terminal_intervals[0].start_at)
        coverage_end = _parse_utc(prepared.terminal_intervals[-1].end_at_exclusive)
        development_news: dict[int, FeatureValues | None] = {}
        development_decisions: dict[int, str] = {}
        for row in development.features.to_dict("records"):
            ordinal = int(row["market_ordinal"])
            development_decisions[ordinal] = format_utc_timestamp(
                row["decision_at"].to_pydatetime()
            )
            development_news[ordinal] = FeatureValues(
                **{name: row[name] for name, _ in FEATURE_DTYPES}
            )
        for exclusion in dataset._json(development_files["exclusions.json"]):
            if "provider_gap_window" in exclusion["reasons"]:
                ordinal = exclusion["market_ordinal"]
                development_decisions[ordinal] = exclusion["decision_at"]
                development_news[ordinal] = None
        # Provider receipts alone do not bind deduplication or scoring. Shared
        # article versions and permanent anchors must retain their identities,
        # even when neither publication exposes a complete common window.
        previous_articles = dataset._json(development_files["parents/articles/articles.json"])
        previous_versions = {item["article_version_id"]: item for item in previous_articles}
        previous_membership = {
            item["article_id"]: item["duplicate_group_id"] for item in previous_articles
        }
        previous_groups = {
            item["duplicate_group_id"]: item
            for item in dataset._json(development_files["parents/articles/groups.json"])
        }
        groups = {item.duplicate_group_id: item.to_dict() for item in prepared.groups}
        for article in prepared.articles:
            previous = previous_versions.get(article.article_version_id)
            if previous is not None and previous != article.to_dict():
                raise EvaluationIntegrityError("M4/M5 overlapping article versions disagree")
            previous_group = previous_membership.get(article.article_id)
            if previous_group is not None and (
                previous_group != article.duplicate_group_id
                or previous_groups[previous_group] != groups[article.duplicate_group_id]
            ):
                raise EvaluationIntegrityError("M4/M5 overlapping permanent anchors disagree")
        previous_scores = {
            item["content_hash"]: item
            for name, raw in development_files.items()
            if name.startswith("parents/scores/") and name.endswith("/record.json")
            for item in (dataset._json(raw),)
        }
        scores_by_content = {item.content_hash: item.to_dict() for item in prepared.scores}
        shared_content = {item["content_hash"] for item in previous_articles} & {
            item.content_hash for item in prepared.articles
        }
        last_development_decision = max(map(_parse_utc, development_decisions.values()))
        for content_hash in shared_content:
            before = previous_scores.get(content_hash)
            after = scores_by_content.get(content_hash)
            # Adding a previously absent score after Development is causal;
            # changing an existing immutable score is never a cache update.
            if before != after and (
                (before is not None and after is not None)
                or any(
                    item is not None and _parse_utc(item["scored_at"]) <= last_development_decision
                    for item in (before, after)
                )
            ):
                raise EvaluationIntegrityError("M4/M5 overlapping causal score records disagree")
        overlap = {
            ordinal: decision
            for ordinal, decision in sorted(development_decisions.items())
            if coverage_start <= _parse_utc(decision) - timedelta(hours=24)
            and _parse_utc(decision) < coverage_end
        }
        replayed = (
            sentiment_aggregation._aggregate_rows(prepared, tuple(overlap.values()))
            if overlap
            else ()
        )
        if _feature_rows_bits(
            {ordinal: row.features for ordinal, row in zip(overlap, replayed, strict=True)}
        ) != _feature_rows_bits({ordinal: development_news[ordinal] for ordinal in overlap}):
            raise EvaluationIntegrityError(
                "M4/M5 overlapping Development news features or gap exclusions disagree"
            )
        development_rows = dataset._json(development_rows_raw)["rows"]
        labeled = development.labeled
        if len(labeled) != len(development_rows):
            raise EvaluationPreflightError("M5 Development label row count differs")
        for expected, (_, observed) in zip(development_rows, labeled.iterrows(), strict=True):
            if (
                int(observed["market_ordinal"]) != expected["market_ordinal"]
                or format_utc_timestamp(observed["decision_at"].to_pydatetime())
                != expected["decision_at"]
                or int(observed["label"]) != expected["label"]
                or [float(observed[name]) for name in dataset.COMBINED_COLUMNS]
                != [float(value) for value in expected["features"]]
            ):
                raise EvaluationPreflightError(
                    "M5 Development labels/features differ from frozen fit rows"
                )
        rows = aggregation.rows
        if len(rows) != last_decision_ordinal - first_ordinal + 1:
            raise EvaluationPreflightError("M4 decision grid does not cover the holdout")
        result: dict[int, FeatureValues | None] = {}
        for ordinal, row in zip(range(first_ordinal, last_decision_ordinal + 1), rows, strict=True):
            if row.decision_at != format_utc_timestamp(opened_at[ordinal] + timedelta(hours=1)):
                raise EvaluationPreflightError("M4 decision is not the exact holdout close")
            result[ordinal] = row.features
        return result
    except EvaluationError:
        raise
    except (
        CryptoAIError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        OSError,
        AttributeError,
        RecursionError,
    ) as exc:
        raise EvaluationPreflightError("immutable M4/M5 news parent replay failed") from exc


def _aggregation_expectations(
    evidence_raw: bytes,
    *,
    articles: tuple[ArticleRecord, ...],
    scores: tuple[ScoreRecord, ...],
    article_sha256: str,
    score_sha256: str,
    feature_sha256: str,
    protocol_sha256: str,
    opened_at: list[datetime],
    first_ordinal: int,
    last_decision_ordinal: int,
    inspected_at: datetime,
) -> dict[int, FeatureValues | None]:
    """Replay a claimed projection; authoritative M4/M5 replay is separate."""
    try:
        evidence = dataset._json(evidence_raw)
        required = {
            "schema_version",
            "synthetic",
            "article_snapshot_sha256",
            "score_snapshot_sha256",
            "feature_snapshot_sha256",
            "aggregation_config_sha256",
            "coverage_start_at",
            "coverage_end_at_exclusive",
            "groups",
            "terminal_gap_evidence",
            "prepared_dataset_manifest",
            "parent_manifest_sha256",
        }
        if type(evidence) is not dict or set(evidence) != required:
            raise EvaluationPreflightError("synthetic aggregation parent schema differs")
        parent = {key: value for key, value in evidence.items() if key != "parent_manifest_sha256"}
        if (
            evidence["schema_version"] != "phase2-synthetic-aggregation-parent-v1"
            or evidence["synthetic"] is not True
            or canonicalize(evidence) != evidence_raw
            or canonical_sha256(parent) != evidence["parent_manifest_sha256"]
            or evidence["article_snapshot_sha256"] != article_sha256
            or evidence["score_snapshot_sha256"] != score_sha256
            or evidence["feature_snapshot_sha256"] != feature_sha256
            or evidence["aggregation_config_sha256"]
            != canonical_sha256(sentiment_aggregation._config())
        ):
            raise EvaluationPreflightError("synthetic aggregation parent is detached")
        start = _parse_utc(evidence["coverage_start_at"])
        end = _parse_utc(evidence["coverage_end_at_exclusive"])
        if (
            end <= start
            or start > opened_at[first_ordinal] + timedelta(hours=1) - timedelta(hours=24)
            or end <= opened_at[last_decision_ordinal] + timedelta(hours=1)
            or end > inspected_at + timedelta(minutes=1)
        ):
            raise EvaluationPreflightError("synthetic terminal coverage is insufficient")
        raw_groups = evidence["groups"]
        if type(raw_groups) is not list or len(raw_groups) > 100_000:
            raise EvaluationPreflightError("synthetic group inventory is invalid")
        groups = tuple(GroupAnchor(**item) for item in raw_groups)
        if [group.to_dict() for group in groups] != raw_groups or len(
            {group.duplicate_group_id for group in groups}
        ) != len(groups):
            raise EvaluationPreflightError("synthetic duplicate anchors are ambiguous")
        eligible = [article for article in articles if article.point_in_time_eligible]
        by_group: dict[str, list[ArticleRecord]] = {}
        for article in eligible:
            if article.duplicate_group_id is None:
                raise EvaluationPreflightError("eligible article lacks permanent group")
            by_group.setdefault(article.duplicate_group_id, []).append(article)
        if set(by_group) != {group.duplicate_group_id for group in groups}:
            raise EvaluationPreflightError("group inventory differs from article snapshot")
        for group in groups:
            members = by_group[group.duplicate_group_id]
            anchors = [item for item in members if item.article_id == group.anchor_article_id]
            if not anchors:
                raise EvaluationPreflightError("group representative is absent")
            anchor = min(anchors, key=lambda item: (item.first_seen_at, item.article_version_id))
            first_seen = min(item.first_seen_at for item in members)
            if (
                group.duplicate_group_id != derive_duplicate_group_id(anchor.article_id)
                or group.initial_first_seen_at != first_seen
                or anchor.first_seen_at != first_seen
                or group.canonical_url != anchor.canonical_url
                or group.source != anchor.source
                or group.dedup_fingerprint != _dedup_fingerprint(anchor.title, anchor.language)
            ):
                raise EvaluationPreflightError("permanent duplicate anchor cannot be replayed")
        raw_gaps = evidence["terminal_gap_evidence"]
        if type(raw_gaps) is not list or len(raw_gaps) > 100_000:
            raise EvaluationPreflightError("synthetic terminal gap inventory is invalid")
        gaps = []
        previous_end = start
        for item in raw_gaps:
            if type(item) is not dict or type(item.get("attempts")) is not list:
                raise EvaluationPreflightError("terminal gap evidence is malformed")
            attempts = tuple(GapAttempt(**attempt) for attempt in item["attempts"])
            replay = TerminalGapEvidence.create(
                interval_start=item["interval_start"],
                interval_end_exclusive=item["interval_end_exclusive"],
                expected_source_locator=item["expected_source_locator"],
                attempts=attempts,
                terminal_at=item["terminal_at"],
                protocol_config_sha256=item["protocol_config_sha256"],
                provider=item["provider"],
                scope=item["scope"],
                collection_mode=item["collection_mode"],
                input_class=item["input_class"],
                network_access_authorized=item["network_access_authorized"],
                observed_snapshot_id=item["observed_snapshot_id"],
                observed_raw_snapshot_sha256=item["observed_raw_snapshot_sha256"],
                retry_policy_version=item["retry_policy_version"],
                final_terminal_disposition=item["final_terminal_disposition"],
            )
            _validate_terminal_gap_evidence(
                replay,
                interval_start=replay.interval_start,
                interval_end_exclusive=replay.interval_end_exclusive,
                expected_source_locator=_expected_source_locator_at(replay.interval_start),
                protocol_config_sha256=protocol_sha256,
                terminal_as_of=format_utc_timestamp(inspected_at),
            )
            gap_start, gap_end = _parse_utc(replay.interval_start), _parse_utc(
                replay.interval_end_exclusive
            )
            if (
                replay.to_dict() != item
                or replay.input_class != "synthetic_fixture"
                or replay.network_access_authorized is not False
                or replay.protocol_config_sha256 != protocol_sha256
                or gap_start < previous_end
                or gap_end - gap_start != timedelta(minutes=1)
                or gap_end > end
                or _parse_utc(replay.terminal_at) > inspected_at
            ):
                raise EvaluationPreflightError("terminal gap is not verified and bounded")
            gaps.append(replay)
            previous_end = gap_end
        intervals = [
            _SyntheticCoverageInterval(
                evidence["coverage_start_at"], evidence["coverage_start_at"], "complete"
            )
        ]
        intervals.extend(
            _SyntheticCoverageInterval(
                item.interval_start, item.interval_end_exclusive, "provider_gap"
            )
            for item in gaps
        )
        intervals.append(
            _SyntheticCoverageInterval(
                evidence["coverage_end_at_exclusive"],
                evidence["coverage_end_at_exclusive"],
                "complete",
            )
        )
        expected: dict[int, FeatureValues | None] = {}
        serialized_rows = []
        for ordinal in range(first_ordinal, last_decision_ordinal + 1):
            decision_at = format_utc_timestamp(opened_at[ordinal] + timedelta(hours=1))
            decision = _parse_utc(decision_at)
            available_scores = tuple(
                score
                for score in scores
                if score.scored_at is not None and _parse_utc(score.scored_at) <= decision
            )
            prepared = sentiment_aggregation._Prepared(
                articles,
                groups,
                available_scores,
                tuple(intervals),
                (),
            )
            row = sentiment_aggregation._aggregate_rows(prepared, (decision_at,))[0]
            expected[ordinal] = row.features
            serialized_rows.append(row.to_dict())
        prepared_manifest = {
            "schema_version": "phase2-synthetic-prepared-dataset-parent-v1",
            "synthetic": True,
            "article_snapshot_sha256": article_sha256,
            "score_snapshot_sha256": score_sha256,
            "sentiment_feature_sha256": canonical_sha256(serialized_rows),
            "combined_feature_sha256": feature_sha256,
            "technical_columns": list(dataset.TECHNICAL_COLUMNS),
            "sentiment_columns": list(dataset.SENTIMENT_COLUMNS),
            "excluded_provider_gap_ordinals": [
                ordinal for ordinal, value in expected.items() if value is None
            ],
        }
        if evidence["prepared_dataset_manifest"] != prepared_manifest:
            raise EvaluationPreflightError(
                "prepared dataset parent differs from aggregation replay"
            )
        return expected
    except EvaluationError:
        raise
    except (CryptoAIError, ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
        raise EvaluationPreflightError("synthetic aggregation parent cannot be replayed") from exc


def _feature_frame(
    raw: bytes,
    market: pd.DataFrame,
    boundary: holdout.BoundaryPurgePlan,
    expected_sentiment: dict[int, FeatureValues | None] | None = None,
) -> pd.DataFrame:
    header = ("market_ordinal", "decision_at") + dataset.COMBINED_COLUMNS
    rows = _csv_rows(raw, header)
    values: list[list[float]] = []
    ordinals: list[int] = []
    decisions: list[str] = []
    for row in rows:
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", row[0]):
            raise EvaluationIntegrityError("feature row has an invalid original ordinal")
        ordinal = int(row[0])
        if ordinal < boundary.first_holdout_ordinal or ordinal + backtests.HORIZON + 1 >= len(
            market
        ):
            raise EvaluationIntegrityError("feature decision is outside claimed holdout bounds")
        if ordinals and ordinal <= ordinals[-1]:
            raise EvaluationIntegrityError("feature decisions are duplicated or reordered")
        expected_at = market.iloc[ordinal].timestamp.to_pydatetime() + timedelta(hours=1)
        if _parse_utc(row[1]) != expected_at:
            raise EvaluationIntegrityError("feature decision timestamp does not close its candle")
        try:
            parsed_values = [float(value) for value in row[2:]]
        except ValueError as exc:
            raise EvaluationIntegrityError("feature row contains malformed numerics") from exc
        if not all(math.isfinite(value) for value in parsed_values):
            raise EvaluationIntegrityError("feature row contains non-finite numerics")
        sentiment_values: dict[str, int | float] = {}
        for name, dtype in FEATURE_DTYPES:
            value = parsed_values[dataset.COMBINED_COLUMNS.index(name)]
            if dtype in {"int64", "int8"}:
                if not value.is_integer():
                    raise EvaluationIntegrityError("integer sentiment feature has a fraction")
                sentiment_values[name] = int(value)
            else:
                sentiment_values[name] = value
        try:
            FeatureValues(**sentiment_values).to_dict()
        except CryptoAIError as exc:
            raise EvaluationIntegrityError(
                "sentiment feature bounds or counts are invalid"
            ) from exc
        if expected_sentiment is None:
            if (
                any(
                    sentiment_values[name] != 0
                    for name, _ in FEATURE_DTYPES
                    if name not in {"hours_since_latest_article", "news_missing_24h"}
                )
                or sentiment_values["hours_since_latest_article"] != 24.0
                or sentiment_values["news_missing_24h"] != 1
            ):
                raise EvaluationIntegrityError("synthetic no-news features lack replay provenance")
        else:
            expected_row = expected_sentiment.get(ordinal)
            if expected_row is None:
                raise EvaluationIntegrityError("provider-gap decision was not excluded")
            expected_values = expected_row.to_dict()
            for name, dtype in FEATURE_DTYPES:
                observed, expected = sentiment_values[name], expected_values[name]
                if dtype in {"int64", "int8"}:
                    matches = type(observed) is int and observed == expected
                else:
                    matches = struct.pack(">d", float(observed)) == struct.pack(
                        ">d", float(expected)
                    )
                if not matches:
                    raise EvaluationIntegrityError("sentiment feature differs from causal replay")
        if sentiment_values["news_count_24h"] == 0 and (
            sentiment_values["hours_since_latest_article"] != 24.0
            or any(
                sentiment_values[name] != 0.0
                for name, dtype in FEATURE_DTYPES
                if dtype == "float64" and name != "hours_since_latest_article"
            )
        ):
            raise EvaluationIntegrityError("no-news row violates zero-plus-indicator contract")
        values.append(parsed_values)
        ordinals.append(ordinal)
        decisions.append(row[1])
    expected_ordinals = [
        ordinal
        for ordinal in range(boundary.first_holdout_ordinal, len(market) - backtests.HORIZON - 1)
        if expected_sentiment is None or expected_sentiment.get(ordinal) is not None
    ]
    if ordinals != expected_ordinals:
        raise EvaluationIntegrityError("holdout decision interval is missing or truncated")
    frame = pd.DataFrame(values, columns=dataset.COMBINED_COLUMNS, dtype=np.float64)
    frame.insert(0, "decision_at", decisions)
    frame.insert(0, "market_ordinal", ordinals)
    return frame


def _verify_technical_replay(features: pd.DataFrame, market: pd.DataFrame) -> None:
    """Pin all 24 Phase 1 technical values to the same captured OHLCV bytes."""
    computed = compute_features(market)
    ordinals = features.market_ordinal.to_numpy(dtype=np.int64)
    if not set(ordinals).issubset(computed.index):
        raise EvaluationIntegrityError("technical warmup is absent from market context")
    expected = computed.loc[ordinals, list(dataset.TECHNICAL_COLUMNS)].to_numpy(
        dtype=np.float64, copy=True
    )
    observed = features.loc[:, list(dataset.TECHNICAL_COLUMNS)].to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.array_equal(expected.view(np.uint64), observed.view(np.uint64)):
        raise EvaluationIntegrityError("saved technical features differ from OHLCV replay")


def _scheduled_exit_ordinals(features: pd.DataFrame, probabilities: pd.Series) -> list[int]:
    """Count frozen-policy completed trades using signals, never exit prices."""
    exits: list[int] = []
    occupied_through = -1
    for ordinal, probability in zip(
        features.market_ordinal.to_numpy(dtype=np.int64),
        probabilities.to_numpy(dtype=np.float64),
        strict=True,
    ):
        decision = int(ordinal)
        if decision < occupied_through or probability < backtests.THRESHOLD:
            continue
        occupied_through = decision + backtests.HORIZON + 1
        exits.append(occupied_through)
    return exits


def _validate_jsonl(raw: bytes, *, description: str) -> None:
    if raw and not raw.endswith(b"\n"):
        raise EvaluationIntegrityError(f"{description} JSONL lacks final newline")
    for line in raw.splitlines():
        if not line:
            raise EvaluationIntegrityError(f"{description} JSONL contains an empty line")
        parsed = dataset._json(line)
        if type(parsed) is not dict:
            raise EvaluationIntegrityError(f"{description} JSONL requires object records")


def _predict(frame: pd.DataFrame, columns: tuple[str, ...], model: _FrozenModel) -> pd.Series:
    if type(model) is not _FrozenModel or model.columns != columns:
        raise EvaluationIntegrityError("model family or feature columns differ from frozen pair")
    matrix = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    if matrix.shape[1] != len(columns) or not np.isfinite(matrix).all():
        raise EvaluationIntegrityError("verified feature matrix is malformed")
    if model.family == "LogisticRegression":
        assert model.scaler_mean is not None
        assert model.scaler_scale is not None
        assert model.coefficients is not None
        assert model.intercept is not None
        scaled = (matrix - model.scaler_mean) / model.scaler_scale
        logits = np.dot(scaled, model.coefficients) + model.intercept
        if not np.isfinite(logits).all():
            raise EvaluationIntegrityError("frozen model produced non-finite logits")
        probabilities = np.where(
            logits >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(logits, 0.0, 700.0))),
            np.exp(np.clip(logits, -700.0, 0.0)) / (1.0 + np.exp(np.clip(logits, -700.0, 0.0))),
        )
    else:
        try:
            model_instance = experiments._make_model("D")
            assert model.booster_ubj is not None
            model_instance.load_model(bytearray(model.booster_ubj))
            probabilities = np.asarray(
                model_instance.predict_proba(pd.DataFrame(matrix, columns=columns))[:, 1],
                dtype=np.float64,
            )
        except Exception as exc:
            raise EvaluationIntegrityError("frozen XGBoost booster cannot predict") from exc
    if (
        probabilities.shape != (len(frame),)
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise EvaluationIntegrityError("frozen model probabilities are invalid")
    return pd.Series(probabilities.astype(np.float64), index=frame.market_ordinal.to_numpy())


def _preflight(
    request: SyntheticEvaluationRequest,
) -> tuple[dict[str, _PinnedSnapshot], bytes, bytes, dict[str, Any], _PinnedDevelopmentRun]:
    if type(request) is not SyntheticEvaluationRequest:
        raise EvaluationInputError("only an explicit synthetic evaluation request is supported")
    _safe_temporary_path(request.development_run_dir, existing=True)
    descriptor = _open_directory_path(
        request.development_run_dir, description="preflight Development run"
    )
    try:
        info = os.fstat(descriptor)
        development = _PinnedDevelopmentRun(
            request.development_run_dir, descriptor, (info.st_dev, info.st_ino), {}
        )
        result = _preflight_pinned(request, development)
        return (*result, development)
    except BaseException:
        os.close(descriptor)
        raise


def _preflight_pinned(
    request: SyntheticEvaluationRequest,
    development: _PinnedDevelopmentRun,
) -> tuple[dict[str, _PinnedSnapshot], bytes, bytes, dict[str, Any]]:
    if type(request) is not SyntheticEvaluationRequest:
        raise EvaluationInputError("only an explicit synthetic evaluation request is supported")
    _safe_temporary_path(request.development_run_dir, existing=True)
    _safe_temporary_path(request.evaluation_root, existing=False)
    if (
        type(request.run_id) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request.run_id) is None
        or request.run_id == request.development_run_dir.name
        or type(request.random_simulations) is not int
        or type(request.reduced_fixture_mode) is not bool
        or not 1 <= request.random_simulations <= backtests.RANDOM_SIMULATIONS
        or (
            request.random_simulations != backtests.RANDOM_SIMULATIONS
            and not request.reduced_fixture_mode
        )
        or (
            request.reduced_fixture_mode
            and request.random_simulations == backtests.RANDOM_SIMULATIONS
        )
        or not all(
            _sha256(value)
            for value in (
                request.protocol_sha256,
                request.input_inventory_sha256,
                request.dependency_lock_sha256,
            )
        )
        or type(request.code_commit) is not str
        or holdout.COMMIT_PATTERN.fullmatch(request.code_commit) is None
    ):
        raise EvaluationInputError("request identity or bounded fixture settings are invalid")
    store = EvaluationStore(request.evaluation_root)
    store_descriptor = _open_directory_path(
        store.root, description="synthetic evaluation preflight root"
    )
    try:
        try:
            os.stat(request.run_id, dir_fd=store_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvaluationPreflightError("evaluation run ID is already occupied")
    finally:
        os.close(store_descriptor)
    boundary, readiness = request.boundary, request.readiness
    if (
        type(boundary) is not holdout.BoundaryPurgePlan
        or type(readiness) is not holdout.ReadinessReport
    ):
        raise EvaluationPreflightError("issued boundary and readiness evidence is required")
    if (
        boundary not in holdout._issued_boundaries
        or holdout._issued_boundaries[boundary] != holdout._boundary_snapshot(boundary)
        or readiness not in holdout._issued_readiness_reports
        or holdout._issued_readiness_reports[readiness] != holdout._readiness_snapshot(readiness)
        or boundary.synthetic is not True
        or readiness.synthetic is not True
        or readiness.ready is not True
        or readiness.trade_threshold_met is not True
        or readiness.elapsed_days < readiness.planned_minimum_days
        or readiness.planned_minimum_days < 180
        or readiness.provider_outage_state not in {"CLEAR", "VERIFIED_GAP"}
        or readiness.frozen_policy_sha256 != request.protocol_sha256
        or readiness.first_holdout_ordinal != boundary.first_holdout_ordinal
    ):
        raise EvaluationPreflightError("frozen zero-outcome readiness is not satisfied")
    fit = boundary.frozen_fit
    if (
        sha256_bytes(fit.augmented_model_bytes) != fit.augmented_model_sha256
        or sha256_bytes(fit.control_model_bytes) != fit.control_model_sha256
        or sha256_bytes(fit.fit_manifest_bytes) != fit.fit_manifest_sha256
    ):
        raise EvaluationPreflightError("frozen development model hashes changed")
    augmented_frozen = _verified_model(fit.augmented_model_bytes, dataset.COMBINED_COLUMNS)
    control_frozen = _verified_model(fit.control_model_bytes, dataset.TECHNICAL_COLUMNS)
    expected_family = (
        "LogisticRegression" if fit.selected_augmented_cell == "C" else "XGBClassifier"
    )
    if augmented_frozen.family != expected_family or control_frozen.family != expected_family:
        raise EvaluationPreflightError("selected cell pair and frozen model family differ")
    _, development_bytes = _development_manifest(request, development)
    inventory = {
        "evaluation_run_id": request.run_id,
        "market_snapshot_sha256": request.market_snapshot.sha256,
        "article_snapshot_sha256": request.article_snapshot.sha256,
        "score_snapshot_sha256": request.score_snapshot.sha256,
        "feature_snapshot_sha256": request.feature_snapshot.sha256,
    }
    if request.aggregation_evidence is not None:
        if type(request.aggregation_evidence) is not SnapshotRef:
            raise EvaluationInputError("synthetic aggregation reference is malformed")
        inventory["aggregation_evidence_sha256"] = request.aggregation_evidence.sha256
        if request.verified_news_parent is None:
            raise EvaluationPreflightError("news requires immutable M4/M5 parent publications")
        inventory["verified_news_parent_sha256"] = sha256_bytes(
            _news_parent_bytes(request.verified_news_parent)
        )
    elif request.verified_news_parent is not None:
        raise EvaluationPreflightError("news parent without aggregation evidence is ambiguous")
    if sha256_bytes(canonicalize(inventory)) != request.input_inventory_sha256:
        raise EvaluationPreflightError("claim inventory hash does not bind exact inputs and run")
    references = {
        "market": request.market_snapshot,
        "article": request.article_snapshot,
        "score": request.score_snapshot,
        "feature": request.feature_snapshot,
        "development_market": request.development_market,
        "development_rows": request.development_rows,
    }
    if request.aggregation_evidence is not None:
        references["aggregation"] = request.aggregation_evidence
    pinned: dict[str, _PinnedSnapshot] = {}
    index = _OpaqueMarketIndex(readiness, boundary)
    article_schema = _NewsSnapshotSchema("article")
    score_schema = _NewsSnapshotSchema("score")
    feature_header = _FeatureHeader()
    try:
        for name, reference in references.items():
            market_index = index if name == "market" else None
            news_schema = (
                article_schema if name == "article" else score_schema if name == "score" else None
            )
            pinned[name] = _pin_snapshot(
                reference,
                market_index=market_index,
                news_schema=news_schema,
                feature_header=feature_header if name == "feature" else None,
                development=(
                    development if name in {"development_market", "development_rows"} else None
                ),
            )
        if (
            request.development_market.path
            != request.development_run_dir / "development_market.csv"
            or request.development_rows.path
            != request.development_run_dir / "development_rows.json"
        ):
            raise EvaluationPreflightError("development evidence is outside the frozen run")
        cutoff = boundary.development_cutoff_ordinal
        prefix = b"".join(
            pinned["market"].capture().splitlines(keepends=True)[: cutoff + backtests.HORIZON + 3]
        )
        if pinned["development_market"].capture() != prefix:
            raise EvaluationPreflightError(
                "development market evidence differs from the pre-holdout market prefix"
            )
        try:
            replay_rows, replay_augmented, replay_control = build_synthetic_development_fit(
                prefix,
                cutoff,
                (fit.selected_augmented_cell, fit.matched_control_cell),
            )
        except (CryptoAIError, ValueError, TypeError, OverflowError) as exc:
            raise EvaluationPreflightError(
                "development fit cannot be independently replayed"
            ) from exc
        if (
            pinned["development_rows"].capture() != replay_rows
            or fit.shared_labeled_rows_sha256 != sha256_bytes(replay_rows)
            or fit.augmented_model_bytes != replay_augmented
            or fit.control_model_bytes != replay_control
        ):
            raise EvaluationPreflightError(
                "frozen development rows, labels, scaler, or model differ from replay"
            )
        dataset_manifest_bytes = _development_dataset_manifest_bytes(
            pinned["development_market"].capture(),
            pinned["development_rows"].capture(),
            cutoff,
        )
        development.files = {
            "development_manifest.json": development_bytes,
            "development_dataset_manifest.json": dataset_manifest_bytes,
            "development_market.csv": pinned["development_market"].capture(),
            "development_rows.json": pinned["development_rows"].capture(),
            "augmented_model.json": fit.augmented_model_bytes,
            "control_model.json": fit.control_model_bytes,
            "fit_manifest.json": fit.fit_manifest_bytes,
        }
        development.verify()
        validate_article_collection(article_schema.articles)
        article_by_version = {
            article.article_version_id: article for article in article_schema.articles
        }
        score_ids: set[str] = set()
        scored_content_hashes: set[str] = set()
        for score in score_schema.scores:
            article = article_by_version.get(score.article_version_id)
            if (
                score.score_id in score_ids
                or score.content_hash in scored_content_hashes
                or article is None
                or article.content_hash != score.content_hash
            ):
                raise EvaluationPreflightError(
                    "score snapshot has a duplicate, competing, or orphan link"
                )
            score_ids.add(score.score_id)
            scored_content_hashes.add(score.content_hash)
        aggregation_rows = None
        if request.aggregation_evidence is None:
            if (
                article_schema.articles
                or score_schema.scores
                or readiness.provider_outage_state != "CLEAR"
            ):
                raise EvaluationPreflightError(
                    "news or provider gaps require replay-verified synthetic aggregation"
                )
        else:
            aggregation_rows = _aggregation_expectations(
                pinned["aggregation"].capture(),
                articles=tuple(article_schema.articles),
                scores=tuple(score_schema.scores),
                article_sha256=request.article_snapshot.sha256,
                score_sha256=request.score_snapshot.sha256,
                feature_sha256=request.feature_snapshot.sha256,
                protocol_sha256=request.protocol_sha256,
                opened_at=index.opened_at,
                first_ordinal=boundary.first_holdout_ordinal,
                last_decision_ordinal=readiness.market_last_ordinal - backtests.HORIZON - 1,
                inspected_at=readiness.inspected_at,
            )
            verified_rows = _verified_news_parent_expectations(
                request.verified_news_parent,
                evidence_raw=pinned["aggregation"].capture(),
                articles=tuple(article_schema.articles),
                scores=tuple(score_schema.scores),
                opened_at=index.opened_at,
                first_ordinal=boundary.first_holdout_ordinal,
                last_decision_ordinal=readiness.market_last_ordinal - backtests.HORIZON - 1,
                development_market_raw=pinned["development_market"].capture(),
                development_rows_raw=pinned["development_rows"].capture(),
                protocol_sha256=request.protocol_sha256,
            )
            if _feature_rows_bits(aggregation_rows) != _feature_rows_bits(verified_rows):
                raise EvaluationPreflightError(
                    "self-attested aggregation differs from immutable M4 replay"
                )
            any_gap = any(value is None for value in aggregation_rows.values())
            if any_gap != (readiness.provider_outage_state == "VERIFIED_GAP"):
                raise EvaluationPreflightError("provider-gap state differs from excluded decisions")
        issued_bytes = holdout._issued_readiness_evidence.get(readiness)
        if (
            type(issued_bytes) is not bytes
            or sha256_bytes(issued_bytes) != readiness.operational_evidence_sha256
        ):
            raise EvaluationPreflightError("issued operational evidence bytes are unavailable")
        issued = dataset._json(issued_bytes)
        if (
            type(issued) is not dict
            or issued.get("raw_snapshot_sha256")
            != [references[name].sha256 for name in ("market", "article", "score", "feature")]
            or issued.get("market_candles") != index.market_candles
        ):
            raise EvaluationPreflightError("readiness proof and pinned snapshot inventory differ")
        # Parse only the saved, previously hashed feature bytes to derive the
        # scheduled-exit count. Market price columns remain opaque pre-claim.
        operational_market = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(index.opened_at),
                "open": np.ones(len(index.opened_at), dtype=np.float64),
            }
        )
        operational_features = _feature_frame(
            pinned["feature"].capture(), operational_market, boundary, aggregation_rows
        )
        augmented_model = _verified_model(fit.augmented_model_bytes, dataset.COMBINED_COLUMNS)
        signals = _predict(operational_features, dataset.COMBINED_COLUMNS, augmented_model)
        if _scheduled_exit_ordinals(operational_features, signals) != issued.get(
            "scheduled_exit_ordinals"
        ):
            raise EvaluationPreflightError(
                "readiness scheduled exits differ from frozen model signals"
            )
        for snapshot in pinned.values():
            snapshot.capture()
        development.verify()
        return pinned, development_bytes, dataset_manifest_bytes, inventory
    except BaseException:
        for entry in pinned.values():
            os.close(entry.descriptor)
        raise


def _json_bytes(value: Any) -> bytes:
    return canonicalize(value)


def _predictions_csv(features: pd.DataFrame, augmented: pd.Series, control: pd.Series) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "market_ordinal",
            "decision_at",
            "augmented_probability",
            "control_probability",
            "augmented_signal",
            "control_signal",
        )
    )
    for row, aug, ctl in zip(
        features.itertuples(index=False), augmented.to_numpy(), control.to_numpy(), strict=True
    ):
        writer.writerow(
            (
                int(row.market_ordinal),
                row.decision_at,
                repr(float(aug)),
                repr(float(ctl)),
                int(aug >= backtests.THRESHOLD),
                int(ctl >= backtests.THRESHOLD),
            )
        )
    return buffer.getvalue().encode("utf-8")


def _execute_claimed(
    request: SyntheticEvaluationRequest,
    claim: holdout.ClaimRecord,
    captured: dict[str, bytes],
    development_manifest_bytes: bytes,
    development_dataset_manifest_bytes: bytes,
    inventory: dict[str, Any],
    store: EvaluationStore,
) -> EvaluationArtifact:
    # This function is called only after HoldoutClaimManager.acquire returns.
    _validate_jsonl(captured["article"], description="article snapshot")
    _validate_jsonl(captured["score"], description="score snapshot")
    market = _market_frame(captured["market"], request.readiness.market_last_ordinal)
    aggregation_rows = None
    if "aggregation" in captured:
        article_schema = _NewsSnapshotSchema("article")
        score_schema = _NewsSnapshotSchema("score")
        article_schema.feed(captured["article"])
        score_schema.feed(captured["score"])
        article_schema.finish()
        score_schema.finish()
        aggregation_rows = _aggregation_expectations(
            captured["aggregation"],
            articles=tuple(article_schema.articles),
            scores=tuple(score_schema.scores),
            article_sha256=sha256_bytes(captured["article"]),
            score_sha256=sha256_bytes(captured["score"]),
            feature_sha256=sha256_bytes(captured["feature"]),
            protocol_sha256=request.protocol_sha256,
            opened_at=[value.to_pydatetime() for value in market.timestamp],
            first_ordinal=request.boundary.first_holdout_ordinal,
            last_decision_ordinal=len(market) - backtests.HORIZON - 2,
            inspected_at=request.readiness.inspected_at,
        )
    features = _feature_frame(captured["feature"], market, request.boundary, aggregation_rows)
    _verify_technical_replay(features, market)
    fit = request.boundary.frozen_fit
    augmented = _predict(
        features,
        dataset.COMBINED_COLUMNS,
        _verified_model(fit.augmented_model_bytes, dataset.COMBINED_COLUMNS),
    )
    control = _predict(
        features,
        dataset.TECHNICAL_COLUMNS,
        _verified_model(fit.control_model_bytes, dataset.TECHNICAL_COLUMNS),
    )
    first_open = int(features.market_ordinal.iloc[0]) + 1
    final_open = int(features.market_ordinal.iloc[-1]) + backtests.HORIZON + 1
    if final_open >= len(market):
        raise EvaluationIntegrityError("final scheduled holdout exit open is absent")
    window = backtests._window(market, first_open, final_open, len(features))
    model_scores = {"augmented": augmented, "control": control}
    strategy: dict[str, dict[str, dict[str, Any]]] = {"augmented": {}, "control": {}}
    files: dict[str, bytes] = {
        "input_market_snapshot.csv": captured["market"],
        "input_article_snapshot.jsonl": captured["article"],
        "input_score_snapshot.jsonl": captured["score"],
        "input_feature_snapshot.csv": captured["feature"],
        "evaluation_models/augmented.json": fit.augmented_model_bytes,
        "evaluation_models/control.json": fit.control_model_bytes,
        "evaluation_models/fit_manifest.json": fit.fit_manifest_bytes,
        "evaluation_models/development_manifest.json": development_manifest_bytes,
        "evaluation_models/development_dataset_manifest.json": (development_dataset_manifest_bytes),
        "evaluation_models/development_market.csv": captured["development_market"],
        "evaluation_models/development_rows.json": captured["development_rows"],
        "holdout_evaluation_claim.json": claim.raw_bytes,
        "holdout_predictions.csv": _predictions_csv(features, augmented, control),
    }
    if "aggregation" in captured:
        files["input_aggregation_evidence.json"] = captured["aggregation"]
        files["input_news_parent.json"] = _news_parent_bytes(request.verified_news_parent)
    for role, scores in model_scores.items():
        previous_schedule: tuple[tuple[str, str], ...] | None = None
        for scenario in backtests.SCENARIO_ORDER:
            result = backtests._simulate(
                market,
                scores,
                None,
                backtests.SCENARIOS[scenario],
                expected_start=first_open,
                expected_end=final_open,
            )
            payload = backtests._result_payload(result)
            strategy[role][scenario] = payload
            files[f"trade_ledgers/{role}_{scenario}.json"] = _json_bytes(payload["trade_ledger"])
            files[f"equity_curves/{role}_{scenario}.json"] = _json_bytes(payload["equity_curve"])
            schedule = tuple(
                (row["entry_timestamp"], row["exit_timestamp"]) for row in payload["trade_ledger"]
            )
            if previous_schedule is not None and schedule != previous_schedule:
                raise EvaluationIntegrityError("cost scenario changed frozen model trade ordinals")
            previous_schedule = schedule
    base_augmented = strategy["augmented"]["base"]
    base_control = strategy["control"]["base"]
    if [
        row["exit_market_ordinal"] for row in base_augmented["trade_ledger"]
    ] != _scheduled_exit_ordinals(features, augmented):
        raise EvaluationIntegrityError("executed trades differ from readiness schedule")
    ordinals = pd.Index(features.market_ordinal.to_numpy(dtype=np.int64))
    ema_scores = pd.Series(
        (features.ema_short.to_numpy() > features.ema_long.to_numpy()).astype(np.float64),
        index=ordinals,
    )
    momentum_scores = pd.Series(
        (features.return_24.to_numpy() > 0.0).astype(np.float64), index=ordinals
    )
    zeros = pd.Series(0.0, index=ordinals)
    deterministic_scores = {"cash": zeros, "ema_9_21": ema_scores, "momentum_24": momentum_scores}
    baselines: dict[str, dict[str, Any]] = {}
    for scenario in backtests.SCENARIO_ORDER:
        config = backtests.SCENARIOS[scenario]
        paths: dict[str, Any] = {}
        for name, scores in deterministic_scores.items():
            result = backtests._simulate(
                market, scores, None, config, expected_start=first_open, expected_end=final_open
            )
            paths[name] = backtests._result_payload(result)
        buy_hold = buy_and_hold_backtest(
            market, ordinals, backtests.HORIZON, config, initial_capital=backtests.INITIAL_CAPITAL
        )
        backtests._result_window(buy_hold, market, first_open, final_open)
        paths["buy_and_hold"] = backtests._result_payload(buy_hold)
        baselines[scenario] = paths
        for name, payload in paths.items():
            files[f"trade_ledgers/{name}_{scenario}.json"] = _json_bytes(payload["trade_ledger"])
            files[f"equity_curves/{name}_{scenario}.json"] = _json_bytes(payload["equity_curve"])
    random_evidence, random_summary, random_ledgers, random_curves = _random_baseline(
        market,
        augmented,
        first_open,
        final_open,
        request.random_simulations,
        model_returns={
            scenario: strategy["augmented"][scenario]["metrics"]["total_return"]
            for scenario in backtests.SCENARIO_ORDER
        },
    )
    files["trade_ledgers/random_exposure.json"] = _json_bytes(random_evidence)
    files["trade_ledgers/random_exposure.jsonl.gz"] = random_ledgers
    files["equity_curves/random_exposure.jsonl.gz"] = random_curves
    final_gates = evaluate_final_gates(
        augmented_metrics=base_augmented["metrics"],
        control_metrics=base_control["metrics"],
        cash_metrics=baselines["base"]["cash"]["metrics"],
        augmented_ledger=base_augmented["trade_ledger"],
        control_ledger=base_control["trade_ledger"],
        elapsed_days=request.readiness.elapsed_days,
        planned_minimum_days=request.readiness.planned_minimum_days,
    )
    metrics = {
        "schema_version": "phase2-synthetic-holdout-metrics-v1",
        "synthetic": True,
        "engineering_status": (
            "REDUCED_SYNTHETIC_FIXTURE" if request.reduced_fixture_mode else "PASS"
        ),
        "research_verdict": final_gates["research_verdict"],
        "production_decision": "NO-GO",
        "common_window": window,
        "base": base_augmented["metrics"],
        "control_base": base_control["metrics"],
        "final_gates": final_gates,
        "reconciliation": {
            "augmented": base_augmented["reconciliation"],
            "control": base_control["reconciliation"],
        },
    }
    baseline_metrics = {
        "schema_version": "phase2-synthetic-holdout-baselines-v1",
        "synthetic": True,
        "common_window": window,
        "deterministic": {
            scenario: {name: path["metrics"] for name, path in paths.items()}
            for scenario, paths in baselines.items()
        },
        "random_exposure": random_summary,
    }
    costs = {
        "schema_version": "phase2-synthetic-holdout-cost-sensitivity-v1",
        "synthetic": True,
        "official_scenario": "base",
        "common_window": window,
        "models": {
            role: {scenario: path["metrics"] for scenario, path in scenarios.items()}
            for role, scenarios in strategy.items()
        },
        "baselines": baseline_metrics["deterministic"],
        "random_exposure": baseline_metrics["random_exposure"],
        "signals_and_trade_ordinals_frozen_across_scenarios": True,
    }
    files["metrics.json"] = _json_bytes(metrics)
    files["baseline_metrics.json"] = _json_bytes(baseline_metrics)
    files["cost_sensitivity.json"] = _json_bytes(costs)
    metadata = {
        "synthetic": True,
        "specification_id": SPECIFICATION_ID,
        "claim_sha256": claim.sha256,
        "claim_bytes_sha256": sha256_bytes(claim.raw_bytes),
        "development_manifest_sha256": sha256_bytes(development_manifest_bytes),
        "fit_manifest_sha256": fit.fit_manifest_sha256,
        "augmented_model_sha256": fit.augmented_model_sha256,
        "control_model_sha256": fit.control_model_sha256,
        "protocol_sha256": request.protocol_sha256,
        "code_commit": request.code_commit,
        "dependency_lock_sha256": request.dependency_lock_sha256,
        "input_inventory": inventory,
        "input_inventory_sha256": request.input_inventory_sha256,
        "selected_augmented_cell": fit.selected_augmented_cell,
        "matched_control_cell": fit.matched_control_cell,
        "augmented_feature_columns": list(dataset.COMBINED_COLUMNS),
        "control_feature_columns": list(dataset.TECHNICAL_COLUMNS),
        "common_window": window,
        "cost_scenarios": list(backtests.SCENARIO_ORDER),
        "random_simulations": request.random_simulations,
        "baseline_mode": (
            "REDUCED_SYNTHETIC_FIXTURE" if request.reduced_fixture_mode else "FULL_1000"
        ),
        "gate_version": "phase2-final-evidence-gates-v1",
    }
    return store.publish(request.run_id, files, metadata)


def _random_baseline(
    market: pd.DataFrame,
    augmented: pd.Series,
    first_open: int,
    final_open: int,
    simulations: int,
    *,
    model_returns: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    if (
        type(model_returns) is not dict
        or set(model_returns) != set(backtests.SCENARIO_ORDER)
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in model_returns.values()
        )
    ):
        raise EvaluationIntegrityError("random baseline requires finite matched model returns")
    probability = float((augmented.to_numpy() >= backtests.THRESHOLD).mean())
    evidence: dict[str, Any] = {
        "schema_version": "phase2-synthetic-random-exposure-v1",
        "synthetic": True,
        "seed_base": backtests.RANDOM_SEED,
        "simulations": simulations,
        "signal_probability": probability,
        "draws_and_metrics": [],
    }
    samples = {
        scenario: {key: [] for key in ("total_return", "sharpe_ratio", "maximum_drawdown")}
        for scenario in backtests.SCENARIO_ORDER
    }
    ledger_buffer = io.BytesIO()
    curve_buffer = io.BytesIO()
    ledger_zip = gzip.GzipFile(fileobj=ledger_buffer, mode="wb", filename="", mtime=0)
    curve_zip = gzip.GzipFile(fileobj=curve_buffer, mode="wb", filename="", mtime=0)
    for simulation in range(simulations):
        rng = np.random.default_rng(backtests.RANDOM_SEED + simulation)
        draws = pd.Series(
            rng.binomial(1, probability, len(augmented)).astype(np.float64),
            index=augmented.index,
        )
        draw_values = [int(value) for value in draws.to_numpy()]
        item: dict[str, Any] = {
            "simulation": simulation,
            "seed": backtests.RANDOM_SEED + simulation,
            "draws": draw_values,
            "draws_sha256": sha256_bytes(_json_bytes(draw_values)),
            "scenarios": {},
        }
        for scenario in backtests.SCENARIO_ORDER:
            result = backtests._simulate(
                market,
                draws,
                None,
                backtests.SCENARIOS[scenario],
                expected_start=first_open,
                expected_end=final_open,
            )
            payload = backtests._result_payload(result)
            item["scenarios"][scenario] = {
                "metrics": payload["metrics"],
                "ledger_sha256": sha256_bytes(_json_bytes(payload["trade_ledger"])),
                "equity_sha256": sha256_bytes(_json_bytes(payload["equity_curve"])),
            }
            ledger_zip.write(
                _json_bytes(
                    {
                        "simulation": simulation,
                        "scenario": scenario,
                        "trade_ledger": payload["trade_ledger"],
                    }
                )
                + b"\n"
            )
            curve_zip.write(
                _json_bytes(
                    {
                        "simulation": simulation,
                        "scenario": scenario,
                        "equity_curve": payload["equity_curve"],
                    }
                )
                + b"\n"
            )
            for key in samples[scenario]:
                samples[scenario][key].append(payload["metrics"][key])
        evidence["draws_and_metrics"].append(item)
    ledger_zip.close()
    curve_zip.close()
    summary = {}
    for scenario, values_by_key in samples.items():
        returns = values_by_key["total_return"]
        if len(returns) != simulations or any(value is None for value in returns):
            raise EvaluationIntegrityError("random total return must be defined for every run")
        summary[scenario] = {
            "simulations": simulations,
            "seed_base": backtests.RANDOM_SEED,
            "signal_probability": probability,
            **{key: backtests._summary(values) for key, values in values_by_key.items()},
            "fraction_return_at_least_model": float(
                np.mean(np.asarray(returns, dtype=np.float64) >= model_returns[scenario])
            ),
        }
    return evidence, summary, ledger_buffer.getvalue(), curve_buffer.getvalue()


def _mark_claim_completed(
    request: SyntheticEvaluationRequest,
    claim: holdout.ClaimRecord,
    artifact: EvaluationArtifact,
    *,
    development: _PinnedDevelopmentRun,
) -> None:
    """Append a durable completion marker; never edit or release the claim."""
    completion_name = "holdout_evaluation_completed.json"
    completion = canonicalize(
        {
            "schema_version": "phase2-synthetic-holdout-completion-v1",
            "synthetic": True,
            "evaluation_run_id": request.run_id,
            "claim_sha256": claim.sha256,
            "evaluation_manifest_sha256": sha256_bytes(canonicalize(artifact.manifest)),
            "completed_at_utc": format_utc_timestamp(datetime.now(UTC)),
        }
    )
    development.verify()
    directory = os.dup(development.descriptor)
    file_descriptor: int | None = None
    try:
        current_claim, _ = _read_regular_file_at_once(
            directory, holdout.CLAIM_NAME, description="consumed holdout claim"
        )
        if current_claim != claim.raw_bytes:
            raise EvaluationIntegrityError("claim bytes changed before completion")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(completion_name, flags, 0o600, dir_fd=directory)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            handle.write(completion)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory_descriptor(directory, description="claim completion parent")
        captured, _ = _read_regular_file_at_once(
            directory, completion_name, description="claim completion record"
        )
        if captured != completion:
            raise EvaluationIntegrityError("claim completion bytes changed")
        development.verify()
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory)


class OfflineEvaluationEngine:
    """Run only explicit temporary, synthetic fixtures through a consumed claim."""

    def evaluate(self, request: SyntheticEvaluationRequest) -> EvaluationArtifact:
        pinned: dict[str, _PinnedSnapshot] = {}
        development: _PinnedDevelopmentRun | None = None
        try:
            pinned, development_bytes, dataset_manifest_bytes, inventory, development = _preflight(
                request
            )
            development.verify()
            claim = holdout.HoldoutClaimManager(
                request.development_run_dir, expected_identity=development.identity
            ).acquire(
                boundary=request.boundary,
                readiness=request.readiness,
                protocol_sha256=request.protocol_sha256,
                input_inventory_sha256=request.input_inventory_sha256,
                code_commit=request.code_commit,
                dependency_lock_sha256=request.dependency_lock_sha256,
                evaluation_run_id=request.run_id,
                development_dataset_manifest_sha256=sha256_bytes(dataset_manifest_bytes),
            )
            development.verify()
            # Captured market bytes are first interpreted as prices after claim fsync.
            captured = {name: entry.capture() for name, entry in pinned.items()}
            store = EvaluationStore(request.evaluation_root)
            artifact = _execute_claimed(
                request,
                claim,
                captured,
                development_bytes,
                dataset_manifest_bytes,
                inventory,
                store,
            )
            try:
                _mark_claim_completed(request, claim, artifact, development=development)
                if (
                    store.get(request.run_id, development_run_dir=request.development_run_dir)
                    != artifact
                ):
                    raise EvaluationIntegrityError(
                        "completed evaluation changed during verification"
                    )
            except BaseException:
                # Publication and the source completion record live in separate
                # directories. A failure after rename must invalidate the exact
                # owned publication before the exception can escape; the
                # irreversible claim itself is never removed or retried.
                store.unpublish_owned(request.run_id, artifact)
                raise
            return artifact
        except EvaluationError:
            raise
        except CryptoAIError as exc:
            raise EvaluationIntegrityError("synthetic evaluation failed closed") from exc
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            OverflowError,
        ) as exc:
            raise EvaluationIntegrityError("synthetic evaluation failed closed") from exc
        finally:
            for entry in pinned.values():
                os.close(entry.descriptor)
            if development is not None:
                os.close(development.descriptor)
