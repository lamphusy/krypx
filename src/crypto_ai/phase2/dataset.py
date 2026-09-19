"""Offline synthetic dataset integration; no market loader, model, or network seam.

Candidate properties inspect immutable buffers. Only builder.prepare / store.publish /
store.get certify transitive parents and replay dataset semantics. Preparation and
publication require a clean local implementation commit; tests mock only local Git evidence.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import pandas as pd

from crypto_ai.config import settings
from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.exceptions import CryptoAIError, PublicationCollisionError
from crypto_ai.features.build import compute_features, get_expected_feature_columns
from crypto_ai.features.dataset import _feature_configuration
from crypto_ai.features.labels import add_labels
from crypto_ai.sentiment import aggregation as aggregation_module
from crypto_ai.sentiment.aggregation import (
    FEATURE_DTYPES,
    AggregationArtifact,
    FeatureValues,
    SyntheticAggregationInput,
)
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.providers.gdelt_gsg import (
    LANGUAGE_MAP_VERSION,
    NORMALIZER_VERSION,
    TEXT_NORMALIZER_VERSION,
    URL_NORMALIZER_VERSION,
)
from crypto_ai.sentiment.scoring import ScoreArtifact
from crypto_ai.sentiment.storage import (
    PUBLICATION_SCHEMA_VERSION,
    ContentAddressedStore,
    _atomic_rename_directory_no_replace,
    _capture_publication_tree,
    _cleanup_staging_at,
    _ensure_directory_chain_at,
    _fsync_directory_descriptor,
    _fsync_tree_directories_at,
    _open_directory_at,
    _open_store_directory,
    _read_regular_file_at_once,
    _require_atomic_rename_directory_no_replace_at,
    _require_descriptor_relative_mutations,
    _validate_relative_path,
    _write_fsynced_at,
)

SCHEMA = "phase2-prepared-dataset-v1"
SPECIFICATION_ID = "phase2-milestone5-offline-dataset-integration-v1"
MANIFEST = "prepared_dataset_manifest.json"
PUBLICATION_PREFIX = "prepared-dataset-"
TECHNICAL_COLUMNS = (
    "ema_short",
    "ema_long",
    "ema_ratio",
    "close_to_ema_short",
    "close_to_ema_long",
    "macd",
    "macd_signal",
    "macd_diff",
    "rsi",
    "stoch_rsi",
    "bb_width",
    "bb_pct",
    "atr",
    "atr_pct",
    "candle_range_pct",
    "body_return",
    "volume_change",
    "volume_ma_ratio",
    "return_1",
    "return_2",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
)
SENTIMENT_COLUMNS = tuple(name for name, _ in FEATURE_DTYPES)
COMBINED_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
RAW_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PREFIX_COLUMNS = ("market_ordinal", "decision_at") + RAW_COLUMNS
LABEL_COLUMNS = (
    "entry_timestamp",
    "exit_timestamp",
    "entry_open",
    "exit_open",
    "gross_forward_return",
    "label",
)
COMBINED_TABLE_COLUMNS = PREFIX_COLUMNS + COMBINED_COLUMNS
LABELED_TABLE_COLUMNS = COMBINED_TABLE_COLUMNS + LABEL_COLUMNS
DTYPES = {
    "market_ordinal": "int64",
    "decision_at": "datetime64[ns, UTC]",
    "timestamp": "datetime64[ns, UTC]",
    **dict.fromkeys(RAW_COLUMNS[1:] + TECHNICAL_COLUMNS, "float64"),
    **dict(FEATURE_DTYPES),
    "entry_timestamp": "datetime64[ns, UTC]",
    "exit_timestamp": "datetime64[ns, UTC]",
    "entry_open": "float64",
    "exit_open": "float64",
    "gross_forward_return": "float64",
    "label": "int8",
}
_TECHNICAL_CONFIG = {
    "ema_short": 9,
    "ema_long": 21,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "stoch_rsi_period": 14,
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "atr_period": 14,
    "volume_ma_period": 20,
    "return_periods": [1, 2, 3, 6, 12, 24],
}
_HASH_FILES = {
    "market_snapshot_sha256": "market.json",
    "article_snapshot_sha256": "parents/articles/state.json",
    "score_snapshot_sha256": "score-inventory.json",
    "sentiment_feature_sha256": "parents/aggregation/envelope.json",
    "combined_feature_sha256": "combined.json",
    "labeled_dataset_sha256": "labeled.json",
    "market_price_context_sha256": "market-price-context.json",
    "row_index_sha256": "row-index.json",
    "exclusions_sha256": "exclusions.json",
    "technical_config_sha256": "config/technical.json",
    "deduplication_config_sha256": "config/deduplication.json",
    "scoring_config_sha256": "config/scoring.json",
    "aggregation_config_sha256": "config/aggregation.json",
    "label_config_sha256": "config/label.json",
    "missing_data_policy_sha256": "config/missing-data.json",
    "integration_config_sha256": "config/integration.json",
    "dependency_lock_sha256": "requirements-lock.txt",
    "phase2_dependency_lock_sha256": "requirements-phase2.txt",
    "protocol_sha256": "protocol.json",
}
MANIFEST_FIELDS = frozenset(_HASH_FILES) | {
    "schema_version",
    "specification_id",
    "synthetic",
    "code_commit",
    "technical_feature_columns",
    "sentiment_feature_columns",
    "combined_feature_columns",
    "label_columns",
    "column_dtypes",
    "coverage_as_of",
    "row_counts",
    "files",
}
_LOCK_HASHES = {
    "requirements-lock.txt": "b17b32ea58d2a8baaed70fbb9ededd984639b3b52873d194af9d2051b54af210",
    "requirements-phase2.txt": "f57ef4ff913e39f002daefe2b717451d342a50bfc2d409b7e8118ff53b6eeafb",
}


class DatasetError(CryptoAIError):
    """A Phase 2 synthetic dataset contract failed."""


class DatasetInputError(DatasetError):
    """An input does not match the frozen fixture/schema boundary."""


class DatasetAuthorizationError(DatasetError):
    """A non-synthetic input or unsupported execution seam was requested."""


class DatasetIntegrityError(DatasetError):
    """Stored evidence does not reproduce its claimed identity or semantics."""


@contextmanager
def _errors():
    try:
        yield
    except DatasetError:
        raise
    except (
        CryptoAIError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
        IndexError,
    ) as exc:
        raise DatasetIntegrityError("invalid synthetic prepared dataset evidence") from exc


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _time(value):
    parsed = parse_utc_timestamp(value, field="dataset timestamp")
    if format_utc_timestamp(parsed) != value:
        raise DatasetInputError("timestamp must use canonical UTC encoding")
    return parsed


def _git(*args):
    """Read local Git evidence only: no lazy object fetch, hooks, filters or fsmonitor."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_NO_LAZY_FETCH="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                *args,
            ],
            env=environment,
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DatasetIntegrityError("local implementation commit evidence unavailable") from exc
    return result.stdout


def _verify_code_provenance(code_commit, *, require_clean):
    """Pin the executing implementation and frozen locks to an existing local commit."""
    with _errors():
        if type(code_commit) is not str or not re.fullmatch("[0-9a-f]{40}", code_commit):
            raise DatasetInputError("exact Git commit identity required")
        expected_commit = code_commit.encode() + b"\n"
        if _git("rev-parse", "--verify", code_commit + "^{commit}") != expected_commit:
            raise DatasetIntegrityError("implementation commit does not resolve exactly")

        def clean():
            if _git("rev-parse", "HEAD") != expected_commit or _git(
                "status", "--porcelain=v1", "--untracked-files=all"
            ):
                raise DatasetIntegrityError(
                    "preparation/publication requires the pinned clean commit"
                )

        if require_clean:
            clean()
        if (
            _git("show", code_commit + ":src/crypto_ai/phase2/dataset.py")
            != Path(__file__).read_bytes()
        ):
            raise DatasetIntegrityError("implementation source differs from pinned commit")
        for name, digest in _LOCK_HASHES.items():
            if sha256_bytes(_git("show", code_commit + ":" + name)) != digest:
                raise DatasetIntegrityError("implementation commit dependency lock mismatch")
        if require_clean:
            clean()


def _json(raw):
    if type(raw) is not bytes or len(raw) > 128_000_000:
        raise DatasetInputError("bounded exact JSON bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DatasetIntegrityError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if canonicalize(value) != raw:
        raise DatasetIntegrityError("noncanonical or non-finite JSON")
    return value


def _table(frame, columns):
    if list(frame.columns) != list(columns):
        raise DatasetInputError("table column order mismatch")
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for name, value in zip(columns, row, strict=True):
            dtype = DTYPES[name]
            if dtype == "datetime64[ns, UTC]":
                value = format_utc_timestamp(value.to_pydatetime())
            elif dtype == "float64":
                value = float(value)
                if not math.isfinite(value):
                    raise DatasetIntegrityError("non-finite table number")
            else:
                value = int(value)
            values.append(value)
        rows.append(values)
    return canonicalize(
        {"columns": list(columns), "dtypes": {n: DTYPES[n] for n in columns}, "rows": rows}
    )


def _frame(raw, columns):
    value = _json(raw)
    if (
        type(value) is not dict
        or set(value) != {"columns", "dtypes", "rows"}
        or value["columns"] != list(columns)
        or value["dtypes"] != {n: DTYPES[n] for n in columns}
        or type(value["rows"]) is not list
    ):
        raise DatasetIntegrityError("invalid ordered table schema")
    rows = []
    for row in value["rows"]:
        if type(row) is not list or len(row) != len(columns):
            raise DatasetIntegrityError("invalid table row")
        converted = []
        for name, item in zip(columns, row, strict=True):
            dtype = DTYPES[name]
            if dtype == "datetime64[ns, UTC]":
                item = pd.Timestamp(_time(item))
            elif dtype == "float64":
                if type(item) not in (int, float) or not math.isfinite(item):
                    raise DatasetIntegrityError("invalid float64 table token")
                item = float(item)
            elif type(item) is not int or not 0 <= item <= (1 if dtype == "int8" else 2**63 - 1):
                raise DatasetIntegrityError("invalid integer table token")
            converted.append(item)
        rows.append(converted)
    frame = pd.DataFrame(rows, columns=columns)
    for name in columns:
        frame[name] = frame[name].astype(DTYPES[name])
    return frame


@dataclass(frozen=True, slots=True)
class SyntheticMarket:
    """Reproducible generated OHLCV only; no caller frame, CSV path, or live loader."""

    start_at: str
    hours: int
    seed: int = 0

    def snapshot_bytes(self) -> bytes:
        with _errors():
            if type(self) is not SyntheticMarket:
                raise DatasetAuthorizationError("exact synthetic market generator required")
            start = _time(self.start_at)
            if (
                start.minute
                or start.second
                or start.microsecond
                or type(self.hours) is not int
                or not 40 <= self.hours <= 10000
                or type(self.seed) is not int
                or not 0 <= self.seed <= 2**32 - 1
            ):
                raise DatasetInputError("invalid synthetic hourly generator bounds")
            rows = []
            for index in range(self.hours):
                noise = int(sha256_bytes(f"{self.seed}:{index}".encode())[:8], 16)
                # Exercise both sides of the frozen cost threshold, not just label zero.
                opening = 10000.0 + index * 0.5 + (noise % 10000) / 100.0
                closing = opening + ((noise >> 10) % 401 - 200) / 100.0
                rows.append(
                    [
                        format_utc_timestamp(start + timedelta(hours=index)),
                        opening,
                        max(opening, closing) + 5.0,
                        min(opening, closing) - 5.0,
                        closing,
                        100.0 + float(noise % 100),
                    ]
                )
            return canonicalize(
                {
                    "schema_version": "synthetic-market-v1",
                    "synthetic": True,
                    "generator": {
                        "start_at": self.start_at,
                        "hours": self.hours,
                        "seed": self.seed,
                    },
                    "columns": list(RAW_COLUMNS),
                    "dtypes": {n: DTYPES[n] for n in RAW_COLUMNS},
                    "rows": rows,
                }
            )

    def frame(self) -> pd.DataFrame:
        with _errors():
            data = _json(self.snapshot_bytes())
            return _frame(
                canonicalize({key: data[key] for key in ("columns", "dtypes", "rows")}), RAW_COLUMNS
            )


@dataclass(frozen=True, slots=True)
class SyntheticDatasetInput:
    market: SyntheticMarket
    aggregation_id: str
    protocol_bytes: bytes
    code_commit: str
    dependency_lock_bytes: bytes
    phase2_dependency_lock_bytes: bytes


def _validate_request(request):
    if type(request) is not SyntheticDatasetInput or type(request.market) is not SyntheticMarket:
        raise DatasetAuthorizationError("only explicit generated synthetic inputs are admitted")
    if not _hash(request.aggregation_id):
        raise DatasetInputError("invalid aggregation identity")
    if type(request.code_commit) is not str or not re.fullmatch(
        "[0-9a-f]{40}", request.code_commit
    ):
        raise DatasetInputError("exact Git commit identity required")
    protocol = _json(request.protocol_bytes)
    if type(protocol) is not dict or protocol.get("synthetic") is not True:
        raise DatasetAuthorizationError("protocol must be an explicit synthetic fixture")
    for name, data in zip(
        _LOCK_HASHES,
        (request.dependency_lock_bytes, request.phase2_dependency_lock_bytes),
        strict=True,
    ):
        if type(data) is not bytes or sha256_bytes(data) != _LOCK_HASHES[name]:
            raise DatasetInputError("exact frozen dependency-lock bytes required")
    if (
        _feature_configuration() != _TECHNICAL_CONFIG
        or tuple(get_expected_feature_columns()) != TECHNICAL_COLUMNS
        or tuple(settings.RAW_COLUMNS) != RAW_COLUMNS
        or tuple(settings.LABEL_COLUMNS) != LABEL_COLUMNS
        or (
            settings.TAKER_FEE_RATE,
            settings.SLIPPAGE_BPS_PER_SIDE,
            settings.HALF_SPREAD_BPS_PER_SIDE,
            settings.MIN_EDGE_BPS,
        )
        != (0.001, 2.0, 1.0, 5.0)
    ):
        raise DatasetInputError("Phase 1 configuration differs from the frozen contract")


def _parent(store, aggregation_id):
    """Capture one M4 publication, hydrate/replay its parents once, retain those buffers."""

    def metadata(value):
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "aggregation_id", "envelope_sha256"}
            or value["schema_version"] != aggregation_module.SCHEMA
            or value["aggregation_id"] != aggregation_id
            or not _hash(value["envelope_sha256"])
        ):
            raise DatasetIntegrityError("invalid aggregation metadata before payload read")

    publication = store.read_publication(
        "aggregation-" + aggregation_id, metadata_prevalidator=metadata
    )
    artifact = AggregationArtifact(tuple(sorted(publication.files.items())))
    decoded = aggregation_module._read_candidate(artifact)
    if (
        artifact.aggregation_id != aggregation_id
        or sha256_bytes(publication.files["envelope.json"])
        != publication.manifest["metadata"]["envelope_sha256"]
    ):
        raise DatasetIntegrityError("aggregation manifest identity mismatch")
    source = decoded["input.json"]
    if (
        type(source) is not dict
        or set(source)
        != {
            "schema_version",
            "synthetic",
            "state_publication_id",
            "coverage_as_of",
            "protocol_config_sha256",
            "score_artifacts",
        }
        or source["synthetic"] is not True
    ):
        raise DatasetIntegrityError("invalid aggregation request evidence")
    scores = tuple(
        ScoreArtifact(tuple(sorted((name, bytes.fromhex(raw)) for name, raw in item.items())))
        for item in source["score_artifacts"]
    )
    request = SyntheticAggregationInput(
        source["state_publication_id"],
        scores,
        source["coverage_as_of"],
        source["protocol_config_sha256"],
    )
    prepared = aggregation_module._prepare(store, request)
    expected = aggregation_module._build(request, prepared, tuple(decoded["decisions.json"]))
    if expected.files != artifact.files:
        raise DatasetIntegrityError("aggregation semantic replay mismatch")
    for _, raw in artifact.files:
        if store.get_bytes(sha256_bytes(raw)) != raw:
            raise DatasetIntegrityError("aggregation CAS dependency mismatch")
    return artifact, request, prepared


def _configs(aggregation):
    cost = {
        "horizon": 4,
        "fee_rate": 0.001,
        "slippage_bps_per_side": 2.0,
        "half_spread_bps_per_side": 1.0,
        "minimum_net_edge_bps": 5.0,
        "minimum_required_return": minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0),
        "positive_label": "gross_forward_return_strictly_greater_than_threshold",
        "entry": "original_ordinal_plus_1",
        "exit": "original_ordinal_plus_5",
    }
    return {
        "config/technical.json": canonicalize(_TECHNICAL_CONFIG),
        "config/deduplication.json": canonicalize(
            {
                "normalizer_version": NORMALIZER_VERSION,
                "url_version": URL_NORMALIZER_VERSION,
                "text_version": TEXT_NORMALIZER_VERSION,
                "language_version": LANGUAGE_MAP_VERSION,
                "fingerprint": "normalized_title_language_dedup-fingerprint-v1",
                "window_hours": 72,
                "match": "identical_fingerprint_distinct_url_or_source",
                "representative": "permanent_earliest_first_seen_then_article_id",
                "revisions": "forward_only_same_time_conflicts_excluded_no_group_merge",
            }
        ),
        "config/aggregation.json": dict(aggregation.files)["config.json"],
        "config/label.json": canonicalize(cost),
        "config/missing-data.json": canonicalize(
            {
                "no_news": "verified_M4_zero_plus_indicator_unchanged",
                "gap": "provider_gap_window",
                "gap_scope": "all_cells_before_folds",
                "unmatched": "fail_closed",
                "fill": False,
                "diagnostics_are_features": False,
            }
        ),
        "config/integration.json": canonicalize(
            {
                "schema_version": SCHEMA,
                "synthetic_only": True,
                "decision": "original_candle_open_plus_1h",
                "join": "one_to_one_exact_left",
                "columns": list(COMBINED_TABLE_COLUMNS),
                "label_columns": list(LABEL_COLUMNS),
                "dtypes": DTYPES,
                "serialization": "RFC8785_columns_dtypes_rows-v1",
                "ordinal": "full_market_before_warmup_and_gap_exclusion",
                "purge_original_rows": 5,
                "purge_boundary": "training_exit_lt_validation_candle_open",
                "code_commit_scope": "clean_local_implementation_commit_with_frozen_locks",
                "python_runtime": sys.version,
                "machine": platform.machine(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "ta")},
                "identity_build_timestamp": None,
            }
        ),
    }


def _build(store, request):
    _validate_request(request)
    raw_market = request.market.snapshot_bytes()
    market_value = _json(raw_market)
    market = _frame(
        canonicalize({k: market_value[k] for k in ("columns", "dtypes", "rows")}), RAW_COLUMNS
    )
    aggregation, parent_input, prepared = _parent(store, request.aggregation_id)
    if parent_input.protocol_config_sha256 != sha256_bytes(request.protocol_bytes):
        raise DatasetIntegrityError("protocol detached from article and sentiment parents")
    technical = compute_features(market)
    threshold = minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0)
    labeled = add_labels(technical, horizon=4, minimum_required_return=threshold)
    sentiment = {row.decision_at: row for row in aggregation.rows}
    files = _configs(aggregation)
    combined_rows, labeled_rows, index_rows, dispositions = [], [], [], []
    warmup = len(market) - len(technical)
    for ordinal in range(warmup):
        dispositions.append(
            {
                "market_ordinal": ordinal,
                "decision_at": format_utc_timestamp(
                    (market.loc[ordinal, "timestamp"] + pd.Timedelta(hours=1)).to_pydatetime()
                ),
                "reasons": ["technical_warmup"],
            }
        )
    gap_count = tail_count = excluded_union = 0
    for ordinal, technical_row in technical.iterrows():
        decision = format_utc_timestamp(
            (technical_row["timestamp"] + pd.Timedelta(hours=1)).to_pydatetime()
        )
        if decision not in sentiment:
            raise DatasetIntegrityError(
                "missing exact sentiment decision; no implicit no-news fill"
            )
        news = sentiment[decision]
        gap = news.exclusion_reason == "provider_gap_window"
        tail = ordinal not in labeled.index
        gap_count += int(gap)
        tail_count += int(tail)
        reasons = (["provider_gap_window"] if gap else []) + (["unlabeled_tail"] if tail else [])
        if reasons:
            excluded_union += 1
            dispositions.append(
                {"market_ordinal": int(ordinal), "decision_at": decision, "reasons": reasons}
            )
        if gap:
            continue
        if news.features is None:
            raise DatasetIntegrityError("sentiment row missing payload without verified exclusion")
        row = [int(ordinal), pd.Timestamp(decision)] + [technical_row[n] for n in RAW_COLUMNS]
        row += [float(technical_row[n]) for n in TECHNICAL_COLUMNS]
        feature_values = news.features.to_dict()
        row += [feature_values[n] for n in SENTIMENT_COLUMNS]
        combined_rows.append(row)
        if not tail:
            labeled_rows.append(row + [labeled.loc[ordinal, n] for n in LABEL_COLUMNS])
        index_rows.append(
            {
                "market_ordinal": int(ordinal),
                "decision_at": decision,
                "inference": True,
                "labeled": not tail,
            }
        )
    if not combined_rows or not labeled_rows:
        raise DatasetInputError(
            "no retained inference/labeled decisions; cannot publish empty model views"
        )
    combined_raw = _table(
        pd.DataFrame(combined_rows, columns=COMBINED_TABLE_COLUMNS), COMBINED_TABLE_COLUMNS
    )
    labeled_raw = _table(
        pd.DataFrame(labeled_rows, columns=LABELED_TABLE_COLUMNS), LABELED_TABLE_COLUMNS
    )
    files.update(
        {
            "market.json": raw_market,
            "combined.json": combined_raw,
            "labeled.json": labeled_raw,
            "market-price-context.json": _table(market, RAW_COLUMNS),
            "row-index.json": canonicalize(index_rows),
            "exclusions.json": canonicalize(dispositions),
            "protocol.json": request.protocol_bytes,
            "requirements-lock.txt": request.dependency_lock_bytes,
            "requirements-phase2.txt": request.phase2_dependency_lock_bytes,
            "input.json": canonicalize(
                {
                    "schema_version": "synthetic-dataset-input-v1",
                    "synthetic": True,
                    "market": market_value["generator"],
                    "aggregation_id": request.aggregation_id,
                    "code_commit": request.code_commit,
                }
            ),
        }
    )
    files.update({"parents/aggregation/" + name: raw for name, raw in aggregation.files})
    files.update({"parents/articles/" + name: raw for name, raw in prepared.state_files})
    score_inventory, configs = [], {}
    for score in parent_input.score_artifacts:
        record = score.record
        score_files = dict(score.files)
        digest = sha256_bytes(score_files["config.json"])
        configs[digest] = score_files["config.json"]
        score_inventory.append(
            {
                "score_id": record.score_id,
                "envelope_sha256": sha256_bytes(score_files["envelope.json"]),
            }
        )
        files.update(
            {f"parents/scores/{record.score_id}/" + name: raw for name, raw in score.files}
        )
    files["score-inventory.json"] = canonicalize(
        sorted(score_inventory, key=lambda v: v["score_id"])
    )
    files["config/scoring.json"] = canonicalize(sorted(configs))
    counts = {
        "market_rows": len(market),
        "technical_warmup_rows": warmup,
        "technical_decisions": len(technical),
        "provider_gap_rows": gap_count,
        "inference_rows": len(combined_rows),
        "realizable_labels_before_gaps": len(labeled),
        "unlabeled_tail_rows_before_gaps": tail_count,
        "labeled_rows": len(labeled_rows),
        "technical_exclusion_union": excluded_union,
    }
    manifest = {
        "schema_version": SCHEMA,
        "specification_id": SPECIFICATION_ID,
        "synthetic": True,
        "code_commit": request.code_commit,
        **{key: sha256_bytes(files[name]) for key, name in _HASH_FILES.items()},
        "technical_feature_columns": list(TECHNICAL_COLUMNS),
        "sentiment_feature_columns": list(SENTIMENT_COLUMNS),
        "combined_feature_columns": list(COMBINED_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "column_dtypes": DTYPES,
        "coverage_as_of": parent_input.coverage_as_of,
        "row_counts": counts,
        "files": {
            n: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
            for n, raw in sorted(files.items())
        },
    }
    files[MANIFEST] = canonicalize(manifest)
    return PreparedDatasetArtifact(tuple(sorted(files.items())))


def _candidate(artifact):
    if type(artifact) is not PreparedDatasetArtifact or type(artifact.files) is not tuple:
        raise DatasetIntegrityError("invalid immutable candidate type")
    files = dict(artifact.files)
    if len(files) != len(artifact.files) or any(type(raw) is not bytes for raw in files.values()):
        raise DatasetIntegrityError("duplicate or non-byte candidate files")
    manifest = _json(files[MANIFEST])
    if (
        type(manifest) is not dict
        or set(manifest) != MANIFEST_FIELDS
        or manifest["schema_version"] != SCHEMA
        or manifest["specification_id"] != SPECIFICATION_ID
        or manifest["synthetic"] is not True
        or type(manifest["files"]) is not dict
    ):
        raise DatasetIntegrityError("invalid prepared manifest schema")
    expected = {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in files.items()
        if name != MANIFEST
    }
    if manifest["files"] != expected:
        raise DatasetIntegrityError("prepared manifest byte inventory mismatch")
    for name in files:
        _validate_relative_path(name)
    for key, name in _HASH_FILES.items():
        if not _hash(manifest[key]) or manifest[key] != sha256_bytes(files[name]):
            raise DatasetIntegrityError("prepared core/config/lock hash mismatch")
    if (
        manifest["technical_feature_columns"] != list(TECHNICAL_COLUMNS)
        or manifest["sentiment_feature_columns"] != list(SENTIMENT_COLUMNS)
        or manifest["combined_feature_columns"] != list(COMBINED_COLUMNS)
        or manifest["label_columns"] != list(LABEL_COLUMNS)
        or manifest["column_dtypes"] != DTYPES
    ):
        raise DatasetIntegrityError("prepared column identity mismatch")
    return files, manifest


@dataclass(frozen=True, slots=True)
class PreparedDatasetArtifact:
    """Immutable candidate buffers; typed properties alone do not certify external parents."""

    files: tuple[tuple[str, bytes], ...]

    @property
    def dataset_id(self) -> str:
        with _errors():
            files, _ = _candidate(self)
            return sha256_bytes(files[MANIFEST])

    @property
    def manifest(self) -> dict:
        with _errors():
            return _candidate(self)[1]

    @property
    def features(self) -> pd.DataFrame:
        with _errors():
            files, _ = _candidate(self)
            frame = _frame(files["combined.json"], COMBINED_TABLE_COLUMNS)
            for values in frame[list(SENTIMENT_COLUMNS)].to_dict("records"):
                FeatureValues(**values).to_dict()
            return frame

    @property
    def labeled(self) -> pd.DataFrame:
        with _errors():
            files, _ = _candidate(self)
            return _frame(files["labeled.json"], LABELED_TABLE_COLUMNS)

    def cell(self, cell_id: str, *, labeled: bool = True) -> pd.DataFrame:
        with _errors():
            if (
                type(cell_id) is not str
                or cell_id not in {"A", "B", "C", "D"}
                or type(labeled) is not bool
            ):
                raise DatasetInputError("unknown cell or labeled-view selector")
            frame = self.labeled if labeled else self.features
            columns = TECHNICAL_COLUMNS if cell_id in {"A", "B"} else COMBINED_COLUMNS
            return frame.set_index("decision_at")[list(columns)].copy()


class OfflineDatasetBuilder:
    """Generated synthetic markets plus verified M4 state only; no real-data integration."""

    def __init__(self, store: ContentAddressedStore):
        if type(store) is not ContentAddressedStore:
            raise DatasetAuthorizationError("exact local content store required")
        self.store = store

    def prepare(self, request: SyntheticDatasetInput) -> PreparedDatasetArtifact:
        with _errors():
            _validate_request(request)
            _verify_code_provenance(request.code_commit, require_clean=True)
            return _build(self.store, request)


def _verify(store, artifact, *, require_clean=False):
    files, manifest = _candidate(artifact)
    value = _json(files["input.json"])
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "synthetic", "market", "aggregation_id", "code_commit"}
        or value["schema_version"] != "synthetic-dataset-input-v1"
        or value["synthetic"] is not True
        or type(value["market"]) is not dict
        or set(value["market"]) != {"start_at", "hours", "seed"}
        or value["code_commit"] != manifest["code_commit"]
    ):
        raise DatasetIntegrityError("invalid synthetic preparation identity")
    request = SyntheticDatasetInput(
        SyntheticMarket(**value["market"]),
        value["aggregation_id"],
        files["protocol.json"],
        value["code_commit"],
        files["requirements-lock.txt"],
        files["requirements-phase2.txt"],
    )
    _validate_request(request)
    _verify_code_provenance(request.code_commit, require_clean=require_clean)
    expected = _build(store, request)
    if dict(expected.files) != files:
        raise DatasetIntegrityError("dataset parent/join/label/config semantic replay mismatch")


def _publish_verified(store, publication_id, files, metadata):
    """Read back every staged byte BEFORE creating the outer completion manifest."""
    _require_descriptor_relative_mutations()
    _require_atomic_rename_directory_no_replace_at()
    staging_name = ".staging-" + publication_id + "-" + uuid.uuid4().hex
    with ExitStack() as descriptors:
        parent = _open_store_directory(
            store.root,
            ("publications",),
            description="prepared datasets",
            expected_root_identity=store._root_identity,
        )
        descriptors.callback(os.close, parent)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent)
        stage = None
        exists = True
        try:
            stage = _open_directory_at(parent, staging_name, description="dataset staging")
            inventory = {}
            for name, raw in sorted(files.items()):
                parts = PurePosixPath(name).parts
                directory = _ensure_directory_chain_at(
                    stage, parts[:-1], description="dataset payload"
                )
                try:
                    _write_fsynced_at(directory, parts[-1], raw)
                    captured, _ = _read_regular_file_at_once(directory, parts[-1], description=name)
                    if captured != raw or sha256_bytes(captured) != sha256_bytes(raw):
                        raise DatasetIntegrityError(
                            "staged payload readback mismatch before manifest"
                        )
                finally:
                    os.close(directory)
                inventory[name] = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
            manifest = canonicalize(
                {
                    "schema_version": PUBLICATION_SCHEMA_VERSION,
                    "publication_id": publication_id,
                    "metadata": metadata,
                    "files": inventory,
                }
            )
            _write_fsynced_at(stage, "manifest.json", manifest)
            captured_manifest, manifest_stat = _read_regular_file_at_once(
                stage, "manifest.json", description="staged manifest"
            )
            if captured_manifest != manifest:
                raise DatasetIntegrityError("staged completion marker mismatch")
            dirs = {
                PurePosixPath(*PurePosixPath(n).parts[:i]).as_posix()
                for n in files
                for i in range(1, len(PurePosixPath(n).parts))
            }
            _capture_publication_tree(
                stage,
                manifest_data=manifest,
                manifest_stat=manifest_stat,
                publication_id=publication_id,
                manifest_files=inventory,
                expected_paths=set(files) | {"manifest.json"},
                expected_directories=dirs,
            )
            _fsync_tree_directories_at(stage, description="prepared staging")
            _atomic_rename_directory_no_replace(parent, staging_name, publication_id)
            exists = False
            _fsync_directory_descriptor(parent, description="prepared publications")
        finally:
            if stage is not None:
                os.close(stage)
            if exists:
                _cleanup_staging_at(parent, staging_name)


class DatasetStore:
    """Separate prepared-dataset namespace in the explicit verified parent CAS root."""

    def __init__(self, store: ContentAddressedStore):
        if type(store) is not ContentAddressedStore:
            raise DatasetAuthorizationError("exact local content store required")
        self.cas = store

    def get(self, dataset_id: str) -> PreparedDatasetArtifact | None:
        with _errors():
            if not _hash(dataset_id):
                raise DatasetIntegrityError("invalid dataset identity")
            publication_id = PUBLICATION_PREFIX + dataset_id
            descriptor = _open_store_directory(
                self.cas.root,
                ("publications",),
                description="dataset cache",
                expected_root_identity=self.cas._root_identity,
            )
            try:
                try:
                    os.stat(publication_id, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    return None
            finally:
                os.close(descriptor)

            def metadata(value):
                if (
                    type(value) is not dict
                    or set(value) != {"schema_version", "dataset_id", "manifest_sha256"}
                    or value["schema_version"] != SCHEMA
                    or value["dataset_id"] != dataset_id
                    or value["manifest_sha256"] != dataset_id
                ):
                    raise DatasetIntegrityError("invalid prepared metadata before payload read")

            publication = self.cas.read_publication(publication_id, metadata_prevalidator=metadata)
            artifact = PreparedDatasetArtifact(tuple(sorted(publication.files.items())))
            if artifact.dataset_id != dataset_id:
                raise DatasetIntegrityError("prepared publication identity mismatch")
            _verify(self.cas, artifact)
            for _, raw in artifact.files:
                if self.cas.get_bytes(sha256_bytes(raw)) != raw:
                    raise DatasetIntegrityError("prepared CAS dependency mismatch")
            return artifact

    def publish(self, artifact: PreparedDatasetArtifact) -> PreparedDatasetArtifact:
        with _errors():
            _verify(self.cas, artifact, require_clean=True)
            dataset_id = artifact.dataset_id
            prior = self.get(dataset_id)
            if prior is not None:
                if dict(prior.files) != dict(artifact.files):
                    raise DatasetIntegrityError("immutable prepared dataset collision")
                return prior
            for _, raw in artifact.files:
                self.cas.put_bytes(raw)
            try:
                _publish_verified(
                    self.cas,
                    PUBLICATION_PREFIX + dataset_id,
                    dict(artifact.files),
                    {
                        "schema_version": SCHEMA,
                        "dataset_id": dataset_id,
                        "manifest_sha256": dataset_id,
                    },
                )
            except PublicationCollisionError as exc:
                winner = self.get(dataset_id)
                if winner is None or dict(winner.files) != dict(artifact.files):
                    raise DatasetIntegrityError("concurrent prepared collision") from exc
            loaded = self.get(dataset_id)
            if loaded is None:
                raise DatasetIntegrityError("prepared publication disappeared")
            return loaded
