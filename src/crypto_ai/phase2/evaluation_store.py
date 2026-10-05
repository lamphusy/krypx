"""Immutable, offline-only publications for synthetic holdout evaluations.

This store records exact evaluation evidence; it does not run a model, fetch data,
or turn synthetic evidence into authorization for a production holdout run.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import tempfile
import uuid
import zlib
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_ai.exceptions import CryptoAIError, PublicationCollisionError
from crypto_ai.phase2 import holdout
from crypto_ai.phase2.backtests import RANDOM_SIMULATIONS, SCENARIO_ORDER
from crypto_ai.phase2.dataset import COMBINED_COLUMNS, TECHNICAL_COLUMNS
from crypto_ai.sentiment.canonical import MAX_SAFE_INTEGER, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.storage import (
    _atomic_rename_directory_no_replace,
    _capture_publication_tree,
    _create_directory_path_without_symlinks,
    _ensure_directory_chain_at,
    _fsync_directory_descriptor,
    _fsync_tree_directories_at,
    _open_directory_at,
    _open_directory_path,
    _read_regular_file_at_once,
    _require_atomic_rename_directory_no_replace_at,
    _require_descriptor_relative_mutations,
    _stat_identity,
    _stat_tree_fingerprint,
    _write_fsynced_at,
)

SCHEMA = "phase2-evaluation-publication-v1"
MANIFEST_NAME = "evaluation_manifest.json"
# Scaler means, variances and scales are retained inside each canonical model
# file. Requiring its exact bytes also binds those fitted preprocessing values.
DEVELOPMENT_FILE_MAP = {
    "development_manifest.json": "evaluation_models/development_manifest.json",
    "development_dataset_manifest.json": "evaluation_models/development_dataset_manifest.json",
    "development_market.csv": "evaluation_models/development_market.csv",
    "development_rows.json": "evaluation_models/development_rows.json",
    "augmented_model.json": "evaluation_models/augmented.json",
    "control_model.json": "evaluation_models/control.json",
    "fit_manifest.json": "evaluation_models/fit_manifest.json",
}
SCENARIOS = ("low", "base", "high")
MODELS = ("augmented", "control")
BASELINES = ("cash", "buy_and_hold", "ema_9_21", "momentum_24")

REQUIRED_PAYLOAD_NAMES = frozenset(
    {
        "input_market_snapshot.csv",
        "input_article_snapshot.jsonl",
        "input_score_snapshot.jsonl",
        "input_feature_snapshot.csv",
        "holdout_evaluation_claim.json",
        "evaluation_models/augmented.json",
        "evaluation_models/control.json",
        "evaluation_models/fit_manifest.json",
        "evaluation_models/development_manifest.json",
        "evaluation_models/development_dataset_manifest.json",
        "evaluation_models/development_market.csv",
        "evaluation_models/development_rows.json",
        "holdout_predictions.csv",
        "metrics.json",
        "baseline_metrics.json",
        "cost_sensitivity.json",
        "trade_ledgers/random_exposure.jsonl.gz",
        "equity_curves/random_exposure.jsonl.gz",
        "trade_ledgers/random_exposure.json",
    }
    | {
        f"{directory}/{model}_{scenario}.json"
        for directory in ("trade_ledgers", "equity_curves")
        for model in MODELS + BASELINES
        for scenario in SCENARIOS
    }
)
OPTIONAL_PAYLOAD_NAMES: frozenset[str] = frozenset(
    {"input_aggregation_evidence.json", "input_news_parent.json"}
)
ALLOWED_PAYLOAD_NAMES = REQUIRED_PAYLOAD_NAMES | OPTIONAL_PAYLOAD_NAMES
# The fixed core inventory is useful to callers constructing a complete bundle.
PAYLOAD_NAMES = REQUIRED_PAYLOAD_NAMES
METADATA_NAMES = frozenset(
    {
        "synthetic",
        "specification_id",
        "claim_sha256",
        "claim_bytes_sha256",
        "development_manifest_sha256",
        "fit_manifest_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "protocol_sha256",
        "code_commit",
        "dependency_lock_sha256",
        "input_inventory",
        "input_inventory_sha256",
        "selected_augmented_cell",
        "matched_control_cell",
        "augmented_feature_columns",
        "control_feature_columns",
        "common_window",
        "cost_scenarios",
        "random_simulations",
        "baseline_mode",
        "gate_version",
    }
)
METADATA_HASH_NAMES = frozenset(
    {
        "claim_sha256",
        "claim_bytes_sha256",
        "development_manifest_sha256",
        "fit_manifest_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "protocol_sha256",
        "dependency_lock_sha256",
        "input_inventory_sha256",
    }
)


class EvaluationError(CryptoAIError):
    """An offline evaluation publication contract failed."""


class EvaluationInputError(EvaluationError):
    """Candidate evidence, metadata, or a store path is invalid."""


class EvaluationIntegrityError(EvaluationError):
    """Published bytes or filesystem state failed verification."""


class EvaluationCollisionError(EvaluationIntegrityError):
    """A run identifier is occupied and must not be replaced."""


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    """A verified manifest and the exact captured payload bytes."""

    manifest: dict
    files: dict[str, bytes]


@contextmanager
def _errors():
    try:
        yield
    except EvaluationError:
        raise
    except PublicationCollisionError as exc:
        raise EvaluationCollisionError("evaluation run already exists") from exc
    except (
        CryptoAIError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        IndexError,
        RuntimeError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise EvaluationIntegrityError("invalid synthetic evaluation evidence") from exc


def _run_id(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise EvaluationInputError("unsafe evaluation run ID")
    return value


def _metadata(value: object) -> tuple[dict, bytes]:
    if (
        type(value) is not dict
        or set(value) != METADATA_NAMES
        or value.get("synthetic") is not True
    ):
        raise EvaluationInputError("evaluation metadata must be a synthetic JSON object")
    try:
        raw = canonicalize(value)
        copy = _json(raw)
    except (CryptoAIError, TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise EvaluationInputError("evaluation metadata is not canonicalizable") from exc
    if type(copy) is not dict or copy.get("synthetic") is not True:
        raise EvaluationInputError("evaluation metadata must be a synthetic JSON object")
    if (
        copy["specification_id"] != "phase2-milestone9-one-time-future-evaluation-v1"
        or any(not _hash(copy[name]) for name in METADATA_HASH_NAMES)
        or copy["claim_sha256"] != copy["claim_bytes_sha256"]
        or type(copy["code_commit"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", copy["code_commit"]) is None
        or (copy["selected_augmented_cell"], copy["matched_control_cell"])
        not in {("C", "A"), ("D", "B")}
        or copy["augmented_feature_columns"] != list(COMBINED_COLUMNS)
        or copy["control_feature_columns"] != list(TECHNICAL_COLUMNS)
        or copy["cost_scenarios"] != list(SCENARIO_ORDER)
        or copy["gate_version"] != "phase2-final-evidence-gates-v1"
        or type(copy["random_simulations"]) is not int
        or not 1 <= copy["random_simulations"] <= RANDOM_SIMULATIONS
        or (
            copy["baseline_mode"] == "FULL_1000"
            and copy["random_simulations"] != RANDOM_SIMULATIONS
        )
        or (
            copy["baseline_mode"] == "REDUCED_SYNTHETIC_FIXTURE"
            and copy["random_simulations"] == RANDOM_SIMULATIONS
        )
        or copy["baseline_mode"] not in {"FULL_1000", "REDUCED_SYNTHETIC_FIXTURE"}
        or type(copy["common_window"]) is not dict
    ):
        raise EvaluationInputError("evaluation metadata violates frozen identities")
    inventory = copy["input_inventory"]
    if (
        type(inventory) is not dict
        or set(inventory)
        not in (
            {
                "evaluation_run_id",
                "market_snapshot_sha256",
                "article_snapshot_sha256",
                "score_snapshot_sha256",
                "feature_snapshot_sha256",
            },
            {
                "evaluation_run_id",
                "market_snapshot_sha256",
                "article_snapshot_sha256",
                "score_snapshot_sha256",
                "feature_snapshot_sha256",
                "aggregation_evidence_sha256",
                "verified_news_parent_sha256",
            },
        )
        or any(not _hash(value) for key, value in inventory.items() if key != "evaluation_run_id")
        or type(inventory["evaluation_run_id"]) is not str
        or sha256_bytes(canonicalize(inventory)) != copy["input_inventory_sha256"]
    ):
        raise EvaluationInputError("evaluation input inventory is not cryptographically bound")
    return copy, raw


def _require_metadata_unchanged(value: object, expected: bytes) -> None:
    try:
        current = canonicalize(value) if type(value) is dict else None
    except (CryptoAIError, TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise EvaluationIntegrityError("caller metadata changed during publication") from exc
    if current != expected:
        raise EvaluationIntegrityError("caller metadata changed during publication")


def _files(value: object) -> dict[str, bytes]:
    if type(value) is not dict:
        raise EvaluationInputError("evaluation files must be a path-to-bytes dictionary")
    try:
        names = set(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError("evaluation file paths are invalid") from exc
    if not REQUIRED_PAYLOAD_NAMES <= names or not names <= ALLOWED_PAYLOAD_NAMES:
        raise EvaluationInputError("evaluation payload inventory is missing or has unknown paths")
    if any(type(name) is not str or type(raw) is not bytes for name, raw in value.items()):
        raise EvaluationInputError("evaluation payloads require exact string paths and bytes")
    return dict(value)


def _inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in sorted(files.items())
    }


def _json(raw: bytes) -> object:
    if type(raw) is not bytes or len(raw) > 128_000_000:
        raise EvaluationIntegrityError("bounded exact JSON bytes are required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvaluationIntegrityError("duplicate evaluation manifest field")
            result[key] = value
        return result

    def invalid_constant(value):
        raise EvaluationIntegrityError(f"non-finite JSON value {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
        if canonicalize(value) != raw:
            raise EvaluationIntegrityError("evaluation JSON is not canonical")
    except (json.JSONDecodeError, UnicodeError, CryptoAIError, ValueError, TypeError) as exc:
        if isinstance(exc, EvaluationIntegrityError):
            raise
        raise EvaluationIntegrityError("invalid evaluation JSON") from exc
    return value


def _hash(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ordinal(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_SAFE_INTEGER


def _utc(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise EvaluationIntegrityError("claim timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EvaluationIntegrityError("claim timestamp is invalid") from exc
    if parsed.tzinfo != UTC or format_utc_timestamp(parsed) != value:
        raise EvaluationIntegrityError("claim timestamp is not canonical UTC")
    return parsed


def _claim_provenance(files: Mapping[str, bytes], metadata: dict, run_id: str) -> dict:
    """Verify the retained claim through the exact frozen development evidence."""
    claim_raw = files[holdout.CLAIM_NAME]
    claim = _json(claim_raw)
    development = _json(files["evaluation_models/development_manifest.json"])
    fit = _json(files["evaluation_models/fit_manifest.json"])
    claim_names = {
        "schema_version",
        "specification_id",
        "synthetic",
        "run_id",
        "evaluation_run_id",
        "protocol_sha256",
        "input_inventory_sha256",
        "code_commit",
        "dependency_lock_sha256",
        "development_cutoff_ordinal",
        "purge_ordinals",
        "first_holdout_ordinal",
        "first_holdout_decision_at",
        "last_development_exit_at",
        "selected_augmented_cell",
        "matched_control_cell",
        "shared_labeled_rows_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "fit_manifest_sha256",
        "development_dataset_manifest_sha256",
        "development_cutoff_iso",
        "frozen_at",
        "readiness_plan_sha256",
        "frozen_policy_sha256",
        "operational_evidence_sha256",
        "readiness",
        "claimed_at_utc",
    }
    development_names = {
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
    fit_names = {
        "schema_version",
        "specification_id",
        "synthetic",
        "selected_augmented_cell",
        "matched_control_cell",
        "development_cutoff_ordinal",
        "shared_labeled_rows_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "augmented_fit_count",
        "control_fit_count",
        "frozen_at_utc",
    }
    if (
        type(claim) is not dict
        or set(claim) != claim_names
        or type(development) is not dict
        or set(development) != development_names
        or type(fit) is not dict
        or set(fit) != fit_names
        or sha256_bytes(claim_raw) != metadata["claim_sha256"]
        or sha256_bytes(claim_raw) != metadata["claim_bytes_sha256"]
        or claim["schema_version"] != holdout.EVALUATION_CLAIM_SCHEMA
        or claim["specification_id"] != holdout.SPECIFICATION_ID
        or claim["synthetic"] is not True
        or development["schema_version"] != "phase2-synthetic-development-provenance-v1"
        or development["synthetic"] is not True
        or fit["schema_version"] != holdout.FIT_SCHEMA
        or fit["specification_id"] != holdout.SPECIFICATION_ID
        or fit["synthetic"] is not True
        or type(claim["run_id"]) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", claim["run_id"]) is None
        or claim["run_id"] != development["run_id"]
        or claim["run_id"] == run_id
        or claim["evaluation_run_id"] != run_id
    ):
        raise EvaluationIntegrityError("retained holdout claim or development schema is invalid")
    for field in (
        "protocol_sha256",
        "input_inventory_sha256",
        "code_commit",
        "dependency_lock_sha256",
        "selected_augmented_cell",
        "matched_control_cell",
        "augmented_model_sha256",
        "control_model_sha256",
        "fit_manifest_sha256",
    ):
        if claim[field] != metadata[field]:
            raise EvaluationIntegrityError(f"retained holdout claim disagrees on {field}")
    if (
        claim["frozen_policy_sha256"] != metadata["protocol_sha256"]
        or any(
            not _hash(claim[field])
            for field in (
                "shared_labeled_rows_sha256",
                "readiness_plan_sha256",
                "operational_evidence_sha256",
                "development_dataset_manifest_sha256",
            )
        )
        or not _ordinal(claim["development_cutoff_ordinal"])
        or not _ordinal(claim["first_holdout_ordinal"])
        or claim["first_holdout_ordinal"]
        != claim["development_cutoff_ordinal"] + holdout.PURGE_ROWS + 1
        or type(claim["purge_ordinals"]) is not list
        or any(type(ordinal) is not int for ordinal in claim["purge_ordinals"])
        or claim["purge_ordinals"]
        != list(range(claim["development_cutoff_ordinal"] + 1, claim["first_holdout_ordinal"]))
    ):
        raise EvaluationIntegrityError("retained holdout claim boundary is invalid")
    for field in (
        "development_cutoff_ordinal",
        "purge_ordinals",
        "first_holdout_ordinal",
        "selected_augmented_cell",
        "matched_control_cell",
        "shared_labeled_rows_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
        "fit_manifest_sha256",
    ):
        if claim[field] != development[field]:
            raise EvaluationIntegrityError(f"development manifest disagrees on {field}")
    for field in (
        "development_cutoff_ordinal",
        "selected_augmented_cell",
        "matched_control_cell",
        "shared_labeled_rows_sha256",
        "augmented_model_sha256",
        "control_model_sha256",
    ):
        if claim[field] != fit[field]:
            raise EvaluationIntegrityError(f"fit manifest disagrees on {field}")
    if (
        any(
            type(record[field]) is not int or record[field] != 1
            for record in (development, fit)
            for field in ("augmented_fit_count", "control_fit_count")
        )
        or not _ordinal(development["fitted_max_decision_ordinal"])
        or development["fitted_max_decision_ordinal"] != claim["development_cutoff_ordinal"]
        or development["augmented_feature_columns"] != metadata["augmented_feature_columns"]
        or development["control_feature_columns"] != metadata["control_feature_columns"]
        or development["development_rows_sha256"] != claim["shared_labeled_rows_sha256"]
        or not _hash(development["development_market_sha256"])
        or sha256_bytes(files["evaluation_models/development_market.csv"])
        != development["development_market_sha256"]
        or sha256_bytes(files["evaluation_models/development_rows.json"])
        != development["development_rows_sha256"]
        or development["model_family"]
        != ("LogisticRegression" if claim["selected_augmented_cell"] == "C" else "XGBClassifier")
        or type(development["signal_threshold"]) not in (int, float)
        or development["signal_threshold"] != 0.5
        or fit["frozen_at_utc"] != claim["frozen_at"]
    ):
        raise EvaluationIntegrityError("retained single-fit provenance is invalid")
    from crypto_ai.phase2 import evaluation

    try:
        replay_rows, replay_augmented, replay_control = evaluation.build_synthetic_development_fit(
            files["evaluation_models/development_market.csv"],
            claim["development_cutoff_ordinal"],
            (claim["selected_augmented_cell"], claim["matched_control_cell"]),
        )
    except (CryptoAIError, ValueError, TypeError, OverflowError) as exc:
        raise EvaluationIntegrityError("retained development fit cannot be replayed") from exc
    if (
        replay_rows != files["evaluation_models/development_rows.json"]
        or replay_augmented != files["evaluation_models/augmented.json"]
        or replay_control != files["evaluation_models/control.json"]
    ):
        raise EvaluationIntegrityError("retained development rows or fitted model changed")
    dataset_manifest = evaluation._development_dataset_manifest_bytes(
        files["evaluation_models/development_market.csv"],
        replay_rows,
        claim["development_cutoff_ordinal"],
    )
    if (
        files["evaluation_models/development_dataset_manifest.json"] != dataset_manifest
        or sha256_bytes(dataset_manifest) != claim["development_dataset_manifest_sha256"]
    ):
        raise EvaluationIntegrityError("development dataset manifest differs from replay")
    replayed_rows = _json(replay_rows)
    if (
        type(replayed_rows) is not dict
        or type(replayed_rows.get("rows")) is not list
        or not replayed_rows["rows"]
        or replayed_rows["rows"][-1].get("decision_at") != claim["development_cutoff_iso"]
    ):
        raise EvaluationIntegrityError("development cutoff differs from retained rows")
    first_decision = _utc(claim["first_holdout_decision_at"])
    cutoff = _utc(claim["development_cutoff_iso"])
    last_exit = _utc(claim["last_development_exit_at"])
    frozen = _utc(claim["frozen_at"])
    claimed = _utc(claim["claimed_at_utc"])
    if not cutoff < last_exit <= frozen < first_decision <= claimed:
        raise EvaluationIntegrityError("retained holdout claim chronology is invalid")
    readiness = claim["readiness"]
    if (
        type(readiness) is not dict
        or set(readiness)
        != {
            "ready",
            "elapsed_days",
            "trade_threshold_met",
            "provider_outage_state",
            "planned_minimum_days",
            "synthetic",
        }
        or readiness["ready"] is not True
        or readiness["trade_threshold_met"] is not True
        or readiness["synthetic"] is not True
        or readiness["provider_outage_state"] not in {"CLEAR", "VERIFIED_GAP"}
        or type(readiness["planned_minimum_days"]) is not int
        or readiness["planned_minimum_days"] < holdout.MINIMUM_DAYS
        or type(readiness["elapsed_days"]) not in (int, float)
        or not math.isfinite(readiness["elapsed_days"])
        or readiness["elapsed_days"] < readiness["planned_minimum_days"]
    ):
        raise EvaluationIntegrityError("retained holdout readiness is invalid")
    window = metadata["common_window"]
    if (
        not _ordinal(window.get("first_open_ordinal"))
        or window["first_open_ordinal"] < claim["first_holdout_ordinal"] + 1
        or _utc(window.get("first_open_at")) < first_decision
    ):
        raise EvaluationIntegrityError("evaluation window is outside the claimed holdout")
    # The market/prediction replay below binds this open to the first retained
    # decision.  A leading verified provider gap may exclude earlier decisions
    # without changing the claimed holdout boundary itself.
    return claim


def _manifest(raw: bytes, run_id: str) -> dict:
    value = _json(raw)
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "run_id", "metadata", "files"}
        or value["schema_version"] != SCHEMA
        or value["run_id"] != run_id
    ):
        raise EvaluationIntegrityError("invalid evaluation manifest")
    try:
        _metadata(value["metadata"])
    except EvaluationInputError as exc:
        raise EvaluationIntegrityError("invalid evaluation manifest metadata") from exc
    inventory = value["files"]
    if (
        type(inventory) is not dict
        or not REQUIRED_PAYLOAD_NAMES <= set(inventory)
        or not set(inventory) <= ALLOWED_PAYLOAD_NAMES
    ):
        raise EvaluationIntegrityError("invalid evaluation payload inventory")
    for entry in inventory.values():
        if (
            type(entry) is not dict
            or set(entry) != {"sha256", "size_bytes"}
            or not _hash(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise EvaluationIntegrityError("invalid evaluation payload descriptor")
    return value


def _directories(files: Mapping[str, object]) -> set[str]:
    return {
        PurePosixPath(*PurePosixPath(name).parts[:index]).as_posix()
        for name in files
        for index in range(1, len(PurePosixPath(name).parts))
    }


def _verify_random_evidence(
    files: Mapping[str, bytes],
    metadata: dict,
    baselines: dict,
    costs: dict,
    models: dict,
    *,
    replay_paths: bool,
) -> None:
    """Bind every path; replay once per distinct captured publication."""
    from crypto_ai.phase2 import backtests, evaluation

    predictions_raw = files["holdout_predictions.csv"]
    if not predictions_raw.endswith(b"\n") or b"\r" in predictions_raw:
        raise EvaluationIntegrityError("prediction CSV has noncanonical line endings")
    try:
        rows = list(csv.reader(io.StringIO(predictions_raw.decode("utf-8")), strict=True))
        expected_header = [
            "market_ordinal",
            "decision_at",
            "augmented_probability",
            "control_probability",
            "augmented_signal",
            "control_signal",
        ]
        if not rows or rows[0] != expected_header or len(rows) < 2:
            raise EvaluationIntegrityError("holdout prediction schema is invalid")
        signals = []
        ordinals = []
        for row in rows[1:]:
            if (
                len(row) != len(expected_header)
                or re.fullmatch(r"(?:0|[1-9][0-9]*)", row[0]) is None
                or row[4] not in {"0", "1"}
            ):
                raise EvaluationIntegrityError("holdout prediction signal is invalid")
            ordinal = int(row[0])
            if ordinals and ordinal <= ordinals[-1]:
                raise EvaluationIntegrityError("holdout prediction ordinals are not ordered")
            probability = float(row[2])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise EvaluationIntegrityError("holdout prediction probability is invalid")
            signal = int(row[4])
            if signal != int(probability >= backtests.THRESHOLD):
                raise EvaluationIntegrityError("random exposure source signal changed")
            ordinals.append(ordinal)
            signals.append(signal)
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise EvaluationIntegrityError("holdout prediction CSV is malformed") from exc
    if metadata["common_window"].get("oof_decisions") != len(signals):
        raise EvaluationIntegrityError("prediction count differs from frozen window")
    window = metadata["common_window"]
    first_open = window.get("first_open_ordinal")
    final_open = window.get("final_open_ordinal")
    market = evaluation._market_frame(files["input_market_snapshot.csv"], None)
    if (
        not _ordinal(first_open)
        or not _ordinal(final_open)
        or ordinals[0] + 1 != first_open
        or ordinals[-1] + backtests.HORIZON + 1 != final_open
        or final_open >= len(market)
        or any(
            row[1] != format_utc_timestamp(market.timestamp.iloc[ordinal + 1].to_pydatetime())
            for row, ordinal in zip(rows[1:], ordinals, strict=True)
        )
        or backtests._window(market, first_open, final_open, len(signals)) != window
    ):
        raise EvaluationIntegrityError("prediction decisions differ from retained market window")
    probability = sum(signals) / len(signals)
    raw = files["trade_ledgers/random_exposure.json"]
    evidence = _json(raw)
    simulations = metadata["random_simulations"]
    if (
        type(evidence) is not dict
        or set(evidence)
        != {
            "schema_version",
            "synthetic",
            "seed_base",
            "simulations",
            "signal_probability",
            "draws_and_metrics",
        }
        or evidence["schema_version"] != "phase2-synthetic-random-exposure-v1"
        or evidence["synthetic"] is not True
        or evidence["seed_base"] != backtests.RANDOM_SEED
        or evidence["simulations"] != simulations
        or evidence["signal_probability"] != probability
        or type(evidence["draws_and_metrics"]) is not list
        or len(evidence["draws_and_metrics"]) != simulations
    ):
        raise EvaluationIntegrityError("random-exposure evidence has a forged envelope")
    samples = {
        scenario: {key: [] for key in ("total_return", "sharpe_ratio", "maximum_drawdown")}
        for scenario in SCENARIOS
    }
    ledger_zip = gzip.GzipFile(
        fileobj=io.BytesIO(files["trade_ledgers/random_exposure.jsonl.gz"]), mode="rb"
    )
    curve_zip = gzip.GzipFile(
        fileobj=io.BytesIO(files["equity_curves/random_exposure.jsonl.gz"]), mode="rb"
    )
    for index, item in enumerate(evidence["draws_and_metrics"]):
        seed = backtests.RANDOM_SEED + index
        if (
            type(item) is not dict
            or set(item) != {"simulation", "seed", "draws", "draws_sha256", "scenarios"}
            or type(item["simulation"]) is not int
            or item["simulation"] != index
            or type(item["seed"]) is not int
            or item["seed"] != seed
            or type(item["draws"]) is not list
            or len(item["draws"]) != len(signals)
            or any(type(draw) is not int or draw not in (0, 1) for draw in item["draws"])
            or item["draws_sha256"] != sha256_bytes(canonicalize(item["draws"]))
            or type(item["scenarios"]) is not dict
            or set(item["scenarios"]) != set(SCENARIOS)
        ):
            raise EvaluationIntegrityError("random-exposure draw record is invalid")
        replayed = np.random.default_rng(seed).binomial(1, probability, len(signals)).tolist()
        if replayed != item["draws"]:
            raise EvaluationIntegrityError("random-exposure draws differ from fixed seed")
        draw_series = pd.Series(
            np.asarray(item["draws"], dtype=np.float64),
            index=pd.Index(ordinals, dtype=np.int64),
        )
        for scenario in SCENARIOS:
            path = item["scenarios"][scenario]
            if (
                type(path) is not dict
                or set(path) != {"metrics", "ledger_sha256", "equity_sha256"}
                or type(path["metrics"]) is not dict
                or not _hash(path["ledger_sha256"])
                or not _hash(path["equity_sha256"])
            ):
                raise EvaluationIntegrityError("random-exposure scenario path is invalid")
            retained_paths = {}
            for stream, path_key, digest_key in (
                (ledger_zip, "trade_ledger", "ledger_sha256"),
                (curve_zip, "equity_curve", "equity_sha256"),
            ):
                try:
                    line = stream.readline(128_000_001)
                    if not line.endswith(b"\n") or len(line) > 128_000_000:
                        raise EvaluationIntegrityError("random trajectory record is incomplete")
                    record = _json(line[:-1])
                except (EOFError, OSError, UnicodeError, ValueError, zlib.error) as exc:
                    raise EvaluationIntegrityError("random trajectory gzip is invalid") from exc
                if (
                    type(record) is not dict
                    or set(record) != {"simulation", "scenario", path_key}
                    or type(record["simulation"]) is not int
                    or record["simulation"] != index
                    or record["scenario"] != scenario
                    or type(record[path_key]) is not list
                    or canonicalize(record) != line[:-1]
                    or sha256_bytes(canonicalize(record[path_key])) != path[digest_key]
                ):
                    raise EvaluationIntegrityError("random trajectory differs from its evidence")
                retained_paths[path_key] = record[path_key]
            if replay_paths:
                try:
                    replay = backtests._result_payload(
                        backtests._simulate(
                            market,
                            draw_series,
                            None,
                            backtests.SCENARIOS[scenario],
                            expected_start=first_open,
                            expected_end=final_open,
                        )
                    )
                except CryptoAIError as exc:
                    raise EvaluationIntegrityError("random trajectory replay failed") from exc
                if any(
                    canonicalize(replay[key]) != canonicalize(retained)
                    for key, retained in (
                        ("metrics", path["metrics"]),
                        ("trade_ledger", retained_paths["trade_ledger"]),
                        ("equity_curve", retained_paths["equity_curve"]),
                    )
                ):
                    raise EvaluationIntegrityError(
                        "random trajectory differs from fixed-seed replay"
                    )
            for key in samples[scenario]:
                if key not in path["metrics"]:
                    raise EvaluationIntegrityError("random-exposure metric is missing")
                samples[scenario][key].append(path["metrics"][key])
    try:
        if ledger_zip.read(1) or curve_zip.read(1):
            raise EvaluationIntegrityError("random trajectory archive has extra records")
    except (EOFError, OSError, zlib.error) as exc:
        raise EvaluationIntegrityError("random trajectory gzip is truncated") from exc
    expected_summary = {}
    for scenario in SCENARIOS:
        scenario_samples = samples[scenario]
        returns = scenario_samples["total_return"]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in returns):
            raise EvaluationIntegrityError("random-exposure return is non-finite")
        model_return = models["augmented"][scenario].get("total_return")
        if type(model_return) not in (int, float) or not math.isfinite(model_return):
            raise EvaluationIntegrityError("matched augmented return is invalid")
        expected_summary[scenario] = {
            "simulations": simulations,
            "seed_base": backtests.RANDOM_SEED,
            "signal_probability": probability,
            **{key: backtests._summary(values) for key, values in scenario_samples.items()},
            "fraction_return_at_least_model": float(
                np.mean(np.asarray(returns, dtype=np.float64) >= model_return)
            ),
        }
    if (
        baselines.get("random_exposure") != expected_summary
        or costs.get("random_exposure") != expected_summary
    ):
        raise EvaluationIntegrityError("random-exposure summaries are not replay-bound")


def _verify_deterministic_evidence(
    files: Mapping[str, bytes],
    metadata: dict,
    claim: dict,
    models: dict,
    deterministic: dict,
    metrics: dict,
) -> None:
    """Replay the frozen model and every deterministic path from captured bytes."""
    from crypto_ai.backtesting.baselines import buy_and_hold_backtest
    from crypto_ai.phase2 import backtests, evaluation

    market = evaluation._market_frame(files["input_market_snapshot.csv"], None)
    boundary = SimpleNamespace(first_holdout_ordinal=claim["first_holdout_ordinal"])
    aggregation_rows = None
    if "input_aggregation_evidence.json" in files:
        article_schema = evaluation._NewsSnapshotSchema("article")
        score_schema = evaluation._NewsSnapshotSchema("score")
        article_schema.feed(files["input_article_snapshot.jsonl"])
        score_schema.feed(files["input_score_snapshot.jsonl"])
        article_schema.finish()
        score_schema.finish()
        aggregation_rows = evaluation._aggregation_expectations(
            files["input_aggregation_evidence.json"],
            articles=tuple(article_schema.articles),
            scores=tuple(score_schema.scores),
            article_sha256=sha256_bytes(files["input_article_snapshot.jsonl"]),
            score_sha256=sha256_bytes(files["input_score_snapshot.jsonl"]),
            feature_sha256=sha256_bytes(files["input_feature_snapshot.csv"]),
            protocol_sha256=claim["protocol_sha256"],
            opened_at=[value.to_pydatetime() for value in market.timestamp],
            first_ordinal=claim["first_holdout_ordinal"],
            last_decision_ordinal=len(market) - backtests.HORIZON - 2,
            inspected_at=_utc(claim["claimed_at_utc"]),
        )
        verified_rows = evaluation._verified_news_parent_expectations(
            evaluation._parse_news_parent(files["input_news_parent.json"]),
            evidence_raw=files["input_aggregation_evidence.json"],
            articles=tuple(article_schema.articles),
            scores=tuple(score_schema.scores),
            opened_at=[value.to_pydatetime() for value in market.timestamp],
            first_ordinal=claim["first_holdout_ordinal"],
            last_decision_ordinal=len(market) - backtests.HORIZON - 2,
            development_market_raw=files["evaluation_models/development_market.csv"],
            development_rows_raw=files["evaluation_models/development_rows.json"],
            protocol_sha256=claim["protocol_sha256"],
        )
        if evaluation._feature_rows_bits(aggregation_rows) != evaluation._feature_rows_bits(
            verified_rows
        ):
            raise EvaluationIntegrityError("retained M4 replay and aggregation evidence differ")
    features = evaluation._feature_frame(
        files["input_feature_snapshot.csv"], market, boundary, aggregation_rows
    )
    evaluation._verify_technical_replay(features, market)
    augmented = evaluation._predict(
        features,
        COMBINED_COLUMNS,
        evaluation._verified_model(files["evaluation_models/augmented.json"], COMBINED_COLUMNS),
    )
    control = evaluation._predict(
        features,
        TECHNICAL_COLUMNS,
        evaluation._verified_model(files["evaluation_models/control.json"], TECHNICAL_COLUMNS),
    )
    if files["holdout_predictions.csv"] != evaluation._predictions_csv(
        features, augmented, control
    ):
        raise EvaluationIntegrityError("retained predictions differ from frozen model replay")
    first_open = int(features.market_ordinal.iloc[0]) + 1
    final_open = int(features.market_ordinal.iloc[-1]) + backtests.HORIZON + 1
    if (
        backtests._window(market, first_open, final_open, len(features))
        != metadata["common_window"]
    ):
        raise EvaluationIntegrityError("retained performance window differs from market replay")

    def verify_path(name: str, scenario: str, payload: dict, reported: dict) -> None:
        if payload["metrics"] != reported:
            raise EvaluationIntegrityError(f"{name} {scenario} metrics differ from replay")
        for directory, key in (
            ("trade_ledgers", "trade_ledger"),
            ("equity_curves", "equity_curve"),
        ):
            if files[f"{directory}/{name}_{scenario}.json"] != canonicalize(payload[key]):
                raise EvaluationIntegrityError(f"{name} {scenario} {key} differs from replay")

    model_scores = {"augmented": augmented, "control": control}
    base_reconciliation = {}
    for name, scores in model_scores.items():
        previous_schedule = None
        for scenario in SCENARIOS:
            payload = backtests._result_payload(
                backtests._simulate(
                    market,
                    scores,
                    None,
                    backtests.SCENARIOS[scenario],
                    expected_start=first_open,
                    expected_end=final_open,
                )
            )
            verify_path(name, scenario, payload, models[name][scenario])
            schedule = tuple(
                (row["entry_timestamp"], row["exit_timestamp"]) for row in payload["trade_ledger"]
            )
            if previous_schedule is not None and schedule != previous_schedule:
                raise EvaluationIntegrityError("model trade schedule changed across cost scenarios")
            previous_schedule = schedule
            if scenario == "base":
                base_reconciliation[name] = payload["reconciliation"]
    if metrics.get("reconciliation") != base_reconciliation:
        raise EvaluationIntegrityError("reported reconciliation differs from replay")

    ordinals = pd.Index(features.market_ordinal.to_numpy(dtype=np.int64))
    zero = pd.Series(0.0, index=ordinals)
    baseline_scores = {
        "cash": zero,
        "ema_9_21": pd.Series(
            (features.ema_short.to_numpy() > features.ema_long.to_numpy()).astype(np.float64),
            index=ordinals,
        ),
        "momentum_24": pd.Series(
            (features.return_24.to_numpy() > 0.0).astype(np.float64), index=ordinals
        ),
    }
    for scenario in SCENARIOS:
        config = backtests.SCENARIOS[scenario]
        for name, scores in baseline_scores.items():
            payload = backtests._result_payload(
                backtests._simulate(
                    market,
                    scores,
                    None,
                    config,
                    expected_start=first_open,
                    expected_end=final_open,
                )
            )
            verify_path(name, scenario, payload, deterministic[scenario][name])
        buy_hold = buy_and_hold_backtest(
            market,
            ordinals,
            backtests.HORIZON,
            config,
            initial_capital=backtests.INITIAL_CAPITAL,
        )
        backtests._result_window(buy_hold, market, first_open, final_open)
        verify_path(
            "buy_and_hold",
            scenario,
            backtests._result_payload(buy_hold),
            deterministic[scenario]["buy_and_hold"],
        )


def _capture(
    descriptor: int,
    run_id: str,
    *,
    expected_manifest_bytes: bytes | None = None,
    replay_random_paths: bool = True,
    replay_deterministic_paths: bool = True,
) -> EvaluationArtifact:
    raw, info = _read_regular_file_at_once(
        descriptor, MANIFEST_NAME, description="evaluation manifest"
    )
    if expected_manifest_bytes is not None and raw != expected_manifest_bytes:
        raise EvaluationIntegrityError("evaluation manifest bytes changed")
    manifest = _manifest(raw, run_id)
    captured, _ = _capture_publication_tree(
        descriptor,
        manifest_data=raw,
        manifest_stat=info,
        publication_id=run_id,
        manifest_files=manifest["files"],
        expected_paths=set(manifest["files"]) | {MANIFEST_NAME},
        expected_directories=_directories(manifest["files"]),
    )
    # The shared tree walker reserves "manifest.json" for older publication
    # contracts. Here the named evaluation manifest is captured as a regular
    # member, so bind that capture back to the original manifest read as well.
    if captured[MANIFEST_NAME] != raw or _stat_tree_fingerprint(
        os.stat(MANIFEST_NAME, dir_fd=descriptor, follow_symlinks=False)
    ) != _stat_tree_fingerprint(info):
        raise EvaluationIntegrityError("evaluation manifest changed during capture")
    files = {name: captured[name] for name in manifest["files"]}
    for name, payload in files.items():
        if name.endswith(".json") and canonicalize(_json(payload)) != payload:
            raise EvaluationIntegrityError(f"noncanonical evaluation JSON payload: {name}")
    metadata = manifest["metadata"]
    inventory = metadata["input_inventory"]
    if (
        inventory["evaluation_run_id"] != run_id
        or sha256_bytes(files["input_market_snapshot.csv"]) != inventory["market_snapshot_sha256"]
        or sha256_bytes(files["input_article_snapshot.jsonl"])
        != inventory["article_snapshot_sha256"]
        or sha256_bytes(files["input_score_snapshot.jsonl"]) != inventory["score_snapshot_sha256"]
        or sha256_bytes(files["input_feature_snapshot.csv"]) != inventory["feature_snapshot_sha256"]
        or ("input_aggregation_evidence.json" in files)
        != ("aggregation_evidence_sha256" in inventory)
        or ("input_news_parent.json" in files) != ("verified_news_parent_sha256" in inventory)
        or (
            "aggregation_evidence_sha256" in inventory
            and sha256_bytes(files["input_aggregation_evidence.json"])
            != inventory["aggregation_evidence_sha256"]
        )
        or (
            "verified_news_parent_sha256" in inventory
            and sha256_bytes(files["input_news_parent.json"])
            != inventory["verified_news_parent_sha256"]
        )
        or sha256_bytes(files["evaluation_models/augmented.json"])
        != metadata["augmented_model_sha256"]
        or sha256_bytes(files["evaluation_models/control.json"]) != metadata["control_model_sha256"]
        or sha256_bytes(files["evaluation_models/fit_manifest.json"])
        != metadata["fit_manifest_sha256"]
        or sha256_bytes(files["evaluation_models/development_manifest.json"])
        != metadata["development_manifest_sha256"]
    ):
        raise EvaluationIntegrityError("evaluation manifest metadata and payloads disagree")
    claim = _claim_provenance(files, metadata, run_id)
    metrics = _json(files["metrics.json"])
    baselines = _json(files["baseline_metrics.json"])
    costs = _json(files["cost_sensitivity.json"])
    if (
        type(metrics) is not dict
        or type(baselines) is not dict
        or type(costs) is not dict
        or metrics.get("common_window") != metadata["common_window"]
        or baselines.get("common_window") != metadata["common_window"]
        or costs.get("common_window") != metadata["common_window"]
        or metrics.get("synthetic") is not True
        or baselines.get("synthetic") is not True
        or costs.get("synthetic") is not True
    ):
        raise EvaluationIntegrityError("evaluation reports diverge from frozen window metadata")
    deterministic = baselines.get("deterministic")
    if (
        type(deterministic) is not dict
        or set(deterministic) != set(SCENARIOS)
        or any(
            type(deterministic[scenario]) is not dict
            or set(deterministic[scenario]) != set(BASELINES)
            for scenario in SCENARIOS
        )
        or costs.get("baselines") != deterministic
    ):
        raise EvaluationIntegrityError("deterministic baseline reports disagree")
    from crypto_ai.phase2.evaluation import evaluate_final_gates

    models = costs.get("models")
    if (
        type(models) is not dict
        or set(models) != set(MODELS)
        or any(
            type(models[role]) is not dict or set(models[role]) != set(SCENARIOS) for role in MODELS
        )
        or metrics.get("base") != models["augmented"]["base"]
        or metrics.get("control_base") != models["control"]["base"]
    ):
        raise EvaluationIntegrityError("strategy reports disagree on official base metrics")
    _verify_random_evidence(
        files, metadata, baselines, costs, models, replay_paths=replay_random_paths
    )
    if replay_deterministic_paths:
        _verify_deterministic_evidence(files, metadata, claim, models, deterministic, metrics)
    expected_gates = evaluate_final_gates(
        augmented_metrics=metrics["base"],
        control_metrics=metrics["control_base"],
        cash_metrics=deterministic["base"]["cash"],
        augmented_ledger=_json(files["trade_ledgers/augmented_base.json"]),
        control_ledger=_json(files["trade_ledgers/control_base.json"]),
        elapsed_days=float(claim["readiness"]["elapsed_days"]),
        planned_minimum_days=claim["readiness"]["planned_minimum_days"],
    )
    if (
        metrics.get("final_gates") != expected_gates
        or metrics.get("research_verdict") != expected_gates["research_verdict"]
        or metrics.get("production_decision") != "NO-GO"
        or metrics.get("engineering_status")
        != ("PASS" if metadata["baseline_mode"] == "FULL_1000" else "REDUCED_SYNTHETIC_FIXTURE")
    ):
        raise EvaluationIntegrityError("reported final gates differ from retained trade evidence")
    for name in REQUIRED_PAYLOAD_NAMES:
        if name.startswith(("trade_ledgers/", "equity_curves/")) and name not in {
            "trade_ledgers/random_exposure.json",
            "trade_ledgers/random_exposure.jsonl.gz",
            "equity_curves/random_exposure.jsonl.gz",
        }:
            if type(_json(files[name])) is not list:
                raise EvaluationIntegrityError("deterministic path is not an ordered JSON array")
    if type(_json(files["trade_ledgers/random_exposure.json"])) is not dict:
        raise EvaluationIntegrityError("random-exposure evidence is invalid")
    return EvaluationArtifact(
        manifest=manifest,
        files=files,
    )


def _verify_development_files(descriptor: int, expected: Mapping[str, bytes]) -> None:
    """Reconcile the complete frozen source bundle through its pinned directory."""
    if set(expected) != set(DEVELOPMENT_FILE_MAP) or any(
        type(raw) is not bytes for raw in expected.values()
    ):
        raise EvaluationIntegrityError("frozen Development file inventory is incomplete")
    for name, retained in expected.items():
        raw, _ = _read_regular_file_at_once(
            descriptor, name, description=f"frozen Development artifact {name}"
        )
        if sha256_bytes(raw) != sha256_bytes(retained) or raw != retained:
            raise EvaluationIntegrityError(f"frozen Development artifact changed: {name}")


def _verify_source_claim(development_run_dir: Path, artifact: EvaluationArtifact) -> None:
    """Bind the publication to its authoritative consumed development claim."""
    if not isinstance(development_run_dir, Path) or not development_run_dir.is_absolute():
        raise EvaluationInputError("development run must be an absolute Path")
    try:
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        resolved = development_run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvaluationIntegrityError("claimed development run is unavailable") from exc
    claim = _json(artifact.files[holdout.CLAIM_NAME])
    if (
        resolved == temp_root
        or not resolved.is_relative_to(temp_root)
        or resolved.name != claim["run_id"]
    ):
        raise EvaluationInputError("development run does not match the retained claim")
    descriptor = _open_directory_path(resolved, description="claimed synthetic development run")
    try:
        identity = _stat_identity(os.fstat(descriptor))
        for _ in range(2):
            _verify_development_files(
                descriptor,
                {
                    source: artifact.files[retained]
                    for source, retained in DEVELOPMENT_FILE_MAP.items()
                },
            )
            raw, _ = _read_regular_file_at_once(
                descriptor, holdout.CLAIM_NAME, description="consumed development-run claim"
            )
            if raw != artifact.files[holdout.CLAIM_NAME]:
                raise EvaluationIntegrityError("development-run claim differs from retained claim")
            generation_sha256 = holdout._generation_claim_sha256(
                protocol_sha256=claim["protocol_sha256"],
                development_dataset_manifest_sha256=claim["development_dataset_manifest_sha256"],
                augmented_model_sha256=claim["augmented_model_sha256"],
                control_model_sha256=claim["control_model_sha256"],
                development_cutoff_iso=claim["development_cutoff_iso"],
            )
            registry_descriptor = _open_directory_path(
                holdout._generation_registry_path(),
                description="authoritative synthetic generation registry",
            )
            try:
                consumed, _ = _read_regular_file_at_once(
                    registry_descriptor,
                    f"{generation_sha256}.claim",
                    description="authoritative consumed generation claim",
                )
            finally:
                os.close(registry_descriptor)
            if consumed != holdout._generation_marker_bytes(raw, resolved, identity):
                raise EvaluationIntegrityError("development claim differs from generation registry")
            completion_raw, _ = _read_regular_file_at_once(
                descriptor,
                "holdout_evaluation_completed.json",
                description="completed synthetic evaluation binding",
            )
            completion = _json(completion_raw)
            if (
                type(completion) is not dict
                or set(completion)
                != {
                    "schema_version",
                    "synthetic",
                    "evaluation_run_id",
                    "claim_sha256",
                    "evaluation_manifest_sha256",
                    "completed_at_utc",
                }
                or completion["schema_version"] != "phase2-synthetic-holdout-completion-v1"
                or completion["synthetic"] is not True
                or completion["evaluation_run_id"] != artifact.manifest["run_id"]
                or completion["claim_sha256"] != sha256_bytes(raw)
                or completion["evaluation_manifest_sha256"]
                != sha256_bytes(canonicalize(artifact.manifest))
                or canonicalize(completion) != completion_raw
                or _utc(completion["completed_at_utc"]) < _utc(claim["claimed_at_utc"])
            ):
                raise EvaluationIntegrityError("completion record does not bind evaluation")
            attached = _open_directory_path(
                resolved,
                description="claimed synthetic development run",
                expected_identity=identity,
            )
            os.close(attached)
    finally:
        os.close(descriptor)


def _rollback_owned_directory(descriptor: int, payload_names: frozenset[str]) -> None:
    """Invalidate the open inode, never a potentially swapped public pathname.

    Moving the manifest *inside* the pinned directory is the fallback when its
    unlink fails. It removes the only completion marker without renaming or
    deleting a directory name that a concurrent actor may have replaced.
    """
    try:
        os.unlink(MANIFEST_NAME, dir_fd=descriptor)
    except FileNotFoundError:
        pass
    except OSError:
        retired_name = f".invalid-evaluation-manifest-{uuid.uuid4().hex}"
        try:
            _atomic_rename_directory_no_replace(descriptor, MANIFEST_NAME, retired_name)
        except FileNotFoundError:
            pass
        except (CryptoAIError, OSError):
            # An owned directory can have lost write permission during the
            # failed verification. Restore it through the already-pinned fd,
            # then retry the descriptor-relative manifest unlink once.
            os.fchmod(descriptor, 0o700)
            try:
                os.unlink(MANIFEST_NAME, dir_fd=descriptor)
            except FileNotFoundError:
                pass
        else:
            try:
                os.unlink(retired_name, dir_fd=descriptor)
            except OSError:
                # A retired file is never a public evaluation completion.
                pass
    try:
        os.stat(MANIFEST_NAME, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise EvaluationIntegrityError("owned evaluation manifest survived rollback")
    # Remove only files that this publication attempt created, through the
    # already-open owned inode. A public pathname may have been replaced, so
    # rollback must never follow that pathname or erase its current occupant.
    for name in sorted(payload_names):
        parts = PurePosixPath(name).parts
        parent = os.dup(descriptor)
        try:
            try:
                for component in parts[:-1]:
                    child = _open_directory_at(
                        parent, component, description="owned evaluation rollback directory"
                    )
                    os.close(parent)
                    parent = child
                os.unlink(parts[-1], dir_fd=parent)
            except FileNotFoundError:
                pass
        finally:
            os.close(parent)
    directories = {
        PurePosixPath(name).parts[:index]
        for name in payload_names
        for index in range(1, len(PurePosixPath(name).parts))
    }
    for parts in sorted(directories, key=lambda item: (-len(item), item)):
        parent = os.dup(descriptor)
        try:
            try:
                for component in parts[:-1]:
                    child = _open_directory_at(
                        parent, component, description="owned evaluation rollback directory"
                    )
                    os.close(parent)
                    parent = child
                os.rmdir(parts[-1], dir_fd=parent)
            except (FileNotFoundError, OSError):
                # A concurrently injected entry does not make the already
                # invalidated publication visible as a complete evaluation.
                pass
        finally:
            os.close(parent)
    _fsync_directory_descriptor(descriptor, description="invalid evaluation publication")


class EvaluationStore:
    """No-overwrite artifact store confined to a synthetic temporary root."""

    def __init__(self, root: Path):
        if not isinstance(root, Path) or not root.is_absolute():
            raise EvaluationInputError("evaluation root must be an absolute Path")
        with _errors():
            temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            normalized = Path(os.path.abspath(os.fspath(root)))
            resolved = normalized.resolve(strict=False)
            if resolved == temp_root or not resolved.is_relative_to(temp_root):
                raise EvaluationInputError("evaluations are limited to temporary synthetic roots")
            # macOS exposes its temporary directory through /var -> /private/var.
            # Traverse the resolved target itself so the descriptor walk never
            # follows that alias (or any other symlink component).
            self.root = resolved
            descriptor = _create_directory_path_without_symlinks(
                self.root, description="synthetic evaluation root"
            )
            try:
                self._identity = _stat_identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
            # Publication ownership is intentionally scoped to this instance.
            # A completion failure may invalidate only the inode just created by
            # this instance, never a concurrently substituted public pathname.
            self._published_owners: dict[str, tuple[tuple[int, int], bytes, frozenset[str]]] = {}

    def _check_attachment(self, run_id: str, identity: tuple[int, int]) -> None:
        parent = _open_directory_path(
            self.root,
            description="synthetic evaluation root",
            expected_identity=self._identity,
        )
        try:
            child = _open_directory_at(parent, run_id, description="published evaluation")
            try:
                if _stat_identity(os.fstat(child)) != identity:
                    raise EvaluationIntegrityError("evaluation directory was replaced")
                if (
                    _stat_identity(os.stat(run_id, dir_fd=parent, follow_symlinks=False))
                    != identity
                ):
                    raise EvaluationIntegrityError("evaluation directory was detached")
                current = _open_directory_path(
                    self.root,
                    description="synthetic evaluation root",
                    expected_identity=self._identity,
                )
                os.close(current)
            finally:
                os.close(child)
        finally:
            os.close(parent)

    def get(self, run_id: str, *, development_run_dir: Path | None = None) -> EvaluationArtifact:
        """Read verified evidence only with its consumed development source."""
        with _errors():
            _run_id(run_id)
            if development_run_dir is None:
                raise EvaluationInputError("authoritative development run is required")
            parent = _open_directory_path(
                self.root,
                description="synthetic evaluation root",
                expected_identity=self._identity,
            )
            try:
                descriptor = _open_directory_at(parent, run_id, description="published evaluation")
                try:
                    artifact = _capture(descriptor, run_id)
                    identity = _stat_identity(os.fstat(descriptor))
                    self._check_attachment(run_id, identity)
                    _verify_source_claim(development_run_dir, artifact)
                    if (
                        _capture(
                            descriptor,
                            run_id,
                            replay_random_paths=False,
                            replay_deterministic_paths=False,
                        )
                        != artifact
                    ):
                        raise EvaluationIntegrityError("evaluation changed during verification")
                    self._check_attachment(run_id, identity)
                    _verify_source_claim(development_run_dir, artifact)
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)
            return artifact

    def unpublish_owned(self, run_id: str, artifact: EvaluationArtifact) -> None:
        """Invalidate this store instance's publication after completion failure.

        The public name is used only to locate a directory.  Every mutation is
        performed through a descriptor whose inode must match the publication
        recorded at the successful no-replace rename.  If that name has been
        replaced, an unrelated directory is left untouched.
        """
        with _errors():
            _run_id(run_id)
            owner = self._published_owners.get(run_id)
            if owner is None or type(artifact) is not EvaluationArtifact:
                raise EvaluationIntegrityError("no owned evaluation publication to invalidate")
            identity, manifest_bytes, payload_names = owner
            if canonicalize(artifact.manifest) != manifest_bytes:
                raise EvaluationIntegrityError("rollback artifact differs from owned publication")
            parent = _open_directory_path(
                self.root,
                description="synthetic evaluation root",
                expected_identity=self._identity,
            )
            try:
                descriptor = _open_directory_at(
                    parent, run_id, description="owned published evaluation"
                )
                try:
                    if _stat_identity(os.fstat(descriptor)) != identity:
                        raise EvaluationIntegrityError("evaluation directory was replaced")
                    _rollback_owned_directory(descriptor, payload_names)
                    self._published_owners.pop(run_id, None)
                    _fsync_directory_descriptor(parent, description="synthetic evaluation root")
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)

    def publish(
        self,
        run_id: str,
        files: dict[str, bytes],
        metadata: dict,
    ) -> EvaluationArtifact:
        """Publish a complete bundle atomically, then return verified exact bytes."""
        with _errors():
            _run_id(run_id)
            checked_files = _files(files)
            checked_metadata, metadata_bytes = _metadata(metadata)
            _require_descriptor_relative_mutations()
            parent = _open_directory_path(
                self.root,
                description="synthetic evaluation root",
                expected_identity=self._identity,
            )
            stage_name = f".staging-{run_id}-{uuid.uuid4().hex}"
            stage: int | None = None
            try:
                _require_atomic_rename_directory_no_replace_at(parent)
                os.mkdir(stage_name, mode=0o700, dir_fd=parent)
                stage = _open_directory_at(parent, stage_name, description="evaluation staging")
                for name, raw in sorted(checked_files.items()):
                    parts = PurePosixPath(name).parts
                    file_parent = _ensure_directory_chain_at(
                        stage,
                        parts[:-1],
                        description=f"parent directory for {name}",
                    )
                    try:
                        _write_fsynced_at(file_parent, parts[-1], raw)
                        readback, _ = _read_regular_file_at_once(
                            file_parent, parts[-1], description=name
                        )
                    finally:
                        os.close(file_parent)
                    if readback != raw or sha256_bytes(readback) != sha256_bytes(raw):
                        raise EvaluationIntegrityError("staged evaluation payload differs")
                _require_metadata_unchanged(metadata, metadata_bytes)
                manifest_bytes = canonicalize(
                    {
                        "schema_version": SCHEMA,
                        "run_id": run_id,
                        "metadata": checked_metadata,
                        "files": _inventory(checked_files),
                    }
                )
                # The manifest is the last file created; nested directories are synced below.
                _write_fsynced_at(stage, MANIFEST_NAME, manifest_bytes)
                staged = _capture(stage, run_id, expected_manifest_bytes=manifest_bytes)
                if staged.files != checked_files:
                    raise EvaluationIntegrityError("staged evaluation bytes changed")
                _fsync_tree_directories_at(stage, description="evaluation staging")
                if (
                    _capture(
                        stage,
                        run_id,
                        expected_manifest_bytes=manifest_bytes,
                        replay_random_paths=False,
                        replay_deterministic_paths=False,
                    )
                    != staged
                ):
                    raise EvaluationIntegrityError("staged evaluation changed during fsync")
                _require_metadata_unchanged(metadata, metadata_bytes)
                _atomic_rename_directory_no_replace(parent, stage_name, run_id)
                _fsync_directory_descriptor(parent, description="synthetic evaluation root")
                published = _capture(
                    stage,
                    run_id,
                    expected_manifest_bytes=manifest_bytes,
                    replay_random_paths=False,
                    replay_deterministic_paths=False,
                )
                if published != staged:
                    raise EvaluationIntegrityError("published evaluation changed after rename")
                self._check_attachment(run_id, _stat_identity(os.fstat(stage)))
                _require_metadata_unchanged(metadata, metadata_bytes)
                self._published_owners[run_id] = (
                    _stat_identity(os.fstat(stage)),
                    manifest_bytes,
                    frozenset(checked_files),
                )
            except Exception as exc:
                # A wrapper may raise after the no-replace rename succeeded. The open
                # descriptor still pins this attempt's inode, even if its name was swapped.
                if stage is not None:
                    try:
                        _rollback_owned_directory(stage, frozenset(checked_files))
                    except Exception as cleanup_exc:
                        raise EvaluationIntegrityError(
                            "failed to invalidate owned evaluation publication"
                        ) from cleanup_exc
                if isinstance(exc, EvaluationError):
                    raise
                if isinstance(exc, PublicationCollisionError):
                    raise EvaluationCollisionError("evaluation run already exists") from exc
                raise EvaluationIntegrityError("evaluation publication failed") from exc
            finally:
                if stage is not None:
                    os.close(stage)
                os.close(parent)
            return published
