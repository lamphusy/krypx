"""Synthetic-only four-cell engineering experiments, never a research/holdout runner.

Prepared datasets are transitively replayed before use. Results are deterministic
canonical buffers; publication and reads replay the synthetic experiment rather
than trusting a caller's recomputed hashes. No pickle, live loader, or tuning seam.
"""

from __future__ import annotations

import math
import os
import platform
import re
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.modeling.metrics import classification_metrics
from crypto_ai.modeling.train import _xgboost_classifier
from crypto_ai.phase2 import dataset as datasets
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.storage import (
    ContentAddressedStore,
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
    _validate_relative_path,
    _write_fsynced_at,
)

SCHEMA = "phase2-experiment-v1"
SPECIFICATION_ID = "phase2-milestone6-offline-four-cell-experiments-v1"
MANIFEST = "experiment_manifest.json"
FOLD_COUNT = 5
PURGE_ROWS = 5
RESEARCH_TEST_ROWS = 2098
THRESHOLD = 0.5
CELLS = ("A", "B", "C", "D")
LOGISTIC_PARAMS = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": 0.0,
    "max_iter": 1000,
    "n_jobs": None,
    "penalty": "deprecated",
    "random_state": 42,
    "solver": "lbfgs",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": False,
}
SCALER_PARAMS = {"copy": True, "with_mean": True, "with_std": True}
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
DEPENDENCIES = {
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
    "xgboost": "3.2.0",
}
PREDICTION_COLUMNS = (
    "market_ordinal",
    "decision_at",
    "entry_timestamp",
    "exit_timestamp",
    "fold_number",
    "actual_label",
    "probability_score",
    "predicted_label",
    "signal",
)


class ExperimentError(CryptoAIError):
    """A synthetic experiment contract failed."""


class ExperimentInputError(ExperimentError):
    """An input or split does not meet the frozen engineering contract."""


class ExperimentSplitError(ExperimentInputError):
    """A shared fold plan cannot satisfy expanding training history."""


class ExperimentIntegrityError(ExperimentError):
    """Experiment bytes, parents, or semantic replay disagree."""


class ExperimentAuthorizationError(ExperimentError):
    """Only generated synthetic prepared datasets may enter this engine."""


class ExperimentTrainingError(ExperimentError):
    """Fitting/prediction failed; no partial result is accepted."""


@contextmanager
def _errors():
    try:
        yield
    except ExperimentError:
        raise
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
    ) as exc:
        raise ExperimentIntegrityError("invalid synthetic experiment evidence") from exc


def _json(raw):
    return datasets._json(raw)


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _instant(value):
    return format_utc_timestamp(value.to_pydatetime())


def _test_size(value):
    if value is None:
        return RESEARCH_TEST_ROWS
    if type(value) is not int or not 1 <= value <= RESEARCH_TEST_ROWS:
        raise ExperimentInputError("explicit synthetic fixture_test_rows must be 1..2098")
    return value


def _cell_columns(cell):
    if type(cell) is not str or cell not in CELLS:
        raise ExperimentInputError("unknown experiment cell")
    return datasets.TECHNICAL_COLUMNS if cell in ("A", "B") else datasets.COMBINED_COLUMNS


def _configuration(test_rows):
    return {
        "specification_id": SPECIFICATION_ID,
        "scope": "synthetic_engineering_only",
        "fold_count": FOLD_COUNT,
        "purge_original_market_rows": PURGE_ROWS,
        "research_test_rows": RESEARCH_TEST_ROWS,
        "fixture_test_rows": test_rows,
        "effective_test_rows": _test_size(test_rows),
        "threshold": THRESHOLD,
        "cells": {
            cell: {
                "classifier": "logistic_regression" if cell in ("A", "C") else "xgboost",
                "feature_columns": list(_cell_columns(cell)),
            }
            for cell in CELLS
        },
        "logistic": LOGISTIC_PARAMS,
        "scaler": SCALER_PARAMS,
        "xgboost": XGBOOST_PARAMS,
        "single_class_training": "explicit_constant_class_probability_with_warning",
        "importance": "sum_weight_and_split_weighted_gain_cover_across_fitted_folds",
        "trading_backtests": False,
        "holdout_evaluation": False,
        "research_gate_evaluation": False,
    }


def shared_folds(frame, *, dataset_id, fixture_test_rows=None):
    """Materialize five shared folds; helper frames cannot enter public training.

    Validation blocks are consecutive in the retained index. Purges are NOT
    compressed: j-5..j-1 are forbidden original market ordinals, even if absent.
    """
    with _errors():
        size = _test_size(fixture_test_rows)
        needed = {"market_ordinal", "timestamp", "decision_at", "exit_timestamp", "label"}
        if (
            type(frame) is not pd.DataFrame
            or frame.empty
            or not _hash(dataset_id)
            or not frame.columns.is_unique
            or not needed <= set(frame.columns)
        ):
            raise ExperimentInputError("a nonempty ordered labeled frame is required")
        ordinals = frame.market_ordinal.to_numpy()
        if (
            str(frame.market_ordinal.dtype) != "int64"
            or (ordinals < 0).any()
            or (np.diff(ordinals) <= 0).any()
        ):
            raise ExperimentInputError("original market ordinals must be increasing int64")
        for name in ("timestamp", "decision_at", "exit_timestamp"):
            values = frame[name]
            if (
                str(values.dtype) != "datetime64[ns, UTC]"
                or values.isna().any()
                or not values.is_monotonic_increasing
                or not values.is_unique
            ):
                raise ExperimentInputError("fold timestamps must be increasing exact UTC")
        opened = frame.timestamp
        if (
            not ((frame.decision_at - opened) == pd.Timedelta(hours=1)).all()
            or not ((frame.exit_timestamp - opened) == pd.Timedelta(hours=5)).all()
            or not np.array_equal(
                (opened - opened.iloc[0]).to_numpy(),
                (ordinals - ordinals[0]) * np.timedelta64(1, "h"),
            )
        ):
            raise ExperimentInputError("original ordinal/time/label alignment mismatch")
        labels = frame.label.to_numpy()
        if str(frame.label.dtype) != "int8" or not np.isin(labels, [0, 1]).all():
            raise ExperimentInputError("binary int8 labels required")
        first = len(frame) - FOLD_COUNT * size
        if first <= 0:
            raise ExperimentInputError("insufficient synthetic rows; no automatic fold shrinking")
        rows = [
            {"market_ordinal": int(o), "decision_at": _instant(d), "label": int(y)}
            for o, d, y in zip(ordinals, frame.decision_at, labels, strict=True)
        ]
        folds = []
        previous_training_length = None
        for number in range(1, FOLD_COUNT + 1):
            start = first + (number - 1) * size
            validation = list(range(start, start + size))
            j = int(ordinals[start])
            training = np.flatnonzero(ordinals < j - PURGE_ROWS).tolist()
            if not training or training[-1] >= start:
                raise ExperimentInputError("purge leaves an empty or noncausal training fold")
            if previous_training_length is not None and len(training) <= previous_training_length:
                raise ExperimentSplitError("training history must strictly expand across folds")
            previous_training_length = len(training)
            if (frame.iloc[training].exit_timestamp >= opened.iloc[start]).any():
                raise ExperimentInputError("training label leaks into validation candle open")
            purge = list(range(j - PURGE_ROWS, j))
            folds.append(
                {
                    "fold_number": number,
                    "training_positions": training,
                    "validation_positions": validation,
                    "training_market_ordinals": [int(ordinals[i]) for i in training],
                    "validation_market_ordinals": [int(ordinals[i]) for i in validation],
                    "purge_market_ordinals": purge,
                    "purged_retained_positions": np.flatnonzero(np.isin(ordinals, purge)).tolist(),
                    "validation_start_open": _instant(opened.iloc[start]),
                    "validation_start_decision": _instant(frame.decision_at.iloc[start]),
                    "validation_end_decision": _instant(frame.decision_at.iloc[start + size - 1]),
                }
            )
        return {
            "schema_version": "phase2-shared-folds-v1",
            "synthetic": True,
            "dataset_id": dataset_id,
            "prepared_dataset_manifest_sha256": dataset_id,
            "row_identity_sha256": sha256_bytes(canonicalize(rows)),
            "fixture_test_rows": fixture_test_rows,
            "effective_test_rows": size,
            "purge_original_market_rows": PURGE_ROWS,
            "folds": folds,
        }


def evaluate_classification(actual, probability):
    """Strictly validate BEFORE invoking unchanged Phase 1 classification math."""
    with _errors():
        # Inspect sequence tokens before numpy can erase mixed bool/int distinctions.
        for values in (actual, probability):
            if type(values) in (list, tuple) and any(
                isinstance(value, (bool, np.bool_, str)) for value in values
            ):
                raise ExperimentInputError("strings and booleans are not metric numbers")
        actual = np.asarray(actual)
        probability = np.asarray(probability)
        if (
            actual.ndim != 1
            or probability.ndim != 1
            or not len(actual)
            or len(actual) != len(probability)
            or actual.dtype.kind not in "iu"
            or not np.isin(actual, [0, 1]).all()
            or probability.dtype.kind not in "fiu"
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ExperimentInputError(
                "strict binary labels and finite [0,1] probabilities required"
            )
        probability = probability.astype(np.float64)
        actual = actual.astype(np.int8)
        predicted = (probability >= THRESHOLD).astype(np.int8)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = classification_metrics(actual, probability, predicted)
        result["warnings"] = [str(item.message) for item in captured]
        canonicalize(result)
        return result


def global_feature_importance(models, feature_columns):
    """Aggregate fold boosters without a full-data refit; preserve frozen order."""
    with _errors():
        models = tuple(models)
        columns = tuple(feature_columns)
        if (
            not columns
            or any(type(n) is not str for n in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ExperimentInputError("unique ordered feature names required")
        weights = {n: [] for n in columns}
        gains = {n: [] for n in columns}
        covers = {n: [] for n in columns}
        for model in models:
            booster = model.get_booster()
            if tuple(booster.feature_names or ()) != columns:
                raise ExperimentIntegrityError("booster feature identity/order mismatch")
            values = {
                kind: booster.get_score(importance_type=kind)
                for kind in ("weight", "total_gain", "total_cover")
            }
            keys = set(values["weight"])
            if not keys <= set(columns) or any(set(v) != keys for v in values.values()):
                raise ExperimentIntegrityError("booster importance feature mismatch")
            for name in columns:
                weight, gain, cover = (
                    values[kind].get(name, 0.0) for kind in ("weight", "total_gain", "total_cover")
                )
                for value in (weight, gain, cover):
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ExperimentIntegrityError("invalid booster importance value")
                if weight != int(weight) or (weight == 0 and (gain != 0 or cover != 0)):
                    raise ExperimentIntegrityError("invalid booster importance split count")
                weights[name].append(float(weight))
                gains[name].append(float(gain))
                covers[name].append(float(cover))
        rows = []
        for name in columns:
            weight = math.fsum(weights[name])
            rows.append(
                {
                    "feature": name,
                    "weight": weight,
                    "gain": math.fsum(gains[name]) / weight if weight else 0.0,
                    "cover": math.fsum(covers[name]) / weight if weight else 0.0,
                }
            )
        result = {
            "applicable": True,
            "method": "split_weighted_across_folds",
            "fitted_fold_count": len(models),
            "warnings": [] if models else ["no fitted boosters: all training folds single-class"],
            "features": rows,
        }
        canonicalize(result)
        return result


def _make_model(cell):
    _cell_columns(cell)
    if cell in ("A", "C"):
        return Pipeline(
            [
                ("scaler", StandardScaler(**SCALER_PARAMS)),
                ("classifier", LogisticRegression(**LOGISTIC_PARAMS)),
            ]
        )
    return _xgboost_classifier(**XGBOOST_PARAMS)


def _predict(model, train, validation, labels):
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(train, labels)
            classes = np.asarray(model.classes_)
            if (
                classes.shape != (2,)
                or classes.dtype.kind not in "iu"
                or not np.array_equal(classes, [0, 1])
            ):
                raise ExperimentTrainingError("classifier must expose ordered binary classes [0,1]")
            raw = np.asarray(model.predict_proba(validation))
        if (
            raw.shape != (len(validation), 2)
            or raw.dtype.kind != "f"
            or not np.isfinite(raw).all()
            or ((raw < 0) | (raw > 1)).any()
            or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-7, rtol=0)
        ):
            raise ExperimentTrainingError("invalid binary probability matrix")
        return raw[:, 1].astype(np.float64), [str(w.message) for w in captured]
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentTrainingError("synthetic fold fit/predict failed") from exc


@dataclass(frozen=True, slots=True)
class SyntheticExperimentInput:
    dataset_id: str
    protocol_bytes: bytes
    fixture_test_rows: int | None = None


def _validate_request(request):
    if type(request) is not SyntheticExperimentInput:
        raise ExperimentAuthorizationError("exact synthetic experiment input required")
    if not _hash(request.dataset_id):
        raise ExperimentInputError("exact prepared dataset identity required")
    protocol = _json(request.protocol_bytes)
    if type(protocol) is not dict or protocol.get("synthetic") is not True:
        raise ExperimentAuthorizationError("explicit canonical synthetic fixture protocol required")
    _test_size(request.fixture_test_rows)
    if {name: version(name) for name in DEPENDENCIES} != DEPENDENCIES:
        raise ExperimentIntegrityError("installed experiment dependencies differ from frozen locks")


def _execution_context(base_commit=None):
    root = Path(__file__).resolve().parents[3]
    locks = {name: (root / name).read_bytes() for name in datasets._LOCK_HASHES}
    if {n: sha256_bytes(raw) for n, raw in locks.items()} != datasets._LOCK_HASHES:
        raise ExperimentIntegrityError("local dependency lock bytes changed")
    commit = (
        datasets._git("rev-parse", "HEAD").decode().strip() if base_commit is None else base_commit
    )
    if type(commit) is not str or not re.fullmatch("[0-9a-f]{40}", commit):
        raise ExperimentIntegrityError("exact local engine base commit required")
    if datasets._git("rev-parse", "--verify", commit + "^{commit}") != commit.encode() + b"\n":
        raise ExperimentIntegrityError("recorded engine base commit is not available locally")
    return {
        "source.py": Path(__file__).read_bytes(),
        **locks,
        "environment.json": canonicalize(
            {
                "engine_base_commit": commit,
                "engine_source_identity": "exact_source_bytes_not_a_clean_commit_claim",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": {name: version(name) for name in DEPENDENCIES},
            }
        ),
    }


def _build(prepared, request, context):
    frame = prepared.labeled
    if not np.isfinite(frame[list(datasets.COMBINED_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise ExperimentInputError("all frozen features must be finite")
    folds = shared_folds(
        frame, dataset_id=request.dataset_id, fixture_test_rows=request.fixture_test_rows
    )
    fold_bytes = canonicalize(folds)
    fold_hash = sha256_bytes(fold_bytes)
    files = {
        **context,
        "input.json": canonicalize(
            {"dataset_id": request.dataset_id, "fixture_test_rows": request.fixture_test_rows}
        ),
        "protocol.json": request.protocol_bytes,
        "config.json": canonicalize(_configuration(request.fixture_test_rows)),
        "prepared_dataset_manifest.json": dict(prepared.files)[datasets.MANIFEST],
        "folds.json": fold_bytes,
    }
    for cell in CELLS:
        columns = list(_cell_columns(cell))
        rows, metrics, traces, models = [], [], [], []
        for fold in folds["folds"]:
            train = frame.iloc[fold["training_positions"]]
            validation = frame.iloc[fold["validation_positions"]]
            classes = np.unique(train.label.to_numpy())
            trace = {
                "fold_number": fold["fold_number"],
                "folds_sha256": fold_hash,
                "training_market_ordinals": fold["training_market_ordinals"],
                "training_label_classes": classes.tolist(),
                "scaler": None,
                "importance": None,
            }
            if len(classes) == 1:
                probability = np.full(len(validation), float(classes[0]), dtype=np.float64)
                notices = ["single-class training: explicit constant class probability"]
                trace["predictor"] = "constant_class"
            else:
                model = _make_model(cell)
                probability, notices = _predict(
                    model, train[columns].copy(), validation[columns].copy(), train.label.copy()
                )
                trace["predictor"] = "fitted_frozen_classifier"
                if cell in ("A", "C"):
                    scaler = model.named_steps["scaler"]
                    trace["scaler"] = {
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                        "variance": scaler.var_.tolist(),
                        "n_samples_seen": int(scaler.n_samples_seen_),
                    }
                else:
                    models.append(model)
                    trace["importance"] = global_feature_importance((model,), columns)
            trace["warnings"] = notices
            traces.append(trace)
            metrics.append(
                {
                    "fold_number": fold["fold_number"],
                    "metrics": evaluate_classification(validation.label.to_numpy(), probability),
                }
            )
            for position, (_, row) in enumerate(validation.iterrows()):
                score = float(probability[position])
                predicted = int(score >= THRESHOLD)
                rows.append(
                    [
                        int(row.market_ordinal),
                        _instant(row.decision_at),
                        _instant(row.entry_timestamp),
                        _instant(row.exit_timestamp),
                        fold["fold_number"],
                        int(row.label),
                        score,
                        predicted,
                        predicted,
                    ]
                )
        importance = (
            global_feature_importance(models, columns)
            if cell in ("B", "D")
            else {"applicable": False, "method": "not_applicable_logistic", "features": []}
        )
        prefix = f"cells/{cell}/"
        files[prefix + "predictions.json"] = canonicalize(
            {
                "columns": list(PREDICTION_COLUMNS),
                "rows": rows,
                "folds_sha256": fold_hash,
                "dataset_id": request.dataset_id,
            }
        )
        files[prefix + "metrics.json"] = canonicalize(
            {
                "folds": metrics,
                "aggregate": evaluate_classification(
                    np.array([r[5] for r in rows], dtype=np.int8),
                    np.array([r[6] for r in rows], dtype=np.float64),
                ),
                "folds_sha256": fold_hash,
            }
        )
        files[prefix + "training.json"] = canonicalize(traces)
        files[prefix + "importance.json"] = canonicalize(importance)
    manifest = {
        "schema_version": SCHEMA,
        "specification_id": SPECIFICATION_ID,
        "synthetic": True,
        "scope": "classification_engineering_only_no_research_or_holdout",
        "dataset_id": request.dataset_id,
        "prepared_dataset_manifest_sha256": request.dataset_id,
        "folds_sha256": fold_hash,
        "config_sha256": sha256_bytes(files["config.json"]),
        "protocol_sha256": sha256_bytes(request.protocol_bytes),
        "engine_source_sha256": sha256_bytes(files["source.py"]),
        "environment_sha256": sha256_bytes(files["environment.json"]),
        "parent_price_context_sha256": prepared.manifest["market_price_context_sha256"],
        "parent_exclusions_sha256": prepared.manifest["exclusions_sha256"],
        "files": _inventory(files),
    }
    files[MANIFEST] = canonicalize(manifest)
    return ExperimentArtifact(tuple(sorted(files.items())))


def _inventory(files):
    return {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in sorted(files.items())
    }


_MANIFEST_FIELDS = {
    "schema_version",
    "specification_id",
    "synthetic",
    "scope",
    "dataset_id",
    "prepared_dataset_manifest_sha256",
    "folds_sha256",
    "config_sha256",
    "protocol_sha256",
    "engine_source_sha256",
    "environment_sha256",
    "parent_price_context_sha256",
    "parent_exclusions_sha256",
    "files",
}


def _candidate(artifact):
    if type(artifact) is not ExperimentArtifact or type(artifact.files) is not tuple:
        raise ExperimentIntegrityError("exact immutable experiment buffers required")
    files = dict(artifact.files)
    if len(files) != len(artifact.files) or any(type(raw) is not bytes for raw in files.values()):
        raise ExperimentIntegrityError("duplicate or non-byte experiment files")
    for name in files:
        _validate_relative_path(name)
    manifest = _json(files[MANIFEST])
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_FIELDS
        or manifest["schema_version"] != SCHEMA
        or manifest["specification_id"] != SPECIFICATION_ID
        or manifest["synthetic"] is not True
        or manifest["scope"] != "classification_engineering_only_no_research_or_holdout"
        or manifest["files"] != _inventory({n: b for n, b in files.items() if n != MANIFEST})
    ):
        raise ExperimentIntegrityError("invalid experiment manifest/inventory")
    for key, name in (
        ("prepared_dataset_manifest_sha256", "prepared_dataset_manifest.json"),
        ("folds_sha256", "folds.json"),
        ("config_sha256", "config.json"),
        ("protocol_sha256", "protocol.json"),
        ("engine_source_sha256", "source.py"),
        ("environment_sha256", "environment.json"),
    ):
        if not _hash(manifest[key]) or manifest[key] != sha256_bytes(files[name]):
            raise ExperimentIntegrityError("experiment transitive hash mismatch")
    if manifest["dataset_id"] != manifest["prepared_dataset_manifest_sha256"]:
        raise ExperimentIntegrityError("experiment parent identity mismatch")
    return files, manifest


@dataclass(frozen=True, slots=True)
class ExperimentArtifact:
    """Candidate buffers. Engine/store methods, not properties, certify semantics."""

    files: tuple[tuple[str, bytes], ...]

    @property
    def experiment_id(self):
        with _errors():
            return sha256_bytes(_candidate(self)[0][MANIFEST])

    @property
    def manifest(self):
        with _errors():
            return _candidate(self)[1]

    def predictions(self, cell):
        with _errors():
            _cell_columns(cell)
            value = _json(_candidate(self)[0][f"cells/{cell}/predictions.json"])
            if value["columns"] != list(PREDICTION_COLUMNS):
                raise ExperimentIntegrityError("prediction column order mismatch")
            frame = pd.DataFrame(value["rows"], columns=PREDICTION_COLUMNS)
            for name in PREDICTION_COLUMNS:
                dtype = (
                    "datetime64[ns, UTC]"
                    if name in ("decision_at", "entry_timestamp", "exit_timestamp")
                    else (
                        "float64"
                        if name == "probability_score"
                        else (
                            "int8"
                            if name in ("actual_label", "predicted_label", "signal")
                            else "int64"
                        )
                    )
                )
                frame[name] = frame[name].astype(dtype)
            return frame

    def metrics(self, cell):
        with _errors():
            _cell_columns(cell)
            return _json(_candidate(self)[0][f"cells/{cell}/metrics.json"])

    def importance(self, cell):
        with _errors():
            _cell_columns(cell)
            return _json(_candidate(self)[0][f"cells/{cell}/importance.json"])


class OfflineExperimentEngine:
    """No arbitrary frames/factories: only exact verified synthetic M5 publications."""

    def __init__(self, store: ContentAddressedStore):
        if type(store) is not ContentAddressedStore:
            raise ExperimentAuthorizationError("exact local parent content store required")
        self.store = store

    def run(self, request: SyntheticExperimentInput) -> ExperimentArtifact:
        with _errors():
            _validate_request(request)
            prepared = datasets.DatasetStore(self.store).get(request.dataset_id)
            if prepared is None:
                raise ExperimentInputError("verified prepared dataset not found")
            return _build(prepared, request, _execution_context())


def _verify(store, artifact):
    files, manifest = _candidate(artifact)
    value = _json(files["input.json"])
    if type(value) is not dict or set(value) != {"dataset_id", "fixture_test_rows"}:
        raise ExperimentIntegrityError("invalid experiment invocation")
    if value["dataset_id"] != manifest["dataset_id"]:
        raise ExperimentIntegrityError("invocation/manifest parent mismatch")
    request = SyntheticExperimentInput(
        value["dataset_id"], files["protocol.json"], value["fixture_test_rows"]
    )
    _validate_request(request)
    # Source/runtime/locks are checked before any fits. A source change needs a new run.
    environment = _json(files["environment.json"])
    context = _execution_context(environment["engine_base_commit"])
    if any(files.get(name) != raw for name, raw in context.items()):
        raise ExperimentIntegrityError("experiment source/runtime/lock context changed")
    prepared = datasets.DatasetStore(store).get(request.dataset_id)
    if prepared is None:
        raise ExperimentIntegrityError("verified prepared parent disappeared")
    expected = _build(prepared, request, context)
    if dict(expected.files) != files:
        raise ExperimentIntegrityError(
            "experiment fold/model/prediction/metric semantic replay mismatch"
        )


def generate_run_id():
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex


def _run_id(value):
    if type(value) is not str or not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z_[0-9a-f]{32}", value):
        raise ExperimentInputError("run ID must be UTC microseconds plus random 32-hex suffix")
    try:
        datetime.strptime(value.split("_")[0], "%Y%m%dT%H%M%S%fZ")
    except ValueError as exc:
        raise ExperimentInputError("invalid UTC run identifier") from exc


def _directories(files):
    return {
        PurePosixPath(*PurePosixPath(name).parts[:i]).as_posix()
        for name in files
        for i in range(1, len(PurePosixPath(name).parts))
    }


def _capture(descriptor, run_id):
    raw, info = _read_regular_file_at_once(descriptor, "manifest.json", description="run manifest")
    manifest = _json(raw)
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "run_id", "metadata", "files"}
        or manifest["schema_version"] != "phase2-experiment-publication-v1"
        or manifest["run_id"] != run_id
        or type(manifest["metadata"]) is not dict
        or set(manifest["metadata"]) != {"experiment_id", "schema_version"}
        or manifest["metadata"]["schema_version"] != SCHEMA
        or not _hash(manifest["metadata"]["experiment_id"])
        or type(manifest["files"]) is not dict
        or not manifest["files"]
    ):
        raise ExperimentIntegrityError("invalid run metadata before payload capture")
    for name, entry in manifest["files"].items():
        _validate_relative_path(name)
        if (
            name == "manifest.json"
            or type(entry) is not dict
            or set(entry) != {"sha256", "size_bytes"}
            or not _hash(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise ExperimentIntegrityError("invalid run payload inventory")
    captured, _ = _capture_publication_tree(
        descriptor,
        manifest_data=raw,
        manifest_stat=info,
        publication_id=run_id,
        manifest_files=manifest["files"],
        expected_paths=set(manifest["files"]) | {"manifest.json"},
        expected_directories=_directories(manifest["files"]),
    )
    artifact = ExperimentArtifact(tuple(sorted((n, captured[n]) for n in manifest["files"])))
    if artifact.experiment_id != manifest["metadata"]["experiment_id"]:
        raise ExperimentIntegrityError("run completion metadata identity mismatch")
    return artifact


def _rollback_owned_directory(descriptor):
    """Invalidate only the directory held open by this publication attempt.

    A public name can be swapped after an identity check.  Never rename or remove
    that name during rollback: on platforms without inode-conditional rename, it
    might now identify someone else's directory.  The manifest is removed first
    so an interrupted rollback cannot leave a valid-looking completed run.  Do
    not recurse into child names: those could also be swapped with unrelated
    directories after an identity check.  Non-publication payload residue may
    remain in the invalidated directory until a separately authorized cleanup.
    """
    try:
        os.unlink("manifest.json", dir_fd=descriptor)
    except FileNotFoundError:
        pass
    _fsync_directory_descriptor(descriptor, description="invalid experiment run")


class ExperimentStore:
    """Pinned runs directory; hidden staging, manifest-last, atomic no-replace."""

    def __init__(self, root: Path, parents: ContentAddressedStore):
        if type(parents) is not ContentAddressedStore:
            raise ExperimentAuthorizationError("exact verified parent store required")
        with _errors():
            self.parents = parents
            self.root = Path(os.path.abspath(os.fspath(root)))
            descriptor = _create_directory_path_without_symlinks(
                self.root, description="experiment runs"
            )
            try:
                self._identity = _stat_identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)

    def _check_attachment(self, run_id, expected_identity, expected_files=None):
        """A pinned fd must still name the requested run, not a detached old root."""
        parent = _open_directory_path(
            self.root, description="experiment runs", expected_identity=self._identity
        )
        try:
            child = _open_directory_at(parent, run_id, description="published experiment")
            try:
                if _stat_identity(os.fstat(child)) != expected_identity:
                    raise ExperimentIntegrityError("experiment directory was replaced")
                if (
                    expected_files is not None
                    and dict(_capture(child, run_id).files) != expected_files
                ):
                    raise ExperimentIntegrityError("published experiment payload changed")
                if (
                    _stat_identity(os.stat(run_id, dir_fd=parent, follow_symlinks=False))
                    != expected_identity
                ):
                    raise ExperimentIntegrityError("published experiment was detached")
                current = _open_directory_path(
                    self.root, description="experiment runs", expected_identity=self._identity
                )
                os.close(current)
            finally:
                os.close(child)
        finally:
            os.close(parent)

    def get(self, run_id: str) -> ExperimentArtifact:
        with _errors():
            _run_id(run_id)
            parent = _open_directory_path(
                self.root, description="experiment runs", expected_identity=self._identity
            )
            try:
                descriptor = _open_directory_at(parent, run_id, description="experiment run")
                try:
                    artifact = _capture(descriptor, run_id)
                    self._check_attachment(run_id, _stat_identity(os.fstat(descriptor)))
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)
            _verify(self.parents, artifact)
            return artifact

    def publish(self, artifact: ExperimentArtifact, run_id: str | None = None) -> str:
        with _errors():
            if run_id is None:
                run_id = generate_run_id()
            _run_id(run_id)
            _verify(self.parents, artifact)
            files, _ = _candidate(artifact)
            _require_descriptor_relative_mutations()
            _require_atomic_rename_directory_no_replace_at()
            parent = _open_directory_path(
                self.root, description="experiment runs", expected_identity=self._identity
            )
            name = ".staging-" + run_id + "-" + uuid.uuid4().hex
            stage = None
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent)
                stage = _open_directory_at(parent, name, description="experiment staging")
                for filename, raw in sorted(files.items()):
                    parts = PurePosixPath(filename).parts
                    directory = _ensure_directory_chain_at(
                        stage, parts[:-1], description="experiment payload"
                    )
                    try:
                        _write_fsynced_at(directory, parts[-1], raw)
                        captured, _ = _read_regular_file_at_once(
                            directory, parts[-1], description=filename
                        )
                        if captured != raw or sha256_bytes(captured) != sha256_bytes(raw):
                            raise ExperimentIntegrityError(
                                "staged payload readback mismatch before manifest"
                            )
                    finally:
                        os.close(directory)
                completion = canonicalize(
                    {
                        "schema_version": "phase2-experiment-publication-v1",
                        "run_id": run_id,
                        "metadata": {
                            "schema_version": SCHEMA,
                            "experiment_id": artifact.experiment_id,
                        },
                        "files": _inventory(files),
                    }
                )
                _write_fsynced_at(stage, "manifest.json", completion)
                if dict(_capture(stage, run_id).files) != files:
                    raise ExperimentIntegrityError("staging bytes changed before publication")
                _fsync_tree_directories_at(stage, description="experiment staging")
                if dict(_capture(stage, run_id).files) != files:
                    raise ExperimentIntegrityError("staging changed during directory fsync")
                _atomic_rename_directory_no_replace(parent, name, run_id)
                _fsync_directory_descriptor(parent, description="experiment runs")
                self._check_attachment(run_id, _stat_identity(os.fstat(stage)), files)
            except Exception as exc:
                # A publish rename can succeed even if its helper raises.  The
                # still-open stage fd pins our own inode in either location.
                # Rollback must not mutate a potentially swapped public name.
                if stage is not None:
                    try:
                        _rollback_owned_directory(stage)
                    except Exception as cleanup_exc:
                        raise ExperimentIntegrityError(
                            "failed to invalidate owned experiment publication"
                        ) from cleanup_exc
                if isinstance(exc, ExperimentIntegrityError):
                    raise
                raise ExperimentIntegrityError("experiment publication failed") from exc
            finally:
                if stage is not None:
                    os.close(stage)
                # Do not remove the staging name here.  A concurrent rename can
                # replace it with an unrelated directory after any identity
                # check; the exception path invalidated only our pinned inode.
                os.close(parent)
            return run_id
