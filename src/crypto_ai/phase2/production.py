"""Offline synthetic production refits, never live training or model activation.

Every fit starts from a source-bound, completed Milestone 9 publication. Reads
replay provenance and the fitted model rather than trusting rehashed metadata.
Only explicit synthetic authorizations and temporary fixture roots are admitted.
"""

from __future__ import annotations

import math
import os
import platform
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np

from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import dataset, evaluation, evaluation_store, experiments, holdout
from crypto_ai.phase2.production_store import (
    ProductionArtifact,
    ProductionCollisionError,
    ProductionError,
    ProductionInputError,
    ProductionIntegrityError,
    ProductionStore,
    _version,
)
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.storage import (
    _capture_publication_tree,
    _open_directory_path,
    _read_regular_file_at_once,
    _stat_identity,
    _stat_tree_fingerprint,
)

SPECIFICATION_ID = "phase2-milestone10-production-decision-registry-v1"
AUTHORIZATION_SCHEMA = "phase2-synthetic-production-authorization-v1"
ROWS_SCHEMA = "phase2-synthetic-production-training-rows-v1"
DATASET_SCHEMA = "phase2-synthetic-production-prepared-dataset-v1"
MODEL_SCHEMA = "phase2-synthetic-production-model-v1"
SCOPE = "offline_synthetic_only"
PAYLOAD_NAMES = frozenset(
    {
        "model.json",
        "feature_columns.json",
        "prepared_dataset_manifest.json",
        "training_rows.json",
        "authorization.json",
        "execution_context.json",
    }
)


class ProductionAuthorizationError(ProductionError):
    """No explicit, matching synthetic-only authority or passing evidence exists."""


@contextmanager
def _errors():
    try:
        yield
    except ProductionError:
        raise
    except (
        CryptoAIError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ImportError,
        OverflowError,
        IndexError,
        RuntimeError,
        RecursionError,
    ) as exc:
        raise ProductionIntegrityError("synthetic production provenance failed closed") from exc


def _instant(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo != UTC:
        raise ProductionInputError("an explicit timezone-aware UTC datetime is required")
    return value


def _utc(value: object) -> datetime:
    parsed = parse_utc_timestamp(value, field="production timestamp")
    assert parsed is not None
    return parsed


def _temporary_path(value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ProductionInputError("synthetic paths must be absolute Paths")
    root_alias = Path(tempfile.gettempdir())
    root = root_alias.resolve(strict=True)
    normalized = Path(os.path.abspath(value))
    # Resolve only the OS temporary-root alias, never user-controlled descendants.
    if normalized.is_relative_to(root_alias):
        normalized = root / normalized.relative_to(root_alias)
    if normalized == root or not normalized.is_relative_to(root):
        raise ProductionInputError("only isolated temporary synthetic directories are permitted")
    return normalized


@dataclass(frozen=True, slots=True)
class SyntheticProductionAuthorization:
    evaluation_run_id: str
    evaluation_manifest_sha256: str
    model_version: str
    authorized_at_utc: datetime
    authorization_id: str
    approved: bool = True
    scope: str = SCOPE
    synthetic: bool = True


@dataclass(frozen=True, slots=True)
class SyntheticProductionRequest:
    evaluation_root: Path
    development_run_dir: Path
    versions_root: Path
    evaluation_run_id: str
    model_version: str
    authorization: SyntheticProductionAuthorization
    training_as_of_utc: datetime
    created_at_utc: datetime


def _authorization(value: object) -> dict:
    if type(value) is not SyntheticProductionAuthorization:
        raise ProductionAuthorizationError("explicit synthetic production authorization required")
    if value.approved is not True or value.synthetic is not True or value.scope != SCOPE:
        raise ProductionAuthorizationError("only approved offline synthetic fitting is permitted")
    _version(value.evaluation_run_id)
    _version(value.model_version)
    _version(value.authorization_id)
    if not evaluation._sha256(value.evaluation_manifest_sha256):
        raise ProductionAuthorizationError("authorization must bind an exact evaluation hash")
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "synthetic": True,
        "scope": SCOPE,
        "approved": True,
        "authorization_id": value.authorization_id,
        "evaluation_run_id": value.evaluation_run_id,
        "evaluation_manifest_sha256": value.evaluation_manifest_sha256,
        "model_version": value.model_version,
        "authorized_at_utc": format_utc_timestamp(_instant(value.authorized_at_utc)),
    }


def _parse_authorization(raw: bytes) -> SyntheticProductionAuthorization:
    parsed = dataset._json(raw)
    if type(parsed) is not dict or parsed.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ProductionAuthorizationError("unknown synthetic authorization schema")
    fields = {key: item for key, item in parsed.items() if key != "schema_version"}
    fields["authorized_at_utc"] = _utc(fields["authorized_at_utc"])
    result = SyntheticProductionAuthorization(**fields)
    if canonicalize(_authorization(result)) != raw:
        raise ProductionAuthorizationError("synthetic authorization has unexpected fields")
    return result


def load_synthetic_authorization(path: Path) -> SyntheticProductionAuthorization:
    """Read one bounded regular temporary fixture file without following links."""
    with _errors():
        path = _temporary_path(path)
        parent = _open_directory_path(path.parent, description="synthetic authorization parent")
        descriptor = None
        try:
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ProductionAuthorizationError("authorization must be one regular file")
            if before.st_size > 16_384:
                raise ProductionAuthorizationError("synthetic authorization exceeds its bound")
            chunks = []
            size = 0
            while chunk := os.read(descriptor, 16_385 - size):
                chunks.append(chunk)
                size += len(chunk)
                if size > 16_384:
                    raise ProductionAuthorizationError("synthetic authorization exceeds its bound")
            raw = b"".join(chunks)
            if (
                len(raw) != before.st_size
                or _stat_tree_fingerprint(os.fstat(descriptor)) != _stat_tree_fingerprint(before)
                or _stat_tree_fingerprint(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
                != _stat_tree_fingerprint(before)
            ):
                raise ProductionIntegrityError("authorization file mutated while being captured")
            return _parse_authorization(raw)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)


class _Source:
    """Reopen only the same verified evaluation and Development directory inodes."""

    def __init__(self, root: Path, development: Path, run_id: str):
        self.root = _temporary_path(root)
        self.development = _temporary_path(development)
        self.run_id = _version(run_id)
        self._captured: tuple[bytes, tuple[tuple[str, bytes], ...]] | None = None
        self.identities = {}
        for path in (self.root, self.root / run_id, self.development):
            descriptor = _open_directory_path(path, description="synthetic production source")
            try:
                self.identities[path] = _stat_identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)

    def attached(self) -> None:
        for path, identity in self.identities.items():
            descriptor = _open_directory_path(
                path, description="pinned production source", expected_identity=identity
            )
            os.close(descriptor)

    def capture(self) -> tuple[evaluation_store.EvaluationArtifact, bytes]:
        self.attached()
        if self._captured is None:
            result = evaluation_store.EvaluationStore(self.root).get(
                self.run_id, development_run_dir=self.development
            )
        else:
            # Semantic replay was already performed for this exact captured
            # bundle. Re-read every byte and source binding, never trust mtimes
            # or reuse a verification result for different content/inodes.
            manifest_bytes, payloads = self._captured
            result = evaluation_store.EvaluationArtifact(
                manifest=dataset._json(manifest_bytes), files=dict(payloads)
            )
            path = self.root / self.run_id
            descriptor = _open_directory_path(
                path, description="pinned evaluation", expected_identity=self.identities[path]
            )
            try:
                raw, info = _read_regular_file_at_once(
                    descriptor,
                    evaluation_store.MANIFEST_NAME,
                    description="verified production parent manifest",
                )
                if raw != canonicalize(result.manifest):
                    raise ProductionIntegrityError("production parent manifest changed")
                inventory = result.manifest["files"]
                captured, _ = _capture_publication_tree(
                    descriptor,
                    manifest_data=raw,
                    manifest_stat=info,
                    publication_id=self.run_id,
                    manifest_files=inventory,
                    expected_paths=set(inventory) | {evaluation_store.MANIFEST_NAME},
                    expected_directories=evaluation_store._directories(inventory),
                )
                if (
                    captured[evaluation_store.MANIFEST_NAME] != raw
                    or {name: captured[name] for name in inventory} != result.files
                ):
                    raise ProductionIntegrityError("production evaluation payloads changed")
            finally:
                os.close(descriptor)
            evaluation_store._verify_source_claim(self.development, result)
        self.attached()
        descriptor = _open_directory_path(
            self.development,
            description="completed evaluation source",
            expected_identity=self.identities[self.development],
        )
        try:
            completion, _ = _read_regular_file_at_once(
                descriptor,
                "holdout_evaluation_completed.json",
                description="production evaluation completion",
            )
        finally:
            os.close(descriptor)
        parsed = dataset._json(completion)
        if (
            parsed["evaluation_manifest_sha256"] != canonical_sha256(result.manifest)
            or parsed["evaluation_run_id"] != self.run_id
            or parsed["claim_sha256"] != sha256_bytes(result.files[holdout.CLAIM_NAME])
        ):
            raise ProductionIntegrityError("evaluation completion changed during capture")
        self.attached()
        if self._captured is None:
            self._captured = (canonicalize(result.manifest), tuple(sorted(result.files.items())))
        return result, completion


def _passing_source(source, completion: bytes, authorization: dict, created: datetime) -> None:
    if (
        source.manifest["run_id"] != authorization["evaluation_run_id"]
        or canonical_sha256(source.manifest) != authorization["evaluation_manifest_sha256"]
        or source.manifest["metadata"]["synthetic"] is not True
    ):
        raise ProductionAuthorizationError("authorization is detached from verified evaluation")
    metrics = dataset._json(source.files["metrics.json"])
    if (
        metrics["research_verdict"] != "PASS"
        or metrics["final_gates"]["research_verdict"] != "PASS"
        or metrics["production_decision"] != "NO-GO"
    ):
        raise ProductionAuthorizationError("a verified research PASS is required even for fixtures")
    if (
        not _utc(dataset._json(completion)["completed_at_utc"])
        <= _utc(authorization["authorized_at_utc"])
        <= created
    ):
        raise ProductionAuthorizationError("authorization must follow completed evidence")


def _training_rows(source, as_of: datetime) -> dict:
    files = source.files
    claim = dataset._json(files[holdout.CLAIM_NAME])
    development = dataset._json(files["evaluation_models/development_rows.json"])
    if development["feature_columns"] != list(dataset.COMBINED_COLUMNS):
        raise ProductionIntegrityError("Development feature order changed")
    market_raw = files["input_market_snapshot.csv"]
    development_raw = files["evaluation_models/development_market.csv"]
    if not market_raw.startswith(development_raw):
        raise ProductionIntegrityError("Development and holdout market histories conflict")
    market = evaluation._market_frame(market_raw, None)
    feature_rows = evaluation._csv_rows(
        files["input_feature_snapshot.csv"],
        ("market_ordinal", "decision_at") + dataset.COMBINED_COLUMNS,
    )
    threshold = minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0)
    rows = list(development["rows"])
    for row in feature_rows:
        ordinal = int(row[0])
        exit_ordinal = ordinal + holdout.HORIZON + 1
        if ordinal < claim["first_holdout_ordinal"] or exit_ordinal >= len(market):
            raise ProductionIntegrityError("production row violates the retained holdout boundary")
        entry = float(market.iloc[ordinal + 1].open)
        exit_price = float(market.iloc[exit_ordinal].open)
        gross_return = exit_price / entry - 1.0
        if not math.isfinite(gross_return):
            raise ProductionIntegrityError("production label return is not finite binary64")
        rows.append(
            {
                "market_ordinal": ordinal,
                "decision_at": row[1],
                "exit_ordinal": exit_ordinal,
                "exit_at": format_utc_timestamp(
                    market.iloc[exit_ordinal].timestamp.to_pydatetime()
                ),
                "label": int(gross_return > threshold),
                "features": [float(item) for item in row[2:]],
            }
        )
    previous = -1
    for row in rows:
        ordinal = row["market_ordinal"]
        if (
            type(ordinal) is not int
            or ordinal <= previous
            or ordinal in claim["purge_ordinals"]
            or row["exit_ordinal"] != ordinal + holdout.HORIZON + 1
            or _utc(row["exit_at"]) > as_of
            or _utc(row["decision_at"])
            != market.iloc[ordinal].timestamp.to_pydatetime() + timedelta(hours=1)
            or type(row["label"]) is not int
            or row["label"] not in (0, 1)
            or len(row["features"]) != len(dataset.COMBINED_COLUMNS)
            or not np.isfinite(np.asarray(row["features"], dtype=np.float64)).all()
        ):
            raise ProductionIntegrityError("production fit row, label or availability is invalid")
        previous = ordinal
    if not rows or not feature_rows:
        raise ProductionIntegrityError("both Development and verified holdout rows are required")
    return {
        "schema_version": ROWS_SCHEMA,
        "synthetic": True,
        "feature_columns": list(dataset.COMBINED_COLUMNS),
        "training_as_of_utc": format_utc_timestamp(as_of),
        "development_row_count": len(development["rows"]),
        "holdout_row_count": len(feature_rows),
        "excluded_boundary_purge_ordinals": claim["purge_ordinals"],
        "rows": rows,
    }


def _context(base_commit: str | None = None) -> dict:
    context = experiments._execution_context(base_commit)
    environment = dataset._json(context["environment.json"])
    for name, required in experiments.DEPENDENCIES.items():
        if version(name) != required:
            raise ProductionIntegrityError("production dependency version differs from frozen fit")
    return {
        "schema_version": "phase2-synthetic-production-execution-v1",
        "engine_base_commit": environment["engine_base_commit"],
        "source_identity": "uncommitted_source_bytes_not_clean_commit_claim",
        "source_sha256": {
            name: sha256_bytes(Path(__file__).with_name(name).read_bytes())
            for name in ("production.py", "production_store.py", "evaluation.py", "experiments.py")
        },
        "python": platform.python_version(),
        "dependencies": environment["dependencies"],
        "dependency_lock_sha256": sha256_bytes(context["requirements-lock.txt"]),
        "phase2_dependency_lock_sha256": sha256_bytes(context["requirements-phase2.txt"]),
    }


def _model(rows: dict, cell: str) -> bytes:
    if cell not in ("C", "D"):
        raise ProductionIntegrityError("only the selected augmented model family can be refitted")
    fitted = dataset._json(
        evaluation._fitted_model_artifact(
            rows["rows"], cell=cell, rows_sha256=canonical_sha256(rows)
        )
    )
    fitted["schema_version"] = MODEL_SCHEMA
    fitted["purpose"] = "production_refit_never_historical_holdout_evidence"
    return canonicalize(fitted)


def _prepared(source, completion: bytes, rows: dict, context: dict) -> dict:
    retained = source.files
    return {
        "schema_version": DATASET_SCHEMA,
        "synthetic": True,
        "evaluation_run_id": source.manifest["run_id"],
        "evaluation_manifest_sha256": canonical_sha256(source.manifest),
        "evaluation_completion_sha256": sha256_bytes(completion),
        "development_dataset_manifest_sha256": sha256_bytes(
            retained["evaluation_models/development_dataset_manifest.json"]
        ),
        "market_snapshot_sha256": sha256_bytes(retained["input_market_snapshot.csv"]),
        "article_snapshot_sha256": sha256_bytes(retained["input_article_snapshot.jsonl"]),
        "score_snapshot_sha256": sha256_bytes(retained["input_score_snapshot.jsonl"]),
        "evaluation_feature_snapshot_sha256": sha256_bytes(retained["input_feature_snapshot.csv"]),
        "verified_news_parent_sha256": (
            sha256_bytes(retained["input_news_parent.json"])
            if "input_news_parent.json" in retained
            else None
        ),
        "aggregation_evidence_sha256": (
            sha256_bytes(retained["input_aggregation_evidence.json"])
            if "input_aggregation_evidence.json" in retained
            else None
        ),
        "labeled_dataset_sha256": canonical_sha256(rows),
        "feature_columns": list(dataset.COMBINED_COLUMNS),
        "label_columns": ["label"],
        "training_as_of_utc": rows["training_as_of_utc"],
        "training_row_count": len(rows["rows"]),
        "development_row_count": rows["development_row_count"],
        "holdout_row_count": rows["holdout_row_count"],
        "boundary_purge_ordinals": rows["excluded_boundary_purge_ordinals"],
        "provider_gap_policy": "retain_only_source_verified_feature_rows_never_reintroduce_gaps",
        "label_horizon": holdout.HORIZON,
        "minimum_required_return": minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0),
        "dependency_lock_sha256": context["dependency_lock_sha256"],
        "phase2_dependency_lock_sha256": context["phase2_dependency_lock_sha256"],
    }


def _candidate(source, completion, authorization, as_of, created, context):
    _passing_source(source, completion, authorization, created)
    rows = _training_rows(source, as_of)
    prepared = _prepared(source, completion, rows, context)
    cell = source.manifest["metadata"]["selected_augmented_cell"]
    model_raw = _model(rows, cell)
    model = dataset._json(model_raw)
    metadata = {
        "synthetic": True,
        "scope": SCOPE,
        "specification_id": SPECIFICATION_ID,
        "model_version": authorization["model_version"],
        "model_type": model["model_family"],
        "selected_augmented_cell": cell,
        "training_start": rows["rows"][0]["decision_at"],
        "training_end": rows["rows"][-1]["decision_at"],
        "training_row_count": len(rows["rows"]),
        "training_as_of_utc": format_utc_timestamp(as_of),
        "feature_columns": list(dataset.COMBINED_COLUMNS),
        "feature_schema_hash": canonical_sha256(list(dataset.COMBINED_COLUMNS)),
        "model_parameters": model["hyperparameters"],
        "preprocessing_parameters": model.get("scaler_parameters"),
        "evaluation_run_id_provenance": source.manifest["run_id"],
        "evaluation_manifest_sha256": canonical_sha256(source.manifest),
        "evaluation_completion_sha256": sha256_bytes(completion),
        "evaluation_model_sha256": source.manifest["metadata"]["augmented_model_sha256"],
        "protocol_sha256": source.manifest["metadata"]["protocol_sha256"],
        "prepared_dataset_manifest_sha256": canonical_sha256(prepared),
        "authorization_sha256": canonical_sha256(authorization),
        "authorization_id": authorization["authorization_id"],
        "created_at_utc": format_utc_timestamp(created),
        "code_commit": context["engine_base_commit"],
        "execution_context_sha256": canonical_sha256(context),
        "dependency_lock_sha256": context["dependency_lock_sha256"],
        "source_baseline_mode": source.manifest["metadata"]["baseline_mode"],
        "production_decision": "NO-GO",
        "activation_authorized": False,
        "model_purpose": "synthetic_production_refit_not_evaluation_model",
    }
    files = {
        "model.json": model_raw,
        "feature_columns.json": canonicalize(list(dataset.COMBINED_COLUMNS)),
        "prepared_dataset_manifest.json": canonicalize(prepared),
        "training_rows.json": canonicalize(rows),
        "authorization.json": canonicalize(authorization),
        "execution_context.json": canonicalize(context),
    }
    return files, metadata


class ProductionRegistry:
    """Mandatory source-bound registry read, including deterministic fit replay."""

    def __init__(self, versions_root: Path, evaluation_root: Path, development_run_dir: Path):
        with _errors():
            self.store = ProductionStore(_temporary_path(versions_root))
            self.evaluation_root = _temporary_path(evaluation_root)
            self.development_run_dir = _temporary_path(development_run_dir)

    def get(self, model_version: str) -> ProductionArtifact:
        with _errors():

            def verify(manifest, files):
                if set(files) != PAYLOAD_NAMES:
                    raise ProductionIntegrityError("production payload schema differs")
                authorization = _authorization(_parse_authorization(files["authorization.json"]))
                if authorization["model_version"] != model_version:
                    raise ProductionIntegrityError("authorization targets a different version")
                metadata = manifest["metadata"]
                source = _Source(
                    self.evaluation_root,
                    self.development_run_dir,
                    authorization["evaluation_run_id"],
                )
                evidence, completion = source.capture()
                context = dataset._json(files["execution_context.json"])
                if (
                    canonicalize(_context(context["engine_base_commit"]))
                    != files["execution_context.json"]
                ):
                    raise ProductionIntegrityError("production runtime/source provenance changed")
                as_of, created = _utc(metadata["training_as_of_utc"]), _utc(
                    metadata["created_at_utc"]
                )
                if as_of > created:
                    raise ProductionIntegrityError("production fit cutoff is in the future")
                expected_files, expected_metadata = _candidate(
                    evidence, completion, authorization, as_of, created, context
                )
                if files != expected_files or canonicalize(metadata) != canonicalize(
                    expected_metadata
                ):
                    raise ProductionIntegrityError(
                        "production bytes differ from verified fit replay"
                    )
                current, current_completion = source.capture()
                if current != evidence or current_completion != completion:
                    raise ProductionIntegrityError("production source mutated during replay")

            return self.store.get(model_version, verify=verify)


class OfflineProductionEngine:
    """Fit only a fresh synthetic production model; never activate or overwrite."""

    def train(self, request: SyntheticProductionRequest) -> ProductionArtifact:
        with _errors():
            if type(request) is not SyntheticProductionRequest:
                raise ProductionInputError(
                    "only an explicit synthetic production request is allowed"
                )
            authorization = _authorization(request.authorization)
            if (
                authorization["evaluation_run_id"] != request.evaluation_run_id
                or authorization["model_version"] != request.model_version
            ):
                raise ProductionAuthorizationError(
                    "authorization targets another evaluation/version"
                )
            as_of = _instant(request.training_as_of_utc)
            created = _instant(request.created_at_utc)
            if as_of > created:
                raise ProductionInputError("production training cutoff cannot follow creation")
            store = ProductionStore(_temporary_path(request.versions_root))
            store.assert_available(request.model_version)
            binding = _Source(
                request.evaluation_root, request.development_run_dir, request.evaluation_run_id
            )
            source, completion = binding.capture()
            context = _context()
            files, metadata = _candidate(source, completion, authorization, as_of, created, context)

            def verify(manifest, captured):
                if (
                    captured != files
                    or canonicalize(manifest["metadata"]) != canonicalize(metadata)
                    or manifest["model_version"] != authorization["model_version"]
                ):
                    raise ProductionIntegrityError("production candidate was changed")
                current, current_completion = binding.capture()
                if current != source or current_completion != completion:
                    raise ProductionIntegrityError(
                        "evaluation source changed during production fit"
                    )

            return store.publish(authorization["model_version"], files, metadata, verify=verify)


__all__ = [
    "OfflineProductionEngine",
    "ProductionRegistry",
    "ProductionArtifact",
    "ProductionError",
    "ProductionInputError",
    "ProductionIntegrityError",
    "ProductionAuthorizationError",
    "ProductionCollisionError",
    "SyntheticProductionAuthorization",
    "SyntheticProductionRequest",
    "load_synthetic_authorization",
]
