"""Synthetic-only experiment contracts, original-ordinal folds, and immutable runs."""

from __future__ import annotations

import gzip
import json
import os
import re
import socket
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.build import compute_features
from crypto_ai.phase2 import dataset as dataset_module
from crypto_ai.phase2 import experiments as experiment_module
from crypto_ai.phase2.dataset import (
    COMBINED_COLUMNS,
    TECHNICAL_COLUMNS,
    DatasetStore,
    OfflineDatasetBuilder,
    PreparedDatasetArtifact,
    SyntheticDatasetInput,
    SyntheticMarket,
)
from crypto_ai.phase2.experiments import (
    ExperimentArtifact,
    ExperimentError,
    ExperimentIntegrityError,
    ExperimentSplitError,
    ExperimentStore,
    OfflineExperimentEngine,
    SyntheticExperimentInput,
    evaluate_classification,
    generate_run_id,
    global_feature_importance,
    shared_folds,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.aggregation import (
    AggregationStore,
    OfflineFeatureAggregator,
    SyntheticAggregationInput,
)
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GSGAdapter,
    GSGNormalizer,
    RightsApproval,
    plan_retrieval,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

START = datetime(2026, 9, 1, tzinfo=UTC)
PROTOCOL = canonicalize({"synthetic": True, "fixture": "milestone-six"})
CELLS = ("A", "B", "C", "D")


def assert_only_unpublished_staging_tombstones(root: Path) -> None:
    for stage in root.glob(".staging-*"):
        assert stage.is_dir()
        assert not stage.is_symlink()
        assert not (stage / "manifest.json").exists()


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 6 tests are synthetic and entirely offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(scope="module")
def synthetic_git():
    """Exercise provenance validation with explicit synthetic read-only Git evidence."""
    commit = "1" * 40
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        if args in (("rev-parse", "--verify", commit + "^{commit}"), ("rev-parse", "HEAD")):
            return commit.encode() + b"\n"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if len(args) == 2 and args[0] == "show" and args[1].startswith(commit + ":"):
            name = args[1].split(":", 1)[1]
            if name in (
                "src/crypto_ai/phase2/dataset.py",
                "requirements-lock.txt",
                "requirements-phase2.txt",
            ):
                return (root / name).read_bytes()
        raise AssertionError(f"unexpected Git operation: {args!r}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dataset_module, "_git", git)
        yield


@dataclass(frozen=True)
class Corpus:
    store: ContentAddressedStore
    dataset: PreparedDatasetArtifact
    request: SyntheticExperimentInput


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory, synthetic_git) -> Corpus:
    """Publish genuine GSG -> M4 -> M5 parents; no prebuilt or real market inputs."""
    root = tmp_path_factory.mktemp("synthetic-m6-parents")
    store = ContentAddressedStore(root / "provider")
    market = SyntheticMarket(format_utc_timestamp(START), 70, seed=17)
    technical = compute_features(market.frame())
    decisions = tuple(
        format_utc_timestamp(value.to_pydatetime() + timedelta(hours=1))
        for value in technical.timestamp
    )
    first = datetime.fromisoformat(decisions[0].replace("Z", "+00:00"))
    last = datetime.fromisoformat(decisions[-1].replace("Z", "+00:00"))
    plan = plan_retrieval(
        format_utc_timestamp(first - timedelta(hours=24)),
        format_utc_timestamp(last + timedelta(minutes=1)),
    )
    clock = [first]
    adapter = GSGAdapter(store, clock=lambda: clock[0])
    raw = gzip.compress(b"", mtime=0)
    snapshots = []
    for interval in plan.intervals:
        source_at = datetime.fromisoformat(interval.filename_timestamp.replace("Z", "+00:00"))
        clock[0] = source_at + timedelta(minutes=30)
        snapshots.append(
            adapter.ingest_snapshot(
                raw,
                filename_timestamp=interval.filename_timestamp,
                ingested_at=format_utc_timestamp(clock[0]),
                source_locator=(
                    "https://data.gdeltproject.org/gdeltv3/gsg/"
                    f"{source_at.strftime('%Y%m%d%H%M%S')}.gsg.json.gz"
                ),
                collection_mode="prospective",
                input_class="synthetic_fixture",
            )
        )
    protocol_hash = sha256_bytes(PROTOCOL)
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=protocol_hash,
        raw_snapshot_sha256={sha256_bytes(raw)},
    )
    as_of = format_utc_timestamp(last + timedelta(hours=1))
    state = GSGNormalizer(protocol_config_sha256=protocol_hash, rights_approval=approval)
    state.normalize(snapshots, retrieval_plan=plan, terminal_as_of=as_of)
    state.publish_state(store, "milestone-six")
    aggregation = OfflineFeatureAggregator(store).aggregate(
        SyntheticAggregationInput(
            state_publication_id="gsg-normalizer-state-milestone-six",
            score_artifacts=(),
            coverage_as_of=as_of,
            protocol_config_sha256=protocol_hash,
        ),
        decisions,
    )
    AggregationStore(store).publish(aggregation)
    repository = Path(__file__).resolve().parents[2]
    dataset = OfflineDatasetBuilder(store).prepare(
        SyntheticDatasetInput(
            market,
            aggregation.aggregation_id,
            PROTOCOL,
            "1" * 40,
            (repository / "requirements-lock.txt").read_bytes(),
            (repository / "requirements-phase2.txt").read_bytes(),
        )
    )
    DatasetStore(store).publish(dataset)
    return Corpus(store, dataset, SyntheticExperimentInput(dataset.dataset_id, PROTOCOL, 3))


@pytest.fixture
def fast_parent(corpus: Corpus, monkeypatch: pytest.MonkeyPatch) -> Corpus:
    """Avoid repeated expensive parent replay in isolated model-output probes."""
    monkeypatch.setattr(DatasetStore, "get", lambda self, identity: corpus.dataset)
    return corpus


def fold_frame(ordinals: list[int] | None = None) -> pd.DataFrame:
    values = np.asarray(list(range(80)) if ordinals is None else ordinals, dtype=np.int64)
    opened = pd.DatetimeIndex([START + timedelta(hours=int(value)) for value in values])
    return pd.DataFrame(
        {
            "market_ordinal": values,
            "timestamp": opened,
            "decision_at": opened + pd.Timedelta(hours=1),
            "entry_timestamp": opened + pd.Timedelta(hours=1),
            "exit_timestamp": opened + pd.Timedelta(hours=5),
            "label": np.asarray(values % 2, dtype=np.int8),
        }
    )


def test_project_specific_error_contract() -> None:
    assert issubclass(ExperimentError, CryptoAIError)


@pytest.mark.parametrize(
    "actual,probability",
    [
        ([], []),
        ([0], []),
        ([0, 2], [0.1, 0.9]),
        ([-1, 1], [0.1, 0.9]),
        ([0.5, 1], [0.1, 0.9]),
        ([0, 1], [float("nan"), 0.9]),
        ([0, 1], [float("inf"), 0.9]),
        ([0, 1], [-0.01, 0.9]),
        ([0, 1], [0.1, 1.01]),
        ([[0, 1]], [[0.1, 0.9]]),
        (["0", "1"], [0.1, 0.9]),
        ([False, True], [0.1, 0.9]),
        ([0, 1], ["0.1", "0.9"]),
        ([0, 1], [False, True]),
        ([0, True], [0.1, 0.9]),
        ([0, 1], [0.1, True]),
    ],
)
def test_classification_rejects_invalid_vectors(actual, probability) -> None:
    with pytest.raises(ExperimentError):
        evaluate_classification(actual, probability)


def test_classification_complete_metrics_and_frozen_threshold() -> None:
    result = evaluate_classification([0, 1, 0, 1], [0.1, 0.5, 0.6, 0.9])
    expected = {
        "accuracy",
        "balanced_accuracy",
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "precision_class_0",
        "recall_class_0",
        "f1_class_0",
        "log_loss",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "confusion_matrix",
        "positive_label_rate",
        "predicted_positive_rate",
    }
    assert expected <= set(result)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert result["predicted_positive_rate"] == 0.75
    assert result["brier_score"] == pytest.approx(0.1575)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_are_explicitly_undefined(label: int) -> None:
    result = evaluate_classification([label] * 3, [0.1, 0.5, 0.9])
    assert result["roc_auc"] is None
    assert result["pr_auc"] is None
    assert len(result["confusion_matrix"]) == 2
    assert np.isfinite(result["log_loss"])


@pytest.mark.parametrize("size", [0, -1, True, 0.5, "3", 2099])
def test_fixture_split_size_rejects_invalid_or_unbounded_values(size) -> None:
    with pytest.raises(ExperimentError):
        shared_folds(fold_frame(), dataset_id="a" * 64, fixture_test_rows=size)


def test_research_split_size_is_not_silently_shrunk() -> None:
    with pytest.raises(ExperimentError):
        shared_folds(fold_frame(), dataset_id="a" * 64)


@pytest.mark.parametrize("identity", ["", "a" * 63, "A" * 64, "g" * 64, 3, None])
def test_folds_reject_invalid_parent_identity(identity) -> None:
    with pytest.raises(ExperimentError):
        shared_folds(fold_frame(), dataset_id=identity, fixture_test_rows=3)


@pytest.mark.parametrize("key", ["market_ordinal", "timestamp", "decision_at", "exit_timestamp"])
def test_folds_reject_missing_causal_columns(key: str) -> None:
    with pytest.raises(ExperimentError):
        shared_folds(fold_frame().drop(columns=key), dataset_id="a" * 64, fixture_test_rows=3)


def test_run_identifiers_have_unique_random_suffix() -> None:
    values = [generate_run_id() for _ in range(30)]
    assert len(set(values)) == len(values)
    assert all(re.search(r"[0-9a-f]{32}$", value) for value in values)
    assert all("/" not in value and ".." not in value for value in values)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_ordinal",
        "reversed",
        "negative_ordinal",
        "float_ordinal",
        "duplicate_time",
        "naive_time",
        "missing_time",
        "decision_offset",
        "exit_overlap",
        "duplicate_columns",
    ],
)
def test_folds_reject_ambiguous_or_noncausal_rows(mutation: str) -> None:
    frame = fold_frame()
    if mutation == "duplicate_ordinal":
        frame.loc[1, "market_ordinal"] = 0
    elif mutation == "reversed":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif mutation == "negative_ordinal":
        frame.loc[0, "market_ordinal"] = -1
    elif mutation == "float_ordinal":
        frame["market_ordinal"] = frame.market_ordinal.astype("float64")
    elif mutation == "duplicate_time":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif mutation == "naive_time":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    elif mutation == "missing_time":
        frame.loc[0, "exit_timestamp"] = pd.NaT
    elif mutation == "decision_offset":
        frame.loc[0, "decision_at"] += pd.Timedelta(hours=1)
    elif mutation == "exit_overlap":
        frame.loc[0, "exit_timestamp"] = frame.timestamp.iloc[-1]
    elif mutation == "duplicate_columns":
        frame.columns = ["timestamp"] + list(frame.columns[1:])
    with pytest.raises(ExperimentError):
        shared_folds(frame, dataset_id="a" * 64, fixture_test_rows=3)


@pytest.mark.parametrize("bad", [None, {}, object(), "store"])
def test_engine_only_accepts_exact_local_parent_store(bad) -> None:
    with pytest.raises(ExperimentError):
        OfflineExperimentEngine(bad)


@pytest.mark.parametrize("bad", [None, {}, object(), "request"])
def test_engine_only_accepts_exact_synthetic_request(tmp_path: Path, bad) -> None:
    with pytest.raises(ExperimentError):
        OfflineExperimentEngine(ContentAddressedStore(tmp_path / "parents")).run(bad)


@pytest.mark.parametrize(
    "protocol",
    [
        b"{}",
        b'{"synthetic":false}',
        b'{"synthetic":1}',
        b'{"synthetic":true,"synthetic":false}',
        b'{"synthetic":true} trailing prose',
        b'{"synthetic":true,"value":NaN}',
        b'{"synthetic":true,"value":1e999}',
        b' { "synthetic" : true } ',
        "not exact bytes",
    ],
)
def test_real_or_noncanonical_protocol_is_rejected(tmp_path: Path, protocol) -> None:
    with pytest.raises(ExperimentError):
        OfflineExperimentEngine(ContentAddressedStore(tmp_path / "parents")).run(
            SyntheticExperimentInput("a" * 64, protocol, 3)
        )


def test_missing_prepared_parent_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError):
        OfflineExperimentEngine(ContentAddressedStore(tmp_path / "parents")).run(
            SyntheticExperimentInput("a" * 64, PROTOCOL, 3)
        )


def test_original_ordinal_purge_does_not_compress_provider_gap_holes() -> None:
    frame = fold_frame([value for value in range(80) if value not in range(56, 61)])
    result = shared_folds(frame, dataset_id="a" * 64, fixture_test_rows=3)
    folds = result["folds"]
    assert len(folds) == 5
    first = folds[0]
    assert first["training_positions"] == list(range(56))
    assert first["validation_positions"] == [60, 61, 62]
    previous_validation = []
    for fold in folds:
        train = frame.iloc[fold["training_positions"]]
        validation = frame.iloc[fold["validation_positions"]]
        assert train.exit_timestamp.max() < validation.timestamp.iloc[0]
        assert train.market_ordinal.max() <= validation.market_ordinal.iloc[0] - 6
        assert len(validation) == 3
        assert set(fold["training_positions"]).isdisjoint(fold["validation_positions"])
        previous_validation.extend(fold["validation_positions"])
    assert previous_validation == list(range(60, 75))


def test_gap_shaped_ordinals_reject_nonexpanding_training_history() -> None:
    frame = fold_frame(list(range(55)) + list(range(60, 75)))
    assert issubclass(ExperimentSplitError, CryptoAIError)
    with pytest.raises(ExperimentSplitError, match="expand"):
        shared_folds(frame, dataset_id="a" * 64, fixture_test_rows=3)


def test_shared_folds_are_expanding_contiguous_deterministic_and_parent_bound() -> None:
    frame = fold_frame()
    first = shared_folds(frame, dataset_id="a" * 64, fixture_test_rows=4)
    second = shared_folds(frame, dataset_id="a" * 64, fixture_test_rows=4)
    rebound = shared_folds(frame, dataset_id="b" * 64, fixture_test_rows=4)
    assert canonicalize(first) == canonicalize(second)
    assert sha256_bytes(canonicalize(first)) != sha256_bytes(canonicalize(rebound))
    assert len(first["folds"]) == 5
    assert [row["validation_positions"] for row in first["folds"]] == [
        list(range(start, start + 4)) for start in range(60, 80, 4)
    ]
    assert [len(row["training_positions"]) for row in first["folds"]] == [55, 59, 63, 67, 71]
    assert all(
        len(left["training_positions"]) < len(right["training_positions"])
        for left, right in zip(first["folds"], first["folds"][1:], strict=False)
    )
    assert all(row["training_positions"][0] == 0 for row in first["folds"])


class FakeBooster:
    def __init__(self, scores: dict[str, dict[str, float]], columns=("a", "b", "c")):
        self.scores = scores
        self.feature_names = list(columns)

    def get_score(self, *, importance_type: str):
        return self.scores[importance_type]


class FakeTree:
    def __init__(self, scores: dict[str, dict[str, float]], columns=("a", "b", "c")):
        self.booster = FakeBooster(scores, columns)

    def get_booster(self):
        return self.booster


def test_frozen_estimators_and_scaler_parameters() -> None:
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "config/phase2_protocol.json").read_bytes()
    )
    matrix = config["experiment_matrix"]
    for cell in ("A", "C"):
        pipeline = experiment_module._make_model(cell)
        assert (
            pipeline.named_steps["classifier"].get_params()
            == matrix["logistic"]["effective_sklearn_1_9_0_params"]
        )
        assert pipeline.named_steps["scaler"].get_params() == {
            "copy": True,
            "with_mean": True,
            "with_std": True,
        }
    for cell in ("B", "D"):
        parameters = experiment_module._make_model(cell).get_params()
        assert {key: parameters[key] for key in matrix["xgboost"]} == matrix["xgboost"]


@pytest.mark.parametrize("cell", ["", "a", "E", 1, None])
def test_model_factory_rejects_unknown_cells(cell) -> None:
    with pytest.raises(ExperimentError):
        experiment_module._make_model(cell)


class FakeEstimator:
    def __init__(self, mode: str = "valid"):
        self.mode = mode
        self.classes_ = np.array([0, 1])
        if mode == "reversed_classes":
            self.classes_ = np.array([1, 0])
        elif mode == "missing_class":
            self.classes_ = np.array([0])

    def fit(self, features, labels):
        if self.mode == "fit_error":
            raise ValueError("synthetic fit failure")
        return self

    def predict_proba(self, features):
        length = len(features)
        if self.mode == "prediction_error":
            raise ValueError("synthetic inference failure")
        if self.mode == "wrong_rows":
            return np.tile([0.25, 0.75], (length + 1, 1))
        if self.mode == "one_column":
            return np.ones((length, 1))
        if self.mode == "one_dimension":
            return np.full(length, 0.75)
        if self.mode == "nan":
            return np.tile([float("nan"), 0.75], (length, 1))
        if self.mode == "infinite":
            return np.tile([0.25, float("inf")], (length, 1))
        if self.mode == "out_of_bounds":
            return np.tile([-0.25, 1.25], (length, 1))
        if self.mode == "nonunit_sum":
            return np.tile([0.25, 0.25], (length, 1))
        if self.mode == "strings":
            return np.tile(["0.25", "0.75"], (length, 1))
        return np.tile([0.25, 0.75], (length, 1))

    def get_booster(self):
        return FakeBooster({name: {} for name in ("total_gain", "weight", "total_cover")})


@pytest.mark.parametrize(
    "mode",
    [
        "reversed_classes",
        "missing_class",
        "fit_error",
        "prediction_error",
        "wrong_rows",
        "one_column",
        "one_dimension",
        "nan",
        "infinite",
        "out_of_bounds",
        "nonunit_sum",
        "strings",
    ],
)
def test_model_failures_and_malformed_probabilities_fail_closed(
    fast_parent: Corpus,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setattr(experiment_module, "_make_model", lambda cell: FakeEstimator(mode))
    with pytest.raises(ExperimentError):
        OfflineExperimentEngine(fast_parent.store).run(fast_parent.request)


@pytest.fixture(scope="module")
def artifact(corpus: Corpus) -> ExperimentArtifact:
    # This path retains the genuine DatasetStore.get and transitive parent replay.
    return OfflineExperimentEngine(corpus.store).run(corpus.request)


def test_genuine_parent_to_four_cell_training_and_continuous_oof(
    corpus: Corpus,
    artifact: ExperimentArtifact,
) -> None:
    assert len(artifact.experiment_id) == 64
    assert artifact.experiment_id == sha256_bytes(dict(artifact.files)["experiment_manifest.json"])
    assert corpus.dataset.dataset_id in canonicalize(artifact.manifest).decode()
    tables = {cell: artifact.predictions(cell) for cell in CELLS}
    for cell, frame in tables.items():
        assert len(frame) == 15
        assert not frame.index.has_duplicates
        assert frame.probability_score.dtype == np.dtype("float64")
        assert frame.probability_score.between(0.0, 1.0).all()
        assert np.isfinite(frame.probability_score).all()
        assert set(frame.fold_number) == {1, 2, 3, 4, 5}
        assert frame.groupby("fold_number").size().tolist() == [3] * 5
        assert frame.actual_label.tolist() == corpus.dataset.labeled.label.iloc[-15:].tolist()
        assert (
            frame.predicted_label.tolist() == (frame.probability_score >= 0.5).astype(int).tolist()
        )
        metrics = artifact.metrics(cell)
        assert metrics
        assert artifact.importance(cell)
    for cell in ("B", "C", "D"):
        for column in (
            "decision_at",
            "entry_timestamp",
            "exit_timestamp",
            "market_ordinal",
            "fold_number",
            "actual_label",
        ):
            pd.testing.assert_series_equal(
                tables["A"][column], tables[cell][column], check_exact=True
            )


def test_verified_engine_rerun_is_exactly_deterministic(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
) -> None:
    rerun = OfflineExperimentEngine(fast_parent.store).run(fast_parent.request)
    assert rerun.experiment_id == artifact.experiment_id
    assert rerun.files == artifact.files


def test_logistic_scaler_is_fitted_on_training_rows_only(
    fast_parent: Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.preprocessing import StandardScaler

    captures = []
    original = StandardScaler.fit

    def record(self, features, labels=None, **kwargs):
        captures.append(features.copy(deep=True))
        return original(self, features, labels, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record)
    OfflineExperimentEngine(fast_parent.store).run(fast_parent.request)
    assert len(captures) == 10
    frame = fast_parent.dataset.labeled
    folds = shared_folds(frame, dataset_id=fast_parent.dataset.dataset_id, fixture_test_rows=3)
    for offset, columns in ((0, TECHNICAL_COLUMNS), (5, COMBINED_COLUMNS)):
        for number, fold in enumerate(folds["folds"]):
            expected = frame.iloc[fold["training_positions"]][list(columns)]
            pd.testing.assert_frame_equal(captures[offset + number], expected, check_exact=True)


@pytest.mark.parametrize("cell", ["", "a", "E", None, 1])
def test_artifact_views_reject_unknown_cells(artifact: ExperimentArtifact, cell) -> None:
    for accessor in (artifact.predictions, artifact.metrics, artifact.importance):
        with pytest.raises(ExperimentError):
            accessor(cell)


def test_atomic_publication_no_replace_and_verified_reload(
    corpus: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    store = ExperimentStore(root, corpus.store)
    run_id = store.publish(artifact)
    assert (root / run_id / "manifest.json").is_file()
    assert (root / run_id / "experiment_manifest.json").is_file()
    assert store.get(run_id).files == artifact.files
    assert not list(root.glob(".staging-*"))
    with pytest.raises(ExperimentError):
        store.publish(artifact, run_id=run_id)
    assert_only_unpublished_staging_tombstones(root)


def test_publication_collision_preserves_existing_run_bytes(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    published = store.root / run_id
    before = {
        path.relative_to(published).as_posix(): path.read_bytes()
        for path in published.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    after = {
        path.relative_to(published).as_posix(): path.read_bytes()
        for path in published.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert store.get(run_id).files == artifact.files
    assert_only_unpublished_staging_tombstones(store.root)


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", ".", "..", "a/b"])
def test_publication_rejects_unsafe_run_identifiers(
    corpus: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    run_id,
) -> None:
    with pytest.raises(ExperimentError):
        ExperimentStore(tmp_path / "runs", corpus.store).publish(artifact, run_id=run_id)


def test_publication_rejects_parent_directory_symlinks(
    corpus: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(CryptoAIError):
        ExperimentStore(alias / "runs", corpus.store).publish(artifact)


def test_global_importance_uses_total_split_weighted_gain_and_cover() -> None:
    first = FakeTree(
        {
            "weight": {"a": 2.0, "b": 1.0},
            "total_gain": {"a": 8.0, "b": 6.0},
            "total_cover": {"a": 20.0, "b": 15.0},
        }
    )
    second = FakeTree(
        {
            "weight": {"a": 1.0, "b": 3.0},
            "total_gain": {"a": 10.0, "b": 10.0},
            "total_cover": {"a": 7.0, "b": 5.0},
        }
    )
    result = global_feature_importance([first, second], ("a", "b", "c"))
    assert result == {
        "applicable": True,
        "method": "split_weighted_across_folds",
        "fitted_fold_count": 2,
        "warnings": [],
        "features": [
            {"feature": "a", "weight": 3.0, "gain": 6.0, "cover": 9.0},
            {"feature": "b", "weight": 4.0, "gain": 4.0, "cover": 5.0},
            {"feature": "c", "weight": 0.0, "gain": 0.0, "cover": 0.0},
        ],
    }


def test_importance_empty_fitted_boosters_preserves_every_feature() -> None:
    result = global_feature_importance([], ("a", "b"))
    assert result["features"] == [
        {"feature": name, "weight": 0.0, "gain": 0.0, "cover": 0.0} for name in ("a", "b")
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "unknown",
        "missing",
        "nan",
        "negative",
        "fractional_weight",
        "boolean",
        "zero_nonzero_gain",
        "zero_nonzero_cover",
    ],
)
def test_importance_rejects_invalid_booster_contract(mutation: str) -> None:
    values = {"weight": {"a": 1.0}, "total_gain": {"a": 1.0}, "total_cover": {"a": 1.0}}
    columns = ("a", "b", "c")
    if mutation == "order":
        columns = ("b", "a", "c")
    elif mutation == "unknown":
        values = {key: {"unexpected": 1.0} for key in values}
    elif mutation == "missing":
        values["total_gain"] = {}
    elif mutation == "nan":
        values["total_gain"]["a"] = float("nan")
    elif mutation == "negative":
        values["total_cover"]["a"] = -1.0
    elif mutation == "fractional_weight":
        values["weight"]["a"] = 0.5
    elif mutation == "boolean":
        values["weight"]["a"] = True
    elif mutation == "zero_nonzero_gain":
        values["weight"]["a"] = 0.0
        values["total_cover"]["a"] = 0.0
    elif mutation == "zero_nonzero_cover":
        values["weight"]["a"] = 0.0
        values["total_gain"]["a"] = 0.0
    with pytest.raises(ExperimentError):
        global_feature_importance([FakeTree(values, columns)], ("a", "b", "c"))


def rehash_file(artifact: ExperimentArtifact, name: str, raw: bytes) -> ExperimentArtifact:
    files = dict(artifact.files)
    old = files[name]
    files[name] = raw
    manifest = json.loads(files["experiment_manifest.json"])
    manifest["files"][name] = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
    for field, value in manifest.items():
        if field.endswith("_sha256") and value == sha256_bytes(old):
            manifest[field] = sha256_bytes(raw)
    files["experiment_manifest.json"] = canonicalize(manifest)
    return replace(artifact, files=tuple(sorted(files.items())))


@pytest.mark.parametrize(
    "name",
    [
        "cells/A/metrics.json",
        "cells/B/predictions.json",
        "cells/C/training.json",
        "cells/D/importance.json",
        "folds.json",
        "config.json",
    ],
)
def test_recomputed_hashes_cannot_publish_forged_semantics(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    name: str,
) -> None:
    value = json.loads(dict(artifact.files)[name])
    if name.endswith("metrics.json"):
        value["aggregate"]["accuracy"] = 1.0 - value["aggregate"]["accuracy"]
    elif name.endswith("predictions.json"):
        value["rows"][0][6] = 0.111111
    elif name.endswith("training.json"):
        value[0]["training_market_ordinals"] = []
    elif name.endswith("importance.json"):
        value["features"][0]["gain"] += 100.0
    elif name == "folds.json":
        value["folds"][0]["purge_market_ordinals"] = []
    else:
        value["threshold"] = 0.75
    forged = rehash_file(artifact, name, canonicalize(value))
    assert forged.experiment_id != artifact.experiment_id
    with pytest.raises(ExperimentError):
        ExperimentStore(tmp_path / "runs", fast_parent.store).publish(forged)
    assert not list((tmp_path / "runs").iterdir())


def test_manifest_is_last_write_and_rename_occurs_after_fsync(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    write = experiment_module._write_fsynced_at
    fsync = experiment_module._fsync_tree_directories_at
    rename = experiment_module._atomic_rename_directory_no_replace

    def observed_write(directory, name, raw):
        events.append(("write", name))
        return write(directory, name, raw)

    def observed_fsync(directory, **kwargs):
        events.append(("fsync", None))
        return fsync(directory, **kwargs)

    def observed_rename(directory, source, target):
        events.append(("rename", source))
        assert source.startswith(".staging-")
        return rename(directory, source, target)

    monkeypatch.setattr(experiment_module, "_write_fsynced_at", observed_write)
    monkeypatch.setattr(experiment_module, "_fsync_tree_directories_at", observed_fsync)
    monkeypatch.setattr(experiment_module, "_atomic_rename_directory_no_replace", observed_rename)
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    writes = [name for event, name in events if event == "write"]
    assert writes[-1] == "manifest.json"
    assert events[-2][0] == "fsync"
    assert events[-1][0] == "rename"
    assert (tmp_path / "runs" / run_id).is_dir()


def test_post_rename_payload_mutation_leaves_no_completed_run(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    rename = experiment_module._atomic_rename_directory_no_replace
    renamed = []

    def mutate_at_rename(directory, source, target):
        if target != run_id:
            return rename(directory, source, target)
        staged = store.root / source / "source.py"
        staged.write_bytes(staged.read_bytes() + b"\n")
        result = rename(directory, source, target)
        renamed.append(True)
        return result

    monkeypatch.setattr(experiment_module, "_atomic_rename_directory_no_replace", mutate_at_rename)
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert renamed, "the adversarial mutation must occur immediately before rename"
    assert not (store.root / run_id / "manifest.json").exists()
    with pytest.raises(ExperimentError):
        store.get(run_id)
    assert_only_unpublished_staging_tombstones(store.root)


def test_post_rename_failure_never_requires_unsafe_quarantine_rename(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    rename = experiment_module._atomic_rename_directory_no_replace
    published = []
    check_attachment = store._check_attachment

    def fail_rollback_rename(directory, source, target):
        if source == run_id:
            raise OSError("synthetic quarantine rename failure")
        result = rename(directory, source, target)
        if target == run_id:
            published.append(True)
        return result

    def fail_post_rename_validation(*args, **kwargs):
        raise ExperimentIntegrityError("synthetic post-rename validation failure")

    monkeypatch.setattr(
        experiment_module, "_atomic_rename_directory_no_replace", fail_rollback_rename
    )
    monkeypatch.setattr(store, "_check_attachment", fail_post_rename_validation)
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert published, "the run must become public before validation fails"
    assert not (store.root / run_id / "manifest.json").exists()
    monkeypatch.setattr(store, "_check_attachment", check_attachment)
    with pytest.raises(ExperimentError):
        store.get(run_id)


def test_rollback_descriptor_does_not_displace_concurrent_replacement(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    unrelated = store.root / ".unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel.bin"
    sentinel.write_bytes(b"unrelated directory must not be moved or changed")
    foreign_identity = (unrelated.stat().st_dev, unrelated.stat().st_ino)
    detached = store.root / ".detached-owned-run"
    rollback = experiment_module._rollback_owned_directory
    check_attachment = store._check_attachment
    swapped = []

    def fail_post_rename_validation(*args, **kwargs):
        raise ExperimentIntegrityError("synthetic post-rename validation failure")

    def swap_before_descriptor_rollback(stage_descriptor):
        assert (store.root / run_id / "manifest.json").is_file()
        os.rename(store.root / run_id, detached)
        os.rename(unrelated, store.root / run_id)
        swapped.append(True)
        return rollback(stage_descriptor)

    monkeypatch.setattr(store, "_check_attachment", fail_post_rename_validation)
    monkeypatch.setattr(
        experiment_module, "_rollback_owned_directory", swap_before_descriptor_rollback
    )
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert swapped, "the replacement must occur after validation and before rollback"
    assert (store.root / run_id).is_dir()
    assert (store.root / run_id / "sentinel.bin").read_bytes() == (
        b"unrelated directory must not be moved or changed"
    )
    assert (store.root / run_id).stat().st_dev == foreign_identity[0]
    assert (store.root / run_id).stat().st_ino == foreign_identity[1]
    assert not (detached / "manifest.json").exists()
    monkeypatch.setattr(store, "_check_attachment", check_attachment)
    with pytest.raises(ExperimentError):
        store.get(run_id)


def test_rollback_does_not_touch_concurrently_replaced_nested_directory(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    unrelated = store.root / ".unrelated-cells"
    unrelated.mkdir()
    (unrelated / "sentinel.bin").write_bytes(b"nested replacement must not be touched")
    foreign_identity = (unrelated.stat().st_dev, unrelated.stat().st_ino)
    detached = store.root / ".detached-owned-cells"
    rollback = experiment_module._rollback_owned_directory
    check_attachment = store._check_attachment
    swapped = []

    def fail_post_rename_validation(*args, **kwargs):
        raise ExperimentIntegrityError("synthetic post-rename validation failure")

    def swap_nested_directory_before_rollback(stage_descriptor):
        nested = store.root / run_id / "cells"
        assert (store.root / run_id / "manifest.json").is_file()
        os.rename(nested, detached)
        os.rename(unrelated, nested)
        swapped.append(True)
        return rollback(stage_descriptor)

    monkeypatch.setattr(store, "_check_attachment", fail_post_rename_validation)
    monkeypatch.setattr(
        experiment_module, "_rollback_owned_directory", swap_nested_directory_before_rollback
    )
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert swapped, "the nested replacement must occur immediately before rollback"
    foreign = store.root / run_id / "cells"
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == foreign_identity
    assert (foreign / "sentinel.bin").read_bytes() == b"nested replacement must not be touched"
    assert (detached / "A").is_dir()
    assert not (store.root / run_id / "manifest.json").exists()
    monkeypatch.setattr(store, "_check_attachment", check_attachment)
    with pytest.raises(ExperimentError):
        store.get(run_id)


def test_pre_rename_rollback_preserves_unrelated_staging_replacement(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    unrelated = store.root / ".unrelated-staging"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel.bin"
    sentinel.write_bytes(b"unrelated staging replacement must remain unchanged")
    foreign_identity = (unrelated.stat().st_dev, unrelated.stat().st_ino)
    detached = store.root / ".detached-owned-staging"
    fsync = experiment_module._fsync_tree_directories_at
    rollback = experiment_module._rollback_owned_directory
    swapped = []

    def fail_before_rename(directory, **kwargs):
        fsync(directory, **kwargs)
        raise ExperimentIntegrityError("synthetic pre-rename verification failure")

    def swap_before_descriptor_rollback(stage_descriptor):
        staged = next(store.root.glob(".staging-*"))
        assert (staged / "manifest.json").is_file()
        original_name = staged.name
        os.rename(staged, detached)
        os.rename(unrelated, store.root / original_name)
        swapped.append(original_name)
        return rollback(stage_descriptor)

    monkeypatch.setattr(experiment_module, "_fsync_tree_directories_at", fail_before_rename)
    monkeypatch.setattr(
        experiment_module, "_rollback_owned_directory", swap_before_descriptor_rollback
    )
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert swapped, "the replacement must occur after staging and before rollback"
    foreign = store.root / swapped[0]
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == foreign_identity
    assert (foreign / "sentinel.bin").read_bytes() == (
        b"unrelated staging replacement must remain unchanged"
    )
    assert not (detached / "manifest.json").exists()
    assert not (store.root / run_id).exists()


def test_rename_succeeds_then_raises_leaves_no_completed_run(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = generate_run_id()
    rename = experiment_module._atomic_rename_directory_no_replace
    renamed = []

    def raise_after_rename(directory, source, target):
        result = rename(directory, source, target)
        if target == run_id:
            renamed.append(True)
            raise OSError("synthetic error after successful rename")
        return result

    monkeypatch.setattr(
        experiment_module, "_atomic_rename_directory_no_replace", raise_after_rename
    )
    with pytest.raises(ExperimentIntegrityError):
        store.publish(artifact, run_id=run_id)
    assert renamed, "the public rename must succeed before the injected error"
    assert not (store.root / run_id / "manifest.json").exists()
    with pytest.raises(ExperimentError):
        store.get(run_id)
    assert_only_unpublished_staging_tombstones(store.root)


def test_incomplete_payload_write_has_no_completed_publication(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_write(*args, **kwargs):
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(experiment_module, "_write_fsynced_at", broken_write)
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    with pytest.raises(ExperimentError):
        store.publish(artifact)
    assert_only_unpublished_staging_tombstones(store.root)
    assert not any(path.name == "manifest.json" for path in store.root.rglob("manifest.json"))


@pytest.mark.parametrize(
    "path",
    [
        "cells/A/predictions.json",
        "cells/B/metrics.json",
        "prepared_dataset_manifest.json",
        "folds.json",
        "experiment_manifest.json",
    ],
)
def test_single_byte_retained_artifact_tampering_fails_closed(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    path: str,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    target = store.root / run_id / path
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    with pytest.raises(ExperimentError):
        store.get(run_id)


@pytest.mark.parametrize("invalid", [7, "not-hex", "A" * 64, None])
def test_malformed_outer_metadata_aborts_before_payload_reads(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    path = store.root / run_id / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["metadata"]["experiment_id"] = invalid
    path.write_bytes(canonicalize(manifest))
    reads = []
    original = experiment_module._read_regular_file_at_once

    def captured(directory, name, **kwargs):
        reads.append(name)
        return original(directory, name, **kwargs)

    monkeypatch.setattr(experiment_module, "_read_regular_file_at_once", captured)
    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", captured)
    with pytest.raises(ExperimentError):
        store.get(run_id)
    assert reads == ["manifest.json"]


def test_late_fifo_injection_during_capture_is_rejected(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    original = storage_module._read_regular_file_at_once
    injected = []

    def captured(directory, name, **kwargs):
        result = original(directory, name, **kwargs)
        if name == "predictions.json" and not injected:
            target = store.root / run_id / "cells" / "A" / "late-fifo"
            os.mkfifo(target)
            injected.append(target)
        return result

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", captured)
    with pytest.raises(ExperimentError):
        store.get(run_id)
    assert injected


def test_verified_run_remains_readable_after_a_later_head_commit(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    git = dataset_module._git

    def later_git(*args):
        if args == ("rev-parse", "HEAD"):
            return b"2" * 40 + b"\n"
        return git(*args)

    monkeypatch.setattr(dataset_module, "_git", later_git)
    assert store.get(run_id).files == artifact.files


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_training_is_explicit_constant_fallback_without_fitting(
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
    label: int,
) -> None:
    # Exercise this internal branch directly. This deliberately altered frame is
    # never admitted through the verified public DatasetStore training boundary.
    class SyntheticSingleClassBranch:
        files = corpus.dataset.files
        manifest = corpus.dataset.manifest
        labeled = corpus.dataset.labeled.assign(label=np.int8(label))

    def forbidden_fit(cell):
        raise AssertionError("single-class folds must not fit a substitute classifier")

    monkeypatch.setattr(experiment_module, "_make_model", forbidden_fit)
    result = experiment_module._build(
        SyntheticSingleClassBranch(), corpus.request, experiment_module._execution_context()
    )
    for cell in CELLS:
        assert result.predictions(cell).probability_score.tolist() == [float(label)] * 15
        traces = json.loads(dict(result.files)[f"cells/{cell}/training.json"])
        assert all(row["predictor"] == "constant_class" for row in traces)
        assert all(row["warnings"] for row in traces)
        assert result.metrics(cell)["aggregate"]["roc_auc"] is None
        assert result.metrics(cell)["aggregate"]["pr_auc"] is None
    for cell in ("B", "D"):
        assert result.importance(cell)["fitted_fold_count"] == 0
        assert result.importance(cell)["warnings"]


def test_directory_replaced_by_symlink_during_capture_is_rejected(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    run_id = store.publish(artifact)
    original = storage_module._read_regular_file_at_once
    injected = []

    def captured(directory, name, **kwargs):
        result = original(directory, name, **kwargs)
        if name == "predictions.json" and not injected:
            target = store.root / run_id / "cells" / "A"
            moved = tmp_path / "moved-cell"
            target.rename(moved)
            target.symlink_to(moved, target_is_directory=True)
            injected.append(target)
        return result

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", captured)
    with pytest.raises(ExperimentError):
        store.get(run_id)
    assert injected


def test_publication_root_replacement_during_fsync_never_reports_success(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    fsync = experiment_module._fsync_tree_directories_at
    injected = []

    def replace_root(directory, **kwargs):
        result = fsync(directory, **kwargs)
        if not injected:
            moved = tmp_path / "detached-runs"
            store.root.rename(moved)
            store.root.mkdir()
            injected.append(moved)
        return result

    monkeypatch.setattr(experiment_module, "_fsync_tree_directories_at", replace_root)
    with pytest.raises(ExperimentError):
        store.publish(artifact)
    assert injected
    assert not list(store.root.iterdir())


def test_payload_mutated_during_fsync_is_rejected_before_final_rename(
    fast_parent: Corpus,
    artifact: ExperimentArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExperimentStore(tmp_path / "runs", fast_parent.store)
    fsync = experiment_module._fsync_tree_directories_at
    rename = experiment_module._atomic_rename_directory_no_replace
    renamed = []
    injected = []

    def mutate_payload(directory, **kwargs):
        result = fsync(directory, **kwargs)
        if not injected:
            target = next(store.root.glob(".staging-*/source.py"))
            target.write_bytes(target.read_bytes() + b"\n")
            injected.append(True)
        return result

    def record_rename(*args, **kwargs):
        renamed.append(True)
        return rename(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "_fsync_tree_directories_at", mutate_payload)
    monkeypatch.setattr(experiment_module, "_atomic_rename_directory_no_replace", record_rename)
    with pytest.raises(ExperimentError):
        store.publish(artifact)
    assert injected
    assert not renamed
    assert_only_unpublished_staging_tombstones(store.root)
    assert not any(path.name == "manifest.json" for path in store.root.rglob("manifest.json"))
