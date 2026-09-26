"""End-to-end synthetic Milestone 6 -> Milestone 7 report integration."""

from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import artifacts as artifact_module
from crypto_ai.phase2 import backtests as backtest_module
from crypto_ai.phase2.artifacts import ArtifactStore
from crypto_ai.phase2.backtests import OfflineBacktestEngine, verify_report
from crypto_ai.phase2.experiments import ExperimentStore, generate_run_id
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from phase2.test_experiments import Corpus

pytest_plugins = ("phase2.test_experiments",)


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 7 integration must stay entirely offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(scope="module")
def report_case(
    corpus: Corpus, artifact, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentStore, str]:
    root = tmp_path_factory.mktemp("milestone7-verified") / "runs"
    source = ExperimentStore(root, corpus.store)
    run_id = source.publish(artifact)
    return source, run_id


@pytest.fixture(scope="module")
def bundle(report_case):
    source, run_id = report_case
    return OfflineBacktestEngine(source).run(run_id)


def test_public_engine_requires_exact_verified_synthetic_experiment_store() -> None:
    with pytest.raises(CryptoAIError):
        OfflineBacktestEngine(object())


def test_implementation_hash_transitively_binds_phase1_execution_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    baseline = backtest_module._implementation_source_hash()

    def changed_engine_bytes(path: Path) -> bytes:
        value = original(path)
        if path.as_posix().endswith("/src/crypto_ai/backtesting/engine.py"):
            return value + b"x"
        return value

    monkeypatch.setattr(Path, "read_bytes", changed_engine_bytes)
    assert backtest_module._implementation_source_hash() != baseline


def test_verified_parent_to_five_deterministic_report_buffers(bundle, report_case) -> None:
    source, run_id = report_case
    assert bundle.metadata["source_run_id"] == run_id
    assert bundle.metadata["synthetic"] is True
    assert set(dict(bundle.files)) == {
        "strategy_metrics.json",
        "cost_sensitivity.json",
        "baseline_metrics.json",
        "ablation_report.json",
        "development_report.md",
    }
    strategy = bundle.json("strategy_metrics.json")
    baselines = bundle.json("baseline_metrics.json")
    costs = bundle.json("cost_sensitivity.json")
    ablation = bundle.json("ablation_report.json")
    assert set(strategy["cells"]) == set("ABCD")
    assert set(costs["scenarios"]) == {"low", "base", "high"}
    assert set(baselines["deterministic"]["base"]) == {
        "cash",
        "buy_and_hold",
        "ema_9_21",
        "momentum_24",
    }
    assert all(
        baselines["random_exposure"][cell][scenario]["simulations"] == 1000
        for cell in "ABCD"
        for scenario in ("low", "base", "high")
    )
    assert len(ablation["folds"]) == 5
    assert ablation["research_gates_evaluated"] is False
    first_window = strategy["common_window"]
    assert first_window["equity_marks"] == first_window["open_to_open_intervals"] + 1
    for cell in "ABCD":
        for scenario in ("low", "base", "high"):
            result = strategy["cells"][cell][scenario]
            assert len(result["equity_curve"]) == first_window["equity_marks"]
            assert result["metrics"]["num_trades"] == len(result["trade_ledger"])
            for trade in result["trade_ledger"]:
                assert trade["signal_market_ordinal"] + 1 == trade["entry_market_ordinal"]
                assert trade["exit_market_ordinal"] - trade["entry_market_ordinal"] == 4
                assert trade["signal_timestamp"] == trade["entry_timestamp"]
    markdown = dict(bundle.files)["development_report.md"]
    assert b"NOT_EVALUATED_SYNTHETIC_ONLY" in markdown
    for name, raw in bundle.files:
        if name.endswith(".json"):
            assert sha256_bytes(raw).encode() in markdown
    assert source.get(run_id).experiment_id == bundle.metadata["source_experiment_id"]


def test_semantic_report_replay_rejects_hash_consistent_false_metrics(bundle, report_case) -> None:
    source, _ = report_case
    files = dict(bundle.files)
    old_hash = sha256_bytes(files["strategy_metrics.json"])
    strategy = bundle.json("strategy_metrics.json")
    strategy["cells"]["A"]["base"]["metrics"]["total_return"] += 0.01
    files["strategy_metrics.json"] = canonicalize(strategy)
    new_hash = sha256_bytes(files["strategy_metrics.json"])
    files["development_report.md"] = files["development_report.md"].replace(
        old_hash.encode(), new_hash.encode()
    )
    with pytest.raises(CryptoAIError):
        verify_report(source, files, bundle.metadata)


def test_verified_synthetic_report_rerun_is_byte_identical(bundle, report_case) -> None:
    source, run_id = report_case
    repeated = OfflineBacktestEngine(source).run(run_id)
    assert repeated.files == bundle.files
    assert repeated.metadata == bundle.metadata


def test_report_store_publishes_separate_immutable_run(bundle, report_case) -> None:
    source, source_run_id = report_case
    report_store = ArtifactStore(source.root, source)
    report_run_id = report_store.publish(dict(bundle.files), bundle.metadata)
    assert report_run_id != source_run_id
    loaded = report_store.get(report_run_id)
    assert loaded.files == dict(bundle.files)
    assert loaded.manifest["metadata"] == bundle.metadata
    assert source.get(source_run_id).experiment_id == bundle.metadata["source_experiment_id"]
    assert (Path(source.root) / source_run_id / "manifest.json").is_file()


def test_forged_staged_manifest_rejected_after_rename(
    bundle, report_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = report_case
    report_store = ArtifactStore(source.root, source)
    run_id = generate_run_id()
    rename = artifact_module._atomic_rename_directory_no_replace
    forged = []

    def forge_then_rename(parent, stage_name, published_name):
        stage = report_store.root / stage_name
        manifest = artifact_module._json((stage / "manifest.json").read_bytes())
        original_hash = manifest["metadata"]["cost_config_sha256"]
        manifest["metadata"]["cost_config_sha256"] = (
            "f" * 64 if original_hash != "f" * 64 else "e" * 64
        )
        substitute = stage / "forged-manifest.tmp"
        substitute.write_bytes(canonicalize(manifest))
        os.replace(substitute, stage / "manifest.json")
        forged.append(True)
        return rename(parent, stage_name, published_name)

    monkeypatch.setattr(artifact_module, "_semantic_replay", lambda *_: None)
    monkeypatch.setattr(artifact_module, "_atomic_rename_directory_no_replace", forge_then_rename)
    with pytest.raises(artifact_module.ArtifactIntegrityError, match="manifest bytes"):
        report_store.publish(dict(bundle.files), copy.deepcopy(bundle.metadata), run_id=run_id)
    assert forged
    assert not (report_store.root / run_id / "manifest.json").exists()
    with pytest.raises(artifact_module.ArtifactError):
        report_store.get(run_id)


def test_caller_metadata_mutation_during_payload_write_fails_closed(
    bundle, report_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = report_case
    report_store = ArtifactStore(source.root, source)
    metadata = copy.deepcopy(bundle.metadata)
    run_id = generate_run_id()
    write = artifact_module._write_fsynced_at
    mutated = []

    def mutate_after_write(descriptor, name, raw):
        result = write(descriptor, name, raw)
        if name == "ablation_report.json":
            original_hash = metadata["source_prediction_sha256"]["A"]
            metadata["source_prediction_sha256"]["A"] = (
                "f" * 64 if original_hash != "f" * 64 else "e" * 64
            )
            mutated.append(True)
        return result

    monkeypatch.setattr(artifact_module, "_semantic_replay", lambda *_: None)
    monkeypatch.setattr(artifact_module, "_write_fsynced_at", mutate_after_write)
    with pytest.raises(artifact_module.ArtifactIntegrityError, match="caller report metadata"):
        report_store.publish(dict(bundle.files), metadata, run_id=run_id)
    assert mutated
    assert not (report_store.root / run_id).exists()
    assert not any(
        (stage / "manifest.json").exists()
        for stage in report_store.root.glob(".staging-" + run_id + "-*")
    )


def test_workflow_entrypoint_passes_only_verified_bundle_to_immutable_store(
    bundle, report_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_run_id = report_case
    runner = OfflineBacktestEngine(source)
    requested = "20260926T000000000000Z_" + "f" * 32
    seen = []

    def verified_run(value):
        assert value == source_run_id
        return bundle

    def record_publish(self, files, metadata, *, run_id):
        seen.append((self.root, files, metadata, run_id))
        return run_id

    monkeypatch.setattr(runner, "run", verified_run)
    monkeypatch.setattr(ArtifactStore, "publish", record_publish)
    assert runner.publish(source_run_id, report_run_id=requested) == requested
    assert seen == [(source.root, dict(bundle.files), bundle.metadata, requested)]
