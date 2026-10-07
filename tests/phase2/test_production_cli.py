"""Offline mocked CLI routing tests; no production fit or real data is loaded."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_ai import cli
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import workflow


def _arguments() -> list[str]:
    return [
        "train-production",
        "--synthetic-only",
        "--evaluation-run-id",
        "synthetic-evaluation",
        "--evaluation-root",
        "/synthetic/evaluations",
        "--development-run-dir",
        "/synthetic/development",
        "--versions-root",
        "/synthetic/versions",
        "--model-version",
        "fixture-v1",
        "--authorization-file",
        "/synthetic/authorization.json",
        "--training-as-of-utc",
        "2026-10-05T00:00:00Z",
        "--created-at-utc",
        "2026-10-05T00:01:00Z",
    ]


@pytest.mark.parametrize("arguments", [[], ["--help"], ["prepare", "--symbol", "BTC/USDT"]])
def test_dispatch_preserves_phase1_arguments_and_result(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    monkeypatch.setattr(cli, "main", lambda value: observed.append(value) or 17)
    assert workflow.dispatch(arguments) == 17
    assert observed == [arguments]
    assert observed[0] is arguments


def test_dispatch_none_preserves_phase1_sysargv_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []
    monkeypatch.setattr(sys, "argv", ["krypx", "prepare"])
    monkeypatch.setattr(cli, "main", lambda value: observed.append(value) or 19)
    assert workflow.dispatch() == 19
    assert observed == [None]


@pytest.mark.parametrize("use_sysargv", [False, True])
def test_dispatch_routes_only_explicit_phase2_prefix(
    use_sysargv: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    monkeypatch.setattr(workflow, "main", lambda value: observed.append(value) or 23)
    monkeypatch.setattr(cli, "main", lambda value: pytest.fail("Phase 1 handler reached"))
    arguments = ["phase2", *_arguments()]
    monkeypatch.setattr(sys, "argv", ["krypx", *arguments])
    assert workflow.dispatch(None if use_sysargv else arguments) == 23
    assert observed == [arguments[1:]]


@pytest.mark.parametrize(
    "flag",
    [
        "--synthetic-only",
        "--evaluation-run-id",
        "--evaluation-root",
        "--development-run-dir",
        "--versions-root",
        "--model-version",
        "--authorization-file",
        "--training-as-of-utc",
        "--created-at-utc",
    ],
)
def test_missing_required_argument_aborts_before_payload_access(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments()
    index = arguments.index(flag)
    del arguments[index : index + (1 if flag == "--synthetic-only" else 2)]
    monkeypatch.setattr(
        workflow, "_train_production", lambda value: pytest.fail("production path reached")
    )
    with pytest.raises(SystemExit) as caught:
        workflow.main(arguments)
    assert caught.value.code == 2


@pytest.mark.parametrize("flag", ["--training-as-of-utc", "--created-at-utc"])
@pytest.mark.parametrize(
    "value",
    ["2026-10-05", "2026-10-05T00:00:00", "2026-10-05T00:00:00+00:00", "2026-02-30T00:00:00Z"],
)
def test_invalid_utc_arguments_rejected_before_payload_access(
    flag: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments()
    arguments[arguments.index(flag) + 1] = value
    monkeypatch.setattr(
        workflow, "_train_production", lambda value: pytest.fail("production path reached")
    )
    with pytest.raises(SystemExit) as caught:
        workflow.main(arguments)
    assert caught.value.code == 2


class _ProductionError(CryptoAIError):
    """Mock the project-specific production error without real artifact access."""


def _install_mock_production(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_error: bool = False,
    fit_error: bool = False,
) -> dict[str, object]:
    observed: dict[str, object] = {}
    authorization = object()

    def load(path: Path) -> object:
        observed["authorization_path"] = path
        if load_error:
            raise _ProductionError("unapproved synthetic fixture")
        return authorization

    def train(request: object) -> object:
        observed["request"] = request
        if fit_error:
            raise _ProductionError("corrupted synthetic evaluation")
        return SimpleNamespace(files={"model.json": b"{}"}, manifest={})

    module = SimpleNamespace(
        ProductionError=_ProductionError,
        load_synthetic_authorization=load,
        SyntheticProductionRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        OfflineProductionEngine=lambda: SimpleNamespace(train=train),
    )
    import crypto_ai.phase2

    monkeypatch.setitem(sys.modules, "crypto_ai.phase2.production", module)
    monkeypatch.setattr(crypto_ai.phase2, "production", module, raising=False)
    observed["authorization"] = authorization
    return observed


def test_production_cli_passes_exact_explicit_arguments_to_verified_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed = _install_mock_production(monkeypatch)
    assert workflow.main(_arguments()) == 0
    request = observed["request"]
    assert request.evaluation_root == Path("/synthetic/evaluations")
    assert request.development_run_dir == Path("/synthetic/development")
    assert request.versions_root == Path("/synthetic/versions")
    assert request.evaluation_run_id == "synthetic-evaluation"
    assert request.model_version == "fixture-v1"
    assert request.authorization is observed["authorization"]
    assert observed["authorization_path"] == Path("/synthetic/authorization.json")
    assert request.training_as_of_utc == datetime(2026, 10, 5, tzinfo=UTC)
    assert request.created_at_utc == datetime(2026, 10, 5, 0, 1, tzinfo=UTC)
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == {
        "activated": False,
        "artifact_count": 1,
        "evaluation_run_id": "synthetic-evaluation",
        "model_version": "fixture-v1",
        "synthetic": True,
    }


@pytest.mark.parametrize("load_error", [False, True])
def test_production_cli_project_error_returns_one_without_traceback(
    load_error: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed = _install_mock_production(
        monkeypatch, load_error=load_error, fit_error=not load_error
    )
    assert workflow.main(_arguments()) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "Production fixture rejected:" in output.err
    assert "Traceback" not in output.err
    if load_error:
        assert "request" not in observed


def test_existing_evaluation_fixture_command_preserves_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request = object()
    monkeypatch.setattr(workflow, "_build_fixture", lambda root: request)

    def evaluate(actual: object) -> object:
        assert actual is request
        return SimpleNamespace(
            files={"metrics.json": b'{"production_decision":"NO-GO","research_verdict":"FAIL"}'}
        )

    monkeypatch.setattr(
        workflow.evaluation, "OfflineEvaluationEngine", lambda: SimpleNamespace(evaluate=evaluate)
    )
    assert workflow.main(["evaluate-synthetic-fixture"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_count": 1,
        "production_decision": "NO-GO",
        "research_verdict": "FAIL",
        "synthetic": True,
    }


def test_evaluation_fixture_still_rejects_caller_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "_build_fixture", lambda root: pytest.fail("fixture created"))
    with pytest.raises(SystemExit) as caught:
        workflow.main(["evaluate-synthetic-fixture", "--evaluation-root", "/not-readable"])
    assert caught.value.code == 2
