"""Mock-only checks for frozen dependency locks and clean local code provenance."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import dataset as dataset_module
from crypto_ai.phase2.dataset import (
    DatasetIntegrityError,
    SyntheticDatasetInput,
    SyntheticMarket,
)
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes

COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40
SOURCE_PATH = "src/crypto_ai/phase2/dataset.py"
FROZEN_LOCK_HASHES = {
    "requirements-lock.txt": "b17b32ea58d2a8baaed70fbb9ededd984639b3b52873d194af9d2051b54af210",
    "requirements-phase2.txt": "f57ef4ff913e39f002daefe2b717451d342a50bfc2d409b7e8118ff53b6eeafb",
}
REPOSITORY = Path(__file__).resolve().parents[2]
COMMIT_QUERY = ("rev-parse", "--verify", COMMIT + "^{commit}")
HEAD_QUERY = ("rev-parse", "HEAD")
STATUS_QUERY = ("status", "--porcelain=v1", "--untracked-files=all")


def _valid_git_responses() -> dict[tuple[str, ...], bytes]:
    return {
        COMMIT_QUERY: (COMMIT + "\n").encode(),
        HEAD_QUERY: (COMMIT + "\n").encode(),
        STATUS_QUERY: b"",
        ("show", f"{COMMIT}:{SOURCE_PATH}"): Path(dataset_module.__file__).read_bytes(),
        **{
            ("show", f"{COMMIT}:{name}"): (REPOSITORY / name).read_bytes()
            for name in FROZEN_LOCK_HASHES
        },
    }


def _mock_git(monkeypatch, responses):
    calls = []

    def git(*args):
        calls.append(args)
        if args not in responses:
            raise DatasetIntegrityError("mock local Git object is absent")
        return responses[args]

    monkeypatch.setattr(dataset_module, "_git", git)
    return calls


def test_frozen_lock_hashes_match_exact_unchanged_repository_bytes() -> None:
    assert dataset_module._LOCK_HASHES == FROZEN_LOCK_HASHES
    for name, expected in FROZEN_LOCK_HASHES.items():
        assert sha256_bytes((REPOSITORY / name).read_bytes()) == expected


def test_clean_implementation_commit_verifies_every_source_and_lock_blob(monkeypatch) -> None:
    responses = _valid_git_responses()
    calls = _mock_git(monkeypatch, responses)
    dataset_module._verify_code_provenance(COMMIT, require_clean=True)
    assert set(calls) == set(responses)


def test_historical_hydration_verifies_pins_without_requiring_current_head_clean(monkeypatch):
    responses = _valid_git_responses()
    del responses[HEAD_QUERY]
    del responses[STATUS_QUERY]
    calls = _mock_git(monkeypatch, responses)
    dataset_module._verify_code_provenance(COMMIT, require_clean=False)
    assert set(calls) == set(responses)
    assert HEAD_QUERY not in calls
    assert STATUS_QUERY not in calls


@pytest.mark.parametrize("resolved", [b"", (OTHER_COMMIT + "\n").encode(), b"not-a-commit\n"])
@pytest.mark.parametrize("require_clean", [False, True])
def test_wrong_or_unresolvable_commit_identity_fails_closed(monkeypatch, resolved, require_clean):
    responses = _valid_git_responses()
    responses[COMMIT_QUERY] = resolved
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=require_clean)


@pytest.mark.parametrize("head", [b"", (OTHER_COMMIT + "\n").encode()])
def test_fresh_publication_rejects_head_other_than_claimed_commit(monkeypatch, head):
    responses = _valid_git_responses()
    responses[HEAD_QUERY] = head
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=True)


@pytest.mark.parametrize(
    "status",
    [
        b" M src/crypto_ai/phase2/dataset.py\n",
        b"M  config/phase2_protocol.json\n",
        b"?? tests/phase2/new-fixture.py\n",
    ],
)
def test_fresh_publication_rejects_tracked_staged_and_untracked_dirt(monkeypatch, status):
    responses = _valid_git_responses()
    responses[STATUS_QUERY] = status
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=True)


@pytest.mark.parametrize("require_clean", [False, True])
@pytest.mark.parametrize("suffix", [b"\n", b"# modified implementation\n"])
def test_commit_source_must_match_executing_dataset_implementation(
    monkeypatch, require_clean, suffix
):
    responses = _valid_git_responses()
    source_query = ("show", f"{COMMIT}:{SOURCE_PATH}")
    responses[source_query] += suffix
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=require_clean)


@pytest.mark.parametrize("require_clean", [False, True])
@pytest.mark.parametrize("lock_name", tuple(FROZEN_LOCK_HASHES))
def test_commit_lock_bytes_must_match_frozen_hashes(monkeypatch, require_clean, lock_name):
    responses = _valid_git_responses()
    lock_query = ("show", f"{COMMIT}:{lock_name}")
    responses[lock_query] += b"\n"
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=require_clean)


@pytest.mark.parametrize("require_clean", [False, True])
@pytest.mark.parametrize(
    "missing_query",
    [
        COMMIT_QUERY,
        ("show", f"{COMMIT}:{SOURCE_PATH}"),
        *(("show", f"{COMMIT}:{name}") for name in FROZEN_LOCK_HASHES),
    ],
)
def test_missing_local_commit_source_or_lock_object_fails_closed(
    monkeypatch, require_clean, missing_query
):
    responses = _valid_git_responses()
    del responses[missing_query]
    _mock_git(monkeypatch, responses)
    with pytest.raises(CryptoAIError):
        dataset_module._verify_code_provenance(COMMIT, require_clean=require_clean)


def _request() -> SyntheticDatasetInput:
    return SyntheticDatasetInput(
        market=SyntheticMarket("2026-09-01T00:00:00Z", 48, seed=17),
        aggregation_id="0" * 64,
        protocol_bytes=canonicalize({"synthetic": True}),
        code_commit=COMMIT,
        dependency_lock_bytes=(REPOSITORY / "requirements-lock.txt").read_bytes(),
        phase2_dependency_lock_bytes=(REPOSITORY / "requirements-phase2.txt").read_bytes(),
    )


@pytest.mark.parametrize("field", ["dependency_lock_bytes", "phase2_dependency_lock_bytes"])
@pytest.mark.parametrize("tampered", [b"fabricated dependency pin", b"\n", b""])
def test_request_rejects_forged_lock_buffers_even_with_valid_clean_code(
    monkeypatch, field, tampered
):
    _mock_git(monkeypatch, _valid_git_responses())
    request = replace(_request(), **{field: tampered})
    with pytest.raises(CryptoAIError):
        dataset_module._validate_request(request)


def test_request_accepts_exact_frozen_lock_buffers_with_mocked_clean_code(monkeypatch):
    _mock_git(monkeypatch, _valid_git_responses())
    dataset_module._validate_request(_request())


def test_git_reader_disables_network_fetch_external_config_and_optional_writes(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/untrusted/repository")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"local evidence")

    monkeypatch.setattr(dataset_module.subprocess, "run", run)
    assert dataset_module._git("rev-parse", "HEAD") == b"local evidence"
    command, options = calls[0]
    assert command == [
        "git",
        "--no-replace-objects",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-C",
        str(REPOSITORY),
        "rev-parse",
        "HEAD",
    ]
    assert options["check"] is True
    assert options["capture_output"] is True
    assert options["timeout"] == 10
    env = options["env"]
    assert env["GIT_NO_LAZY_FETCH"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == dataset_module.os.devnull
    assert "GIT_DIR" not in env
    assert "GIT_CONFIG_COUNT" not in env


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git unavailable"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 10),
    ],
)
def test_git_reader_wraps_process_failures_in_project_exception(monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(dataset_module.subprocess, "run", run)
    with pytest.raises(DatasetIntegrityError):
        dataset_module._git("rev-parse", "HEAD")
