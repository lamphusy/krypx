"""Live authority, retention and runner acceptance exercised without any network."""

import base64
import json
import os
import plistlib
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment import pilot_runner as runner
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.exceptions import NetworkSafetyError, ReceiptValidationError
from crypto_ai.sentiment.live_contracts import LIVE_GAP_SCHEMA, LIVE_RECEIPT_SCHEMA, LiveAuthority
from crypto_ai.sentiment.live_transport import RealGSGTransport
from crypto_ai.sentiment.network import (
    GSGNetworkClient,
    RetrievalFailure,
    verify_retained_closeout,
    verify_terminal_gap_evidence,
)
from crypto_ai.sentiment.providers.gdelt_gsg import plan_retrieval
from crypto_ai.sentiment.receipts import RECEIPT_DOMAIN, verify_receipt, verify_receipt_chain
from crypto_ai.sentiment.storage import ContentAddressedStore

from .test_gdelt_gsg_network import KEY, PUBLIC, RAW, Clock, Response, Transport


def configuration():
    config = json.loads((Path(__file__).parents[2] / "config/phase2_protocol.json").read_bytes())
    config["batch_b_live_pilot"]["public_key_hex"] = PUBLIC.public_bytes_raw().hex()
    config["batch_b_live_pilot"]["signer_key_id"] = sha256_bytes(PUBLIC.public_bytes_raw())
    # Synthetic tests retain their fixed midnight plan independently of a human
    # changing the actual future operational schedule in governance.
    for field, value in {
        "anchor_utc": "2026-09-13T00:00:00Z",
        "end_exclusive_utc": "2026-09-14T00:00:00Z",
        "closeout_deadline_utc": "2026-09-14T00:45:00Z",
    }.items():
        config["batch_b_live_pilot"][field] = value
        config["batch_b_live_pilot"]["approval_record"][field] = value
    config["batch_b_live_pilot"]["approval_record"][
        "window_authorized_at_utc"
    ] = "2026-09-12T17:46:20Z"
    return config


def live_client(tmp_path, *responses, clock=None, create=True):
    config = configuration()
    pilot = config["batch_b_live_pilot"]
    authority = LiveAuthority.from_config(config)
    plan = plan_retrieval(pilot["anchor_utc"], pilot["end_exclusive_utc"])
    clock = clock or Clock()
    clock.start = datetime(2026, 9, 13, 0, 45, tzinfo=UTC)
    fake = Transport(clock, *responses)
    transport = RealGSGTransport(plan)
    transport.open = fake.open  # Explicitly mocked before any client execution.
    store = ContentAddressedStore(tmp_path)
    client = GSGNetworkClient(
        store,
        pilot_id=pilot["pilot_id"],
        specification_id=pilot["specification_id"],
        protocol_sha256=canonical_sha256(config),
        code_commit="b" * 40,
        plan=plan,
        transport=transport,
        private_key=KEY,
        public_key=PUBLIC,
        clock=clock.utc,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        create_pilot=create,
        real_network_calls_prohibited=False,
        live_authority=authority,
    )
    return client, clock, fake, authority, plan


def test_live_snapshots_are_not_fixtures_and_closeout_replays(tmp_path):
    client, _, transport, authority, plan = live_client(tmp_path, *(Response() for _ in range(15)))
    with client:
        for interval in plan.intervals[:15]:
            result = client.retrieve(interval.filename_timestamp)
            assert result.snapshot.receipt.input_class == "provider_response"
            assert client.store.get_bytes(result.snapshot.receipt.raw_snapshot_sha256) == RAW
        assert all(b - a >= 5 for a, b in zip(transport.starts, transport.starts[1:], strict=False))
        envelope = client.receipts[0]
        assert envelope["schema_version"] == LIVE_RECEIPT_SCHEMA
        body = verify_receipt(envelope, PUBLIC, live_authority=authority)
        assert body["input_class"] == "provider_response"
        assert body["real_network_calls_prohibited"] is False
        assert body["live_evidence"]["authority_sha256"] == authority.sha256
        with pytest.raises(ReceiptValidationError):
            verify_receipt(envelope, PUBLIC)
        closeout = client.closeout()
        result = verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=plan.plan_id,
            live_authority=authority,
        )
        assert result["slot_outcomes"][0]["outcome"] == "verified"
        assert result["live_authority_sha256"] == authority.sha256


@pytest.mark.parametrize(
    "response",
    [
        lambda: Response(status=404),
        lambda: Response(headers={"Content-Length": str(len(RAW) + 10)}),
    ],
)
def test_live_terminal_gap_early_closeout_is_signed_and_replayable(tmp_path, response):
    client, _, transport, authority, plan = live_client(tmp_path, response())
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(plan.intervals[0].filename_timestamp)
        gap = failure.value.result.gap_evidence
        assert gap.version == LIVE_GAP_SCHEMA and gap.input_class == "provider_response"
        assert gap.network_access_authorized is True
        bindings = dict(
            plan=plan,
            filename_timestamp=plan.intervals[0].filename_timestamp,
            receipts=client.receipts,
            public_key=PUBLIC,
            protocol_sha256=client.context["protocol_sha256"],
            live_authority=authority,
        )
        verify_terminal_gap_evidence(gap, **bindings)
        with pytest.raises(NetworkSafetyError):
            verify_terminal_gap_evidence(replace(gap, input_class="synthetic_fixture"), **bindings)
        closeout = client.closeout()
        verified = verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=plan.plan_id,
            live_authority=authority,
        )
        assert verified["slot_outcomes"][0]["outcome"] == "provider_gap"
        assert all(s["outcome"] != "verified" for s in verified["slot_outcomes"])
        transport.open.assert_called_once()


def test_live_restart_keeps_budget_pacing_and_signer(tmp_path):
    first, clock, transport1, _, plan = live_client(tmp_path, Response())
    with first:
        first.retrieve(plan.intervals[0].filename_timestamp)
    second, _, transport2, authority, _ = live_client(
        tmp_path, Response(), clock=clock, create=False
    )
    with second:
        second.retrieve(plan.intervals[1].filename_timestamp)
        bodies = verify_receipt_chain(second.receipts, PUBLIC, live_authority=authority)
        assert bodies[-1]["live_evidence"]["cumulative_download_bytes"] == 2 * len(RAW)
        assert transport2.starts[0] - transport1.starts[0] >= 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved", False),
        ("network_pilot_authorized", False),
        ("real_provider_rights_approved", False),
        ("incidental_raw_fields_retention_approved", False),
        ("immutable_retention_through_review", False),
        ("external_redistribution_authorized", True),
        ("milestone_3_scoring_authorized", True),
        ("publisher_scraping_authorized", True),
        ("body_text_collection_authorized", True),
        ("provider", "other"),
        ("scope", "all_news"),
        ("attribution", "none"),
        ("incremental_cost_usd", True),
        ("incremental_cost_usd", 0.01),
        ("approved_at_utc", "2099-01-01T00:00:00Z"),
        ("window_authorized_at_utc", "2099-01-01T00:00:00Z"),
    ],
)
def test_live_authority_rejects_missing_or_expanded_approval(field, value):
    config = configuration()
    config["batch_b_live_pilot"]["approval_record"][field] = value
    with pytest.raises(CryptoAIError):
        LiveAuthority.from_config(config)


@pytest.mark.parametrize("mutation", ["rights_hash", "count", "future", "relabel"])
def test_validly_resigned_live_forgeries_fail(tmp_path, mutation):
    client, _, _, authority, plan = live_client(tmp_path, Response())
    with client:
        envelope = deepcopy(client.retrieve(plan.intervals[0].filename_timestamp).receipts[0])
    body = envelope["body"]
    if mutation == "rights_hash":
        body["live_evidence"]["approval_sha256"] = "f" * 64
    elif mutation == "count":
        body["live_evidence"]["cumulative_download_bytes"] += 1
    elif mutation == "relabel":
        body["input_class"] = "synthetic_fixture"
    else:
        for field in (
            "requested_at_utc",
            "dispatch_confirmed_at_utc",
            "completed_at_utc",
            "raw_published_at_utc",
        ):
            body[field] = "2099-01-01T00:00:00Z"
    envelope["body_sha256"] = canonical_sha256(body)
    envelope["signature"] = base64.b64encode(KEY.sign(RECEIPT_DOMAIN + canonicalize(body))).decode()
    PUBLIC.verify(base64.b64decode(envelope["signature"]), RECEIPT_DOMAIN + canonicalize(body))
    with pytest.raises(CryptoAIError):
        verify_receipt_chain([envelope], PUBLIC, live_authority=authority)


@pytest.mark.parametrize("bad", [None, True, "253", [], {}])
def test_malformed_live_byte_counts_fail_with_project_exception(tmp_path, bad):
    client, _, _, authority, plan = live_client(tmp_path, Response())
    with client:
        envelope = deepcopy(client.retrieve(plan.intervals[0].filename_timestamp).receipts[0])
    envelope["body"]["bytes_received"] = bad
    with pytest.raises(CryptoAIError):
        verify_receipt(envelope, PUBLIC, live_authority=authority)


def test_real_replacement_schedule_is_exact_and_authority_bound():
    config = json.loads((Path(__file__).parents[2] / "config/phase2_protocol.json").read_bytes())
    pilot = config["batch_b_live_pilot"]
    authority = LiveAuthority.from_config(config)
    assert authority.value["approval_record"]["anchor_utc"] == "2026-09-13T14:00:00Z"
    assert pilot["end_exclusive_utc"] == "2026-09-14T14:00:00Z"
    assert pilot["closeout_deadline_utc"] == "2026-09-14T14:45:00Z"
    assert len(runner.calendar_jobs(Path("/example"), pilot, "/python")) == 96


def test_runner_calendar_has_96_absolute_nonoverlapping_workers():
    pilot = configuration()["batch_b_live_pilot"]
    jobs = runner.calendar_jobs(Path("/example/repo"), pilot, "/example/python")
    assert len(jobs) == len(set(label for label, _ in jobs)) == 96
    for index, (_, encoded) in enumerate(jobs):
        job = plistlib.loads(encoded)
        when = (
            datetime(2026, 9, 13, 0, 45, tzinfo=UTC) + timedelta(minutes=index * 15)
        ).astimezone()
        assert job["StartCalendarInterval"] == {
            "Month": when.month,
            "Day": when.day,
            "Hour": when.hour,
            "Minute": when.minute,
        }
        assert job["ProgramArguments"][-2:] == ["--slot", str(index)]
        assert job["RunAtLoad"] is False and job["KeepAlive"] is False
        assert job["StandardOutPath"] == job["StandardErrorPath"] == "/dev/null"


@pytest.mark.parametrize("bad", ["../../escape", "x/y", "", "x" * 97])
def test_runner_rejects_unsafe_scheduler_names(bad):
    pilot = configuration()["batch_b_live_pilot"]
    pilot["pilot_id"] = bad
    with pytest.raises(NetworkSafetyError):
        runner.calendar_jobs(Path("/example"), pilot, "python")


def test_runner_provisions_private_key_once_outside_git(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        runner, "_git", lambda repo, *args: "data/" if args[0] == "check-ignore" else ""
    )
    public = runner.provision(tmp_path, runner.DEFAULT_ROOT)
    root = tmp_path / runner.DEFAULT_ROOT
    assert root.stat().st_mode & 0o777 == 0o700
    path = root / "secrets/pilot_ed25519.key"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_size == 32
    assert set(public) == {"public_key_hex", "signer_key_id"}
    assert sha256_bytes(bytes.fromhex(public["public_key_hex"])) == public["signer_key_id"]
    with pytest.raises(FileExistsError):
        runner.provision(tmp_path, runner.DEFAULT_ROOT)
    path.chmod(0o644)
    with pytest.raises(NetworkSafetyError):
        runner._read(path, private=True)


def test_runner_key_reader_rejects_fifo_without_opening(tmp_path):
    path = tmp_path / "key"
    os.mkfifo(path)
    with pytest.raises(NetworkSafetyError):
        runner._read(path, private=True)


@pytest.mark.parametrize("mutation", ["config", "code", "signature", "late"])
def test_launch_manifest_pins_preanchor_config_code_and_key(tmp_path, mutation):
    config = configuration()
    pilot = config["batch_b_live_pilot"]
    when = datetime(2026, 9, 12, tzinfo=UTC)
    if mutation == "late":
        when += timedelta(days=2)
    body = runner._launch_body(config, pilot, "a" * 40, when)
    envelope = runner._sign_launch(body, KEY)
    if mutation == "signature":
        envelope["signature"] = base64.b64encode(b"x" * 64).decode()
    store = ContentAddressedStore(tmp_path)
    store.publish_bundle(
        runner.ARM_PUBLICATION,
        {"authority.json": canonicalize(envelope)},
        metadata={"authority_sha256": canonical_sha256(envelope)},
    )
    if mutation == "config":
        config["network_pilot_authorized"] = False
    with pytest.raises(NetworkSafetyError):
        runner._verify_launch(
            config, pilot, store, KEY, "b" * 40 if mutation == "code" else "a" * 40
        )


def test_runner_arm_rejects_elapsed_anchor_before_schedule_calls(tmp_path, monkeypatch):
    config = configuration()
    pilot = config["batch_b_live_pilot"]
    monkeypatch.setattr(runner, "load_setup", lambda repo: (config, pilot, None, KEY))
    launch = Mock(side_effect=AssertionError("scheduler must not run"))
    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises(NetworkSafetyError, match="anchor elapsed"):
        runner.arm(tmp_path, now=datetime(2026, 9, 13, tzinfo=UTC))
    launch.assert_not_called()


@pytest.mark.parametrize(
    "branch,status", [("other", ""), ("main", " M config/phase2_protocol.json")]
)
def test_runner_requires_clean_main(tmp_path, monkeypatch, branch, status):
    monkeypatch.setattr(
        runner, "_git", lambda repo, *args: branch if args[0] == "branch" else status
    )
    with pytest.raises(NetworkSafetyError):
        runner._require_clean_main(tmp_path)


@pytest.mark.parametrize("terminal", [False, True])
def test_complete_runner_arming_and_first_worker_are_offline(tmp_path, monkeypatch, terminal):
    config = configuration()
    pilot = config["batch_b_live_pilot"]
    (tmp_path / "data").mkdir()

    def git(repo, *args):
        if args[0] == "check-ignore":
            return "data/"
        if args[0] == "branch":
            return "main"
        if args[0] == "rev-parse":
            return "b" * 40
        return ""

    monkeypatch.setattr(runner, "_git", git)
    pins = runner.provision(tmp_path, runner.DEFAULT_ROOT)
    # Provision a synthetic key in this temporary test namespace; never the operational key.
    pilot.update(pins)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/phase2_protocol.json").write_bytes(canonicalize(config))
    clock = Clock()
    clock.start = datetime(2026, 9, 12, 17, tzinfo=UTC)

    class FakeDateTime:
        @staticmethod
        def now(tz):
            return clock.utc()

    monkeypatch.setattr(runner, "datetime", FakeDateTime)
    monkeypatch.setattr(runner, "hard_deadline", lambda seconds: nullcontext())
    monkeypatch.setattr(runner.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runner.time, "sleep", clock.sleep)
    launch = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(runner.subprocess, "run", launch)
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    armed = runner.arm(tmp_path)
    assert armed["status"] == "ARMED" and launch.call_count == 96
    for call in launch.call_args_list:
        assert call.args[0][:2] == ["/bin/launchctl", "bootstrap"]
    clock.start = datetime(2026, 9, 13, 0, 45, tzinfo=UTC)
    responses = [Response(status=404)] if terminal else [Response() for _ in range(15)]
    fake = Transport(clock, *responses)
    monkeypatch.setattr(
        RealGSGTransport, "open", lambda self, url, *, timeout: fake.open(url, timeout=timeout)
    )
    result = runner.worker(tmp_path, 0)
    assert result["status"] == ("HALTED_CLOSED_OUT_NOT_ACCEPTED" if terminal else "WORKER_FINISHED")
    assert fake.open.call_count == (1 if terminal else 15)
    with pytest.raises(NetworkSafetyError):
        runner.worker(tmp_path, 0)
    assert fake.open.call_count == (1 if terminal else 15)
