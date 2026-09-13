"""Explicitly armed, calendar-bounded live pilot; importing never opens a network.

The macOS scheduler starts independent workers, not a 24-hour collector. Each
worker has a process watchdog and an immutable invocation claim. Nothing here
launches a job until the human runs ``arm`` before the observation anchor.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import plistlib
import re
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.live_contracts import LiveAuthority
from crypto_ai.sentiment.live_deadline import hard_deadline
from crypto_ai.sentiment.network_budget import PilotBudget
from crypto_ai.sentiment.providers.gdelt_gsg import plan_retrieval
from crypto_ai.sentiment.storage import (
    ContentAddressedStore,
    _create_directory_path_without_symlinks,
    _ensure_directory_at,
    _open_directory_path,
    _read_regular_file_at_once,
    _write_fsynced_at,
)

DEFAULT_ROOT = "data/phase2-pilot-20260913"
ARM_DOMAIN = b"KrypX Batch B launch authority v1\n"
READY_DOMAIN = b"KrypX Batch B schedule ready v1\n"
ARM_PUBLICATION = "live-pilot-launch-authority-v1"
KEY_NAME = "pilot_ed25519.key"


def _utc(value: str) -> datetime:
    result = parse_utc_timestamp(value, field="pilot UTC")
    if result is None or format_utc_timestamp(result) != value:
        raise NetworkSafetyError("pilot UTC must be canonical")
    return result


def _json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise NetworkSafetyError("duplicate configuration key")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise NetworkSafetyError("non-finite configuration value")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
        if type(value) is not dict:
            raise NetworkSafetyError("configuration must be an object")
        canonicalize(value)
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise NetworkSafetyError("malformed pilot configuration") from exc


def _read(path: Path, *, maximum: int = 4_000_000, private: bool = False) -> bytes:
    descriptor = _open_directory_path(path.parent, description="pilot file parent")
    try:
        info = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum or info.st_nlink != 1:
            raise NetworkSafetyError("pilot file must be a bounded unique regular file")
        if private and (stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid()):
            raise NetworkSafetyError("pilot private key must be owned by this user with mode 0600")
        raw, opened = _read_regular_file_at_once(descriptor, path.name, description="pilot file")
        if len(raw) > maximum or opened.st_nlink != 1:
            raise NetworkSafetyError("pilot file changed its bound or link count")
        if private and (stat.S_IMODE(opened.st_mode) != 0o600 or opened.st_uid != os.getuid()):
            raise NetworkSafetyError("private key permissions changed")
        return raw
    finally:
        os.close(descriptor)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL, timeout=10
    ).strip()


def _pilot_root(repo: Path, relative: str) -> Path:
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise NetworkSafetyError("pilot directory must be an explicit ignored repository path")
    root = repo / relative
    if not relative.startswith("data/phase2-pilot-"):
        raise NetworkSafetyError("pilot directory must use the isolated Phase 2 data namespace")
    if not _git(repo, "check-ignore", "--", relative + "/probe"):
        raise NetworkSafetyError("pilot namespace must be excluded from Git")
    if _git(repo, "ls-files", "--", relative):
        raise NetworkSafetyError("pilot namespace must not contain tracked files")
    return root


def provision(repo: Path, relative: str) -> dict[str, str]:
    """Create a new isolated random signer once; never print or overwrite its secret."""
    root = _pilot_root(repo, relative)
    parent = _open_directory_path(root.parent, description="pilot data parent")
    try:
        os.mkdir(root.name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    directory = _open_directory_path(root, description="isolated pilot root")
    try:
        os.mkdir("secrets", mode=0o700, dir_fd=directory)
        secret = _ensure_directory_at(directory, "secrets", description="pilot signer directory")
        try:
            key = Ed25519PrivateKey.generate()
            _write_fsynced_at(secret, KEY_NAME, key.private_bytes_raw())
            os.fsync(secret)
        finally:
            os.close(secret)
        os.fsync(directory)
    finally:
        os.close(directory)
    public = key.public_key().public_bytes_raw()
    return {"public_key_hex": public.hex(), "signer_key_id": sha256_bytes(public)}


def load_setup(repo: Path) -> tuple[dict, dict, ContentAddressedStore, Ed25519PrivateKey]:
    config = _json(_read(repo / "config/phase2_protocol.json"))
    pilot = config["batch_b_live_pilot"]
    LiveAuthority.from_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", pilot["pilot_id"]):
        raise NetworkSafetyError("invalid pilot scheduler identity")
    approval = pilot["approval_record"]
    if (
        config["network_pilot_authorized"] is not True
        or config["milestone_3_scoring_authorized"] is not False
        or approval["network_pilot_authorized"] is not True
        or approval["real_provider_rights_approved"] is not True
        or approval["milestone_3_scoring_authorized"] is not False
        or approval["incidental_raw_fields_retention_approved"] is not True
        or approval["immutable_retention_through_review"] is not True
        or approval["external_redistribution_authorized"] is not False
        or approval["incremental_cost_usd"] != 0
    ):
        raise NetworkSafetyError(
            "live pilot requires explicit matching rights and network authority"
        )
    for field in ("anchor_utc", "end_exclusive_utc", "closeout_deadline_utc"):
        if approval[field] != pilot[field]:
            raise NetworkSafetyError("approval and pilot schedule disagree")
    start, end = _utc(pilot["anchor_utc"]), _utc(pilot["end_exclusive_utc"])
    if end - start != timedelta(days=1) or _utc(pilot["closeout_deadline_utc"]) != end + timedelta(
        minutes=45
    ):
        raise NetworkSafetyError("live pilot requires the exact 24-hour window and 45-minute tail")
    if _utc(approval["approved_at_utc"]) >= start:
        raise NetworkSafetyError("human approval must precede the anchor")
    root = _pilot_root(repo, pilot["store_path"])
    descriptor = _open_directory_path(root, description="pilot root")
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise NetworkSafetyError("pilot root must be owned by this user with mode 0700")
    finally:
        os.close(descriptor)
    if pilot["private_key_path"] != "secrets/" + KEY_NAME:
        raise NetworkSafetyError("unexpected pilot key location")
    raw = _read(root / pilot["private_key_path"], maximum=32, private=True)
    if len(raw) != 32:
        raise NetworkSafetyError("pilot key must contain exactly 32 Ed25519 private bytes")
    key = Ed25519PrivateKey.from_private_bytes(raw)
    public = key.public_key().public_bytes_raw()
    if public.hex() != pilot["public_key_hex"] or sha256_bytes(public) != pilot["signer_key_id"]:
        raise NetworkSafetyError("pilot signer does not match the governance pin")
    return config, pilot, ContentAddressedStore(root), key


def _require_clean_main(repo: Path) -> str:
    if _git(repo, "branch", "--show-current") != "main" or _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise NetworkSafetyError("live execution requires the committed, clean main checkout")
    return _git(repo, "rev-parse", "HEAD")


def _launch_body(config: dict, pilot: dict, commit: str, when: datetime) -> dict:
    return {
        "schema_version": "batch-b-live-launch-v1",
        "pilot_id": pilot["pilot_id"],
        "specification_id": pilot["specification_id"],
        "protocol_sha256": canonical_sha256(config),
        "approval_sha256": canonical_sha256(pilot["approval_record"]),
        "code_commit": commit,
        "plan_sha256": plan_retrieval(pilot["anchor_utc"], pilot["end_exclusive_utc"]).plan_id,
        "signer_key_id": pilot["signer_key_id"],
        "armed_at_utc": format_utc_timestamp(when),
    }


def _sign_launch(body: dict, key: Ed25519PrivateKey, *, domain: bytes = ARM_DOMAIN) -> dict:
    return {
        "body": body,
        "body_sha256": canonical_sha256(body),
        "signature": base64.b64encode(key.sign(domain + canonicalize(body))).decode("ascii"),
    }


def _verify_launch(config: dict, pilot: dict, store: ContentAddressedStore, key, commit: str):
    publication = store.read_publication(ARM_PUBLICATION)
    if set(publication.files) != {"authority.json"}:
        raise NetworkSafetyError("unexpected launch authority payload")
    value = _json(publication.files["authority.json"])
    if set(value) != {"body", "body_sha256", "signature"}:
        raise NetworkSafetyError("invalid launch authority envelope")
    when = _utc(value["body"]["armed_at_utc"])
    expected = _launch_body(config, pilot, commit, when)
    if (
        when >= _utc(pilot["anchor_utc"])
        or value["body"] != expected
        or value["body_sha256"] != canonical_sha256(expected)
        or canonicalize(value) != publication.files["authority.json"]
        or publication.manifest["metadata"] != {"authority_sha256": canonical_sha256(value)}
    ):
        raise NetworkSafetyError("launch authority differs from the frozen checkout or approval")
    try:
        signature = base64.b64decode(value["signature"], validate=True)
        key.public_key().verify(signature, ARM_DOMAIN + canonicalize(expected))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise NetworkSafetyError("launch authority signature is invalid") from exc
    return expected


def calendar_jobs(repo: Path, pilot: dict, python: str) -> list[tuple[str, bytes]]:
    """Absolute local calendar triggers; worker guards additionally enforce UTC/year."""
    jobs = []
    if not isinstance(pilot["pilot_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", pilot["pilot_id"]
    ):
        raise NetworkSafetyError("invalid pilot scheduler identity")
    for index in range(96):
        start = _utc(pilot["anchor_utc"]) + timedelta(minutes=45 + index * 15)
        local = start.astimezone()
        label = f"org.krypx.{pilot['pilot_id']}.slot-{index:02d}"
        payload = {
            "Label": label,
            "ProgramArguments": [
                python,
                str(repo / "scripts/run_phase2_pilot.py"),
                "worker",
                "--slot",
                str(index),
            ],
            "WorkingDirectory": str(repo),
            "StartCalendarInterval": {
                "Month": local.month,
                "Day": local.day,
                "Hour": local.hour,
                "Minute": local.minute,
            },
            "ProcessType": "Background",
            "RunAtLoad": False,
            "KeepAlive": False,
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "/dev/null",
            "EnvironmentVariables": {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        }
        jobs.append((label, plistlib.dumps(payload, sort_keys=True)))
    return jobs


def arm(repo: Path, *, now: datetime | None = None) -> dict:
    config, pilot, store, key = load_setup(repo)
    when = now or datetime.now(UTC)
    if when >= _utc(pilot["anchor_utc"]):
        raise NetworkSafetyError("anchor elapsed: no arming, backfill or automatic date shift")
    if sys.platform != "darwin":
        raise NetworkSafetyError("automatic calendar scheduling requires macOS launchd")
    commit = _require_clean_main(repo)
    budget = PilotBudget(store, pilot["pilot_id"], canonical_sha256(config), create=True)
    # Creation is deliberately one-shot. A failed partial arm requires review,
    # never a second arm which resets counters or silently replaces evidence.
    with budget.locked():
        body = _launch_body(config, pilot, commit, when)
        envelope = _sign_launch(body, key)
        budget.assert_storage_capacity(2_000_000)
        store.publish_bundle(
            ARM_PUBLICATION,
            {"authority.json": canonicalize(envelope)},
            metadata={"authority_sha256": canonical_sha256(envelope)},
        )
        descriptor = _create_directory_path_without_symlinks(
            store.root / "control", description="pilot scheduler control"
        )
        try:
            jobs = calendar_jobs(repo, pilot, sys.executable)
            for label, payload in jobs:
                _write_fsynced_at(descriptor, label + ".plist", payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        for label, _ in jobs:
            if datetime.now(UTC) >= _utc(pilot["anchor_utc"]):
                budget.stop("arming_exceeded_anchor")
                raise NetworkSafetyError("calendar setup did not complete before the anchor")
            result = subprocess.run(
                [
                    "/bin/launchctl",
                    "bootstrap",
                    f"gui/{os.getuid()}",
                    str(store.root / "control" / (label + ".plist")),
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if result.returncode:
                budget.stop("scheduler_install_failed")
                raise NetworkSafetyError(
                    "calendar installation failed; pilot halted, no automatic retry"
                )
        budget.assert_storage_capacity(0)
        ready_at = datetime.now(UTC)
        if ready_at >= _utc(pilot["anchor_utc"]):
            budget.stop("arming_exceeded_anchor")
            raise NetworkSafetyError("calendar readiness did not precede the anchor")
        ready_envelope = _sign_launch(
            {
                "schema_version": "batch-b-live-schedule-ready-v1",
                "authority_sha256": canonical_sha256(envelope),
                "jobs": 96,
                "ready_at_utc": format_utc_timestamp(ready_at),
            },
            key,
            domain=READY_DOMAIN,
        )
        store.publish_bundle(
            "live-pilot-schedule-ready-v1",
            {"ready.json": canonicalize(ready_envelope)},
            metadata={
                "pilot_id": pilot["pilot_id"],
                "ready_sha256": canonical_sha256(ready_envelope),
            },
        )
        if datetime.now(UTC) >= _utc(pilot["anchor_utc"]):
            budget.stop("readiness_publication_exceeded_anchor")
            raise NetworkSafetyError("durable scheduler readiness missed the anchor")
    return {"status": "ARMED", "jobs": 96, "launch_authority_sha256": canonical_sha256(envelope)}


def worker(repo: Path, index: int) -> dict:
    if type(index) is not int or not 0 <= index < 96:
        raise NetworkSafetyError("worker slot must be an integer from zero through 95")
    config, pilot, store, key = load_setup(repo)
    now = datetime.now(UTC)
    slot_start = _utc(pilot["anchor_utc"]) + timedelta(minutes=45 + index * 15)
    slot_end = slot_start + timedelta(minutes=15)
    if not slot_start <= now < slot_end:
        raise NetworkSafetyError("worker is outside its fixed UTC slot; no catch-up")
    with hard_deadline(min(900, (slot_end - now).total_seconds())):
        return _worker_run(repo, index, config, pilot, store, key)


def _worker_run(repo: Path, index: int, config: dict, pilot: dict, store, key) -> dict:
    # Deferred imports keep provisioning/help entirely independent of live I/O.
    from crypto_ai.sentiment.live_transport import RealGSGTransport
    from crypto_ai.sentiment.network import (
        GSGNetworkClient,
        RetrievalFailure,
        verify_retained_closeout,
    )

    commit = _require_clean_main(repo)
    launch = _verify_launch(config, pilot, store, key, commit)
    ready = store.read_publication("live-pilot-schedule-ready-v1")
    ready_value = _json(ready.files.get("ready.json", b"{}"))["body"]
    ready_at = _utc(ready_value["ready_at_utc"])
    expected_ready = {
        "schema_version": "batch-b-live-schedule-ready-v1",
        "authority_sha256": canonical_sha256(_sign_launch(launch, key)),
        "jobs": 96,
        "ready_at_utc": format_utc_timestamp(ready_at),
    }
    expected_ready_envelope = _sign_launch(expected_ready, key, domain=READY_DOMAIN)
    if (
        ready.files != {"ready.json": canonicalize(expected_ready_envelope)}
        or ready.manifest["metadata"]
        != {
            "pilot_id": pilot["pilot_id"],
            "ready_sha256": canonical_sha256(expected_ready_envelope),
        }
        or not _utc(launch["armed_at_utc"]) <= ready_at < _utc(pilot["anchor_utc"])
    ):
        raise NetworkSafetyError("complete pre-anchor schedule installation is not established")
    plan = plan_retrieval(pilot["anchor_utc"], pilot["end_exclusive_utc"])
    authority = LiveAuthority.from_config(config)
    client = GSGNetworkClient(
        store,
        pilot_id=pilot["pilot_id"],
        specification_id=pilot["specification_id"],
        protocol_sha256=canonical_sha256(config),
        code_commit=commit,
        plan=plan,
        transport=RealGSGTransport(plan),
        private_key=key,
        public_key=key.public_key(),
        clock=lambda: datetime.now(UTC),
        monotonic=time.monotonic,
        sleep=time.sleep,
        real_network_calls_prohibited=False,
        live_authority=authority,
    )
    with client:
        # The immutable publication makes invocation one-shot even if a previous
        # process ended between two completed requests, leaving no pending intent.
        client.budget.assert_storage_capacity(131_072)
        if (store.publications_root / f"live-worker-claim-{index:02d}").exists():
            raise NetworkSafetyError(
                "worker invocation already claimed; automatic relaunch prohibited"
            )
        store.publish_bundle(
            f"live-worker-claim-{index:02d}",
            {
                "claim.json": canonicalize(
                    {"slot": index, "started_at_utc": format_utc_timestamp(datetime.now(UTC))}
                )
            },
            metadata={"launch_authority_sha256": canonical_sha256(_sign_launch(launch, key))},
        )
        terminal_failure = False
        try:
            for minute in plan.intervals[index * 15 : (index + 1) * 15]:
                client.retrieve(minute.filename_timestamp)
        except RetrievalFailure:
            terminal_failure = True
        if index == 95 or terminal_failure:
            closeout = client.closeout()
            verify_retained_closeout(
                client.budget,
                public_key=key.public_key(),
                receipts=client.receipts,
                expected_closeout_sha256=canonical_sha256(closeout),
                expected_plan_sha256=plan.plan_id,
                live_authority=authority,
            )
            return {
                "status": (
                    "HALTED_CLOSED_OUT_NOT_ACCEPTED"
                    if terminal_failure
                    else "CLOSED_OUT_NOT_YET_ACCEPTED"
                ),
                "closeout_sha256": canonical_sha256(closeout),
            }
        return {"status": "WORKER_FINISHED", "slot": index, "receipt_count": len(client.receipts)}


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).absolute().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    provision_parser = sub.add_parser("provision", help="create isolated signer once; no network")
    provision_parser.add_argument("--root", default=DEFAULT_ROOT)
    sub.add_parser("check", help="validate local pins/configuration; no network or scheduling")
    sub.add_parser("arm", help="before anchor, install 96 bounded live calendar jobs")
    slot_parser = sub.add_parser("worker", help="run one pre-armed fixed slot; real HTTPS")
    slot_parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        with hard_deadline(900):
            if args.command == "provision":
                result = provision(repo, args.root)
            elif args.command == "check":
                config, pilot, store, _ = load_setup(repo)
                commit = _require_clean_main(repo)
                if datetime.now(UTC) >= _utc(pilot["anchor_utc"]):
                    raise NetworkSafetyError("anchor elapsed; launch readiness cannot be certified")
                if (store.publications_root / ARM_PUBLICATION).exists():
                    raise NetworkSafetyError("pilot already armed; do not arm again")
                if sys.platform != "darwin":
                    raise NetworkSafetyError("automatic calendar scheduling requires macOS")
                subprocess.run(
                    ["/bin/launchctl", "print", f"gui/{os.getuid()}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=10,
                )
                result = {
                    "status": "READY_TO_ARM_NOT_ARMED",
                    "code_commit": commit,
                    "anchor_utc": pilot["anchor_utc"],
                    "closeout_deadline_utc": pilot["closeout_deadline_utc"],
                    "protocol_sha256": canonical_sha256(config),
                    "signer_key_id": pilot["signer_key_id"],
                }
            elif args.command == "arm":
                result = arm(repo)
            else:
                result = worker(repo, args.slot)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (CryptoAIError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # Provider messages, response headers and secret paths are not printed.
        print(
            "Pilot stopped: local authority, schedule, integrity or safety validation failed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
