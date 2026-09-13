"""Explicit, independently pinned live authority; never an upgrade of fixture evidence."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.providers.gdelt_gsg import plan_retrieval

LIVE_RECEIPT_SCHEMA = "batch-b-live-signed-receipt-v1"
LIVE_CLOSEOUT_SCHEMA = "batch-b-live-signed-closeout-v1"
LIVE_GAP_SCHEMA = "batch-b-live-terminal-gap-evidence-v1"
CAPS = {
    "download_bytes": 500_000_000,
    "storage_bytes": 2_000_000_000,
    "session_seconds": 900,
    "request_spacing_seconds": 5,
    "attempts": 4,
    "cost_usd": 0,
}
PARSER = {
    "provider": "gdelt_gsg",
    "version": "gdelt-gsg-jsonl-v1",
    "policy": "gdelt-gsg-parser-policy-v1",
    "compressed_bytes": 67_108_864,
    "decompressed_bytes": 268_435_456,
    "json_lines": 1_000_000,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NetworkSafetyError(message)


@dataclass(frozen=True, slots=True)
class LiveAuthority:
    """Canonical immutable authority bytes derived from the final governance record."""

    encoded: bytes

    def __post_init__(self) -> None:
        try:
            value = json.loads(self.encoded)
            _require(
                type(self.encoded) is bytes and canonicalize(value) == self.encoded,
                "live authority must be exact canonical bytes",
            )
            _require(
                set(value)
                == {
                    "pilot_id",
                    "specification_id",
                    "protocol_sha256",
                    "plan_sha256",
                    "public_key_hex",
                    "signer_key_id",
                    "approval_record",
                    "caps",
                    "parser",
                },
                "unknown live authority fields",
            )
            for field in ("pilot_id", "specification_id"):
                _require(
                    isinstance(value[field], str)
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value[field]) is not None,
                    "invalid live authority identity",
                )
            for field in ("protocol_sha256", "plan_sha256", "public_key_hex", "signer_key_id"):
                _require(
                    isinstance(value[field], str)
                    and re.fullmatch(r"[0-9a-f]{64}", value[field]) is not None,
                    "invalid live authority hash or public key",
                )
            _require(
                sha256_bytes(bytes.fromhex(value["public_key_hex"])) == value["signer_key_id"],
                "live signer pin mismatch",
            )
            approval = value["approval_record"]
            for field in (
                "approved",
                "network_pilot_authorized",
                "real_provider_rights_approved",
                "incidental_raw_fields_retention_approved",
                "immutable_retention_through_review",
                "disposal_requires_separate_authorization",
            ):
                _require(approval[field] is True, "live rights approval is missing")
            for field in (
                "milestone_3_scoring_authorized",
                "external_redistribution_authorized",
                "publisher_scraping_authorized",
                "body_text_collection_authorized",
            ):
                _require(approval[field] is False, "live authority exceeds the permitted scope")
            _require(
                approval["provider"] == "gdelt_gsg"
                and approval["scope"] == "gdelt_gsg_english_btc_titles"
                and approval["attribution"] == "GDELT",
                "wrong live provider or rights scope",
            )
            _require(
                approval["permitted_url_template"]
                == "https://data.gdeltproject.org/gdeltv3/gsg/{timestamp}.gsg.json.gz",
                "wrong live endpoint",
            )
            _require(
                type(approval["incremental_cost_usd"]) in (int, float)
                and approval["incremental_cost_usd"] == 0,
                "unknown or positive live cost",
            )
            start = parse_utc_timestamp(approval["anchor_utc"], field="anchor")
            approved = parse_utc_timestamp(approval["approved_at_utc"], field="approval")
            window_approved = parse_utc_timestamp(
                approval["window_authorized_at_utc"], field="window approval"
            )
            _require(
                start is not None
                and approved is not None
                and window_approved is not None
                and approved < start
                and window_approved < start,
                "live authority must precede anchor",
            )
            _require(
                format_utc_timestamp(start) == approval["anchor_utc"]
                and format_utc_timestamp(approved) == approval["approved_at_utc"]
                and format_utc_timestamp(window_approved) == approval["window_authorized_at_utc"],
                "noncanonical live UTC",
            )
            plan = plan_retrieval(approval["anchor_utc"], approval["end_exclusive_utc"])
            end = parse_utc_timestamp(approval["end_exclusive_utc"], field="end")
            _require(
                end is not None
                and format_utc_timestamp(end + timedelta(minutes=45))
                == approval["closeout_deadline_utc"],
                "live closeout deadline differs from the plan",
            )
            _require(
                len(plan.intervals) == 1440 and plan.plan_id == value["plan_sha256"],
                "live authority plan mismatch",
            )
            _require(
                canonicalize(value["caps"]) == canonicalize(CAPS) and value["parser"] == PARSER,
                "live authority changes frozen caps or parser",
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise NetworkSafetyError("malformed live authority") from exc

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LiveAuthority:
        try:
            pilot = config["batch_b_live_pilot"]
            _require(
                config["network_pilot_authorized"] is True
                and config["milestone_3_scoring_authorized"] is False
                and pilot["network_pilot_authorized"] is True
                and pilot["real_provider_rights_approved"] is True
                and pilot["milestone_3_scoring_authorized"] is False,
                "live execution is not authorized",
            )
            for field, expected in {
                "maximum_download_bytes": 500_000_000,
                "maximum_storage_bytes": 2_000_000_000,
                "minimum_dispatch_spacing_seconds": 5,
                "http_timeout_seconds": 10,
                "maximum_session_seconds": 900,
                "maximum_retries": 3,
                "maximum_attempts": 4,
                "incremental_cost_usd": 0,
                "expected_reporting_intervals": 96,
                "expected_minute_files": 1440,
            }.items():
                _require(
                    type(pilot[field]) in (int, float) and pilot[field] == expected,
                    "live configuration changes a frozen bound",
                )
            for field in ("anchor_utc", "end_exclusive_utc", "closeout_deadline_utc"):
                _require(
                    pilot[field] == pilot["approval_record"][field],
                    "live schedule and approval disagree",
                )
            plan = plan_retrieval(pilot["anchor_utc"], pilot["end_exclusive_utc"])
            return cls(
                canonicalize(
                    {
                        "pilot_id": pilot["pilot_id"],
                        "specification_id": pilot["specification_id"],
                        "protocol_sha256": canonical_sha256(config),
                        "plan_sha256": plan.plan_id,
                        "public_key_hex": pilot["public_key_hex"],
                        "signer_key_id": pilot["signer_key_id"],
                        "approval_record": pilot["approval_record"],
                        "caps": CAPS,
                        "parser": PARSER,
                    }
                )
            )
        except (KeyError, TypeError) as exc:
            raise NetworkSafetyError("missing live pilot authority configuration") from exc

    @property
    def value(self) -> dict[str, Any]:
        return json.loads(self.encoded)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.encoded)

    def validate_context(self, body: dict[str, Any]) -> None:
        value = self.value
        for field in ("pilot_id", "specification_id", "protocol_sha256", "plan_sha256"):
            _require(body[field] == value[field], "live receipt detached from pinned authority")

    def evidence(
        self,
        *,
        cumulative_bytes: int,
        dispatch: float,
        completed: float,
        elapsed: float,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "authority_sha256": self.sha256,
            "approval_sha256": canonical_sha256(self.value["approval_record"]),
            "cumulative_download_bytes": cumulative_bytes,
            "dispatch_monotonic_seconds": dispatch,
            "completed_monotonic_seconds": completed,
            "session_elapsed_seconds": elapsed,
            "response_headers": {
                k: v for k, v in headers.items() if k in {"content-length", "content-type"}
            },
        }

    def validate_evidence(self, body: dict[str, Any]) -> None:
        self.validate_context(body)
        data = body["live_evidence"]
        _require(
            type(data) is dict
            and set(data)
            == {
                "authority_sha256",
                "approval_sha256",
                "cumulative_download_bytes",
                "dispatch_monotonic_seconds",
                "completed_monotonic_seconds",
                "session_elapsed_seconds",
                "response_headers",
            },
            "invalid live receipt evidence fields",
        )
        _require(
            data["authority_sha256"] == self.sha256
            and data["approval_sha256"] == canonical_sha256(self.value["approval_record"]),
            "live receipt rights binding mismatch",
        )
        count = data["cumulative_download_bytes"]
        _require(
            type(count) is int and body["bytes_received"] <= count <= CAPS["download_bytes"],
            "invalid live cumulative byte evidence",
        )
        for field in (
            "dispatch_monotonic_seconds",
            "completed_monotonic_seconds",
            "session_elapsed_seconds",
        ):
            v = data[field]
            _require(
                type(v) in (int, float) and math.isfinite(v) and v >= 0,
                "invalid live monotonic evidence",
            )
        _require(
            data["dispatch_monotonic_seconds"] <= data["completed_monotonic_seconds"]
            and data["session_elapsed_seconds"] < 900,
            "live duration or monotonic chronology violated",
        )
        headers = data["response_headers"]
        _require(
            type(headers) is dict
            and set(headers) <= {"content-length", "content-type"}
            and all(
                type(v) is str and len(v) <= 256 and not any(ord(c) < 32 for c in v)
                for v in headers.values()
            ),
            "live receipt headers exceed the safe allowlist",
        )
        if "content-length" in headers:
            _require(
                headers["content-length"] == str(body["content_length"]),
                "live length header evidence mismatch",
            )
