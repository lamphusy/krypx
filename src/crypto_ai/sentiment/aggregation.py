"""Synthetic-only point-in-time aggregation of verified immutable fixture evidence.

No market-data loader, scorer execution, network transport, folds or model seam is
provided. Provider state must pass full Batch A raw-byte replay at every public
aggregation/publication boundary. Output publications share that parent CAS root.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from crypto_ai.exceptions import PublicationCollisionError, SentimentError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    ArticleRecord,
    ScoreRecord,
    format_utc_timestamp,
    parse_utc_timestamp,
    validate_article_record,
)
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GroupAnchor,
    GSGNormalizer,
    TerminalInterval,
)
from crypto_ai.sentiment.scoring import ScoreArtifact
from crypto_ai.sentiment.storage import ContentAddressedStore, _open_store_directory

SCHEMA = "synthetic-hourly-aggregation-v1"
SPECIFICATION_ID = "phase2-milestone4-offline-aggregation-v1"
FEATURE_DTYPES = (
    ("sentiment_mean_6h", "float64"),
    ("sentiment_mean_24h", "float64"),
    ("news_count_1h", "int64"),
    ("news_count_6h", "int64"),
    ("news_count_24h", "int64"),
    ("sentiment_recency_6h", "float64"),
    ("sentiment_recency_24h", "float64"),
    ("positive_share_24h", "float64"),
    ("negative_share_24h", "float64"),
    ("sentiment_dispersion_24h", "float64"),
    ("source_count_24h", "int64"),
    ("hours_since_latest_article", "float64"),
    ("news_missing_24h", "int8"),
)
_FILES = frozenset(
    {"config.json", "input.json", "decisions.json", "rows.json", "lineage.json", "envelope.json"}
)


class AggregationError(SentimentError):
    """An offline aggregation contract cannot be satisfied."""


class AggregationInputError(AggregationError):
    """Malformed synthetic inputs, decision grid or numeric output."""


class AggregationAuthorizationError(AggregationError):
    """A caller attempted to leave the explicit synthetic-only boundary."""


class AggregationIntegrityError(AggregationError):
    """An identity, immutable publication or semantic replay is inconsistent."""


class AggregationCoverageError(AggregationError):
    """Unresolved coverage cannot certify an hourly feature window."""


def _hash(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _time(value: object) -> datetime:
    try:
        parsed = parse_utc_timestamp(value, field="aggregation timestamp")
        if type(value) is not str or format_utc_timestamp(parsed) != value:
            raise ValueError("noncanonical UTC spelling")
        return parsed
    except (ValueError, TypeError, OverflowError) as exc:
        raise AggregationInputError("timestamps must use canonical UTC spelling") from exc


def _json(raw: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON constant")

    if type(raw) is not bytes:
        raise ValueError("expected exact bytes")
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if canonicalize(result) != raw:
        raise ValueError("noncanonical JSON")
    return result


def _zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True, slots=True)
class FeatureValues:
    sentiment_mean_6h: float
    sentiment_mean_24h: float
    news_count_1h: int
    news_count_6h: int
    news_count_24h: int
    sentiment_recency_6h: float
    sentiment_recency_24h: float
    positive_share_24h: float
    negative_share_24h: float
    sentiment_dispersion_24h: float
    source_count_24h: int
    hours_since_latest_article: float
    news_missing_24h: int

    def to_dict(self) -> dict:
        values = asdict(self)
        for name, dtype in FEATURE_DTYPES:
            value = values[name]
            if dtype == "float64":
                lower = -1.0 if name.startswith(("sentiment_mean", "sentiment_recency")) else 0.0
                upper = 24.0 if name == "hours_since_latest_article" else 1.0
                if (
                    type(value) is not float
                    or not math.isfinite(value)
                    or not lower <= value <= upper
                ):
                    raise AggregationInputError(f"invalid finite float64 feature: {name}")
                if value == 0.0 and math.copysign(1.0, value) < 0:
                    raise AggregationInputError("feature zero must be positive")
            elif type(value) is not int or not 0 <= value <= (1 if dtype == "int8" else 2**63 - 1):
                raise AggregationInputError(f"invalid integer feature: {name}")
        if not (
            self.news_count_1h <= self.news_count_6h <= self.news_count_24h
            and self.source_count_24h <= self.news_count_24h
            and self.news_missing_24h == int(self.news_count_24h == 0)
        ):
            raise AggregationInputError("feature count/missingness invariants disagree")
        return values


@dataclass(frozen=True, slots=True)
class FeatureDiagnostics:
    scoring_failure_count_6h: int
    scoring_failure_count_24h: int
    scoring_failure_rate_6h: float
    scoring_failure_rate_24h: float
    provider_gap_indicator: int
    low_relevance_count: int

    def to_dict(self) -> dict:
        values = asdict(self)
        for name, value in values.items():
            if "rate" in name:
                if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                    raise AggregationInputError("invalid diagnostic rate")
            elif type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise AggregationInputError("invalid diagnostic count")
        if self.provider_gap_indicator not in (0, 1):
            raise AggregationInputError("invalid provider gap indicator")
        return values


@dataclass(frozen=True, slots=True)
class FeatureRow:
    decision_at: str
    features: FeatureValues | None
    diagnostics: FeatureDiagnostics
    exclusion_reason: str | None = None

    def to_dict(self) -> dict:
        _decisions((self.decision_at,))
        if type(self.diagnostics) is not FeatureDiagnostics:
            raise AggregationInputError("invalid diagnostic schema")
        gap = self.exclusion_reason == "provider_gap_window"
        if self.exclusion_reason not in (None, "provider_gap_window") or (
            gap != (self.features is None) or self.diagnostics.provider_gap_indicator != int(gap)
        ):
            raise AggregationInputError("invalid row exclusion state")
        if self.features is not None and type(self.features) is not FeatureValues:
            raise AggregationInputError("invalid feature schema")
        return {
            "decision_at": self.decision_at,
            "features": self.features.to_dict() if self.features is not None else None,
            "diagnostics": self.diagnostics.to_dict(),
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class SyntheticAggregationInput:
    """References to synthetic state and exact mock artifacts, never supplied article lists."""

    state_publication_id: str
    score_artifacts: tuple[ScoreArtifact, ...]
    coverage_as_of: str
    protocol_config_sha256: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    articles: tuple[ArticleRecord, ...]
    groups: tuple[GroupAnchor, ...]
    scores: tuple[ScoreRecord, ...]
    terminal_intervals: tuple[TerminalInterval, ...]
    state_files: tuple[tuple[str, bytes], ...]


def _prepare(store: ContentAddressedStore, request: SyntheticAggregationInput) -> _Prepared:
    if type(store) is not ContentAddressedStore or type(request) is not SyntheticAggregationInput:
        raise AggregationAuthorizationError("only explicit synthetic fixture requests are admitted")
    if (
        type(request.state_publication_id) is not str
        or not request.state_publication_id.startswith("gsg-normalizer-state-")
        or not _hash(request.protocol_config_sha256)
        or type(request.score_artifacts) is not tuple
        or len(request.score_artifacts) > 10000
    ):
        raise AggregationInputError("invalid synthetic state/score inventory")
    as_of = _time(request.coverage_as_of)
    normalizer = GSGNormalizer.hydrate(store, request.state_publication_id)
    approval = normalizer.rights_approval
    if (
        approval is None
        or approval.approval_kind != "synthetic_fixture_only"
        or approval.approved is not True
        or approval.network_access_authorized is not False
        or normalizer.protocol_config_sha256 != request.protocol_config_sha256
    ):
        raise AggregationAuthorizationError("state must have the exact synthetic-only approval")
    intervals = normalizer.terminal_intervals
    if not intervals:
        raise AggregationCoverageError("no verified terminal coverage")
    for interval in intervals:
        if interval.input_class != "synthetic_fixture" or interval.collection_mode != "prospective":
            raise AggregationAuthorizationError(
                "real or historical provider inputs are not admitted"
            )
        if (
            _time(interval.terminal_at) > as_of
            or _time(interval.start_at) + timedelta(minutes=30) > as_of
        ):
            raise AggregationCoverageError("terminal coverage lies after its frozen as_of")
    for evidence in normalizer.terminal_gap_evidence:
        if (
            evidence.input_class != "synthetic_fixture"
            or evidence.network_access_authorized is not False
        ):
            raise AggregationAuthorizationError("live terminal evidence is not admitted")
    files = normalizer.export_state_files()
    articles = tuple(validate_article_record(item) for item in _json(files["articles.json"]))
    groups = tuple(GroupAnchor(**item) for item in _json(files["groups.json"]))
    by_content = {article.content_hash: article for article in articles}
    scores = {}
    for artifact in request.score_artifacts:
        if type(artifact) is not ScoreArtifact:
            raise AggregationAuthorizationError(
                "only replay-verified MockScorer artifacts are admitted"
            )
        record = artifact.record
        article = by_content.get(record.content_hash)
        synthetic = _json(dict(artifact.files)["input.json"])
        # The frozen mock corpus may include future inputs removed by perturbation
        # tests. Unused, fully verified scores stay in run lineage, never in a row.
        if record.asset != "BTC" or (
            article is not None
            and (
                article.asset != "BTC"
                or article.content is not None
                or synthetic["source"] != article.source
                or synthetic["title"] != article.title
            )
        ):
            raise AggregationIntegrityError(
                "mock score is detached from its synthetic article content"
            )
        if record.content_hash in scores:
            raise AggregationIntegrityError("duplicate or competing score for one content hash")
        scores[record.content_hash] = record
    return _Prepared(
        articles,
        groups,
        tuple(scores[key] for key in sorted(scores)),
        intervals,
        tuple(sorted(files.items())),
    )


def _decisions(values: tuple[str, ...]) -> tuple[datetime, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= 10000:
        raise AggregationInputError("decision grid must be a bounded nonempty immutable tuple")
    parsed = tuple(_time(value) for value in values)
    if any(value.minute or value.second or value.microsecond for value in parsed):
        raise AggregationInputError("decisions must be exact UTC hourly candle closes")
    if tuple(sorted(set(parsed))) != parsed:
        raise AggregationInputError("decisions must be unique and strictly increasing")
    return parsed


def _window_metrics(items: list[tuple[float, ScoreRecord | None]], half_life: int) -> tuple:
    numerator = denominator = recency_numerator = recency_denominator = 0.0
    positive = negative = 0.0
    failed = low = 0
    valid = []
    for age, record in items:
        if record is None or record.state != "succeeded":
            failed += 1
            continue
        s, r = float(record.payload.sentiment_score), float(record.payload.relevance_score)
        if r < 0.20:
            low += 1
            continue
        numerator += r * s
        denominator += r
        weight = r * math.pow(2.0, -age / float(half_life))
        recency_numerator += weight * s
        recency_denominator += weight
        positive += r * float(s > 0.20)
        negative += r * float(s < -0.20)
        valid.append((s, r))
    mean = numerator / denominator if denominator else 0.0
    variance = 0.0
    for s, r in valid:
        delta = s - mean
        variance += r * (delta * delta)
    return (
        _zero(mean),
        _zero(recency_numerator / recency_denominator) if recency_denominator else 0.0,
        _zero(positive / denominator) if denominator else 0.0,
        _zero(negative / denominator) if denominator else 0.0,
        _zero(math.sqrt(variance / denominator)) if denominator else 0.0,
        failed,
        failed / len(items) if items else 0.0,
        low,
    )


def _aggregate_rows(prepared: _Prepared, decision_times: tuple[str, ...]) -> tuple[FeatureRow, ...]:
    """Index window membership and version lookup; retain ordered scalar reductions."""
    times = _decisions(decision_times)
    coverage_start = _time(prepared.terminal_intervals[0].start_at)
    coverage_end = _time(prepared.terminal_intervals[-1].end_at_exclusive)
    if any(coverage_start > t - timedelta(hours=24) or coverage_end <= t for t in times):
        raise AggregationCoverageError(
            "unresolved or insufficient full 24-hour coverage including minute at t"
        )
    gaps = [
        (_time(item.start_at), _time(item.end_at_exclusive))
        for item in prepared.terminal_intervals
        if item.outcome == "provider_gap"
    ]
    score_by_content = {record.content_hash: record for record in prepared.scores}
    versions = {}
    for article in prepared.articles:
        if article.point_in_time_eligible and article.asset == "BTC":
            versions.setdefault(article.article_id, []).append(article)
    version_indexes = {}
    for article_id, articles in versions.items():
        ordered = sorted(
            articles, key=lambda article: (_time(article.first_seen_at), article.article_version_id)
        )
        version_indexes[article_id] = (
            tuple(_time(article.first_seen_at) for article in ordered),
            ordered,
        )
    groups = sorted(
        prepared.groups,
        key=lambda group: (_time(group.initial_first_seen_at), group.duplicate_group_id),
    )
    arrivals = tuple(_time(group.initial_first_seen_at) for group in groups)
    rows = []
    for t, decision_at in zip(times, decision_times, strict=True):
        window_start = t - timedelta(hours=24)
        left, right = bisect_right(arrivals, window_start), bisect_right(arrivals, t)
        selected = []
        sources = set()
        for index in range(left, right):
            group = groups[index]
            chronology, articles = version_indexes.get(group.anchor_article_id, ((), ()))
            selected_index = bisect_right(chronology, t) - 1
            if selected_index < 0:
                raise AggregationIntegrityError("group anchor has no eligible version at decision")
            selected.append(
                (
                    (t - arrivals[index]).total_seconds() / 3600.0,
                    score_by_content.get(articles[selected_index].content_hash),
                )
            )
            sources.add(group.source)
        six = [item for item in selected if item[0] < 6.0]
        one_count = sum(1 for age, _ in selected if age < 1.0)
        m6, r6, _, _, _, f6, rate6, _ = _window_metrics(six, 6)
        m24, r24, pos, neg, dispersion, f24, rate24, low = _window_metrics(selected, 24)
        is_gap = any(start <= t and end > window_start for start, end in gaps)
        diagnostics = FeatureDiagnostics(f6, f24, rate6, rate24, int(is_gap), low)
        features = (
            None
            if is_gap
            else FeatureValues(
                m6,
                m24,
                one_count,
                len(six),
                len(selected),
                r6,
                r24,
                pos,
                neg,
                dispersion,
                len(sources),
                min(24.0, (t - arrivals[right - 1]).total_seconds() / 3600.0) if right else 24.0,
                int(not selected),
            )
        )
        row = FeatureRow(
            decision_at, features, diagnostics, "provider_gap_window" if is_gap else None
        )
        row.to_dict()
        rows.append(row)
    return tuple(rows)


def _config() -> dict:
    return {
        "schema_version": SCHEMA,
        "specification_id": SPECIFICATION_ID,
        "synthetic_only": True,
        "feature_dtypes": dict(FEATURE_DTYPES),
        "feature_order": [name for name, _ in FEATURE_DTYPES],
        "windows_hours": [1, 6, 24],
        "relevance_floor": 0.20,
        "sentiment_cutoffs": [-0.20, 0.20],
        "half_lives_hours": [6, 24],
        "reduction": "ordered-left-to-right-binary64-two-pass-population-v1",
        "group_order": ["group_first_seen_at", "duplicate_group_id"],
        "zero_sign": "positive",
        "python_runtime": sys.version,
        "machine": platform.machine(),
        "float_mantissa_bits": sys.float_info.mant_dig,
        "power_sqrt": "python-math.pow-math.sqrt",
    }


def _request_payload(request: SyntheticAggregationInput) -> dict:
    return {
        "schema_version": "synthetic-aggregation-input-v1",
        "synthetic": True,
        "state_publication_id": request.state_publication_id,
        "coverage_as_of": request.coverage_as_of,
        "protocol_config_sha256": request.protocol_config_sha256,
        "score_artifacts": [
            {name: raw.hex() for name, raw in artifact.files}
            for artifact in sorted(request.score_artifacts, key=lambda item: item.envelope_sha256)
        ],
    }


def _build(
    request: SyntheticAggregationInput, prepared: _Prepared, decisions: tuple[str, ...]
) -> AggregationArtifact:
    rows = _aggregate_rows(prepared, decisions)
    state_files = dict(prepared.state_files)
    lineage = {
        "schema_version": "synthetic-aggregation-lineage-v1",
        "state_sha256": _json(state_files["state.json"])["state_sha256"],
        "state_files": {name: sha256_bytes(raw) for name, raw in prepared.state_files},
        "articles": {
            item.article_version_id: canonical_sha256(item.to_dict()) for item in prepared.articles
        },
        "groups_sha256": sha256_bytes(state_files["groups.json"]),
        "terminal_ledger_sha256": sha256_bytes(state_files["chronology.json"]),
        "gap_evidence_sha256": sha256_bytes(state_files["gap-evidence.json"]),
        "score_envelopes": sorted(item.envelope_sha256 for item in request.score_artifacts),
        "coverage_as_of": request.coverage_as_of,
        "protocol_config_sha256": request.protocol_config_sha256,
    }
    files = {
        "config.json": canonicalize(_config()),
        "input.json": canonicalize(_request_payload(request)),
        "decisions.json": canonicalize(decisions),
        "rows.json": canonicalize([row.to_dict() for row in rows]),
        "lineage.json": canonicalize(lineage),
    }
    identity = {
        "schema_version": SCHEMA,
        "files": {name: sha256_bytes(raw) for name, raw in sorted(files.items())},
    }
    files["envelope.json"] = canonicalize(
        {**identity, "aggregation_id": canonical_sha256(identity)}
    )
    return AggregationArtifact(tuple(sorted(files.items())))


def _read_candidate(artifact: AggregationArtifact) -> dict:
    """Structural inspection only; publish/get additionally replay all parent evidence."""
    try:
        if type(artifact) is not AggregationArtifact or type(artifact.files) is not tuple:
            raise ValueError("invalid candidate")
        files = dict(artifact.files)
        if len(files) != len(artifact.files) or set(files) != _FILES:
            raise ValueError("invalid file inventory")
        decoded = {name: _json(raw) for name, raw in files.items()}
        envelope = decoded["envelope.json"]
        identity = {
            "schema_version": SCHEMA,
            "files": {
                name: sha256_bytes(raw)
                for name, raw in sorted(files.items())
                if name != "envelope.json"
            },
        }
        if (
            envelope != {**identity, "aggregation_id": canonical_sha256(identity)}
            or decoded["config.json"] != _config()
        ):
            raise ValueError("candidate identity/config mismatch")
        return decoded
    except (SentimentError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise AggregationIntegrityError("invalid aggregation artifact buffers") from exc


def _hydrate_row(value: dict) -> FeatureRow:
    if type(value) is not dict or set(value) != {
        "decision_at",
        "features",
        "diagnostics",
        "exclusion_reason",
    }:
        raise AggregationIntegrityError("invalid feature row schema")
    payload = value["features"]
    if payload is not None:
        if type(payload) is not dict or set(payload) != dict(FEATURE_DTYPES).keys():
            raise AggregationIntegrityError("invalid 13-feature schema")
        converted = dict(payload)
        for name, dtype in FEATURE_DTYPES:
            if dtype == "float64":
                if type(converted[name]) not in (int, float):
                    raise AggregationIntegrityError("invalid float64 JSON token")
                converted[name] = float(converted[name])
        payload = FeatureValues(**converted)
    if type(value["diagnostics"]) is not dict:
        raise AggregationIntegrityError("invalid diagnostic schema")
    diagnostics = dict(value["diagnostics"])
    for name in ("scoring_failure_rate_6h", "scoring_failure_rate_24h"):
        if type(diagnostics[name]) not in (int, float):
            raise AggregationIntegrityError("invalid diagnostic rate JSON token")
        diagnostics[name] = float(diagnostics[name])
    row = FeatureRow(
        value["decision_at"], payload, FeatureDiagnostics(**diagnostics), value["exclusion_reason"]
    )
    row.to_dict()
    return row


@dataclass(frozen=True, slots=True)
class AggregationArtifact:
    """Exact immutable candidate buffers; only store.get/publish confer parent verification."""

    files: tuple[tuple[str, bytes], ...]

    @property
    def aggregation_id(self) -> str:
        return _read_candidate(self)["envelope.json"]["aggregation_id"]

    @property
    def rows(self) -> tuple[FeatureRow, ...]:
        try:
            decoded = _read_candidate(self)
            values = decoded["rows.json"]
            if type(values) is not list:
                raise ValueError("invalid rows array")
            rows = tuple(_hydrate_row(value) for value in values)
            _decisions(tuple(row.decision_at for row in rows))
            if [row.decision_at for row in rows] != decoded["decisions.json"]:
                raise ValueError("row/grid mismatch")
            return rows
        except (
            SentimentError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise AggregationIntegrityError("invalid typed aggregation rows") from exc


class OfflineFeatureAggregator:
    """Replay synthetic state and mock artifacts; no scoring or external data execution."""

    def __init__(self, store: ContentAddressedStore) -> None:
        if type(store) is not ContentAddressedStore:
            raise AggregationAuthorizationError(
                "aggregator requires the exact local CAS implementation"
            )
        self.store = store

    def aggregate(
        self, request: SyntheticAggregationInput, decision_times: tuple[str, ...]
    ) -> AggregationArtifact:
        try:
            _decisions(decision_times)
            return _build(request, _prepare(self.store, request), decision_times)
        except AggregationError:
            raise
        except (
            SentimentError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise AggregationIntegrityError(
                "synthetic aggregation input verification failed"
            ) from exc


def _verify_artifact(store: ContentAddressedStore, artifact: AggregationArtifact) -> None:
    try:
        decoded = _read_candidate(artifact)
        value = decoded["input.json"]
        if type(value) is not dict or set(value) != {
            "schema_version",
            "synthetic",
            "state_publication_id",
            "coverage_as_of",
            "protocol_config_sha256",
            "score_artifacts",
        }:
            raise ValueError("invalid input schema")
        if (
            value["schema_version"] != "synthetic-aggregation-input-v1"
            or value["synthetic"] is not True
            or type(value["score_artifacts"]) is not list
        ):
            raise ValueError("not a synthetic input manifest")
        artifacts = []
        for files in value["score_artifacts"]:
            if type(files) is not dict or any(type(raw) is not str for raw in files.values()):
                raise ValueError("invalid score buffers")
            artifacts.append(
                ScoreArtifact(
                    tuple(sorted((name, bytes.fromhex(raw)) for name, raw in files.items()))
                )
            )
        request = SyntheticAggregationInput(
            value["state_publication_id"],
            tuple(artifacts),
            value["coverage_as_of"],
            value["protocol_config_sha256"],
        )
        if type(decoded["decisions.json"]) is not list:
            raise ValueError("invalid decision grid")
        expected = OfflineFeatureAggregator(store).aggregate(
            request, tuple(decoded["decisions.json"])
        )
        if dict(expected.files) != dict(artifact.files):
            raise ValueError("feature/lineage semantic replay mismatch")
    except (
        SentimentError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise AggregationIntegrityError(
            "aggregation artifact failed parent/feature replay"
        ) from exc


class AggregationStore:
    """Immutable feature publications beside their required provider parent publications."""

    def __init__(self, store: ContentAddressedStore) -> None:
        if type(store) is not ContentAddressedStore:
            raise AggregationAuthorizationError("feature store requires the exact local CAS")
        self.cas = store

    def get(self, aggregation_id: str) -> AggregationArtifact | None:
        if not _hash(aggregation_id):
            raise AggregationIntegrityError("invalid aggregation identity")
        publication_id = "aggregation-" + aggregation_id
        descriptor = _open_store_directory(
            self.cas.root,
            ("publications",),
            description="aggregation cache",
            expected_root_identity=self.cas._root_identity,
        )
        try:
            try:
                os.stat(publication_id, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise AggregationIntegrityError("cannot inspect feature publication") from exc
        finally:
            os.close(descriptor)

        def validate_metadata(metadata):
            if (
                type(metadata) is not dict
                or set(metadata) != {"schema_version", "aggregation_id", "envelope_sha256"}
                or metadata["schema_version"] != SCHEMA
                or metadata["aggregation_id"] != aggregation_id
                or not _hash(metadata["envelope_sha256"])
            ):
                raise AggregationIntegrityError("invalid aggregation publication metadata")

        publication = self.cas.read_publication(
            publication_id, metadata_prevalidator=validate_metadata
        )
        artifact = AggregationArtifact(tuple(sorted(publication.files.items())))
        _verify_artifact(self.cas, artifact)
        if (
            artifact.aggregation_id != aggregation_id
            or sha256_bytes(publication.files["envelope.json"])
            != publication.manifest["metadata"]["envelope_sha256"]
        ):
            raise AggregationIntegrityError("feature publication identity mismatch")
        for _, raw in artifact.files:
            if self.cas.get_bytes(sha256_bytes(raw)) != raw:
                raise AggregationIntegrityError("feature CAS dependency mismatch")
        return artifact

    def publish(self, artifact: AggregationArtifact) -> AggregationArtifact:
        _verify_artifact(self.cas, artifact)
        aggregation_id = artifact.aggregation_id
        existing = self.get(aggregation_id)
        if existing is not None:
            if dict(existing.files) != dict(artifact.files):
                raise AggregationIntegrityError("immutable aggregation collision")
            return existing
        for _, raw in artifact.files:
            self.cas.put_bytes(raw)
        try:
            self.cas.publish_bundle(
                "aggregation-" + aggregation_id,
                dict(artifact.files),
                metadata={
                    "schema_version": SCHEMA,
                    "aggregation_id": aggregation_id,
                    "envelope_sha256": sha256_bytes(dict(artifact.files)["envelope.json"]),
                },
            )
        except PublicationCollisionError as exc:
            winner = self.get(aggregation_id)
            if winner is None or dict(winner.files) != dict(artifact.files):
                raise AggregationIntegrityError(
                    "concurrent immutable aggregation collision"
                ) from exc
        result = self.get(aggregation_id)
        if result is None:
            raise AggregationIntegrityError("feature publication disappeared")
        return result
