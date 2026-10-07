"""Run explicitly synthetic, wholly offline Phase 2 fixture workflows.

Usage: ``python -m crypto_ai.phase2.workflow evaluate-synthetic-fixture``.
The evaluation smoke command accepts no paths, models, endpoints, or real holdout inputs. It
creates and consumes its own deterministic evidence inside a fresh temporary
directory, then prints a compact JSON summary before removing that directory.
``krypx phase2 train-production`` admits verified synthetic artifacts only;
real production training and automatic model activation remain prohibited.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from crypto_ai.features.build import compute_features
from crypto_ai.phase2 import evaluation, holdout
from crypto_ai.phase2.dataset import COMBINED_COLUMNS, SENTIMENT_COLUMNS, TECHNICAL_COLUMNS
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp

_MARKET_START = datetime(2025, 12, 1, tzinfo=UTC)
_OOF_START = datetime(2025, 1, 1, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_CUTOFF = 100
_FIRST_HOLDOUT = _CUTOFF + holdout.PURGE_ROWS + 1
_LAST_MARKET = _FIRST_HOLDOUT + holdout.MINIMUM_DAYS * 24 - 1
_PROTOCOL_SHA256 = "1" * 64
_CODE_COMMIT = "3" * 40
_LOCK_SHA256 = "4" * 64
_RUN_ID = "synthetic-evaluation"


def _snapshot(directory: Path, name: str, raw: bytes) -> evaluation.SnapshotRef:
    path = directory / name
    path.write_bytes(raw)
    return evaluation.SnapshotRef(path=path, sha256=sha256_bytes(raw))


def _build_fixture(root: Path) -> evaluation.SyntheticEvaluationRequest:
    """Issue all required evidence from fixed, local, synthetic values only."""
    development_run_dir = root / f"synthetic-development-{uuid4().hex}"
    development_run_dir.mkdir()
    evaluation_root = root / "evaluations"
    evaluation_root.mkdir()
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir()

    holdout_started_at = _MARKET_START + _FIRST_HOLDOUT * _HOUR
    inspected_at = holdout_started_at + holdout.MINIMUM_DAYS * _DAY
    signal_ordinals = {_FIRST_HOLDOUT + 24 * index for index in range(holdout.MINIMUM_TRADES)}

    market_lines = ["market_ordinal,timestamp,open,high,low,close,volume"]
    market_rows: list[dict[str, object]] = []
    previous_close = 100.0
    for ordinal in range(_LAST_MARKET + 1):
        opened_at = _MARKET_START + ordinal * _HOUR
        open_price = previous_close
        development_signal = ordinal in {36, 48, 60, 72, 84, 96}
        body_return = (
            0.08
            if ordinal in signal_ordinals or development_signal
            else 0.0003 + 0.0001 * (ordinal % 11 - 5)
        )
        close_price = open_price * (1.0 + body_return)
        high_price = max(open_price, close_price) * 1.0015
        low_price = min(open_price, close_price) * 0.9985
        volume = float(1000 + ordinal % 23)
        market_rows.append(
            {
                "timestamp": pd.Timestamp(opened_at),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
        market_lines.append(
            f"{ordinal},{format_utc_timestamp(opened_at)},"
            + ",".join(
                repr(value) for value in (open_price, high_price, low_price, close_price, volume)
            )
        )
        previous_close = close_price
    market_raw = ("\n".join(market_lines) + "\n").encode("utf-8")
    market_ref = _snapshot(snapshots_dir, "market.csv", market_raw)
    development_market_raw = (
        "\n".join(market_lines[: _CUTOFF + holdout.HORIZON + 3]) + "\n"
    ).encode("utf-8")
    development_market_ref = _snapshot(
        development_run_dir, "development_market.csv", development_market_raw
    )
    development_rows_raw, augmented_model_bytes, control_model_bytes = (
        evaluation.build_synthetic_development_fit(development_market_raw, _CUTOFF)
    )
    development_rows_ref = _snapshot(
        development_run_dir, "development_rows.json", development_rows_raw
    )
    article_ref = _snapshot(snapshots_dir, "articles.jsonl", b"")
    score_ref = _snapshot(snapshots_dir, "scores.jsonl", b"")

    technical = compute_features(pd.DataFrame(market_rows))
    no_news = {name: 0.0 for name in SENTIMENT_COLUMNS}
    no_news["hours_since_latest_article"] = 24.0
    no_news["news_missing_24h"] = 1.0
    feature_lines = ["market_ordinal,decision_at," + ",".join(COMBINED_COLUMNS)]
    prediction_rows = []
    for ordinal in range(_FIRST_HOLDOUT, _LAST_MARKET - holdout.HORIZON):
        decision_at = _MARKET_START + (ordinal + 1) * _HOUR
        values = [float(technical.loc[ordinal, name]) for name in TECHNICAL_COLUMNS]
        values += [no_news[name] for name in SENTIMENT_COLUMNS]
        prediction_rows.append(
            {"market_ordinal": ordinal, **dict(zip(COMBINED_COLUMNS, values, strict=True))}
        )
        feature_lines.append(
            f"{ordinal},{format_utc_timestamp(decision_at)},"
            + ",".join(repr(value) for value in values)
        )
    feature_raw = ("\n".join(feature_lines) + "\n").encode("utf-8")
    feature_ref = _snapshot(snapshots_dir, "features.csv", feature_raw)

    prediction_frame = pd.DataFrame(prediction_rows)
    augmented_probabilities = evaluation._predict(
        prediction_frame,
        COMBINED_COLUMNS,
        evaluation._verified_model(augmented_model_bytes, COMBINED_COLUMNS),
    )
    completed_exit_ordinals = evaluation._scheduled_exit_ordinals(
        prediction_frame, augmented_probabilities
    )
    augmented_model_sha256 = sha256_bytes(augmented_model_bytes)
    control_model_sha256 = sha256_bytes(control_model_bytes)
    frozen_fit_at = _MARKET_START + 105 * _HOUR + timedelta(minutes=30)
    shared_rows_sha256 = development_rows_ref.sha256
    fit_manifest_bytes = canonicalize(
        {
            "schema_version": holdout.FIT_SCHEMA,
            "specification_id": holdout.SPECIFICATION_ID,
            "synthetic": True,
            "selected_augmented_cell": "C",
            "matched_control_cell": "A",
            "development_cutoff_ordinal": _CUTOFF,
            "shared_labeled_rows_sha256": shared_rows_sha256,
            "augmented_model_sha256": augmented_model_sha256,
            "control_model_sha256": control_model_sha256,
            "augmented_fit_count": 1,
            "control_fit_count": 1,
            "frozen_at_utc": format_utc_timestamp(frozen_fit_at),
        }
    )
    frozen_fit = holdout.FrozenFitProof(
        selected_augmented_cell="C",
        matched_control_cell="A",
        development_cutoff_ordinal=_CUTOFF,
        shared_labeled_rows_sha256=shared_rows_sha256,
        augmented_model_sha256=augmented_model_sha256,
        control_model_sha256=control_model_sha256,
        fit_manifest_sha256=sha256_bytes(fit_manifest_bytes),
        augmented_fit_count=1,
        control_fit_count=1,
        frozen_at=frozen_fit_at,
        fit_manifest_bytes=fit_manifest_bytes,
        augmented_model_bytes=augmented_model_bytes,
        control_model_bytes=control_model_bytes,
    )
    labels = (
        holdout.DevelopmentLabel(99, 104, _MARKET_START + 104 * _HOUR),
        holdout.DevelopmentLabel(100, 105, _MARKET_START + 105 * _HOUR),
    )
    purge_ordinals = tuple(range(_CUTOFF + 1, _FIRST_HOLDOUT))
    boundary = holdout.BoundaryPurgeManager.validate(
        development_cutoff_ordinal=_CUTOFF,
        development_labels=labels,
        purge_ordinals=purge_ordinals,
        first_holdout_ordinal=_FIRST_HOLDOUT,
        first_holdout_decision_at=_MARKET_START + (_FIRST_HOLDOUT + 1) * _HOUR,
        frozen_fit=frozen_fit,
    )

    oof_spans = tuple(
        holdout.OofSpan(
            first_test_decision_at=_OOF_START + 36 * index * _DAY + _HOUR,
            last_test_decision_at=_OOF_START + 36 * (index + 1) * _DAY,
        )
        for index in range(holdout.OOF_FOLDS)
    )
    readiness_plan = holdout.ZeroOutcomeReadinessInspector.plan(
        oof_spans, holdout.MINIMUM_TRADES, frozen_at=_MARKET_START + 105 * _HOUR
    )
    market_first_ordinal = _FIRST_HOLDOUT - holdout.TECHNICAL_CONTEXT_ROWS + 1
    candles = tuple(
        holdout.ClosedCandle(
            ordinal=ordinal,
            opened_at=_MARKET_START + ordinal * _HOUR,
            closed_at=_MARKET_START + (ordinal + 1) * _HOUR,
        )
        for ordinal in range(market_first_ordinal, _LAST_MARKET + 1)
    )
    scheduled_exits = tuple(
        holdout.ScheduledExit(
            decision_ordinal=exit_ordinal - holdout.HORIZON - 1,
            ordinal=exit_ordinal,
            exit_at=_MARKET_START + exit_ordinal * _HOUR,
            policy_sha256=_PROTOCOL_SHA256,
        )
        for exit_ordinal in completed_exit_ordinals
    )
    readiness = holdout.ZeroOutcomeReadinessInspector.inspect(
        plan=readiness_plan,
        market_first_ordinal=market_first_ordinal,
        first_holdout_ordinal=_FIRST_HOLDOUT,
        market_last_ordinal=_LAST_MARKET,
        candles=candles,
        raw_snapshots=tuple(
            holdout.RawSnapshot(raw, sha256_bytes(raw))
            for raw in (market_raw, b"", b"", feature_raw)
        ),
        provider_outage_state="CLEAR",
        provider_gap_exclusions_verified=True,
        holdout_started_at=holdout_started_at,
        inspected_at=inspected_at,
        scheduled_exits=scheduled_exits,
        frozen_policy_sha256=_PROTOCOL_SHA256,
    )

    development_manifest = {
        "schema_version": evaluation.DEVELOPMENT_SCHEMA,
        "synthetic": True,
        "run_id": development_run_dir.name,
        "development_cutoff_ordinal": _CUTOFF,
        "purge_ordinals": list(purge_ordinals),
        "first_holdout_ordinal": _FIRST_HOLDOUT,
        "selected_augmented_cell": "C",
        "matched_control_cell": "A",
        "shared_labeled_rows_sha256": shared_rows_sha256,
        "fit_manifest_sha256": sha256_bytes(fit_manifest_bytes),
        "augmented_model_sha256": augmented_model_sha256,
        "control_model_sha256": control_model_sha256,
        "augmented_fit_count": 1,
        "control_fit_count": 1,
        "fitted_max_decision_ordinal": _CUTOFF,
        "augmented_feature_columns": list(COMBINED_COLUMNS),
        "control_feature_columns": list(TECHNICAL_COLUMNS),
        "signal_threshold": 0.5,
        "development_market_sha256": development_market_ref.sha256,
        "development_rows_sha256": development_rows_ref.sha256,
        "model_family": "LogisticRegression",
    }
    (development_run_dir / "development_manifest.json").write_bytes(
        canonicalize(development_manifest)
    )
    for name, raw in {
        "augmented_model.json": augmented_model_bytes,
        "control_model.json": control_model_bytes,
        "fit_manifest.json": fit_manifest_bytes,
        "development_dataset_manifest.json": evaluation._development_dataset_manifest_bytes(
            development_market_raw, development_rows_raw, _CUTOFF
        ),
    }.items():
        (development_run_dir / name).write_bytes(raw)
    inventory_sha256 = sha256_bytes(
        canonicalize(
            {
                "evaluation_run_id": _RUN_ID,
                "market_snapshot_sha256": market_ref.sha256,
                "article_snapshot_sha256": article_ref.sha256,
                "score_snapshot_sha256": score_ref.sha256,
                "feature_snapshot_sha256": feature_ref.sha256,
            }
        )
    )
    return evaluation.SyntheticEvaluationRequest(
        run_id=_RUN_ID,
        development_run_dir=development_run_dir,
        evaluation_root=evaluation_root,
        boundary=boundary,
        readiness=readiness,
        protocol_sha256=_PROTOCOL_SHA256,
        input_inventory_sha256=inventory_sha256,
        code_commit=_CODE_COMMIT,
        dependency_lock_sha256=_LOCK_SHA256,
        market_snapshot=market_ref,
        article_snapshot=article_ref,
        score_snapshot=score_ref,
        feature_snapshot=feature_ref,
        development_market=development_market_ref,
        development_rows=development_rows_ref,
        random_simulations=1,
        reduced_fixture_mode=True,
    )


def _utc_argument(value: str) -> datetime:
    """Accept only an unambiguous, valid RFC3339 UTC timestamp."""
    try:
        parsed = parse_utc_timestamp(value, field="timestamp")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert parsed is not None
    return parsed


def _train_production(args: argparse.Namespace) -> int:
    """Dispatch only after explicit synthetic opt-in and argument validation."""
    from crypto_ai.phase2 import production

    try:
        authorization = production.load_synthetic_authorization(args.authorization_file)
        request = production.SyntheticProductionRequest(
            evaluation_root=args.evaluation_root,
            development_run_dir=args.development_run_dir,
            versions_root=args.versions_root,
            evaluation_run_id=args.evaluation_run_id,
            model_version=args.model_version,
            authorization=authorization,
            training_as_of_utc=args.training_as_of_utc,
            created_at_utc=args.created_at_utc,
        )
        artifact = production.OfflineProductionEngine().train(request)
    except production.ProductionError as exc:
        print(f"Production fixture rejected: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "activated": False,
                "artifact_count": len(artifact.files),
                "evaluation_run_id": args.evaluation_run_id,
                "model_version": args.model_version,
                "synthetic": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run synthetic-only Phase 2 workflows; never enable real-data execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "evaluate-synthetic-fixture",
        help="evaluate the built-in temporary synthetic fixture only",
    )
    production_parser = commands.add_parser(
        "train-production",
        help="fit and publish a verified synthetic-only production registry fixture",
    )
    production_parser.add_argument("--evaluation-run-id", required=True)
    production_parser.add_argument("--synthetic-only", action="store_true", required=True)
    production_parser.add_argument("--evaluation-root", type=Path, required=True)
    production_parser.add_argument("--development-run-dir", type=Path, required=True)
    production_parser.add_argument("--versions-root", type=Path, required=True)
    production_parser.add_argument("--model-version", required=True)
    production_parser.add_argument("--authorization-file", type=Path, required=True)
    production_parser.add_argument("--training-as-of-utc", type=_utc_argument, required=True)
    production_parser.add_argument("--created-at-utc", type=_utc_argument, required=True)
    args = parser.parse_args(argv)
    if args.command == "train-production":
        return _train_production(args)

    with tempfile.TemporaryDirectory(prefix="krypx-phase2-synthetic-") as temporary:
        request = _build_fixture(Path(temporary).resolve(strict=True))
        artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
        metrics = json.loads(artifact.files["metrics.json"])
        summary = {
            "artifact_count": len(artifact.files),
            "production_decision": metrics["production_decision"],
            "research_verdict": metrics["research_verdict"],
            "synthetic": True,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def dispatch(argv: Sequence[str] | None = None) -> int:
    """Preserve Phase 1 CLI delegation; route only an explicit phase2 prefix."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "phase2":
        return main(arguments[1:])
    from crypto_ai.cli import main as phase1_main

    return phase1_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
