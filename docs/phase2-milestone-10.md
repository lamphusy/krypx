# Phase 2 Milestone 10 — Production Decision and Model Registry Specification

Specification ID: `phase2-milestone10-production-decision-registry-v1`

Status: **SPECIFICATION_FROZEN** — documentation and protocol configuration only.

Human authorization recorded: **2026-10-05**. Accepted Milestone 9 implementation:
`468e8221bd9a4566c75ce71818a25e8a7f23001b` on `main`, with 1,693 passing tests.

## 1. Authority and evidence boundary

This instruction authorizes only this specification, changes to
`config/phase2_protocol.json` and `docs/phase2-research-protocol.md`, offline
verification, and a local documentation commit on `main`. It does not authorize
Milestone 10 source/test implementation, production training, promotion, activation,
deployment, or a Git push. `milestone_10_implementation_authorized`,
`milestone_10_production_training_authorized`, `milestone_10_promotion_authorized`
and `milestone_10_activation_authorized` remain `false`.

The active Milestone 9 status progresses to `ACCEPTED`; its accepted scope remains
**OFFLINE_ONLY** synthetic engineering, not a real holdout result. Its immutable
historical acceptance record remains `ACCEPTED_OFFLINE_ONLY`. Synthetic gate passes,
test counts and engineering acceptance are not evidence of production suitability.

The live pilot remains `DEFERRED_WITHOUT_BACKFILL`, with
`network_pilot_authorized: false` and `real_network_calls_prohibited: true`.
Real data acquisition, model downloads, paid scoring, real model training, and real
holdout access/evaluation remain unauthorized. No old observation window may be
backfilled. Phase 1 code, tests, data and artifacts remain byte-identical.

## 2. Production promotion decision

A future production decision requires an immutable, completed and independently
verified Milestone 9 evaluation bundle at
`artifacts/phase2/evaluations/{run_id}/`. Verify the exact
`evaluation_manifest.json` bytes and its complete payload inventory, immutable
Development/model/dataset/protocol parents, one-time generation claim, and durable
claim-completion record. The completion must bind this exact evaluation; directory
presence or a caller-supplied `PASS` string is insufficient.

Source-binding verification and accounting reconciliation remain mandatory, including
predictions, ledgers, equity curves, cost scenarios, baselines and final numerical
gate operands. Do not loosen the accepted Milestone 9 gates or repeat its consumed
evaluation to obtain a different decision.

| Verified evidence | Production decision and action |
|---|---|
| Missing/incomplete bundle or completion; malformed/corrupted/hash-mismatched evidence; inconsistent parents, outputs or gates | `NO-GO`; reject before fitting or publishing any production model. |
| Research verdict `FAIL`, missing/unknown verdict, or undefined required gate | `NO-GO`; no override by a CLI flag or successful engineering tests. |
| Synthetic-only bundle, even with research verdict `PASS` | `NO-GO` for real production; usable only in separately authorized offline implementation tests. |
| Verified real research `PASS`, but no explicit applicable human authorization | `NO-GO` pending authorization; no training, promotion or deployment. |
| Verified real research `PASS` and explicit applicable human authorization | Eligible only for the specifically authorized production operation; no automatic activation. |

Human authorization must identify the accepted evaluation run and exact manifest
hash, permitted operation and target production version. Training, promotion and
activation are distinct permissions; none is implied by a research verdict or by
the existence of a version directory. Preserve the authorization reference with
the production manifest. Record engineering verification separately from research
verdict and operational permission. The current state authorizes none of these
production operations.

## 3. Future `train-production` contract

`train-production --evaluation-run-id <accepted-run-id>` is the specified future
entry point, **not an implemented or authorized executable in this step**. The
argument is required for provenance and authorization lookup; a run ID by itself
is not human authority. Resolve it to the verified evidence in Section 2 and reject
an absent, unaccepted, synthetic-for-real-production, incomplete or failed run.

Before fitting, freeze a training input snapshot and `training_as_of_utc`. Fit on
all eligible currently labeled history in **Development plus verified holdout
labeled rows** in that snapshot. Labels and their complete next-open exit context
must be available by that cutoff. Do not use unlabeled tails, unverified data,
provider-gap exclusions, duplicate decision rows, or silently reinstate the
boundary-purge rows excluded by the frozen evaluation protocol. This is not
authorization to acquire additional history or to evaluate a new holdout.

Verify the prepared dataset manifest transitively against market, raw news, score,
coverage and aggregation parents; preserve causal feature/missingness contracts,
shared row identities and the unchanged Phase 1 H=4 cost-aware label. Bind the
exact training row/label identities and input snapshot hashes. A provenance copy
must describe the actual combined production training dataset, not merely copy a
Development-only manifest while fitting additional holdout rows.

Use the selected augmented cell's frozen model family, effective hyperparameters
and seed from Milestone 6 and the accepted evaluation provenance:

- Cell C: `LogisticRegression` with its required `StandardScaler`.
- Cell D: `XGBClassifier` with frozen Phase 1 XGBoost settings.

Reject cross-labeled families, unsupported models, changed parameters, altered
features or outcome-driven tuning. Fit a fresh production model; fit any required
production scaler only on the exact production training rows. Preserve the frozen
evaluation models and their original scalers unchanged. Retain the production
scaler as part of the serialized model contract or as an explicitly hashed payload.

The production input is exactly the authoritative **37-column** order in
[Milestone 5 Section 2](phase2-milestone-5.md): 24 technical columns followed by
13 sentiment columns. Verify this against both the accepted augmented evaluation
model and the production prepared dataset. Do not infer columns from numeric data
or reorder them. `feature_schema_hash` is SHA-256 of the RFC 8785 canonical ordered
array in `feature_columns.json`; its values must equal the manifest's
`feature_columns` array exactly.

The production refit is a distinct model trained with information unavailable to
the original evaluation fit. It must never replace an evaluation artifact, be
used to recompute claimed historical holdout performance, reset a consumed claim,
or be presented as the model whose one-time holdout result justified this decision.

## 4. Versioned storage and registry

Reserve one safe, nonempty single-component `model_version` under the fixed root;
reject path traversal, separators and symlinked storage ancestors. Publish:

```text
artifacts/phase2/production/versions/{model_version}/
├── model.json                         # or declared serialized model payload
├── feature_columns.json               # exact ordered 37-name JSON array
├── prepared_dataset_manifest.json     # actual fit dataset provenance copy
└── manifest.json                      # completion marker, written last
```

Additional required scaler/provenance payloads must be explicitly inventoried.
Record the model serialization format and dependency versions; verify hashes and
format before loading. Never deserialize an unverified model. A registry entry
is valid only after complete manifest, payload and parent verification, not because
the directory or a model file exists.

The canonical `manifest.json` contains these required metadata fields:

| Field | Contract |
|---|---|
| `model_version` | Unique version identifier, exactly equal to the final directory name. |
| `model_type` | Selected augmented `LogisticRegression` or `XGBClassifier`; bound to the evaluated cell. |
| `training_start`, `training_end` | UTC decision timestamps of the first/last included training rows, not filesystem times. |
| `training_row_count` | Positive integer equal to the verified unique fit-row count. |
| `feature_columns` | Exact 37 ordered names, equal to `feature_columns.json`. |
| `feature_schema_hash` | Exact-byte SHA-256 of that RFC 8785 canonical ordered array. |
| `model_parameters` | Complete effective frozen classifier parameters and preprocessing contract. |
| `evaluation_run_id_provenance` | Accepted run ID, also bound to its exact evaluation manifest hash. |
| `created_at_utc` | Explicit UTC creation time; never inferred from mutable file metadata. |

Also bind specification/protocol identity, training-as-of cutoff, model/scaler fit
provenance, prepared dataset manifest hash, code commit, dependency lock hash,
authorization reference and **`production_artifact_hashes`**. That inventory maps
each relative payload path to its exact-byte SHA-256 and byte length; it includes
the model, columns, prepared manifest and every additional payload, but not the
self-referential hash of `manifest.json`. Parent data must remain available for
transitive verification. Hashes are 64-character lowercase hexadecimal SHA-256.

JSON rejects duplicate/unknown schema keys, missing required fields, invalid types,
non-finite numbers and malformed UTC timestamps, and uses RFC 8785 serialization.
Capture, hash and parse the same byte sequence; fail closed with project-specific
errors on any metadata, model, inventory, provenance or hash mismatch. Define and
version the concrete parser schema during separately authorized implementation;
no silent legacy migration or replacement is permitted.

## 5. Atomic publication and durability

The future publisher must implement all of the following in order:

1. Acquire a **version-scoped exclusive filesystem lock**, anchored to the verified
   `versions/` directory. Concurrent publishers of the same version cannot proceed.
   A lock never grants permission to replace an existing directory.
2. Fail if the final version path already exists, **including an empty directory**,
   partial prior publication, symlink or other filesystem object. Do not delete it,
   reuse it, merge into it or treat it as an idempotent replacement.
3. Create hidden same-filesystem staging
   `.staging-{model_version}-{random}` under `versions/`. Write and fsync all model,
   schema and provenance payloads. Read back and verify their exact captured bytes,
   lengths and SHA-256 hashes before creating the completion marker.
4. Deep-snapshot metadata. Write and fsync canonical `manifest.json` **last**, with
   `production_artifact_hashes`; verify its exact bytes and full inventory. Reject
   symlinks, non-regular objects, extra/missing entries and concurrent mutations
   through descriptor-anchored verification and two-pass directory inventory.
5. Fsync the staging directory, atomically rename it with **no-replace** semantics
   to `{model_version}`, then fsync the parent `versions/` directory. Use the accepted
   cross-platform no-replace primitive; no check-then-overwriting-rename fallback.
   Unsupported platform/filesystem capability fails closed.
6. Re-verify exact manifest and payload bytes after rename through the owned directory
   descriptor. Do not report success before verification and durability complete.
   On failure, unpublish the owned completion marker with an ownership-safe rollback
   so no invalid manifest-bearing version remains discoverable; preserve unrelated
   collision or concurrently replaced directories. Release the version lock on exit.

Failures and interrupted staging cannot yield a usable registry version. A visible
directory without a verified manifest is not a completed model and cannot be loaded.
Previously completed versions are immutable and retained for provenance. Recovery
must not overwrite another version, bypass authorization or rewrite evaluation history.

**No automatic activation:** neither `train-production` nor registry publication
creates, updates or switches `active_model.json`. Activation is a separate,
explicitly authorized operational action; deployment remains out of scope here.

## 6. Future offline acceptance and current handoff

Separately authorized offline implementation must use synthetic fixtures and test
missing/corrupted/FAIL evidence, absent human authority despite PASS, run-ID/manifest
substitution, model-family/parameter/feature-order mismatches, training-row and
label provenance, preservation of evaluation models, and separation from activation.
Storage tests must cover version-lock contention, existing empty destinations,
manifest-last ordering, exact-byte tampering, fsync/rename failures, replacement
races and ownership-safe rollback. Synthetic acceptance cannot authorize real fitting.

This docs/config-only freeze requires all 1,693 existing repository tests to pass,
formatting/lint checks, strict repository JSON validation, local RFC 8785 differential
checks, and unchanged Phase 1 bytes (48/48 source/test blobs; 50/50 including the two
previously selected fixtures). No implementation acceptance is claimed by this file.

Next proposed action: `implement_milestone_10_production_registry_offline`, **awaiting
separate human authorization**. This freeze does not execute that action.
