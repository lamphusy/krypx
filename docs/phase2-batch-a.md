# Phase 2 Batch A engineering record

This record is limited to the offline engineering work authorized on 2026-08-14 and
the global-watermark corrective task authorized on 2026-08-15, plus the causal-
availability and terminal-gap-evidence corrective task authorized on 2026-08-16 and
the five-finding acceptance-blocker correction authorized on 2026-08-25 and completed
offline on 2026-08-27.
Milestone 0 is approved only as the engineering specification for offline Milestones 1
and 2. It does not approve provider network access, historical or forward collection,
scorer selection, model or tokenizer downloads, research-gate execution, feature
generation, model training, backtesting, future-holdout collection, or holdout access.

## Milestone 1 — acceptance blockers corrected, pending independent review

Milestone 1 provides strict article and score envelopes, UTC-only timestamp validation,
dependency-free RFC 8785/JCS canonical serialization, exact-byte SHA-256 hashing,
content-addressed immutable objects, and atomic manifest-last/no-overwrite bundle
publication. Its project-specific failures inherit from `CryptoAIError`.

Original Batch A evidence before corrective review:

- Focused Phase 2 foundation suite: 24 passed.
- Full Phase 1 plus Phase 2 suite: 277 passed with 12 pre-existing single-class metric
  warnings.
- Formatting: passed for all `src` and `tests` Python files.
- Lint: passed for all `src` and `tests` Python files.
- Tests are protected by the repository-wide real-network connection guard.

## Milestone 2 — acceptance blockers corrected, pending independent review

Milestone 2 provides one offline GDELT GSG adapter. It creates bounded one-minute
retrieval schedules and pure retry decisions but contains no HTTP client. Caller-supplied
gzip bytes are stored exactly, receipt locators and headers are redacted, and normalization
can advance its durable exclusive filename and causal-availability boundaries only after
every contiguous expected minute is terminally complete or recorded as a provider gap backed
by verified immutable evidence. Historical-backfill
observations are reported and excluded rather than
assigned a model-eligible availability time.

The adapter preserves first observations, repeated observations, immutable article
versions, later title revisions, exact causal duplicate groups, expected-interval gaps,
zero-line valid files, coverage hashes, and deterministic normalized publications.

Original Batch A evidence before corrective review:

- Focused Phase 2 suite: 35 passed.
- Full Phase 1 plus Phase 2 suite: 288 passed with 12 expected single-class metric
  warnings from the existing synthetic Phase 1 integration test.
- Formatting, lint, bytecode compilation, and installed-package consistency: passed.
- No test or implementation path opened a network connection; the adapter has no network
  execution method.

## First corrective integrity implementation

The independent review findings were corrected without rewriting the three existing Batch A
commits. The corrective implementation adds:

- single-open object and publication reads that hash and return or parse the same captured
  byte buffer;
- canonical, transitively hashed normalizer state containing article versions, repeated-
  observation links, exclusions/conflicts, article-group assignments, permanent anchors,
  protocol hash, and rights approval;
- manifest-last, no-overwrite state publication and hydration only from fully verified byte
  buffers, including verification of referenced raw content-addressed objects;
- transactional batches whose new logical articles are grouped only after sorting by
  `(initial_first_seen_at, article_id)`;
- whole-set `revision_time_unknown` exclusion for incoming same-time conflicts and a
  project-specific fail-closed error when a conflict reaches already-published state;
- an explicit provider/scope/protocol-bound rights gate that defaults to
  `license_restricted`; no real GDELT rights approval is recorded; and
- deterministic primary exclusion selection from the frozen precedence table.

Milestones 1 and 2 are presented for independent review rather than marked finally accepted.
Milestone 3 and all later milestones remain incomplete.

First corrective verification evidence:

- Focused integrity regression suite: 19 passed.
- Complete Phase 2 sentiment suite: 57 passed.
- Complete Phase 1 plus Phase 2 repository suite: 310 passed with 12 expected
  single-class metric warnings from the unchanged synthetic Phase 1 integration test.
- `git diff --check`: passed.
- Formatter: passed with 61 files unchanged.
- Lint and bytecode compilation: passed.
- Installed dependency consistency: no broken requirements.
- All execution remained under the repository's real-network connection guard; no network
  access occurred.

## Prior global-watermark corrective implementation

The filename-order cross-call and cross-restart chronology defect was corrected in commit
`8bd1050` without amending the existing Batch A history. Normalizer state v2 added
canonical `chronology.json`, whose minute-level terminal ledger and
`next_expected_interval_start` exclusive watermark are transitively covered by
`state_sha256`.

The correction provides:

- one immutable, strictly ordered, contiguous terminal fact per minute, recording either a
  retrieved-and-normalized raw snapshot with its exact SHA-256 and line count or an explicit
  missing/invalid provider gap;
- fail-closed rejection before mutation for every regressing, overlapping, replayed,
  pending, or forward-gapped plan; exact replay is intentionally unsupported;
- watermark advancement only after the whole plan and proposed normalized state validate;
- single-read, manifest-verified hydration that also verifies canonical chronology,
  transitive file size/hash descriptors, raw content-addressed objects, ledger continuity,
  terminal-fact consistency, and watermark justification;
- canonical state bytes intended to be independent of equivalent normalize-call partitions
  and persist/hydrate boundaries; this prior unconditional claim is superseded by the valid-
  stream scope frozen below; and
- atomic publication-failure coverage for the chronology file, state index, manifest, and
  final rename, including preservation of unrelated staging directories.

Global-watermark corrective verification evidence:

- Focused global-watermark regression suite: 24 passed.
- Complete Phase 2 sentiment suite: 81 passed.
- Complete Phase 1 plus Phase 2 repository suite: 334 passed with 12 expected
  single-class metric warnings from the unchanged synthetic Phase 1 integration test.
- `git diff --check`, formatting, lint, bytecode compilation, installed-dependency
  consistency, JSON validation, and Markdown/JSON status reconciliation: passed.
- All tests used synthetic fixtures and temporary storage under the repository-wide
  real-network guard; no network or external service was accessed.

Milestone 2 remains pending independent review. This correction does not approve real
GDELT rights, network collection, Batch B, or any scorer/research/holdout activity.

## Causal-availability and terminal-gap-evidence corrective implementation

The filename watermark and minute ledger from `8bd1050` remain intact. The current isolated
correction advances the normalizer to state v3 and chronology v2 while adding a separate
causal-availability invariant and immutable synthetic gap evidence.

The correction freezes these contracts:

- `next_expected_interval_start` remains the exclusive end of the last terminal filename
  interval. Regressing, overlapping, replayed, pending, and forward-gapped plans still fail
  before mutation.
- `closed_availability_through` is a separate exclusive causal boundary. Every complete
  interval uses its canonical `raw_published_at` as `terminal_at`; every missing or invalid
  gap uses the matching verified evidence's `terminal_at`. All terminal event times must be
  strictly increasing, and the next boundary is exactly the final event plus one UTC
  microsecond. It is persisted and never inferred from article rows.
- Each complete snapshot requires canonical UTC `raw_published_at >= ingested_at`. Distinct
  raw snapshots must have strictly increasing publication times across calls and restarts;
  a publication before the persisted causal boundary fails before article/group mutation.
- `gdelt-gsg-snapshot-v2` binds `gdelt-gsg-parser-policy-v1` and the exact compressed-byte,
  decompressed-byte, and JSON-line bounds. Hydration uses those same bounded settings to
  reparse each single-read raw buffer and chronologically replay normalization, rejecting any
  canonically rehashed article/link/exclusion state that disagrees with the raw observations.
- Equal-time conflicting fingerprints for one `article_id` inside one raw snapshot are all
  excluded as `revision_time_unknown`. Equal publication time across distinct raw snapshots
  is invalid. A conflict against prior immutable state remains fail-closed and cannot rewrite
  a version or duplicate-group anchor.
- Partition- and restart-independent canonical state bytes are required only for valid input
  streams satisfying both filename and causal-availability chronology. Invalid regressions,
  distinct-snapshot equal times, and conflicts against immutable prior state are not claimed
  to match an atomic rebuild. Recovery requires a separate chronological state generation
  outside this task.
- Absence plus caller-supplied time is not provider-gap evidence. Every missing or invalid
  interval requires canonical immutable `gdelt-gsg-terminal-gap-evidence-v1`, including an
  evidence ID plus a detached canonical-body SHA-256, provider/scope/interval/expected-
  locator/mode/input-class bindings,
  `gdelt-gsg-retry-policy-v1`, attempt count, and strictly ordered attempt facts containing
  attempt number/time, HTTP status or bounded error kind, retry disposition, and Retry-After
  when applicable. It also binds a verified terminal outcome/time and the protocol hash.
  Invalid-snapshot evidence binds snapshot ID, raw SHA-256, and the bounded parser error. Only
  a non-retryable terminal result or exact retry exhaustion may close the interval. Synthetic
  evidence grants no provider or network authority.
- Canonical `gap-evidence.json`, chronology, and both boundaries are transitively covered by
  `state_sha256`. Hydration rejects legacy state and uses the existing single-read,
  manifest-verified buffers to validate canonical bytes, descriptors, raw receipts, both
  watermarks, complete snapshot times, gap evidence, and closed-world relations among ledger
  records, articles, links, exclusions, receipts, and evidence. It never silently migrates,
  infers, repairs, or fills missing chronology.
- Any normalization or publication failure leaves articles, groups, exclusions, links,
  ledger, gap evidence, and both boundaries unchanged and publishes no partial generation.

Current causal-availability and terminal-gap-evidence corrective verification evidence:

- Focused causal-availability regressions: 17 passed.
- Focused terminal-gap-evidence regressions: 44 passed.
- All five focused GSG files: 117 passed.
- Complete Phase 2 sentiment suite: 144 passed.
- Complete Phase 1 plus Phase 2 repository suite: 397 passed with 12 expected single-class
  metric warnings from the unchanged synthetic Phase 1 integration test.
- Diff validation, formatting, lint, bytecode compilation, installed-package consistency,
  JSON validation, and Markdown/JSON reconciliation: passed.
- All tests used synthetic fixtures and temporary storage under the repository-wide real-
  network guard. No network, provider, publisher, credential, model, market-data, or holdout
  access occurred.

Milestones 1 and 2 remain corrected but pending independent review. Milestone 3 and all later
milestones remain incomplete. The earlier totals above remain clearly identified as prior
evidence.

## Five-finding acceptance-blocker corrective implementation

An independent adversarial review of commit
`3e31538e79be85002fc4f11633f22b189058a42a` returned **NOT ACCEPTED** despite the
then-green repository suite. The current isolated offline correction addresses every reported
failure without rewriting prior commits:

- Article validation now checks `title` and `content` are exactly `str | None` before any
  string operation or identity recomputation. Integer, list, and object payloads fail closed
  with `ArticleValidationError`, including when their dependent hashes are recomputed.
- Exact-byte storage opens descriptors with `O_NONBLOCK` where available and validates the
  opened descriptor with `fstat`. FIFOs and every other non-regular object fail immediately
  with `SentimentStorageError` rather than blocking before validation.
- Every outer publication manifest must contain exactly `files`, `metadata`,
  `publication_id`, and `schema_version`; the version must be
  `immutable-publication-v1`, and metadata must be an object. State-v3 hydration therefore
  rejects unknown schemas, extra fields, and malformed metadata before deserialization.
- Normalized-publication boundaries validate immutable tuple contents, canonical order,
  article/link/exclusion relationships, UTC `as_of`, finite retrieval rates in `[0,1]`,
  nonnegative integer counters, due/complete/gap arithmetic, zero-line bounds, gap duration
  and grouping invariants, all declared hashes, and both recomputed RFC 8785 semantic
  identities before writing any final artifact.
- Idempotent normalization collisions compare the exact payload buffers and the complete
  expected manifest, including schema, file descriptors, metadata, provider, scope, protocol
  hash, rights hash, and semantic hashes. Altered metadata is rejected as a collision rather
  than accepted as an idempotent rerun.

Corrective verification evidence:

- Strict article contract tests: 27 passed.
- Immutable storage tests: 15 passed, including a subprocess-bounded FIFO regression.
- GSG adapter and normalized-publication tests: 24 passed.
- Normalizer state-integrity tests: 24 passed, including state-v3 outer-manifest rejection.
- Complete Phase 2 sentiment suite: 179 passed.
- Frozen Phase 1 suite: 253 passed with 12 expected single-class synthetic metric warnings.
- Complete repository suite: 432 passed with the same 12 expected warnings.
- Diff validation, formatting (64 Python files unchanged), lint, bytecode compilation,
  installed-package consistency, RFC 8785 differential verification, JSON validation, and
  Markdown/JSON reconciliation: passed.
- Tests used only synthetic/captured fixtures and temporary storage. No network, provider,
  publisher, credential, model, market-data, or holdout access occurred, and no push was
  performed during this corrective task.

The governance record now distinguishes the current no-push correction from historical
repository activity. Local reflogs record commit `3e31538e79be85002fc4f11633f22b189058a42a`
updating `origin/codex/phase2-foundation` at `2026-08-16T16:59:32+07:00`, fast-forwarding
local `main` at `2026-08-16T16:59:45+07:00`, and updating `origin/main` at
`2026-08-16T16:59:50+07:00`. The local evidence does not identify the actor, and no matching
push authorization is recorded in this protocol. The current corrective task explicitly
prohibits and performs no push.

Milestones 1 and 2 are corrected but remain pending a new independent read-only review.
Milestone 3 and all later milestones remain incomplete.

## Remaining authorization boundary

Real GDELT retrieval, GDELT title-only/right-use acceptance, all network/provider and
publisher access, accounts, credentials, paid services, historical or prospective collection,
scorer/model selection, model downloads, scoring, feature construction, numerical research
gates, training, backtests, and all holdout access/evaluation remain unapproved. Batch B is
not authorized. Milestone 3 and later milestones are not complete.

The exact next action is independent read-only review of the single five-finding
acceptance-blocker corrective commit based on
`3e31538e79be85002fc4f11633f22b189058a42a`, including reproduction of all five prior
adversarial probes and verification of the governance reconciliation.
Batch B is not authorized. A later bounded prospective GSG collection pilot would require a new
authorization that explicitly:

1. approves GDELT GSG title-only use and the documented-use interpretation;
2. permits network access only to the frozen GSG archive endpoint, prospectively from the
   authorization time, with exact interval/request, download-byte, retained-storage, elapsed-
   time, and retry caps;
3. permits durable storage of those exact raw bytes and receipt metadata;
4. confirms that historical retrieval, publisher pages, credentials, paid services, scoring,
   features, model activity, research gates, and holdout activity remain forbidden unless
   separately authorized.

Until that authorization is supplied, the repository must remain fixture-only and offline.
