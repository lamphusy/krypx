# Phase 2 Batch A engineering record

This record is limited to the offline engineering work authorized on 2026-08-14 and
the global-watermark corrective task authorized on 2026-08-15.
Milestone 0 is approved only as the engineering specification for offline Milestones 1
and 2. It does not approve provider network access, historical or forward collection,
scorer selection, model or tokenizer downloads, research-gate execution, feature
generation, model training, backtesting, future-holdout collection, or holdout access.

## Milestone 1 — corrected implementation complete, pending independent review

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

## Milestone 2 — corrected implementation complete, pending independent review

Milestone 2 provides one offline GDELT GSG adapter. It creates bounded one-minute
retrieval schedules and pure retry decisions but contains no HTTP client. Caller-supplied
gzip bytes are stored exactly, receipt locators and headers are redacted, and normalization
can advance its durable exclusive global watermark only after every contiguous expected
minute is terminally complete or recorded as a provider gap. Historical-backfill
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

## Global-watermark corrective implementation

The remaining cross-call and cross-restart chronology defect is corrected in one new,
isolated commit without amending the existing Batch A history. Normalizer state v2 now
includes canonical `chronology.json`, whose minute-level terminal ledger and
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
- canonical state bytes independent of equivalent normalize-call partitions and
  persist/hydrate boundaries; and
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

## Remaining authorization boundary

Real GDELT retrieval, GDELT title-only/right-use acceptance, prospective collection,
scorer/model selection, model downloads, scoring, feature construction, numerical research
gates, training, backtests, and any future-holdout activity remain unapproved. Milestone 3
and later milestones are not complete.

The exact next action is independent review of the new global-watermark corrective commit.
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
