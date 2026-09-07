# Phase 2 Batch B — prospective ingestion specification

Specification ID: `phase2-batch-b-prospective-ingestion-v1`. Frozen on 2026-09-07.
Status: `SPECIFICATION_FROZEN`. `network_pilot_authorized: false`.
`milestone_3_scoring_authorized: false`.
Next action: `await_human_network_pilot_authorization`.

The human instruction authorizes this document, its protocol-configuration companion,
offline verification, and one local documentation commit on `main`. Batch A remains
ACCEPTED and COMPLETED at engineering commit
`bb6d4d3854103d41d5f4c7338de9445aa1b3dbe5`, signed off by governance commit
`e7c189bf180c9b3fd72892544fa72805998f765d`. This specification grants no execution,
provider-rights, signing-key, scheduler, or network authority. It freezes the requested
requirements; implementation readiness is `BLOCKED_ON_COMPATIBILITY_AND_IMPLEMENTATION`.

## Source, usage, and retention boundary

The requested research scope is GDELT GSG English Bitcoin titles. Normalize only English
titles matching the frozen direct-BTC selector (`bitcoin`, `btc`, `xbt`, or
`satoshi nakamoto`, excluding the existing `BTC City` false positive). Retain provider
URLs solely as identifiers; `content` remains null. Never follow a publisher URL,
redirect to a publisher, image URL, related link, or search-result link.

The sole candidate endpoint for a later authorized pilot is exactly
`https://api.gdeltproject.org/api/v2/doc/doc`, using HTTPS GET with certificate validation.
No alternate host/path, archive fallback, redirect, account, credential, Google Cloud,
paid API, or publisher scraping is permitted. Query parameters, output mode, response
schema, completeness/truncation handling, and their canonical identity must be frozen in
a reviewed endpoint-compatible implementation before authorization. No request may be
built from an arbitrary caller-supplied URL or query.

**Compatibility blocker:** this URL is the DOC API, not the GSG gzip/JSONL archive
implemented in Batch A. The local research protocol already distinguishes those products.
DOC JSON must never be labeled `gdelt_gsg`, converted into invented `from`/`to` records,
or assigned GSG snapshot, observation, chronology, or gap-evidence identities. No DOC
adapter exists in the accepted implementation. Human reconciliation must either explicitly
amend the endpoint to the GSG archive or approve a separately versioned DOC provider
contract and its implementation. This task does neither. The specified DOC URL remains
the sole candidate allowlist entry while this blocker is unresolved.

Freeze the intended rights scope as internal title-metadata ingestion, immutable raw
response retention, normalization, deduplication, integrity replay, and engineering coverage
reporting with GDELT attribution. No publisher bodies, external redistribution, public
corpus publication, or raw-response commits to Git are in scope. Full raw provider responses
can contain rejected languages, unrelated titles, and incidental metadata; exact-byte
retention of those fields requires the same rights approval even though normalized output
is English Bitcoin titles only. A raw response cannot be redacted and still described as
the original bytes.

The Milestone 0 source register records GDELT dataset-use statements observed on 2026-08-12.
Those statements were not refreshed during this offline task, and this specification is
not a new rights grant. Before retrieval, an explicit human rights record must cover the
actual chosen product, title metadata and incidental raw fields, internal use, attribution,
and immutable retention. Bind that record to the exact protocol SHA-256 and provider scope.
`real_provider_rights_approved` remains false. Missing, expired, or mismatched approval
fails before any request. Synthetic-fixture approval is never valid for real responses.

Retain accepted raw CAS objects, receipts, signatures, attempts, evidence, and manifests
immutably through review until a separately authorized retention/disposal decision. There
is no automatic expiration, overwrite, eviction, or deletion to make room. The retention
rights record must permit that duration; otherwise the pilot cannot start. Allocate an
isolated pilot namespace and keep all records local and outside Git.

## Prospective schedule and hard bounds

The fixed prospective UTC anchor is `2026-09-08T00:00:00Z`. The pilot observation window
is half-open: `[2026-09-08T00:00:00Z, 2026-09-09T00:00:00Z)`. Freeze 96 reporting
intervals, each 900 seconds, with interval `i` equal to
`[anchor + i*900s, anchor + (i+1)*900s)` for `0 <= i < 96`.

Authorization and readiness must exist before the anchor. If either is missing, the pilot
does not start: it cannot backfill the elapsed interval, infer historical availability,
move the anchor automatically, or shrink the denominator. A missed anchor requires a
human-approved successor specification with a new prospective date.

| Bound | Frozen limit and accounting |
|---|---|
| Download | 500 MB = 500,000,000 response-body bytes in total, counting successful, failed, retried, and partial transfers before content decompression |
| Retained storage | 2.0 GB = 2,000,000,000 bytes across the complete pilot namespace, including CAS, staging, receipts, signatures, manifests, logs, and reports; enforce allocated and logical bytes conservatively |
| Process duration | At most 900 seconds of monotonic wall time per invocation, including connection, waits, retries, parsing, publication, and cleanup |
| Rate | At most 0.2 Hz globally: one request start per five seconds, no bursts, one in-flight request, including retries |
| Retries | At most three retries after the initial request: four attempts per logical retrieval; counters persist across restarts |
| Cost | $0.00 incremental billed cost; no paid service or infrastructure; any positive or unknown third-party charge blocks execution |

The 900-second cap applies to each process, not to the 24-hour calendar span. A future
reviewed scheduler would invoke at most one worker for each of the 96 slots; workers do
not overlap, and each deadline is the earlier of process start plus 900 seconds and its
slot end. There is no continuously running 24-hour collector and no scheduler is created
by this task. Global byte, storage, retry, and rate budgets survive all invocations and
restarts. A reset or a fresh process cannot replenish them. Budget state must be durable
and locked against concurrent workers; missing or inconsistent budget state stops the pilot.

These 96 reporting intervals are not the GSG terminal ledger. Batch A freezes 60-second
GSG intervals (1,440 per day), each due 30 minutes later. If GSG is retained, every reporting
interval needs all 15 constituent minute files verified, and the final 30-minute arrival
lag needs an explicitly reconciled execution schedule before launch. Neither a 15-minute
state-v3 ledger nor an unapproved extension past the frozen pilot window is permitted.
The endpoint/schedule reconciliation is a launch prerequisite, not a silent change to
Batch A semantics.

Check remaining budgets before every request and before each bounded body read, decompression,
write, retry, and publication. Reserve room for manifests and terminal incident records
before accepting payloads. Do not rely on `Content-Length`; count bytes during streaming,
stop before exceeding the allowance, and include partial bytes in the durable total.
Parser compressed/decompressed/record limits must be versioned, persisted in snapshot
identity, and no larger than the approved product-specific limits. An unknown length does
not waive the cap. Honor a valid longer `Retry-After`; if it cannot fit the remaining slot
and process budget, stop without retrying early. Otherwise retry only transport failures,
HTTP 429, and HTTP 5xx with minimum backoffs of 5, 10, and 20 seconds, also observing the
global rate limit. Other responses terminate the logical retrieval; authentication,
payment, redirect, or scope violations stop the entire pilot. Malformed retry metadata
fails closed. The existing GSG policy has three total attempts, so this four-attempt pilot
policy needs its own version and offline verification before use.

## Raw CAS, signed receipts, and availability

Capture each allowed response body once, before content decoding or JSON reserialization.
Hash those exact bytes with SHA-256 and publish under that digest using immutable CAS,
regular-file descriptor checks, no symlink traversal, fsync, and atomic manifest-last,
no-overwrite publication. Preserve content encoding and parser-policy identity separately.
Verify an existing object against its exact bytes and full manifest before treating a
collision as idempotent. Changed metadata is a collision, not a replacement. Truncated or
invalid responses remain incident evidence, never verified complete snapshots.

Record canonical UTC `requested_at`, `response_started_at`, and `ingested_at` (final body
byte), then `raw_published_at` only after successful durable raw publication. Availability
cannot precede `raw_published_at` and signature/receipt publication. Enforce strictly
increasing terminal availability across distinct snapshots; retain immutable first-seen
times, every revision, deterministic observation links, and causal deduplication. Provider
publication/seen times are audit fields and cannot backdate KrypX availability. Equivalent
offline replay must reproduce identities, links, revisions, exclusions, and coverage exactly.

A signed receipt is a future collector attestation, not a provider signature. Freeze
Ed25519 detached signatures over RFC 8785 UTF-8 bytes of a versioned receipt body containing:

- pilot ID, specification ID, protocol SHA-256, code commit, approval-record SHA-256,
  actual provider ID, parser version/policy and bounds;
- reporting interval and logical retrieval ID, exact canonical allowed request parameters,
  attempt number, HTTP status or bounded transport error, byte count, encoding, raw SHA-256
  and CAS reference when a response exists;
- UTC request/receipt/publication times, monotonic durations, rate and budget counters,
  previous receipt SHA-256, and allowlisted non-secret response headers.

Domain-separate the signing input with UTF-8 bytes of
`KrypX Batch B receipt v1` followed by one LF byte (`0x0A`), then the canonical receipt
body. This is the decoded `\n` in the JSON prefix, not literal backslash-plus-n bytes.
The envelope contains
the body, its SHA-256, `algorithm: Ed25519`, signer key ID, and canonical base64 signature.
Define the key ID as SHA-256 of the raw 32-byte Ed25519 public key. Verification requires an
independently pinned public key whose fingerprint is included in the later human authority
record; never trust a key solely because it accompanies a receipt. First receipt uses a
null predecessor. The previous-receipt hash chain detects internal modification and
reordering, but cannot alone detect removal of a valid suffix.

Require a signed closeout record binding the pilot ID, fixed expected-plan SHA-256,
all 96 slot outcomes, final receipt-chain head and count, and SHA-256 of the canonical
complete payload inventory (including signatures, gaps, and reports but excluding the
closeout itself). Sign its RFC 8785 body with the approved Ed25519 key and domain prefix
`KrypX Batch B closeout v1` followed by LF. Independently pin the final closeout hash and
chain head in the later pilot acceptance record before accepting the result. Verification
against that checkpoint detects suffix removal or substitution; the chain alone does not.
A missing or invalid closeout is an incomplete run, never a coverage pass. An interrupted
run may preserve partial evidence without certifying successful completion.

Hash the final envelope and publish it with its referenced raw object atomically verifiable
as a complete generation. The signed body records raw-object publication time; the signed
receipt envelope has its own later publication timestamp for eligibility, avoiding a
self-referential signature over its own publication. A missing/invalid signature, raw hash,
approval binding, inventory entry, or timestamp prevents verification and normalization.
No signing key is generated or accessed now. Key provisioning, signing support, and offline
tamper tests require separate implementation authority. Batch A SHA-256 receipts are
unsigned and cannot be represented as satisfying this requirement.

Redact credentials from locators, request logs, headers, and error diagnostics before those
metadata fields are serialized and signed. Requests must contain no credentials. A
credential-bearing raw response triggers containment and a stop; do not publish it as an
ordinary snapshot or claim redacted bytes retain the original raw hash. Do not follow or
fetch any URL embedded in a response.

## Gap evidence, circuit breakers, and pilot acceptance

Maintain the immutable expected reporting schedule independently of observed responses.
Each missing or invalid logical retrieval requires authenticated terminal evidence binding
provider, scope, protocol SHA-256, interval, expected locator/query identity, ordered actual
attempt facts, versioned retry policy, terminal disposition, and UTC terminal time. Invalid
responses also bind the raw snapshot hash and parser error. Never invent a failed attempt,
terminal timestamp, provider outage, or complete interval from absence alone.

Preserve `TerminalGapEvidence` semantics: only an observed terminal non-retryable outcome
or verified retry exhaustion establishes a provider gap. Batch A's current evidence contract
is restricted to synthetic fixtures and cannot encode a real gap by changing a flag.
A separately versioned, signed live evidence contract and offline tamper/replay validation
are prerequisites. DOC evidence must never be passed off as GSG terminal evidence.

Local cap, deadline, cancellation, missing authority, and missed-schedule stops are local
incident records, not fabricated provider gaps. They remain uncovered in the fixed
denominator; they cannot advance a GSG terminal watermark. Missing evidence remains
incomplete and fails closed. Previously committed receipts, gaps, revisions, and watermarks
are immutable; recovery requires the documented restart rules or a new approved generation.

Stop the pilot immediately on any cap exhaustion, charge uncertainty, scope/redirect
violation, invalid authority, broken signature/hash, conflicting identity, unsafe storage
object, publication collision, clock regression, incompatible schema, missing signing key,
or inconsistent durable counter. Attempt-level exhausted retries/invalid snapshots can be
recorded as genuine evidenced gaps if the future compatible implementation permits it.
After a fifth irrecoverably uncovered reporting interval, 95% is unattainable; stop and
retain a failed report with the denominator still 96. Never delete evidence or expand any
budget to reach acceptance.

Before human network authorization, separately authorized offline implementation must prove:

1. Provider/endpoint/query/parser and schedule compatibility, including empty-result and
   result-limit behavior. DOC's documented 250-result ceiling and lack of a stable cursor
   cannot be treated as complete article coverage. Truncation or unknown completeness
   makes the affected interval unverified; no unapproved query splitting is allowed.
2. Exact-byte CAS replay, cryptographic receipt verification, immutable first-seen times,
   revisions, duplicate links, and deterministic restart/rerun behavior.
3. Atomic publication, symlink/non-regular-object rejection, invalid manifest rejection,
   persistent global budgets, retry exhaustion, `Retry-After`, credential redaction, and
   fail-closed live gap-evidence validation using synthetic fixtures only.
4. All 466 accepted repository tests still pass and Phase 1 source/data/artifacts remain
   unchanged. These are prerequisites for new implementation; this documentation task
   does not implement or certify a collector.

After a separately authorized live pilot, acceptance requires at least **92 of 96** fully
verified reporting intervals (`92/96 = 95.833333...%`, satisfying `>=95%`). An interval
counts once only when every required response is complete, exact-byte and signature
verified, within bounds, replayed under the frozen policy, and fully reconciled with its
receipt/provenance inventory. A valid empty response counts only when that product's
completeness contract is proved. Gaps, truncation, local stops, missing signatures,
unattempted slots, and duplicate receipts never count as verified coverage.

If GSG is retained after explicit endpoint reconciliation, each verified reporting interval
requires 15 verified minute intervals; report the 1,440-minute denominator separately.
Publish coverage from the frozen plan plus verified receipts and terminal evidence, never
caller-supplied counters. Report verified, gap, invalid, local-stop, and unattempted counts,
byte/storage/runtime/cost totals, bounds, all hashes, and failures. Do not round 91/96
up to a pass or shrink the denominator after stopping. This is a pilot engineering
criterion; it neither changes nor approves the frozen 99.5% research coverage gate.

## Authority and next action

`SPECIFICATION_FROZEN` records the requested specification, not permission to execute it.
Before any request, a later human instruction must reference the final protocol and
specification hashes, resolve DOC/GSG and schedule compatibility, approve the actual
title-use/retention scope, pin the signer public key and verified implementation commit,
and explicitly authorize the bounded prospective pilot. Missing any prerequisite fails
before network access. An elapsed anchor requires a new specification and approval.

Publisher scraping, sentiment scoring, model/tokenizer downloads, feature construction or
joining, market-data access, training, backtesting, research-gate execution, future holdout
collection, holdout access/evaluation, paid services, credentials, and `git push` are excluded.
Milestone 3 remains unauthorized. No pilot, scheduler, or collection is started here.

The current action is `await_human_network_pilot_authorization`, subject to the stated
readiness prerequisites. This document and current protocol configuration supersede the
historical `prepare_batch_b_specification` next action in the 2026-09-02 Batch A sign-off
record. Earlier one-GiB/seven-day collection proposals do not apply to this pilot. No
accepted Batch A parser, interval, retry, storage, or Phase 1 contract is changed.
