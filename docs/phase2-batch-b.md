# Phase 2 Batch B — prospective ingestion specification

Current live specification ID: `phase2-batch-b-live-prospective-v1`.
Base specification ID: `phase2-batch-b-prospective-ingestion-v2`, reconciled on 2026-09-08.
Supersedes v1, frozen on 2026-09-07 in commit `b8b72d50133647ebc99846bfd69bb8866d185136`.
Base status: `SPECIFICATION_FROZEN_COMPATIBLE`. Live setup: `VERIFIED_SETUP_NOT_ARMED`.
`network_pilot_authorized: true`; `real_provider_rights_approved: true`;
`milestone_3_scoring_authorized: false`. No collection or scheduler arming occurs during setup.

## September 13 live pilot authority and setup

The human approved a replacement anchor of `2026-09-13T14:00:00Z`, recorded at
`2026-09-13T13:03:24Z`. The previously authorized midnight September 13 anchor was
missed before arming; no requests, responses or live receipts were produced for it.
It is not backfilled. The existing rights approval and provisioned public-key pin
apply to this expressly approved successor window; no date was shifted automatically.

The human authority accepted offline corrective commit
`14769aff7d72a44c17e520c99f6f90588e83101a` with **723 passing tests** (470 sentiment,
253 Phase 1), and separately authorized the real prospective GSG pilot, its HTTPS runner,
and an isolated Ed25519 key pair. The additional explicit actual-product rights approval,
recorded at `2026-09-12T17:46:20Z`, covers internal GSG title metadata, incidental exact raw
fields including unrelated titles and rejected languages, and immutable retention through
review. GDELT attribution is required; external redistribution, publisher scraping, body-text
collection and automatic disposal remain prohibited. This is the human's recorded approval,
not a claim of a newly researched provider license or a broader research/scoring rights grant.

Pilot ID: `phase2-batch-b-20260913`. The authorized window is
`[2026-09-13T14:00:00Z, 2026-09-14T14:00:00Z)`, with 96 reporting slots and 1,440 minute
files. First worker: `2026-09-13T14:45:00Z`; last worker: `2026-09-14T14:30:00Z`;
retrieval and signed closeout deadline: `2026-09-14T14:45:00Z`. The September 9 plan remains
historical, missed and unexecuted; it is not backfilled. No further anchor shift is implied.

The implementation must bind the canonical approval-record SHA-256 and exact protocol
SHA-256, actual provider/scope, fixed plan, pinned public key and verified code commit.
Approval identity is computed from the separate approval object when loaded, not stored as
a self-referential hash of its enclosing protocol. The public key is 32-byte lowercase hex;
its key ID is SHA-256 of those bytes. Missing pins, key mismatch, altered authority, unsafe
paths, unverified code, a missed anchor or unavailable durable state prevent arming.

Keep the exclusive pilot namespace at `data/phase2-pilot-20260913` (mode `0700`, ignored by
Git), including raw CAS, publications, control files and `secrets/pilot_ed25519.key`
(mode `0600`). Never print or commit the private key. All retained files, including secrets
and scheduler/control metadata, count toward the same 2,000,000,000-byte storage cap.
The public pin belongs in `batch_b_live_pilot` in the protocol JSON; a null pin means setup
is incomplete and cannot be armed. The exact-byte responses are never redacted or committed.

After setup passes offline verification, the explicit launch command is:

```text
.venv/bin/python scripts/run_phase2_pilot.py arm
```

Run it before `2026-09-13T14:00:00Z`. It installs 96 local launchd calendar workers; setup
itself does not execute that command or contact GDELT. Workers are sequential and exclusive,
each protected by an OS watchdog capped at the earlier of 900 monotonic seconds and its
fixed lag-adjusted slot end. There is no always-running 24-hour collector, catch-up window,
or budget reset. Unbounded stdout/stderr are discarded; only bounded local status evidence
is retained. The machine must remain available for the scheduled slots; sleep, missed starts,
local failures and caps cannot be represented as provider outages or verified intervals.

The real path must preserve verified TLS, exact URL allowlisting/no redirects, bounded raw
stream accounting, five-second dispatch spacing, four attempts only for HTTP 429/5xx,
10-second HTTP timeouts, atomic CAS, signature chaining and signed closeout verification.
The $0.00 incremental cost cap and all numerical limits below remain unchanged. Production
inputs use `provider_response` and separate live authority/evidence contracts, never mocked
flags or synthetic receipt/gap relabeling. Accepted offline validators remain fail-closed.

Local `origin/main` reflog records `14769aff` updated by push at
`2026-09-12T23:45:28+07:00`; actor and authority are not established by that record.
This setup performs no push or live remote query. The authorized current commit is local
on `main`; no completion commit hash, successful collection, live coverage pass, or later
milestone completion is invented in advance.

Next action: `execute_live_prospective_pilot`, after setup validation and explicit arming.
Scoring, models, indicators/features, market data, training/backtests, research gates and
holdout collection/access/evaluation remain unauthorized.

### Final setup verification — replacement 14:00 UTC anchor

All verification was offline. The 121 new tests exercise the actual HTTP parser against
in-memory responses, verified TLS configuration, strict transfer completion, process deadlines,
live authority/signatures/CAS replay, signed failure closeouts and mocked calendar installation.
The watchdog is a fresh isolated interpreter, with a startup acknowledgement and parent
liveness pipe; no Python runs in a forked multi-threaded interpreter. A separate ten-second
watchdog bounds DNS/TLS/headers; it cannot extend the outer worker/slot deadline.

| Command | Result, exit 0 |
|---|---|
| `git diff --check` | No output |
| `.venv/bin/black --check .` | `80 files would be left unchanged.` |
| `.venv/bin/ruff check --no-cache .` | `All checks passed!` |
| `.venv/bin/python -m compileall -q src tests scripts` | No output |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `.venv/bin/pytest` | `844 passed, 12 warnings in 18.50s` |
| `.venv/bin/pytest tests/sentiment/test_live_transport.py tests/sentiment/test_live_pilot.py -q` | `121 passed in 8.57s` |

Only the 12 pre-existing synthetic single-class metric warnings remain. Phase 1 source/tests
are 48/48 byte-identical to `14769aff`; all 94 pre-existing tracked blobs outside the five
modified Phase 2 files are unchanged. Strict JSON/fixture checks and six invalid-JSON controls
passed. RFC 8785 binary64 differential matched local Node.js for 49,972/49,972 values.

Live receipt/closeout/gap schemas are `batch-b-live-signed-receipt-v1`,
`batch-b-live-signed-closeout-v1`, and `batch-b-live-terminal-gap-evidence-v1`. They bind
the approved provider/parser/caps through the immutable authority, plus actual attempt
byte/monotonic/header facts. Fixture schemas retain their original strict defaults.
The runner produces raw CAS, verified parser observations, signed attempts/gaps and
96-slot closeout evidence. It does not migrate the Batch A state-v3 normalizer, publish
sentiment features, or claim an independently accepted live result.

Run `.venv/bin/python scripts/run_phase2_pilot.py check` from the committed clean `main`
checkout before arming. It validates local governance, key custody/pins, the future anchor,
and launchd GUI availability without scheduling or making a request. Keep this checkout
unchanged and this Mac awake/logged in through the closeout deadline. A missed slot is not
backfilled. Transfer-Encoding is rejected and a real peer EOF is required; incompatible
server framing, volume or availability may fail the pilot rather than weaken its caps.
Authenticated terminal failures produce an early signed failure closeout when still safe.
Hard deadline, capacity or unresolved-transfer failures can leave an incomplete run without
a closeout; they never certify coverage. Signing-key custody remains local and unchanged.

## Historical offline authority (preserved, not live permissions)

Offline implementation amendment: `phase2-batch-b-offline-implementation-v1`, authorized
2026-09-10 against base commit `ec01f7d411fddc93f959eed06e6e399565538805`.
Current offline implementation status: `ACCEPTED_OFFLINE_ONLY` at `14769aff`.
Four-P1 corrective authority: 2026-09-11; verification completed 2026-09-12 against
`a4ce157c0a8306ea6b760bdfab3812295f3e86be`. The initial 648-test implementation was
subsequently **NOT ACCEPTED**; the corrective patch subsequently passed independent review.

The September 8 instruction authorized only documentation/configuration. The subsequent
September 10 human instruction authorizes source/test implementation of the network-client
interface, circuit breakers and Ed25519 receipts, with **offline/mock-only execution** and
a local feature commit on `main`. `real_network_calls_prohibited: true`. Batch A remains
ACCEPTED and COMPLETED at engineering commit
`bb6d4d3854103d41d5f4c7338de9445aa1b3dbe5`, signed off by governance commit
`e7c189bf180c9b3fd72892544fa72805998f765d`. No real provider rights, operational signing-key
access, scheduler, live HTTP implementation or network execution was authorized by that
offline instruction. Deterministic
in-memory fixture keys and mocked transport responses are permitted for the new tests.
Archive compatibility remains resolved; the subsequent independent offline acceptance is
not evidence of a successful live pilot. The September 13 authority above supersedes only
the operational permissions and schedule, not the synthetic identities of offline evidence.

The dated offline amendment below originally superseded v2's retry restriction only for
the mock adapter. The live instruction separately adopts the same ceiling of three retries
and four attempts. The accepted Batch A retry policy and contract identities stay unchanged.

## Source, usage, and retention boundary

The requested research scope is GDELT GSG English Bitcoin titles. Normalize only English
titles matching the frozen direct-BTC selector (`bitcoin`, `btc`, `xbt`, or
`satoshi nakamoto`, excluding the existing `BTC City` false positive). Retain provider
URLs solely as identifiers; `content` remains null. Never follow a publisher URL,
redirect to a publisher, image URL, related link, or search-result link.

The human selected Option 1, the GDELT GSG archive, replacing the DOC API. The sole
permitted source for the authorized September 13 pilot is HTTPS GET with certificate validation at
`https://data.gdeltproject.org/gdeltv3/gsg/{YYYYMMDDHHMMSS}.gsg.json.gz`.
Only the 1,440 timestamped locators generated by Batch A's `plan_retrieval()` and
`expected_gsg_source_locator()` for this window are allowlisted. No directory listing,
query string, arbitrary URL, redirect, alternate host/path, HTTP downgrade, account,
credential, Google Cloud, paid API, or publisher scraping is permitted.

**Locator reconciliation:** the instruction's `http://data.gdeltproject.org/gdeltv2/`
and `.gsg.jsonl.gz` spelling do not match the accepted code. To satisfy the human's
explicit exact-Batch-A-compatibility decision, this specification uses the code's HTTPS
`gdeltv3/gsg/` path and `.gsg.json.gz` suffix; these are gzip files containing JSONL.
The supplied HTTP/v2 spelling is not an additional endpoint. The former
`https://api.gdeltproject.org/api/v2/doc/doc` endpoint is removed from the allowlist,
not retained as a fallback. No endpoint was contacted or newly verified in this task.

The archive payload and identity contracts align with accepted
`src/crypto_ai/sentiment/providers/gdelt_gsg.py` without relabeling or conversion:

| Contract | Frozen Batch A identity / bound |
|---|---|
| Provider / scope | `gdelt_gsg` / `gdelt_gsg_english_btc_titles` |
| Parser / policy | `gdelt-gsg-jsonl-v1` / `gdelt-gsg-parser-policy-v1` |
| Snapshot | `gdelt-gsg-snapshot-v2`; exact raw SHA-256, file timestamp, input class, collection mode, parser identity and bounds |
| Observation | `gdelt-gsg:<file_timestamp>:<raw_snapshot_sha256>:<zero_based_line_number>:<from\|to>`; both endpoints retain their actual provider fields |
| Normalizer / persisted state | `gdelt-gsg-normalizer-v3` / `gdelt-gsg-normalizer-state-v3`; legacy state fails closed |
| Chronology | `gdelt-gsg-terminal-chronology-v2`; separate minute watermark and causal-availability boundary |
| Gap evidence / retry | `gdelt-gsg-terminal-gap-evidence-v1` / `gdelt-gsg-retry-policy-v1`; existing synthetic-only authority restriction preserved |
| Parser maximums | 67,108,864 compressed bytes, 268,435,456 decompressed bytes, 1,000,000 JSON lines per snapshot |

The DOC/GSG product blocker is resolved. Offline adapter, budget and signing components
are independently accepted; the separately authorized live transport and evidence setup
must pass its own verification before arming. Signed mock envelopes bind the accepted identities,
not redefine them. The mock gap version described below is deliberately rejected by unchanged
Batch A v1 validators/state-v3. Changing flags cannot convert synthetic evidence into real
authority, and no state migration or live evidence approval is implied.

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
not a new rights grant. The September 13 human approval now explicitly covers the actual
chosen product, title metadata and incidental raw fields, internal use, attribution,
and immutable retention. Its separate record is bound to the exact protocol SHA-256 and
provider scope. `real_provider_rights_approved: true` applies only to that approved pilot
scope. Missing, expired or mismatched approval fails before any request. Synthetic-fixture
approval is never valid for real responses.

Retain accepted raw CAS objects, receipts, signatures, attempts, evidence, and manifests
immutably through review until a separately authorized retention/disposal decision. There
is no automatic expiration, overwrite, eviction, or deletion to make room. The retention
rights record must permit that duration; otherwise the pilot cannot start. Allocate an
isolated pilot namespace and keep all records local and outside Git.

## Prospective schedule and hard bounds

Historical schedule status: `MISSED_UNAUTHORIZED_NOT_EXECUTED`, recorded on 2026-09-10.
The September 9 anchor elapsed without network authorization or execution. Preserve these
dates solely as the historical frozen plan and synthetic test fixtures; never backfill or
shift them automatically. The human has now approved the September 13 successor anchor.

The current fixed prospective UTC anchor is `2026-09-13T14:00:00Z`. Its observation window
is half-open: `[2026-09-13T14:00:00Z, 2026-09-14T14:00:00Z)`. Freeze 96 reporting
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
| Retries | Live authority: at most three retries/four attempts, HTTP 429/5xx only, delays 5/10/20 seconds; durable counters; accepted Batch A three-attempt policy unchanged |
| HTTP timeout | 10.0 seconds with verified-TLS real transport and a hard worker watchdog; mocked transport remains available only for offline tests |
| Cost | $0.00 incremental billed cost; no paid service or infrastructure; any positive or unknown third-party charge blocks execution |

The 900-second cap applies to each process, not to the 24-hour calendar span. The authorized
runner schedules at most one worker for each of the 96 reporting intervals.
Worker `i` has a processing slot from `anchor + (i+1)*900s + 1800s` to that time plus
900 seconds. Workers do not overlap; each deadline is the earlier of actual process
start plus 900 seconds and its fixed processing-slot end. Late starts cannot move a
deadline, and a missed worker cannot be relaunched to extend it. There is no continuously
running 24-hour collector; scheduler setup and explicit arming are separate operations.
Global byte, storage, retry, and rate budgets survive all invocations and
restarts. A reset or a fresh process cannot replenish them. Budget state must be durable
and locked against concurrent workers; missing or inconsistent budget state stops the pilot.

These 96 reporting intervals are not the GSG terminal ledger: each contains exactly 15
one-minute intervals, for 1,440 distinct planned gzip files. All 15 must verify for their
reporting interval to count. Keep the minute-level state-v3 ledger and coverage report;
aggregate only in a separate pilot report. Request planned minutes in ascending order
and preserve Batch A's terminal watermark and strictly increasing publication chronology.

Batch A's frozen `PROVIDER_LAG` is 1,800 seconds: a file timestamp `t` is due at `t+30m`.
The schedule accommodates this release-arrival allowance by starting each worker at its
reporting interval's end plus 30 minutes, when all 15 files are due. The first worker
starts `2026-09-13T14:45:00Z`; the last starts `2026-09-14T14:30:00Z` and must finish,
including closeout, by `2026-09-14T14:45:00Z`. This explicitly specified 45-minute
retrieval/closeout tail does not extend the 24-hour observation window or admit next-day
file timestamps. No later catch-up is allowed. The 30-minute value is the accepted
engineering policy, not a newly verified provider SLA; late files remain subject to
bounded retries and genuine gap evidence. No live arrival behavior is claimed offline.

Under the authorized four-attempt live policy, all attempts for 15 files require at most 60
request starts per worker (at least 295 seconds from first to last at five-second spacing;
retry-completion delays impose additional bounds), but transfer time,
provider delay, full-feed volume, and signing/storage work remain unmeasured. These
counts do not guarantee completion under 900 seconds or 500 MB; caps override coverage.

Check remaining budgets before every request and before each bounded body read, decompression,
write, retry, and publication. Reserve room for manifests and terminal incident records
before accepting payloads. Do not rely on `Content-Length`; count bytes during streaming,
stop before exceeding the allowance, and include partial bytes in the durable total.
Parser compressed/decompressed/record limits must be versioned, persisted in snapshot
identity, and no larger than the approved product-specific limits. An unknown length does
not waive the cap. Honor a valid longer `Retry-After`; if it cannot fit the remaining slot
and process budget, stop without retrying early. The September 10 implementation amendment
uses `batch-b-gsg-retry-policy-v1`: only HTTP 429 and 5xx retry, for four total attempts
with delays 5, 10 and 20 seconds after prior completion. Transport failures, HTTP 408,
other client errors and malformed payloads fail closed without retries. HTTP 200 proceeds
to bounded parsing. Preserve global five-second request-start spacing and valid longer
`Retry-After` for 429/503 in addition to completion-relative delays. Authentication, payment,
redirect and scope violations stop the pilot. A restart cannot reset per-file attempts;
deadline/cap cancellation is not retry exhaustion. Accepted Batch A `GSGRetryPolicy`
remains unchanged at three attempts, transport/408/429/5xx and base delays 2/4 seconds;
the v2 historical retry record is retained in JSON with an explicit amendment pointer.

## Raw CAS, signed receipts, and availability

Capture each allowed response body once, before content decoding or JSON reserialization.
Hash those exact bytes with SHA-256 and publish under that digest using immutable CAS,
regular-file descriptor checks, no symlink traversal, fsync, and atomic manifest-last,
no-overwrite publication. Preserve content encoding and parser-policy identity separately.
Verify an existing object against its exact bytes and full manifest before treating a
collision as idempotent. Changed metadata is a collision, not a replacement. Truncated or
invalid responses remain incident evidence, never verified complete snapshots.

Record canonical UTC `requested_at`, `response_started_at`, and `ingested_at` (final body
byte), then `raw_published_at` only after successful durable raw publication. Preserve
Batch A's `first_seen_at = raw_published_at` and terminal time for valid snapshots;
never replace those values with signature-publication times. Separately record signed
receipt publication: a pilot observation cannot become verified/eligible for use until
both raw and valid signed receipt are published. This verification gate does not rewrite
the article's first-seen time or accepted state schema. Enforce strictly
increasing terminal availability across distinct snapshots; retain immutable first-seen
times, every revision, deterministic observation links, and causal deduplication. Provider
publication/seen times are audit fields and cannot backdate KrypX availability. Equivalent
offline replay must reproduce identities, links, revisions, exclusions, and coverage exactly.

A signed receipt is a collector attestation, not a provider signature or rights grant.
The live-pilot design continues to require the broader authority metadata below; the exact
implemented synthetic receipt schema is separately listed in the implementation amendment.
Do not represent that mock schema as a completed live-authority contract. The broader
design requires Ed25519 detached signatures over RFC 8785 UTF-8 bytes of a body containing:

- pilot ID, specification ID, protocol SHA-256, code commit, approval-record SHA-256,
  actual provider ID, parser version/policy and bounds;
- reporting interval and logical retrieval ID, exact planned minute and expected GSG
  source locator (no query),
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
The September 10 instruction authorized only in-memory deterministic fixture signing keys
and offline tamper tests; the current separate live instruction authorizes isolated key
provisioning and public-key pinning as specified above. Batch A SHA-256 receipts are unsigned
and cannot be represented as satisfying the signature requirement.

Redact credentials from locators, request logs, headers, and error diagnostics before those
metadata fields are serialized and signed. Requests must contain no credentials. A
credential-bearing raw response triggers containment and a stop; do not publish it as an
ordinary snapshot or claim redacted bytes retain the original raw hash. Do not follow or
fetch any URL embedded in a response.

## Gap evidence, circuit breakers, and pilot acceptance

Maintain the immutable expected reporting schedule independently of observed responses.
Each missing or invalid logical retrieval requires authenticated terminal evidence binding
provider, scope, protocol SHA-256, planned minute, exact expected GSG locator, ordered actual
attempt facts, versioned retry policy, terminal disposition, and UTC terminal time. Invalid
responses also bind the raw snapshot hash and parser error. Never invent a failed attempt,
terminal timestamp, provider outage, or complete interval from absence alone.

Preserve `TerminalGapEvidence` semantics: only an observed terminal non-retryable outcome
or verified retry exhaustion establishes a provider gap. Batch A's current evidence contract
is restricted to synthetic fixtures and cannot encode a real gap by changing a flag.
The new mock-only contract reuses the `TerminalGapEvidence` dataclass shape under
`batch-b-gsg-terminal-gap-evidence-v2`, bound to `batch-b-gsg-retry-policy-v1` and
authenticated actual attempt receipts. This is not Batch A's
`gdelt-gsg-terminal-gap-evidence-v1`; unchanged Batch A validators and state-v3 hydration
reject the new version. No migration is implemented. The shape reuse and synthetic flags
do not authorize real gaps or a real signed-authority envelope. No DOC evidence is permitted.

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
After a fifth irrecoverably uncovered reporting interval, 92/96 is unattainable; stop and
retain a failed report with the denominator still 96. Never delete evidence or expand any
budget to reach acceptance.

Before human network authorization, the authorized offline implementation and independent
review must prove:

1. Exact planned HTTPS GSG locators and the 96-to-1,440 schedule mapping, gzip/JSONL
   parsing under the accepted snapshot-v2 bounds, complete transfer and gzip integrity,
   and empty-file behavior. Truncation or unknown completeness makes a minute unverified;
   an HTML error or empty transport body is not a valid empty gzip snapshot. No DOC
   query, pagination, query splitting, product substitution, or directory listing exists.
2. Exact-byte CAS replay, cryptographic receipt verification, immutable first-seen times,
   revisions, duplicate links, and deterministic restart/rerun behavior.
3. Atomic publication, symlink/non-regular-object rejection, invalid manifest rejection,
   persistent global budgets, retry exhaustion, `Retry-After`, credential redaction, and
   fail-closed live gap-evidence validation using synthetic fixtures only.
4. All 466 accepted repository tests still pass alongside the new mock/signature/budget tests,
   and Phase 1 source/data/artifacts remain byte-identical. Full verification passed;
   passing offline tests does not certify live transport, rights or collection feasibility.

## September 10 offline implementation amendment

`phase2-batch-b-offline-implementation-v1` records the new source/test authority independently
of the frozen v2 archive specification. Its implementation status is
`ACCEPTED_OFFLINE_ONLY` following review of `14769aff`; no real pilot pass is claimed.

- `network.py` implements an explicitly injected mock-only streaming interface, exact
  planned GSG locators, accepted bounded parser integration, four-attempt Batch B retry
  policy, durable raw CAS/receipts and gap recomputation. There is no built-in HTTP client,
  scheduler or executable live collection path; `real_network_calls_prohibited` must be
  true, input must be synthetic, and transport cost must be known and exactly zero.
- `network_budget.py` implements pilot-wide durable byte counts, request starts and attempts,
  exclusive whole-session locks, fail-closed interrupted intents/stops and no automatic
  budget resets. Entire-store logical/allocated capacity includes CAS, staging, logs,
  publication metadata and the budget journal. Full payload inventory hashes every regular
  store file under pinned descriptors with pre-read and bottom-up post-read verification.
- `receipts.py` implements RFC 8785 domain-separated Ed25519 signing and strict verification,
  predecessor hash chains, immutable context/plan binding and a 96-slot signed closeout.
  A pinned closeout plus independently checked full payload inventory detects chain suffix
  truncation. A local hash chain alone cannot detect a simultaneous rollback of the entire
  store and its checkpoint.

Historical initial verification on 2026-09-10: **648 tests passed** (395 sentiment, 253 Phase 1), including
182 new offline tests. Black left 72 files unchanged; Ruff, compilation and dependency
checks passed. Strict tracked JSON/fixture JSONL checks passed, and 49,972/49,972 RFC 8785
binary64 comparisons matched local Node.js. All 85 pre-existing tracked blobs outside the
four modified Phase 2 governance/exception files are byte-identical to `ec01f7d`, covering
all Phase 1 source/data/artifacts and the accepted Batch A code/tests. See
[the verification record](phase2-batch-b-verification.md) for commands, outputs and limits.
The Ed25519 dependency is already present as `cryptography==49.0.0` in the existing lock;
`requirements-phase2.txt` declares this direct Phase 2 dependency without changing Phase 1
dependency files or installing anything.

Retained closeout verification binds the signed pilot/protocol to its actual stored chain,
durable byte/attempt counters, exact CAS bytes, parser replay and immutable gap publications.
A signature over an unrelated chain cannot certify that store. Restarts wait at least five
monotonic seconds even after a forward UTC clock jump. Closeout and retrieval cannot overlap;
successful closeout prevents later client error paths from mutating its signed budget inventory.

The current exact `batch-b-signed-receipt-v2` body fields are:
`pilot_id`, `specification_id`, `protocol_sha256`, `code_commit`, `plan_sha256`,
`plan_start_at_utc`, `interval_index`, `filename_timestamp`, `source_locator`,
`retry_policy_version`, `input_class`, `real_network_calls_prohibited`, `attempt_number`,
`http_status`, `bytes_received`, `content_length`, `transfer_complete`, `raw_sha256`,
`snapshot_id`, `snapshot_state`, `raw_published_at_utc`, `retry_after_seconds`,
`requested_at_utc`, `dispatch_confirmed_at_utc`, `completed_at_utc`,
and `previous_receipt_sha256`. Unknown fields are rejected, not silently accepted as authority.
The envelope contains exactly `schema_version`, `body`, `body_sha256`, `algorithm`,
`signer_key_id`, and `signature`. Keys are supplied in memory; verification requires the
independently supplied fixture public key, not a key trusted from the envelope itself.

The closeout body contains exactly `pilot_id`, `specification_id`, `protocol_sha256`,
`code_commit`, `plan_sha256`, `final_receipt_sha256`, `receipt_count`, `slot_outcomes`,
and `payload_inventory_sha256`. The inventory covers the complete pilot tree immediately
before closeout publication, including budget checkpoints/events, raw CAS, receipts, gaps,
reports and logs; only the not-yet-created closeout bundle is outside that snapshot. It is
not a filtered inventory of known receipt files. Later acceptance must independently pin
the closeout hash and chain head; an unpinned local result cannot establish completeness.

Historical offline implementation boundaries: the 10.0-second HTTP timeout is a contract passed to the
injected callback. The 900-second deadline is checked cooperatively between operations,
not enforced by an OS-preemptive kill; a real transport and hard watchdog are not implemented.
The network boundary additionally rejects a zero-byte HTTP body even though the accepted
gzip reader can treat an empty stream as empty; a valid gzip member containing zero records
remains allowed. This framing check does not modify Batch A's parser. Real transport,
operational keys, live rights/approval metadata and the new prospective schedule are now
separately authorized for live setup; this does not make mock-only evidence production data
or migrate the accepted Batch A state/gap contracts.

## Four-P1 offline correction — September 11 authority, September 12 verification

The correction does not change the frozen anchor, caps, retry ceiling, accepted Batch A
contracts, Phase 1 behavior, or network/scoring authorization. It tightens four contracts:

1. Capacity checks pin **every** traversed directory until one global bottom-up postpass
   verifies the complete tree. Entry sets, identities, logical sizes and allocated blocks
   must remain unchanged; deep FIFO, sparse-file growth and symlink injections fail closed.
   This is a bounded stability check, not an atomic filesystem snapshot against an external
   writer acting after the final observation. The pilot namespace remains exclusive.
2. The signed request window is `[requested_at_utc, dispatch_confirmed_at_utc]`, bracketing
   `transport.open()`. The durable journal records a separate earlier intent. The next
   request waits at least five monotonic seconds after the preceding open **returns** and
   five UTC seconds after its signed dispatch confirmation. This conservative boundary
   cannot precede physical initiation; a dispatch delay cannot shorten request spacing.
3. A present Content-Length is a strict bounded nonnegative decimal integer and must match
   received bytes exactly for success. The mock transport must explicitly attest completion
   even for an unknown-length stream; bare EOF is insufficient. Known truncation or overrun
   retains exact received CAS bytes and a signed `transfer_complete: false` receipt, clears
   HTTP-200 observations, and produces non-retryable terminal failure evidence, including
   for HTTP 429/5xx. Such minutes never count as verified. Missing/malformed completion
   attestation or a raised stream exception instead leaves a halted unresolved intent;
   neither successful evidence nor an unobserved provider gap is fabricated.
4. Request, dispatch confirmation, completion and raw publication must all lie in the
   authoritative half-open worker slot: `[plan_start + 45 minutes + i*15 minutes,
   plan_start + 60 minutes + i*15 minutes)`. Signatures do not exempt records from these
   bounds; retained replay rejects future-dated evidence too.

Receipt schema v2 authenticates the three additional fields above. Gap evidence is now
`batch-b-gsg-terminal-gap-evidence-v2`. Both signature prefixes remain exactly
`KrypX Batch B receipt v1\n` and `KrypX Batch B closeout v1\n`; closeout schema v1 is unchanged.
Old receipt-v1 evidence is rejected rather than silently supplied new fields or migrated.
The operational `provider_gap` slot label records a terminal unusable retrieval, not proof
of a provider-wide outage; signed transfer facts distinguish truncation from HTTP failure.

See [the corrective verification record](phase2-batch-b-verification.md) for current results.
Local `origin/main` reflog records a push of `a4ce157` at `2026-09-11T00:11:49+07:00`;
the actor and authorization cannot be established by that record. This corrective task
performs only the authorized local commit on `main`, with no push or remote verification.

After a separately authorized live pilot, acceptance requires at least **92 of 96** fully
verified reporting intervals (`92/96 = 95.833333...%`, satisfying `>=95.83%`). The
integer gate of 92, not a rounded percentage, is authoritative. An interval
counts once only when every required response is complete, exact-byte and signature
verified, within bounds, replayed under the frozen policy, and fully reconciled with its
receipt/provenance inventory. A valid empty response counts only when that product's
completeness contract is proved. Gaps, truncation, local stops, missing signatures,
unattempted slots, and duplicate receipts never count as verified coverage.

Each verified reporting interval requires 15 verified minute intervals; report the
1,440-minute denominator separately. GSG archive-file coverage is not a claim of complete
news coverage: isolated stories absent from GSG remain outside this source's coverage.
Publish coverage from the frozen plan plus verified receipts and terminal evidence, never
caller-supplied counters. Report verified, gap, invalid, local-stop, and unattempted counts,
byte/storage/runtime/cost totals, bounds, all hashes, and failures. Do not round 91/96
up to a pass or shrink the denominator after stopping. This is a pilot engineering
criterion; it neither changes nor approves the frozen 99.5% research coverage gate.

## Authority and next action

`SPECIFICATION_FROZEN_COMPATIBLE` records the base archive-compatible specification.
The separately recorded September 13 human instruction authorizes the exact prospective
pilot, actual rights scope, key provisioning and live runner. Before any request, the live
path must verify the final protocol and specification identity, approved rights record,
pinned signer and verified implementation commit, durable budgets and scheduled bounds.
Missing any prerequisite fails before network access. An elapsed anchor requires a new
specification and approval; authorization is never substituted for readiness or coverage.

Publisher scraping, sentiment scoring, model/tokenizer downloads, feature construction or
joining, market-data access, training, backtesting, research-gate execution, future holdout
collection, holdout access/evaluation, paid services, credentials, and `git push` are excluded.
Milestone 3 remains unauthorized. Setup neither arms the scheduler nor starts collection.

The next action is `execute_live_prospective_pilot` after verified setup and explicit arming.
The offline review passed; actual transport and authority-envelope setup must still pass
offline verification before launch. This amendment supersedes the completed offline review action;
the document and current protocol configuration had already superseded v1's
`await_human_network_pilot_authorization` action and the
historical `prepare_batch_b_specification` next action in the 2026-09-02 Batch A sign-off
record. Earlier one-GiB/seven-day collection proposals do not apply to this pilot. No
accepted Batch A parser, interval, retry, storage, or Phase 1 contract is changed.
