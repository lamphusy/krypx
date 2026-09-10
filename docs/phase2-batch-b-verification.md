# Batch B offline implementation — verification record

Date: 2026-09-10. Base: `ec01f7d411fddc93f959eed06e6e399565538805` on `main`.
Implementation: `phase2-batch-b-offline-implementation-v1`.
Result: **VERIFIED OFFLINE; independent acceptance pending**.
This is not a live-pilot acceptance, network authorization, or Milestone 3 completion.

## Scope and files

- `src/crypto_ai/sentiment/network.py`: injected mock streaming client, exact GSG locators,
  four-attempt retry policy, circuit breakers, CAS/receipt/gap publication and verified closeout.
- `src/crypto_ai/sentiment/network_budget.py`: locked persistent byte/rate/attempt ledger,
  no automatic resets, storage bounds and complete two-pass payload inventory.
- `src/crypto_ai/sentiment/receipts.py`: strict Ed25519 receipt/chain/closeout contracts.
- `src/crypto_ai/sentiment/exceptions.py`: Phase 2-only project exception refinements.
- `tests/sentiment/test_gdelt_gsg_network.py`: 47 mock integration tests.
- `tests/sentiment/test_network_budget.py`: 37 durable-budget and filesystem tests.
- `tests/sentiment/test_receipts.py`: 83 signature/schema/chain tests.
- `tests/sentiment/test_network_adversarial.py`: 15 independent-review regressions.
- `requirements-phase2.txt`: direct declaration of already-installed/locked
  `cryptography==49.0.0`; no installation or dependency upgrade performed.
- `config/phase2_protocol.json`, `docs/phase2-batch-b.md`,
  `docs/phase2-research-protocol.md`, and this record: authority and verification reconciliation.

All original 466 tests remain passing; the four new test modules add 182 tests.
All 85 pre-existing tracked blobs outside the four modified Phase 2 governance/exception
files are byte-identical to the base. This covers every Phase 1 source/data/artifact file
and the accepted Batch A provider, storage, schemas, canonicalizer and existing tests.
The 69/69 figure in the historical Batch A sign-off is preserved as its original
scope-specific result, not overwritten by this broader base-relative comparison.

## Commands and exact final result lines

Executed from `/Users/sylam/Project/KrypX`, entirely offline. All commands below exited 0.
Test progress lines are omitted here; the actual final result lines are retained.

| Command | Final output |
|---|---|
| `git diff --check` | No output |
| `.venv/bin/black --check .` | `All done! ✨ 🍰 ✨` / `72 files would be left unchanged.` |
| `.venv/bin/ruff check --no-cache .` | `All checks passed!` |
| `.venv/bin/python -m compileall -q src tests` | No output |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `.venv/bin/pytest tests/sentiment` | `395 passed in 13.03s` |
| `.venv/bin/pytest --ignore=tests/sentiment` | `253 passed, 12 warnings in 4.56s` |
| `.venv/bin/pytest` | `648 passed, 12 warnings in 16.36s` |

The dependency check also emitted this non-fatal local cache-permission warning:

```text
WARNING: The directory '/Users/sylam/Library/Caches/pip' or its parent directory is not owned or is not writable by the current user. The cache has been disabled. Check the permissions and owner of that directory. If executing pip with sudo, you should use sudo's -H flag.
```

The 12 test warnings are the existing Phase 1 synthetic single-class metric warnings at
`src/crypto_ai/modeling/train.py:133` and `:145`: ROC-AUC and PR-AUC are undefined for a
single-class sample. No real training/backtest or holdout evaluation was performed.

## Strict JSON and RFC 8785 differential

An inline `.venv/bin/python` check enumerated Git-tracked `.json` and fixture `.jsonl`
files, decoded strict UTF-8, rejected duplicate object keys, rejected non-finite constants
and float overflow, then passed each value through the existing RFC 8785 canonicalizer.
Negative controls included duplicate keys, NaN, Infinity, `1e999`, an unpaired surrogate,
and invalid UTF-8. Ignored runtime/holdout artifacts were not inspected.

Exact outputs:

```text
Strict repository JSON: 2/2 tracked files valid
Strict fixture JSONL: 2/2 files; 2/2 records valid
Strict JSON negative controls: 6/6 rejected
Ignored runtime and holdout artifacts were not inspected.
RFC 8785 binary64 differential: 49972/49972 matched; 0 mismatches
Reference: local Node.js v18.20.8; 49995 deterministic candidates; no network
```

The differential used these exact deterministic inputs: for ascending integer `i` from
zero, take the first eight bytes of SHA-256 of ASCII
`krypx-rfc8785-differential-v1:` followed by `str(i)`, interpret as big-endian binary64,
and discard non-finite values until 49,972 remain. Pass hex bit patterns to local Node.js
`Buffer.readDoubleBE(0)` / `JSON.stringify`; compare each output byte-for-byte with Python
`canonicalize(struct.unpack(">d", raw)[0])`. Transporting bit patterns avoids introducing
an intermediate decimal conversion. The existing canonicalizer unit tests also passed.

## Adversarial confirmations

- Exact successful gzip bytes survive CAS publication and replay; invalid/truncated/empty
  HTTP bodies fail closed. A valid gzip member containing zero records is distinct from
  a zero-byte response.
- Every returned body chunk counts, including error/retry/partial and cap-crossing bytes.
  Literal 500,000,001-byte cumulative facts are tested without allocating or downloading
  that body. A sparse 2,000,000,001-byte temporary fixture proves logical storage rejection.
- Request starts are at least five seconds apart, including restarts with a forward UTC
  jump. Transient 429/5xx attempts stop at four, with 5/10/20-second backoffs; longer valid
  Retry-After values are honored. HTTP 408, other client errors and malformed bodies do
  not retry under the new, separately versioned policy.
- Forged, missing, reordered, cross-pilot or detached signed receipts cannot certify
  retained payloads. Verification reconciles actual publications, raw CAS/parser replay,
  immutable gaps and durable byte/attempt counters with the complete signed inventory.
- Terminal gap time cannot precede raw evidence publication. New four-attempt evidence
  uses a distinct version and is rejected by unchanged Batch A v1/state-v3 validation.
- Hydration counts toward the session deadline. Cost/mock-authority changes during reads,
  overlapping closeout/retrieval, and five irrecoverable reporting slots fail closed.
- Successful closeout prevents later client failures from mutating its committed budget
  inventory. A pinned closeout detects truncated chains, extra objects and missing evidence.
- FIFO, symlink, directory replacement and late-injection probes reject unsafe storage
  without opening non-regular payloads. Partial journal/checkpoint updates cannot reset budgets.

## Remaining boundaries

Only explicit mock transports are accepted; there is no built-in live HTTP client,
production launcher or scheduler. The 10-second timeout is passed to injected callbacks;
the 900-second deadline is checked between operations, not by an OS-preemptive watchdog.
An unresponsive callback cannot be forcibly interrupted by this synchronous prototype.
Whole-store rollback requires an independently held closeout pin to detect; an interrupted
request fails closed and needs explicit recovery, not invented receipts or automatic retry.

New gap evidence has no automatic migration into accepted Batch A state-v3. Operational
keys, real rights/approval envelopes, actual transport and live-scale feasibility remain
outside this mock execution. The September 9 anchor is recorded as
`MISSED_UNAUTHORIZED_NOT_EXECUTED`; there is no date shift or backfill.

Next action: `independent_offline_batch_b_acceptance_review` of the resulting local commit.
Network collection, real provider rights, scoring/models, research gates and holdout access
remain unauthorized. No push was performed by this task.
