# Phase 2 Milestone 4 — Offline Point-in-Time Feature-Aggregation Specification

Specification ID: `phase2-milestone4-offline-aggregation-v1`

Status: **SPECIFICATION_FROZEN**

Human authorization recorded: **2026-09-15**

Baseline on `main`: `d129094c9483f69431eeb6e3698e52ee273d5d0e`

## 1. Authority and scope

This instruction freezes documentation and protocol configuration only. Milestone 3's
synthetic/mock scoring foundation is `ACCEPTED_OFFLINE_ONLY` at the baseline commit,
with 1,003 passing tests, including 159 scoring tests and 253 Phase 1 tests. This freeze
does not claim a Milestone 4 implementation, successful feature build, or acceptance.

Current governance is:

- `milestone_4_status: SPECIFICATION_FROZEN`
- `milestone_4_implementation_authorized: false`
- `milestone_4_feature_aggregation_authorized: false`
- `next_action: implement_milestone_4_aggregation_offline`

The next action names the proposed successor task, **not permission to execute it**.
Human review and separate explicit implementation authorization are required first.
Only this file, `config/phase2_protocol.json`, and `docs/phase2-research-protocol.md`
may change in this step. One local documentation commit on `main` is authorized; push is not.

The live pilot remains `DEFERRED_WITHOUT_BACKFILL`, `network_pilot_authorized: false`,
and `real_network_calls_prohibited: true`. No deferred window may be backfilled.
Real scoring, model/scorer selection or downloads, provider/publisher access, paid services,
market-data joins, model training, backtests, research-gate execution, and holdout access
remain unauthorized. Historical approvals, signer pins and pilot evidence are unchanged.

This document specializes the research protocol's
[Point-in-time hourly sentiment features](phase2-research-protocol.md#point-in-time-hourly-sentiment-features)
section (the section referred to as Section 10 in the task) and its JSON
`aggregation_contract` / `missing_data_policy`. The existing 13 feature definitions,
thresholds, window sizes, deduplication and score-availability semantics are preserved.

## 2. Future offline input and information-set contract

After separate authorization, implementations may consume only synthetic article/version,
duplicate-group, mock-score, retrieval-plan and verified terminal-gap fixtures, with a
synthetic hourly decision grid. No real corpus or market-data loader is in scope.
The grid represents candle closes; `t = candle_open_timestamp + 1 hour`, in UTC.
Constructing this grid in tests does not authorize joining the Phase 1 market dataset.

For each decision `t`:

1. Verify immutable input identities and parent hashes using the accepted article,
   scoring, provider and storage contracts. Reject malformed records, ambiguous versions,
   mismatched score/cache identities, conflicting group anchors, or invalid evidence with
   project-specific exceptions inheriting from `CryptoAIError`.
2. Admit only articles with `point_in_time_eligible = true`, exact `asset == "BTC"`,
   and `first_seen_at <= t`. Provider/publication dates do not replace causal availability.
3. Use each permanent duplicate-group anchor established by accepted causal deduplication.
   `group_first_seen_at` is the anchor's initial eligible `first_seen_at`; it must be `<= t`.
   Do not regroup, change the representative, merge existing groups or reset group arrival
   when later articles or revisions arrive.
4. Select the anchor's **latest eligible version** with version `first_seen_at <= t`.
   Then inspect that version's matching immutable score record. Only `succeeded` scores
   with `relevance_score >= 0.20` enter sentiment aggregates. There is no fallback to an
   older successful version or another group member when the selected version is pending,
   failed, missing a score, or below the relevance floor.
5. `scored_at` remains audit time only, exactly as in the existing scoring contract.
   It is not a new article-eligibility timestamp or a second historical availability gate.
   The frozen synthetic score corpus and scorer/configuration identity must be held fixed
   when checking point-in-time perturbations; this is not a claim of live scoring latency.
6. Window membership and age use permanent `group_first_seen_at`, never revision or score
   time. Every group counts once in each applicable window; nested windows may each include
   the same group. Revisions affect values only from their own availability onward and
   never create counts, reset age, or re-enter expired windows.

All windows are open-left, closed-right: `(t-W, t]`. A group exactly at `t-W` is excluded;
a group exactly at `t` is included. Equal-time version ambiguity remains excluded under
the accepted `revision_time_unknown` rule, with no outcome-dependent tie-break.

## 3. Exact feature definitions

Let `G_W(t)` be eligible anchored groups whose permanent arrival lies in `(t-W, t]`.
Let `S_W(t)` be the subset whose selected version has a successful, integrity-verified
score with relevance `r_g >= 0.20`. Its sentiment is `s_g`, and age in hours is
`a_g = (t - group_first_seen_at_g) / 1 hour`. All sums below are over `S_W(t)` unless
stated otherwise. Let `D_W = sum(r_g)` and `N_W = |G_W(t)|`.

- `mu_W = sum(r_g * s_g) / D_W`.
- `rho_W,h = sum((r_g * 2^(-a_g/h)) * s_g) / sum(r_g * 2^(-a_g/h))`.
- `pos_W = sum(r_g * I[s_g > 0.20]) / D_W`.
- `neg_W = sum(r_g * I[s_g < -0.20]) / D_W`.
- `disp_W = sqrt(sum(r_g * (s_g - mu_W)^2) / D_W)`, a relevance-weighted
  **population** standard deviation, without a sample correction.

Every ratio is `0.0` when its denominator is zero. Relevance exactly `0.20` is included;
sentiment exactly `-0.20` or `0.20` is neutral for the share features. Relevance is a
weight, not a calibrated probability. Counts do not require a successful or relevant score.

The canonical feature order below matches `aggregation_contract.features` unchanged.
These are the exact **13** features; diagnostics are not additional model inputs.

| Feature | Formula | Type | Value bounds / empty value |
|---|---|---|---|
| `sentiment_mean_6h` | `mu_6h` | `float64` | `[-1.0, 1.0]`; `0.0` |
| `sentiment_mean_24h` | `mu_24h` | `float64` | `[-1.0, 1.0]`; `0.0` |
| `news_count_1h` | `N_1h` | `int64` | `[0, 2^63-1]`; `0` |
| `news_count_6h` | `N_6h` | `int64` | `[0, 2^63-1]`; `0` |
| `news_count_24h` | `N_24h` | `int64` | `[0, 2^63-1]`; `0` |
| `sentiment_recency_6h` | `rho_6h,6h`, half-life 6h | `float64` | `[-1.0, 1.0]`; `0.0` |
| `sentiment_recency_24h` | `rho_24h,24h`, half-life 24h | `float64` | `[-1.0, 1.0]`; `0.0` |
| `positive_share_24h` | `pos_24h` | `float64` | `[0.0, 1.0]`; `0.0` |
| `negative_share_24h` | `neg_24h` | `float64` | `[0.0, 1.0]`; `0.0` |
| `sentiment_dispersion_24h` | `disp_24h` | `float64` | `[0.0, 1.0]`; `0.0` |
| `source_count_24h` | distinct canonical representative `source` among `G_24h(t)` | `int64` | `[0, 2^63-1]`; `0` |
| `hours_since_latest_article` | `min(24, (t - latest eligible group arrival <= t) / 1 hour)` | `float64` | `[0.0, 24.0]`; `24.0` if no prior group |
| `news_missing_24h` | `1` iff `N_24h == 0`, else `0` | `int8` | `{0, 1}`; `1` |

The source is the permanent `GroupAnchor.source` fixed when the anchor was established,
not a union of syndicated members' sources or a later revision's source. Integer overflow,
non-finite floats, booleans masquerading as numbers and out-of-bound values fail closed;
no silent cast, clipping, or display rounding may repair them.

## 4. Missingness, failures and verified provider gaps

| Condition | Required behavior |
|---|---|
| No eligible news in 24h, with sufficient verified coverage | All counts/source count `0`; means, recency, shares and dispersion `0.0`; hours `24.0`; missing `1`. Keep the row. |
| News exists but all selected scores are below `0.20` relevance | Counts, sources and hours reflect actual eligible groups; sentiment aggregates `0.0`; missing `0`. Keep the row and report low relevance. |
| News exists but selected scores fail or are unavailable | Counts, sources and hours still reflect news; an aggregate with no valid scores is `0.0`; missing `0`. Keep the row and report scoring diagnostics. Never fabricate a successful neutral score. |
| Some selected scores succeed | Aggregate only relevant successful scores; preserve counts for all eligible groups. |
| Verified terminal gap intersects the trailing 24h window | Exclude from the prospective modeling index as `provider_gap_window`; do not emit a substitute no-news modeling row. |
| Missing, pending, corrupt or invalid provider input without matching terminal evidence | Unresolved, not no-news and not a certified gap; fail closed rather than certify the decision window. |

No forward fill, backward fill, daily copying, neutral pseudo-articles, or future revision
backfill is permitted. Keep `scoring_failure_count_6h`, `scoring_failure_count_24h`, their
rates, `provider_gap_indicator` and `low_relevance_count` outside the 13 model features.
They must reconcile to selected-version/group counts and must never redefine eligibility.

### Terminal evidence and exact overlap

A gap must be derived from verified immutable `TerminalGapEvidence` bound to the exact
`RetrievalPlan`, expected source locator, retry disposition, raw-response references,
protocol identity and coverage `as_of`. A caller-supplied gap label, forged coverage report,
HTTP error alone, or arbitrary year-2099 interval is not evidence. Preserve accepted
provider/state verification, global watermark and causal chronology checks.

The excluded gap range is the evidence's expected one-minute interval
`[interval_start, interval_end_exclusive)`. Adjacent verified gaps may be coalesced without
changing their union. The operational 30-minute release lag affects retrieval scheduling
only; neither `due_at`, `terminal_at` nor a later recovery time replaces the bound interval.
`terminal_at` remains required causal audit evidence and must be valid at the frozen
coverage `as_of`. A valid zero-line gzip is a delivered empty interval, not a gap.

For gap `[g_start, g_end)` and decision window `(t-24h, t]`, exclusion is exactly:

`g_start <= t AND g_end > t-24h`.

Thus a gap starting at `t` intersects, a gap ending at `t-24h` does not, and a gap starting
strictly after `t` does not. Coverage evidence and its cutoff are pinned inputs; later
evidence cannot silently rewrite an already published artifact. Article perturbation
tests hold that coverage evidence fixed. Newly verified coverage is a different input
snapshot, not an excuse to mutate earlier immutable results.

The future integration contract excludes these decisions from the **shared four-cell row
index before fold generation**. Labels and the five-row purge retain the underlying
continuous hourly market ordinal. This step neither creates folds nor joins market data;
later implementation tests this contract with a synthetic grid and exclusion reasons only.

## 5. Deterministic binary64 and immutable output contract

The slow reference explicitly enumerates decisions and groups. The optimized path may
accelerate membership/indexing but must preserve identical selection and scalar arithmetic:

- Order groups by `(group_first_seen_at, duplicate_group_id)` before every reduction.
- Convert validated score/relevance values and elapsed hours to IEEE-754 binary64.
  Accumulate each sum from positive `0.0`, left to right, with a binary64 result after
  every operation. No float32 intermediate, reassociation, fused multiply-add, fast-math,
  approximate reduction, or parallel reduction with a different order is permitted.
- Compute recency weight as `r_g * 2^(-a_g/h)` before multiplying by `s_g`. Compute
  dispersion in two passes: the frozen mean, then `delta = s_g - mu_W`, `delta * delta`,
  then relevance weighting and ordered accumulation. Divide before square root.
- Both paths must use the same pinned binary64 power/square-root primitives and runtime.
  Any different runtime/platform must pass the same differential gate before acceptance;
  this document does not claim universal cross-platform transcendental bit identity.
- Compare all float feature values by their 64-bit representations (**zero ULP difference**),
  integers exactly, and canonical feature-row bytes exactly. Approximate `allclose` or
  rounded CSV equality is insufficient. Normalize exact zero outputs to positive `0.0`.
  Display rounding never changes stored numbers.

Future synthetic artifacts must retain their feature schema/dtypes and input lineage,
including article/score, group, verified coverage and protocol/configuration hashes.
Reuse accepted exact-byte hashing, immutable CAS and atomic manifest-last no-overwrite
publication; identical reruns reuse verified artifacts, while altered payload or metadata
under an existing identity is an integrity failure. This is a future implementation
requirement, not authority to populate feature storage in this task.

RFC 8785 canonical JSON and exact-byte SHA-256 must remain unchanged. Integral float
tokens may serialize as JSON integers; typed hydration must restore `float64` fields
without rewriting bytes. Do not serialize an `int64` outside the existing JCS exact-safe
integer range as an imprecise JSON number: reject it at that serialization boundary.
Row feature bytes must not depend on future records or nondeterministic build timestamps;
a separate run manifest may identify the complete frozen input snapshot.

## 6. Adversarial acceptance requirements for later implementation

These tests are **specified, not newly implemented or claimed passing** by this freeze.
All later execution remains synthetic/mock-only unless separately authorized.

1. **Hand-calculated formulas:** independently specified expected values for all 13 fields,
   relevance weighting, half-life decay and weighted population dispersion, not just
   agreement between two implementations sharing an error.
2. **Window and threshold boundaries:** arrivals at `t-W`, immediately after it, exactly
   `t`, and immediately after `t`; relevance below/equal/above `0.20`; sentiment below,
   equal and above both share cutoffs; capped age, zero denominators and empty input.
3. **Causal perturbation:** add, remove or modify only articles/versions strictly after
   cutoff `T`, including same-group revisions, later duplicates and extreme scores.
   With earlier inputs, scorer identity and coverage snapshot held fixed, every feature
   row at every decision `t <= T` must remain byte-identical. Also test shuffled input
   order and valid chronological stream partitions.
4. **Permanent anchors and revisions:** duplicate groups count once per window; later
   members cannot replace a representative or increase source count; newer failed,
   missing, pending or low-relevance versions never resurrect an older success; revisions
   do not reset age. Ambiguous same-time versions retain accepted exclusion semantics.
5. **Missingness and gaps:** distinguish delivered empty gzip, genuinely no eligible news,
   failed scores, low relevance, unresolved provider inputs and verified terminal gaps.
   Test both overlap endpoints, adjacent gaps, invalid evidence, out-of-plan intervals,
   altered hashes and cutoff/chronology mismatches. Exclusion reasons must be deterministic.
6. **Dual implementation:** deterministic randomized and edge-case fixtures must produce
   zero-bit-difference float64 results, exact integers, exact dtypes and identical canonical
   bytes between slow reference and optimized implementations. Include cancellation,
   uneven weights, tiny representable values, equal-time ordering and multiple windows.
7. **Storage and validation:** reject extra/duplicate/missing schema fields, malformed UTC
   times, non-finite/out-of-range values, bad parent hashes, conflicting cache/publication
   metadata, symlinks and non-regular files. Verify incomplete writes never become visible
   as complete, and deterministic reruns preserve first-published bytes and timestamps.
8. **Baseline and authority:** all 1,003 existing tests must remain passing; Phase 1 source,
   tests, data and artifacts remain byte-identical. No network/model/real-score/market-join
   seam may be exercised; real pilot deferral and explicit authorization checks remain.

## 7. Freeze verification and next approval

For this docs/config-only commit, run formatting and lint checks, the complete existing
suite, strict repository JSON validation, RFC 8785 differential checks, and tracked-blob
identity checks against the baseline. Verify only the three authorized files changed.
Execution results are reported at handoff; the future acceptance requirements above are
not confused with those existing regression results.

The next human decision is to review this specification and separately authorize the
**offline synthetic-only Milestone 4 implementation**, its tests and local commit scope.
Until then `milestone_4_implementation_authorized` remains `false`, regardless of the
machine-readable next-action label. Real feature use, live collection, model execution,
market-data integration, backtesting and holdout evaluation need their own authorization.
