# Phase 2 Milestone 5 — Offline Dataset Integration Specification

Specification ID: `phase2-milestone5-offline-dataset-integration-v1`

Status: **SPECIFICATION_FROZEN** — documentation/configuration only.

Human authorization recorded: **2026-09-18**.

Accepted Milestone 4 implementation on `main`:
`c177a762affb101fa37ad200fe8542fb9136f880`.

## 1. Authority and preservation boundary

Milestone 4 is **ACCEPTED**, scoped to offline synthetic feature aggregation, after
independent review against `683f76d97e132c0d245391c82317b163d61f2e32`: zero blocking
findings, 1,130 passing tests (127 aggregation, 253 Phase 1), and unchanged Phase 1
source, tests, data and artifacts. Its four reviewed files were committed unchanged.

This instruction permits that local implementation commit and a separate specification
commit modifying only this document, `config/phase2_protocol.json`, and
`docs/phase2-research-protocol.md`. It does not authorize Milestone 5 source/test
implementation, dataset construction, market-data joins, fold creation or model execution.
`milestone_5_implementation_authorized` and `milestone_5_dataset_integration_authorized`
remain `false`. The next-action label is a proposal awaiting separate human authorization.

Future implementation under this specification is **offline, synthetic-only**. No real
market/news corpus, provider call, model download, paid scoring, training, research
backtest, research gate or holdout access/evaluation is authorized. The live pilot stays
`DEFERRED_WITHOUT_BACKFILL`, with `network_pilot_authorized: false` and
`real_network_calls_prohibited: true`. No push is authorized. Historical rights, signer
pins, approvals, and accepted Milestone 3/4 numerical contracts remain unchanged.

## 2. Exact ordered feature space

The technical block is exactly the Phase 1 `get_expected_feature_columns()` output with
`RETURN_PERIODS = [1, 2, 3, 6, 12, 24]`, in this order (positions 1–24):

```text
ema_short, ema_long, ema_ratio, close_to_ema_short, close_to_ema_long,
macd, macd_signal, macd_diff, rsi, stoch_rsi, bb_width, bb_pct, atr, atr_pct,
candle_range_pct, body_return, volume_change, volume_ma_ratio,
return_1, return_2, return_3, return_6, return_12, return_24
```

The sentiment block is exactly Milestone 4's frozen order (positions 25–37):

```text
sentiment_mean_6h, sentiment_mean_24h, news_count_1h, news_count_6h, news_count_24h,
sentiment_recency_6h, sentiment_recency_24h, positive_share_24h, negative_share_24h,
sentiment_dispersion_24h, source_count_24h, hours_since_latest_article, news_missing_24h
```

`combined_feature_columns = technical_feature_columns + sentiment_feature_columns`:
exactly **24 + 13 = 37**, unique, in that order. No alphabetical sorting, inference from
all numeric columns, automatic suffixing, diagnostics, row identifiers, OHLCV context or
future-derived labels may enter the feature matrix. Controls A/B project the first 24;
augmented cells C/D project all 37 from the same prepared rows. This is a data-view
contract, not permission to train any experiment cell.

Technical values remain finite `float64` with unchanged Phase 1 formulas/settings and
warm-up removal. Compute them on continuous hourly market history before sentiment-gap
filtering, never on the compressed retained-decision sequence. Sentiment values preserve
the exact types, bounds, positive-zero and
ordered binary64 semantics in [Milestone 4](phase2-milestone-4.md): counts/source count
are non-negative `int64`; `news_missing_24h` is `int8` in `{0,1}`; the remaining eight
fields are finite `float64` with their frozen bounds. No scaling, clipping, rounding,
type widening of stored integer columns, or feature recomputation is introduced by a join.

## 3. Timestamp, exact join and shared-row contracts

Let `i` be a candle's immutable ordinal in the full continuous synthetic hourly market
snapshot, and `u_i` its UTC **open** timestamp. Phase 1's `timestamp` column is `u_i`;
it must not be renamed/reinterpreted as close time or changed in Phase 1 storage.
Define the Phase 2 decision join key `decision_at = t = u_i + 1 hour`.

The exact left join is `technical.decision_at == sentiment.decision_at`, one-to-one,
using timezone-aware UTC instants on exact hourly boundaries. Preserve the technical
left-row ordering. Reject duplicate/nonmonotonic keys, naive/non-UTC timestamps,
off-grid times, extra/missing feature columns, unmatched technical decisions, or multiple
sentiment parents for one decision. No nearest/as-of match, tolerance, local-time
conversion, string-prefix match, silent inner join or fill is permitted. A parent may
contain additional sentiment decisions, but they remain pinned lineage and never add rows.

For each otherwise-valid technical decision, the accepted M4 artifact must supply either
a verified 13-feature row or an explicit verified exclusion:

- Legitimate no-news is an existing verified M4 row: all counts/source count zero,
  means/recency/shares/dispersion `0.0`, hours `24.0`, missing `1`. Preserve it exactly.
- Low relevance and missing/failed scores retain the M4 counts, missingness and diagnostics;
  they are not missing join rows. Diagnostics are separate from the 37 features.
- A verified `provider_gap_window` has no feature payload. Record the exclusion, then remove
  that decision from the shared eligible index for **all four cells before any folds**.
  Never zero-fill it or expose it as `news_missing_24h`.
- A missing row, unverified gap claim, incomplete coverage, malformed parent or hash
  mismatch fails the entire preparation. Do not silently drop it as a certified gap.

Use the M4 gap rule unchanged: `[g_start,g_end)` intersects `(t-24h,t]` iff
`g_start <= t and g_end > t-24h`. Do not shift it by release lag or recovery time.
Verify parent bytes and recompute these classifications; a trusted-looking caller flag
or a rehashed forged row is insufficient. Scored timestamps remain audit-only as frozen;
this contract does not claim live scoring latency or authorize retrospective real scoring.

Persist `market_ordinal` and `decision_at` as non-feature row identity alongside original
OHLCV context. The ordinal is derived from the immutable full market snapshot, not a
reset pandas index or filtered row number. All cells share identical decision IDs,
ordinals, labels, exclusions, and exact market-price-context hash. Record warm-up,
provider-gap and unlabeled-tail dispositions separately, with reconcilable counts.

## 4. Target and execution alignment — unchanged Phase 1 semantics

The frozen expression is `r_t^(H) = O_(t+H+1) / O_(t+1) - 1`, `H = 4`.
Here subscripts denote candle ordinals, not an additional offset from the UTC close key.
Equivalently, for ordinal `i` and close key `t = u_i + 1h`:

- `entry_open = O_(i+1)`, `entry_timestamp = u_(i+1) = t`;
- `exit_open = O_(i+5)`, `exit_timestamp = u_(i+5) = t + 4h`;
- `gross_forward_return = exit_open / entry_open - 1.0`.

Do **not** shift entry to `t + 1h`: that would introduce a second one-hour shift.
Compute labels on the full continuous market ordinal sequence after unchanged technical
warm-up handling and **before** sentiment-gap filtering. Removed decision rows remain
available as price context for other decisions' entry/exit calculations. Never call
positional shifting on a gap-filtered frame or substitute the next retained market row.
Missing market hours fail closed; they are not sentiment gaps and are not forward-filled.

The unchanged cost-aware binary target is:

`label = int8(gross_forward_return > minimum_required_return)`.

Using Phase 1's existing ordered binary64 cost function, let `f = 0.001`,
`e = (2.0 + 1.0) / 10000.0`, and `m = 5.0 / 10000.0`. Then:

`minimum_required_return = (1 + m) * (1 + e) / ((1 - e) * (1 - f)^2) - 1`

which is `0.003105688415184993` in the pinned environment. Equality is label **0**.
Both fees and adverse entry/exit execution remain multiplicative, not an additive
cost approximation. Do not retune horizon, threshold, costs or class balance.

Exact label column order is `entry_timestamp`, `exit_timestamp`, `entry_open`,
`exit_open`, `gross_forward_return`, `label`. Timestamp fields are UTC, the three price/
return fields finite `float64` (prices positive), and label `int8` in `{0,1}`.
Labels are separate from model features. The final five market decisions lack complete
labels and are excluded only from the labeled view, not otherwise-valid inference rows.

No folds are created in this step. The future handoff must preserve the original
**five-market-row purge**, not five compressed retained-row positions. For first validation
ordinal `j`, ordinals `j-5` through `j-1` are unavailable for training even if some were
already removed by gaps; every training label must satisfy
`exit_timestamp < u_j` (the original validation candle **open**, not its close `t_j`).
Keep continuous market ordinals and shared exclusion evidence available for this check.
Do not call Phase 1 holdout-splitting workflows as part of integration; no holdout boundary,
development cutoff or research fold plan is selected by this documentation freeze.

## 5. Prepared dataset manifest and immutable publication

The future Phase 2 completion artifact is `prepared_dataset_manifest.json`, schema
`phase2-prepared-dataset-v1`, distinct from the unchanged Phase 1 manifest/schema/paths.
It is RFC 8785 canonical JSON, with strict exact fields, no duplicate keys, trailing text,
unknown versions, non-finite numbers or silent legacy migration. All digest values are
64-character lowercase hex SHA-256 over exact captured bytes; file descriptors also bind
byte lengths. Verify all dependencies before accepting a prepared bundle.

The required top-level fields are frozen below; every listed field is required:

| Fields | Binding / validation |
|---|---|
| `schema_version`, `specification_id`, `synthetic` | Exact schema above, this specification ID, and literal `true` |
| `protocol_sha256`, `code_commit` | Exact protocol JSON bytes and clean implementation commit; no self-referential commit hash |
| `market_snapshot_sha256` | Exact original synthetic full hourly OHLCV snapshot, including execution-price context |
| `article_snapshot_sha256` | Exact verified GSG `state.json` index bytes; traverse all state files, raw receipts/CAS, anchors, exclusions, chronology and gap evidence |
| `score_snapshot_sha256` | Exact canonical score-corpus inventory bytes, sorted by score ID, binding every score ID and envelope SHA-256; traverse every envelope, config and raw mock response |
| `sentiment_feature_sha256` | Exact M4 `envelope.json` bytes; traverse configuration, decisions, rows, lineage and parents, including excluded decisions |
| `combined_feature_sha256`, `labeled_dataset_sha256` | Exact serialized combined inference table and labeled table bytes, respectively |
| `market_price_context_sha256`, `row_index_sha256`, `exclusions_sha256` | Exact canonical complete execution-price context, retained row identities and exclusion/disposition ledger bytes |
| `technical_feature_columns`, `sentiment_feature_columns`, `combined_feature_columns`, `label_columns` | Exact ordered arrays from Sections 2/4; concatenation is asserted, not inferred |
| `column_dtypes` | Exact mapping for all context, feature and label columns; validate types on serialization and hydration |
| `technical_config_sha256`, `deduplication_config_sha256`, `scoring_config_sha256`, `aggregation_config_sha256`, `label_config_sha256`, `missing_data_policy_sha256`, `integration_config_sha256` | Exact canonical, retained configuration objects; all effective settings/versions are included, not just descriptive names |
| `dependency_lock_sha256`, `phase2_dependency_lock_sha256` | Exact `requirements-lock.txt` and `requirements-phase2.txt` bytes, respectively |
| `coverage_as_of`, `row_counts` | Pinned verified UTC coverage cutoff and recomputed non-negative integer disposition counts |
| `files` | Exact complete payload inventory: safe relative paths to `{sha256, size_bytes}`; no unmanifested files or non-regular objects |

`scoring_config_sha256` binds a canonical sorted inventory of every effective mock
configuration hash in the frozen score corpus, including unused verified future scores.
Each inventory entry is unique and its configuration bytes are retained/verified.
`deduplication_config_sha256` binds the accepted normalizer version, all effective
deduplication policies/thresholds and permanent-anchor policy; never create new groups
at integration time. `aggregation_config_sha256` binds the exact accepted M4 `config.json`.
`integration_config_sha256` binds this schema, hourly open-to-close mapping, exact join,
column/dtype orders, serialization policy, row dispositions and original-ordinal rules.
Changing any effective input/configuration must produce a different prepared identity.

Combined/labeled table payloads use canonical JSON objects with exact keys `columns`,
`dtypes`, `rows`; rows are ordered arrays whose values follow their explicit column array.
Combined table order is `market_ordinal`, `decision_at`, the unchanged raw-column order
(`timestamp`, `open`, `high`, `low`, `close`, `volume`), then the 37 features. The labeled
table appends the six label columns. UTC timestamps use the accepted canonical UTC
encoding. Restore integral JSON tokens to declared float64 types on hydration without
rewriting bytes; enforce int8/int64 and the accepted JCS safe-integer serialization bound.
No datetime precision loss, lossy CSV defaults, nondeterministic build timestamp or
platform-dependent pretty-printing may define identity. Retain runtime/serialization
versions in integration configuration and apply the accepted binary64 differential gate.

Counts must reconcile the full market input, initial technical warm-up, technical
decisions, gap exclusions, retained inference rows, realizable labels and unlabeled tail.
The exclusion ledger records overlapping reasons without double counting the union;
the row index binds both inference and labeled decisions and original ordinals. Hash
verification alone is insufficient: replay verified parents, join, exclusions, formulas
and labels, then require exact output bytes/schema/row order to match the manifest.

The exact manifest-byte SHA-256 is the prepared dataset identity, held by its outer
immutable publication metadata (not a self-hash inside the manifest). Publish all payloads
and dependency descriptors through accepted CAS, staging/fsync and atomic manifest-last,
no-overwrite publication. The outer storage `manifest.json` remains the final visibility
marker. Validate its metadata before payload reads, reject symlinks/non-regular objects
and mutations using accepted descriptor/inventory checks, and compare complete metadata
on collisions. Unfinished writes are not cache hits; identical reruns reuse exact bytes.
Use a separate Phase 2 namespace, never overwrite Phase 1 dataset/manifest paths.

At this freeze, `requirements-lock.txt` is
`b17b32ea58d2a8baaed70fbb9ededd984639b3b52873d194af9d2051b54af210` and
`requirements-phase2.txt` is
`f57ef4ff913e39f002daefe2b717451d342a50bfc2d409b7e8118ff53b6eeafb`.
These are offline file observations, not permission to install/upgrade dependencies.
No prepared dataset, manifest or feature join is created by this freeze.

## 6. Future synthetic acceptance criteria

After separate implementation authorization, tests must establish:

1. Exactly 24/13/37 ordered features and six ordered label columns; reject reordered,
   duplicate, missing/extra columns, booleans, non-finite values, invalid dtypes/bounds
   and label/diagnostic leakage into feature matrices.
2. One-to-one exact UTC close-key joins including open/close off-by-one controls;
   reject missing/duplicate keys, unresolved coverage and non-hourly market gaps.
3. Verified no-news rows preserved bit-for-bit; low relevance/failures distinct;
   verified gap rows excluded identically across all cell views, never neutral-filled.
4. Unchanged H=4 entry/exit prices and strict cost labels, including cutoff equality,
   full five-row tail and retained-price context across consecutive gap exclusions.
5. Original-ordinal row IDs, leakage separation and shared indexes; future fold consumers
   must not weaken purge distance by compressing rows or comparing to validation close.
6. Future-only article/revision perturbations preserve earlier feature bytes with earlier
   scores/coverage fixed. Future market outcome perturbations may change affected labels,
   never earlier technical/sentiment inputs; assert feature and label invariants separately.
7. Transitive parent/configuration/lock hash verification and semantic replay reject
   forged rehashed payloads; deterministic reruns, collision rejection, incomplete writes,
   early metadata rejection and non-regular/symlink/mutation failures remain fail-closed
   with project-specific exceptions inheriting from `CryptoAIError`.
8. All 1,130 existing tests and Phase 1 byte identity remain intact. Tests use only synthetic
   fixtures/temporary stores, deny network access and never load a holdout or invoke models.

These are specified requirements, not implemented tests or acceptance of Milestone 5.
For this freeze, rerun formatting, lint, all existing tests, strict repository JSON,
RFC 8785 checks and baseline blob comparisons. Then await separate human authorization
for `implement_milestone_5_dataset_integration_offline`.
