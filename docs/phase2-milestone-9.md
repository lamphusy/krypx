# Phase 2 Milestone 9 — One-Time Future Evaluation Specification

Specification ID: `phase2-milestone9-one-time-future-evaluation-v1`

Specification status: **SPECIFICATION_FROZEN** on 2026-09-28, under human
authority for documentation and protocol configuration only. Its parent is the
Milestone 8 `ACCEPTED_OFFLINE_ONLY` sign-off at `main` commit
`bf2cb8c179704fece6b5edbd3f95d8fe310a5aa4` (101 focused and 1,582
repository tests). This freeze defines a future `evaluate-holdout` contract;
it does **not** authorize implementing or executing that workflow, choosing a
real development cutoff, collecting data, claiming or opening a real holdout,
fitting or downloading a real model, running a real backtest, or pushing Git.
The live pilot remains `DEFERRED_WITHOUT_BACKFILL` and there is no backfill
exception. All Phase 1 files and semantics remain unchanged.

**Current implementation status: ACCEPTED_OFFLINE_ONLY.** The 2026-10-04
human sign-off authorizes final verification, governance finalization and one
local `main` commit of the synthetic-only implementation against specification
commit `54978ada26a472c4cdacf8cb0e614aacf2f25d1a`. The completion commit is the
commit containing this acceptance record. The original specification-freeze
authority above is historical; it did not grant the later implementation
authority. This acceptance does not authorize real holdout access/evaluation,
network collection, model downloads, production training or Git push.

## 1. Frozen inputs and zero-outcome preflight

An eventual, separately authorized `evaluate-holdout` invocation accepts one
immutable development run and one prospective holdout generation. It must
identify the exact frozen protocol and implementation commits, dependency
lock, article/score/market input inventory, final cutoff `d`, five-row purge,
readiness plan, selected augmented model, matched technical-only control,
saved feature schemas and column order, threshold, cost rules and gates. A
caller-supplied frame, alternate model, changed threshold or reconstructed
column order is not an acceptable substitute for the frozen artifacts.

Preflight is an outcome firewall, not a preliminary backtest. It must:

1. Verify the exact snapshot bytes and SHA-256 hashes against their frozen
   descriptors and transitive parent manifests, including article and score
   provenance, original market-row identities, schema, duplicate IDs, closure
   state and publication identity. Capture each byte sequence once from a
   descriptor-anchored regular file; do not parse a different path or a later
   read after verification. The market snapshot may be streamed opaquely for
   hashing, but preflight must not materialize holdout outcome price columns.
2. Verify a timestamp/ordinal-and-closure-only market view for 100% authentic,
   closed, hourly-contiguous candles over every required technical warmup,
   decision, entry, four-interval holding path, scheduled exit and benchmark
   mark. This continuity check must not load the holdout open/high/low/close
   values, derive forward labels, compute returns or disclose PnL. Missing,
   synthetic or filled candles, unresolved provider gaps and post-hoc row
   selection fail closed.
3. Verify frozen development fit metadata, rather than infer provenance from
   a model filename: the selected augmented/control pair was fitted once on
   identical eligible labeled Development decisions through `d`, inclusive;
   their scaler, if any, was fitted only there; their model and fit-manifest
   hashes match; and neither fit used any `d+1` through `d+5` boundary-purge
   decision or label, or any `d+6` onward holdout decision or label. The purge
   has no model prediction, trade or outcome score. Causal purge-period news
   may enter only a later holdout decision's lagged feature context.
4. Verify the Milestone 8 outcome-free readiness proof. The frozen plan uses
   `planned_minimum_days = max(180, ceil(50 / q))`, where `q` is the positive,
   finite augmented Development OOF completed-trade rate per OOF elapsed UTC
   calendar day. Elapsed time from the frozen first holdout market-candle open
   must reach that plan (hence at least 180 days), and at least 50 frozen-policy
   trades must have reached their scheduled exit opens. Count scheduled exits
   from frozen signals, ordinals and timestamps without reading exit prices or
   calculating trade returns. A slower rate extends the wait; it never reduces
   either threshold. Outward readiness may disclose elapsed days and the
   `completed_trades >= 50` boolean, not exact predictions, exit timestamps or
   outcome values.

Any missing, changed, ambiguous or non-finite prerequisite prevents claim
acquisition. Before the claim, do not compute or reveal forward returns,
labels, trade PnL, equity, Sharpe, profit factor, drawdown, benchmark outcomes
or augmented-versus-control outcome differences, including through logs,
notebooks or diagnostics. Preflight cannot be used to tune the provider,
scorer, features, model, costs, gates or waiting period.

## 2. Exclusive claim and single-use state transition

Only after a successful zero-outcome preflight, and immediately before the
first outcome-bearing holdout read, create
`holdout_evaluation_claim.json` **under the frozen development run
directory**. The claim is a generation-specific, atomic failure-if-exists
creation using no-replace semantics; fsync the claim file and its parent run
directory before opening holdout outcome prices or labels. It binds the
protocol and input-inventory hashes, code/dependency lock, cutoff/purge,
readiness proof, both fitted model/fit-manifest hashes, feature-column schema,
and destination evaluation `run_id`. The claim's exact bytes and SHA-256 are
later referenced by the evaluation manifest, not treated as a disposable lock.

The claim is consumed whether execution succeeds, fails, crashes, times out,
is cancelled, or leaves partial staged output. An existing claim makes any
routine retry fail closed, even if no final evaluation directory exists. A
second evaluation requires a documented incident, a new protocol and research
generation, fresh model/input/code artifact hashes, separate human authority,
and a **new genuinely future holdout**; the consumed sample becomes
Development evidence. Deleting or replacing the claim to retry is forbidden.

## 3. Claimed holdout execution and matched comparisons

After the durable claim, parse only the verified, captured input bytes.
Build the point-in-time holdout feature matrix for original decision ordinals
`d+6` onward, preserving the frozen eligible row set and missing-news policy.
Apply the exact saved feature-column order and types for each of the frozen
augmented and technical-only models; reject missing, extra, reordered or
non-finite feature values. Use the fitted artifacts once, without refit,
recalibration, threshold selection or explanation-driven tuning. Record
decision IDs, timestamps and probabilities in `holdout_predictions.csv`.

Replay the unchanged Milestone 7 long-or-cash, full-equity, no-overlap state
machine on continuous original hourly market context. A decision at the close
of candle `t` enters, if signaled, at open `t+1` and exits after `H=4`
open-to-open holding intervals at open `t+H+1 = t+5`. Include every authentic
open from the first eligible decision's next open through the **final required
scheduled exit open**, even where no decision was retained. The augmented
model, matched control, costs and baselines share this exact inclusive open
mark sequence, initial $10,000, eligible decision IDs and frozen execution
rules. Process a scheduled exit, then pending entry, then equity mark at an
open; evaluate a new signal only after that candle closes. A missing final
exit open is an integrity failure, not permission to drop its decision.

Run all three frozen multiplicative cost scenarios on unchanged signals and
trade ordinals: `base` is the official result (10 bps taker fee, 2 bps
slippage and 1 bp half-spread per side); `low` uses 10/1/0.5 bps; `high`
uses 10/5/2 bps. Report base, low and high side by side, never select the
best-cost result after viewing outcomes. Reconcile ledger net currency PnL,
open-level equity, total return and descriptive costs to the frozen Phase 1
metric definitions and tolerance.

On the **same** holdout performance window, run all five Milestone 7 trading
baselines under the same applicable cost scenarios: Cash; cost-aware Buy &
Hold; EMA 9/21 state rule; Momentum 24 state rule; and deterministic Random
Exposure (1,000 fixed-seed Bernoulli sequences with draws reused across cost
scenarios and the Milestone 7 summary percentiles). The technical indicators
come from verified point-in-time features. Cash has no fabricated trade/risk
ratios. Baselines and control are comparison context, not an alternate window
or a post-hoc way to redefine the final verdict.

## 4. Final evidence gates and verdict

Evaluate these conjunctive, predeclared gates exactly once for the selected
augmented model on the claimed holdout, using the official **base-cost**
results unless a gate explicitly says otherwise:

| Gate | Frozen pass condition |
|---|---|
| Readiness and sample | Elapsed duration is at least the frozen `planned_minimum_days` (itself at least 180 calendar days), and completed augmented trades are at least 50. |
| Return and comparisons | Augmented total return is strictly `> 0`; it strictly beats Cash and the matched frozen technical control on the identical window. |
| Risk-adjusted return | Augmented annualized Sharpe is strictly `> 0`. |
| Trade quality | Augmented profit factor is strictly `> 1.05`. |
| Drawdown | Augmented maximum-drawdown magnitude is `<= 20.0%` and no more than `2.0` percentage points above the control's magnitude. |
| Temporal concentration | The maximum rolling 30-day share of positive incremental trade PnL is `<= 40.0%`. |

For the rolling gate, align augmented and control ledgers on the sorted union
of UTC exit timestamps. At exit hour `e`, use each strategy's frozen-backtester
net currency PnL, or zero if that strategy has no exit; set
`p(e) = max(0, P_aug(e) - P_ctl(e))`. The denominator is `sum_e p(e)` over the
whole holdout, and zero or undefined denominator **fails**. For every UTC
calendar date from the first through last exit date, include the 30-day
half-open window `[date 00:00Z, date + 30 days 00:00Z)`; zero-exit dates remain
in the scan. The largest window numerator divided by the common denominator
must be `<= 0.40`. Do not match trades, resample or select windows afterward.

Required integrity, provenance, authentic-candle continuity, readiness and
reconciliation failures also fail closed; an undefined Sharpe or profit factor
does not become a passing zero or infinity. Failing **any** required gate
produces research verdict `FAIL` and production decision `NO-GO`. There is no
post-hoc model, scorer, feature, cost, threshold, gate or window tuning, no
second look and no automatic production approval even if every gate passes.
The low/high scenarios and all five baselines must still be published as
sensitivity/comparison evidence.

These explicit Milestone 9 inequalities supersede the older *draft* final
return, control-delta, Sharpe and profit-factor floors in the research
protocol for this specification. Its earlier proposed high-cost pass floor and
source/day/group counterfactual concentration caps are **not silently added**
to this final verdict; they may be reported as predeclared diagnostics, but
changing or adding a pass/fail gate requires a new approved protocol generation
before any outcome inspection. This specification freezes numerical semantics
only; it does not itself approve real-data gate execution.

## 5. Evaluation directory and immutable publication

Reserve one distinct `run_id` and publish exactly one immutable evaluation
directory under `artifacts/phase2/evaluations/{run_id}/`:

```text
artifacts/phase2/evaluations/{run_id}/
├── input_market_snapshot.csv
├── input_article_snapshot.jsonl
├── input_score_snapshot.jsonl
├── evaluation_models/
├── holdout_predictions.csv
├── trade_ledgers/
├── equity_curves/
├── metrics.json
├── baseline_metrics.json
├── cost_sensitivity.json
└── evaluation_manifest.json
```

The three `input_*` files are exact, verified captured bytes used for this
evaluation, not refreshed or transformed substitutes. `evaluation_models/`
holds verified copies of both frozen development fit artifacts and their
provenance/column-order descriptors. `trade_ledgers/` and `equity_curves/`
hold complete named augmented, control and baseline outputs for every
applicable cost scenario; the exact recursive inventory, including random
exposure evidence, is closed by the manifest. `metrics.json` records the
official base result, every gate operand, per-gate pass/fail, engineering
integrity status, research verdict and production decision.
`baseline_metrics.json` records all five matched-window baseline metrics and
random-exposure distributions; `cost_sensitivity.json` records low/base/high
model, control and baseline comparisons with fixed-signal checks. Every JSON
file uses strict duplicate-key/non-finite-number rejection and RFC 8785
canonicalization; undefined ratios are `null` with reasons.

Publish through a hidden same-filesystem staging directory. Write and fsync
each payload, then read back and verify every exact byte length and SHA-256
through anchored directory/file descriptors that reject symlinks, non-regular
files, unmanifested entries and path replacement. The
`evaluation_manifest.json` is written **last**. It binds the claim bytes/hash,
frozen development and parent manifest hashes, protocol/code/dependency
identities, selected pair and feature-column order, exact input inventory,
decision/performance-window identities, cost/baseline/gate versions, and the
complete recursive payload inventory with byte lengths and hashes. Fsync the
manifest and staging directory, then atomically expose the immutable final
directory with no-replace semantics and fsync its parent. Verification must
reopen the published directory by descriptor, recheck the manifest and every
listed payload, and fail on mutation, missing or extra entries. A publication
failure must not expose a valid-looking manifest-bearing partial evaluation;
it never restores the consumed claim or permits a routine retry.

## 6. Offline engineering acceptance and authorization boundary

Milestone 9 is `ACCEPTED_OFFLINE_ONLY` after 1,693 passing repository tests,
including all 253 Phase 1 tests, and 211 passing focused evaluation/dataset
tests. Final checks also cover formatting, linting, compilation, dependency
consistency, strict JSON/JSONL validation, and 49,972/49,972 local RFC 8785
binary64 differential matches. Phase 1 preservation is 48/48 source/test
blobs, or 50/50 including the two selected provider fixtures, byte-identical
to `43889acff2651c696f318ff6780455a06bbfcb35`. The explicitly authorized removal
of two incidental trailing spaces in `backtesting/baselines.py` restores the
original blob; no Phase 1 behavior or test was changed.

The accepted synthetic implementation pins the verified Development directory
inode through preflight, claim acquisition and completion. Retrieval verifies
all seven retained Development artifacts, including the frozen models and
embedded scaler parameters, against their exact bytes and hashes. Immutable
M4/M5 parents are cross-reconciled for shared minute chronology, article
versions, permanent anchors and causal score records, including partial
coverage overlaps; fully overlapping trailing windows must agree bit-for-bit
on all 13 features and provider-gap exclusions. Prior claim-generation,
source-binding, completion-rollback and fail-closed error contracts remain
enforced.

The next proposed action is `prepare_milestone_10_specification`, requiring
separate human authority. This sign-off authorizes the local Milestone 9
commit only after final checks pass; it does not authorize Milestone 10
implementation or production training. The live pilot remains
`DEFERRED_WITHOUT_BACKFILL`, `network_pilot_authorized` remains `false`, and
`real_network_calls_prohibited` remains `true`. No real claim, holdout
unblinding/evaluation, real-data research gate, model download, paid scoring,
production decision or Git push is authorized by this offline acceptance.
