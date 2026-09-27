# Phase 2 Milestone 7 — Offline Backtests and Development Report

Specification ID: `phase2-milestone7-offline-development-backtests-v1`

Specification status: **SPECIFICATION_FROZEN** on 2026-09-24. That historical
documentation-only freeze did not authorize source, tests or backtest execution;
a separate September 26 human instruction authorized verified synthetic-only
implementation. Current Milestone 7 engineering status: **ACCEPTED_OFFLINE_ONLY**
by human sign-off on 2026-09-27. The accepted Milestone 6 parent is
`b3726f3c02587b470ab85e552559091e7aeeaa42` on `main` (1,425 passing tests).
The next proposed action is `prepare_milestone_8_specification`, which requires
separate human authorization and does not permit holdout access.

## 1. Authority and input boundary

The eventual implementation may use only synthetic, fully verified Milestone 5
prepared datasets and their accepted Milestone 6 four-cell experiment publications.
Before any simulation, verify the M6 outer publication and deterministic replay, the
M5 prepared manifest and its transitive parents, the shared fold hash, each cell's
chronological OOF prediction bytes, the exact 0.50 threshold, and the complete
`market-price-context.json` hash. Accept neither arbitrary caller-supplied frames nor
unverified market-price rows. Every A/B/C/D cell must have identical OOF decision
ordinals, timestamps, labels, entry/exit timestamps, fold assignments and price
context; only model probabilities may differ. All price bars required through the
last scheduled exit open must be present, finite, positive and hourly-contiguous in
their **original market ordinals**. Removed `provider_gap_window` decision rows are
never synthesized, traded as signals, or compressed into shorter holding periods.

The accepted four cells remain A=Logistic/24 technical, B=XGBoost/24 technical,
C=Logistic/37 combined and D=XGBoost/37 combined. This milestone consumes their
already-frozen OOF probabilities; it does not refit a model, tune a threshold, select
a cell, score news, join real market data, run research gates or inspect a holdout.
Phase 1 source, data, artifacts, settings and semantics remain byte-identical.

## 2. Common performance window and execution state machine

Let `i` be a continuous original hourly market-row ordinal. The decision is made
from candle `i` only after it closes. An accepted score at that close can schedule
entry at the open of row `i+1`; with fixed `H=4`, its exit is the open of row `i+5`.
The frozen gross return is `O[i+5] / O[i+1] - 1`. If row `i` opens at UTC `u_i`,
its decision close and row `i+1` open both have UTC timestamp `u_i + 1h`; `i+1`
means the **next candle ordinal**, not an additional hour after the close. Four
complete open-to-open holding intervals end at `i+5`. Missing execution context
fails closed; do not silently discard a terminal OOF decision to improve results.

Resolve one common inclusive sequence of open marks for **all** cells, scenarios
and baselines: the next open after the earliest OOF decision through the scheduled
exit open of the final OOF decision. Preserve every original hourly market open
between those endpoints, even where no OOF decision was retained. There are `N+1`
equity marks and `N` open-to-open returns. The first mark has no preceding interval;
costs paid at that first open are included in the first interval return. Every mark,
trade and net return must reconcile with the common initial capital of $10,000.

At each market open, the event order is **process a scheduled exit → process a
pending entry → mark candle-level equity**. Only after that row's candle later
closes may its signal be evaluated; place any resulting order for the following
candle's open. A buy signal is a finite class-1 probability `>= 0.50`. Trade long or stay
in cash: invest 100% of available equity after the entry fee, with no borrowing,
shorting, partial sizing or overlapping trades. Ignore a new buy while a position
or entry order is active. An exit at one open may be followed by a newly scheduled
entry only at the **next** open. A trade must close at its scheduled `i+5` open;
an unresolved position/order at the common window end is an integrity error.

The overall OOF backtest is one chronological state-machine run across all five
validation segments, without resetting cash or position at fold boundaries.
Separately reported fold-level backtests reset to $10,000 independently in each
validation segment; no equity or position is carried across a purge boundary.
Fold-local results are diagnostics, never multiplied or averaged into the overall
continuous OOF equity curve.

## 3. Frozen multiplicative transaction costs

For each side let `f` be the taker-fee fraction and let
`e = (slippage_bps_per_side + half_spread_bps_per_side) / 10,000`.
For entry at `i+1` and exit at `i+5`:

- `entry_fill = O[i+1] × (1 + e)` and the entry fee is charged against available cash.
- `exit_fill = O[i+5] × (1 - e)` and the exit fee is charged against gross sale proceeds.
- `net_growth = (exit_fill / entry_fill) × (1 - f)^2`;
  `net_trade_return = net_growth - 1`.

| Scenario | Taker fee per side | Slippage per side | Half-spread per side | `e` |
|---|---:|---:|---:|---:|
| `low` | 10 bps (`0.001`) | 1 bp | 0.5 bp | `0.00015` |
| `base` (official) | 10 bps (`0.001`) | 2 bps | 1 bp | `0.00030` |
| `high` | 10 bps (`0.001`) | 5 bps | 2 bps | `0.00070` |

Keep the fee constant across all three scenarios. Reject negative, non-finite or
out-of-contract rates. Do not substitute additive cost subtraction for the frozen
multiplicative model, and do not deduct the same costs again from already-net
trade or interval returns. The base scenario is the official direct-comparison
result; low/high are sensitivity analyses, not choices after viewing outcomes.

## 4. Five trading baselines on the same window

Use the same full market context, eligible OOF decision set, $10,000 capital,
threshold, horizon, price marks and cost scenario as the four cells.

1. **Cash:** no orders, zero total return, zero costs and exposure. Undefined
   trade/risk ratios are `null` with an explicit reason, not fabricated zeros.
2. **Buy and hold:** one cost-aware full-equity entry at the first common open and
   one cost-aware exit at the final common open; mark every intervening open.
3. **EMA 9/21:** the frozen Phase 1 **state** rule `ema_short > ema_long` at each
   eligible OOF decision, not a signal only on the crossing candle. Apply the
   same fixed-horizon, non-overlapping engine.
4. **Momentum 24:** the frozen Phase 1 state rule `return_24 > 0` at each eligible
   OOF decision; use the same fixed-horizon engine.
5. **Deterministic random exposure:** exactly 1,000 Bernoulli signal sequences
   per matched cell, indexed by the same OOF decisions. For cell `X`, freeze
   `p_X = mean(probability_X >= 0.50)`. Simulation `j` uses a fresh
   `numpy.random.default_rng(RANDOM_SEED + j)`, `RANDOM_SEED = 42`,
   `j = 0..999`, and draws one signal per OOF row with probability `p_X`.
   Replay each sequence under low/base/high with identical draws across costs;
   report median, 5th and 95th percentiles for total return, Sharpe and maximum
   drawdown, plus the fraction of random total returns at least the matched
   cell's return. The non-overlap engine determines realized exposure, so it
   need not equal `p_X`.

Cash, buy-and-hold, EMA and momentum paths are common to all cells for a given
scenario; the random distribution is cell-specific. Technical indicators come
from the verified point-in-time prepared features, never recomputed using future
prices or provider-gap rows. A baseline cannot use a different performance
window or eligible decision index to flatter a comparison.

## 5. Complete Phase 1-compatible metric contract

Compute every metric on net open-to-open interval returns and completed net trade
cash flows. Hourly annualization uses 8,760 intervals/year and annual risk-free
rate `0.0`. Preserve Phase 1's `null` plus warning convention whenever a ratio is
undefined; never serialize NaN or infinity.

| Metric | Frozen definition |
|---|---|
| Total return | `final_equity / 10000 - 1` |
| Annualized return | `(final_equity / 10000)^(8760/N) - 1`, `N > 0` |
| Annualized volatility | Sample standard deviation of `N` interval returns (`ddof=1`) × `sqrt(8760)`; `null` for fewer than two intervals |
| Sharpe ratio | Mean excess interval return divided by sample interval-return standard deviation, × `sqrt(8760)`; `null` if undefined |
| Sortino ratio | Mean excess interval return divided by `sqrt(mean(min(excess, 0)^2))`, × `sqrt(8760)`; `null` if undefined |
| Maximum drawdown | Most negative `equity / running_peak - 1`, with $10,000 included as the initial peak |
| Maximum drawdown duration | Longest underwater span in open-to-open intervals, including an unrecovered terminal drawdown |
| Calmar ratio | Annualized return / absolute maximum drawdown; `null` if undefined |
| Profit factor | Gross positive **net currency PnL** / absolute gross negative net currency PnL; `null` when no losing trade exists |
| Completed trades | Number of closed trades; open orders/positions at termination are errors |
| Win rate | Fraction of completed trades with positive net return; `null` for no trades |
| Market exposure | Mean prior-interval held/not-held flag over all common open-to-open intervals |
| Turnover | Sum of entry and exit **market-price** notionals divided by mean open-mark equity |
| Average holding period | Mean completed-trade `exit_ordinal - entry_ordinal` in candles; `null` for no trades |
| Total estimated costs | Currency sum of entry/exit fees and adverse entry/exit execution shortfalls |

The trade ledger records signal, entry and exit timestamps, original ordinals,
market/fill prices, quantity, gross and net returns, each fee/execution cost,
net currency PnL and post-exit equity. The candle-level equity curve records all
common-window opens, equity, position state, exposure and one return per interval.
Its compounded returns, ledger PnL and final equity must agree within the frozen
Phase 1 reconciliation tolerance. `total_estimated_costs` is descriptive, not an
additional deduction. Preserve classification context separately from trading
performance. Cash's zero volatility, no trades and no losses do not become an
infinite Sharpe or profit factor.

## 6. Cost sensitivity, ablation and fold consistency

Backtest every A/B/C/D cell and deterministic baseline under **all three**
scenarios. Require the exact same price window, OOF row IDs, capital and event
sequence at each cost level. Cost changes may affect equity and metrics, not
signals or trade entry/exit ordinals. Where all fills are valid and fee is fixed,
the same trade sequence must not have a higher net return at higher costs.

For each scenario and each metric where a difference is meaningful, report
`C - A` as **linear incremental value** and `D - B` as **nonlinear incremental
value**. Return deltas are percentage points (`100 × (R_aug - R_control)`), not
relative percentage growth; risk/cost/trade-count deltas retain their stated
units. Do not turn a `null` parent metric into zero or an apparently numeric
delta. Include matched controls side by side, not only augmented cells.

For each of the five folds, run an independent base-cost backtest per cell on
that fold's validation predictions, resetting capital and position. Report
per-fold A/B/C/D return, completed trades and the C–A and D–B return deltas;
also report the count of positive deltas, median delta and largest positive
fold contribution divided by the sum of positive deltas (`null` if that sum is
zero). These are **descriptive engineering outputs**. Proposed numerical
development gates in the research protocol remain unapproved and must not be
silently evaluated, used to select a winner or reported as a research verdict.

## 7. Immutable report publication and layout

Milestone 6 publications use an exact immutable inventory. **Never append**
Milestone 7 files to an existing M6 `run_id` or rewrite its manifest. Publish a
new, distinct M7 report `run_id` under the requested common root
`artifacts/phase2/runs/{run_id}/`, with its own exact outer schema
`phase2-backtest-publication-v1` and verifier. The five required payload files are:

| File | Required content |
|---|---|
| `strategy_metrics.json` | Four cells × three scenarios; full metrics, trade ledgers, open-level equity curves, exact common performance window and reconciliation diagnostics |
| `cost_sensitivity.json` | Low/base/high per-cell metrics and frozen-signal/cost monotonicity checks |
| `baseline_metrics.json` | Cash, buy-and-hold, EMA, momentum under all scenarios; deterministic 1,000-run random summaries for each matched cell/scenario |
| `ablation_report.json` | C–A and D–B direct deltas, five fold-local comparisons and consistency summaries; no gate verdict |
| `development_report.md` | Human-readable, synthetic-only report with the layout below and exact references to the four JSON payload hashes |

The M7 `manifest.json` is written **last** and is not an unmanifested sixth
payload. Its exact metadata binds the distinct M7 and source M6 run IDs, M6
outer manifest bytes/hash and experiment ID, M6 fold and prediction hashes,
M5 prepared-dataset ID and market-price-context hash, common decision/window
identities, all cost/baseline/metric configuration hashes, implementation source
and dependency-lock hashes, and each of the five file names with exact SHA-256
and byte length. Canonicalize JSON with RFC 8785; hash the exact UTF-8 bytes of
Markdown. Capture each file once for hash/parse, reject duplicate keys and
non-finite JSON numbers, and replay the verified synthetic parents and outputs.
Use same-filesystem hidden staging, payload fsync/readback, manifest-last fsync,
and atomic no-replace publication. Reject collisions, symlinked ancestors,
non-regular/unmanifested entries, incomplete writes, mutation during verification
and altered manifest metadata with project-specific errors. A failed publication
must never leave a manifest-bearing valid report. Existing M6 runs remain
untouched and continue to verify with their original exact schema.

The `development_report.md` layout is: evidence boundary and parent hashes;
shared window/folds and coverage exclusions; four-cell classification context;
base-cost strategy/risk/trading table; low/base/high sensitivity; five baseline
comparisons including random distributions; C–A/D–B ablations and fold
consistency; integrity/engineering checks; limitations and required next
authorization. Distinguish **Engineering Status** (`PASS`/`FAIL` for contract
verification) from **Research Outcome** (`NOT_EVALUATED_SYNTHETIC_ONLY`). Do not
call a synthetic PnL, gate, benchmark win or apparent alpha real research
evidence. No model choice, production claim or holdout result belongs here.

## 8. Frozen offline acceptance tests and stop conditions

Before the later implementation could be accepted, synthetic fixture tests had to prove
identical A/B/C/D and baseline windows, exact `i+1`/`i+5` execution boundaries,
exit-before-entry ordering, full-equity no-overlap behavior, final exit context,
and no trades on excluded provider-gap decisions. Perturbing prices or scores
strictly after a decision must not change earlier decisions or fills; changing
a single retained parent/output byte must break report verification. Test exact
low/base/high fills and multiplicative fees, cost monotonicity with fixed
signals, independent fold resets versus continuous OOF, the EMA state rule,
momentum boundary, 1,000 deterministic seeds, cash/no-trade null metrics,
ledger/equity/cost reconciliation, collision and post-rename failure handling,
and deterministic byte-identical reruns. Fail closed on malformed timestamps,
rates, prices, ordinals, labels, gaps, metrics, JSON or provenance.

The September 24 freeze **alone did not** authorize those tests or the engine.
The separate September 26 instruction authorized them on verified synthetic
fixtures, and the September 27 human sign-off accepts that engineering result
offline only. None of these steps authorizes real collection, scoring, model
downloads, real-data training/backtests, research gates, future collection,
holdout access, paid services or Git push. The live pilot stays
`DEFERRED_WITHOUT_BACKFILL` with `network_pilot_authorized: false` and
`real_network_calls_prohibited: true`.

## 9. Final offline engineering acceptance

Human authority accepted the fixed-horizon four-cell backtest engine, low/base/high
cost sensitivity, five trading baselines, side-by-side C–A and D–B ablations,
reconciliation diagnostics, immutable report publication and cross-platform
atomic no-replace storage. The implementation and report commit is
`e84944721f5ba2199992c7fc547bc0c868436056`; the subsequent storage-fix
commit is `29f311660c047cc31b365fd4029e788bff9150a9`. This is an
**engineering acceptance on verified synthetic inputs**, not a research outcome,
model selection, real-data backtest or holdout result.

Final verification: **1,481/1,481** repository tests passed, including **253/253**
Phase 1 tests. Against base `43889acff2651c696f318ff6780455a06bbfcb35`,
the selected **50/50** source/test/fixture blobs remain byte-identical: **48/48**
existing non-Phase-2 source/test paths and two GSG fixture JSONL files. This
selected baseline is not a claim that all 50 were part of the original Phase 1 tree.
Strict repository JSON validation passed, and the local Node.js RFC 8785
binary64 differential matched **49,972/49,972** cases. Atomic no-replace
publication uses descriptor-relative macOS `renameatx_np(RENAME_EXCL)` and Linux
`renameat2(RENAME_NOREPLACE)`, with a syscall fallback on supported Linux
architectures and an on-filesystem success/collision capability probe. Linux
paths were exercised with offline simulations; the final full suite ran on
Darwin arm64. No real network call, model download, real-data backtest, holdout
access or push was performed as part of this acceptance.

`next_action: prepare_milestone_8_specification` is a governance placeholder,
not authorization to begin Milestone 8. Its specification, any future holdout
planning or execution, and live-pilot resumption each require separate explicit
human authorization.
