# Phase 2 Milestone 8 — Future Holdout Accumulation and Protocol Freeze

Specification ID: `phase2-milestone8-future-holdout-v1`

Specification status: **SPECIFICATION_FROZEN** on 2026-09-27. This is a
documentation/configuration contract only. Milestone 8 claim/readiness code,
real collection, model fitting, holdout access, evaluation and publication are
**not authorized** by this freeze. Milestone 7 is **ACCEPTED** within its
existing offline, verified-synthetic-only scope at signed-off `main` commit
`42f41ecbdadae6d048d2095ec71477591cea7385`. The live pilot remains
`DEFERRED_WITHOUT_BACKFILL`; `network_pilot_authorized` remains `false`.

## 1. Development cutoff, boundary purge and single fit

`d` is the last eligible **labeled development decision row**, measured in
continuous original hourly market ordinals. All observations through `d` are
irrevocably Development; the exact Phase 2 cutoff remains unset. Selection of
`d` requires complete authentic decision features, entry open `d+1`, exit open
`d+5`, label and required market/news inputs before the final freeze. No
pre-freeze observation can later be relabeled as holdout evidence.

After the single permitted development OOF selection chooses the augmented
cell, fit that selected augmented model and its matched technical-only control
**exactly once** on the same eligible labeled Development rows through `d`,
inclusive. Fit any logistic scaler only on those rows. Reuse frozen parameters,
seeds and thresholds; no retuning or refit follows. Atomically publish and
cryptographically bind the two fitted artifacts and the final protocol before
any outcome-bearing holdout read.

Decision rows `d+1` through `d+5` are the five-row boundary purge. They are
never fitted, assigned model predictions or trades, labeled, or scored for
outcomes. A Development label at `d` exits at open `d+5`, strictly before the
first holdout decision at close `d+6`. The first holdout decision is `d+6`,
whose information must not have been fully available before final freeze.
Prospectively received purge-period articles may contribute only causal,
lagged news context to a later holdout decision; this does not permit outcome
scoring of the purge decision rows. No historical backfill can become holdout.

## 2. Future accumulation and readiness

Compute `OOF_elapsed_days` from the UTC duration of the union of the five
closed OOF validation spans, each `[first_test_decision_at - 1h,
last_test_decision_at)`, divided by 86,400 seconds. Excluded or gap hours
within those calendar spans still count. Define
`q = augmented_OOF_completed_trades / OOF_elapsed_days`. If either operand or
`q` is non-positive or non-finite, fail closed. Otherwise freeze, before any
future collection, the planned minimum duration in calendar days:

`planned_minimum_days = max(180, ceil(50 / q))`.

Readiness requires **both** the frozen planned duration and at least 50
completed trades whose scheduled exits occurred under the frozen policy.
Slower trading extends the wait; it never lowers the threshold. Require 100%
closed, authentic, hourly market candles for technical warmup, decisions,
entries, holding periods, exits and benchmark context used by either model.
Missing required candles block readiness; synthetic candles, forward/backward
filling and post-hoc row selection are forbidden. Verified provider-gap
exclusions retain their established fail-closed semantics.

## 3. Pre-claim information firewall

Before the exclusive claim, operational checks may inspect file presence and
arrival, schema, byte counts, exact hashes, duplicate IDs, candle closure and
continuity, provider outage state, scorer retry state, eligibility/coverage
counts, dependency and storage health, and outcome-free frozen-model load or
inference health. The outward readiness report may expose elapsed days and
only the boolean `completed_trades >= 50`; it must not disclose exact trade
timestamps or prediction values.

Before the claim, no process may read or reveal forward returns, labels,
trade PnL, equity curves, Sharpe, drawdown, profit factor, hit rate, benchmark
outcomes, augmented-versus-control outcome differences, or outcome-conditioned
features, explanations or article attribution. Dashboards, logs, notebooks,
alerts and debugging output are subject to the same prohibition. No
outcome-informed tuning of provider, scorer, features, models, thresholds,
costs, research gates or waiting period is permitted.

## 4. Exclusive single-use evaluation claim

The dedicated claim path is `holdout_evaluation_claim.json` **under the run
directory** of one frozen research generation. Create it atomically with
failure-if-exists/no-replace semantics; fsync both the claim file and its
parent run directory **immediately before any outcome-bearing read**. Bind
the claim to the frozen protocol, input inventory,
both model hashes, code/dependency lock, cutoff, purge and readiness proof.
The claim is irreversible: success, exception, crash, timeout, cancellation
or partial output all consume it. A routine repeat evaluation is prohibited.
A failure requires an incident record, a new protocol/research generation,
fresh artifact hashes, separate human authority and a genuinely future new
holdout; the consumed sample becomes Development evidence.

## 5. Final evaluation integrity and publication

Before the claim, freeze the exact provider/rights and retention contract,
article and score schemas, normalization/deduplication rules, point-in-time
feature contract, market/label/cost semantics, chosen augmented/control pair,
cutoff, purge, duration/trade thresholds, research gates, code commit,
dependency lock and parent manifest hashes. Actual scorer selection, numerical
research-gate approval and future collection still require separate decisions.

The eventual one-shot evaluator must capture each exact market/news input byte
sequence once, transitively verify its hash and parent manifests, and parse
only those captured bytes. It must use the same fixed execution and cost rules
for the augmented model, control and frozen baselines. Publish evaluation
artifacts through same-filesystem hidden staging, verify all staged hashes,
fsync, and atomically expose an immutable directory with its manifest written
last. Publication failure must fail closed without permitting a second look.
Engineering PASS/FAIL and research outcome remain separate statements; this
specification asserts neither.

## 6. Authorization boundary and next step

The next proposed action is `implement_milestone_8_claim_and_readiness_offline`.
It requires **separate human authorization** and, if approved, is limited to
synthetic fixtures and mock-only, outcome-free tests. This document creates no
claim and authorizes no live collection, real fitting, real-data backtest,
holdout inspection/evaluation, network call, model download or Git push.
