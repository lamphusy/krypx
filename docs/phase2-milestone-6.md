# Phase 2 Milestone 6 — Offline Four-Cell Experiment Engine

Specification ID: `phase2-milestone6-offline-four-cell-experiments-v1`

Specification status: **SPECIFICATION_FROZEN**. Implementation status:
**ACCEPTED_OFFLINE_ONLY** after independent review and human sign-off on 2026-09-24.

Human authorization recorded: **2026-09-20**. Accepted Milestone 5 base on `main`:
`d7cfa57542ea6c44c0b6dead7f76dc376e8d25d2` (1,280 passing tests).

## 1. Authority and evidence boundary

The September 20 instruction authorized this specification and direct implementation of
`src/crypto_ai/phase2/experiments.py` with adversarial tests in
`tests/phase2/test_experiments.py`. Only fully verified synthetic Milestone 5 prepared
datasets may be used. Local fitting of the already installed, frozen LogisticRegression
and XGBoost classifiers is explicitly permitted on those synthetic fixtures. This is
not authorization for a sentiment scorer, real model/data research or model downloads.
The separate September 24 human sign-off accepts this offline implementation after
independent review and authorizes one local commit on `main`, without push.

Do not read real market/news or pilot data, access a holdout, make network calls, install
dependencies, run trading backtests, select a model from outcomes, execute research gates,
or push repository changes. The live pilot remains
`DEFERRED_WITHOUT_BACKFILL`; `network_pilot_authorized` is `false` and
`real_network_calls_prohibited` is `true`. All Phase 1 source, tests, data and artifacts,
accepted M3–M5 contracts, feature/label formulas and historical governance remain intact.
The original September 20 implementation instruction did not request a local commit;
the September 24 sign-off supplies that authority. Synthetic run artifacts are not
production models or evidence that news improves trading performance.

The future real experiment still requires its separate prospective corpus, at least
730 consecutive modeled UTC days after news warmup, approved research gates and explicit
execution authority. No synthetic fixture waives these requirements. Acceptance here is
limited to the reviewed offline engineering implementation, not real research execution.

## 2. Frozen four-cell matrix

| Cell | Classifier | Exact ordered input columns | Matched comparison |
|---|---|---|---|
| A | Logistic regression | Milestone 5's 24 technical columns | Control for C |
| B | XGBoost | Same 24 technical columns | Control for D |
| C | Logistic regression | Same 24 technical + 13 sentiment columns | C versus A |
| D | XGBoost | Same 24 technical + 13 sentiment columns | D versus B |

The technical and sentiment orders are unchanged from
[Milestone 5 Sections 2–4](phase2-milestone-5.md). No timestamp, ordinal, price context,
label, coverage diagnostic or automatically selected numeric column enters a feature
matrix. Every cell consumes identical labeled decision identities, labels, exclusion
ledger, market-price-context hash, validation rows and purge identities. Legitimate
zero-plus-indicator no-news rows stay included; verified `provider_gap_window` rows
were removed once by M5, before any fold construction, and cannot reappear.

The binary label remains the strict Phase 1 H=4 cost threshold
`gross_forward_return > 0.003105688415184993`. Entry/exit prices are never recomputed from
a compressed retained index. At original candle ordinal `i`, decision time is its UTC
open plus one hour, entry is original open `i+1` at that decision instant, and exit is
original open `i+5`. M6 consumes verified M5 labels, not a new target definition.

## 3. Shared expanding folds and original-market purge

Create **one** canonical `folds.json`, bind its exact-byte SHA-256 to the exact
`prepared_dataset_manifest.json` bytes and shared row/context identities, and require
all four cells to consume it. There are exactly five expanding chronological folds.
No cell reconstructs an independent split or removes an inconvenient validation row.

For `N` verified labeled rows and an explicitly fixed validation block length `k`, the
first validation position is `N - 5*k`. Fold `f` (zero-based) validates the next `k`
consecutive retained labeled positions starting at `N - 5*k + f*k`. Validation blocks
are disjoint, adjacent in the retained sequence and cover its final `5*k` rows exactly.
Training is an expanding eligible prefix, not a sliding window; validation never
participates in fitting or scaler statistics for its own fold.

Let `j` be the first validation row's original full-market ordinal and `u_j` its UTC
candle **open**. Purge the five original ordinals `j-5` through `j-1`, whether or not
some were already excluded as gaps. Retain for training only labeled earlier rows with
`market_ordinal < j-5`, and assert every training `exit_timestamp < u_j`. Comparing to
decision close `u_j+1h` or subtracting five positions from the filtered index is wrong.
Bind the original purge identities and the retained subset separately so exclusions
cannot shorten separation. Empty training, insufficient rows, overlapping folds,
non-expanding histories (including equal eligible training counts in successive folds)
or chronology/label-boundary violations fail closed.

### Explicit synthetic fixture sizing is not research retuning

The frozen research rule is still **2,098 validation rows per fold**. The accepted
`SyntheticMarket` generator is capped at 10,000 hourly rows, which cannot supply five
blocks of 2,098 rows plus nonempty training. M6 therefore exposes an explicit
`fixture_test_rows` positive integer in `[1, 2098]` solely for synthetic engineering.
Its default remains `2098` and must fail if a synthetic dataset is too small. There is
no automatic shrinking, outcome-based choice, search or change to the research rule.
Every fixture run records its selected size, synthetic-only purpose and immutable fold
identity; such a result cannot be represented as the real 2,098-row research evaluation.

## 4. Frozen classifiers and fold-local scaling

Use the unchanged objects at `experiment_matrix.logistic` and
`experiment_matrix.xgboost` in `config/phase2_protocol.json`; reject substitutions.
All effective parameters are frozen, not merely user-supplied constructor arguments.

LogisticRegression uses the scikit-learn 1.9.0 effective contract:

```text
C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, l1_ratio=0.0, max_iter=1000, n_jobs=None,
penalty="deprecated", random_state=42, solver="lbfgs", tol=0.0001,
verbose=0, warm_start=False
```

Its StandardScaler uses `copy=True`, `with_mean=True`, `with_std=True`, fitted only on
that cell's current training fold. Create a fresh scaler/classifier for every fold;
never fit on validation, the whole dataset, another fold's expanded history or another
cell. Record the training row identity and scaler statistics so leakage can be tested.

XGBoost uses:

```text
objective="binary:logistic", eval_metric="logloss", n_estimators=300,
learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
random_state=42, n_jobs=-1
```

No early stopping, validation-based model selection, parameter overrides, balancing,
feature tuning or full-history/production refit is introduced. Installed dependency
versions, exact lock bytes, effective configuration and implementation-source hashes
are retained. Cross-version/platform bit-identity is not assumed; differing required
runtime or source provenance fails closed rather than silently migrating an artifact.

If a training fold has one label class, follow the explicit Phase 1 engineering
fallback: do not fit either classifier; emit the constant probability equal to that
class and an explicit single-class-training diagnostic. This is neither a neutral
fallback nor a successful trained model. A model exception, invalid probability or
unexpected output shape is an error, not permission to substitute a constant.

## 5. Continuous out-of-fold predictions and classification

Each validation row has exactly one probability for each cell, in the shared
chronological sequence, with its original ordinal, UTC `decision_at`, entry/exit
timestamps, fold number and unchanged label. The original candle open is derived as
`decision_at - 1h` and verified against the parent; it is not a duplicated prediction
column. Exact prediction-column order is `market_ordinal`, `decision_at`,
`entry_timestamp`, `exit_timestamp`, `fold_number`, `actual_label`, `probability_score`,
`predicted_label`, `signal`. Probabilities hydrate as finite binary64 values in `[0,1]`;
predicted labels are exactly `int(probability >= 0.5)`. Positive-class probability must
refer to class `1`, not an unchecked positional class assignment. Reject missing,
duplicated, reordered, out-of-range or non-finite outputs before publication.

Continuous OOF means concatenating all five disjoint validation blocks in order.
Provider-excluded market hours remain absent: do not fill them, reset labels or claim
continuous authentic market coverage. Evaluate aggregate metrics once over the full
concatenated OOF sequence, **not** as the mean of per-fold metrics.

Retain the complete unchanged Phase 1 classification metric set per fold and overall:

- Accuracy and balanced accuracy.
- Precision, recall and F1 for each of classes `0` and `1`.
- Binary log loss, ROC AUC, precision-recall curve AUC and Brier score.
- Confusion matrix in fixed label order `[0,1]`.
- Positive-label rate and predicted-positive rate.

PR AUC means area under the precision-recall curve, not average precision. Undefined
single-class ROC AUC/PR AUC are JSON `null` with an explicit diagnostic, never NaN or
a made-up score. Zero-division precision/recall/F1 follow the frozen Phase 1 value `0`.
No Sharpe, PnL, trading return, gate verdict, winner selection or holdout result is
generated by this classification-only engineering step.

## 6. Global tree feature importance

For each XGBoost cell, derive global `gain`, `weight` and `cover` from every fitted
fold's booster using its exact ordered feature names and raw `weight`, `total_gain`
and `total_cover` statistics. Unknown feature names, negative or
non-finite importance, or inconsistent split-count evidence fail closed. Include all
24 or 37 features in frozen order, with zero values for features absent from every tree.

For feature `q` and fold `f`, let `w_fq` be split count (`weight`), `G_fq` total gain
and `C_fq` total cover. Across fitted folds:

```text
weight_q = math.fsum(w_fq for f in fitted_folds)
gain_q   = math.fsum(G_fq for f in fitted_folds) / weight_q if weight_q > 0 else 0.0
cover_q  = math.fsum(C_fq for f in fitted_folds) / weight_q if weight_q > 0 else 0.0
```

Use a fixed fold/feature reduction order. Weight is summed; average gain and cover
are split-count weighted, not averaged equally across folds. Reading totals directly
avoids multiplying separately rounded per-fold averages back into approximate totals.
Preserve per-fold
evidence and global output. A single-class fold contributes no booster or split counts;
an all-single-class run has explicit no-fitted-booster diagnostics and zero tree
importance. For logistic cells report **not applicable**, rather than fabricating tree
importance from coefficients. Importance is synthetic training evidence, not causal
attribution or evidence about real articles.

## 7. Manifest-bound provenance and atomic run publication

Only an accepted `DatasetStore.get()` path that verifies the complete synthetic M5
publication and transitive raw/article/mock-score/aggregation dependencies is an input.
A caller-supplied DataFrame, rehashed forged parent, real-input flag or superficially
valid candidate manifest cannot bypass that boundary. Resolve parents before fitting.
Capture/hash/parse each retained payload from the same byte sequence.

The repository artifact namespace is `artifacts/phase2/runs/{run_id}/`, never Phase 1
run paths; tests pass explicit isolated temporary runs roots to `ExperimentStore`.
Run IDs are safe UTC identifiers with a random suffix to avoid same-clock collisions;
the chosen ID is bound into the outer `manifest.json`, not the deterministic inner
experiment identity. Retain:

- `experiment_manifest.json`, binding exact payload descriptors, the prepared manifest,
  shared folds, frozen configuration and parent/configuration identities. Its exact-byte
  hash is the deterministic `experiment_id`; the outer publication binds that identity
  to its independently generated `run_id`.
- `prepared_dataset_manifest.json` as the exact verified M5 bytes.
- `folds.json`, `config.json`, `input.json`, `protocol.json`, `environment.json` and
  `source.py`, binding synthetic invocation, fold plan, effective policy, runtime/locks,
  implementation source bytes and the recorded engine base commit.
- Exact `requirements-lock.txt` and `requirements-phase2.txt` dependency bytes.
- For each A/B/C/D cell, `predictions.json`, `metrics.json`, `importance.json` and
  `training.json` under `cells/{cell}/`.

All JSON is strict canonical RFC 8785, with no duplicate keys, non-finite values,
implicit legacy migration or trailing data. Generated artifact schemas are exact.
The caller's fixture `protocol.json` is an explicitly synthetic canonical object whose
additional fixture annotations remain hash-bound; it is not a live authorization record
or a replacement for repository governance. Digests are lowercase
64-hex SHA-256 over exact retained bytes, with exact sizes and complete inventories.
The recorded base commit plus source hash identifies the exact reviewed engine bytes;
the completion commit SHA is reported after creation rather than embedded in its own
manifest or this self-referential sign-off record.

Verification includes deterministic synthetic replay/refitting from verified parents,
not just digest checks or trust in caller-supplied metrics. Loading and publication
therefore intentionally require the recorded engine base commit to remain locally
resolvable, the exact source, compatible pinned runtime and intact parent store, and
can be computationally expensive. A later `HEAD` alone does not invalidate a run:
verification preserves its recorded base commit rather than substituting current `HEAD`.
Preserve reproducible invocation
and per-fold training/scaler evidence. Do not load pickles or persist production-ready
models; no full-history refit or holdout artifact is created.

If validation after the atomic rename fails, invalidate only the directory pinned
by the still-open staging descriptor: unlink `manifest.json` and fsync. Do not
rename or remove the public run name, recursively delete mutable child names,
or clean a hidden staging name by path during rollback; a concurrent swap could
make any such name refer to an unrelated directory. An invalidated run-ID or
hidden-stage tombstone may retain payload bytes, but it cannot retain its
completion manifest or hydrate as a completed publication. Separately
authorized, ownership-aware residue cleanup is required to reclaim those bytes.
An unrelated replacement remains untouched.

Stage all payloads in a hidden directory on the destination filesystem. Write/fsync,
read back and hash-verify payloads before creating the outer `manifest.json` last.
Sync directories and publish atomically with no-replace semantics. Invalid or interrupted
work cannot become a visible completed run. Reject symlinked ancestors, non-regular
objects, unmanifested entries and detected directory/payload mutation through accepted
descriptor and before/after inventory protections. A run ID is an exclusive claim:
every existing destination is rejected, even for identical deterministic experiment
bytes. Conflicting metadata or payloads therefore cannot become idempotent cache hits.
An explicit new run ID may bind the same deterministic experiment identity without
overwriting or minting different timestamps inside an existing immutable run.

After staging-directory fsync, recapture the complete staged tree before rename.
After atomic rename and parent-directory fsync, recapture through the requested run
path and recheck the pinned root/run identities before reporting success. An inode
held by a descriptor is not proof that it still occupies the requested pathname.

## 8. Adversarial acceptance and offline verification

Tests must cover exact 24/37-feature inputs and shared row/label/fold identities;
original-ordinal purge with exclusions around boundaries; train exits compared to
validation open; fold-local scaler isolation; frozen model parameters; single-class
fallback and undefined metrics; probability shape/class/range/type failures; chronological
OOF completeness; weighted importance with unused features; parent/configuration/source
and payload tampering; deterministic reruns; collisions, interrupted staging and unsafe
filesystem objects. All data and stores are synthetic temporary fixtures. Tests deny
network access and do not inspect historical or prospective holdouts.

Required verification is whitespace, Black, Ruff, compilation, dependency consistency,
focused M6 tests, the complete repository suite, strict JSON/JSONL validation,
RFC 8785 differential checks and Phase 1 byte identity. All 1,280 pre-existing tests,
including the 253 Phase 1 tests, must remain passing. Record actual counts and limitations
after execution; this specification does not pre-claim new results or acceptance.

## 9. Engineering verification results

Verified September 24, 2026 against accepted M5 commit
`d7cfa57542ea6c44c0b6dead7f76dc376e8d25d2`. These are implementation verification
results; the separate acceptance sign-off is recorded below. They are not a research outcome.

| Command / check | Exact result |
|---|---|
| `git diff --check` | Exit 0; no output. New files separately checked for trailing whitespace. |
| `.venv/bin/black --check .` | `90 files would be left unchanged.` |
| `.venv/bin/ruff check --no-cache .` | `All checks passed!` |
| `.venv/bin/python -m compileall -q src tests` | Exit 0; no output. |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `.venv/bin/pytest tests/phase2/test_experiments.py` | `145 passed in 98.00s (0:01:37)` |
| `.venv/bin/pytest --ignore=tests/sentiment --ignore=tests/phase2` | `253 passed, 12 warnings in 5.02s` |
| `.venv/bin/pytest` | `1425 passed, 12 warnings in 341.10s (0:05:41)` |
| Strict repository JSON/JSONL validation | 25 JSON files and 2 JSONL files valid; all six malformed controls rejected. |
| RFC 8785 differential against installed Node `JSON.stringify` | 49,972/49,972 finite binary64 values matched; four nested/Unicode canonical vectors matched. |
| Accepted-base blob comparison | 114/114 pre-existing non-governance files byte-identical, including 88/88 source/test files and 50/50 Phase 1 source/test files. |

The 12 existing warnings are undefined ROC/PR AUC in the original synthetic Phase 1
single-class integration fixture. Pip also reported an unwritable local cache and
disabled that cache; dependency consistency still passed. No dependencies were changed.
The strict JSON check rejects duplicate keys, non-finite numbers/overflow and trailing
prose; the differential uses deterministic seed `20260919`, IEEE-754 bit patterns and
four nested/Unicode vectors, entirely locally.

New tests cover genuine synthetic GSG → M4 → M5 → M6 lineage, original-ordinal purges,
fold-local scaling, frozen parameters, continuous OOF identities, deterministic reruns,
strict probabilities and metric inputs, explicit single-class behavior, weighted tree
importance, rehashed semantic forgeries, manifest-last/no-replace publication, incomplete
writes, malformed metadata before payload reads, late FIFO/symlink injection, payload
mutation during fsync, detached output roots and later-HEAD historical replay.
All injected publication races fail with `ExperimentIntegrityError`; invalid post-rename
runs are invalidated through their pinned descriptors, ambiguous rename completion is
handled, and concurrent replacement directories retain their bytes and inode.
Non-publishable payload tombstones are possible, never completed runs. Gap-shaped
equal-length training histories fail with `ExperimentSplitError`.

Implementation SHA-256:
`939ea7f56273186f5a2646eae9f331345e5db29fa131915613bafd4a24b03cf7`.
Test SHA-256:
`087c8c2ace2a93b0f03e6b506f3d82c432c1aa1d5c585287d7f9a76364e04f21`.

All run artifacts were isolated temporary synthetic fixtures, not repository research
artifacts. Independent review found zero blocking findings against the accepted M5 base;
the September 24 human sign-off authorizes one local commit with message
`feat(phase2): implement four-cell experiment engine and shared fold evaluation`.
Its SHA is reported after successful creation, not embedded here. No push, real
network/data access, model download, trading backtest or holdout evaluation was
performed by M6. Milestone 7 specification and implementation remain unauthorized;
the next action requires separate human authority.
