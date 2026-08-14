# Phase 2 Batch A engineering record

This record is limited to the offline engineering work authorized on 2026-08-14.
Milestone 0 is approved only as the engineering specification for offline Milestones 1
and 2. It does not approve provider network access, historical or forward collection,
scorer selection, model or tokenizer downloads, research-gate execution, feature
generation, model training, backtesting, future-holdout collection, or holdout access.

## Milestone 1 — accepted

Milestone 1 provides strict article and score envelopes, UTC-only timestamp validation,
dependency-free RFC 8785/JCS canonical serialization, exact-byte SHA-256 hashing,
content-addressed immutable objects, and atomic manifest-last/no-overwrite bundle
publication. Its project-specific failures inherit from `CryptoAIError`.

Acceptance evidence:

- Focused Phase 2 foundation suite: 24 passed.
- Full Phase 1 plus Phase 2 suite: 277 passed with 12 pre-existing single-class metric
  warnings.
- Formatting: passed for all `src` and `tests` Python files.
- Lint: passed for all `src` and `tests` Python files.
- Tests are protected by the repository-wide real-network connection guard.

## Milestone 2 — in progress

Only synthetic and repository-captured test fixtures are permitted. Real GDELT access
and all other external activity remain unapproved.
