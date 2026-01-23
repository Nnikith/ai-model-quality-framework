# Model Card — v2 (Robustness-Oriented Upgrade)

This document describes **Model v2**, a robustness-focused upgrade to the baseline
Model v1 used in **ai-model-quality-framework**.

Model v2 exists to address known limitations of word-level text representations
under realistic, noisy input conditions.

For system-level context, see [Architecture](architecture.md).

---

## Model overview

- **Model version:** v2
- **Task:** Binary text classification (fake vs real news)
- **Model type:** Logistic Regression
- **Feature representation:** Character n-gram TF-IDF (`char_wb`, n=3–5)
- **Intended use:** Robust production-facing classifier

Model v2 preserves the simplicity and interpretability of a linear model
while improving tolerance to spelling noise and minor textual perturbations.

---

## Motivation

Model v1 achieves strong accuracy on clean data but is sensitive to:
- minor typos
- spelling variations
- casing and punctuation changes

These behaviors are acceptable for a baseline but represent a realistic
risk in user-generated or scraped text.

Model v2 is explicitly designed to mitigate this risk.

Details of v1 limitations are documented in
[Model Card — v1](model-card-v1.md).

---

## Training data

- Source: ISOT Fake News dataset
- Text field: `text`
- Label mapping:
  - `0` — real
  - `1` — fake

The same ingestion, validation, and split logic is used for v1 and v2 to
ensure comparability (see [Architecture](architecture.md)).

---

## Feature representation

Model v2 replaces word-level features with **character n-grams**.

Key properties:
- subword patterns are preserved
- small spelling changes affect fewer features
- representation remains sparse and linear

This choice improves robustness without increasing model complexity
or inference latency.

---

## Evaluation summary

Model v2 achieves strong performance on validation and test splits.
Metrics are recorded in:

- `artifacts/reports/eval_metrics_v2.json`

Evaluation is enforced via gates that require:
- minimum absolute quality
- non-regression relative to v1
- modest improvement when the baseline is not near ceiling

Gate behavior and thresholds are described in
[Testing Strategy](testing-strategy.md).

---

## Robustness guarantees

Unlike v1, Model v2 provides **explicit robustness guarantees**.

Guaranteed behaviors:
- predictions remain stable under minor typographical errors
- casing and punctuation variations do not cause large probability swings

These guarantees are enforced via end-to-end tests against the inference API.

See:
- [Testing Strategy — Robustness](testing-strategy.md)

---

## Known limitations

Model v2 improves robustness but does not eliminate all risks.

Remaining limitations include:
- susceptibility to heavy adversarial corruption
- reliance on surface text patterns rather than semantic understanding
- no fairness or bias guarantees beyond dataset composition

These limitations are documented rather than enforced.

---

## Failure modes

Expected failure modes include:
- degraded performance under extreme text corruption
- reduced confidence calibration on out-of-distribution inputs

Monitoring is used to surface these conditions post-training
(see [Monitoring](monitoring.md)).

---

## Relationship to Model v1

Model v2 is evaluated relative to v1 under controlled conditions.

Constraints:
- v2 must meet minimum absolute quality thresholds
- v2 must not regress relative to v1
- v2 should improve robustness even when accuracy gains are marginal

This ensures that v2 represents a **meaningful upgrade**, not a parallel model.

---

## Deployment considerations

Model v2:
- uses the same serving API contract as v1
- produces artifacts in the same format
- can be swapped at serving time by changing the artifact directory

This allows safe comparison, rollback, and A/B-style evaluation.

Serving behavior is described in
[Architecture](architecture.md).

---

## Additional documentation

- **[Architecture](architecture.md)** — system design and artifact contracts
- **[Testing Strategy](testing-strategy.md)** — robustness and evaluation gates
- **[Model Card: v1](model-card-v1.md)** — baseline model and limitations
- **[CI/CD](ci-cd.md)** — continuous enforcement of quality
- **[Monitoring](monitoring.md)** — drift detection and observability
