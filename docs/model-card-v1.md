# Model Card — v1 (Baseline)

This document describes **Model v1**, the baseline model used in
**ai-model-quality-framework**.

Model v1 establishes a strong reference point for accuracy-oriented performance
and serves as the comparison baseline for subsequent model versions.

For system context and training pipeline details, see
[Architecture](architecture.md).

---

## Model overview

- **Model version:** v1
- **Task:** Binary text classification (fake vs real news)
- **Model type:** Logistic Regression
- **Feature representation:** Word-level TF-IDF
- **Intended use:** Baseline production-style classifier

Model v1 is intentionally simple and interpretable.
Its purpose is not to maximize robustness, but to establish a clear,
well-performing baseline with known tradeoffs.

---

## Training data

- Source: ISOT Fake News dataset
- Text field: `text`
- Label mapping:
  - `0` — real
  - `1` — fake

Data ingestion, validation, and splitting are handled by the data layer
(see [Architecture](architecture.md)).

---

## Evaluation summary

Model v1 achieves strong performance on clean validation and test data.
Exact metrics are recorded in:

- `artifacts/reports/eval_metrics_v1.json`

Evaluation thresholds and enforcement are defined via evaluation gates
(see [Testing Strategy](testing-strategy.md)).

---

## Strengths

- High accuracy and F1 on clean, well-formed text
- Simple, interpretable linear model
- Fast training and inference
- Stable baseline for regression comparisons

These characteristics make v1 suitable as a reference model and a
control point for evaluating improvements.

---

## Known limitations

Model v1 relies on **word-level TF-IDF features**, which introduces
well-understood weaknesses:

- sensitivity to minor typos
- sensitivity to spelling variations
- sensitivity to casing and punctuation changes
- brittle behavior under noisy, user-generated text

These limitations are not accidental; they are explicitly documented
and tested.

---

## Robustness expectations

Model v1 does **not** provide robustness guarantees against
minor textual perturbations.

In particular:
- predictions may change significantly under small typos
- invariance to spelling noise is *not* enforced

This behavior is captured in the test suite:
- robustness tests exist
- v1 is explicitly excluded from strict typo-invariance requirements

See [Testing Strategy — Robustness](testing-strategy.md) for details.

---

## Failure modes

Expected failure modes include:
- degraded performance on noisy or misspelled input
- instability under adversarial or malformed text

These failure modes are considered acceptable for v1 and motivate
the introduction of Model v2.

---

## Relationship to Model v2

Model v2 is a **robustness-oriented upgrade** over v1.

Key differences:
- character n-gram features replace word-level features
- robustness expectations are explicitly enforced
- v2 must not regress relative to v1 under evaluation gates

Details are documented in
[Model Card — v2](model-card-v2.md).

---

## Additional documentation

- **[Architecture](architecture.md)** — system design and data flow
- **[Testing Strategy](testing-strategy.md)** — quality gates and robustness tests
- **[Model Card: v2](model-card-v2.md)** — robustness-focused successor model
- **[CI/CD](ci-cd.md)** — continuous enforcement of model quality
