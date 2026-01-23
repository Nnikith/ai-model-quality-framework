# Monitoring

This document describes how **ai-model-quality-framework** approaches post-training
monitoring and observability for machine learning models.

Monitoring is designed to **surface signals**, not to make deployment decisions.
Warnings are produced as structured reports and are intentionally non-blocking.

For system context, see [Architecture](architecture.md).

---

## Scope

This document covers:
- what is monitored
- how drift is detected
- how monitoring outputs are produced
- how monitoring relates to CI and evaluation

Out of scope:
- automated retraining
- alerting or paging infrastructure
- production deployment topology

---

## Monitoring goals

Monitoring is designed to:

1. **Detect change**
   - data distributions
   - model prediction behavior

2. **Remain non-disruptive**
   - warnings do not crash pipelines
   - monitoring runs in CI and local environments

3. **Produce interpretable outputs**
   - structured JSON reports
   - explicit statistics and thresholds

4. **Complement evaluation**
   - monitoring does not replace training-time gates
   - signals inform investigation rather than enforce decisions

---

## What is monitored

### Data drift

Data drift measures changes in input characteristics between
reference and current datasets.

Examples:
- text length distribution shifts
- token frequency changes
- label distribution shifts

Implementation:
- computed using lightweight statistical summaries
- does not require model retraining

---

### Prediction drift

Prediction drift measures changes in model output behavior
over time or across datasets.

Examples:
- probability distribution shifts
- increased prediction entropy
- skew toward one class

Prediction drift can indicate:
- data distribution changes
- emerging model brittleness
- silent failures not captured by accuracy metrics

---

## Monitoring execution

Entry points:
- `scripts/run_drift_report.py`
- `scripts/run_prediction_drift.py`

Core logic:
- `src/fakenews/monitoring/drift.py`
- `src/fakenews/monitoring/prediction_drift.py`

Monitoring scripts:
- load reference and comparison datasets
- compute summary statistics
- write reports to disk

---

## Monitoring outputs

Monitoring produces structured JSON artifacts.

Outputs:
- `artifacts/monitoring/drift_report.json`
- `artifacts/monitoring/pred_drift_report.json`

Each report includes:
- computed statistics
- thresholds or reference values
- pass / warn / skipped status
- human-readable warnings

These outputs are designed for both automated inspection
and manual review.

---

## CI integration

Monitoring scripts are executed in CI as part of the pipeline.

CI behavior:
- monitoring runs on CI-trained artifacts
- warnings are allowed
- crashes or malformed outputs fail CI

This ensures monitoring code remains functional and
compatible with evolving artifacts.

CI design is described in [CI/CD](ci-cd.md).

---

## Relationship to evaluation gates

Evaluation gates and monitoring serve different purposes:

- **Evaluation gates**
  - enforce minimum quality at training time
  - block bad models from progressing

- **Monitoring**
  - observes changes after training
  - surfaces risks and investigation signals

Monitoring does not override evaluation gates and does not
make automatic decisions.

---

## Expected failure modes

Monitoring may surface:
- drift warnings due to small sample sizes
- noisy statistics on CI datasets
- ambiguous signals requiring human interpretation

These are expected behaviors and should not cause pipeline failures.

---

## What monitoring does not guarantee

Monitoring does **not** guarantee:
- detection of all failure modes
- semantic correctness of predictions
- fairness or bias mitigation
- immediate detection of adversarial behavior

Monitoring is a diagnostic tool, not a safety proof.

---

## Summary

Monitoring in this project is intentionally:

- lightweight
- non-blocking
- transparent
- complementary to evaluation and testing

It provides early signals of change while preserving
pipeline stability.

---

## Additional documentation

- **[Architecture](architecture.md)** — system design and data flow
- **[Testing Strategy](testing-strategy.md)** — robustness and quality enforcement
- **[Model Card: v1](model-card-v1.md)** — baseline model behavior
- **[Model Card: v2](model-card-v2.md)** — robustness guarantees
- **[CI/CD](ci-cd.md)** — CI execution and enforcement
