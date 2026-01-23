# CI / CD

This document describes how continuous integration is used to **enforce model quality**
in **ai-model-quality-framework**.

CI is treated as a **quality gate**, not just a test runner. A change is considered valid
only if it preserves data validity, model quality, robustness guarantees, and system contracts.

For overall system context, see [Architecture](architecture.md).

---

## Scope

This document covers:
- what CI executes
- how CI differs from full local evaluation
- how CI enforces quality and robustness
- expected failure modes

Out of scope:
- cloud-specific deployment pipelines
- release orchestration or canarying
- production infrastructure details

---

## CI goals

CI is designed to:

1. **Fail fast on broken assumptions**
   - invalid data
   - missing artifacts
   - broken contracts

2. **Prevent silent regressions**
   - model quality regressions
   - robustness regressions
   - API behavior changes

3. **Exercise the full system**
   - ingestion → training → evaluation → tests
   - same interfaces as local runs

4. **Remain deterministic and stable**
   - small, tracked datasets
   - relaxed gates where appropriate

---

## CI execution path

On every push or pull request, CI executes the following steps:

1. **Environment setup**
   - install dependencies
   - install parquet engine for dataset loading

2. **Data ingestion**
   - run ingestion using `configs/data_ci.yaml`
   - use repository-tracked sample data
   - write processed dataset and validation reports

3. **Model training**
   - train Model v1 and Model v2
   - use `configs/eval_ci.yaml` to avoid tiny-sample brittleness
   - write versioned artifacts and evaluation reports

4. **Evaluation gates**
   - enforce minimum quality and non-regression
   - fail CI if gates are violated

5. **Test execution**
   - unit tests
   - integration tests
   - end-to-end API and robustness tests

6. **Monitoring scripts**
   - run drift and prediction drift reports
   - warnings are allowed; crashes are not

This mirrors a reduced but complete production-style pipeline.

---

## Configuration strategy

CI uses paired configuration files to balance signal and stability.

### Data configuration

- `configs/data.yaml` — full dataset ingestion
- `configs/data_ci.yaml` — CI-safe sample ingestion

Both share:
- schema expectations
- split strategy
- validation rules

CI differs only in data source and size.

---

### Evaluation configuration

- `configs/eval.yaml` — full evaluation with strict gates
- `configs/eval_ci.yaml` — relaxed gates for CI

CI gates are intentionally permissive to:
- avoid flakiness on small samples
- validate pipeline correctness rather than absolute performance

Full evaluation is expected to run locally or in scheduled jobs.

---

## Robustness enforcement

CI enforces robustness guarantees explicitly.

Examples:
- v2 must satisfy typo-invariance tests
- v1 is allowed to fail these tests by design

This ensures that:
- improvements remain enforced over time
- regressions are caught immediately

Details of robustness testing are documented in
[Testing Strategy](testing-strategy.md).

---

## Artifact contracts in CI

CI relies on stable artifact paths:

- `artifacts/models/v1/`
- `artifacts/models/v2/`
- `artifacts/reports/`
- `artifacts/monitoring/`

Tests and monitoring scripts depend only on these contracts,
not on training internals.

Breaking an artifact contract is treated as a CI failure.

---

## Failure modes

CI failures are expected and informative.

Common causes include:
- data validation errors
- evaluation gate violations
- robustness regressions
- missing or malformed artifacts
- API contract changes

Each failure mode maps to a specific report or test output,
making debugging straightforward.

---

## What CI does not guarantee

CI does **not** guarantee:
- optimal model performance
- fairness or bias mitigation
- semantic understanding
- real-world distribution alignment

These concerns require production feedback and are therefore
documented and monitored rather than enforced in CI.

---

## Relationship to monitoring

CI validates **pre-deployment** quality.
Monitoring validates **post-training and post-deployment** behavior.

Monitoring design and outputs are documented in
[Monitoring](monitoring.md).

---

## Additional documentation

- **[Architecture](architecture.md)** — system design and execution flow
- **[Testing Strategy](testing-strategy.md)** — test layers and guarantees
- **[Model Card: v1](model-card-v1.md)** — baseline model behavior
- **[Model Card: v2](model-card-v2.md)** — robustness guarantees
- **[Monitoring](monitoring.md)** — drift detection and observability
