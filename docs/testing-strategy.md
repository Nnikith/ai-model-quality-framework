# Testing Strategy

This document describes how **ai-model-quality-framework** approaches testing and quality assurance
for machine learning systems.

The goal is not to maximize test coverage, but to **make modeling assumptions explicit and testable**
across the full lifecycle: data ingestion, training, evaluation, serving, and monitoring.

For system context and component boundaries, see [Architecture](architecture.md).

---

## Scope

This document covers:

- how data quality is validated
- how model quality is gated
- how robustness is enforced
- how regressions are detected
- how CI exercises the system end-to-end

Out of scope:

- model-specific design decisions (see model cards: [v1](model-card-v1.md), [v2](model-card-v2.md))
- system architecture details (see [Architecture](architecture.md))
- monitoring design and report interpretation (see [Monitoring](monitoring.md))

---

## Testing philosophy

Traditional ML projects often rely on offline metrics and ad-hoc evaluation.
This project treats the model as a **system under test**.

Key principles:

1. Assumptions must be explicit
2. Assumptions must be testable
3. Tests must fail deterministically
4. CI must enforce guarantees continuously

Testing is therefore layered rather than monolithic.

---

## Test layers

### 1) Data validation tests

Data validation occurs during ingestion, before any training begins.

Responsibilities:
- enforce required columns and schema
- validate text length and null constraints
- detect invalid or malformed records early

Mechanism:
- ingestion produces a validation report
- validation failures fail the pipeline immediately

Artifacts:
- `artifacts/reports/data_validation.json`

Rationale:
Failing early prevents downstream metrics from becoming misleading.

Related implementation:
- ingestion and validation logic in the data layer (see [Architecture](architecture.md))

---

### 2) Unit tests

Unit tests focus on **pure logic** and small, deterministic components.

Examples:
- drift statistic calculations
- helper utilities
- small transformation functions

Characteristics:
- no model training
- no external state
- fast execution

Unit tests ensure core logic behaves correctly in isolation.

---

### 3) Integration tests

Integration tests validate **contracts between components**.

Examples:
- training produces expected artifact files
- evaluation reports contain required fields
- monitoring scripts run and write reports without crashing

Characteristics:
- limited I/O
- controlled datasets
- stable execution in CI

These tests ensure that components fit together as expected (see [Architecture](architecture.md)).

---

### 4) End-to-end (E2E) tests

E2E tests exercise the system as a user would interact with it.

Primary focus:
- inference API behavior
- robustness under realistic input perturbations
- regression protection across model versions

Examples:
- API returns valid probabilities
- predictions remain stable under minor text perturbations
- known limitations are documented and bounded

E2E tests operate against trained artifacts rather than mocked components
(see [CI/CD](ci-cd.md) for how artifacts are produced in CI).

---

## Robustness testing

Robustness is treated as a **first-class quality dimension**, not an afterthought.

Approach:
- identify realistic failure modes (typos, casing, punctuation, spacing)
- encode expectations as tests
- enforce guarantees in CI

Versioned behavior:
- v1 documents lack of robustness guarantees (see [Model Card: v1](model-card-v1.md))
- v2 enforces strict typo invariance within a defined tolerance (see [Model Card: v2](model-card-v2.md))

This allows known weaknesses to be documented while preventing regressions
in improved models.

---

## Evaluation gates

Evaluation gates encode **minimum acceptable quality** as code.

Gate types:
- absolute thresholds (minimum F1, PR-AUC)
- non-regression vs previous versions
- minimum improvement when applicable

Behavior:
- training exits with a non-zero code if gates fail
- CI uses relaxed gates to avoid tiny-sample flakiness
- full evaluation uses strict gates for meaningful quality control

Gates are configured in:
- `configs/eval.yaml` (full evaluation)
- `configs/eval_ci.yaml` (CI-safe evaluation)

Implementation lives in the evaluation module (see [Architecture](architecture.md)).

---

## CI as a test executor

CI is responsible for enforcing the testing strategy continuously.

CI executes:
1. data ingestion on a tracked sample
2. model training (v1 and v2)
3. evaluation gate checks
4. unit, integration, and E2E tests
5. robustness enforcement

CI behavior and rationale are documented in [CI/CD](ci-cd.md).

---

## What is intentionally not tested

Some aspects are explicitly out of scope:

- absolute model correctness
- semantic understanding of text
- real-world fairness guarantees

These require domain context and production feedback and are therefore
documented rather than enforced.

---

## Summary

The testing strategy prioritizes:

- explicit assumptions
- deterministic failures
- layered validation
- continuous enforcement via CI

This approach mirrors how production ML systems are validated,
where correctness is defined by contracts and guarantees rather than metrics alone.

---

## Additional documentation

- **[Architecture](architecture.md)** — system design and artifact contracts
- **[Model Card: v1](model-card-v1.md)** — baseline behavior and known limitations
- **[Model Card: v2](model-card-v2.md)** — robustness upgrade and enforced guarantees
- **[CI/CD](ci-cd.md)** — CI execution path and quality enforcement
- **[Monitoring](monitoring.md)** — drift detection and monitoring outputs
