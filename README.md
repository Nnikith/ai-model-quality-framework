# AI Model Quality Framework

A production-style, end-to-end machine learning system designed to demonstrate
**model development, evaluation, robustness testing, and CI-enforced quality gates**.

This repository is intentionally structured to support:
- AI / ML Engineer roles
- AI Quality Engineer / AI Tester roles
- MLOps-focused interviews

The model is treated as a **system under test**, not just a notebook experiment.

---

## Project Goals

This project demonstrates:

- Data ingestion and validation
- Baseline and improved model development
- Automated evaluation and quality gates
- Robustness testing as a first-class concern
- Drift detection and monitoring
- CI pipelines that enforce model quality
- Clear, production-style repo structure and documentation

---

## Models

### Model v1 (Baseline)
- Word-level TF-IDF features
- Logistic Regression classifier
- Strong performance on clean data
- **Known limitation:** sensitivity to typos and spelling variations

This limitation is intentionally documented and tested.

### Model v2 (Robustness Upgrade)
Model v2 is a deliberate robustness-focused improvement over v1.

**What changed:**
- Character n-gram TF-IDF (`char_wb`, n=3–5)
- Same linear classifier and API contract

**Why it matters:**
Character n-grams preserve subword structure, making the model significantly more tolerant
to minor typos, spacing issues, and spelling noise.

**Robustness as a contract:**
- v1 has *no guaranteed typo invariance*
- v2 introduces a **strict robustness test** that must pass
- CI fails if this guarantee is violated

---

## Evaluation & Quality Gates

Models are gated on:
- Minimum absolute quality (F1, PR-AUC)
- Non-regression vs previous versions
- Required improvement when the baseline is not near ceiling

When baseline performance is already near theoretical limits,
improvement thresholds are intentionally reduced to avoid forcing artificial gains
while still preventing regressions.

---

## Continuous Integration

The CI pipeline enforces:
1. Data ingestion on a tracked CI sample
2. Model training (v1 and v2)
3. Evaluation against configured gates
4. Robustness and regression tests

A commit cannot pass CI if model quality or robustness regresses.

---

## Repository Structure

High-level structure:

configs/ # Data, model, and evaluation configs
src/fakenews/ # Ingestion, modeling, evaluation, monitoring, serving
scripts/ # CLI entrypoints for training, ingestion, monitoring
tests/ # Unit, integration, and end-to-end tests
artifacts/ # Models, reports, and monitoring outputs


Detailed design notes are available in:
- `docs/architecture.md`
- `docs/testing-strategy.md`
- `docs/model-card-v2.md`

---

## Why this repo exists

This project is intentionally opinionated:
- Every modeling assumption is made testable
- Robustness is enforced, not assumed
- CI is used as a quality gate, not just a syntax checker

The goal is to demonstrate **how ML systems should be built and validated in production**.
