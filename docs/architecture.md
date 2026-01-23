# Architecture

This document describes the end-to-end architecture of **ai-model-quality-framework**.

The system is intentionally designed as a **production-style ML system under test**.
Modeling, testing, evaluation, and CI are treated as first-class architectural components,
not as supporting scripts.

---

## Scope

This document covers:

- major system components and responsibilities
- data and model lifecycle (ingest → train → evaluate → serve → monitor)
- artifact contracts between stages
- the role of CI in enforcing quality

Out of scope:

- model-specific assumptions and limitations
  (see model cards: [v1](model-card-v1.md), [v2](model-card-v2.md))
- detailed testing logic
  (see [testing-strategy.md](testing-strategy.md))
- deployment to a specific cloud or platform

---

## Design goals

1. **Reproducibility**
   - config-driven behavior
   - deterministic CI splits
   - versioned artifacts

2. **Testability**
   - assumptions expressed as tests
   - failures surface early and clearly

3. **Environment parity**
   - CI exercises the same pipeline as full runs
   - relaxed CI configs avoid flakiness without changing interfaces

4. **Separation of concerns**
   - ingestion, modeling, serving, and monitoring are decoupled
   - each stage communicates via explicit artifacts

---

## High-level component map

The repository is organized around **clear system responsibilities** rather than implementation convenience.
Each top-level area corresponds to a stage in the model lifecycle or a cross-cutting concern.

### Configuration

```
configs/
```

Centralized, declarative configuration for the system.

- data ingestion and splitting behavior
- model architecture and training parameters
- evaluation gates and CI-specific relaxations

Paired configs (e.g., full vs CI) allow the same pipeline to run in different environments
without changing code.

---

### Data layer

```
data/
src/fakenews/data/
```

Responsible for data ingestion, validation, and canonicalization.

- raw CI sample data lives under `data/`
- ingestion and validation logic lives under `src/fakenews/data/`
- outputs a single canonical, validated dataset consumed by training

This layer is designed to **fail early** when schema or quality assumptions are violated.

---

### Modeling and evaluation

```
src/fakenews/models/
src/fakenews/evaluation/
```

Responsible for training and quality enforcement.

- model implementations are versioned (v1, v2)
- training produces explicit, versioned artifacts
- evaluation gates encode minimum quality and non-regression expectations

Training and evaluation are intentionally decoupled so that
quality checks can evolve independently of model code.

---

### Serving

```
src/fakenews/serving/
```

Responsible for inference only.

- loads pre-trained artifacts
- exposes a stable API contract
- does not retrain or mutate model state

Serving depends only on artifact contracts, not on training logic.

---

### Monitoring

```
src/fakenews/monitoring/
```

Responsible for post-training observability.

- feature distribution drift
- prediction distribution drift
- structured JSON outputs for CI or review

Monitoring is designed to surface warnings without breaking pipelines.

---

### Execution entrypoints

```
scripts/
```

Thin CLI wrappers that connect configuration to core logic.

- ingestion
- training
- serving
- monitoring

Scripts are intentionally lightweight; all behavior lives in `src/`.

---

### Testing

```
tests/
src/fakenews/testing/
```

Responsible for validating system behavior.

- unit tests for core logic
- integration tests for artifact contracts
- end-to-end tests for API robustness and regression protection

Tests treat the model as a **system under test**, not a black-box function.

---

### Artifacts

```
artifacts/
```

Generated outputs produced by the system.

- trained models and vectorizers
- evaluation reports
- drift and monitoring reports

Artifacts form the explicit contracts between stages and are consumed by
serving, monitoring, and CI.

---

### Why this structure matters

This layout enforces:

- clear ownership of responsibilities
- explicit contracts between stages
- minimal coupling between ingestion, training, serving, and monitoring
- CI-friendly execution paths

As a result, the system can be tested, evolved, and extended
without collapsing into a monolithic pipeline.

---

## Data flow

### 1) Ingestion and validation

Entry point:
- `scripts/ingest_isot.py`

Core logic:
- `src/fakenews/data/ingest_isot.py`
- `src/fakenews/data/validate.py`

Responsibilities:
- load raw data sources
- canonicalize schema
- validate structural and content constraints
- assign splits (`train`, `val`, `test`)
- write processed dataset and validation reports

Outputs (artifact contract):
- `data/processed/isot.parquet`
- `artifacts/reports/split_manifest.json`
- `artifacts/reports/data_validation.json`

Configuration:
- `configs/data.yaml` — full dataset behavior
- `configs/data_ci.yaml` — CI-safe sample ingestion

Notes:
- CI ingestion uses repository-tracked sample CSVs to avoid external dependencies.
- Validation failures are treated as hard failures.

Testing considerations are described in
[Testing Strategy – Data Validation](testing-strategy.md).

---

### 2) Model training (versioned)

Entry points:
- `scripts/train_v1.py`
- `scripts/train_v2.py`

Core logic:
- `src/fakenews/models/train_v1.py`
- `src/fakenews/models/train_v2.py`

Responsibilities:
- load processed dataset
- vectorize text
- train classifier
- evaluate on holdout splits
- write versioned artifacts

Artifact contracts:

**v1**
- `artifacts/models/v1/model.joblib`
- `artifacts/models/v1/vectorizer.joblib`
- `artifacts/reports/eval_metrics_v1.json`

**v2**
- `artifacts/models/v2/model.joblib`
- `artifacts/models/v2/vectorizer.joblib`
- `artifacts/reports/eval_metrics_v2.json`

Configuration:
- `configs/model_v1.yaml`
- `configs/model_v2.yaml`

Model-specific design choices and limitations are documented in:
- [Model Card – v1](model-card-v1.md)
- [Model Card – v2](model-card-v2.md)

---

### 3) Evaluation gates

Core logic:
- `src/fakenews/evaluation/gates.py`

Responsibilities:
- encode quality expectations as deterministic checks
- fail training when expectations are not met

Gate types:
- absolute thresholds (F1, PR-AUC)
- non-regression vs previous model versions
- minimum improvement when applicable

Configuration:
- `configs/eval.yaml` — full evaluation
- `configs/eval_ci.yaml` — CI-safe relaxed gates

Rationale:
- CI should validate pipeline correctness and regression safety
- full evaluation should enforce meaningful quality standards

Details of gate rationale and failure modes are covered in
[Testing Strategy – Evaluation Gates](testing-strategy.md).

---

### 4) Serving (stable API)

Entry point:
- `scripts/run_api.py`

Core logic:
- `src/fakenews/serving/api.py`

Responsibilities:
- load model artifacts from a versioned directory
- expose a stable inference API

Endpoints:
- `GET /health`
- `POST /predict`

Contract guarantees:
- serving does not retrain or modify artifacts
- model version is inferred from artifact directory
- API remains stable across model versions

Serving-related tests are described in
[Testing Strategy – API & E2E Tests](testing-strategy.md).

---

### 5) Monitoring

Entry points:
- `scripts/run_drift_report.py`
- `scripts/run_prediction_drift.py`

Core logic:
- `src/fakenews/monitoring/drift.py`
- `src/fakenews/monitoring/prediction_drift.py`

Responsibilities:
- compute feature-level drift
- compute prediction distribution drift
- write structured monitoring reports

Outputs:
- `artifacts/monitoring/drift_report.json`
- `artifacts/monitoring/pred_drift_report.json`

Monitoring is designed to:
- surface warnings without crashing pipelines
- run in CI and local environments

Monitoring design and interpretation are described in
[Monitoring](monitoring.md).

---

## CI as an architectural component

CI is treated as part of the system architecture, not just a test runner.

The CI pipeline executes a reduced but complete path:

1. ingest CI sample data
2. train v1 and v2
3. enforce evaluation gates
4. run unit, integration, and E2E tests
5. enforce robustness guarantees

CI behavior and configuration are documented in
[CI/CD](ci-cd.md).

---

## Failure modes and expected behavior

- **Ingestion validation failure**
  Pipeline fails early with a validation report.

- **Tiny CI splits**
  Training code includes safe fallbacks; CI gates are relaxed.

- **Missing artifacts**
  Serving health endpoint reflects load state; tests skip appropriately.

- **Drift warnings**
  Monitoring emits warnings but does not crash pipelines.

---

## Versioning policy

- v1 is a stable baseline with documented limitations.
- v2 is a robustness-oriented upgrade.
- both versions share:
  - serving contract
  - monitoring interfaces
  - artifact conventions

Model-specific details are captured in the model cards rather than duplicated here.

---

## Additional documentation

- **[Testing Strategy](testing-strategy.md)** — how model quality and robustness are enforced
- **[Model Card: v1](model-card-v1.md)** — baseline design and known limitations
- **[Model Card: v2](model-card-v2.md)** — robustness upgrade and guarantees
- **[CI/CD](ci-cd.md)** — how CI enforces model quality
- **[Monitoring](monitoring.md)** — drift detection and monitoring outputs
