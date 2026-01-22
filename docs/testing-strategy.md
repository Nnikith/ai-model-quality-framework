# AI Model Quality Framework – Testing Strategy

## 1. Philosophy: Model as a System Under Test

We treat the ML model + data + preprocessing + inference API as a production system.
Every assumption becomes a **testable requirement** and a **CI/CD quality gate**.

Testing goals:
- Catch data issues before training
- Prevent model regressions
- Measure robustness to input noise
- Detect bias/sensitivity risks
- Detect drift and trigger re-evaluation
- Ensure inference contracts remain stable

---

## 2. Test Levels

### 2.1 Unit Tests (Fast, deterministic)
Scope: pure functions and small modules
Examples:
- text cleaning keeps non-empty content
- vectorizer/tokenizer functions return expected shapes/types
- config parsing and validation

Pass criteria:
- 100% pass
- run in < 30 seconds

### 2.2 Integration Tests (Pipeline behavior)
Scope: ingestion → preprocessing → training → evaluation on a small sample
Examples:
- pipeline can train v1 on a tiny dataset subset
- artifacts are created in the expected locations
- evaluation report schema is valid

Pass criteria:
- 100% pass
- run in < 2–3 minutes (CPU)

### 2.3 End-to-End Tests (System contract)
Scope: API + model loading + prediction behavior
Examples:
- FastAPI /predict responds with valid schema
- invalid input rejected with clear error
- response includes model/version metadata

Pass criteria:
- 100% pass
- run in < 2 minutes (CPU)

---

## 3. AI Testing Categories

## 3.1 Data Quality Tests (Block training if failed)

### Schema & Types
- Required columns exist (e.g., `text`, `label`)
- Labels within allowed set
- Types correct (text is string-like, label is categorical/int)

### Missingness & Validity
- % empty or null text below threshold
- Minimum and maximum text length thresholds

### Duplicates & Leakage
- No exact duplicates above threshold
- No duplicates across train/test split (leakage guard)
- Near-duplicate detection (optional enhancement)

### Distribution & Balance
- Label distribution within acceptable bounds
- Train/validation/test label distributions comparable

### Ingestion Artifacts

Ingestion must produce the following artifacts:

- processed dataset (parquet or csv)
- split manifest with row counts per split
- data validation report containing:
  - total rows ingested
  - rows rejected (with reasons)
  - label distribution per split
  - duplicate counts

**Gate outcome:** FAIL blocks training and CI merge.

---

## 3.2 Model Quality Tests (Promotion gates)

### Baseline Metrics (v1)
- Minimum thresholds (example targets; adjustable):
  - F1 >= 0.80 (or dataset-appropriate)
  - PR-AUC >= 0.80
- Training must be reproducible (seeded)
- Calibration check: expected confidence behavior

### Improved Model Metrics (v2)
- Must meet an absolute threshold OR improve upon v1 by X%
- Must not degrade significantly on any critical slice

### Slice Metrics
Evaluate by:
- text length buckets (short/medium/long)
- topic/source buckets (if available)
- uncertainty buckets (low/high confidence)

**Gate outcome:** FAIL blocks promotion of new model artifacts.

---

## 3.3 Regression Tests (Prevent “quiet” breakages)

### Golden Set Predictions
- Fixed set of representative inputs stored in repo
- Expected output format and ranges must remain stable
- Prediction drift beyond tolerance triggers investigation

### Artifact Compatibility
- New code can load old model artifacts (backward compatibility)
- API response schema remains stable

**Gate outcome:** FAIL blocks merge.

---

## 3.4 Robustness Tests (Noise + perturbations)

### Invariance / Stability Checks
- Case changes should not flip predictions unexpectedly
- Minor punctuation/whitespace changes should not flip labels frequently
- Small typo injection should degrade gracefully

### Out-of-Distribution (OOD) Behavior
- Empty input → should return safe error
- Gibberish input → low confidence / predictable handling
- Very short input → handled consistently

**Gate outcome:** FAIL blocks merge if robustness breaks beyond thresholds.

---

## 3.5 Bias / Sensitivity Risk Checks

This project cannot “solve fairness” for fake news detection, but it can detect risk.

### Named Entity Sensitivity
- Swap person/organization names and measure prediction volatility
- Excessive sensitivity triggers a warning or gate failure (configurable)

### Topic Sensitivity
- Compare performance across topics (politics, health, finance) if feasible
- Flag underperforming topics

### Documentation Requirement
- Model cards must document limitations and known risks

**Gate outcome:** WARN by default (can be configured to FAIL).

---

## 3.6 Drift Monitoring Tests (Post-deploy signals)

### Data Drift
- Compare new input distribution vs training baseline
- Metrics may include:
  - PSI on TF-IDF feature summaries (v1)
  - embedding distribution shift (v2)
  - text length and language distribution shift

### Prediction Drift
- Monitor confidence distribution shift
- Monitor predicted label distribution shift

### Alerting / Re-evaluation
- If drift exceeds thresholds:
  - create an alert report
  - trigger scheduled re-evaluation workflow

**Gate outcome:** does not fail CI by default, but triggers re-eval workflow.

---

## 4. CI/CD Quality Gates (What blocks a PR)

### Required on Every PR (CPU-only)
- Lint: ruff
- Unit tests: pytest unit
- Integration tests: small sample pipeline
- Data tests: schema + leakage + basic distribution tests on sample
- Regression tests: golden set + API schema

### Optional / Scheduled (Nightly or manual)
- Full training run on larger dataset subset
- Full evaluation + error analysis report
- Drift simulation run

---

## 5. Reporting Artifacts

All quality runs generate machine-readable artifacts:
- `artifacts/reports/data_validation.json`
- `artifacts/reports/eval_metrics.json`
- `artifacts/reports/test_summary.json`
- optional: HTML summary for humans

These are uploaded by CI for traceability.

---

## 6. Configuration

All thresholds and gates are configurable via `configs/`:
- `configs/data.yaml`
- `configs/eval.yaml`
- `configs/monitoring.yaml`

---

## 7. Definition of Done (v1 milestone)

v1 is complete when:
- Data ingestion + validation pipeline exists
- Baseline model trains reproducibly
- Evaluation report is generated
- CI runs unit + integration + regression tests
- Inference API serves predictions locally
- Drift baseline stats are saved

v2 extends v1 by adding transformer model + embedding drift + enhanced robustness testing.
