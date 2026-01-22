# AI Model Quality Framework – Architecture

## 1. Purpose and Scope

This project implements a production-style AI system where the **ML model is treated as a system under test**.

The reference use case is **Fake News Detection**, but the framework is designed to be reusable for other NLP classification tasks.

Primary goals:
- Build an end-to-end NLP ML pipeline
- Layer automated AI testing and validation on top
- Add drift monitoring and CI/CD quality gates
- Demonstrate AI Tester, ML Engineer, and MLOps competencies in a single system

---

## 2. High-Level System Overview

The system consists of the following layers:

1. Data ingestion and validation
2. Feature engineering and preprocessing
3. Model training (baseline + improved)
4. Evaluation and error analysis
5. Automated AI testing and quality gates
6. Model artifact management
7. Inference service
8. Monitoring and drift detection
9. CI/CD orchestration

Each layer produces explicit artifacts that are versioned and testable.

---

## 3. Data Layer

### Responsibilities
- Ingest raw datasets from external sources
- Validate schema, labels, and distributions
- Prevent data leakage across splits
- Version datasets and splits for reproducibility

# Ingestion guarantees:
- labels are normalized to binary {0,1}
- text column is finalized (optionally title + body)
- split assignment is explicit and persisted
- invalid rows are rejected with reason codes


### Artifacts
- Raw dataset snapshots
- Processed datasets
- Train/validation/test split manifests
- Data validation reports

### Canonical Dataset Schema (Internal)

All ingested datasets are normalized into a single internal schema before
any downstream processing.

Canonical columns:
- id: string (unique, deterministic)
- text: string (model input)
- label: int (0 = real, 1 = fake)
- source: string (dataset identifier)
- subject: string | null
- date: datetime | null
- split: one of {train, val, test}

No downstream component is allowed to depend on raw dataset-specific schemas.

---

## 4. Feature and Preprocessing Layer

### Responsibilities
- Text normalization and cleaning
- Language checks and filtering
- Feature extraction (TF-IDF for v1, tokenization for v2)
- Persist preprocessing artifacts

### Artifacts
- Fitted vectorizers/tokenizers
- Preprocessing configuration
- Feature statistics summaries

---

## 5. Modeling Layer

### Model v1 (Baseline)
- TF-IDF + Logistic Regression
- Focus on interpretability and stability

### Model v2 (Improved)
- Transformer-based classifier
- Focus on performance and robustness

### Responsibilities
- Reproducible training
- Deterministic artifact generation
- Model versioning

### Artifacts
- Serialized model files
- Training metadata
- Model signatures

---

## 6. Evaluation and Error Analysis

### Responsibilities
- Compute standard metrics (F1, ROC-AUC, PR-AUC)
- Perform slice-based evaluation
- Analyze failure modes

### Artifacts
- Evaluation reports
- Confusion matrices
- Error analysis summaries

---

## 7. AI Testing and Quality Gates

### Responsibilities
- Enforce data quality tests
- Validate model performance thresholds
- Run robustness and bias checks
- Block model promotion if gates fail

### Artifacts
- Test reports
- Pass/fail gate summaries

---

## 8. Inference Service

### Responsibilities
- Serve predictions via API
- Enforce input validation
- Log inference metadata safely

### Artifacts
- API schemas
- Deployed model version info

---

## 9. Monitoring and Drift Detection

### Responsibilities
- Detect data and feature drift
- Monitor prediction confidence shifts
- Trigger alerts and re-evaluation workflows

### Artifacts
- Drift reports
- Monitoring dashboards (offline)

---

## 10. CI/CD Integration

### Responsibilities
- Run automated tests on every PR
- Enforce quality gates before merge
- Track model regressions over time

---

## 11. Non-Goals

- Real-time streaming ingestion
- Online learning
- Full production deployment

These are intentionally excluded to keep scope focused and evaluable.
