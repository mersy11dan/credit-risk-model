# Bati Bank Credit Risk Probability Model

End-to-end machine learning solution for estimating **probability of default (PD)** on loan applications for Bati Bank. This repository covers data processing, model training with MLflow tracking, batch inference, a REST API, and CI/CD.

## Project Overview

<!-- Brief description of the challenge, objective, and deliverables -->

- **Goal:** Build a reproducible PD model to support credit decisioning at Bati Bank.
- **Output:** Calibrated probability scores (0–1) and risk bands (Low / Medium / High).
- **Stack:** Python, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions.

## Setup

### Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose

### Local installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Project structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # Lint + test on push to main
├── data/
│   ├── raw/                   # Raw datasets (not committed)
│   └── processed/             # Cleaned feature tables
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── config.py              # Paths, column names, defaults
│   ├── data_processing.py     # Load, validate, clean
│   ├── features.py            # Preprocessing pipeline
│   ├── train.py               # Training + MLflow logging
│   ├── predict.py             # Batch inference
│   └── api/                   # FastAPI scoring service
├── tests/
├── models/                    # Serialized pipelines (gitignored)
├── mlruns/                    # MLflow artifacts (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Business Understanding

<!-- Document the credit risk problem, stakeholders, and success criteria -->

### Context

- Bati Bank needs to assess whether a loan applicant is likely to default before approval.
- The model should align with regulatory expectations for transparent, auditable credit scoring.

### Key questions

- What defines a **default** in the provided dataset?
- Which features are legally and ethically acceptable for scoring?
- What PD threshold maps to approve / review / reject decisions?

### Success metrics

| Metric | Target / Notes |
|--------|----------------|
| ROC-AUC | <!-- fill after baseline --> |
| PR-AUC  | <!-- important for imbalanced data --> |
| Business KPI | <!-- e.g. approval rate at fixed default rate --> |

## Pipeline

<!-- High-level flow from raw data to deployed model -->

1. **Ingest** — Place raw CSV in `data/raw/`.
2. **EDA** — Explore distributions, missingness, and target balance in `notebooks/eda.ipynb`.
3. **Configure** — Set `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`, and `TARGET_COLUMN` in `src/config.py`.
4. **Preprocess** — `src/data_processing.py` validates and cleans data; output saved to `data/processed/`.
5. **Train** — `src/train.py` fits a sklearn Pipeline and logs metrics to MLflow.
6. **Serve** — Deploy via FastAPI (`src/api/`) or batch scoring (`src/predict.py`).

## Training

```bash
# Train with MLflow tracking
python -m src.train --data data/raw/credit_data.csv

# View experiments
mlflow ui --backend-store-uri ./mlruns
```

Artifacts:

- `models/model.pkl` — Serialized sklearn Pipeline
- `mlruns/` — Experiment runs, parameters, and metrics

## API

### Run locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health and model load status |
| POST | `/predict` | Score a single credit application |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 20000,
    "loan_term_months": 36,
    "credit_score": 680,
    "employment_years": 5
  }'
```

### Docker

```bash
docker compose up --build
```

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

CI runs linting and pytest on every push to `main` (see `.github/workflows/ci.yml`).

## Report

<!-- Link or summarize your final write-up: methodology, EDA findings, model comparison, SHAP/explainability, limitations -->

- **Interim findings:** <!-- --> 
- **Final model choice:** <!-- -->
- **Deployment notes:** <!-- -->
- **Full report:** <!-- link to PDF / Notion / Google Doc -->

## License

<!-- Add license if required by the challenge -->
