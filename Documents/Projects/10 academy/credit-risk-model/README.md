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

## Credit Scoring Business Understanding

Bati Bank's credit risk initiative sits at the intersection of commercial lending goals and regulatory accountability. Before any model is deployed, stakeholders need a shared understanding of *why* the bank builds probability-of-default (PD) models the way it does, *what* the available data can and cannot represent, and *how* modeling choices affect both performance and compliance.

### Context

Bati Bank must estimate the likelihood that a borrower will fail to meet repayment obligations before extending credit. In practice, the bank often works with behavioral and transactional data rather than a clean, long-horizon default label. That gap between business need and available ground truth shapes every downstream modeling decision in this project.

---

### How Basel II influences the need for an interpretable and well-documented model

Under the **Basel II** framework, banks that use internal ratings-based (IRB) approaches must estimate key risk parameters—**Probability of Default (PD)**, Loss Given Default (LGD), and Exposure at Default (EAD)—using rigorous, validated methodologies. Even where Bati Bank is not yet on a full IRB path, Basel II sets the industry standard for how credit risk models are governed.

Basel II creates a strong preference for models that are:

- **Transparent** — Regulators and internal audit must understand how inputs translate into risk estimates. Black-box outputs are difficult to defend during validation.
- **Well documented** — Model development, data lineage, assumptions, limitations, and performance monitoring must be recorded in a way that supports periodic review and stress testing.
- **Statistically sound and stable** — PD estimates feed directly into capital requirements and portfolio decisions; unexplained drift or opaque feature interactions increase regulatory and financial exposure.

For Bati Bank, this means the PD model is not merely a prediction tool—it is part of the bank's **risk management infrastructure**. Interpretability and documentation are therefore design requirements, not optional enhancements.

---

### Why a proxy variable is necessary when there is no direct default label

A **default** is typically defined by a formal event: sustained delinquency (e.g., 90+ days past due), charge-off, bankruptcy, or similar credit loss trigger observed over a defined performance window. In many real-world datasets—including those available for this challenge—the bank does not have a sufficiently long or complete history to label every customer with that outcome.

When a direct default label is unavailable or immature, practitioners use a **proxy variable**: an observable behavior that correlates with future default but can be measured today. Common examples include:

- Severe delinquency or repeated missed payments in the recent past
- A "bad customer" flag derived from internal collections data
- Deterioration in repayment behavior relative to contractual terms

A proxy is necessary because the model still needs a **supervised learning target**. Without one, the bank cannot train a classifier to distinguish higher-risk from lower-risk applicants. The proxy acts as a practical stand-in for true default, allowing Bati Bank to learn patterns from historical behavior even when full default outcomes are incomplete or not yet realized.

The critical caveat: **a proxy is an assumption, not a fact.** Its validity depends on how closely it tracks the business definition of default the bank intends to manage.

---

### Business risks introduced by proxy-based prediction

Relying on a proxy target introduces risks that extend beyond standard model error:

| Risk | Description | Potential impact |
|------|-------------|------------------|
| **Definition mismatch** | The proxy may capture short-term delinquency while the bank cares about long-term default. | Under- or over-estimation of true PD; misaligned lending decisions. |
| **Selection bias** | Customers with observable proxy events may differ systematically from the full applicant population. | Model performs well on historical "bad" cases but poorly on new applicants. |
| **Concept drift** | Economic conditions, product mix, or collections policy change how the proxy relates to actual default. | Silent degradation of model accuracy after deployment. |
| **Fairness and reputational risk** | Proxies tied to past hardship may correlate with protected or vulnerable groups. | Regulatory scrutiny, customer harm, and brand damage. |
| **Overconfidence in automation** | Strong offline metrics on a proxy can create false certainty at the point of credit decision. | Approvals granted to high-risk borrowers; unnecessary declines of creditworthy applicants. |

Mitigation requires explicit documentation of the proxy definition, ongoing monitoring against realized outcomes as they mature, and human review for borderline cases—especially during early deployment.

---

### Trade-offs: Logistic Regression with WoE vs. Gradient Boosting in a regulated context

Two modeling paths are commonly considered for credit scoring in regulated environments:

#### Logistic Regression with Weight of Evidence (WoE)

WoE transforms categorical and binned numeric features into a monotonic, linearly compatible scale. Paired with logistic regression, this approach is a long-standing industry standard.

**Strengths**

- Highly **interpretable** — Each feature's contribution to the score is transparent and easy to explain to credit officers and regulators.
- **Stable and auditable** — Coefficient signs and magnitudes support straightforward validation and policy alignment (e.g., "higher debt burden increases PD").
- **Regulatory familiarity** — Validators and risk teams are accustomed to reviewing scorecards built on this methodology.

**Limitations**

- Assumes **linear relationships** (after WoE transformation); may underfit complex interaction effects.
- Feature engineering (binning, monotonicity constraints) is **labor-intensive** and requires domain expertise.

#### Gradient Boosting (e.g., XGBoost, LightGBM)

Gradient boosting often delivers **superior predictive performance** by capturing non-linear patterns and feature interactions automatically.

**Strengths**

- Higher **discrimination** (e.g., ROC-AUC, Gini) on complex datasets.
- Less manual feature engineering when raw features are informative.

**Limitations**

- **Lower inherent interpretability** — Explaining individual decisions requires post-hoc tools (SHAP, LIME), which add complexity to validation.
- Greater risk of **overfitting** and **unstable feature importance** if not carefully tuned and monitored.
- Harder to align with **policy constraints** (e.g., monotonicity requirements on certain variables).

#### Recommendation for Bati Bank

| Dimension | Logistic Regression + WoE | Gradient Boosting |
|-----------|---------------------------|-------------------|
| Regulatory defensibility | Strong | Moderate (requires extra explainability work) |
| Predictive power | Moderate | Strong |
| Development effort | Higher (manual binning/WoE) | Lower (automated splits) |
| Ongoing monitoring | Straightforward | Requires robust drift and explainability monitoring |
| Best suited when | Interpretability and auditability are paramount | Performance gains justify added governance overhead |

For a regulated financial institution like Bati Bank, a pragmatic approach is to **benchmark both**: use logistic regression with WoE as the interpretable baseline and challenger models (e.g., gradient boosting) to quantify the performance uplift. The final production choice should weigh predictive gain against the bank's appetite for validation effort, explainability requirements, and the reliability of the proxy target.

---

### Success metrics

| Metric | Target / Notes |
|--------|----------------|
| ROC-AUC | Primary discrimination metric; compare baseline vs. challenger models |
| PR-AUC | Important when defaults (or proxy events) are rare |
| Gini coefficient | Common credit risk reporting metric (2 × AUC − 1) |
| Population Stability Index (PSI) | Monitor score distribution drift over time |
| Business KPI | Approval rate, loss rate, or margin at a fixed PD cutoff |

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
