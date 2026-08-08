# Bati Bank Credit Risk Probability Model

End-to-end machine learning solution for estimating **proxy credit risk** from alternative transaction data (Xente) for Bati Bank.

**Stack:** Python · scikit-learn · MLflow · FastAPI · Docker · GitHub Actions

**Important:** The target `is_high_risk` is an **RFM-based proxy**, not observed default. Use scores for ranking and review until validated against true default outcomes.

---

## Project overview

| Goal | Deliverable |
|------|-------------|
| Credit decisioning with thin-file / alternative data | Customer-level high-risk probability |
| Regulatory-aware modeling | Documented Basel II context, WoE/IV, interpretable baseline |
| Production readiness | FastAPI, Docker, CI (Ruff + pytest), MLflow tracking |

---

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place the Xente file at `data/raw/data.csv`, then:

```bash
# Build customer dataset + RFM proxy target
python -c "from src.data_processing import load_transactions, build_modeling_dataset_with_proxy_target; build_modeling_dataset_with_proxy_target(load_transactions())"

# Train & compare models (logs to MLflow)
python -m src.train

# API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Tests & lint
pytest tests/ -v
ruff check src/ tests/
```

---

## Project structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # Ruff + pytest on push to main
├── data/
│   ├── raw/                      # Xente CSV (not committed)
│   └── processed/                # modeling_dataset.csv (not committed)
├── notebooks/eda.ipynb           # Exploration
├── reports/                      # Interim + final reports & figures
├── scripts/                      # Figure generators for reports
├── src/
│   ├── config.py
│   ├── data_processing.py        # Features + RFM proxy target
│   ├── eda.py
│   ├── woe_iv.py
│   ├── train.py
│   ├── predict.py
│   ├── results.py                # Model comparison exports
│   └── api/                      # FastAPI scoring service
├── tests/
├── models/                       # model.pkl (gitignored); comparison CSVs kept
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Credit scoring business understanding

Bati Bank must estimate whether a customer is likely to become a credit risk when **direct default labels are unavailable**. Basel II–style governance favors models that are **transparent, documented, and validated**.

### Why Basel II matters for model choice

| Expectation | Project response |
|-------------|------------------|
| Interpretability | WoE/IV helpers + Logistic Regression baseline |
| Documentation | README, proxy-target section, final report |
| Challenge models | LR, Decision Tree, Random Forest, Gradient Boosting |
| Stability | Fixed `random_state`, RFM snapshot date, CI gates |

**Pragmatic recommendation:** use **Gradient Boosting** for ranking performance and keep **Logistic Regression (+ WoE analysis)** as the regulatory reference until SHAP/explainability is productionized.

### Proxy target (`is_high_risk`)

1. Compute **RFM** (recency, frequency, monetary) per `CustomerId`
2. Standardize → **K-Means (k=3, random_state=42)**
3. Label the **least engaged** cluster as high risk
4. Merge into `data/processed/modeling_dataset.csv`

This is a **modeling assumption**, not ground truth.

---

## Training & results

```bash
python -m src.train
mlflow ui --backend-store-uri ./mlruns
```

Artifacts:

- `models/model.pkl` — best pipeline (local; not committed)
- `models/model_comparison.csv` — ranked metrics (committed)
- `models/best_model_summary.json` — winner summary (committed)
- `mlruns/` — experiment runs (local; not committed)

Latest comparison snapshot (re-run training to refresh):

| Rank | Model | ROC-AUC (approx.) |
|------|-------|-------------------|
| 1 | Gradient Boosting | ~0.9999 |
| 2 | Logistic Regression | ~0.9999 |
| 3 | Random Forest | ~0.9997 |
| 4 | Decision Tree | ~0.9963 |

Treat these metrics as **proxy separation**, not proven default prediction.

---

## API

```bash
uvicorn src.api.main:app --reload --port 8000
# or
docker compose up --build
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + model load source |
| POST | `/predict` | Customer features → `risk_probability` + `risk_category` |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"total_transaction_amount\": 1000,
    \"avg_transaction_amount\": 200,
    \"transaction_count\": 5,
    \"std_transaction_amount\": 25,
    \"txn_hour\": 10,
    \"txn_day\": 12,
    \"txn_month\": 6,
    \"txn_year\": 2019,
    \"mode_ProductCategory\": \"airtime\"
  }"
```

---

## Portfolio website

A portfolio case-study site lives in [`docs/`](docs/) (GitHub Pages ready).

- Local: open [`docs/index.html`](docs/index.html) or `python -m http.server 5500 --directory docs`
- Live (after Pages setup): `https://mersy11dan.github.io/credit-risk-model/`

## Reports

| Report | Path |
|--------|------|
| **Final report** | [`reports/final_report.md`](reports/final_report.md) · [`reports/final_report.html`](reports/final_report.html) |
| Portfolio copy | [`docs/final-report.html`](docs/final-report.html) |
| Interim report | [`reports/Interim_Report.md`](reports/Interim_Report.md) |
| Figures | `reports/figures_final/` |

Export PDF: open the HTML in Chrome/Edge → **Ctrl+P** → **Save as PDF** (enable background graphics).

Regenerate charts:

```bash
set PYTHONPATH=.
python scripts/generate_final_report_figures.py
```

---

## CI/CD

On every push to `main`:

1. `ruff check` + `ruff format --check`
2. `pytest tests/ -v`

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## License

Educational project for the 10 Academy Week 4 Credit Risk Probability Model challenge.
