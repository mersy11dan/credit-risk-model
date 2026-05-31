# Credit Risk Model

Machine learning pipeline for credit default prediction, including data processing, model training, batch inference, and a REST API.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI pipeline
├── data/
│   ├── raw/                   # Raw input data
│   └── processed/             # Processed datasets
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── data_processing.py     # Data loading & preprocessing
│   ├── train.py               # Model training script
│   ├── predict.py             # Batch prediction script
│   └── api/                   # FastAPI scoring service
├── tests/                     # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train a model:**

```bash
python -m src.train --data data/raw/credit.csv --model models/model.pkl
```

**Run batch predictions:**

```bash
python -m src.predict --model models/model.pkl --data data/processed/test.csv --output data/processed/predictions.csv
```

**Start the API:**

```bash
uvicorn src.api.main:app --reload
```

**Run tests:**

```bash
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.
