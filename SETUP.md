# Setup and Running Guide

This repository contains:

- A Streamlit dashboard (`dashboard/app.py`)
- A FastAPI prediction API (`api/app.py`)
- A data pipeline that downloads/enriches the IBM Telco churn dataset and builds CSVs + SQLite (`generate_data.py`)

## Prerequisites

- Python 3.10+ recommended
- Docker Desktop (optional)

## Quick Start (Docker)

From the repo root:

```bash
docker-compose up --build
```

- Dashboard: `http://localhost:8501`
- API docs (Swagger): `http://localhost:8000/docs`

## Quick Start (Local, No Docker)

1. Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Build the datasets + SQLite database:

```bash
python generate_data.py
```

This will populate:

- `data/raw/*.csv`
- `data/telecom.db`

4. Train and save the API-facing churn model:

```bash
python train_model.py
```

This writes:

- `artifacts/churn_model.joblib`
- `data/processed/churn_metrics.json`
- `reports/metrics.md`

5. Start the API (Terminal 1):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

6. Start the dashboard (Terminal 2):

```bash
streamlit run dashboard/app.py
```

## Data Notes

- The raw dataset is the IBM Telco Customer Churn dataset (Kaggle).
- `src/data_loader.py` enriches the raw data with derived CRM / usage / engagement / campaign fields to create a multi-table "customer 360" schema.
- If `kagglehub` download fails, the loader prints manual download instructions and the expected local file path.

## Optional: Generate Processed Artifacts

Some dashboard pages and the PDF report use precomputed outputs under `data/processed/`. You can generate them with:

```bash
python src/retention_engine.py
python src/nbo_engine.py
python src/explainability.py
python src/cohort_analysis.py
```
