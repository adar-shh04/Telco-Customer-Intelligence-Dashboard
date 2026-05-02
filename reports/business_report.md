# Telco Customer Intelligence Platform: Executive Brief

## Objective

Build an end-to-end analytics and machine learning workflow to help a telecom business:

- Predict churn risk and identify key drivers
- Prioritize retention actions by expected value, not just risk
- Recommend next-best-offers to drive revenue uplift
- Deliver results in business-friendly surfaces: dashboard, API, and PDF report

## Data

Base dataset:

- IBM Telco Customer Churn dataset (Kaggle), 7,043 customers

Project enrichment:

- The pipeline derives additional customer 360 signals such as usage, CRM, engagement, and campaign fields to simulate a richer enterprise schema while keeping the original churn label.

## Data Science Methods

- Feature engineering: RFM scores, usage ratios, health scores, risk indicators, engagement metrics, customer segments
- Predictive modeling: gradient boosting / LightGBM / XGBoost options for churn prediction
- Serving pipeline: saved sklearn artifact with train-only preprocessing for API inference
- Explainability: SHAP global importance and customer-level explanation support
- Decisioning: churn driver analysis plus ROI-ranked retention actions
- Experimentation: A/B test simulation with p-values, confidence intervals, churn reduction, and ROI
- Monitoring: anomaly detection across customer and segment-level KPIs

## Solution Overview

This repository ships three ways to consume the work:

1. Streamlit dashboard: interactive BI and decisioning workflows.
2. FastAPI service: real-time predictions for churn and next-best-offer.
3. PDF reporting: executive-style summaries generated from computed artifacts.

## Key Deliverables

- `reports/Telecom_Revenue_Optimization_Report.pdf`: sample executive PDF output
- `data/telecom.db`: SQLite database for ad-hoc SQL analysis
- `data/raw/*.csv`: normalized customer tables plus master dataset
- `data/processed/*.csv`: model and analytics artifacts used by dashboard/reporting
- `reports/metrics.md`: generated churn model evaluation report after running `python train_model.py`

## Limitations

- Several customer 360 fields are derived to make the project closer to a real product analytics environment.
- The NBO conversion labels are simulated because the source dataset does not include historical offer-response labels.

Recommended production upgrade:

- Add a single saved sklearn pipeline for training and serving.
- Add threshold tuning, calibration, PR-AUC, and model monitoring.
- Replace simulated NBO labels with real offer exposure and acceptance history when available.
