# Resume Notes - Telco Customer Intelligence Platform

Use this project as a **Data Scientist** portfolio project, not just a dashboard project.

## Best One-Line Project Title

Telco Customer Intelligence Platform: Churn Prediction, Explainable Retention Analytics, and Next-Best-Offer Decisioning

## Resume Bullets

Pick 2-3 depending on space:

- Built an end-to-end telecom customer intelligence platform using Python, scikit-learn, SHAP, Streamlit, and FastAPI to predict churn, explain customer risk drivers, and recommend retention actions from IBM Telco customer data.
- Engineered customer health, usage, billing, CRM, engagement, campaign, and RFM features from a 7K-customer churn dataset to support churn modeling, segmentation, retention prioritization, and business reporting.
- Trained gradient boosting churn models and translated prediction outputs into ROI-ranked intervention recommendations using estimated revenue saved, treatment cost, churn probability, and retention lift assumptions.
- Built a reusable sklearn training pipeline with train-only preprocessing, saved model artifacts, and generated evaluation reports covering ROC-AUC, PR-AUC, calibration error, threshold metrics, and confusion matrix results.
- Implemented SHAP explainability to identify global churn drivers and produce customer-level explanations for business stakeholders.
- Built an experimentation module to simulate A/B tests for retention offers, estimate churn reduction, test statistical significance, and calculate campaign ROI.
- Productized analytics through a Streamlit dashboard and FastAPI prediction API, demonstrating the full path from exploratory analysis to stakeholder-facing delivery.

## Interview Story

Use this structure:

1. **Problem:** Telecom churn directly affects recurring revenue, so the goal was to predict risk and recommend actions, not just classify customers.
2. **Data:** Started from IBM Telco churn data and enriched it into a customer 360 schema with usage, billing, CRM, campaign, and engagement signals.
3. **Modeling:** Used tree-based models for churn prediction because they handle nonlinear effects well and work naturally with SHAP.
4. **Explainability:** Added SHAP to show which factors drive churn globally and for individual customers.
5. **Decisioning:** Converted risk scores into business recommendations using health score, churn driver, intervention cost, expected revenue saved, and ROI.
6. **Delivery:** Saved the trained churn pipeline as a reusable artifact, loaded it in FastAPI for serving, and wrapped insights in a Streamlit dashboard.

## Honest Limitations To Mention

- Some customer 360 fields are derived from the base dataset to simulate richer enterprise data.
- NBO conversion labels are simulated because the source dataset does not contain real offer-acceptance history.
- Next production improvement would be drift monitoring, automated retraining, stronger CI tests, and replacing simulated NBO labels with real offer-response history.

## GitHub Description

End-to-end data science project for telecom churn prediction, SHAP explainability, retention ROI modeling, A/B test simulation, next-best-offer recommendations, Streamlit dashboard, and FastAPI model serving.
