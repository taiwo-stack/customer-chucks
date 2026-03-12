# Customer Churn Prediction & CLV Analysis

## Business Context
Churn is a critical metric for SaaS companies. This project predicts churn risk and estimates Customer Lifetime Value (CLV) to help prioritize retention efforts.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data preparation: `python src/data_prep.py`
3. Train models: `python src/train_models.py`
4. Launch app: `streamlit run app.py`

## CLV Assumptions
- **Expected Tenure**: 24 months. This is a conservative estimate for long-term customer value in the telecommunications sector.
- **CLV Formula**: `MonthlyCharges * 24`.

## Features
- **Predictive Modeling**: Logistic Regression and Random Forest (AUC-ROC ~0.84).
- **Interpretability**: Coefficient analysis for LR and Feature Importance for RF.
- **Interactive App**: Multi-tab interface for predictions, performance, and business insights.

## Repository Structure
- `src/`: Modular logic for data, clv, and training.
- `models/`: Processed models and metrics.
- `data/`: Raw and processed datasets.
- `app.py`: Streamlit entry point.
