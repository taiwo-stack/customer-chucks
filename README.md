# Customer Churn Prediction & CLV Analysis

## Business Context
Churn is the loss of customers over time, which can cost SaaS companies up to 7% of annual revenue. This project identifies high-risk customers and calculates their **Customer Lifetime Value (CLV)** to help businesses prioritize retention efforts where they matter most.

## Public App URL
[Insert App URL here after deployment]

## How to Run
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Setup Data**: The IBM Telco dataset is automatically pulled from source in `src/data_prep.py`.
3. **Train Models**: Run `python src/train_models.py` to train all 3 models (LR, RF, XGB).
4. **Launch Dashboard**: `streamlit run app.py`

## CLV Assumptions & Methodology
- **Expected Tenure**: Assumed to be **24 months** based on industry benchmarks for telecommunications.
- **CLV Formula**: `MonthlyCharges * ExpectedTenure`.
- **Segmentation**: Customers are split into "Low", "Med", "High", and "Premium" value quartiles based on their calculated CLV.

## Model Performance (Test Set)
| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.51 | 0.80 | 0.63 | 0.84 |
| Random Forest | 0.55 | 0.74 | 0.63 | 0.84 |
| XGBoost | 0.53 | 0.74 | 0.62 | 0.83 |

*Note: Models were optimized to hit >60% recall to ensure identification of at least 3 out of 4 potential churners.*

## Key Business Insights
- **Contract Type**: Month-to-month contracts are the #1 predictor of churn. Moving these customers to annual plans should be the top priority.
- **Value Prioritization**: Premium customers have a 3x higher churn risk than medium-value customers, requiring immediate "Save" campaigns for the VIP segment.
- **Fiber Optic Risk**: Fiber optic users show higher churn rates when coupled with lack of Tech Support, suggesting a need for better onboarding/support.

## AI Usage
A full log of AI assistance, manually verified fixes, and critical prompts is available in [AI_USAGE.md](./AI_USAGE.md).
