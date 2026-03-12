# AI Usage Report

## What AI helped with
- **Project Structure**: Suggested a modular Python structure.
- **Data Engineering**: Assisted in defining `tenure_bucket` and `monthly_to_total_ratio` logic.
- **Model Debugging**: Helped transition from a unified training script to one that handles missing dependencies (XGBoost) gracefully.
- **Streamlit UI**: Provided the layout and tab structure for the interactive dashboard.

## Prompts that mattered
- "IBM Telco Customer Churn dataset raw csv url"
- "Standardized coefficients for logistic regression feature importance"
- "Streamlit caching for heavy models"

## What was fixed/verified manually
- Verified the `TotalCharges` cleanup logic (converting " " to numeric).
- Manually checked the Senior Citizen + Fiber Optic churn risk scenario.
- Corrected a typo in the Stremlit metrics loading logic.
- Handled the environment-specific installation issue for XGBoost and SHAP by implementing robust error handling and fallback models.
