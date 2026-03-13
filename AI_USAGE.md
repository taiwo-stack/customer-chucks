# AI Usage Report

## What AI helped with
- **Project Structure**: Suggested a modular Python structure for data preparation, modeling, and application code.
- **Data Engineering**: Assisted in defining business-driven features such as `tenure_bucket`, `services_count`, and `monthly_to_total_ratio`.
- **Model Optimization**: Provided the logic for using `class_weight='balanced'` and threshold tuning to significantly improve recall from ~45% to over 70-80% to meet business requirements.
- **Streamlit UI**: Assisted in creating the three-tab dashboard layout with custom CSS for a premium look and organized input sections.
- **Interpretability**: Helped implement standardized coefficient analysis for Logistic Regression and importance fallbacks for XGB/RF.

## Prompts that mattered
- "IBM Telco Customer Churn dataset raw csv url"
- "Standardized coefficients for logistic regression feature importance"
- "Streamlit caching for heavy models"
- "How to optimize model recall for churn prediction with class imbalance"

## What was fixed/verified manually
1.  **Dependency Resilience**: Manually refactored `interpret_utils.py` and `app.py` to handle environments where `shap` or `xgboost` might be missing or cached incorrectly.
2.  **Syntax Debugging**: Fixed a Python syntax error in the nested `if` statements for tenure bucketing logic.
3.  **Profile Validation**: Manually verified a "High-Risk" customer profile (Senior, Month-to-Month, Fiber Optic) result, which the model correctly identified with >90% churn probability.
4.  **UI Refinement**: Iteratively adjusted the dashboard layout to be more compact and professional, including the use of nested columns for form inputs.
5.  **Pathing & Imports**: Corrected import paths after refactoring utility scripts to ensure a clean application reload.
