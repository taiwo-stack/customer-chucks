# Project Deep Dive: Customer Churn & CLV Analysis

I have completed a deep dive into the "Customer Chucks" project. This report summarizes the technology stack, workflow, and strategic choices made to optimize the model and business outcomes.

## 🏗️ Technology Stack
- **Language**: Python 3.x
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest, XGBoost
- **Interpretability**: SHAP (Tree-based Models), Logistic Coefficients (Linear Models)
- **Frontend/Dashboard**: Streamlit (with custom CSS for premium aesthetics)
- **Deployment & Persistence**: Joblib for model serialization

## 🔄 Project Workflow
The project follows a modular, scalable architecture:
1. **Data Ingestion**: Automatically downloads the raw IBM Telco dataset.
2. **Modular Preprocessing**: `src/data_prep.py` handles cleaning (e.g., numeric coercion) and feature engineering.
3. **Automated Training**: `src/train_models.py` trains multiple models, handles class imbalance, and optimizes thresholds automatically.
4. **Interactive Dashboard**: `app.py` serves as the user interface, providing real-time predictions and executive-level strategy insights.

## 🚀 Key "Betterment" Choices
- **Business-Driven Features**: We engineered features like `internet_no_tech_support` and `tenure_buckets` which are highly correlated with churn, rather than relying solely on raw data.
- **Imbalance Mitigation**: Instead of standard accuracy, we used `class_weight='balanced'` and `scale_pos_weight`.
- **Threshold Optimization**: We manually tuned the classification thresholds to ensure **Recall > 60%**, acknowledging that in churn prediction, missing a real churner is more costly than a false alarm.
- **CLV Integration**: By calculating Customer Lifetime Value (CLV), we transitioned the project from a simple "Yes/No" churn model to a **Value-Based Retention Tool**.

## 📊 Business Insights
- **Contract Risk**: Month-to-month contracts are the most significant predictor of churn.
- **Onboarding Gap**: Fiber optic customers without Tech Support show a significantly higher risk profile.
- **Value Concentration**: Premium customers are at high risk, and their retention is 5x more valuable than those in the "Low" segment.

## 📑 Presentation Content
A full slide-by-slide presentation outline has been prepared to help communicate these findings to stakeholders.
[View Presentation Content](file:///C:/Users/HP/.gemini/antigravity/brain/5927d7dc-10ab-420a-8021-c8d834c89afb/presentation_content.md)
