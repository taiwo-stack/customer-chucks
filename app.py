import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.clv_analysis import analyze_clv
from src.interpret_utils import get_shap_explanations, get_logistic_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- Page Config ---
st.set_page_config(page_title="Customer Churn & CLV Dashboard", layout="wide")

# --- Caching ---
@st.cache_data
def load_data():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, val, test

@st.cache_resource
def load_models():
    models = {}
    for name in ['logistic', 'rf', 'xgb']:
        path = f"models/{name}.pkl"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    preprocessor = joblib.load("models/preprocessor.pkl")
    return models, preprocessor

# --- Main App ---
def main():
    st.title("🎯 Customer Churn Prediction & CLV Analysis")
    
    if not os.path.exists("models/metrics.csv") or not os.path.exists("data/processed/train.csv"):
        st.warning("Project data or models not found. Please ensure all preparation scripts have run.")
        return

    train_df, val_df, test_df = load_data()
    models, preprocessor = load_models()
    
    tabs = st.tabs(["🚀 Predict Churn", "📊 Model Performance", "💰 CLV Overview"])
    
    with tabs[0]:
        st.header("Predict Individual Churn Risk")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Customer Details")
            # Feature inputs
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 1)
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            multiple = st.selectbox("Multiple Lines", ["No", "No Phone Service", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "No Internet Service", "Yes"])
            backup = st.selectbox("Online Backup", ["No", "No Internet Service", "Yes"])
            protection = st.selectbox("Device Protection", ["No", "No Internet Service", "Yes"])
            support = st.selectbox("Tech Support", ["No", "No Internet Service", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "No Internet Service", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "No Internet Service", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges", value=50.0)
            total = st.number_input("Total Charges", value=monthly * tenure if tenure > 0 else 0.0)
            
        # Feature Engineering for Single Input
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        # New Feature construction
        def get_tenure_bucket(t):
            if t <= 6: return '0-6m'
            elif t <= 12: return '6-12m'
            elif t <= 24: return '12-24m'
            else: return '24m+'
        
        input_data['tenure_bucket'] = get_tenure_bucket(tenure)
        input_data['services_count'] = [phone, multiple, internet, security, backup, protection, support, tv, movies].count("Yes") + (1 if internet != "No" else 0)
        input_data['monthly_to_total_ratio'] = total / max(1, tenure * monthly)
        input_data['internet_no_tech_support'] = 1 if internet != "No" and support == "No" else 0
        
        input_df = pd.DataFrame([input_data])
        
        with col2:
            st.subheader("Prediction Results")
            available_models = list(models.keys())
            if not available_models:
                st.error("No models available. Please run training first.")
                return
                
            selected_model_name = st.selectbox("Select Model for Prediction", available_models)
            model = models[selected_model_name]
            
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            
            if prob > 0.6:
                st.error("Churn Risk: HIGH")
            elif prob > 0.3:
                st.warning("Churn Risk: MEDIUM")
            else:
                st.success("Churn Risk: LOW")
                
            clv = monthly * 24 # Assuming 24 months
            st.info(f"Estimated CLV: ${clv:,.2f} (24 months expected tenure)")
            st.code("Formula: MonthlyCharges * 24", language="python")

    with tabs[1]:
        st.header("Model Performance Metrics")
        if os.path.exists("models/metrics.csv"):
            metrics_df = pd.read_csv("models/metrics.csv", index_col=0)
            st.table(metrics_df)
        else:
            st.info("Metrics not found. Training might be in progress.")
        
        st.subheader("Visual Analysis")
        model_choice = st.selectbox("Select Model for Visuals", list(models.keys()))
        model = models[model_choice]
        
        col_v1, col_v2 = st.columns(2)
        
        # Confusion Matrix
        y_test = test_df['Churn']
        X_test = test_df.drop('Churn', axis=1)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        with col_v1:
            st.write("**Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            
        # ROC Curve
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        with col_v2:
            st.write(f"**ROC Curve (AUC = {roc_auc:.2f})**")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            st.pyplot(fig_roc)

        st.subheader("Importance Analysis")
        if model_choice == 'logistic':
            importance_df = get_logistic_importance(model, train_df.drop('Churn', axis=1))
            st.bar_chart(importance_df.set_index('Feature')['Importance'].head(10))
        else:
            # Fallback for RF importance
            importances = model.named_steps['classifier'].feature_importances_
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature')['Importance'].head(10))

    with tabs[2]:
        st.header("CLV & Business Insights")
        churn_rates, insights = analyze_clv(train_df)
        
        st.subheader("Churn Rate by CLV Segment")
        st.bar_chart(churn_rates)
        
        st.subheader("Key Takeaways")
        for insight in insights:
            st.write(f"- {insight}")

if __name__ == "__main__":
    main()
