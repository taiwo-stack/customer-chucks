import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_shap_explanations(model_name, X_processed, feature_names):
    """
    Generate SHAP explanations for tree-based models (RF/XGB).
    """
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Returning None.")
        return None, None
    model_path = f"models/{model_name}.pkl"
    if not os.path.exists(model_path):
        return None
    
    pipeline = joblib.load(model_path)
    model = pipeline.named_steps['classifier']
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    # For binary classification, shap_values might be a list (one for each class) 
    # or a single array depending on the model/version.
    if isinstance(shap_values, list):
        # Using index 1 for 'Churn=Yes'
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    return explainer, shap_values_to_plot

def get_logistic_importance(pipeline, X_train):
    """
    Calculate importance for Logistic Regression.
    importance = |coefficient * std_dev_of_feature|
    """
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after preprocessing
    # This is tricky with ColumnTransformer, but we can approximate or use get_feature_names_out in newer sklearn
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    coeffs = model.coef_[0]
    # For simplicity, we'll use absolute coefficients if scaling was already done
    # Standardized coefficients = coeff * std(x) / std(y)
    # Since we use StandardScaler, std(x) is 1 for numerical, and we'll just use |coeff|
    
    importance = np.abs(coeffs)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

if __name__ == "__main__":
    # This is a placeholder for local testing
    # The actual logic will be used in the Streamlit app
    pass
