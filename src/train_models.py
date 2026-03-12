import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate():
    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_val = val_df.drop('Churn', axis=1)
    y_val = val_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']
    
    # Define categorical and numerical columns
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod', 'tenure_bucket']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'services_count', 
                'monthly_to_total_ratio', 'internet_no_tech_support', 'SeniorCitizen']
    
    # Preprocessing
    # Specify categories for OrdinalEncoder to match tips
    # Gender: Female, Male (alphabetical: Female=0, Male=1)
    # PaymentMethod: alphabetical
    # MultipleLines: No, No Phone Service, Yes (matches tips)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ])
    
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'rf': RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=4),
    }
    
    # Try to add XGBoost if available
    try:
        from xgboost import XGBClassifier
        models['xgb'] = XGBClassifier(random_state=42, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        print("XGBoost is available and added to models.")
    except ImportError:
        print("XGBoost is not available. Skipping.")
    
    trained_models = {}
    results = {}
    
    os.makedirs("models", exist_ok=True)
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_proba)
            }
            
            joblib.dump(clf, f"models/{name}.pkl")
            trained_models[name] = clf
        except Exception as e:
            print(f"Error training {name}: {e}")
        
    # Save preprocessor separately if needed, but it's inside the pipeline
    # We might want just the fitted preprocessor for interpretability tool
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    # Print results
    print("\nModel Performance on Test Set:")
    perf_df = pd.DataFrame(results).T
    print(perf_df)
    
    # Save metrics to a file for the app
    perf_df.to_csv("models/metrics.csv")
    
    return trained_models, results

if __name__ == "__main__":
    train_and_evaluate()
