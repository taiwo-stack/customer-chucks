import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import requests

def download_data(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Data already exists.")

def clean_data(df):
    # TotalCharges has empty spaces which make it an object instead of float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing TotalCharges with 0 (assuming new customers with tenure 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Drop CustomerID as it's not useful for modeling
    df = df.drop('customerID', axis=1)
    
    # Binary encoding for categorical with 2 values
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def feature_engineering(df):
    # tenure_bucket: 0-6m, 6-12m, 12-24m, 24m+
    def bucket_tenure(t):
        if t <= 6: return '0-6m'
        elif t <= 12: return '6-12m'
        elif t <= 24: return '12-24m'
        else: return '24m+'
    
    df['tenure_bucket'] = df['tenure'].apply(bucket_tenure)
    
    # services_count: total number of services
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Count 'Yes' (and 'Fiber optic'/'DSL' for InternetService)
    # MultipleLines No phone service = 0
    # InternetService No = 0
    
    df['services_count'] = (df[service_cols] != 'No').sum(axis=1)
    # Adjust for InternetService 'No' vs 'DSL'/'Fiber optic'
    # Adjust for MultipleLines 'No phone service' vs 'No'/'Yes'
    
    # More precise count:
    service_count = (df['PhoneService'] == 'Yes').astype(int) + \
                    (df['MultipleLines'] == 'Yes').astype(int) + \
                    (df['InternetService'] != 'No').astype(int) + \
                    (df['OnlineSecurity'] == 'Yes').astype(int) + \
                    (df['OnlineBackup'] == 'Yes').astype(int) + \
                    (df['DeviceProtection'] == 'Yes').astype(int) + \
                    (df['TechSupport'] == 'Yes').astype(int) + \
                    (df['StreamingTV'] == 'Yes').astype(int) + \
                    (df['StreamingMovies'] == 'Yes').astype(int)
    df['services_count'] = service_count

    # monthly_to_total_ratio: TotalCharges / max(1, tenure * MonthlyCharges)
    df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, df['tenure'] * df['MonthlyCharges'])
    
    # flags like “internet but no tech support”
    df['internet_no_tech_support'] = ((df['InternetService'] != 'No') & (df['TechSupport'] == 'No')).astype(int)
    
    return df

def split_and_save(df, processed_path):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['Churn']) # 0.25 * 0.8 = 0.2
    
    train_df.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_path, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
    
    print(f"Data split and saved to {processed_path}")

def run_data_prep():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_path = "data/raw/Telco-Customer-Churn.csv"
    processed_path = "data/processed"
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    download_data(url, raw_path)
    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    split_and_save(df, processed_path)

if __name__ == "__main__":
    run_data_prep()
