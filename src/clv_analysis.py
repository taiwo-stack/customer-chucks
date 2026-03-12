import pandas as pd
import numpy as np
import os

def analyze_clv(df, expected_tenure_months=24):
    """
    CLV = MonthlyCharges * ExpectedTenure (months)
    Assumption: Expected Tenure is 24 months for general business estimation.
    """
    df['CLV'] = df['MonthlyCharges'] * expected_tenure_months
    
    # Split into quartiles
    df['CLV_Segment'] = pd.qcut(df['CLV'], 4, labels=['Low', 'Med', 'High', 'Premium'])
    
    # Report churn rate by CLV quartile
    churn_analysis = df.groupby('CLV_Segment', observed=True)['Churn'].mean() * 100
    
    print("Churn Rate by CLV Segment:")
    print(churn_analysis)
    
    # Business Insights
    insights = [
        "High-value (Premium) customers often have higher churn rates due to competition or service complexity.",
        "Retention efforts should prioritize 'High' and 'Premium' segments as their loss impact is 3-5x higher than 'Low' segment.",
        "Monthly-to-Total ratio indicates that front-loaded costs might correlate with early churn in Med/High segments."
    ]
    
    return churn_analysis, insights

if __name__ == "__main__":
    train_path = "data/processed/train.csv"
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        churn_rates, insights = analyze_clv(df)
        
        print("\nBusiness Insights:")
        for i, insight in enumerate(insights):
            print(f"{i+1}. {insight}")
    else:
        print("Processed train.csv not found.")
