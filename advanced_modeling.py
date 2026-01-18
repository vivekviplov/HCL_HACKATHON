import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess

# Auto-install libraries
required_packages = ['shap', 'xgboost', 'scikit-learn', 'pandas', 'numpy', 'matplotlib']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import shap
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def advanced_analysis():
    print("Loading data...")
    try:
        df = pd.read_csv("House Price Prediction Dataset_Fixed.csv")
    except FileNotFoundError:
        print("Fixed dataset not found. Please run fix_data.py first.")
        return

    # Preprocessing (Same as before)
    df_encoded = pd.get_dummies(df, columns=['Location', 'Condition'], drop_first=True)
    df_encoded['Garage'] = df_encoded['Garage'].map({'Yes': 1, 'No': 0})
    
    df_eng = df_encoded.copy()
    df_eng['House_Age'] = 2026 - df_eng['YearBuilt']
    df_eng['Bath_Bed_Ratio'] = df_eng['Bathrooms'] / (df_eng['Bedrooms'] + 0.0001)
    df_eng['Location_Score'] = (df_eng.get('Location_Urban', 0) * 3 + 
                                df_eng.get('Location_Suburban', 0) * 2 + 
                                df_eng.get('Location_Rural', 0) * 1)
    
    feature_cols = [col for col in df_eng.columns if col not in ['Id', 'Price']]
    X = df_eng[feature_cols]
    y = df_eng['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. STACKING REGRESSOR (The "Unique" Model)
    # Combines the strengths of Linear (for trend) and Tree (for non-linear nuances)
    print("\n--- Training Stacking Regressor ---")
    
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
    ]
    
    # Final estimator uses the predictions of the above to make a final decision
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    )
    
    stacking_reg.fit(X_train, y_train)
    y_pred_stack = stacking_reg.predict(X_test)
    r2_stack = r2_score(y_test, y_pred_stack)
    
    print(f"Stacking Regressor R2: {r2_stack:.5f}")
    print("(This model combines Random Forest and XGBoost, usually providing better stability than either alone.)")

    # 2. SHAP ANALYSIS (Explainable AI)
    # This explains WHY the model made a specific prediction. Very impressive for showcases.
    print("\n--- Performing SHAP Analysis ---")
    
    # We use the XGBoost model for SHAP as it's tree-based and fast
    model_xgb = estimators[1][1]
    model_xgb.fit(X_train, y_train)
    
    explainer = shap.Explainer(model_xgb)
    shap_values = explainer(X_test)

    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance (Global Interpretability)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("Saved 'shap_summary.png'")

    # Waterfall Plot for a single prediction (Local Interpretability)
    # Explains the first house in the test set
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"Why was this house predicted at ${model_xgb.predict(X_test.iloc[[0]])[0]:,.0f}?")
    plt.tight_layout()
    plt.savefig("shap_waterfall.png")
    print("Saved 'shap_waterfall.png'")
    
    print("\nDone! 'Stacking' makes your model robust, and 'SHAP' makes it explainable.")

if __name__ == "__main__":
    advanced_analysis()
