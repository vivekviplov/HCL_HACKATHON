import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Advanced House Price Predictor", layout="wide")

@st.cache_resource
def load_data_and_model():
    # Load Data
    try:
        df = pd.read_csv("House Price Prediction Dataset_Rich_v5.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please run generate_rich_data.py first.")
        return None, None, None, None, None

    # Preprocessing
    df_encoded = pd.get_dummies(df, columns=['Location', 'Condition', 'Property_Type'], drop_first=True)
    df_encoded['Garage'] = df_encoded['Garage'].map({'Yes': 1, 'No': 0})
    
    # New Features are already numeric
    
    df_eng = df_encoded.copy()
    df_eng['House_Age'] = 2026 - df_eng['YearBuilt']
    df_eng['Bath_Bed_Ratio'] = df_eng['Bathrooms'] / (df_eng['Bedrooms'] + 0.0001)
    df_eng['Location_Score'] = (df_eng.get('Location_Urban', 0) * 3 + 
                                df_eng.get('Location_Suburban', 0) * 2 + 
                                df_eng.get('Location_Rural', 0) * 1)
    
    # Condition Score (Approximation)
    df_eng['Condition_Score'] = (df_eng.get('Condition_Good', 0) * 3 + 
                                 df_eng.get('Condition_Fair', 0) * 2 + 
                                 df_eng.get('Condition_Poor', 0) * 1)

    # Feature Engineering: Neighborhood Price per SqFt (Target Encoding)
    df['PPSF'] = df['Price'] / df['Area']
    ppsf_map = df.groupby('Location')['PPSF'].mean().to_dict()
    df_eng['Neighborhood_PPSF'] = df['Location'].map(ppsf_map)

    # Drop YearBuilt, ID, Price, and Price_History (string)
    feature_cols = [col for col in df_eng.columns if col not in ['Id', 'Price', 'YearBuilt', 'PPSF', 'Price_History']]
    X = df_eng[feature_cols]
    y = df_eng['Price']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "Best_Model": "XGBoost Regressor",
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2_Score": round(r2, 4)
    }
    
    return model, X.columns, X, ppsf_map, metrics

model, feature_names, X_train_sample, ppsf_map, metrics = load_data_and_model()

if model is not None:
    st.title("🏡 Advanced House Price Predictor")
    
    # Display Metrics in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Model Performance")
    st.sidebar.json(metrics)

    st.markdown("""
    This application uses an **XGBoost** model trained on the **Rich Dataset**. 
    It features **SHAP** to explain predictions and shows **Price Trends**.
    """)

    # Sidebar Inputs
    st.sidebar.header("House Details")
    
    property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Individual House"])
    area = st.sidebar.slider("Area (SqFt)", 500, 5000, 2000)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
    floors = st.sidebar.slider("Floors", 1, 3, 1)
    year_built = st.sidebar.slider("Year Built", 1900, 2023, 1980)
    garage = st.sidebar.selectbox("Garage", ["Yes", "No"])
    location = st.sidebar.selectbox("Location", ["Downtown", "Suburban", "Urban", "Rural"])
    condition = st.sidebar.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Neighborhood & Amenities")
    
    dist_school = st.sidebar.slider("Dist. to School (km)", 0.1, 20.0, 2.0)
    dist_hospital = st.sidebar.slider("Dist. to Hospital (km)", 0.5, 30.0, 5.0)
    dist_transport = st.sidebar.slider("Dist. to Transport (km)", 0.1, 20.0, 1.0)
    dist_center = st.sidebar.slider("Dist. to City Center (km)", 0.5, 60.0, 10.0)
    
    has_pool = st.sidebar.checkbox("Has Pool")
    is_furnished = st.sidebar.checkbox("Is Furnished")
    is_renovated = st.sidebar.checkbox("Is Renovated")

    if st.sidebar.button("Predict Price"):
        # Create Input DataFrame
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'YearBuilt': [year_built],
            'Garage': [garage],
            'Location': [location],
            'Condition': [condition],
            'Property_Type': [property_type],
            'Distance_to_School': [dist_school],
            'Distance_to_Hospital': [dist_hospital],
            'Distance_to_Transport': [dist_transport],
            'Distance_to_Center': [dist_center],
            'Has_Pool': [1 if has_pool else 0],
            'Is_Furnished': [1 if is_furnished else 0],
            'Is_Renovated': [1 if is_renovated else 0],
            'Id': [0], 
            'Price': [0]
        })

        # Preprocess Input
        # Initialize all potential dummy columns to 0
        for col in feature_names:
            input_data[col] = 0
            
        # Fill known values
        input_data['Area'] = area
        input_data['Bedrooms'] = bedrooms
        input_data['Bathrooms'] = bathrooms
        input_data['Floors'] = floors
        input_data['Garage'] = 1 if garage == 'Yes' else 0
        
        # New Features
        input_data['Distance_to_School'] = dist_school
        input_data['Distance_to_Hospital'] = dist_hospital
        input_data['Distance_to_Transport'] = dist_transport
        input_data['Distance_to_Center'] = dist_center
        input_data['Has_Pool'] = 1 if has_pool else 0
        input_data['Is_Furnished'] = 1 if is_furnished else 0
        input_data['Is_Renovated'] = 1 if is_renovated else 0
        
        # Location One-Hot
        if location == 'Rural': input_data['Location_Rural'] = 1
        elif location == 'Suburban': input_data['Location_Suburban'] = 1
        elif location == 'Urban': input_data['Location_Urban'] = 1
        
        # Condition One-Hot
        if condition == 'Fair': input_data['Condition_Fair'] = 1
        elif condition == 'Good': input_data['Condition_Good'] = 1
        elif condition == 'Poor': input_data['Condition_Poor'] = 1
        
        # Property Type One-Hot
        if property_type == 'Individual House':
            if 'Property_Type_Individual House' in input_data.columns:
                input_data['Property_Type_Individual House'] = 1
        
        # Feature Engineering
        input_data['House_Age'] = 2026 - year_built
        input_data['Bath_Bed_Ratio'] = bathrooms / (bedrooms + 0.0001)
        input_data['Is_New_House'] = 1 if (2026 - year_built) < 5 else 0
        
        input_data['Location_Score'] = (input_data.get('Location_Urban', 0) * 3 + 
                                        input_data.get('Location_Suburban', 0) * 2 + 
                                        input_data.get('Location_Rural', 0) * 1)
        
        input_data['Condition_Score'] = (input_data.get('Condition_Good', 0) * 3 + 
                                         input_data.get('Condition_Fair', 0) * 2 + 
                                         input_data.get('Condition_Poor', 0) * 1)
        
        # Map Neighborhood PPSF
        input_data['Neighborhood_PPSF'] = ppsf_map.get(location, 0)

        # Select only feature columns in correct order
        final_input = input_data[feature_names]
        
        # Predict
        prediction = model.predict(final_input)[0]
        
        st.success(f"### Predicted Price: ${prediction:,.2f}")
        


        # SHAP Explanation
        st.subheader("Why this price?")
        explainer = shap.Explainer(model)
        shap_values = explainer(final_input)
        
        # Waterfall Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        st.info("The waterfall plot shows how each feature contributed to moving the price from the average (base value) to the final prediction.")
