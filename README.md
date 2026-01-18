# 🏡 Advanced House Price Predictor

## Overview
This project is an advanced machine learning application designed to predict house prices with high accuracy and interpretability. It goes beyond standard datasets by incorporating realistic, synthetic features such as proximity to amenities (schools, hospitals, transport), renovation status, and property type.

The core of the project is an interactive **Streamlit Web Application** powered by an **XGBoost** regressor. It features **SHAP (SHapley Additive exPlanations)** to provide transparency, explaining exactly *why* a specific price was predicted.

## ✨ Key Features
*   **Rich Feature Set**: Includes standard features (Area, Bedrooms) plus advanced metrics:
    *   **Proximity**: Distance to Schools, Hospitals, Transport, and City Center.
    *   **Property Details**: Apartment vs. Individual House, Furnished status, Renovation status.
    *   **Neighborhood**: Location-based price trends and school ratings.
*   **High-Performance Model**: Uses **XGBoost**, a state-of-the-art gradient boosting algorithm, achieving high $R^2$ scores.
*   **Explainable AI (XAI)**: Integrated **SHAP Waterfall Plots** visualize the positive or negative contribution of every feature to the final price.
*   **Interactive UI**: User-friendly sidebar to adjust all house parameters in real-time.
*   **Real-time Metrics**: Displays model performance metrics (RMSE, MAE, $R^2$) directly in the app.

## 🛠️ Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit
    ```

## 🚀 Usage

### 1. Generate the Dataset
If the dataset is missing or you want to regenerate it with new random seed values:
```bash
python generate_rich_data.py
```
*This creates `House Price Prediction Dataset_Rich_v5.csv` with all the latest features.*

### 2. Run the Web Application
Launch the interactive dashboard:
```bash
streamlit run app.py
```
The app will open in your default web browser (usually at `http://localhost:8501`).

### 3. (Optional) Advanced Analysis
To run a standalone script for model training, stacking, and static plot generation:
```bash
python advanced_modeling.py
```

## 📂 Project Structure

*   **`app.py`**: The main Streamlit application file. Handles data loading, model training, UI rendering, and prediction.
*   **`generate_rich_data.py`**: Script to generate the synthetic "Rich" dataset. It adds logic for distances, renovation premiums, and inflation-adjusted price history.
*   **`advanced_modeling.py`**: A script demonstrating advanced techniques like Stacking Regressors and static SHAP summary plots.
*   **`train_models.py`**: Basic script for initial model comparison (Linear Regression vs. Random Forest vs. XGBoost).
*   **`House Price Prediction Dataset_Rich_v5.csv`**: The dataset used by the application.

## 📊 Model Logic
The model pricing logic considers realistic real-estate factors:
*   **Base Price**: Derived from Area and Room counts.
*   **Premiums**: Added for "Individual House" ($150k), "Renovated" ($40k), "Furnished" ($25k), and "Pool" ($30k).
*   **Penalties**: Subtracted for distance from key amenities (School, Hospital, Transport).
*   **Inflation**: Historical price trends account for a ~3% annual inflation rate plus location-specific appreciation.

## 🤝 Contributing
Feel free to fork this project and submit pull requests. Suggestions for new features or better model tuning are welcome!
