import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import requests
import json
import time
import joblib

# --- Setup logging for better visibility ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the Crop Yield Prediction Model setup...")

# A placeholder API key for the Gemini API
apiKey = "AIzaSyCjFAHyFzSZdqSew1IA4UHspe18hRgMg1M"

# --- Helper function to simulate data loading from CSVs ---
def load_and_merge_data():
    """Simulates loading agricultural data, weather, and soil data from files."""
    logging.info("Simulating data loading and merging with more data features...")
    
    # Generate synthetic data for demonstration with more features
    np.random.seed(42)
    num_samples = 2000 # Increased for better model accuracy
    
    # Simulate a time series for crop growth stages
    dates = pd.date_range('2022-01-01', periods=num_samples, freq='D')
    crop_growth_stages = np.random.choice(['Planting', 'Vegetative', 'Flowering', 'Harvest'], num_samples, p=[0.1, 0.4, 0.3, 0.2])
    
    data = pd.DataFrame({
        'crop_type': np.random.choice(['Corn', 'Wheat', 'Soybeans', 'Rice'], num_samples),
        'region': np.random.choice(['Region A', 'Region B', 'Region C'], num_samples),
        'planting_date': dates,
        'crop_growth_stage': crop_growth_stages,
        'yield_kg_per_hectare': np.random.randint(2000, 8000, num_samples) + np.random.normal(0, 500, num_samples)
    })
    
    weather = pd.DataFrame({
        'region': np.random.choice(['Region A', 'Region B', 'Region C'], num_samples),
        'planting_date': dates,
        'rainfall_mm': np.random.uniform(0, 100, num_samples),
        'avg_temp_c': np.random.uniform(15, 35, num_samples),
        'humidity_percent': np.random.uniform(40, 95, num_samples)
    })
    
    soil = pd.DataFrame({
        'region': np.random.choice(['Region A', 'Region B', 'Region C'], num_samples),
        'planting_date': dates,
        'soil_ph': np.random.uniform(5.5, 7.5, num_samples),
        'soil_nitrogen': np.random.uniform(5, 20, num_samples),
        'soil_moisture': np.random.uniform(10, 50, num_samples),
        'soil_organic_matter': np.random.uniform(1, 5, num_samples) # New feature
    })

    # Simulate actionable recommendations as ground truth for training
    recommendations = pd.DataFrame({
        'region': np.random.choice(['Region A', 'Region B', 'Region C'], num_samples),
        'planting_date': dates,
        'irrigation_rec': np.random.choice(['High', 'Medium', 'Low'], num_samples, p=[0.3, 0.5, 0.2]),
        'fertilizer_rec': np.random.choice(['NPK 15-15-15', 'NPK 10-20-10', 'None'], num_samples, p=[0.4, 0.3, 0.3]),
        'pest_control_rec': np.random.choice(['Organic Spray', 'Chemical Treatment', 'Monitor'], num_samples, p=[0.2, 0.1, 0.7])
    })
    
    # Merge all datasets on common keys
    df = data.merge(weather, on=['region', 'planting_date']).merge(soil, on=['region', 'planting_date']).merge(recommendations, on=['region', 'planting_date'])
    logging.info(f"Merged dataset shape with new features: {df.shape}")
    
    return df

# --- Preprocessing Pipeline ---
def preprocess_data(df):
    """
    Handles feature engineering, normalization, and encoding for the dataset.
    
    Args:
        df (pd.DataFrame): The raw, merged DataFrame.
        
    Returns:
        tuple: A tuple containing the preprocessed features (X) and labels (y_yield),
               and the ColumnTransformer for future use.
    """
    logging.info("Preprocessing data...")
    
    # Define features and labels for the yield model
    features = df[['crop_type', 'region', 'crop_growth_stage', 'rainfall_mm', 'avg_temp_c',
                   'humidity_percent', 'soil_ph', 'soil_nitrogen', 'soil_moisture', 'soil_organic_matter']]
    y_yield = df['yield_kg_per_hectare']
    
    # Define labels for the recommendation models
    y_irrigation = df['irrigation_rec']
    y_fertilizer = df['fertilizer_rec']
    y_pest = df['pest_control_rec']
    
    # Identify numerical and categorical features
    numerical_features = ['rainfall_mm', 'avg_temp_c', 'humidity_percent',
                          'soil_ph', 'soil_nitrogen', 'soil_moisture', 'soil_organic_matter']
    categorical_features = ['crop_type', 'region', 'crop_growth_stage']
    
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Fit and transform the features
    X = preprocessor.fit_transform(features)
    
    logging.info("Preprocessing complete.")
    return X, y_yield, y_irrigation, y_fertilizer, y_pest, preprocessor

# --- Recommendation System Logic using Gemini API ---
def get_recommendation_from_gemini(crop_type, current_weather, soil_health, predicted_yield):
    """
    Generates an actionable farming recommendation using a Large Language Model.
    
    Args:
        crop_type (str): The type of crop.
        current_weather (str): A summary of weather conditions.
        soil_health (str): A summary of soil health.
        predicted_yield (float): The predicted crop yield.
        
    Returns:
        str: A generated recommendation for the farmer.
    """
    logging.info("Generating recommendation using Gemini API...")
    
    prompt = (
        f"You are a helpful AI agricultural expert. Based on the following data, "
        f"provide a concise, single-paragraph recommendation for a small-scale farmer to "
        f"optimize their {crop_type} crop.\n\n"
        f"Current Conditions:\n"
        f"Weather: {current_weather}\n"
        f"Soil: {soil_health}\n"
        f"Predicted Yield: {predicted_yield:.2f} kg/hectare.\n\n"
        f"Your recommendation should focus on actionable advice for irrigation, "
        f"fertilization, or pest control to improve the predicted yield."
    )
    
    # Gemini API endpoint
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    retries = 3
    for i in range(retries):
        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No recommendation available.')
            return text
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed: {e}")
            if i < retries - 1:
                wait_time = 2 ** i
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return "Failed to get recommendation from AI after multiple retries."

# --- Main function to run the pipeline ---
def main():
    """Main function to run the entire ML pipeline and save the models."""
    try:
        # Step 1: Load and prepare data
        df = load_and_merge_data()
        X, y_yield, y_irrigation, y_fertilizer, y_pest, preprocessor = preprocess_data(df)
        
        # Step 2: Split data for training and testing
        X_train, X_test, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
        X_train_rec, X_test_rec, y_irrigation_train, y_irrigation_test = train_test_split(X, y_irrigation, test_size=0.2, random_state=42)
        X_train_rec, X_test_rec, y_fertilizer_train, y_fertilizer_test = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
        X_train_rec, X_test_rec, y_pest_train, y_pest_test = train_test_split(X, y_pest, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")
        
        # Step 3: Train and evaluate Random Forest Regressor for Yield
        logging.info("Training Random Forest Regressor model for Yield...")
        rf_model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_yield.fit(X_train, y_yield_train)
        y_pred_yield = rf_model_yield.predict(X_test)
        
        mae_yield = mean_absolute_error(y_yield_test, y_pred_yield)
        r2_yield = r2_score(y_yield_test, y_pred_yield)
        logging.info(f"Random Forest Yield MAE: {mae_yield:.2f}, R² Score: {r2_yield:.2f}")

        # Step 4: Train and evaluate Random Forest Classifiers for Recommendations
        logging.info("Training models for Irrigation, Fertilization, and Pest Control...")
        
        rf_model_irrigation = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_irrigation.fit(X_train_rec, y_irrigation_train)
        
        rf_model_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_fertilizer.fit(X_train_rec, y_fertilizer_train)
        
        rf_model_pest = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_pest.fit(X_train_rec, y_pest_train)
        
        # Step 5: Save all trained models and the preprocessor
        joblib.dump(rf_model_yield, 'rf_model_yield.pkl')
        joblib.dump(rf_model_irrigation, 'rf_model_irrigation.pkl')
        joblib.dump(rf_model_fertilizer, 'rf_model_fertilizer.pkl')
        joblib.dump(rf_model_pest, 'rf_model_pest.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        logging.info("All models and preprocessor saved as .pkl files.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
