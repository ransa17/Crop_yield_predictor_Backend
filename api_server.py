import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import requests
import json
import time

# --- Setup logging for better visibility ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting API server setup...")

# A placeholder API key for the Gemini API
apiKey = "AIzaSyCjFAHyFzSZdqSew1IA4UHspe18hRgMg1M"

# --- Helper function for simple AI-powered recommendations ---
def get_ai_recommendation(prompt: str):
    """
    Generates a single-paragraph recommendation from a simple prompt.
    """
    logging.info("Generating AI recommendation...")
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    
    retries = 3
    for i in range(retries):
        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No recommendation available.')
            return text
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed: {e}")
            if i < retries - 1:
                wait_time = 2 ** i
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return "Failed to get recommendation from AI after multiple retries."
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse AI response: {e}")
            return "AI response was unreadable."

# --- Load the pre-trained models and preprocessor ---
try:
    logging.info("Loading pre-trained models and preprocessor...")
    rf_model_yield = joblib.load('rf_model_yield.pkl')
    rf_model_irrigation = joblib.load('rf_model_irrigation.pkl')
    rf_model_fertilizer = joblib.load('rf_model_fertilizer.pkl')
    rf_model_pest = joblib.load('rf_model_pest.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    logging.info("Models and preprocessor loaded successfully.")
except FileNotFoundError:
    logging.error("Model files not found. Please run 'python yield_predictor.py' first.")
    raise HTTPException(status_code=500, detail="Model files not found. Please train the model first.")

# --- FastAPI app setup ---
app = FastAPI(title="AI Farming Platform API")

# --- CORS middleware to allow requests from the React app ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define the input data model for the API endpoint ---
class PredictionRequest(BaseModel):
    crop_type: str
    region: str
    rainfall: float
    temperature: float
    humidity: float
    soilPh: float
    soilNitrogen: float
    soilMoisture: float
    cropGrowthStage: str
    soilOrganicMatter: float = 0.0

# --- API endpoint for prediction ---
@app.post("/predict")
async def predict_yield(request: PredictionRequest):
    """
    Predicts crop yield and provides actionable recommendations.
    """
    logging.info(f"Received prediction request for {request.crop_type} in {request.region}.")

    input_data = pd.DataFrame([{
        'crop_type': request.crop_type,
        'region': request.region,
        'crop_growth_stage': request.cropGrowthStage,
        'rainfall_mm': request.rainfall,
        'avg_temp_c': request.temperature,
        'humidity_percent': request.humidity,
        'soil_ph': request.soilPh,
        'soil_nitrogen': request.soilNitrogen,
        'soil_moisture': request.soilMoisture,
        'soil_organic_matter': request.soilOrganicMatter
    }])
    
    try:
        processed_data = preprocessor.transform(input_data)
        predicted_yield = rf_model_yield.predict(processed_data)[0]
        
        weather_summary = f"rainfall: {request.rainfall}mm, temp: {request.temperature}°C, humidity: {request.humidity}%"
        soil_summary = f"pH: {request.soilPh}, Nitrogen: {request.soilNitrogen}%, Moisture: {request.soilMoisture}%, Organic Matter: {request.soilOrganicMatter}%"
        
        # Make a separate, simple API call for each recommendation
        irrigation_prompt = f"As an agricultural expert, provide a concise, single-paragraph irrigation recommendation for a {request.crop_type} crop,not more than 300 words.at the {request.cropGrowthStage} stage with the following conditions: {weather_summary} and {soil_summary}. The predicted yield is {predicted_yield:.2f} kg/hectare. Focus on specific, actionable advice."
        fertilizer_prompt = f"As an agricultural expert, provide a concise, single-paragraph fertilization recommendation for a {request.crop_type} crop,not more than 300 words.at the {request.cropGrowthStage} stage with the following conditions: {weather_summary} and {soil_summary}. The predicted yield is {predicted_yield:.2f} kg/hectare. Focus on specific, actionable advice."
        pest_control_prompt = f"As an agricultural expert, provide a concise, single-paragraph pest control recommendation for a {request.crop_type} crop,not more than 300 words.at the {request.cropGrowthStage} stage with the following conditions: {weather_summary} and {soil_summary}. The predicted yield is {predicted_yield:.2f} kg/hectare. Focus on specific, actionable advice."
        summary_prompt = f"As an agricultural expert, provide a brief, overall summary for a small-scale farmer to improve their {request.crop_type} crop,not more than 300 words. Include key insights from the following conditions: {weather_summary}, {soil_summary}, and a predicted yield of {predicted_yield:.2f} kg/hectare."
        
        irrigation_rec = get_ai_recommendation(irrigation_prompt)
        fertilizer_rec = get_ai_recommendation(fertilizer_prompt)
        pest_control_rec = get_ai_recommendation(pest_control_prompt)
        summary_rec = get_ai_recommendation(summary_prompt)

        return {
            "predicted_yield": float(predicted_yield),
            "irrigation_rec": irrigation_rec,
            "fertilizer_rec": fertilizer_rec,
            "pest_control_rec": pest_control_rec,
            "summary_rec": summary_rec
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
