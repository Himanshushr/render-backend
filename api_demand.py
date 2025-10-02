import io
import pandas as pd
import random
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# --- Pydantic Models ---
class ForecastItem(BaseModel):
    Month: str
    Project_ID: str
    Material_Type: str
    DemandQuantity: int = Field(alias="Actual_DemandQuantity") 
    Predicted_DemandQuantity: int

class ForecastResponse(BaseModel):
    status: str
    count: int
    forecasts: List[ForecastItem]

# --- FastAPI Setup ---
app = FastAPI(title="PowerGrid API")

# Crucial for allowing Netlify's domain to access your API's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins during development. RESTRICT this in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock Prediction Logic ---
def generate_mock_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Mocks the machine learning prediction process."""
    if 'DemandQuantity' not in df.columns:
        raise ValueError("Input data must contain a 'DemandQuantity' column.")

    df['Predicted_DemandQuantity'] = df['DemandQuantity'].apply(
        lambda x: max(0, x + random.randint(-15, 15)) 
    ).astype(int)
    
    df = df.rename(columns={'DemandQuantity': 'Actual_DemandQuantity'})
    return df

# --- API Endpoint ---
@app.post("/predict", response_model=ForecastResponse)
async def predict_demand(file: UploadFile = File(...)):
    """Handles the CSV upload and returns the demand forecast."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type.")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        forecast_df = generate_mock_prediction(df)

        forecast_list = forecast_df.to_dict(orient='records')
        
        return ForecastResponse(
            status="success",
            count=len(forecast_list),
            forecasts=forecast_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")