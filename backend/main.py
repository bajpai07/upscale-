import os
import sys
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# Add project root directory to python path for module imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_customer_upsell, load_upsell_pipeline
from src.shap_explainer import explain_customer_prediction

# Data path for lookup
PROCESSED_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "cdr_features.csv"))

app = FastAPI(
    title="Telco Customer Upsell Classification API",
    description="Machine Learning REST API for predicting upsell readiness with SHAP explanations and Responsible AI guardrails.",
    version="1.0.0"
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerPayload(BaseModel):
    phone_number: Optional[str] = Field(default="Unknown", alias="Phone Number")
    account_length: int = Field(default=100, alias="Account Length")
    vmail_message: int = Field(default=0, alias="VMail Message")
    custserv_calls: int = Field(default=1, alias="CustServ Calls")
    intl_calls: int = Field(default=2, alias="Intl Calls")
    day_calls: int = Field(default=100, alias="Day Calls")
    eve_calls: int = Field(default=100, alias="Eve Calls")
    night_calls: int = Field(default=100, alias="Night Calls")
    total_calls: Optional[int] = Field(default=None, alias="Total Calls")

    class Config:
        populate_by_name = True

def format_prediction_response(customer_dict: dict) -> dict:
    """Helper to generate unified prediction + SHAP response payload"""
    pred_res = predict_customer_upsell(customer_dict, apply_guardrail=True)
    shap_res = explain_customer_prediction(customer_dict)
    
    # Top 3 SHAP drivers
    top_3_drivers = shap_res.get("top_features", [])[:3]
    
    return {
        "phone_number": pred_res["phone_number"],
        "upsell_probability": pred_res["upsell_probability"],
        "raw_model_eligible": pred_res["raw_model_eligible"],
        "guardrail_triggered": pred_res["guardrail_triggered"],
        "guardrail_reason": pred_res["guardrail_reason"],
        "final_upsell_eligible": pred_res["final_upsell_eligible"],
        "campaign_recommendation": pred_res["recommendation"],
        "top_shap_drivers": top_3_drivers,
        "explanation_narrative": shap_res["explanation_narrative"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Telco Upsell API",
        "version": "1.0.0",
        "model_loaded": True
    }

@app.post("/predict")
def predict_upsell_endpoint(payload: Dict[str, Any]):
    """
    Predict upsell readiness for a single customer payload.
    Accepts JSON with keys matching feature names.
    """
    try:
        # Standardize keys (handling spaces/underscores)
        normalized_dict = {}
        key_mapping = {
            "phone_number": "Phone Number",
            "account_length": "Account Length",
            "vmail_message": "VMail Message",
            "custserv_calls": "CustServ Calls",
            "intl_calls": "Intl Calls",
            "day_calls": "Day Calls",
            "eve_calls": "Eve Calls",
            "night_calls": "Night Calls",
            "total_calls": "Total Calls"
        }
        
        for k, v in payload.items():
            norm_k = key_mapping.get(k.lower().replace(" ", "_"), k)
            normalized_dict[norm_k] = v
            
        return format_prediction_response(normalized_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{phone_number}")
def get_customer_prediction(phone_number: str):
    """
    Lookup a customer from dataset by phone number and return prediction + SHAP explanation.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=500, detail="Processed dataset file not found.")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    customer_rows = df[df['Phone Number'] == phone_number]
    
    if customer_rows.empty:
        raise HTTPException(status_code=404, detail=f"Customer phone number '{phone_number}' not found.")
        
    cust_dict = customer_rows.iloc[0].to_dict()
    return format_prediction_response(cust_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
