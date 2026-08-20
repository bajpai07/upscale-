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

# Global cached lookup map for memory optimization (loaded once at startup)
_customer_lookup_map: Optional[Dict[str, dict]] = None

def get_customer_lookup_map() -> Dict[str, dict]:
    """Load only 8 required columns into a lightweight global dictionary once at startup"""
    global _customer_lookup_map
    if _customer_lookup_map is None:
        if not os.path.exists(PROCESSED_DATA_PATH):
            _customer_lookup_map = {}
            return _customer_lookup_map
            
        lookup_cols = [
            'Phone Number', 'Account Length', 'VMail Message', 
            'CustServ Calls', 'Intl Calls', 'Day Calls', 'Eve Calls', 'Night Calls'
        ]
        df_lookup = pd.read_csv(PROCESSED_DATA_PATH, usecols=lookup_cols)
        _customer_lookup_map = df_lookup.set_index('Phone Number').to_dict(orient='index')
        del df_lookup  # Free dataframe object memory immediately
    return _customer_lookup_map

@app.on_event("startup")
def startup_event():
    """Pre-warm model pipeline and lookup map at application startup"""
    load_upsell_pipeline()
    get_customer_lookup_map()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Telco Upsell API",
        "version": "1.0.0",
        "model_loaded": True,
        "customers_cached": len(get_customer_lookup_map())
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
    Lookup a customer from memory map by phone number and return prediction + SHAP explanation.
    """
    lookup_map = get_customer_lookup_map()
    if phone_number not in lookup_map:
        raise HTTPException(status_code=404, detail=f"Customer phone number '{phone_number}' not found.")
        
    cust_dict = dict(lookup_map[phone_number])
    cust_dict['Phone Number'] = phone_number
    return format_prediction_response(cust_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
