import os
import joblib
import pandas as pd
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "final_upsell_model.joblib"))

_pipeline_cache = None

def load_upsell_pipeline(model_path=DEFAULT_MODEL_PATH):
    """
    Load and cache the final trained XGBoost model package and guardrail settings.
    """
    global _pipeline_cache
    if _pipeline_cache is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _pipeline_cache = joblib.load(model_path)
    return _pipeline_cache

def predict_customer_upsell(customer_data: dict, apply_guardrail: bool = True) -> dict:
    """
    Predict upsell readiness for a customer dictionary and apply Responsible AI guardrails.
    
    Expected input keys:
      - Account Length (int)
      - VMail Message (int)
      - CustServ Calls (int)
      - Intl Calls (int)
      - Day Calls (int)
      - Eve Calls (int)
      - Night Calls (int)
      - Total Calls (int, optional - calculated if missing)
    """
    pipeline = load_upsell_pipeline()
    model = pipeline['model']
    feature_cols = pipeline['feature_cols']
    
    # Copy dict and ensure required keys exist
    input_dict = dict(customer_data)
    if 'Total Calls' not in input_dict or input_dict['Total Calls'] is None:
        input_dict['Total Calls'] = (
            input_dict.get('Day Calls', 0) + 
            input_dict.get('Eve Calls', 0) + 
            input_dict.get('Night Calls', 0) + 
            input_dict.get('Intl Calls', 0)
        )
    
    # Construct DataFrame in exact feature order
    df_input = pd.DataFrame([input_dict])[feature_cols]
    
    # Predict raw model probability
    raw_prob = float(model.predict_proba(df_input)[0, 1])
    raw_eligible = bool(raw_prob >= 0.50)
    
    # Guardrail evaluation (CustServ Calls >= 4)
    custserv_calls = int(input_dict.get('CustServ Calls', 0))
    guardrail_triggered = bool(custserv_calls >= 4)
    
    if apply_guardrail and guardrail_triggered:
        final_eligible = False
        reason = f"Guardrail Exclusion: Customer Service Calls ({custserv_calls}) >= 4 indicates high complaint friction."
        recommendation = "Route to Customer Success / Retention Team (Do NOT Upsell)"
    elif raw_eligible:
        final_eligible = True
        reason = "Passes model probability threshold (>= 50%) and Responsible AI guardrail."
        recommendation = "Target for High-Value Upsell Campaign"
    else:
        final_eligible = False
        reason = f"Model probability ({raw_prob*100:.1f}%) is below 50% threshold."
        recommendation = "Standard Service Tier Maintenance"

    return {
        "phone_number": input_dict.get("Phone Number", "Unknown"),
        "upsell_probability": round(raw_prob, 4),
        "raw_model_eligible": raw_eligible,
        "guardrail_applied": apply_guardrail,
        "guardrail_triggered": guardrail_triggered,
        "guardrail_reason": reason,
        "final_upsell_eligible": final_eligible,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    # Self-test with mock input
    sample_customer = {
        "Phone Number": "382-4657",
        "Account Length": 128,
        "VMail Message": 25,
        "CustServ Calls": 1,
        "Intl Calls": 3,
        "Day Calls": 110,
        "Eve Calls": 99,
        "Night Calls": 91
    }
    res = predict_customer_upsell(sample_customer)
    print("Inference Test Result:")
    print(res)
