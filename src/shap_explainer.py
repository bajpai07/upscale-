import os
import joblib
import pandas as pd
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "final_upsell_model.joblib"))

_explainer_cache = None

def load_explainer_pipeline(model_path=DEFAULT_MODEL_PATH):
    """
    Load cached SHAP explainer and model artifact.
    """
    global _explainer_cache
    if _explainer_cache is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _explainer_cache = joblib.load(model_path)
    return _explainer_cache

def explain_customer_prediction(customer_data: dict) -> dict:
    """
    Compute feature-level SHAP explanations and return natural language summary.
    """
    pipeline = load_explainer_pipeline()
    explainer = pipeline['explainer']
    feature_cols = pipeline['feature_cols']
    
    input_dict = dict(customer_data)
    if 'Total Calls' not in input_dict or input_dict['Total Calls'] is None:
        input_dict['Total Calls'] = (
            input_dict.get('Day Calls', 0) + 
            input_dict.get('Eve Calls', 0) + 
            input_dict.get('Night Calls', 0) + 
            input_dict.get('Intl Calls', 0)
        )
    
    df_input = pd.DataFrame([input_dict])[feature_cols]
    
    # Calculate SHAP values
    shap_vals = explainer(df_input)
    values = shap_vals.values[0]
    base_val = float(shap_vals.base_values[0])
    
    contributions = []
    for col, val, shap_imp in zip(feature_cols, df_input.iloc[0].values, values):
        contributions.append({
            "feature": col,
            "feature_value": float(val),
            "shap_value": round(float(shap_imp), 4),
            "impact_direction": "Positive (Increases Upsell Likelihood)" if shap_imp > 0 else "Negative (Decreases Upsell Likelihood)"
        })
    
    # Sort contributions by absolute SHAP impact
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    pos_factors = [c for c in contributions if c['shap_value'] > 0]
    neg_factors = [c for c in contributions if c['shap_value'] < 0]
    
    narrative_parts = []
    if pos_factors:
        top_pos = pos_factors[0]
        narrative_parts.append(f"{top_pos['feature']} ({top_pos['feature_value']:.0f}) strongly boosted upsell readiness (+{top_pos['shap_value']:.3f}).")
    if neg_factors:
        top_neg = neg_factors[0]
        narrative_parts.append(f"{top_neg['feature']} ({top_neg['feature_value']:.0f}) reduced upsell readiness ({top_neg['shap_value']:.3f}).")
        
    narrative = " ".join(narrative_parts) if narrative_parts else "Balanced behavioral features across metrics."
    
    return {
        "base_value": round(base_val, 4),
        "top_features": contributions,
        "primary_positive_driver": pos_factors[0] if pos_factors else None,
        "primary_negative_driver": neg_factors[0] if neg_factors else None,
        "explanation_narrative": narrative
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
    exp = explain_customer_prediction(sample_customer)
    print("SHAP Explanation Test Result:")
    print(exp)
