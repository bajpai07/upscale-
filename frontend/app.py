import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Backend API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="Telco Upsell Intelligence Engine",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .status-badge-green {
        background-color: #2e7d32;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin: 10px 0px;
    }
    .status-badge-red {
        background-color: #c62828;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin: 10px 0px;
    }
    .guardrail-card {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 16px;
        border-radius: 6px;
        margin: 15px 0px;
    }
    .info-callout {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #0d47a1;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📶 Telco Upsell Intelligence & Persona Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Upsell Qualification with SHAP Explainability & Responsible AI Guardrails</div>', unsafe_allow_html=True)

# Helper functions to query backend API
def query_backend_predict(payload):
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend service at {BACKEND_URL}. Error: {e}")
        return None

def query_backend_customer(phone_number):
    try:
        response = requests.get(f"{BACKEND_URL}/customer/{phone_number}", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.warning(f"Phone number '{phone_number}' not found in dataset.")
            return None
        else:
            st.error(f"Backend API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend service at {BACKEND_URL}. Error: {e}")
        return None

# Sidebar - Preset Scenarios
st.sidebar.header("🎯 Preset Test Scenarios")
st.sidebar.markdown("Click an example preset button to auto-evaluate representative customer profiles:")

preset_selection = None
if st.sidebar.button("🟢 Preset A: Upsell Ready"):
    preset_selection = "A"
if st.sidebar.button("🟠 Preset B: Borderline"):
    preset_selection = "B"
if st.sidebar.button("🔴 Preset C: Guardrail Excluded"):
    preset_selection = "C"

st.sidebar.markdown("---")
st.sidebar.header("🔍 Input Mode")
input_mode = st.sidebar.radio("Select Input Method:", ["Phone Number Lookup", "Manual Feature Entry"])

# Initialize session state for features
if "feature_data" not in st.session_state:
    st.session_state.feature_data = {
        "Phone Number": "382-4657",
        "Account Length": 200,
        "VMail Message": 25,
        "CustServ Calls": 1,
        "Intl Calls": 4,
        "Day Calls": 100,
        "Eve Calls": 100,
        "Night Calls": 100
    }

# Apply Presets
if preset_selection == "A":
    st.session_state.feature_data = {
        "Phone Number": "799-8985",
        "Account Length": 408,
        "VMail Message": 25,
        "CustServ Calls": 0,
        "Intl Calls": 5,
        "Day Calls": 120,
        "Eve Calls": 150,
        "Night Calls": 444
    }
    input_mode = "Phone Number Lookup"
elif preset_selection == "B":
    st.session_state.feature_data = {
        "Phone Number": "985-9755",
        "Account Length": 20,
        "VMail Message": 0,
        "CustServ Calls": 1,
        "Intl Calls": 2,
        "Day Calls": 98,
        "Eve Calls": 322,
        "Night Calls": 328
    }
    input_mode = "Phone Number Lookup"
elif preset_selection == "C":
    st.session_state.feature_data = {
        "Phone Number": "999-0000",
        "Account Length": 500,
        "VMail Message": 50,
        "CustServ Calls": 5,  # Trigger Guardrail
        "Intl Calls": 10,
        "Day Calls": 300,
        "Eve Calls": 300,
        "Night Calls": 300
    }
    input_mode = "Manual Feature Entry"

api_response = None

if input_mode == "Phone Number Lookup":
    st.subheader("🔍 Dataset Customer Lookup")
    phone_input = st.text_input("Enter Phone Number:", value=st.session_state.feature_data.get("Phone Number", "382-4657"))
    if st.button("Evaluate Customer", type="primary"):
        with st.spinner("Fetching customer evaluation from model backend..."):
            api_response = query_backend_customer(phone_input.strip())

elif input_mode == "Manual Feature Entry":
    st.subheader("🎛️ Manual Feature Input & Scenario Simulator")
    
    st.markdown("""
    <div class="info-callout">
        ℹ️ <b>Billing Cycle Note:</b> All call counts reflect aggregated call volume across a full billing cycle (~1 month), not a single calendar day.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        phone_num = st.text_input("Phone Identifier", value=st.session_state.feature_data.get("Phone Number", "999-0000"))
        account_len = st.slider(
            "Account Tenure (days as a customer)", 
            min_value=1, max_value=1000, 
            value=int(st.session_state.feature_data.get("Account Length", 200)), step=5,
            help="Total days customer has been subscribed (Dataset Median: 202 days)"
        )
        vmail_msg = st.slider(
            "Voicemail Messages (per billing cycle)", 
            min_value=0, max_value=100, 
            value=int(st.session_state.feature_data.get("VMail Message", 25)), step=1,
            help="Voicemails sent in ~1 month"
        )
        custserv_calls = st.slider(
            "Customer Service Calls (Friction)", 
            min_value=0, max_value=10, 
            value=int(st.session_state.feature_data.get("CustServ Calls", 1)), step=1,
            help="Complaints / support calls in ~1 month (Guardrail triggers at >= 4)"
        )
        intl_calls = st.slider(
            "International Calls (per billing cycle)", 
            min_value=0, max_value=30, 
            value=int(st.session_state.feature_data.get("Intl Calls", 4)), step=1,
            help="International calls placed in ~1 month (Dataset Median: 4)"
        )
        
    with c2:
        day_calls = st.slider(
            "Daytime Calls (per billing cycle)", 
            min_value=0, max_value=400, 
            value=int(st.session_state.feature_data.get("Day Calls", 100)), step=5,
            help="Daytime call volume in ~1 month (Dataset Median: 202 calls)"
        )
        eve_calls = st.slider(
            "Evening Calls (per billing cycle)", 
            min_value=0, max_value=400, 
            value=int(st.session_state.feature_data.get("Eve Calls", 100)), step=5,
            help="Evening call volume in ~1 month (Dataset Median: 202 calls)"
        )
        night_calls = st.slider(
            "Night Calls (per billing cycle)", 
            min_value=0, max_value=400, 
            value=int(st.session_state.feature_data.get("Night Calls", 100)), step=5,
            help="Night call volume in ~1 month (Dataset Median: 202 calls)"
        )

    manual_payload = {
        "Phone Number": phone_num,
        "Account Length": account_len,
        "VMail Message": vmail_msg,
        "CustServ Calls": custserv_calls,
        "Intl Calls": intl_calls,
        "Day Calls": day_calls,
        "Eve Calls": eve_calls,
        "Night Calls": night_calls
    }

    if st.button("Predict Upsell Qualification", type="primary"):
        with st.spinner("Executing prediction pipeline & Responsible AI guardrail..."):
            api_response = query_backend_predict(manual_payload)

# If preset was selected, query backend automatically for initial view
if preset_selection and api_response is None:
    if input_mode == "Phone Number Lookup":
        api_response = query_backend_customer(st.session_state.feature_data["Phone Number"])
    else:
        api_response = query_backend_predict(st.session_state.feature_data)

# Display Prediction Output
if api_response:
    st.markdown("---")
    st.header("📊 Qualification & Explainability Results")
    
    # 1. Guardrail Alert (If Triggered)
    if api_response.get("guardrail_triggered"):
        st.markdown(f"""
        <div class="guardrail-card">
            <h3 style="color: #c62828; margin-top:0;">🚨 RESPONSIBLE AI GUARDRAIL TRIGGERED</h3>
            <p><b>Reason:</b> {api_response.get('guardrail_reason')}</p>
            <p><b>Action:</b> {api_response.get('campaign_recommendation')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Main Status Badge & Metrics
    m1, m2, m3 = st.columns([2, 2, 3])
    
    with m1:
        prob_val = api_response.get("upsell_probability", 0.0)
        st.metric("Upsell Probability Score", f"{prob_val * 100:.1f}%")
        st.progress(prob_val)
        
    with m2:
        final_eligible = api_response.get("final_upsell_eligible", False)
        if final_eligible:
            st.markdown('<div class="status-badge-green">✅ ELIGIBLE FOR UPSELL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-red">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
            
    with m3:
        st.markdown(f"**Campaign Recommendation:**")
        st.info(api_response.get("campaign_recommendation", "N/A"))

    # 3. SHAP Explainability & Top 3 Drivers
    st.markdown("### 💡 SHAP Model Feature Drivers")
    
    c_chart, c_text = st.columns([1, 1])
    
    with c_chart:
        top_drivers = api_response.get("top_shap_drivers", [])
        if top_drivers:
            df_shap = pd.DataFrame(top_drivers)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_shap['shap_value']]
            
            bars = ax.barh(df_shap['feature'], df_shap['shap_value'], color=colors, height=0.55)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_title("Top 3 Feature Contributions (SHAP Impact)", fontsize=11, fontweight='bold')
            ax.set_xlabel("SHAP Value (Log-Odds Contribution)", fontsize=9)
            ax.invert_yaxis()
            
            for bar in bars:
                width = bar.get_width()
                offset = 0.02 if width >= 0 else -0.02
                ha = 'left' if width >= 0 else 'right'
                ax.annotate(f'{width:+.3f}',
                            xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha=ha, va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
    with c_text:
        st.markdown("**Plain-English Explanation Narrative:**")
        st.success(api_response.get("explanation_narrative", "No narrative generated."))
        
        st.markdown("**Top Feature Breakdown:**")
        for d in top_drivers:
            impact_icon = "🟢" if d["shap_value"] > 0 else "🔴"
            st.markdown(f"- {impact_icon} **{d['feature']}** = `{d['feature_value']:.0f}` (Impact: `{d['shap_value']:+.3f}`)")

else:
    st.info("👈 Select a preset scenario from the sidebar or enter a phone number / parameters above to view evaluation results.")
