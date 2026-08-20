import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Backend API Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "https://telco-upsell-backend.onrender.com")

# Page Configuration
st.set_page_config(
    page_title="Telco Upsell Opportunity Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism & High-Contrast CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Outfit:wght@500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #10B981 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    .status-badge-green {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        margin: 10px 0px;
        letter-spacing: 0.03em;
    }

    .status-badge-red {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        margin: 10px 0px;
        letter-spacing: 0.03em;
    }

    .guardrail-card {
        background: rgba(239, 68, 68, 0.1);
        border-left: 6px solid #EF4444;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0px;
    }

    .guardrail-title {
        color: #EF4444;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .info-callout {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #60A5FA;
        margin-bottom: 18px;
    }

    .metric-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">⚡ Telco Upsell Opportunity Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Enterprise Customer Upgrade Prioritization • SHAP Explainability • Responsible AI Guardrails</div>', unsafe_allow_html=True)

# Helper functions to query backend API
def query_backend_predict(payload):
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
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
        response = requests.get(f"{BACKEND_URL}/customer/{phone_number}", timeout=60)
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

# Sidebar - Preset Scenarios & Navigation
st.sidebar.header("🎯 Preset Scenarios")
st.sidebar.caption("Click to evaluate sample customer profiles:")

preset_selection = None
if st.sidebar.button("🟢 Preset A: Upsell Ready", use_container_width=True):
    preset_selection = "A"
if st.sidebar.button("🟠 Preset B: Borderline", use_container_width=True):
    preset_selection = "B"
if st.sidebar.button("🔴 Preset C: Guardrail Excluded", use_container_width=True):
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

# Input Forms
if input_mode == "Phone Number Lookup":
    st.subheader("🔍 Dataset Customer Lookup")
    phone_input = st.text_input("Enter Customer Phone Number:", value=st.session_state.feature_data.get("Phone Number", "382-4657"))
    if st.button("Evaluate Customer Profile", type="primary", use_container_width=True):
        with st.spinner("Connecting to backend service (Waking up service, this may take up to 50 seconds on first request)..."):
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

    if st.button("Predict Upsell Qualification", type="primary", use_container_width=True):
        with st.spinner("Connecting to backend service (Waking up service, this may take up to 50 seconds on first request)..."):
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
    
    tab_eval, tab_model = st.tabs(["⚡ Customer Evaluation & SHAP Drivers", "📊 Audited Model Intelligence"])
    
    with tab_eval:
        st.header("📊 Qualification & Explainability Results")
        
        # 1. Guardrail Alert (If Triggered)
        if api_response.get("guardrail_triggered"):
            st.markdown(f"""
            <div class="guardrail-card">
                <div class="guardrail-title">🚨 RESPONSIBLE AI GUARDRAIL TRIGGERED</div>
                <p style="margin: 4px 0;"><b>Reason:</b> {api_response.get('guardrail_reason')}</p>
                <p style="margin: 4px 0;"><b>Mandated Action:</b> <span style="color:#EF4444; font-weight:700;">{api_response.get('campaign_recommendation')}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # 2. Main Status Badge & Metrics
        m1, m2, m3 = st.columns([2, 2, 3])
        
        with m1:
            prob_val = api_response.get("upsell_probability", 0.0)
            st.metric("Upsell Opportunity Score", f"{prob_val * 100:.1f}%")
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
                
                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor('#0F172A')
                ax.set_facecolor('#0F172A')
                
                colors = ['#10B981' if x > 0 else '#EF4444' for x in df_shap['shap_value']]
                
                bars = ax.barh(df_shap['feature'], df_shap['shap_value'], color=colors, height=0.55)
                ax.axvline(0, color='#64748B', linestyle='--', linewidth=0.9)
                ax.set_title("Top 3 Feature Contributions (SHAP Impact)", fontsize=11, fontweight='bold', color='#F8FAFC', pad=12)
                ax.set_xlabel("SHAP Value (Log-Odds Contribution)", fontsize=9, color='#94A3B8')
                ax.tick_params(colors='#94A3B8')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#334155')
                ax.spines['left'].set_color('#334155')
                ax.invert_yaxis()
                
                for bar in bars:
                    width = bar.get_width()
                    offset = 0.02 if width >= 0 else -0.02
                    ha = 'left' if width >= 0 else 'right'
                    ax.annotate(f'{width:+.3f}',
                                xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                                xytext=(0, 0), textcoords="offset points",
                                ha=ha, va='center', fontsize=9, fontweight='bold', color='#F8FAFC')
                
                plt.tight_layout()
                st.pyplot(fig)
                
        with c_text:
            st.markdown("**Plain-English Explanation Narrative:**")
            st.success(api_response.get("explanation_narrative", "No narrative generated."))
            
            st.markdown("**Top Feature Contribution Breakdown:**")
            for d in top_drivers:
                impact_icon = "🟢" if d["shap_value"] > 0 else "🔴"
                st.markdown(f"- {impact_icon} **{d['feature']}** = `{d['feature_value']:.0f}` (SHAP Log-Odds: `{d['shap_value']:+.3f}`)")

    with tab_model:
        st.header("📊 Model Performance & Responsible AI Architecture")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Clean Test Accuracy", "91.83%", delta="+4.61% vs Baseline")
        with col_b:
            st.metric("Test ROC-AUC Score", "0.9525", delta="5-Fold Mean: 0.9518")
        with col_c:
            st.metric("Test Recall (Upsell Ready)", "98.96%", delta="Only 5 Misses out of 481")

        st.markdown("""
        ### 🛡️ Production Model & Data Quality Guarantee
        - **Deduplicated Dataset**: Evaluated strictly on **7,467 true unique customers** (100% deduplicated by `Phone Number`).
        - **Leakage-Free Features**: Trained exclusively on non-leaked usage counts & tenure (`CustServ Calls`, `Total Calls`, `Account Length`, `Intl Calls`, `Day Calls`, `Eve Calls`, `Night Calls`, `VMail Message`).
        - **Responsible AI Guardrail Policy**: Customers with `CustServ Calls >= 4` are automatically excluded from sales pitches and routed to **Customer Success / Retention** to prevent pitching churn-risk customers.
        """)

else:
    st.info("👈 Select a preset scenario from the sidebar or enter a phone number / parameters above to view evaluation results.")
