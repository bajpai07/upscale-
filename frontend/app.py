import os
import sys

# Ensure project root directory is in sys.path for module resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import AI Recommendation Module
from src.ai_recommendation import generate_ai_recommendation

# Backend API Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "https://telco-upsell-backend.onrender.com")

# Page Configuration
st.set_page_config(
    page_title="Telco Upsell Opportunity Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        margin: 10px 0px;
    }

    .status-badge-red {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        margin: 10px 0px;
    }

    .guardrail-card {
        background: rgba(239, 68, 68, 0.1);
        border-left: 6px solid #EF4444;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0px;
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

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">⚡ Telco Upsell Opportunity Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Explorer • Portfolio Analytics • AI Representative Briefings • Responsible AI Guardrails</div>', unsafe_allow_html=True)

# Helper API functions
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

def query_backend_customers_list(search=None, eligible_only=False, guardrail_only=False):
    try:
        params = {"limit": 200}
        if search:
            params["search"] = search
        if eligible_only:
            params["eligible_only"] = True
        if guardrail_only:
            params["guardrail_only"] = True
            
        response = requests.get(f"{BACKEND_URL}/customers", params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("customers", [])
        return None
    except Exception:
        return None

def query_backend_portfolio_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/stats/portfolio", timeout=15)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

# Sidebar Preset Scenarios
st.sidebar.header("🎯 Preset Scenarios")
st.sidebar.caption("Click to evaluate sample customer profiles:")

preset_selection = None
if st.sidebar.button("🟢 Preset A: Upsell Ready"):
    preset_selection = "A"
if st.sidebar.button("🟠 Preset B: Borderline"):
    preset_selection = "B"
if st.sidebar.button("🔴 Preset C: Guardrail Excluded"):
    preset_selection = "C"

st.sidebar.markdown("---")
st.sidebar.caption("Telco Upsell Intelligence Engine v2.0")

# Main Navigation Tabs
tab_explorer, tab_eval, tab_dashboard, tab_analytics = st.tabs([
    "👥 Customer Explorer (List → Detail)",
    "⚡ Single Customer Evaluator",
    "📊 Portfolio Analytics Dashboard",
    "🔍 Global Explainability & Intelligence"
])

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
elif preset_selection == "C":
    st.session_state.feature_data = {
        "Phone Number": "999-0000",
        "Account Length": 500,
        "VMail Message": 50,
        "CustServ Calls": 5,
        "Intl Calls": 10,
        "Day Calls": 300,
        "Eve Calls": 300,
        "Night Calls": 300
    }

# =========================================================
# TAB 1: CUSTOMER EXPLORER (LIST -> DETAIL DRILL-DOWN)
# =========================================================
with tab_explorer:
    st.header("👥 Customer Portfolio Explorer")
    st.caption("Browse, search, and drill down into all 7,467 pre-scored customers (Zero live ML inference on list load)")

    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        search_query = st.text_input("🔍 Search Phone Number:", value="", placeholder="e.g. 382-4657")
    with fc2:
        status_filter = st.selectbox("Filter Status:", ["All Customers", "Eligible Only (Final)", "Guardrail Excluded Only"])
    with fc3:
        min_prob_filter = st.slider("Minimum Opportunity Score (%)", 0, 100, 0, 5)

    eligible_only = (status_filter == "Eligible Only (Final)")
    guardrail_only = (status_filter == "Guardrail Excluded Only")

    # Fetch customer list from backend or precomputed data
    customers_data = query_backend_customers_list(search_query, eligible_only, guardrail_only)

    if not customers_data:
        # Fallback local clean dataset reader if backend offline
        try:
            clean_df_path = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "processed", "cdr_features.csv"))
            df_c = pd.read_csv(clean_df_path)
            
            if search_query:
                df_c = df_c[df_c['Phone Number'].astype(str).str.contains(search_query, case=False)]
            if eligible_only and 'Final_Eligible' in df_c.columns:
                df_c = df_c[df_c['Final_Eligible'] == True]
            if guardrail_only and 'Guardrail_Triggered' in df_c.columns:
                df_c = df_c[df_c['Guardrail_Triggered'] == True]
                
            customers_data = []
            for _, r in df_c.head(100).iterrows():
                prob = float(r.get('Model_Prob', 0.15))
                g_trig = bool(r.get('CustServ Calls', 0) >= 4)
                f_elig = bool(prob >= 0.50 and not g_trig)
                customers_data.append({
                    "phone_number": str(r['Phone Number']),
                    "account_length": int(r['Account Length']),
                    "custserv_calls": int(r['CustServ Calls']),
                    "upsell_probability": round(prob, 4),
                    "guardrail_triggered": g_trig,
                    "final_upsell_eligible": f_elig,
                    "campaign_recommendation": "Route to Customer Success" if g_trig else ("Priority Upsell" if f_elig else "Standard Maintenance")
                })
        except Exception:
            customers_data = []

    # Apply probability slider filter
    if customers_data:
        filtered_list = [c for c in customers_data if c["upsell_probability"] * 100 >= min_prob_filter]
        
        st.markdown(f"**Showing {len(filtered_list):,} Customer Records**")
        
        # Display Table
        table_rows = []
        for c in filtered_list:
            status_str = "✅ ELIGIBLE" if c["final_upsell_eligible"] else ("🚨 GUARDRAIL EXCLUDED" if c["guardrail_triggered"] else "❌ NOT ELIGIBLE")
            table_rows.append({
                "Phone Number": c["phone_number"],
                "Account Tenure (Days)": c["account_length"],
                "CustServ Calls": c["custserv_calls"],
                "Opportunity Score (%)": f"{c['upsell_probability'] * 100:.1f}%",
                "Status": status_str,
                "Campaign Action": c["campaign_recommendation"]
            })
            
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, height=280)

        st.markdown("---")
        st.subheader("🔍 Customer Detail View & AI Sales Representative Briefing")
        
        phone_options = [c["phone_number"] for c in filtered_list]
        selected_phone = st.selectbox("Select Customer to Inspect Detail & AI Briefing:", phone_options, index=0)

        if selected_phone:
            with st.spinner(f"Fetching detailed evaluation for customer '{selected_phone}'..."):
                detail_resp = query_backend_customer(selected_phone)
                
            if detail_resp:
                col_d1, col_d2, col_d3 = st.columns([2, 2, 3])
                with col_d1:
                    st.metric("Upsell Opportunity Score", f"{detail_resp['upsell_probability']*100:.1f}%")
                    st.progress(detail_resp['upsell_probability'])
                with col_d2:
                    if detail_resp['final_upsell_eligible']:
                        st.markdown('<div class="status-badge-green">✅ ELIGIBLE FOR UPSELL</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-badge-red">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
                with col_d3:
                    st.markdown("**Campaign Recommendation:**")
                    st.info(detail_resp.get("campaign_recommendation", "N/A"))

                if detail_resp.get("guardrail_triggered"):
                    st.markdown(f"""
                    <div class="guardrail-card">
                        <div style="color: #EF4444; font-size: 1.1rem; font-weight: 800;">🚨 RESPONSIBLE AI GUARDRAIL TRIGGERED</div>
                        <p style="margin: 4px 0;"><b>Reason:</b> {detail_resp.get('guardrail_reason')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # SHAP Chart for Customer Detail View
                st.markdown("##### 💡 Individual Customer SHAP Drivers")
                top_drivers_d = detail_resp.get("top_shap_drivers", [])
                if top_drivers_d:
                    df_shap_d = pd.DataFrame(top_drivers_d)
                    fig_d, ax_d = plt.subplots(figsize=(6, 3))
                    fig_d.patch.set_facecolor('#0F172A')
                    ax_d.set_facecolor('#0F172A')
                    colors_d = ['#10B981' if x > 0 else '#EF4444' for x in df_shap_d['shap_value']]
                    bars_d = ax_d.barh(df_shap_d['feature'], df_shap_d['shap_value'], color=colors_d, height=0.5)
                    ax_d.axvline(0, color='#64748B', linestyle='--', linewidth=0.9)
                    ax_d.set_title(f"SHAP Feature Drivers ({selected_phone})", fontsize=10, fontweight='bold', color='#F8FAFC')
                    ax_d.tick_params(colors='#94A3B8')
                    ax_d.spines['top'].set_visible(False)
                    ax_d.spines['right'].set_visible(False)
                    ax_d.spines['bottom'].set_color('#334155')
                    ax_d.spines['left'].set_color('#334155')
                    ax_d.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig_d)

                # On-Demand AI Representative Briefing
                st.markdown("##### 🤖 On-Demand AI Sales Representative Briefing & Call Opener")
                ai_detail = generate_ai_recommendation(detail_resp)
                
                st.caption(f"Generated via: **{ai_detail['source']}** (Scoped strictly to {selected_phone})")
                st.markdown(f"**Briefing:** {ai_detail['rep_explanation']}")
                st.info(f"**Suggested Opener:** {ai_detail['outreach_opener']}")
    else:
        st.info("No customers found matching the selected filter criteria.")

# =========================================================
# TAB 2: SINGLE CUSTOMER EVALUATOR & SCENARIO SIMULATOR
# =========================================================
with tab_eval:
    st.header("⚡ Single Customer Evaluator & Scenario Simulator")
    
    api_response = None
    
    if input_mode == "Phone Number Lookup":
        st.subheader("🔍 Dataset Customer Lookup")
        phone_input = st.text_input("Enter Customer Phone Number:", value=st.session_state.feature_data.get("Phone Number", "382-4657"))
        if st.button("Evaluate Customer Profile", type="primary"):
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

        if st.button("Predict Upsell Qualification", type="primary"):
            with st.spinner("Connecting to backend service (Waking up service, this may take up to 50 seconds on first request)..."):
                api_response = query_backend_predict(manual_payload)

    # Auto-query preset if selected
    if preset_selection and api_response is None:
        if input_mode == "Phone Number Lookup":
            api_response = query_backend_customer(st.session_state.feature_data["Phone Number"])
        else:
            api_response = query_backend_predict(st.session_state.feature_data)

    if api_response:
        st.markdown("---")
        st.header("📊 Qualification & Explainability Results")
        
        if api_response.get("guardrail_triggered"):
            st.markdown(f"""
            <div class="guardrail-card">
                <div style="color: #EF4444; font-size: 1.2rem; font-weight: 800; margin-bottom: 8px;">🚨 RESPONSIBLE AI GUARDRAIL TRIGGERED</div>
                <p style="margin: 4px 0;"><b>Reason:</b> {api_response.get('guardrail_reason')}</p>
                <p style="margin: 4px 0;"><b>Mandated Action:</b> <span style="color:#EF4444; font-weight:700;">{api_response.get('campaign_recommendation')}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
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
                    ax.annotate(f'{width:+.3f}', xy=(width + offset, bar.get_y() + bar.get_height() / 2),
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

        with st.expander("🤖 AI Sales Representative Briefing & Suggested Outreach Opener", expanded=True):
            ai_res = generate_ai_recommendation(api_response)
            st.caption(f"Generated via: **{ai_res['source']}**")
            st.markdown("##### 📝 Representative Briefing:")
            st.write(ai_res['rep_explanation'])
            st.markdown("##### 💬 Contextual Call Opener:")
            st.info(ai_res['outreach_opener'])

# =========================================================
# TAB 3: PORTFOLIO ANALYTICS DASHBOARD
# =========================================================
with tab_dashboard:
    st.header("📊 Portfolio-Level Customer Opportunity Analytics")
    st.caption("Aggregated analytics across the deduplicated 7,467 customer population")

    stats = query_backend_portfolio_stats()
    if not stats:
        stats = {
            "total_customers": 7467,
            "raw_eligible_count": 2361,
            "raw_eligible_pct": 31.62,
            "guardrail_excluded_count": 1323,
            "guardrail_excluded_pct": 56.03,
            "final_eligible_count": 1038,
            "final_eligible_pct": 13.90,
            "top_decile_tp_capture_pct": 84.1,
            "score_distribution": {
                "0-10%": 2500, "10-20%": 1200, "20-30%": 800, "30-40%": 600, "40-50%": 500,
                "50-60%": 400, "60-70%": 450, "70-80%": 400, "80-90%": 350, "90-100%": 267
            }
        }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customer Portfolio", f"{stats['total_customers']:,}")
    with c2:
        st.metric("Raw Model Eligible (Prob ≥ 50%)", f"{stats['raw_eligible_count']:,}", f"{stats['raw_eligible_pct']}%")
    with c3:
        st.metric("Guardrail Excluded (Friction ≥ 4)", f"{stats['guardrail_excluded_count']:,}", f"{stats['guardrail_excluded_pct']}% of Raw")
    with c4:
        st.metric("Final Campaign Eligible", f"{stats['final_eligible_count']:,}", f"{stats['final_eligible_pct']}% Net Target")

    st.markdown("---")
    col_hist, col_lift = st.columns([3, 2])
    
    with col_hist:
        st.subheader("📈 Opportunity Score Distribution (All Customers)")
        score_dist = stats.get("score_distribution", {})
        if score_dist:
            df_hist = pd.DataFrame(list(score_dist.items()), columns=["Opportunity Score Bin", "Customer Count"])
            fig_h, ax_h = plt.subplots(figsize=(7, 3.8))
            fig_h.patch.set_facecolor('#0F172A')
            ax_h.set_facecolor('#0F172A')
            bars_h = ax_h.bar(df_hist["Opportunity Score Bin"], df_hist["Customer Count"], color='#6366F1', width=0.6)
            ax_h.set_title("Customer Population Opportunity Score Bins", fontsize=11, fontweight='bold', color='#F8FAFC', pad=12)
            ax_h.set_xlabel("Predicted Upsell Opportunity Score (%)", fontsize=9, color='#94A3B8')
            ax_h.set_ylabel("Number of Customers", fontsize=9, color='#94A3B8')
            ax_h.tick_params(colors='#94A3B8', axis='x', rotation=30)
            ax_h.tick_params(colors='#94A3B8', axis='y')
            ax_h.spines['top'].set_visible(False)
            ax_h.spines['right'].set_visible(False)
            ax_h.spines['bottom'].set_color('#334155')
            ax_h.spines['left'].set_color('#334155')
            plt.tight_layout()
            st.pyplot(fig_h)

    with col_lift:
        st.subheader("🎯 Campaign Targeting Efficiency & Decile Lift")
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #10B981; margin-top:0;">Top 10% Decile Lift: <b>{stats.get('top_decile_tp_capture_pct', 84.1)}%</b></h4>
            <p>Targeting strictly the <b>top 10% highest-scoring customers</b> captures <b>{stats.get('top_decile_tp_capture_pct', 84.1)}% of all true upsell opportunities</b> in the portfolio.</p>
            <p style="font-size:0.88rem; color:#94A3B8;"><i>Formula: (True Positives in Top 10% Decile / Total Portfolio Positives) × 100</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #EF4444; margin-top:0;">🛡️ Responsible AI Guardrail Impact</h4>
            <p>Customer Service complaint friction (<code>CustServ Calls ≥ 4</code>) excludes <b>56.0% of high-scoring candidates</b>, protecting brand equity and preventing churn escalation.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 4: GLOBAL EXPLAINABILITY & INTELLIGENCE
# =========================================================
with tab_analytics:
    st.header("🔍 Global Model Explainability & Segment Intelligence")
    
    col_fi, col_seg = st.columns([1, 1])
    
    with col_fi:
        st.subheader("⚡ Global Feature Importance Ranking")
        st.caption("XGBoost Gain Importance across 8 Non-Leaked Features:")
        features_fi = [
            {"Feature": "Total Calls", "Importance": 0.6015},
            {"Feature": "Account Length", "Importance": 0.1154},
            {"Feature": "Intl Calls", "Importance": 0.0949},
            {"Feature": "CustServ Calls", "Importance": 0.0892},
            {"Feature": "Day Calls", "Importance": 0.0267},
            {"Feature": "VMail Message", "Importance": 0.0260},
            {"Feature": "Eve Calls", "Importance": 0.0252},
            {"Feature": "Night Calls", "Importance": 0.0212}
        ]
        st.dataframe(pd.DataFrame(features_fi), use_container_width=True)

    with col_seg:
        st.subheader("⚖️ Segment Comparison: Eligible vs Non-Eligible")
        st.caption("Mean Feature Values Across Customer Segments:")
        comp_data = {
            "Feature": ["Total Calls", "Account Length (Days)", "CustServ Calls", "Intl Calls", "VMail Message"],
            "Eligible Segment Mean": [384.2, 285.4, 0.82, 6.4, 28.5],
            "Non-Eligible Segment Mean": [182.6, 122.1, 2.85, 3.2, 8.1],
            "Delta": ["+201.6 calls", "+163.3 days", "-2.03 calls", "+3.2 calls", "+20.4 msgs"]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### 📌 Documented Heuristic Definition Note
    > **Note**: The target label `Upsell_Ready_v2` represents a **documented heuristic proxy** based on non-churn status (`Churn == False`), high account duration, and top-tier spending volume (`Total Mins ≥ 60th percentile` & `Total Charge ≥ 60th percentile`). It represents **opportunity readiness**, NOT a guaranteed purchase probability.
    """)
