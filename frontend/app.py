import os
import sys

# Ensure project root directory is in sys.path for module resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_dotenv(path):
    """Minimal .env reader, so local runs pick up GEMINI_API_KEY without an extra
    dependency. Uses setdefault: a real environment variable (as set in the Render
    dashboard) always wins over the file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except OSError:
        pass


_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.ai_recommendation import generate_ai_recommendation

BACKEND_URL = os.environ.get("BACKEND_URL", "https://telco-upsell-backend.onrender.com")

st.set_page_config(
    page_title="Upsell IQ",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN TOKENS
# Light console theme. Mirrored in .streamlit/config.toml so native widgets and
# this stylesheet resolve to the same values.
# =============================================================================
TOKENS = {
    "plane": "#F7F8FA",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "foreground": "#111827",
    "muted": "#6B7280",
    "faint": "#9CA3AF",
    # Validated light-mode data colours (dataviz reference, checked on #FFFFFF).
    "primary": "#2563EB",
    "series": "#2A78D6",
    "positive": "#2A78D6",
    "negative": "#E34948",
    "ord_high": "#1C5CAB",
    "ord_mid": "#3987E5",
    "ord_low": "#86B6EF",
    "grid": "#E5E7EB",
}

ROWS_PER_PAGE = 25

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --plane: #F7F8FA;
    --card: #FFFFFF;
    --border: #E5E7EB;
    --border-soft: #F0F1F4;
    --fg: #111827;
    --muted: #6B7280;
    --faint: #9CA3AF;
    --primary: #2563EB;
    --primary-soft: #EFF6FF;
    --primary-line: #BFDBFE;
    --radius: 10px;

    --green-fg: #15803D; --green-bg: #F0FDF4; --green-br: #BBF7D0;
    --teal-fg:  #0F766E; --teal-bg:  #ECFDF5; --teal-br:  #99F6E4;
    --red-fg:   #DC2626; --red-bg:   #FEF2F2; --red-br:   #FECACA;
    --amber-fg: #B45309; --amber-bg: #FFFBEB; --amber-br: #FDE68A;
    --violet-fg:#6D28D9; --violet-bg:#FAF5FF; --violet-br:#E9D5FF;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--plane); }
.stMainBlockContainer { padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1500px; }
[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }

/* ---------- Typography ------------------------------------------------- */
h1,h2,h3,h4,h5 { letter-spacing: -0.02em; font-weight: 600; color: var(--fg); }
.page-title { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.03em; color: var(--fg); margin: 0 0 0.3rem; }
.page-sub   { font-size: 0.9rem; color: var(--muted); margin: 0 0 1.6rem; }
.eyebrow {
    font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--faint); margin: 0 0 0.5rem;
}
.sep { height: 1px; background: var(--border); border: 0; margin: 1.6rem 0; }

/* ---------- Card ------------------------------------------------------- */
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 1.1rem; overflow: hidden;
}
.card-head { padding: 1.05rem 1.25rem; border-bottom: 1px solid var(--border-soft); }
.card-head.plain { border-bottom: 0; padding-bottom: 0.4rem; }
.card-title { font-size: 0.9375rem; font-weight: 600; color: var(--fg); letter-spacing: -0.012em; margin: 0; }
.card-desc  { font-size: 0.8125rem; color: var(--muted); margin: 0.18rem 0 0; line-height: 1.5; }
.card-body  { padding: 1.15rem 1.25rem; }
.card-foot  {
    padding: 0.8rem 1.25rem; border-top: 1px solid var(--border-soft);
    font-size: 0.8125rem; color: var(--muted);
    display: flex; justify-content: space-between; align-items: center;
}

/* Native bordered containers match the HTML cards exactly */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card); border-radius: var(--radius);
}

/* ---------- Stat card -------------------------------------------------- */
.stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.05rem 1.2rem; height: 100%;
}
.stat-label {
    font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted); margin-bottom: 0.55rem;
}
.stat-value {
    font-size: 1.9rem; font-weight: 600; letter-spacing: -0.035em;
    line-height: 1.05; color: var(--fg); font-variant-numeric: tabular-nums;
}
.stat-value.pos { color: var(--teal-fg); }
.stat-value.neg { color: var(--red-fg); }
.stat-hint { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }
.stat-unit { font-size: 0.9rem; font-weight: 500; color: var(--faint); letter-spacing: 0; }

/* ---------- Pill / badge ----------------------------------------------- */
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    border-radius: 6px; padding: 0.12rem 0.5rem;
    font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.03em;
    border: 1px solid var(--border); background: #F9FAFB; color: var(--muted);
    white-space: nowrap; line-height: 1.6;
}
.pill-teal   { background: var(--teal-bg);   border-color: var(--teal-br);   color: var(--teal-fg); }
.pill-green  { background: var(--green-bg);  border-color: var(--green-br);  color: var(--green-fg); }
.pill-red    { background: var(--red-bg);    border-color: var(--red-br);    color: var(--red-fg); }
.pill-amber  { background: var(--amber-bg);  border-color: var(--amber-br);  color: var(--amber-fg); }
.pill-violet { background: var(--violet-bg); border-color: var(--violet-br); color: var(--violet-fg); }
.pill-blue   { background: var(--primary-soft); border-color: var(--primary-line); color: #1D4ED8; }

/* ---------- Segment cards (action matrix) ------------------------------ */
.seg-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem; }
.seg {
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.05rem 1.15rem; background: var(--card);
}
.seg-name  { font-size: 0.9375rem; font-weight: 600; margin: 0 0 0.5rem; }
.seg-value { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.seg-pct   { font-size: 0.75rem; color: var(--muted); margin-top: 0.15rem; }
.seg-desc  { font-size: 0.75rem; color: var(--muted); margin-top: 0.75rem; line-height: 1.5; }
.seg-act   { font-size: 0.8125rem; font-weight: 600; margin-top: 0.35rem; }
.seg-blue   { background: var(--primary-soft); border-color: var(--primary-line); }
.seg-blue .seg-name, .seg-blue .seg-value, .seg-blue .seg-act { color: #1D4ED8; }
.seg-violet { background: var(--violet-bg); border-color: var(--violet-br); }
.seg-violet .seg-name, .seg-violet .seg-value, .seg-violet .seg-act { color: var(--violet-fg); }
.seg-amber  { background: var(--amber-bg); border-color: var(--amber-br); }
.seg-amber .seg-name, .seg-amber .seg-value, .seg-amber .seg-act { color: var(--amber-fg); }

/* ---------- Data table -------------------------------------------------- */
.tbl { width: 100%; }
.tbl-row {
    display: grid;
    grid-template-columns: 1.5fr 1.7fr 1fr 1fr 0.9fr 1.4fr;
    gap: 0.75rem; align-items: center;
    padding: 0.7rem 1.25rem; border-bottom: 1px solid var(--border-soft);
}
.tbl-row:last-child { border-bottom: 0; }
.tbl-row:hover { background: #FAFBFC; }
.tbl-head {
    background: #FCFCFD; border-bottom: 1px solid var(--border);
    font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); padding-top: 0.65rem; padding-bottom: 0.65rem;
}
.tbl-head:hover { background: #FCFCFD; }
.cust-link { font-size: 0.875rem; font-weight: 600; color: var(--primary); text-decoration: none; }
.cust-link:hover { text-decoration: underline; }
.cust-sub  { font-size: 0.75rem; color: var(--faint); margin-top: 0.1rem; font-variant-numeric: tabular-nums; }
.cell { font-size: 0.8125rem; color: var(--fg); }
.cell-muted { font-size: 0.8125rem; color: var(--muted); }
.num { font-variant-numeric: tabular-nums; }

.bar-cell { display: flex; align-items: center; gap: 0.6rem; }
.bar-track { flex: 0 0 78px; height: 6px; background: #EEF0F3; border-radius: 999px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; background: #0F766E; }
.bar-val   { font-size: 0.8125rem; font-weight: 600; color: var(--fg); font-variant-numeric: tabular-nums; }

/* ---------- Contribution bars ------------------------------------------ */
.contrib { margin-bottom: 1.05rem; }
.contrib-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
.contrib-name { font-size: 0.875rem; font-weight: 500; color: var(--fg); }
.contrib-meta { font-size: 0.75rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.contrib-track { height: 7px; background: #EEF0F3; border-radius: 999px; overflow: hidden; }
.contrib-fill  { height: 100%; border-radius: 999px; background: var(--primary); }
.contrib-fill.neg { background: #E34948; }

/* ---------- Key/value grid --------------------------------------------- */
.kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 2rem; }
.kv-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.52rem 0; border-bottom: 1px solid var(--border-soft); font-size: 0.8125rem;
}
.kv-k { color: var(--muted); }
.kv-v { color: var(--fg); font-weight: 600; font-variant-numeric: tabular-nums; }

/* ---------- Note / alert ------------------------------------------------ */
.note {
    display: flex; gap: 0.55rem; align-items: flex-start;
    font-size: 0.75rem; color: var(--muted); line-height: 1.55; margin-top: 0.9rem;
}
.alert { border: 1px solid var(--border); border-radius: var(--radius); padding: 0.85rem 1.05rem; margin-bottom: 1rem; background: var(--card); }
.alert-title { font-size: 0.8125rem; font-weight: 600; margin: 0 0 0.25rem; }
.alert-body  { font-size: 0.8125rem; color: var(--muted); line-height: 1.55; margin: 0; }
.alert-body b { color: var(--fg); font-weight: 600; }
.alert-red    { background: var(--red-bg);    border-color: var(--red-br); }
.alert-red .alert-title { color: var(--red-fg); }
.alert-amber  { background: var(--amber-bg);  border-color: var(--amber-br); }
.alert-amber .alert-title { color: var(--amber-fg); }
.alert-blue   { background: var(--primary-soft); border-color: var(--primary-line); }
.alert-blue .alert-title { color: #1D4ED8; }
.alert-green  { background: var(--green-bg);  border-color: var(--green-br); }
.alert-green .alert-title { color: var(--green-fg); }

.quote {
    background: #F9FAFB; border: 1px solid var(--border-soft);
    border-radius: 8px; padding: 0.9rem 1.05rem;
    font-size: 0.875rem; color: var(--fg); line-height: 1.6;
}

/* ---------- Back link --------------------------------------------------- */
.backlink {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.875rem; font-weight: 500; color: var(--muted);
    text-decoration: none; margin-bottom: 1.1rem;
}
.backlink:hover { color: var(--fg); }

/* ---------- Pager ------------------------------------------------------- */
.pager { display: inline-flex; gap: 0.35rem; }
.pager a, .pager span {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 30px; height: 30px; padding: 0 0.5rem;
    border: 1px solid var(--border); border-radius: 7px;
    font-size: 0.8125rem; text-decoration: none; color: var(--fg); background: var(--card);
}
.pager a:hover { background: #F3F4F6; }
.pager span { color: var(--faint); background: #FAFAFB; }

/* ---------- Sidebar ------------------------------------------------------ */
[data-testid="stSidebar"] { background: var(--card); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
.brand { display: flex; align-items: center; gap: 0.65rem; padding: 0 0.25rem 1.1rem; border-bottom: 1px solid var(--border); margin-bottom: 0.9rem; }
.brand-mark {
    width: 34px; height: 34px; border-radius: 9px; flex: none;
    background: var(--primary); color: #fff;
    display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: 700;
}
.brand-name { font-size: 0.9375rem; font-weight: 700; letter-spacing: -0.015em; line-height: 1.25; color: var(--fg); }
.brand-sub  { font-size: 0.6875rem; color: var(--muted); line-height: 1.3; }

/* Sidebar radio rendered as a nav list */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%; padding: 0.5rem 0.7rem; border-radius: 8px;
    cursor: pointer; transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #F3F4F6; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 0.875rem !important; font-weight: 500 !important; color: #374151 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #EFF4FE; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #1D4ED8 !important; font-weight: 600 !important; }
.side-note { font-size: 0.75rem; color: var(--faint); line-height: 1.5; padding: 0 0.25rem; }
.side-label { font-size: 0.6875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--faint); margin: 1.3rem 0 0.5rem; padding: 0 0.25rem; }

/* ---------- Widgets ------------------------------------------------------ */
.stButton > button { font-size: 0.8125rem; font-weight: 500; border-radius: 8px; box-shadow: none; }
.stButton > button[kind="primary"] { font-weight: 600; }
label, [data-testid="stWidgetLabel"] p {
    font-size: 0.75rem !important; font-weight: 500 !important; color: var(--muted) !important;
}
.stTextInput input, .stNumberInput input { font-size: 0.8125rem !important; }
[data-testid="stCaptionContainer"] p { font-size: 0.75rem !important; color: var(--muted) !important; }
[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); }
[data-testid="stExpander"] summary { font-size: 0.8125rem; font-weight: 500; }
[data-testid="stAlert"] { border-radius: var(--radius); font-size: 0.8125rem; }
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# UI PRIMITIVES
# =============================================================================
def pill(text, tone="default"):
    cls = "pill" if tone == "default" else f"pill pill-{tone}"
    return f'<span class="{cls}">{text}</span>'


def stat_card(label, value, hint="", tone="", unit=""):
    tone_cls = f" {tone}" if tone else ""
    unit_html = f'<span class="stat-unit"> {unit}</span>' if unit else ""
    hint_html = f'<div class="stat-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value{tone_cls}">{value}{unit_html}</div>{hint_html}</div>'
    )


def card_html(title, desc, body, foot=None):
    foot_html = f'<div class="card-foot">{foot}</div>' if foot else ""
    desc_html = f'<p class="card-desc">{desc}</p>' if desc else ""
    return (
        f'<div class="card"><div class="card-head">'
        f'<p class="card-title">{title}</p>{desc_html}</div>'
        f'<div class="card-body">{body}</div>{foot_html}</div>'
    )


def card_header(title, desc=None):
    """Header for a native st.container(border=True), matching .card-head."""
    desc_html = f'<p class="card-desc">{desc}</p>' if desc else ""
    st.markdown(
        f'<p class="card-title">{title}</p>{desc_html}'
        f'<div style="height:1px;background:#F0F1F4;margin:0.85rem -1rem 1rem"></div>',
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div><p class="page-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def alert(title, body, tone="blue"):
    return (
        f'<div class="alert alert-{tone}"><p class="alert-title">{title}</p>'
        f'<p class="alert-body">{body}</p></div>'
    )


def kv_grid(pairs):
    rows = "".join(
        f'<div class="kv-row"><span class="kv-k">{k}</span><span class="kv-v">{v}</span></div>'
        for k, v in pairs
    )
    return f'<div class="kv-grid">{rows}</div>'


def contrib_bar(name, meta, fraction, negative=False):
    pct = max(0.0, min(1.0, abs(fraction))) * 100
    neg = " neg" if negative else ""
    return (
        f'<div class="contrib"><div class="contrib-top">'
        f'<span class="contrib-name">{name}</span><span class="contrib-meta">{meta}</span></div>'
        f'<div class="contrib-track"><div class="contrib-fill{neg}" style="width:{pct:.1f}%"></div></div></div>'
    )


def status_pill(row):
    if row.get("guardrail_triggered"):
        return pill("GUARDRAIL", "red")
    if row.get("final_upsell_eligible"):
        return pill("ELIGIBLE", "green")
    return pill("NOT ELIGIBLE")


def action_pill(row):
    """Derived from the decision itself, not from the recommendation string —
    the backend's wording is long prose ("Route to Customer Success / Retention
    Team (Do NOT Upsell)") and would neither fit a pill nor match reliably."""
    if row.get("guardrail_triggered"):
        return pill("Retain First", "violet")
    if row.get("final_upsell_eligible"):
        return pill("Upsell", "blue")
    return pill("Maintain")


def segment_pill(prob):
    if prob >= 0.66:
        return pill("HIGH", "teal")
    if prob >= 0.33:
        return pill("MEDIUM", "amber")
    return pill("LOW")


# =============================================================================
# BACKEND CLIENT
# =============================================================================
def query_backend_predict(payload):
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"Backend API error ({r.status_code}): {r.text}")
        return None
    except Exception as e:
        st.error(f"Could not reach the backend at {BACKEND_URL} — {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def query_backend_customer(phone_number):
    try:
        r = requests.get(f"{BACKEND_URL}/customer/{phone_number}", timeout=60)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def query_backend_customers_list(search=None, eligible_only=False, guardrail_only=False):
    try:
        params = {"limit": 500}
        if search:
            params["search"] = search
        if eligible_only:
            params["eligible_only"] = True
        if guardrail_only:
            params["guardrail_only"] = True
        r = requests.get(f"{BACKEND_URL}/customers", params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("customers", [])
        return None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def query_backend_portfolio_stats():
    try:
        r = requests.get(f"{BACKEND_URL}/stats/portfolio", timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def check_backend_health():
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=600, show_spinner=False)
def load_processed_df():
    """One row per phone number, matching what the backend serves.

    The processed CSV holds 60,445 rows across 7,467 distinct phone numbers, and
    every feature column varies within a phone-number group — so these are not
    repeated copies of one customer, they are separate records that share a
    number. `keep="last"` is deliberate: the backend builds its lookup with
    `set_index('Phone Number').to_dict('index')`, which keeps the last occurrence,
    and the list must agree with the detail view it links to.
    """
    path = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "processed", "cdr_features.csv"))
    df = pd.read_csv(path)
    return df.drop_duplicates(subset=["Phone Number"], keep="last").reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def load_local_customers(search_query, eligible_only, guardrail_only):
    """Offline fallback: read precomputed scores straight from the processed CSV."""
    try:
        df = load_processed_df()
        if search_query:
            df = df[df["Phone Number"].astype(str).str.contains(search_query, case=False)]
        if eligible_only and "Final_Eligible" in df.columns:
            df = df[df["Final_Eligible"] == True]
        if guardrail_only and "Guardrail_Triggered" in df.columns:
            df = df[df["Guardrail_Triggered"] == True]

        rows = []
        for r in df.to_dict("records"):
            prob = float(r.get("Model_Prob", 0.0))
            g = bool(r.get("CustServ Calls", 0) >= 4)
            rows.append({
                "phone_number": str(r["Phone Number"]),
                "account_length": int(r["Account Length"]),
                "custserv_calls": int(r["CustServ Calls"]),
                "upsell_probability": round(prob, 4),
                "guardrail_triggered": g,
                "final_upsell_eligible": bool(prob >= 0.50 and not g),
            })
        return rows
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def compute_local_stats():
    """Derive real portfolio figures from the processed CSV.

    The deployed backend predates /stats/portfolio, so without this the dashboard
    would fall back to hardcoded placeholder numbers that do not match the data.
    """
    try:
        df = load_processed_df()
        total = len(df)
        if not total:
            return None

        raw = int(df["Raw_Eligible"].sum())
        guard = int((df["Raw_Eligible"] & df["Guardrail_Triggered"]).sum())
        final = int(df["Final_Eligible"].sum())

        bins = [i / 10 for i in range(11)]
        labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(10)]
        binned = pd.cut(df["Model_Prob"], bins=bins, labels=labels,
                        include_lowest=True, right=False)
        dist = {lbl: int(binned.value_counts().get(lbl, 0)) for lbl in labels}

        # Top-decile lift: share of all true positives captured by the highest
        # scoring 10% of the base.
        lift = 0.0
        if "Upsell_Ready_v2" in df.columns:
            positives = int(df["Upsell_Ready_v2"].sum())
            if positives:
                cut = max(1, total // 10)
                top = df.nlargest(cut, "Model_Prob")
                lift = round(int(top["Upsell_Ready_v2"].sum()) / positives * 100, 1)

        return {
            "total_customers": total,
            "raw_eligible_count": raw,
            "raw_eligible_pct": round(raw / total * 100, 2),
            "guardrail_excluded_count": guard,
            "guardrail_excluded_pct": round(guard / raw * 100, 2) if raw else 0.0,
            "final_eligible_count": final,
            "final_eligible_pct": round(final / total * 100, 2),
            "top_decile_tp_capture_pct": lift,
            "score_distribution": dist,
        }
    except Exception:
        return None


def get_customers(search, eligible_only, guardrail_only):
    data = query_backend_customers_list(search, eligible_only, guardrail_only)
    if not data:
        data = load_local_customers(search, eligible_only, guardrail_only)
    return data or []


def get_stats():
    """Backend first, then figures computed locally, then documented placeholders."""
    return query_backend_portfolio_stats() or compute_local_stats() or FALLBACK_STATS


FEATURE_KEYS = ("Account Length", "VMail Message", "CustServ Calls", "Intl Calls",
                "Day Calls", "Eve Calls", "Night Calls")


@st.cache_data(ttl=600, show_spinner=False)
def local_feature_row(phone):
    """Per-customer feature values straight from the processed CSV."""
    try:
        df = load_processed_df()
        hit = df[df["Phone Number"].astype(str) == str(phone)]
        if hit.empty:
            return {}
        row = hit.iloc[0]
        return {k: int(row[k]) for k in FEATURE_KEYS if k in hit.columns}
    except Exception:
        return {}


def resolve_features(response, phone):
    """Assemble a behavioural profile from whatever the response actually carries.

    The deployed backend predates the `features` field, so fall back to the SHAP
    driver values it does return, then to the local CSV.
    """
    features = dict(local_feature_row(phone))
    for driver in response.get("top_shap_drivers", []) or []:
        name, value = driver.get("feature"), driver.get("feature_value")
        if name in FEATURE_KEYS and value is not None:
            features[name] = int(value)
    features.update(response.get("features") or {})
    return features


# =============================================================================
# CHARTS  (light surface, validated palette)
# =============================================================================
def _style(ax, fig, grid_axis="y"):
    fig.patch.set_facecolor(TOKENS["card"])
    ax.set_facecolor(TOKENS["card"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TOKENS["border"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=TOKENS["grid"], linewidth=0.9, linestyle=(0, (3, 3)))
        ax.set_axisbelow(True)


def chart_score_distribution(dist):
    df = pd.DataFrame(list(dist.items()), columns=["bin", "count"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _style(ax, fig)
    ax.bar(df["bin"], df["count"], color=TOKENS["series"], width=0.62)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Customers", fontsize=8, color=TOKENS["muted"], labelpad=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def chart_segments(counts):
    """Ordinal HIGH/MEDIUM/LOW — one hue, dark to light, never a rainbow."""
    labels = ["HIGH", "MEDIUM", "LOW"]
    colors = [TOKENS["ord_high"], TOKENS["ord_mid"], TOKENS["ord_low"]]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    _style(ax, fig)
    bars = ax.bar(labels, [counts.get(k, 0) for k in labels], color=colors, width=0.55)
    for b in bars:
        ax.annotate(f"{int(b.get_height()):,}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold", color=TOKENS["foreground"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def chart_feature_importance(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _style(ax, fig, grid_axis="x")
    d = df.sort_values("Importance")
    ax.barh(d["Feature"], d["Importance"], color=TOKENS["series"], height=0.62)
    ax.tick_params(axis="y", labelsize=8.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def chart_shap(drivers, height=2.9):
    df = pd.DataFrame(drivers)
    fig, ax = plt.subplots(figsize=(6, height))
    _style(ax, fig, grid_axis="x")
    colors = [TOKENS["positive"] if v > 0 else TOKENS["negative"] for v in df["shap_value"]]
    bars = ax.barh(df["feature"], df["shap_value"], color=colors, height=0.5)
    ax.axvline(0, color=TOKENS["border"], linewidth=1)
    ax.invert_yaxis()
    span = max(abs(df["shap_value"].min()), abs(df["shap_value"].max()), 0.01)
    ax.set_xlim(-span * 1.4, span * 1.4)
    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w:+.3f}", xy=(w + (span * 0.05 if w >= 0 else -span * 0.05),
                                     b.get_y() + b.get_height() / 2),
                    ha="left" if w >= 0 else "right", va="center",
                    fontsize=8.5, fontweight="bold", color=TOKENS["foreground"])
    ax.set_xlabel("SHAP value (log-odds)", fontsize=8, color=TOKENS["muted"], labelpad=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# ROUTING  (URL-driven, so a drill-down link survives a page load)
# =============================================================================
VIEWS = {
    "dashboard": "Executive Dashboard",
    "explorer": "Customer Explorer",
    "analytics": "Analytics",
    "simulator": "Scenario Simulator",
}
qp = st.query_params
view = qp.get("view", "dashboard")
if view not in VIEWS:
    view = "dashboard"
selected_customer = qp.get("customer")

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◎</div>'
        '<div><div class="brand-name">Upsell IQ</div>'
        '<div class="brand-sub">Telecom Sales Intelligence</div></div></div>',
        unsafe_allow_html=True,
    )

    keys, labels = list(VIEWS.keys()), list(VIEWS.values())
    choice = st.radio("Navigation", labels, index=keys.index(view), label_visibility="collapsed")
    new_view = keys[labels.index(choice)]
    if new_view != view:
        st.query_params.clear()
        st.query_params["view"] = new_view
        st.rerun()

    st.markdown('<div class="side-label">Service</div>', unsafe_allow_html=True)
    live = check_backend_health()
    st.markdown(
        f'<div style="padding:0 0.25rem">'
        f'{pill("BACKEND ONLINE", "green") if live else pill("BACKEND WAKING", "amber")}</div>',
        unsafe_allow_html=True,
    )
    if not live:
        st.markdown(
            '<p class="side-note" style="margin-top:0.55rem">Free-tier instances sleep when idle; '
            "the first request can take up to 50 seconds.</p>",
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="side-note">Opportunity Score is a behavioural prioritisation score, '
        "not a purchase prediction.</p>",
        unsafe_allow_html=True,
    )

FALLBACK_STATS = {
    "total_customers": 7467,
    "raw_eligible_count": 2361, "raw_eligible_pct": 31.62,
    "guardrail_excluded_count": 1323, "guardrail_excluded_pct": 56.03,
    "final_eligible_count": 1038, "final_eligible_pct": 13.90,
    "top_decile_tp_capture_pct": 84.1,
    "score_distribution": {
        "0-10%": 2500, "10-20%": 1200, "20-30%": 800, "30-40%": 600, "40-50%": 500,
        "50-60%": 400, "60-70%": 450, "70-80%": 400, "80-90%": 350, "90-100%": 267,
    },
}


# =============================================================================
# VIEW — EXECUTIVE DASHBOARD
# =============================================================================
def render_dashboard():
    stats = get_stats()
    total = stats["total_customers"]
    page_header("Executive Dashboard",
                f"Upsell prioritisation across {total:,} verified customers.")

    maintenance = max(total - stats["final_eligible_count"] - stats["guardrail_excluded_count"], 0)
    avg_hint = f"{stats['top_decile_tp_capture_pct']}% of positives in top decile"

    cols = st.columns(4)
    tiles = [
        ("Total customers", f"{total:,}", "Deduplicated records", ""),
        ("Campaign eligible", f"{stats['final_eligible_count']:,}", f"{stats['final_eligible_pct']}% of base", "pos"),
        ("Guardrail excluded", f"{stats['guardrail_excluded_count']:,}", f"{stats['guardrail_excluded_pct']}% of model-eligible", "neg"),
        ("Top-decile lift", f"{stats['top_decile_tp_capture_pct']}%", avg_hint, ""),
    ]
    for col, (label, value, hint, tone) in zip(cols, tiles):
        with col:
            st.markdown(stat_card(label, value, hint, tone), unsafe_allow_html=True)

    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        segs = (
            ("seg-blue", "Priority Upsell", stats["final_eligible_count"],
             "High score · no support friction", "Prioritise for upgrade outreach"),
            ("seg-violet", "Retain First", stats["guardrail_excluded_count"],
             "High score · support friction", "Stabilise the relationship first"),
            ("seg-amber", "Standard Maintenance", maintenance,
             "Below the scoring threshold", "Grow usage before upselling"),
        )
        cards = "".join(
            f'<div class="seg {cls}"><p class="seg-name">{name}</p>'
            f'<div class="seg-value">{count:,}</div>'
            f'<div class="seg-pct">{(count / total * 100):.1f}% of base</div>'
            f'<div class="seg-desc">{desc}</div>'
            f'<div class="seg-act">{act}</div></div>'
            for cls, name, count, desc, act in segs
        )
        st.markdown(
            card_html("Campaign action matrix",
                      "Model score crossed with the responsible-AI guardrail",
                      f'<div class="seg-grid">{cards}</div>',
                      foot="Open Customer Explorer to work these segments."),
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            card_header("Opportunity distribution", "Customers per predicted score band")
            chart_score_distribution(stats.get("score_distribution", {}))

    render_top_table()


def render_top_table():
    rows = sorted(get_customers("", False, False),
                  key=lambda c: c["upsell_probability"], reverse=True)[:8]
    if not rows:
        return
    st.markdown(
        card_html("Top upsell opportunities", "Highest-scoring customers across the base",
                  build_table(rows, compact=True),
                  foot='<span></span><a class="cust-link" href="?view=explorer" target="_self">View all →</a>')
        .replace('<div class="card-body">', '<div class="card-body" style="padding:0">'),
        unsafe_allow_html=True,
    )


# =============================================================================
# TABLE BUILDER
# =============================================================================
def build_table(rows, compact=False):
    heads = ["CUSTOMER", "OPPORTUNITY", "SEGMENT", "STATUS", "SUPPORT", "ACTION"]
    html = ['<div class="tbl">', '<div class="tbl-row tbl-head">']
    html += [f"<div>{h}</div>" for h in heads]
    html.append("</div>")

    for c in rows:
        prob = c["upsell_probability"]
        phone = c["phone_number"]
        html.append('<div class="tbl-row">')
        html.append(
            f'<div><a class="cust-link" href="?view=explorer&customer={phone}" target="_self">{phone}</a>'
            f'<div class="cust-sub">{c["account_length"]}-day tenure</div></div>'
        )
        html.append(
            f'<div class="bar-cell"><div class="bar-track">'
            f'<div class="bar-fill" style="width:{prob * 100:.1f}%"></div></div>'
            f'<span class="bar-val">{prob * 100:.1f}</span></div>'
        )
        html.append(f"<div>{segment_pill(prob)}</div>")
        html.append(f"<div>{status_pill(c)}</div>")
        html.append(f'<div class="cell num">{c["custserv_calls"]}</div>')
        html.append(f'<div>{action_pill(c)}</div>')
        html.append("</div>")
    html.append("</div>")
    return "".join(html)


# =============================================================================
# VIEW — CUSTOMER EXPLORER
# =============================================================================
def render_explorer():
    page_header("Customer Explorer", "Filter the scored base, then open any customer.")

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([3, 1.2, 1.2, 1.2])
        with f1:
            search = st.text_input("Search phone number", value="", placeholder="e.g. 382-4657")
        with f2:
            status = st.selectbox("Status", ["All", "Eligible", "Guardrail excluded"])
        with f3:
            min_score = st.selectbox("Min score", ["Any", "50%", "70%", "85%"])
        with f4:
            sort_by = st.selectbox("Sort by", ["Opportunity score", "Support calls", "Tenure"])

    rows = get_customers(search, status == "Eligible", status == "Guardrail excluded")
    floor = {"Any": 0, "50%": 50, "70%": 70, "85%": 85}[min_score]
    rows = [c for c in rows if c["upsell_probability"] * 100 >= floor]
    key = {"Opportunity score": "upsell_probability", "Support calls": "custserv_calls",
           "Tenure": "account_length"}[sort_by]
    rows.sort(key=lambda c: c[key], reverse=True)

    if not rows:
        st.markdown(alert("No matching customers",
                          "Adjust the search term, status or minimum score.", "blue"),
                    unsafe_allow_html=True)
        return

    try:
        page = max(1, int(qp.get("page", 1)))
    except ValueError:
        page = 1
    pages = max(1, (len(rows) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
    page = min(page, pages)
    start = (page - 1) * ROWS_PER_PAGE
    window = rows[start:start + ROWS_PER_PAGE]

    def pager_link(target, glyph):
        if 1 <= target <= pages:
            return f'<a href="?view=explorer&page={target}" target="_self">{glyph}</a>'
        return f"<span>{glyph}</span>"

    foot = (
        f"<span>Page {page} of {pages} · {len(rows):,} customers</span>"
        f'<span class="pager">{pager_link(page - 1, "‹")}{pager_link(page + 1, "›")}</span>'
    )
    st.markdown(
        card_html("Customers", f"{len(rows):,} match the current filters",
                  build_table(window), foot=foot)
        .replace('<div class="card-body">', '<div class="card-body" style="padding:0">'),
        unsafe_allow_html=True,
    )


# =============================================================================
# VIEW — CUSTOMER DETAIL
# =============================================================================
def render_detail(phone):
    st.markdown('<a class="backlink" href="?view=explorer" target="_self">← Back to Customer Explorer</a>',
                unsafe_allow_html=True)

    with st.spinner(f"Loading {phone}…"):
        d = query_backend_customer(phone)

    if not d:
        st.markdown(alert("Customer unavailable",
                          f"No scored record returned for <b>{phone}</b>. The backend may still be "
                          "waking up — retry in a few seconds.", "amber"), unsafe_allow_html=True)
        return

    prob = d.get("upsell_probability", 0.0)
    guard = d.get("guardrail_triggered", False)
    features = resolve_features(d, phone)
    tenure = features.get("Account Length", "—")
    support = features.get("CustServ Calls", "—")

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(
            f'<div class="page-title" style="margin-bottom:0.15rem">{phone}</div>'
            f'<p class="page-sub" style="margin-bottom:1.2rem">{tenure}-day tenure · '
            f"{support} support calls</p>",
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown(
            f'<div style="text-align:right;padding-top:0.6rem">'
            f'{action_pill(d)}</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_card("Upsell opportunity", f"{prob * 100:.1f}", unit="/ 100",
                              hint=segment_pill(prob), tone="pos" if prob >= 0.5 else ""),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Qualification",
                              "Excluded" if guard else ("Eligible" if d.get("final_upsell_eligible") else "Not eligible"),
                              hint=status_pill(d), tone="neg" if guard else ""),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Support friction", f"{support}",
                              hint="Guardrail threshold is 4", tone="neg" if guard else ""),
                    unsafe_allow_html=True)

    if guard:
        st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
        st.markdown(alert("Responsible AI guardrail triggered",
                          f'<b>Reason:</b> {d.get("guardrail_reason", "N/A")}<br/>'
                          f'<b>Mandated action:</b> {d.get("campaign_recommendation", "N/A")}',
                          "red"), unsafe_allow_html=True)

    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)

    drivers = d.get("top_shap_drivers", [])
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            card_header("Why is this customer prioritised?",
                        "Weighted contribution of each scoring component")
            if drivers:
                span = max(abs(x["shap_value"]) for x in drivers) or 1.0
                bars = "".join(
                    contrib_bar(
                        x["feature"],
                        f'{x["feature_value"]:.0f} · {x["shap_value"]:+.3f} log-odds',
                        x["shap_value"] / span,
                        negative=x["shap_value"] < 0,
                    )
                    for x in drivers
                )
                st.markdown(bars, unsafe_allow_html=True)
            st.markdown(
                '<div class="note">ⓘ SHAP contributions are correlational behavioural signals '
                "used for prioritisation — not a prediction that this customer will purchase.</div>",
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            card_header("Behavioural metrics", "Observed usage from the source CDR data")
            pairs = [
                ("Day calls", features.get("Day Calls", "—")),
                ("Evening calls", features.get("Eve Calls", "—")),
                ("Night calls", features.get("Night Calls", "—")),
                ("International", features.get("Intl Calls", "—")),
                ("Voicemails", features.get("VMail Message", "—")),
                ("Support calls", support),
                ("Account tenure", f"{tenure} days"),
                ("Opportunity score", f"{prob * 100:.1f}"),
            ]
            st.markdown(kv_grid(pairs), unsafe_allow_html=True)

    with st.container(border=True):
        card_header("AI recommendation",
                    "Generated from the computed facts above — never from raw model guesswork")

        store = st.session_state.setdefault("ai_briefings", {})
        if phone not in store:
            with st.spinner("Generating recommendation…"):
                store[phone] = generate_ai_recommendation(d)
        ai = store[phone]

        if ai.get("degraded"):
            st.markdown(alert("AI service unavailable — showing template text",
                              f'The Gemini call did not succeed, so this came from the deterministic '
                              f'fallback.<br/><b>Details:</b> {ai.get("error", "unknown error")}',
                              "amber"), unsafe_allow_html=True)

        st.markdown(
            f'<p class="eyebrow">Recommended action</p>'
            f'<p style="font-size:0.9375rem;font-weight:600;margin:0 0 1.1rem">'
            f'{d.get("campaign_recommendation", "—")}</p>'
            f'<p class="eyebrow">Reason</p>'
            f'<p style="font-size:0.875rem;line-height:1.65;margin:0 0 1.1rem">'
            f'{ai["rep_explanation"]}</p>'
            f'<p class="eyebrow">Suggested outreach message</p>'
            f'<div class="quote">{ai["outreach_opener"]}</div>'
            f'<p style="font-size:0.75rem;color:#9CA3AF;margin:0.85rem 0 0">'
            f'Generated by {ai["source"]}.</p>',
            unsafe_allow_html=True,
        )


# =============================================================================
# VIEW — ANALYTICS
# =============================================================================
def render_analytics():
    stats = get_stats()
    page_header("Analytics",
                f"All figures computed from the {stats['total_customers']:,} verified customer records.")

    dist = stats.get("score_distribution", {})
    high = sum(v for k, v in dist.items() if int(k.split("-")[0]) >= 70)
    med = sum(v for k, v in dist.items() if 30 <= int(k.split("-")[0]) < 70)
    low = sum(v for k, v in dist.items() if int(k.split("-")[0]) < 30)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            card_header("Opportunity segments", "Customers by opportunity band")
            chart_segments({"HIGH": high, "MEDIUM": med, "LOW": low})
    with c2:
        with st.container(border=True):
            card_header("Score distribution", "Customers per predicted score band")
            chart_score_distribution(dist)

    fi = pd.DataFrame([
        {"Feature": "Total Calls", "Importance": 0.6015},
        {"Feature": "Account Length", "Importance": 0.1154},
        {"Feature": "Intl Calls", "Importance": 0.0949},
        {"Feature": "CustServ Calls", "Importance": 0.0892},
        {"Feature": "Day Calls", "Importance": 0.0267},
        {"Feature": "VMail Message", "Importance": 0.0260},
        {"Feature": "Eve Calls", "Importance": 0.0252},
        {"Feature": "Night Calls", "Importance": 0.0212},
    ])

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            card_header("Model feature importance",
                        "XGBoost gain across the 8 leakage-free features")
            chart_feature_importance(fi)
    with c4:
        with st.container(border=True):
            card_header("Segment comparison", "Mean feature values, eligible vs non-eligible")
            st.markdown(kv_grid([
                ("Total calls", "384.2 / 182.6"),
                ("Tenure (days)", "285.4 / 122.1"),
                ("Support calls", "0.82 / 2.85"),
                ("Intl calls", "6.4 / 3.2"),
                ("Voicemails", "28.5 / 8.1"),
                ("", ""),
            ]), unsafe_allow_html=True)
            st.markdown('<div class="note">Values shown as <b>eligible / non-eligible</b> means.</div>',
                        unsafe_allow_html=True)

    st.markdown(alert(
        "How the target label is defined",
        "<code>Upsell_Ready_v2</code> is a documented heuristic proxy: non-churned status, high "
        "account duration, and top-tier spend (Total Mins and Total Charge both at or above the "
        "60th percentile). It measures <b>opportunity readiness</b>, not a guaranteed purchase "
        "probability — the source data contains no historical upsell outcomes to validate against.",
        "blue"), unsafe_allow_html=True)


# =============================================================================
# VIEW — SCENARIO SIMULATOR
# =============================================================================
PRESETS = {
    "Upsell ready": {"Phone Number": "799-8985", "Account Length": 408, "VMail Message": 25,
                     "CustServ Calls": 0, "Intl Calls": 5, "Day Calls": 120, "Eve Calls": 150,
                     "Night Calls": 444},
    "Borderline": {"Phone Number": "985-9755", "Account Length": 20, "VMail Message": 0,
                   "CustServ Calls": 1, "Intl Calls": 2, "Day Calls": 98, "Eve Calls": 322,
                   "Night Calls": 328},
    "Guardrail excluded": {"Phone Number": "999-0000", "Account Length": 500, "VMail Message": 50,
                           "CustServ Calls": 5, "Intl Calls": 10, "Day Calls": 300,
                           "Eve Calls": 300, "Night Calls": 300},
}


def render_simulator():
    page_header("Scenario Simulator",
                "Simulate a customer profile to see how the score and the guardrail respond.")

    if "feature_data" not in st.session_state:
        st.session_state.feature_data = PRESETS["Upsell ready"].copy()

    with st.container(border=True):
        card_header("Scenario presets", "Load a representative profile")
        p = st.columns(3)
        for col, name in zip(p, PRESETS):
            with col:
                if st.button(name, width="stretch"):
                    st.session_state.feature_data = PRESETS[name].copy()

    fd = st.session_state.feature_data

    with st.container(border=True):
        card_header("Customer inputs",
                    "Call counts are aggregated over a full billing cycle (~1 month)")
        a, b = st.columns(2)
        with a:
            phone_num = st.text_input("Phone identifier", value=fd["Phone Number"])
            account_len = st.slider("Account tenure (days)", 1, 1000, int(fd["Account Length"]), 5)
            vmail = st.slider("Voicemail messages", 0, 100, int(fd["VMail Message"]), 1)
            custserv = st.slider("Customer service calls", 0, 10, int(fd["CustServ Calls"]), 1,
                                 help="The guardrail triggers at 4 or more.")
        with b:
            intl = st.slider("International calls", 0, 30, int(fd["Intl Calls"]), 1)
            day = st.slider("Daytime calls", 0, 400, int(fd["Day Calls"]), 5)
            eve = st.slider("Evening calls", 0, 400, int(fd["Eve Calls"]), 5)
            night = st.slider("Night calls", 0, 400, int(fd["Night Calls"]), 5)

    if custserv >= 4:
        st.markdown(alert("Guardrail will trigger",
                          f"{custserv} support calls is at or above the threshold of 4, so this "
                          "profile is excluded from upsell targeting regardless of its score.",
                          "red"), unsafe_allow_html=True)

    payload = {"Phone Number": phone_num, "Account Length": account_len, "VMail Message": vmail,
               "CustServ Calls": custserv, "Intl Calls": intl, "Day Calls": day,
               "Eve Calls": eve, "Night Calls": night}

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Scoring…"):
            st.session_state.sim_result = query_backend_predict(payload)

    resp = st.session_state.get("sim_result")
    if not resp:
        return

    st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
    prob = resp.get("upsell_probability", 0.0)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(stat_card("Opportunity score", f"{prob * 100:.1f}", unit="/ 100",
                              hint=segment_pill(prob), tone="pos" if prob >= 0.5 else ""),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(stat_card("Qualification",
                              "Excluded" if resp.get("guardrail_triggered") else
                              ("Eligible" if resp.get("final_upsell_eligible") else "Not eligible"),
                              hint=status_pill(resp),
                              tone="neg" if resp.get("guardrail_triggered") else ""),
                    unsafe_allow_html=True)
    with r3:
        st.markdown(stat_card("Recommended action", "—",
                              hint=action_pill(resp)),
                    unsafe_allow_html=True)

    if resp.get("guardrail_triggered"):
        st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
        st.markdown(alert("Responsible AI guardrail triggered",
                          f'<b>Reason:</b> {resp.get("guardrail_reason", "N/A")}', "red"),
                    unsafe_allow_html=True)

    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            card_header("Feature drivers", "Top SHAP contributions to this score")
            if resp.get("top_shap_drivers"):
                chart_shap(resp["top_shap_drivers"], height=3.1)
    with d2:
        with st.container(border=True):
            card_header("Explanation", "Plain-English reading of the drivers")
            st.markdown(
                f'<p style="font-size:0.875rem;line-height:1.65;color:#374151">'
                f'{resp.get("explanation_narrative", "No narrative generated.")}</p>',
                unsafe_allow_html=True,
            )


# =============================================================================
# DISPATCH
# =============================================================================
if view == "explorer" and selected_customer:
    render_detail(selected_customer)
elif view == "explorer":
    render_explorer()
elif view == "analytics":
    render_analytics()
elif view == "simulator":
    render_simulator()
else:
    render_dashboard()
