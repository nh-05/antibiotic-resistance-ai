"""
Antibiotic Resistance AI Dashboard
===================================
Frontend-only Streamlit app. All backend calls are stubbed with realistic
mock data so the UI is fully demo-able without a live model.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AMR AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f0f4f8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #c9d8ec !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c9d8ec !important; }

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #142038 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8f0fb !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: white;
    border-radius: 10px 10px 0 0;
    border: 1px solid #dde5ef;
    border-bottom: none;
    padding: 0 0.5rem;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    padding: 0.85rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    white-space: nowrap;
}

.stTabs [aria-selected="true"] {
    color: #0284c7 !important;
    border-bottom: 2px solid #0284c7 !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: white;
    border: 1px solid #dde5ef;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.75rem !important;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e2eaf4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    height: 100%;
}

.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.metric-card .sub {
    font-size: 0.75rem;
    color: #64748b;
}

/* ── Prediction card ── */
.pred-card {
    border-radius: 12px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.pred-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}

.pred-card .pred-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 0.6rem;
}

.pred-card .pred-result {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    color: white;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.pred-card .pred-meaning {
    font-size: 1rem;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1.25rem;
}

.pred-card .pred-organism {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-style: italic;
}

.pred-resistant  { background: linear-gradient(135deg, #dc2626, #b91c1c); }
.pred-susceptible{ background: linear-gradient(135deg, #16a34a, #15803d); }
.pred-intermediate{ background: linear-gradient(135deg, #d97706, #b45309); }

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1.25rem; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; font-weight: 600; color: #475569;
    margin-bottom: 0.4rem;
}
.conf-track {
    background: #e2eaf4;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Drug recommendation cards ── */
.drug-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border: 1px solid #e2eaf4;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.drug-card.best {
    border-color: #86efac;
    background: #f0fdf4;
}

.drug-card.worst {
    border-color: #fca5a5;
    background: #fef2f2;
}

.drug-card .drug-name { font-weight: 600; font-size: 0.9rem; color: #0f172a; }
.drug-card .drug-class { font-size: 0.72rem; color: #64748b; }
.drug-card .drug-badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
}

.badge-best   { background: #dcfce7; color: #15803d; }
.badge-good   { background: #dbeafe; color: #1d4ed8; }
.badge-mod    { background: #fef3c7; color: #92400e; }
.badge-poor   { background: #fee2e2; color: #b91c1c; }

.drug-resist-pct { font-family: 'DM Mono', monospace; font-size: 0.85rem; font-weight: 500; color: #334155; }

/* ── Alerts ── */
.alert {
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}
.alert-warn { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
.alert-success { background: #f0fdf4; border: 1px solid #86efac; color: #15803d; }
.alert-danger { background: #fef2f2; border: 1px solid #fca5a5; color: #b91c1c; }

/* ── Section header ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 1.25rem;
}

/* ── Table style ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.styled-table th {
    background: #f1f5f9;
    color: #475569;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.65rem 1rem;
    border-bottom: 1px solid #e2eaf4;
    text-align: left;
}
.styled-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #f8fafc; }

.td-best { color: #15803d; font-weight: 700; }
.td-worst { color: #b91c1c; font-weight: 700; }

/* ── SHAP bar ── */
.shap-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
.shap-name { font-size: 0.78rem; color: #475569; font-family: 'DM Mono', monospace; width: 180px; flex-shrink: 0; }
.shap-track { flex: 1; height: 8px; background: #e2eaf4; border-radius: 4px; overflow: hidden; }
.shap-fill-pos { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #0ea5e9, #0284c7); }
.shap-fill-neg { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #f87171, #dc2626); }
.shap-val { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #64748b; width: 55px; text-align: right; flex-shrink: 0; }

/* ── Performance cards ── */
.perf-card {
    background: linear-gradient(135deg, #0a1628, #142038);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    color: white;
}
.perf-card .perf-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #38bdf8;
}
.perf-card .perf-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7ea8cc;
    margin-top: 0.2rem;
}

/* ── Top header banner ── */
.top-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0c2240 50%, #0a1628 100%);
    border-radius: 12px;
    padding: 1.75rem 2.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #1e3a5f;
}

.banner-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 900;
    color: white;
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.banner-sub {
    font-size: 0.82rem;
    color: #7ea8cc;
    margin-top: 0.3rem;
    font-weight: 400;
}

.banner-badge {
    background: rgba(14,165,233,0.15);
    border: 1px solid rgba(14,165,233,0.3);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
}

/* ── Sidebar inner ── */
.sb-section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e3a5f;
}
.sb-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7fa5 !important;
    margin-bottom: 0.6rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #e2eaf4;
    margin: 1.25rem 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MOCK DATA  (swap these with real backend calls)
# ═══════════════════════════════════════════════════════════════════════════

BACTERIA_LIST = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",
    "Staphylococcus aureus (MRSA)",
    "Streptococcus pneumoniae",
    "Enterococcus faecalis",
    "Salmonella typhi",
    "Mycobacterium tuberculosis",
    "Neisseria gonorrhoeae",
]

ANTIBIOTICS = [
    "Meropenem", "Amikacin", "Ciprofloxacin", "Ceftriaxone",
    "Piperacillin-Tazobactam", "Ampicillin", "Trimethoprim",
    "Vancomycin", "Linezolid", "Azithromycin",
    "Colistin", "Doxycycline", "Rifampicin", "Cefepime",
]

RESISTANCE_MAP = {
    "Escherichia coli":           [12, 18, 68, 47, 24, 81, 72, 5, 4, 55, 8, 42, 30, 38],
    "Klebsiella pneumoniae":      [28, 22, 74, 61, 38, 90, 78, 6, 5, 60, 14, 50, 35, 55],
    "Pseudomonas aeruginosa":     [20, 15, 52, 80, 30, 95, 88, 3, 2, 70, 10, 65, 45, 28],
    "Acinetobacter baumannii":    [45, 30, 82, 88, 60, 97, 92, 8, 6, 78, 18, 72, 55, 70],
    "Staphylococcus aureus (MRSA)":[8, 70, 40, 75, 55, 85, 60, 2, 1, 35, 50, 20, 10, 65],
    "Streptococcus pneumoniae":   [5, 88, 30, 20, 15, 25, 40, 3, 2, 20, 60, 10, 8, 18],
    "Enterococcus faecalis":      [10, 75, 45, 80, 50, 30, 55, 5, 3, 50, 40, 30, 20, 60],
    "Salmonella typhi":           [18, 40, 55, 30, 20, 60, 50, 12, 8, 40, 25, 35, 22, 32],
    "Mycobacterium tuberculosis": [35, 50, 60, 70, 45, 80, 65, 20, 15, 55, 30, 25, 15, 50],
    "Neisseria gonorrhoeae":      [22, 85, 75, 40, 35, 90, 70, 8, 6, 25, 45, 15, 12, 45],
}

FEATURE_SHAP = {
    "mdr_flag":             +0.421,
    "carbapenem_resist":    +0.318,
    "specimen_blood":       +0.241,
    "age_band_65+":         +0.183,
    "icu_flag":             +0.124,
    "prev_hospitalisation": +0.098,
    "diabetes":             +0.072,
    "hypertension":         +0.051,
    "fluoroquinolone_s":    -0.193,
    "aminoglyco_suscept":   -0.132,
    "penicillin_s":         -0.088,
}

def get_resistance_df(bacteria):
    vals = RESISTANCE_MAP.get(bacteria, [random.randint(10,80) for _ in ANTIBIOTICS])
    return pd.DataFrame({"Antibiotic": ANTIBIOTICS, "Resistance %": vals})

def mock_prediction(bacteria, diabetes, hypertension, prev_hosp):
    """Returns (label, confidence, mdr_rate) — replace with real model call."""
    seed = hash(bacteria + str(diabetes) + str(hypertension) + str(prev_hosp)) % 1000
    rng = random.Random(seed)
    risk = rng.random()
    if hypertension and diabetes and prev_hosp:
        risk = min(risk + 0.35, 0.99)
    if risk > 0.6:
        label, conf = "R", round(rng.uniform(0.72, 0.97), 3)
    elif risk > 0.35:
        label, conf = "I", round(rng.uniform(0.61, 0.78), 3)
    else:
        label, conf = "S", round(rng.uniform(0.68, 0.95), 3)
    mdr = round(rng.uniform(24, 62), 1)
    return label, conf, mdr

def badge_for_rank(i, n=5):
    if i == 0:     return "badge-best",  "⭐ Highly Effective"
    if i == n - 1: return "badge-poor",  "⚠ Least Effective"
    if i <= 1:     return "badge-good",  "✓ Effective"
    return "badge-mod", "~ Moderate"

LABEL_META = {
    "R": {"name": "Resistant",     "css": "pred-resistant",    "conf_color": "#dc2626"},
    "I": {"name": "Intermediate",  "css": "pred-intermediate", "conf_color": "#d97706"},
    "S": {"name": "Susceptible",   "css": "pred-susceptible",  "conf_color": "#16a34a"},
}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_label" not in st.session_state:
    st.session_state.pred_label = None
if "pred_conf" not in st.session_state:
    st.session_state.pred_conf = None
if "pred_mdr" not in st.session_state:
    st.session_state.pred_mdr = None
if "pred_bacteria" not in st.session_state:
    st.session_state.pred_bacteria = BACTERIA_LIST[0]


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.25rem 0 0.5rem 0;'>
      <div style='font-size:1.4rem; font-weight:800; color:#e8f0fb; font-family:"Playfair Display",serif;'>🧬 AMR<span style="color:#38bdf8;">AI</span></div>
      <div style='font-size:0.72rem; color:#4a7fa5; margin-top:0.2rem; letter-spacing:0.06em;'>CLINICAL DECISION SUPPORT</div>
    </div>
    <hr style='border-color:#1e3a5f; margin: 1rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">🦠 Select Bacteria</div>', unsafe_allow_html=True)
    bacteria = st.selectbox("", BACTERIA_LIST, label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1e3a5f; margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">👤 Patient Risk Factors</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=False) == "Yes"
    with col_b:
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=False) == "Yes"

    prev_hosp = st.radio("Previous Hospitalisation", ["No", "Yes"], horizontal=True) == "Yes"

    st.markdown('<hr style="border-color:#1e3a5f; margin:1rem 0;"/>', unsafe_allow_html=True)
    run_clicked = st.button("▶  Run Prediction", use_container_width=True)

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin:1.5rem 0 0.75rem 0;'/>
    <div style='font-size:0.68rem; color:#4a7fa5; line-height:1.6;'>
    Model: XGBoost v2.1<br/>
    Dataset: 12,400 isolates<br/>
    Last updated: 2025-06
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION TRIGGER
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    with st.spinner("⏳  Analysing resistance profile…"):
        time.sleep(1.6)
    label, conf, mdr = mock_prediction(bacteria, diabetes, hypertension, prev_hosp)
    st.session_state.prediction_done = True
    st.session_state.pred_label  = label
    st.session_state.pred_conf   = conf
    st.session_state.pred_mdr    = mdr
    st.session_state.pred_bacteria = bacteria
    st.toast("✅  Prediction complete!", icon="🧬")


# ═══════════════════════════════════════════════════════════════════════════
# TOP BANNER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-banner">
  <div>
    <div class="banner-title">🧬 Antibiotic Resistance AI Dashboard</div>
    <div class="banner-sub">AI-powered clinical decision support system · Real-time resistance profiling</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🎯 Prediction",
    "💊 Recommendation",
    "📊 Comparison",
    "🔍 Insights",
    "🧠 Explainability",
    "📈 Model Performance",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.prediction_done:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem;'>
          <div style='font-size:3rem; margin-bottom:1rem;'>🧬</div>
          <div style='font-family:"Playfair Display",serif; font-size:1.5rem; font-weight:700; color:#0f172a; margin-bottom:0.5rem;'>
            Ready to Analyse
          </div>
          <div style='font-size:0.88rem; color:#64748b; max-width:420px; margin:0 auto;'>
            Select a bacteria and patient risk factors in the sidebar, then click <strong>Run Prediction</strong> to generate a resistance profile.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        lbl   = st.session_state.pred_label
        conf  = st.session_state.pred_conf
        mdr   = st.session_state.pred_mdr
        bact  = st.session_state.pred_bacteria
        meta  = LABEL_META[lbl]
        pct   = int(conf * 100)

        # Alert
        if lbl == "R":
            st.markdown(f"""
            <div class="alert alert-danger">
              ⚠️ <strong>High resistance detected</strong> — <em>{bact}</em> shows resistance to the selected profile.
              Immediate antibiogram review recommended.
            </div>""", unsafe_allow_html=True)
        elif lbl == "I":
            st.markdown(f"""
            <div class="alert alert-warn">
              ⚡ <strong>Intermediate resistance</strong> — treatment may be effective at higher doses or in specific sites.
              Consider alternative agents.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert alert-success">
              ✅ <strong>Effective treatment options available</strong> — <em>{bact}</em> appears susceptible.
              Standard dosing protocols apply.
            </div>""", unsafe_allow_html=True)

        # Main prediction + details
        col1, col2 = st.columns([1, 1.4], gap="large")

        with col1:
            st.markdown(f"""
            <div class="pred-card {meta['css']}">
              <div class="pred-label">Resistance Prediction</div>
              <div class="pred-result">{lbl}</div>
              <div class="pred-meaning">{meta['name']}</div>
              <div class="pred-organism">{bact}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="conf-wrap">
              <div class="conf-label">
                <span>Model Confidence</span>
                <span>{pct}%</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{pct}%; background:{meta['conf_color']};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Patient Summary</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Input risk factors used for this prediction</div>', unsafe_allow_html=True)

            r_vals = {
                "Selected Bacteria":        bact,
                "Diabetes":                 "Yes ✓" if diabetes else "No",
                "Hypertension":             "Yes ✓" if hypertension else "No",
                "Previous Hospitalisation": "Yes ✓" if prev_hosp else "No",
                "MDR Rate (cohort)":        f"{mdr}%",
            }
            for k, v in r_vals.items():
                col_k, col_v = st.columns([1, 1])
                col_k.markdown(f"<span style='font-size:0.8rem;color:#64748b;font-weight:500;'>{k}</span>", unsafe_allow_html=True)
                col_v.markdown(f"<span style='font-size:0.8rem;color:#0f172a;font-weight:600;'>{v}</span>", unsafe_allow_html=True)
                st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

            if mdr > 40:
                st.markdown(f"""
                <div class="alert alert-warn" style="margin-top:0.75rem;">
                  ⚠️ MDR rate of <strong>{mdr}%</strong> in this cohort exceeds the 40% alert threshold.
                </div>""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — RECOMMENDATION
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    df_res = get_resistance_df(bacteria)
    df_top5 = df_res.nsmallest(5, "Resistance %").reset_index(drop=True)

    c1, c2 = st.columns([1.1, 1], gap="large")

    with c1:
        st.markdown('<div class="section-header">💊 Top 5 Recommended Antibiotics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Ranked by lowest resistance rate for <em>{bacteria}</em></div>', unsafe_allow_html=True)

        for i, row in df_top5.iterrows():
            badge_cls, badge_txt = badge_for_rank(i, 5)
            card_cls = "best" if i == 0 else ("worst" if i == 4 else "")
            st.markdown(f"""
            <div class="drug-card {card_cls}">
              <div>
                <div class="drug-name">#{i+1} &nbsp; {row['Antibiotic']}</div>
              </div>
              <div style="display:flex;align-items:center;gap:0.75rem;">
                <span class="drug-resist-pct">{row['Resistance %']}% R</span>
                <span class="drug-badge {badge_cls}">{badge_txt}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">📊 Resistance Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Visual comparison of top candidates</div>', unsafe_allow_html=True)
        st.bar_chart(df_top5.set_index("Antibiotic")["Resistance %"], color="#0ea5e9", height=310)

    # Summary alerts
    best_drug = df_top5.iloc[0]
    worst_drug = df_res.nlargest(1, "Resistance %").iloc[0]

    st.markdown(f"""
    <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.75rem;'>
      <div class="alert alert-success" style='flex:1; min-width:220px;'>
        ✅ <strong>Best antibiotic:</strong> {best_drug['Antibiotic']} ({best_drug['Resistance %']}% resistance)
      </div>
      <div class="alert alert-danger" style='flex:1; min-width:220px;'>
        ❌ <strong>Worst antibiotic:</strong> {worst_drug['Antibiotic']} ({worst_drug['Resistance %']}% resistance) — avoid
      </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — COMPARISON
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    df_all = get_resistance_df(bacteria).copy()
    min_r = df_all["Resistance %"].min()
    max_r = df_all["Resistance %"].max()

    st.markdown('<div class="section-header">📊 Full Drug Resistance Comparison</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">All antibiotics tested against <em>{bacteria}</em> — sorted by resistance rate</div>', unsafe_allow_html=True)

    sort_order = st.radio("Sort by:", ["Lowest resistance first ↑", "Highest resistance first ↓", "Alphabetical A→Z"],
                          horizontal=True, label_visibility="visible")

    if "Lowest" in sort_order:
        df_all = df_all.sort_values("Resistance %", ascending=True).reset_index(drop=True)
    elif "Highest" in sort_order:
        df_all = df_all.sort_values("Resistance %", ascending=False).reset_index(drop=True)
    else:
        df_all = df_all.sort_values("Antibiotic").reset_index(drop=True)

    # Build styled table HTML
    rows_html = ""
    for _, row in df_all.iterrows():
        r = row["Resistance %"]
        td_cls = "td-best" if r == min_r else ("td-worst" if r == max_r else "")
        suffix = " ← Best" if r == min_r else (" ← Worst" if r == max_r else "")
        bar_w = int(r)
        bar_col = "#16a34a" if r < 30 else ("#d97706" if r < 60 else "#dc2626")
        rows_html += f"""
        <tr>
          <td>{row['Antibiotic']}</td>
          <td class="{td_cls}">{r}%{suffix}</td>
          <td>
            <div style="background:#e2eaf4;border-radius:4px;height:8px;width:100%;overflow:hidden;">
              <div style="height:100%;width:{bar_w}%;background:{bar_col};border-radius:4px;"></div>
            </div>
          </td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto; border:1px solid #e2eaf4; border-radius:10px; background:white;">
    <table class="styled-table">
      <thead><tr><th>Antibiotic</th><th>Resistance %</th><th style="min-width:180px">Visual</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;">Bar Chart View</div>', unsafe_allow_html=True)
    st.bar_chart(df_all.set_index("Antibiotic")["Resistance %"], color="#0ea5e9", height=340)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — INSIGHTS
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    df_res4 = get_resistance_df(bacteria)
    most_effective   = df_res4.loc[df_res4["Resistance %"].idxmin(), "Antibiotic"]
    least_effective  = df_res4.loc[df_res4["Resistance %"].idxmax(), "Antibiotic"]
    avg_resist       = round(df_res4["Resistance %"].mean(), 1)

    # Most resistant bacteria globally
    all_avgs = {b: np.mean(v) for b, v in RESISTANCE_MAP.items()}
    most_resist_bact = max(all_avgs, key=all_avgs.get)

    st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Summary metrics for <em>{bacteria}</em> and the full dataset</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Most Effective Drug</div>
          <div class="value" style="font-size:1.25rem;color:#15803d;">{most_effective}</div>
          <div class="sub">Lowest resistance rate</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Most Resistant Drug</div>
          <div class="value" style="font-size:1.25rem;color:#b91c1c;">{least_effective}</div>
          <div class="sub">Avoid for this organism</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Avg Resistance</div>
          <div class="value">{avg_resist}%</div>
          <div class="sub">Across all antibiotics</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Highest MDR Organism</div>
          <div class="value" style="font-size:1.05rem;color:#b45309;">{most_resist_bact.split()[0]}</div>
          <div class="sub">{round(all_avgs[most_resist_bact],1)}% avg resistance</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Cross-organism heatmap data
    st.markdown('<div class="section-header" style="font-size:1.1rem; margin-bottom:0.25rem;">🌡 Cross-Organism Resistance Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Resistance % across all organisms and antibiotics</div>', unsafe_allow_html=True)

    heat_df = pd.DataFrame(RESISTANCE_MAP, index=ANTIBIOTICS).T
    st.dataframe(
        heat_df.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100)
                     .format("{:.0f}%"),
        use_container_width=True, height=340
    )

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # SIR distribution mock
    st.markdown('<div class="section-header" style="font-size:1.1rem;">📊 SIR Distribution</div>', unsafe_allow_html=True)
    sir_data = pd.DataFrame({"Category": ["Susceptible (S)", "Intermediate (I)", "Resistant (R)"],
                              "Count": [421, 118, 289]})
    c_pie, c_txt = st.columns([1, 1], gap="large")
    with c_pie:
        st.bar_chart(sir_data.set_index("Category"), color="#0ea5e9", height=260)
    with c_txt:
        total = 828
        st.markdown(f"""
        <div style='margin-top:1rem;'>
          <div class='alert alert-success'>✅ Susceptible: <strong>421 isolates (50.8%)</strong></div>
          <div class='alert alert-warn'>⚡ Intermediate: <strong>118 isolates (14.3%)</strong></div>
          <div class='alert alert-danger'>⚠️ Resistant: <strong>289 isolates (34.9%)</strong></div>
          <div style='font-size:0.78rem;color:#64748b;margin-top:0.75rem;'>
            Based on {total} total isolates in the cohort. MDR threshold (≥3 classes): 34.9% of cases exceed criteria.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — EXPLAINABILITY
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">🧠 Feature Contribution Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
      SHAP-style feature attributions for the most recent prediction.
      Positive values push toward <strong>Resistant</strong>; negative values push toward <strong>Susceptible</strong>.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.prediction_done:
        lbl   = st.session_state.pred_label
        conf  = st.session_state.pred_conf
        pos_n = sum(1 for v in FEATURE_SHAP.values() if v > 0)
        neg_n = sum(1 for v in FEATURE_SHAP.values() if v < 0)
        st.markdown(f"""
        <div class="alert alert-{'danger' if lbl=='R' else 'warn' if lbl=='I' else 'success'}">
          🔬 Prediction <strong>{lbl} ({LABEL_META[lbl]['name']})</strong> was influenced by
          <strong>{pos_n} resistance-promoting</strong> and <strong>{neg_n} susceptibility-promoting</strong> features.
          Model confidence: <strong>{int(conf*100)}%</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert alert-warn">
          ℹ️ Run a prediction first to see feature-level explanations.
        </div>""", unsafe_allow_html=True)

    c_shap, c_legend = st.columns([1.6, 1], gap="large")

    with c_shap:
        sorted_shap = sorted(FEATURE_SHAP.items(), key=lambda x: abs(x[1]), reverse=True)
        max_val = max(abs(v) for _, v in sorted_shap)
        bars_html = ""
        for feat, val in sorted_shap:
            pct_w = int(abs(val) / max_val * 100)
            is_pos = val > 0
            fill_cls = "shap-fill-pos" if is_pos else "shap-fill-neg"
            val_str = f"+{val:.3f}" if is_pos else f"{val:.3f}"
            bars_html += f"""
            <div class="shap-row">
              <div class="shap-name">{feat}</div>
              <div class="shap-track"><div class="{fill_cls}" style="width:{pct_w}%;"></div></div>
              <div class="shap-val" style="color:{'#0284c7' if is_pos else '#dc2626'};">{val_str}</div>
            </div>"""
        st.markdown(f'<div style="margin-top:0.5rem;">{bars_html}</div>', unsafe_allow_html=True)

    with c_legend:
        st.markdown("""
        <div class="metric-card" style="margin-top:0.5rem;">
          <div class="label">Legend</div>
          <div style="margin-top:0.75rem; font-size:0.8rem; line-height:2;">
            <div>🔵 <strong>Blue bars</strong> → Increase resistance risk</div>
            <div>🔴 <strong>Red bars</strong> → Decrease resistance risk</div>
            <div style="margin-top:0.75rem; color:#64748b; font-size:0.74rem;">
              Bar length = relative contribution magnitude.<br/>
              Values are normalised SHAP scores.
            </div>
          </div>
        </div>

        <div class="metric-card" style="margin-top:0.75rem;">
          <div class="label">Top Driver</div>
          <div class="value" style="font-size:1.2rem;">mdr_flag</div>
          <div class="sub">Strongest positive predictor</div>
        </div>

        <div class="metric-card" style="margin-top:0.75rem;">
          <div class="label">Top Suppressor</div>
          <div class="value" style="font-size:1.2rem; color:#0284c7;">fluoroquinolone_s</div>
          <div class="sub">Strongest negative predictor</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;">📊 Overall Feature Importance (Model-Level)</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({"Feature": list(FEATURE_SHAP.keys()),
                           "Importance": [abs(v) for v in FEATURE_SHAP.values()]})
    fi_df = fi_df.sort_values("Importance", ascending=True)
    st.bar_chart(fi_df.set_index("Feature"), color="#0ea5e9", height=340)


# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — MODEL PERFORMANCE
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Evaluation results on held-out test set (20% stratified split, n=2,480 isolates)</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4, gap="small")
    for col, label, val in [
        (p1, "Accuracy",  "91.4%"),
        (p2, "F1 Score",  "0.893"),
        (p3, "ROC-AUC",   "0.961"),
        (p4, "Precision", "90.2%"),
    ]:
        with col:
            st.markdown(f"""
            <div class="perf-card">
              <div class="perf-val">{val}</div>
              <div class="perf-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    cm_col, cl_col = st.columns([1, 1], gap="large")

    with cm_col:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = pd.DataFrame(
            {"Predicted R": [689, 42, 18], "Predicted I": [31, 198, 22], "Predicted S": [19, 28, 1433]},
            index=["Actual R", "Actual I", "Actual S"]
        )
        st.dataframe(
            cm_data.style.background_gradient(cmap="Blues"),
            use_container_width=True
        )
        st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")

    with cl_col:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Per-Class Report</div>', unsafe_allow_html=True)
        cr_df = pd.DataFrame({
            "Class": ["Resistant (R)", "Intermediate (I)", "Susceptible (S)"],
            "Precision": ["93.2%", "81.4%", "97.3%"],
            "Recall":    ["91.8%", "74.6%", "96.2%"],
            "F1":        ["92.5%", "77.8%", "96.7%"],
            "Support":   [739, 268, 1473],
        })
        st.dataframe(cr_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="alert alert-success" style="margin-top:1rem;">
          ✅ Model achieves clinical-grade accuracy (>90%) on both Resistant and Susceptible classes.
          Intermediate class remains hardest to classify — a known challenge in AMR literature.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;">Training History</div>', unsafe_allow_html=True)
    epochs = list(range(1, 21))
    train_acc = [0.61 + 0.015*i + random.uniform(-0.005,0.005) for i in epochs]
    val_acc   = [0.58 + 0.014*i + random.uniform(-0.008,0.008) for i in epochs]
    hist_df = pd.DataFrame({"Train Accuracy": train_acc, "Validation Accuracy": val_acc}, index=epochs)
    st.line_chart(hist_df, color=["#0ea5e9", "#f59e0b"], height=260)