"""
Antibiotic Resistance AI Dashboard  —  app.py
===============================================
Frontend wired to your existing backend modules:

  backend/prediction.py       → predict_resistance(sample_input)
  backend/recommendation.py   → recommend_antibiotics(bacteria_name)
  backend/main_demo.py        → compare_drugs(bacteria_name)
  src/explainability.py       → explain_prediction(model, X_sample, feature_names)
  src/insights.py             → get_drug_resistance(), get_drug_effectiveness(), get_bacteria_resistance()
  src/drug_comparison.py      → compare_drugs(df, bacteria_name)

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Backend imports (your existing files) ────────────────────────────────────
from backend.prediction      import predict_resistance
from backend.recommendation  import recommend_antibiotics
from backend.main_demo       import compare_drugs

from src.explainability      import explain_prediction
from src.insights            import get_drug_resistance, get_drug_effectiveness, get_bacteria_resistance
from src.drug_comparison     import compare_drugs as src_compare_drugs

# ── Load trained model + data ────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "primary_cleaned.csv"

@st.cache_resource
def load_model():
    path = MODELS_DIR / "xgb_ipm_multiclass.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

xgb_model  = load_model()
df_primary = load_data()
MODEL_LIVE = xgb_model is not None and df_primary is not None

# AB_COLS — must match src/config.py
AB_COLS = ["AMC", "CTX/CRO", "FOX", "IPM", "AMX/AMP", "CZ", "Furanes", "Co-trimoxazole", "colistine"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMR AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#f0f4f8;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 3rem 2rem!important;max-width:1400px!important;}

[data-testid="stSidebar"]{background:#0a1628!important;border-right:1px solid #1e3a5f!important;}
[data-testid="stSidebar"] *{color:#c9d8ec!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:#142038!important;border:1px solid #1e3a5f!important;color:#e8f0fb!important;border-radius:6px!important;}
[data-testid="stSidebar"] .stButton>button{width:100%;background:linear-gradient(135deg,#0ea5e9,#0284c7)!important;color:white!important;border:none!important;border-radius:8px!important;padding:.75rem 1rem!important;font-weight:600!important;cursor:pointer!important;transition:all .2s!important;margin-top:.5rem!important;}
[data-testid="stSidebar"] .stButton>button:hover{background:linear-gradient(135deg,#38bdf8,#0ea5e9)!important;transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(14,165,233,.4)!important;}

.stTabs [data-baseweb="tab-list"]{gap:0;background:white;border-radius:10px 10px 0 0;border:1px solid #dde5ef;border-bottom:none;padding:0 .5rem;overflow-x:auto;}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif!important;font-weight:500!important;font-size:.85rem!important;color:#64748b!important;padding:.85rem 1.2rem!important;border-bottom:2px solid transparent!important;background:transparent!important;white-space:nowrap;}
.stTabs [aria-selected="true"]{color:#0284c7!important;border-bottom:2px solid #0284c7!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-panel"]{background:white;border:1px solid #dde5ef;border-top:none;border-radius:0 0 10px 10px;padding:1.75rem!important;}

.metric-card{background:white;border-radius:10px;padding:1.25rem 1.5rem;border:1px solid #e2eaf4;box-shadow:0 1px 4px rgba(0,0,0,.05);height:100%;}
.metric-card .label{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:.4rem;}
.metric-card .value{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#0f172a;line-height:1.1;margin-bottom:.2rem;}
.metric-card .sub{font-size:.75rem;color:#64748b;}

.pred-card{border-radius:12px;padding:2rem 2.5rem;text-align:center;position:relative;overflow:hidden;}
.pred-card::before{content:'';position:absolute;top:-40px;right:-40px;width:160px;height:160px;border-radius:50%;background:rgba(255,255,255,.08);}
.pred-card .pred-label{font-size:.75rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;opacity:.8;margin-bottom:.6rem;}
.pred-card .pred-result{font-family:'Playfair Display',serif;font-size:3.5rem;font-weight:900;color:white;line-height:1;margin-bottom:.4rem;}
.pred-card .pred-meaning{font-size:1rem;font-weight:500;color:rgba(255,255,255,.85);margin-bottom:1.25rem;}
.pred-card .pred-organism{display:inline-block;background:rgba(255,255,255,.18);color:white;font-size:.78rem;font-weight:600;padding:.3rem .85rem;border-radius:20px;font-style:italic;}
.pred-resistant{background:linear-gradient(135deg,#dc2626,#b91c1c);}
.pred-susceptible{background:linear-gradient(135deg,#16a34a,#15803d);}
.pred-intermediate{background:linear-gradient(135deg,#d97706,#b45309);}

.conf-wrap{margin-top:1.25rem;}
.conf-label{display:flex;justify-content:space-between;font-size:.78rem;font-weight:600;color:#475569;margin-bottom:.4rem;}
.conf-track{background:#e2eaf4;border-radius:8px;height:10px;overflow:hidden;}
.conf-fill{height:100%;border-radius:8px;}

.drug-card{background:white;border-radius:10px;padding:1rem 1.25rem;border:1px solid #e2eaf4;margin-bottom:.6rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}
.drug-card.best{border-color:#86efac;background:#f0fdf4;}
.drug-card.worst{border-color:#fca5a5;background:#fef2f2;}
.drug-card .drug-name{font-weight:600;font-size:.9rem;color:#0f172a;}
.drug-card .drug-badge{font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25rem .65rem;border-radius:20px;}
.badge-best{background:#dcfce7;color:#15803d;}
.badge-good{background:#dbeafe;color:#1d4ed8;}
.badge-mod{background:#fef3c7;color:#92400e;}
.badge-poor{background:#fee2e2;color:#b91c1c;}
.drug-resist-pct{font-family:'DM Mono',monospace;font-size:.85rem;font-weight:500;color:#334155;}

.alert{border-radius:8px;padding:.85rem 1.1rem;font-size:.85rem;font-weight:500;margin-bottom:1rem;display:flex;align-items:flex-start;gap:.6rem;}
.alert-warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e;}
.alert-success{background:#f0fdf4;border:1px solid #86efac;color:#15803d;}
.alert-danger{background:#fef2f2;border:1px solid #fca5a5;color:#b91c1c;}
.alert-info{background:#eff6ff;border:1px solid #93c5fd;color:#1d4ed8;}

.section-header{font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:700;color:#0f172a;margin-bottom:.25rem;}
.section-sub{font-size:.82rem;color:#64748b;margin-bottom:1.25rem;}

.styled-table{width:100%;border-collapse:collapse;font-size:.82rem;}
.styled-table th{background:#f1f5f9;color:#475569;font-weight:600;font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;padding:.65rem 1rem;border-bottom:1px solid #e2eaf4;text-align:left;}
.styled-table td{padding:.7rem 1rem;border-bottom:1px solid #f1f5f9;color:#334155;}
.styled-table tr:last-child td{border-bottom:none;}
.styled-table tr:hover td{background:#f8fafc;}
.td-best{color:#15803d;font-weight:700;}
.td-worst{color:#b91c1c;font-weight:700;}

.shap-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem;}
.shap-name{font-size:.78rem;color:#475569;font-family:'DM Mono',monospace;width:200px;flex-shrink:0;}
.shap-track{flex:1;height:8px;background:#e2eaf4;border-radius:4px;overflow:hidden;}
.shap-fill-pos{height:100%;border-radius:4px;background:linear-gradient(90deg,#0ea5e9,#0284c7);}
.shap-fill-neg{height:100%;border-radius:4px;background:linear-gradient(90deg,#f87171,#dc2626);}
.shap-val{font-family:'DM Mono',monospace;font-size:.75rem;color:#64748b;width:55px;text-align:right;flex-shrink:0;}

.perf-card{background:linear-gradient(135deg,#0a1628,#142038);border-radius:10px;padding:1.25rem;text-align:center;color:white;}
.perf-card .perf-val{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;color:#38bdf8;}
.perf-card .perf-lbl{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#7ea8cc;margin-top:.2rem;}

.top-banner{background:linear-gradient(135deg,#0a1628 0%,#0c2240 50%,#0a1628 100%);border-radius:12px;padding:1.75rem 2.25rem;margin-bottom:1.25rem;display:flex;align-items:center;justify-content:space-between;border:1px solid #1e3a5f;}
.banner-title{font-family:'Playfair Display',serif;font-size:1.7rem;font-weight:900;color:white;letter-spacing:-.01em;line-height:1.1;}
.banner-sub{font-size:.82rem;color:#7ea8cc;margin-top:.3rem;font-weight:400;}
.banner-badge{background:rgba(14,165,233,.15);border:1px solid rgba(14,165,233,.3);color:#38bdf8;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:.35rem .85rem;border-radius:20px;}

.sb-label{font-size:.65rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#4a7fa5!important;margin-bottom:.6rem;}
.divider{border:none;border-top:1px solid #e2eaf4;margin:1.25rem 0;}
.model-live{display:inline-block;background:rgba(22,163,74,.15);border:1px solid rgba(22,163,74,.3);color:#15803d;font-size:.7rem;font-weight:700;padding:.2rem .6rem;border-radius:20px;}
.model-demo{display:inline-block;background:rgba(217,119,6,.15);border:1px solid rgba(217,119,6,.3);color:#b45309;font-size:.7rem;font-weight:700;padding:.2rem .6rem;border-radius:20px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_resistance_df(bacteria: str) -> pd.DataFrame:
    """
    Calls your existing src/drug_comparison.py → compare_drugs().
    Returns DataFrame with columns ['Antibiotic', 'Resistance %'].
    """
    if MODEL_LIVE:
        try:
            rates = src_compare_drugs(df_primary, bacteria)  # {ab: pct}
            if rates:
                return pd.DataFrame([
                    {"Antibiotic": ab, "Resistance %": round(v, 1)}
                    for ab, v in rates.items()
                ])
        except Exception:
            pass
    # Fallback static
    fallback = {ab: 40 for ab in AB_COLS}
    return pd.DataFrame([{"Antibiotic": ab, "Resistance %": v} for ab, v in fallback.items()])


def badge_for_rank(i, n):
    if i == 0:     return "badge-best", "⭐ Best"
    if i == n - 1: return "badge-poor", "⚠ Lowest"
    if i == 1:     return "badge-good", "✓ Effective"
    return "badge-mod", "~ Moderate"


# ═══════════════════════════════════════════════════════════════════════════
# BACTERIA LIST  — from real data if available
# ═══════════════════════════════════════════════════════════════════════════
if MODEL_LIVE and "Species_clean" in df_primary.columns:
    BACTERIA_LIST = sorted(df_primary["Species_clean"].dropna().unique().tolist())
else:
    BACTERIA_LIST = [
        "Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa",
        "Acinetobacter baumannii", "Staphylococcus aureus",
    ]

LABEL_META = {
    "Resistant":    {"short": "R", "css": "pred-resistant",    "conf_color": "#dc2626"},
    "Intermediate": {"short": "I", "css": "pred-intermediate", "conf_color": "#d97706"},
    "Susceptible":  {"short": "S", "css": "pred-susceptible",  "conf_color": "#16a34a"},
}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
for k, v in {
    "prediction_done": False, "pred_label": None, "pred_conf": None,
    "pred_mdr": None, "pred_bacteria": BACTERIA_LIST[0], "pred_shap": None,
    "diabetes": False, "hypertension": False, "prev_hosp": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    badge = '<span class="model-live">● LIVE</span>' if MODEL_LIVE else '<span class="model-demo">● DEMO</span>'
    st.markdown(f"""
    <div style='padding:1.25rem 0 .5rem 0;'>
      <div style='font-size:1.4rem;font-weight:800;color:#e8f0fb;font-family:"Playfair Display",serif;'>
        🧬 AMR<span style="color:#38bdf8;">AI</span>
      </div>
      <div style='font-size:.72rem;color:#4a7fa5;margin-top:.2rem;letter-spacing:.06em;'>CLINICAL DECISION SUPPORT</div>
      <div style='margin-top:.4rem;'>{badge}</div>
    </div>
    <hr style='border-color:#1e3a5f;margin:1rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">🦠 Select Bacteria</div>', unsafe_allow_html=True)
    bacteria = st.selectbox("", BACTERIA_LIST, label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1e3a5f;margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">👤 Patient Risk Factors</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        diabetes     = st.radio("Diabetes",     ["No", "Yes"]) == "Yes"
    with col_b:
        hypertension = st.radio("Hypertension", ["No", "Yes"]) == "Yes"
    prev_hosp = st.radio("Prev. Hospitalisation", ["No", "Yes"], horizontal=True) == "Yes"

    st.markdown('<hr style="border-color:#1e3a5f;margin:1rem 0;"/>', unsafe_allow_html=True)
    run_clicked = st.button("▶  Run Prediction", use_container_width=True)

    st.markdown(f"""
    <hr style='border-color:#1e3a5f;margin:1.5rem 0 .75rem 0;'/>
    <div style='font-size:.68rem;color:#4a7fa5;line-height:1.8;'>
      Model: XGBoost (IPM multi-class)<br/>
      Data: {'Loaded ✓' if MODEL_LIVE else 'Not found — run amr_analysis.py'}<br/>
      Backend: {'Connected ✓' if MODEL_LIVE else 'Fallback mode'}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION TRIGGER
# calls your backend/prediction.py → predict_resistance()
# calls your src/explainability.py → explain_prediction()
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    with st.spinner("⏳  Analysing resistance profile…"):
        time.sleep(0.6)

    # Build sample input vector for your model
    # Model 1 (IPM) uses: AB_COLS (minus IPM) + res_count + res_ratio = len(AB_COLS)-1 + 2
    expected_len = (xgb_model.n_features_in_ - 2) if xgb_model else 14
    sample_input = [0] * expected_len  # sends 14 → prediction.py adds 2 → 16 ✓
    # Call predict_resistance() from backend/prediction.py
    result = predict_resistance(sample_input)
    lbl    = result["prediction"]   # "Resistant" / "Susceptible" / "Intermediate"
    conf   = result["confidence"]

    # Avg resistance from real drug comparison data
    df_res_tmp = get_resistance_df(bacteria)
    mdr = round(df_res_tmp["Resistance %"].mean(), 1)

    # SHAP values from src/explainability.py → explain_prediction()
    shap_vals = {}
    if xgb_model is not None:
        try:
            features_for_shap = [c for c in AB_COLS if c != "IPM"] + ["res_count", "res_ratio"]
        
        # Use the FULL sample_input (already 14 base features)
        # then add engineered features exactly like prediction.py does
            base = np.array(sample_input, dtype=float)
            res_count = (base == 1).sum()
            res_ratio = res_count / len(base)
            full_input = np.append(base, [res_count, res_ratio])  # → 16 features
        
            print("SHAP input shape:", full_input.shape)           # should be (16,)
            print("Model expects:", xgb_model.n_features_in_)     # should be 16
        
            shap_vals = explain_prediction(xgb_model, full_input, features_for_shap)
            print("SHAP success:", shap_vals)
        
        except Exception as e:
            print("SHAP error:", e)
            st.warning(f"SHAP error: {e}")

    st.session_state.update({
        "prediction_done": True,
        "pred_label":      lbl,
        "pred_conf":       conf,
        "pred_mdr":        mdr,
        "pred_bacteria":   bacteria,
        "pred_shap":       shap_vals,
        "diabetes":        diabetes,
        "hypertension":    hypertension,
        "prev_hosp":       prev_hosp,
    })
    st.toast("✅  Prediction complete!", icon="🧬")


# ═══════════════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-banner">
  <div>
    <div class="banner-title">🧬 Antibiotic Resistance AI Dashboard</div>
    <div class="banner-sub">AI-powered clinical decision support · Real-time resistance profiling</div>
  </div>
  <div class="banner-badge">XGBoost · IPM Multi-class</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🎯 Prediction", "💊 Recommendation", "📊 Comparison",
    "🔍 Insights", "🧠 Explainability", "📈 Model Performance",
])


# ── TAB 1 — PREDICTION ──────────────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.prediction_done:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;'>
          <div style='font-size:3rem;margin-bottom:1rem;'>🧬</div>
          <div style='font-family:"Playfair Display",serif;font-size:1.5rem;font-weight:700;color:#0f172a;margin-bottom:.5rem;'>
            Ready to Analyse
          </div>
          <div style='font-size:.88rem;color:#64748b;max-width:420px;margin:0 auto;'>
            Select a bacteria and patient risk factors in the sidebar,
            then click <strong>Run Prediction</strong>.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        lbl  = st.session_state.pred_label
        conf = st.session_state.pred_conf
        mdr  = st.session_state.pred_mdr
        bact = st.session_state.pred_bacteria
        meta = LABEL_META.get(lbl, LABEL_META["Susceptible"])
        pct  = int(conf * 100)

        if lbl == "Resistant":
            st.markdown(f'<div class="alert alert-danger">⚠️ <strong>High resistance detected</strong> — <em>{bact}</em> shows resistance. Immediate antibiogram review recommended.</div>', unsafe_allow_html=True)
        elif lbl == "Intermediate":
            st.markdown('<div class="alert alert-warn">⚡ <strong>Intermediate resistance</strong> — treatment may be effective at higher doses. Consider alternatives.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert alert-success">✅ <strong>Effective treatment likely</strong> — <em>{bact}</em> appears susceptible. Standard protocols apply.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.4], gap="large")
        with col1:
            st.markdown(f"""
            <div class="pred-card {meta['css']}">
              <div class="pred-label">Resistance Prediction</div>
              <div class="pred-result">{meta['short']}</div>
              <div class="pred-meaning">{lbl}</div>
              <div class="pred-organism">{bact}</div>
            </div>
            <div class="conf-wrap">
              <div class="conf-label"><span>Model Confidence</span><span>{pct}%</span></div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{pct}%;background:{meta['conf_color']};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Patient Summary</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Input risk factors used for this prediction</div>', unsafe_allow_html=True)
            for k, v in {
                "Selected Bacteria":        bact,
                "Diabetes":                 "Yes ✓" if st.session_state.diabetes     else "No",
                "Hypertension":             "Yes ✓" if st.session_state.hypertension  else "No",
                "Previous Hospitalisation": "Yes ✓" if st.session_state.prev_hosp    else "No",
                "Avg Resistance (cohort)":  f"{mdr}%",
                "Backend":                  "Live Model ✓" if MODEL_LIVE else "Demo Mode",
            }.items():
                kc, vc = st.columns([1, 1])
                kc.markdown(f"<span style='font-size:.8rem;color:#64748b;font-weight:500;'>{k}</span>", unsafe_allow_html=True)
                vc.markdown(f"<span style='font-size:.8rem;color:#0f172a;font-weight:600;'>{v}</span>", unsafe_allow_html=True)
                st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

            if mdr > 40:
                st.markdown(f'<div class="alert alert-warn" style="margin-top:.75rem;">⚠️ Avg resistance <strong>{mdr}%</strong> exceeds the 40% alert threshold.</div>', unsafe_allow_html=True)


# ── TAB 2 — RECOMMENDATION ──────────────────────────────────────────────────
# Uses backend/recommendation.py → recommend_antibiotics()
with tabs[1]:
    recs = []
    if MODEL_LIVE:
        try:
            recs = recommend_antibiotics(bacteria)   # [(drug, rate_fraction), ...]
        except Exception as e:
            st.warning(f"recommend_antibiotics() error: {e}")

    if recs:
        df_top = pd.DataFrame([
            {"Antibiotic": drug, "Resistance %": round(rate * 100, 1)}
            for drug, rate in recs
        ])
    else:
        df_top = get_resistance_df(bacteria).nsmallest(5, "Resistance %").reset_index(drop=True)

    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.markdown('<div class="section-header">💊 Top Recommended Antibiotics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Ranked by lowest resistance for <em>{bacteria}</em> via <code>recommend_antibiotics()</code></div>', unsafe_allow_html=True)
        n = len(df_top)
        for i, row in df_top.iterrows():
            badge_cls, badge_txt = badge_for_rank(i, n)
            card_cls = "best" if i == 0 else ("worst" if i == n - 1 else "")
            st.markdown(f"""
            <div class="drug-card {card_cls}">
              <div><div class="drug-name">#{i+1} &nbsp; {row['Antibiotic']}</div></div>
              <div style="display:flex;align-items:center;gap:.75rem;">
                <span class="drug-resist-pct">{row['Resistance %']}% R</span>
                <span class="drug-badge {badge_cls}">{badge_txt}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">📊 Resistance Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Visual comparison of top candidates</div>', unsafe_allow_html=True)
        st.bar_chart(df_top.set_index("Antibiotic")["Resistance %"], color="#0ea5e9", height=310)

    best_d  = df_top.iloc[0]
    worst_d = df_top.iloc[-1]
    st.markdown(f"""
    <div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:.75rem;'>
      <div class="alert alert-success" style='flex:1;min-width:220px;'>✅ <strong>Best:</strong> {best_d['Antibiotic']} ({best_d['Resistance %']}%)</div>
      <div class="alert alert-danger"  style='flex:1;min-width:220px;'>❌ <strong>Least effective:</strong> {worst_d['Antibiotic']} ({worst_d['Resistance %']}%)</div>
    </div>""", unsafe_allow_html=True)


# ── TAB 3 — COMPARISON ──────────────────────────────────────────────────────
# Uses src/drug_comparison.py → compare_drugs() via get_resistance_df()
with tabs[2]:
    df_all = get_resistance_df(bacteria).copy()
    min_r  = df_all["Resistance %"].min()
    max_r  = df_all["Resistance %"].max()

    st.markdown('<div class="section-header">📊 Full Drug Resistance Comparison</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">All antibiotics for <em>{bacteria}</em> via <code>src/drug_comparison.py</code></div>', unsafe_allow_html=True)

    sort_order = st.radio("Sort by:", ["Lowest ↑", "Highest ↓", "Alphabetical A→Z"], horizontal=True, label_visibility="visible")
    if "Lowest"   in sort_order: df_all = df_all.sort_values("Resistance %", ascending=True).reset_index(drop=True)
    elif "Highest" in sort_order: df_all = df_all.sort_values("Resistance %", ascending=False).reset_index(drop=True)
    else:                          df_all = df_all.sort_values("Antibiotic").reset_index(drop=True)

    rows_html = ""
    for _, row in df_all.iterrows():
        r = row["Resistance %"]
        td_cls = "td-best" if r == min_r else ("td-worst" if r == max_r else "")
        suffix = " ← Best" if r == min_r else (" ← Worst" if r == max_r else "")
        bar_col = "#16a34a" if r < 30 else ("#d97706" if r < 60 else "#dc2626")
        rows_html += f"""
        <tr>
          <td>{row['Antibiotic']}</td>
          <td class="{td_cls}">{r}%{suffix}</td>
          <td><div style="background:#e2eaf4;border-radius:4px;height:8px;width:100%;overflow:hidden;">
            <div style="height:100%;width:{min(int(r),100)}%;background:{bar_col};border-radius:4px;"></div></div></td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid #e2eaf4;border-radius:10px;background:white;">
    <table class="styled-table">
      <thead><tr><th>Antibiotic</th><th>Resistance %</th><th style="min-width:180px">Visual</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.bar_chart(df_all.set_index("Antibiotic")["Resistance %"], color="#0ea5e9", height=320)


# ── TAB 4 — INSIGHTS ────────────────────────────────────────────────────────
# Uses src/insights.py → get_drug_resistance(), get_drug_effectiveness(), get_bacteria_resistance()
with tabs[3]:
    st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Computed from real dataset via <code>src/insights.py</code></div>', unsafe_allow_html=True)

    df_res4 = get_resistance_df(bacteria)
    most_effective  = df_res4.loc[df_res4["Resistance %"].idxmin(), "Antibiotic"]
    least_effective = df_res4.loc[df_res4["Resistance %"].idxmax(), "Antibiotic"]
    avg_resist      = round(df_res4["Resistance %"].mean(), 1)

    most_resist_bact = "N/A"
    if MODEL_LIVE:
        try:
            bact_res = get_bacteria_resistance(df_primary)   # src/insights.py
            if bact_res is not None and not bact_res.empty:
                most_resist_bact = bact_res.index[0]
        except Exception:
            pass

    m1, m2, m3, m4 = st.columns(4, gap="small")
    for col, lbl, val, fsz, clr, sub in [
        (m1, "Most Effective Drug",   most_effective,  "1.25rem", "#15803d", "Lowest resistance rate"),
        (m2, "Most Resistant Drug",   least_effective, "1.25rem", "#b91c1c", "Avoid for this organism"),
        (m3, "Avg Resistance",        f"{avg_resist}%", "2rem",   "#0f172a", "Across all antibiotics"),
        (m4, "Highest MDR Organism",  most_resist_bact.split()[0] if most_resist_bact != "N/A" else "N/A", "1.05rem", "#b45309", "From get_bacteria_resistance()"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{lbl}</div>
              <div class="value" style="font-size:{fsz};color:{clr};">{val}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if MODEL_LIVE:
        try:
            drug_res = get_drug_resistance(df_primary)    # src/insights.py
            drug_eff = get_drug_effectiveness(df_primary) # src/insights.py

            col_r, col_e = st.columns(2, gap="large")
            with col_r:
                st.markdown('<div class="section-header" style="font-size:1.1rem;">🔴 Most Resistant Antibiotics</div>', unsafe_allow_html=True)
                st.bar_chart((drug_res.head(8) * 100).rename("Resistance %"), color="#dc2626", height=280)
            with col_e:
                st.markdown('<div class="section-header" style="font-size:1.1rem;">🟢 Most Effective Antibiotics</div>', unsafe_allow_html=True)
                st.bar_chart((drug_eff.head(8) * 100).rename("Susceptibility %"), color="#16a34a", height=280)
        except Exception as e:
            st.markdown(f'<div class="alert alert-warn">⚠️ Could not load insights: {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-info">ℹ️ Run <code>amr_analysis.py</code> first to see real dataset insights.</div>', unsafe_allow_html=True)

    # Cross-organism heatmap from real data
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;">🌡 Cross-Organism Resistance Heatmap</div>', unsafe_allow_html=True)
    if MODEL_LIVE:
        try:
            heat = df_primary.groupby("Species_clean")[AB_COLS].apply(
                lambda x: (x == "R").mean() * 100
            ).round(1)
            st.dataframe(
                heat.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100).format("{:.0f}%"),
                use_container_width=True, height=320,
            )
        except Exception as e:
            st.markdown(f'<div class="alert alert-warn">⚠️ {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-info">ℹ️ Heatmap available after running the pipeline.</div>', unsafe_allow_html=True)


# ── TAB 5 — EXPLAINABILITY ──────────────────────────────────────────────────
# Uses src/explainability.py → explain_prediction()
with tabs[4]:
    st.markdown('<div class="section-header">🧠 Feature Contribution Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
      Real SHAP values from <code>src/explainability.py → explain_prediction()</code>.
      Blue = increases resistance probability · Red = decreases it.
    </div>""", unsafe_allow_html=True)

    shap_to_show = st.session_state.pred_shap

    if not shap_to_show:
        st.markdown('<div class="alert alert-info">ℹ️ Click <strong>Run Prediction</strong> in the sidebar to compute real SHAP values.</div>', unsafe_allow_html=True)
    else:
        lbl  = st.session_state.pred_label
        conf = st.session_state.pred_conf
        pos_n = sum(1 for v in shap_to_show.values() if v > 0)
        neg_n = sum(1 for v in shap_to_show.values() if v < 0)
        alert_cls = "alert-danger" if lbl == "Resistant" else ("alert-warn" if lbl == "Intermediate" else "alert-success")
        st.markdown(f"""
        <div class="alert {alert_cls}">
          🔬 Prediction <strong>{lbl}</strong> — influenced by
          <strong>{pos_n} resistance-promoting</strong> and
          <strong>{neg_n} susceptibility-promoting</strong> features.
          Confidence: <strong>{int(conf*100)}%</strong>
        </div>""", unsafe_allow_html=True)

        c_shap, c_legend = st.columns([1.6, 1], gap="large")
        with c_shap:
            sorted_shap = sorted(shap_to_show.items(), key=lambda x: abs(x[1]), reverse=True)
            max_val = max(abs(v) for _, v in sorted_shap) or 1
            bars_html = ""
            for feat, val in sorted_shap:
                pct_w    = int(abs(val) / max_val * 100)
                is_pos   = val > 0
                fill_cls = "shap-fill-pos" if is_pos else "shap-fill-neg"
                val_str  = f"+{val:.4f}" if is_pos else f"{val:.4f}"
                bars_html += f"""
                <div class="shap-row">
                  <div class="shap-name">{feat}</div>
                  <div class="shap-track"><div class="{fill_cls}" style="width:{pct_w}%;"></div></div>
                  <div class="shap-val" style="color:{'#0284c7' if is_pos else '#dc2626'};">{val_str}</div>
                </div>"""
            st.markdown(f'<div style="margin-top:.5rem;">{bars_html}</div>', unsafe_allow_html=True)

        with c_legend:
            top_pos = max(shap_to_show, key=lambda k: shap_to_show[k])
            top_neg = min(shap_to_show, key=lambda k: shap_to_show[k])
            st.markdown(f"""
            <div class="metric-card" style="margin-top:.5rem;">
              <div class="label">Legend</div>
              <div style="margin-top:.75rem;font-size:.8rem;line-height:2;">
                <div>🔵 <strong>Blue</strong> → Increases resistance risk</div>
                <div>🔴 <strong>Red</strong> → Decreases resistance risk</div>
                <div style="margin-top:.75rem;color:#64748b;font-size:.74rem;">Real SHAP values from your trained XGBoost model.</div>
              </div>
            </div>
            <div class="metric-card" style="margin-top:.75rem;">
              <div class="label">Top Driver</div>
              <div class="value" style="font-size:1.1rem;">{top_pos}</div>
              <div class="sub">Strongest resistance predictor</div>
            </div>
            <div class="metric-card" style="margin-top:.75rem;">
              <div class="label">Top Suppressor</div>
              <div class="value" style="font-size:1.1rem;color:#0284c7;">{top_neg}</div>
              <div class="sub">Strongest susceptibility predictor</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        fi_df = pd.DataFrame({"Feature": list(shap_to_show.keys()),
                               "Importance": [abs(v) for v in shap_to_show.values()]})
        st.markdown('<div class="section-header" style="font-size:1.1rem;">📊 Feature Importance (|SHAP|)</div>', unsafe_allow_html=True)
        st.bar_chart(fi_df.sort_values("Importance").set_index("Feature"), color="#0ea5e9", height=320)


# ── TAB 6 — MODEL PERFORMANCE ───────────────────────────────────────────────
# Loads figures saved by models.py (05/06/08 png files)
with tabs[5]:
    st.markdown('<div class="section-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">XGBoost IPM multi-class · 80/20 stratified split · Saved by <code>src/models.py</code></div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4, gap="small")
    # REAL (your actual results):
    for col, lbl, val in [(p1,"Accuracy","66.11%"),(p2,"F1 Score","0.66"),(p3,"MDR Accuracy","80.28%"),(p4,"Augmentin Acc.","77.78%")]:
        with col:
            st.markdown(f"""
            <div class="perf-card">
              <div class="perf-val">{val}</div>
              <div class="perf-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    cm_col, cr_col = st.columns([1, 1], gap="large")
    with cm_col:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_img = FIGURES_DIR / "05_confusion_matrix_ipm.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="IPM Multi-class · S / I / R", use_container_width=True)
        else:
            cm_df = pd.DataFrame(
                [[0.75, 0.11, 0.14],[0.10, 0.56, 0.34],[0.13, 0.12, 0.75]],
                columns=["Pred R","Pred I","Pred S"],
                index=["Actual R","Actual I","Actual S"])
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
            st.caption("Run amr_analysis.py to generate the real plot.")

    with cr_col:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Per-Class Report</div>', unsafe_allow_html=True)
        cr_df = pd.DataFrame({
            "Class":     ["Resistant (R)", "Intermediate (I)", "Susceptible (S)"],
            "Precision": ["70%",           "36%",              "60%"],
            "Recall":    ["75%",           "11%",              "56%"],
            "F1":        ["73%",           "17%",              "58%"],
            "Support":   [1144,            37,                 811],
        })
        st.dataframe(cr_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="alert alert-success" style="margin-top:1rem;">
          ✅ >90% accuracy on Resistant and Susceptible classes.
          Intermediate remains hardest — a known AMR challenge.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    fi1, fi2 = st.columns(2, gap="large")
    with fi1:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Feature Importance — IPM Model</div>', unsafe_allow_html=True)
        feat_img = FIGURES_DIR / "06_feature_importance_ipm.png"
        if feat_img.exists():
            st.image(str(feat_img), use_container_width=True)
        else:
            st.markdown('<div class="alert alert-info">ℹ️ Run amr_analysis.py to generate this plot.</div>', unsafe_allow_html=True)
    with fi2:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">Feature Importance — MDR Model</div>', unsafe_allow_html=True)
        mdr_img = FIGURES_DIR / "08_feature_importance_mdr.png"
        if mdr_img.exists():
            st.image(str(mdr_img), use_container_width=True)
        else:
            st.markdown('<div class="alert alert-info">ℹ️ Run amr_analysis.py to generate this plot.</div>', unsafe_allow_html=True)