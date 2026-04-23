"""
MLT Career Prep · Job Fit Scorer
=================================
Your structure (Excel upload, session persistence, manual entry, .pkl files)
+ Advanced features (gauge, fairness, contribution charts, company explorer)
+ Official MLT brand theme (Navy #1B2A4A, Gold #C9A84C, White)

Model  : LASSO Logistic Regression — 13 features, AUC 0.903
Author : MLT Analytics Team
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    f1_score, accuracy_score, confusion_matrix,
)

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MLT Career Prep · Job Fit Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# MLT BRAND THEME — Navy #1B2A4A · Gold #C9A84C · White #FFFFFF
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px;
}

/* ── HERO BANNER ── */
.mlt-hero {
    background: linear-gradient(135deg, #1B2A4A 0%, #243659 50%, #1B2A4A 100%);
    border-left: 6px solid #C9A84C;
    padding: 1.6rem 2rem; border-radius: 12px;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px rgba(27,42,74,0.3);
}
.mlt-hero h1 {
    color: #FFFFFF; font-size: 1.75rem; font-weight: 800;
    margin: 0 0 0.2rem 0; letter-spacing: -0.3px;
}
.mlt-hero .subtitle { color: #C9A84C; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.mlt-hero p { color: #94a3b8; margin: 0; font-size: 0.85rem; }
.mlt-badge {
    display: inline-block; background: rgba(201,168,76,0.15);
    border: 1px solid #C9A84C; color: #C9A84C;
    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; margin-right: 6px; margin-top: 6px;
}

/* ── KPI CARDS ── */
.kpi-card {
    background: #FFFFFF; border-radius: 10px;
    padding: 1rem 1.1rem; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 4px solid #1B2A4A;
}
.kpi-card.gold  { border-top-color: #C9A84C; }
.kpi-card.green { border-top-color: #059669; }
.kpi-card.red   { border-top-color: #DC2626; }
.kpi-card.amber { border-top-color: #F59E0B; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: #1B2A4A; line-height: 1; }
.kpi-label { font-size: 0.68rem; color: #64748b; margin-top: 0.25rem;
             text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }

/* ── SECTION CARDS ── */
.sec-card {
    background: #FFFFFF; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f3;
}
.sec-title {
    font-size: 0.95rem; font-weight: 700; color: #1B2A4A;
    border-bottom: 2px solid #C9A84C;
    padding-bottom: 0.25rem; margin-bottom: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.4px;
}
.sec-caption { font-size: 0.78rem; color: #6B7280; margin-top: -0.4rem; margin-bottom: 0.6rem; }

/* ── LIKELIHOOD BADGES ── */
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px; }
.badge-red    { background: #FEE2E2; color: #DC2626; }
.badge-yellow { background: #FEF3C7; color: #D97706; }
.badge-green  { background: #D1FAE5; color: #059669; }

/* ── UPLOAD ZONE ── */
.upload-zone {
    background: #f8f9ff; border: 2px dashed #C9A84C;
    border-radius: 10px; padding: 1.2rem; text-align: center; margin-bottom: 1rem;
}
.upload-zone p { color: #1B2A4A; font-size: 0.88rem; margin: 0; }

/* ── STATUS BADGE ── */
.badge-saved   { background:#D1FAE5; color:#065F46; padding:2px 10px;
                 border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-unsaved { background:#FEF3C7; color:#92400E; padding:2px 10px;
                 border-radius:20px; font-size:0.78rem; font-weight:600; }

/* ── LEGEND ── */
.legend-row { display:flex; gap:1.2rem; flex-wrap:wrap; align-items:center; margin-bottom:0.8rem; }
.legend-item { display:flex; align-items:center; gap:6px; font-size:0.78rem; color:#374151; }
.legend-dot  { width:11px; height:11px; border-radius:50%; display:inline-block; }

/* ── SIDEBAR ── */
div[data-testid="stSidebarContent"] { background: #1B2A4A !important; }
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p,
div[data-testid="stSidebarContent"] span { color: #cbd5e1 !important; font-size: 0.83rem !important; }
div[data-testid="stSidebarContent"] h2,
div[data-testid="stSidebarContent"] h3 { color: #FFFFFF !important; }
div[data-testid="stSidebarContent"] .sec-title { color: #C9A84C !important; border-color: #C9A84C !important; }

/* ── TABS ── */
button[data-baseweb="tab"] { font-size: 0.85rem !important; font-weight: 600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1B2A4A !important; border-bottom-color: #C9A84C !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
BASE      = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(BASE, "mlt_session_data.json")
THRESHOLD = 0.43

LIKELIHOOD_COLORS = {"Red": "#DC2626", "Yellow": "#F59E0B", "Green": "#059669"}
LIKELIHOOD_LABELS = {
    "Red":    "High Support Needed",
    "Yellow": "Moderate Support Needed",
    "Green":  "Likely Competitive",
}

PARTNER_MAP = {
    "Partner - Active": 1, "Premier Partner - Active": 1, "Core Partner - Active": 1,
    "Partner - Non Active": 0, "Partner - Prospect": 0, "Non-Partner": 0,
}

FUNCTIONAL_INTERESTS = [
    "Consulting (Management Consulting / Strategy)", "Finance (Corporate Finance)",
    "Finance (Investment Banking)", "Marketing", "Product Management",
    "Software Development", "Engineering", "Information Technology",
    "Project Management", "Business Development", "Operations",
    "Human Resources", "Supply Chain", "Sales", "Other",
]

POSITIVE_STATUSES = ["Offered", "Offered & Committed", "Offered & Declined",
                     "Offer Rescinded", "My offer has been rescinded."]
NEGATIVE_STATUSES = ["Denied", "Pending"]

# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def assign_likelihood(prob):
    if prob < 0.35: return "Red"
    if prob <= 0.60: return "Yellow"
    return "Green"

def fit_label(prob):
    flag = assign_likelihood(prob / 100 if prob > 1 else prob)
    icons = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢"}
    return f"{icons[flag]} {LIKELIHOOD_LABELS[flag]}"

def suggest_action(flag):
    return {
        "Red":    "Immediate intervention: refine strategy, target fit, interview prep",
        "Yellow": "Moderate coaching: strengthen positioning, sharpen application materials",
        "Green":  "Maintain momentum: prepare for interviews and close opportunities",
    }.get(flag, "")

def legend_html():
    items = [("#DC2626", "Red — High Support Needed"),
             ("#F59E0B", "Yellow — Moderate Support Needed"),
             ("#059669", "Green — Likely Competitive")]
    parts = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{c}"></span>{l}</span>'
        for c, l in items)
    return f'<div class="legend-row">{parts}</div>'

def kpi(label, value, accent=""):
    cls = f"kpi-card {accent}" if accent else "kpi-card"
    return f'<div class="{cls}"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'

def plotly_mlt(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

def compute_fairness(df, group_col):
    if group_col not in df.columns: return None
    subset = df.dropna(subset=[group_col, "Actual_Label"])
    if len(subset) == 0: return None
    rows = []
    for grp, gdf in subset.groupby(group_col):
        n = len(gdf)
        if n < 5: continue
        y_true = gdf["Actual_Label"].astype(int).values
        y_pred = gdf["Predicted_Label"].astype(int).values
        y_prob = gdf["Predicted_Probability"].values
        if len(np.unique(y_true)) < 2:
            rows.append({"Subgroup": grp, "Count": n,
                         "Actual Offer Rate": round(y_true.mean(), 3),
                         "Avg Predicted Prob": round(y_prob.mean(), 3),
                         "Precision": None, "Recall": None, "FPR": None, "FNR": None})
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({"Subgroup": grp, "Count": n,
                     "Actual Offer Rate": round(y_true.mean(), 3),
                     "Avg Predicted Prob": round(y_prob.mean(), 3),
                     "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
                     "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
                     "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 3),
                     "FNR": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 3)})
    if not rows: return None
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD LASSO MODEL
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_lasso():
    model = pickle.load(open(os.path.join(BASE, 'lasso_model.pkl'), 'rb'))
    pre   = pickle.load(open(os.path.join(BASE, 'lasso_preprocessor.pkl'), 'rb'))
    meta  = pickle.load(open(os.path.join(BASE, 'lasso_metadata.pkl'), 'rb'))
    return model, pre, meta

try:
    lasso_model, lasso_pre, lasso_meta = load_lasso()
    feature_cols = lasso_meta['feature_cols']
    numeric_cols = lasso_meta['numeric_cols']
    cat_cols     = lasso_meta['cat_cols']
    medians      = lasso_meta['medians']
    modes        = lasso_meta['modes']
    MODEL_LOADED = True
    # Try to get coefficients for contribution charts
    try:
        coefs = lasso_model.coef_[0]
        intercept = float(lasso_model.intercept_[0])
        coef_df = (pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs})
                   .assign(Abs=lambda d: d["Coefficient"].abs())
                   .query("Coefficient != 0")
                   .sort_values("Abs", ascending=False)
                   .drop(columns="Abs")
                   .reset_index(drop=True))
    except:
        coefs, intercept, coef_df = None, None, pd.DataFrame()
except Exception as e:
    MODEL_LOADED = False
    coefs, intercept, coef_df = None, None, pd.DataFrame()
    st.warning(f"⚠️ Model files not found ({e}). Scoring disabled — browse & upload still work.")

# ══════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def save_to_file():
    try:
        with open(SAVE_FILE, 'w') as f:
            json.dump({"saved_at": datetime.now().isoformat(),
                       "applicants": st.session_state.applicants}, f, indent=2)
        st.session_state.data_saved = True
    except Exception as e:
        st.error(f"Save failed: {e}")

def load_from_file():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE) as f:
                data = json.load(f)
            return data.get("applicants", []), data.get("saved_at", "")
        except: pass
    return [], ""

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
if 'applicants' not in st.session_state:
    st.session_state.applicants = []
if 'data_saved' not in st.session_state:
    st.session_state.data_saved = False
if 'auto_loaded' not in st.session_state:
    st.session_state.auto_loaded = True
    saved_apps, _ = load_from_file()
    if saved_apps:
        st.session_state.applicants = saved_apps

# ══════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════
def score_application(student: dict) -> float:
    if not MODEL_LOADED: return 0.0
    row = {}
    for col in feature_cols:
        val = student.get(col)
        if val is None:
            val = medians.get(col, 0) if col in numeric_cols else modes.get(col, "Unknown")
        row[col] = val
    df_row = pd.DataFrame([row])
    for col in numeric_cols:
        if col in df_row:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(medians.get(col, 0))
    for col in cat_cols:
        if col in df_row:
            df_row[col] = df_row[col].fillna(modes.get(col, "Unknown")).astype(str)
    try:
        X = lasso_pre.transform(df_row[feature_cols])
        return round(lasso_model.predict_proba(X)[0][1] * 100, 1)
    except: return 0.0

FORTUNE_500 = {
    "Amazon", "Target", "Google", "Visa Inc.", "Dell Technologies Inc.",
    "Citi", "AT&T", "Morgan Stanley", "JPMorgan Chase", "Goldman Sachs",
    "Bank of America", "Wells Fargo", "Microsoft", "Apple", "Meta",
    "Meta Platforms", "IBM", "Intel", "Oracle", "Cisco",
    "Johnson & Johnson", "Procter & Gamble", "PepsiCo", "Coca-Cola",
    "Nike", "Walt Disney", "Netflix", "Salesforce", "Adobe", "PayPal",
    "American Express", "Capital One", "T-Mobile", "Verizon",
    "Home Depot", "Walmart", "Costco", "General Electric",
    "Honeywell", "Lockheed Martin", "Boeing", "General Motors", "Ford",
    "ExxonMobil", "Chevron", "Pfizer", "Merck", "Eli Lilly",
    "Accenture", "Uber", "Mastercard", "Starbucks Coffee Company",
    "LinkedIn", "Deloitte", "BlackRock", "Charles Schwab",
    "Kearney", "Boston Consulting", "McKinsey", "Bain",
    "FICO", "Chick-Fil-A",
}

def infer_seniority(title: str) -> str:
    t = title.lower()
    if any(x in t for x in ['mba', 'graduate', 'grad']): return 'MBA / Graduate'
    if any(x in t for x in ['senior', 'sr.', 'lead', 'principal']): return 'Senior'
    if any(x in t for x in ['manager', 'director', 'vp', 'vice president']): return 'Manager/Director'
    if any(x in t for x in ['intern', 'internship', 'summer']): return 'Undergraduate'
    return 'Undergraduate'

def infer_position_type(title: str) -> str:
    t = title.lower()
    if any(x in t for x in ['analyst', 'analysis']): return 'Analyst'
    if any(x in t for x in ['engineer', 'developer', 'software']): return 'Engineer'
    if any(x in t for x in ['manager', 'management']): return 'Manager'
    if any(x in t for x in ['consult']): return 'Consultant'
    if any(x in t for x in ['intern', 'internship']): return 'Intern'
    return 'Intern'

def app_to_features(app: dict) -> dict:
    title     = str(app.get('job_title', ''))
    company   = str(app.get('company', ''))
    job_type  = str(app.get('job_type', ''))
    industry  = str(app.get('industry', ''))

    is_summer    = 1 if any(x in title.lower() for x in ['summer', 'intern']) else 0
    is_fortune   = 1 if company.strip() in FORTUNE_500 else 0
    title_words  = len(title.split()) if title else 1
    seniority    = infer_seniority(title)
    position     = infer_position_type(title)

    # Company size — default mid if unknown
    company_size = app.get('company_size', 'Mid (1K-10K)')

    # Industry subsector from stored field or default
    ind_subsector = industry if industry and industry not in ['', 'nan', 'None'] else 'Other'

    return {
        'Primary Functional Interest': app.get('func_interest', ''),
        'Designated Low Income':       int(app.get('low_income', False)),
        'First Generation College':    'Yes' if app.get('first_gen') else 'No',
        'Undergrad GPA':               app.get('gpa') or medians.get('Undergrad GPA', 3.5),
        'Pell Grant Count':            app.get('pell', 0),
        'SAT Score':                   app.get('sat', 0),
        'Summer_Internship':           is_summer,
        'Position_Type':               position,
        'Seniority_Level':             seniority,
        'Title_Word_Count':            title_words,
        'Industry_Subsector':          ind_subsector,
        'Company_Size_Bucket':         company_size,
        'Is_Fortune500':               is_fortune,
    }

def excel_row_to_applicant(row: pd.Series) -> dict:
    partner_raw = str(row.get('Partner Org?', ''))
    is_partner  = PARTNER_MAP.get(partner_raw, 0)
    status = str(row.get('Application Status', 'Applied'))
    return {
        "id":           str(row.get('Program Enrollment: Enrollment ID', f"ID-{id(row)}")),
        "name":         str(row.get('Program Enrollment: Enrollment ID', 'Unknown')),
        "gpa":          float(row['Undergrad GPA']) if pd.notna(row.get('Undergrad GPA')) else None,
        "sat":          int(row['SAT Score'])        if pd.notna(row.get('SAT Score')) else 0,
        "pell":         int(row['Pell Grant Count'])  if pd.notna(row.get('Pell Grant Count')) else 0,
        "low_income":   bool(row.get('Designated Low Income', False)),
        "first_gen":    row.get('First Generation College', 'No') == 'Yes',
        "gender":       str(row.get('Gender', '')),
        "race":         str(row.get('Race', '')),
        "func_interest":str(row.get('Primary Functional Interest', '')),
        "program":      str(row.get('Program Enrollment: Program', '')),
        "company":      str(row.get('Related Organization', '')),
        "job_title":    str(row.get('Title', '')),
        "job_type":     str(row.get('Type', '')),
        "partner_org":  is_partner,
        "app_status":   status,
        "coach":        str(row.get('Program Enrollment: Coach', '')),
        "track":        str(row.get('Program Enrollment: Program Track', '')),
        "industry":     str(row.get('Primary Industry Interest', '')),
        "company_size": "Mid (1K-10K)",
        "notes":        "",
        "score":        None,
        "actual_offer": 1 if status in POSITIVE_STATUSES else (0 if status in NEGATIVE_STATUSES else None),
        "added_at":     datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="mlt-hero">
  <div class="subtitle">MANAGEMENT LEADERSHIP FOR TOMORROW</div>
  <h1>🎯 Career Prep · Job Fit Scorer</h1>
  <p>LASSO Model · 13 features · AUC 0.903 &nbsp;|&nbsp; Upload applicants · Score offers · Track results · Fairness monitoring</p>
  <div style="margin-top:0.6rem;">
    <span class="mlt-badge">CP 2018–2023 Training</span>
    <span class="mlt-badge">CP 2024 Validated</span>
    <span class="mlt-badge">Threshold 0.43</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Manual Entry
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👤 Add Student Manually")
    st.markdown('<div class="sec-title">📚 Academic</div>', unsafe_allow_html=True)
    s_gpa  = st.slider("Undergrad GPA", 0.0, 4.0, 3.5, 0.01)
    s_sat  = st.number_input("SAT Score (0 = unknown)", 0, 1600, 0, 10)
    s_pell = st.number_input("Pell Grant Count", 0, 10, 0)

    st.markdown('<div class="sec-title">🏠 Background</div>', unsafe_allow_html=True)
    s_low_income = st.selectbox("Designated Low Income?", ["No", "Yes"]) == "Yes"
    s_first_gen  = st.selectbox("First Generation College?", ["No", "Yes"]) == "Yes"
    s_gender     = st.selectbox("Gender", ["Female", "Male", "Prefer not to identify", "Transgender", ""])
    s_race       = st.selectbox("Race", ["Black or African American", "Hispanic / Latino",
                                          "White", "Asian", "American Indian or Alaskan Native", "Other", ""])
    st.markdown('<div class="sec-title">🎓 Program</div>', unsafe_allow_html=True)
    s_func   = st.selectbox("Primary Functional Interest", FUNCTIONAL_INTERESTS)
    s_track  = st.selectbox("Program Track", ["Corporate Management", "Software Engineering/Technology",
                                               "Finance", "Consulting", "Other", ""])
    s_name   = st.text_input("Student Label / ID", placeholder="e.g. Jane D.")

    st.markdown('<div class="sec-title">💼 Job Application</div>', unsafe_allow_html=True)
    s_company    = st.text_input("Company", placeholder="e.g. Goldman Sachs")
    s_title      = st.text_input("Job Title", placeholder="e.g. Summer Analyst")
    s_partner    = st.selectbox("Partner Organization?", ["Yes", "No"]) == "Yes"
    s_app_status = st.selectbox("Application Status", [
        "Applied", "Pending", "Offered & Committed", "Offered & Declined",
        "Denied", "Withdrew Application", "Offered"])

    if st.button("➕ Add Applicant", use_container_width=True, type="primary"):
        app_entry = {
            "id": f"manual-{datetime.now().strftime('%H%M%S')}",
            "name": s_name or s_company or "Student",
            "gpa": s_gpa, "sat": s_sat, "pell": s_pell,
            "low_income": s_low_income, "first_gen": s_first_gen,
            "gender": s_gender, "race": s_race,
            "func_interest": s_func, "track": s_track,
            "program": "", "company": s_company, "job_title": s_title,
            "job_type": "Internship (Undergrad)", "partner_org": int(s_partner),
            "app_status": s_app_status, "coach": "", "industry": s_func, "company_size": "Mid (1K-10K)",
            "notes": "", "score": None, "actual_offer": None,
            "added_at": datetime.now().isoformat(),
        }
        st.session_state.applicants.append(app_entry)
        st.session_state.data_saved = False
        st.success("✅ Added!")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save", use_container_width=True):
            save_to_file(); st.success("Saved!")
    with c2:
        if st.button("🔄 Reload", use_container_width=True):
            saved, _ = load_from_file()
            if saved:
                st.session_state.applicants = saved
                st.success(f"Loaded {len(saved)}")
            else: st.info("No saved data")

    badge = ('<span class="badge-saved">● Saved</span>' if st.session_state.data_saved
             else '<span class="badge-unsaved">● Unsaved changes</span>')
    st.markdown(badge, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Total applicants:** {len(st.session_state.applicants)}")
    scored = sum(1 for a in st.session_state.applicants if a.get('score') is not None)
    st.markdown(f"**Scored:** {scored}")

# ══════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload Excel",
    "📋 Applicant List",
    "📊 Score & Results",
    "🔍 Application Detail",
    "⚖️ Fairness Monitor",
    "🔧 Model Insights",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD EXCEL
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-card"><div class="sec-title">Upload Your MLT Data File</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
      <p>📎 Upload your <strong>MLT_CP_Anon_Data</strong> Excel file (.xlsx)<br>
      Columns auto-detected · Data stays loaded for the full session · Auto-saved after import</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose Excel file", type=["xlsx", "xls"], label_visibility="collapsed")

    if uploaded:
        try:
            with st.spinner("Reading file…"):
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Select sheet", xls.sheet_names)
                df = pd.read_excel(uploaded, sheet_name=sheet)

            st.success(f"✅ Loaded **{len(df):,} rows** × **{len(df.columns)} columns** from `{sheet}`")

            with st.expander("👀 Preview (first 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            expected = ['Program Enrollment: Enrollment ID', 'Undergrad GPA', 'SAT Score',
                        'Pell Grant Count', 'Designated Low Income', 'First Generation College',
                        'Gender', 'Race', 'Primary Functional Interest',
                        'Related Organization', 'Title', 'Partner Org?', 'Application Status',
                        'Program Enrollment: Coach', 'Program Enrollment: Program',
                        'Program Enrollment: Program Track']
            found   = [c for c in expected if c in df.columns]
            missing = [c for c in expected if c not in df.columns]

            cc1, cc2 = st.columns(2)
            with cc1:
                st.success(f"**{len(found)} columns found**")
                for c in found: st.markdown(f"✅ `{c}`")
            with cc2:
                if missing:
                    st.warning(f"**{len(missing)} columns missing** (defaults used)")
                    for c in missing: st.markdown(f"⚠️ `{c}`")

            st.markdown('<div class="sec-title" style="margin-top:1rem;">Filter Before Importing</div>', unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                prog_opts = ["All"] + sorted(df['Program Enrollment: Program'].dropna().unique().tolist()) if 'Program Enrollment: Program' in df.columns else ["All"]
                sel_prog = st.selectbox("Program cohort", prog_opts)
            with fc2:
                stat_opts = ["All"] + sorted(df['Application Status'].dropna().unique().tolist()) if 'Application Status' in df.columns else ["All"]
                sel_stat = st.selectbox("Application Status", stat_opts)
            with fc3:
                max_rows = st.number_input("Max rows (0 = all)", 0, 10000, 0)

            fdf = df.copy()
            if sel_prog != "All" and 'Program Enrollment: Program' in fdf.columns:
                fdf = fdf[fdf['Program Enrollment: Program'] == sel_prog]
            if sel_stat != "All" and 'Application Status' in fdf.columns:
                fdf = fdf[fdf['Application Status'] == sel_stat]
            if max_rows > 0:
                fdf = fdf.head(max_rows)

            st.info(f"📦 **{len(fdf):,} rows** will be imported")

            if st.button("⬇️ Import into Applicant List", type="primary", use_container_width=True):
                new_apps, errors = [], 0
                for _, row in fdf.iterrows():
                    try: new_apps.append(excel_row_to_applicant(row))
                    except: errors += 1
                st.session_state.applicants.extend(new_apps)
                st.session_state.data_saved = False
                save_to_file()
                st.success(f"✅ Imported **{len(new_apps)}** applicants" +
                           (f" ({errors} skipped)" if errors else "") + " · Auto-saved.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        saved_apps, saved_at = load_from_file()
        if saved_apps:
            st.info(f"💾 **{len(saved_apps)} applicants** from last session ({saved_at[:16] if saved_at else ''})")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — APPLICANT LIST
# ══════════════════════════════════════════════════════════════════════
with tab2:
    apps = st.session_state.applicants
    if not apps:
        st.info("No applicants yet. Add manually from the sidebar or upload an Excel file.")
    else:
        total  = len(apps)
        scored = sum(1 for a in apps if a.get('score') is not None)
        strong = sum(1 for a in apps if (a.get('score') or 0) >= 65)
        avg_sc = round(np.mean([a['score'] for a in apps if a.get('score') is not None]), 1) if scored else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("Total Applicants", total), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Scored", scored, "gold"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Strong Fits (≥65%)", strong, "green"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("Avg Score", f"{avg_sc}%" if scored else "—"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb, cc = st.columns([2, 1, 1])
        with ca: search = st.text_input("🔍 Search by name, company, or ID")
        with cb: show_scored = st.selectbox("Filter", ["All", "Scored only", "Unscored only"])
        with cc: sort_by = st.selectbox("Sort by", ["Added (newest)", "Score (high→low)", "Score (low→high)", "Company"])

        display = apps.copy()
        if search:
            s = search.lower()
            display = [a for a in display if s in a.get('name','').lower() or
                       s in a.get('company','').lower() or s in a.get('id','').lower()]
        if show_scored == "Scored only":   display = [a for a in display if a.get('score') is not None]
        elif show_scored == "Unscored only": display = [a for a in display if a.get('score') is None]
        if sort_by == "Score (high→low)": display = sorted(display, key=lambda a: a.get('score') or -1, reverse=True)
        elif sort_by == "Score (low→high)": display = sorted(display, key=lambda a: a.get('score') or 999)
        elif sort_by == "Company": display = sorted(display, key=lambda a: a.get('company',''))
        else: display = list(reversed(display))

        st.markdown(f"**{len(display)} shown**")

        for idx, app in enumerate(display):
            sc = app.get('score')
            status = fit_label(sc/100 if sc is not None else 0) if sc is not None else "⬜ Not scored"
            score_text = f"{sc}%" if sc is not None else "—"

            with st.expander(f"{status}  ·  **{app.get('name','?')}**  ·  {app.get('company','?')}  ·  {app.get('job_title','?')}  ·  {score_text}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Academic**")
                    st.write(f"GPA: {app.get('gpa','—')}  |  SAT: {app.get('sat') or '—'}  |  Pell: {app.get('pell',0)}")
                    st.write(f"Low Income: {'Yes' if app.get('low_income') else 'No'}  |  First Gen: {'Yes' if app.get('first_gen') else 'No'}")
                with col2:
                    st.markdown("**Demographics & Program**")
                    st.write(f"Gender: {app.get('gender','—')}  |  Race: {app.get('race','—')}")
                    st.write(f"Track: {app.get('track','—')}  |  Interest: {app.get('func_interest','—')}")
                with col3:
                    st.markdown("**Application**")
                    st.write(f"Status: {app.get('app_status','—')}")
                    st.write(f"Partner: {'Yes' if app.get('partner_org') else 'No'}  |  Coach: {app.get('coach','—')}")

                new_note = st.text_area("Coach Notes", value=app.get('notes',''),
                                        key=f"note_{app['id']}_{idx}", height=60)
                if new_note != app.get('notes',''):
                    real_idx = next((i for i, a in enumerate(st.session_state.applicants) if a['id'] == app['id']), -1)
                    if real_idx >= 0:
                        st.session_state.applicants[real_idx]['notes'] = new_note
                        st.session_state.data_saved = False

                if st.button("🗑️ Remove", key=f"del_{app['id']}_{idx}"):
                    st.session_state.applicants = [a for a in st.session_state.applicants if a['id'] != app['id']]
                    st.session_state.data_saved = False
                    st.rerun()

        st.markdown("---")
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("💾 Save All Changes", use_container_width=True, type="primary"):
                save_to_file(); st.success("Saved!")
        with col_y:
            if st.button("🗑️ Clear All Applicants", use_container_width=True):
                st.session_state.applicants = []
                if os.path.exists(SAVE_FILE): os.remove(SAVE_FILE)
                st.session_state.data_saved = True
                st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SCORE & RESULTS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    apps = st.session_state.applicants
    if not apps:
        st.info("No applicants to score. Add them via the sidebar or Upload tab.")
    elif not MODEL_LOADED:
        st.error("Model not loaded. Ensure lasso_model.pkl, lasso_preprocessor.pkl, lasso_metadata.pkl are in the app folder.")
    else:
        st.markdown(legend_html(), unsafe_allow_html=True)
        unscored = [a for a in apps if a.get('score') is None]
        st.info(f"**{len(unscored)} unscored**  |  **{len(apps) - len(unscored)} already scored**")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Score All Applicants", type="primary", use_container_width=True):
                with st.spinner(f"Scoring {len(apps)} applicants…"):
                    for i, app in enumerate(st.session_state.applicants):
                        st.session_state.applicants[i]['score'] = score_application(app_to_features(app))
                save_to_file()
                st.success("✅ Done — results saved.")
                st.rerun()
        with c2:
            if st.button("🔁 Score Unscored Only", use_container_width=True):
                with st.spinner("Scoring…"):
                    for i, app in enumerate(st.session_state.applicants):
                        if app.get('score') is None:
                            st.session_state.applicants[i]['score'] = score_application(app_to_features(app))
                save_to_file(); st.success("Done."); st.rerun()

        scored_apps = [a for a in apps if a.get('score') is not None]
        if scored_apps:
            # KPI row
            scores = [a['score'] for a in scored_apps]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.markdown(kpi("Scored", len(scored_apps)), unsafe_allow_html=True)
            with c2: st.markdown(kpi("Best Fit", f"{max(scores):.1f}%", "green"), unsafe_allow_html=True)
            with c3: st.markdown(kpi("Strong Fits ≥65%", sum(1 for s in scores if s >= 65), "green"), unsafe_allow_html=True)
            with c4: st.markdown(kpi("Moderate 40–65%", sum(1 for s in scores if 40 <= s < 65), "amber"), unsafe_allow_html=True)
            with c5: st.markdown(kpi("Reach <40%", sum(1 for s in scores if s < 40), "red"), unsafe_allow_html=True)

            # Results table
            st.markdown('<div class="sec-card"><div class="sec-title">Ranked Results</div>', unsafe_allow_html=True)
            results_df = pd.DataFrame([{
                "Rank": 0, "ID": a.get('id',''), "Name": a.get('name',''),
                "Company": a.get('company',''), "Job Title": a.get('job_title',''),
                "Coach": a.get('coach',''), "Track": a.get('track',''),
                "Fit": fit_label(a['score']/100), "Score (%)": a['score'],
                "Status": a.get('app_status',''), "Notes": a.get('notes',''),
            } for a in scored_apps]).sort_values("Score (%)", ascending=False).reset_index(drop=True)
            results_df['Rank'] = results_df.index + 1

            st.dataframe(results_df, use_container_width=True, height=400,
                column_config={"Score (%)": st.column_config.ProgressColumn(
                    "Score (%)", format="%.1f%%", min_value=0, max_value=100)})
            st.markdown("</div>", unsafe_allow_html=True)

            # Distribution chart
            st.markdown('<div class="sec-card"><div class="sec-title">Score Distribution</div>', unsafe_allow_html=True)
            bins = np.arange(0, 105, 5)
            counts, edges = np.histogram(scores, bins=bins)
            mids = (edges[:-1] + edges[1:]) / 2
            colors = [LIKELIHOOD_COLORS[assign_likelihood(m/100)] for m in mids]
            fig_dist = go.Figure(go.Bar(x=mids, y=counts, marker_color=colors, width=4.5))
            fig_dist.update_layout(xaxis_title="Score (%)", yaxis_title="Count")
            plotly_mlt(fig_dist, 280)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Export
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", data=csv,
                file_name=f"mlt_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — APPLICATION DETAIL
# ══════════════════════════════════════════════════════════════════════
with tab4:
    scored_apps = [a for a in st.session_state.applicants if a.get('score') is not None]
    if not scored_apps:
        st.info("Score some applicants first (go to Score & Results tab).")
    else:
        sorted_apps = sorted(scored_apps, key=lambda a: a.get('score', 0))
        labels = [f"{a.get('name','?')} | {a.get('company','?')} | {a.get('score',0):.1f}%" for a in sorted_apps]
        sel_idx = st.selectbox("Select Application", range(len(sorted_apps)),
                               format_func=lambda i: labels[i], key="detail_sel")
        app = sorted_apps[sel_idx]
        prob = app['score'] / 100
        flag = assign_likelihood(prob)
        color = LIKELIHOOD_COLORS[flag]

        # Profile card
        st.markdown('<div class="sec-card"><div class="sec-title">Application Profile</div>', unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.markdown(f"**ID:** {app.get('id','—')}")
            st.markdown(f"**Coach:** {app.get('coach','—')}")
        with dc2:
            st.markdown(f"**Track:** {app.get('track','—')}")
            st.markdown(f"**Company:** {app.get('company','—')}")
        with dc3:
            st.markdown(f"**Title:** {app.get('job_title','—')}")
            st.markdown(f"**Interest:** {app.get('func_interest','—')}")
        with dc4:
            st.markdown(f"**Score:** {app['score']:.1f}%")
            badge_cls = flag.lower()
            st.markdown(f'<span class="badge badge-{badge_cls}">{flag} — {LIKELIHOOD_LABELS[flag]}</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Gauge + contributions
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.markdown('<div class="sec-card"><div class="sec-title">Probability Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"valueformat": ".1%"},
                gauge=dict(
                    axis=dict(range=[0, 1], tickformat=".0%"),
                    bar=dict(color=color),
                    steps=[dict(range=[0, 0.35], color="#FEE2E2"),
                           dict(range=[0.35, 0.60], color="#FEF3C7"),
                           dict(range=[0.60, 1.0], color="#D1FAE5")],
                    threshold=dict(line=dict(color="#1B2A4A", width=3), thickness=0.8, value=THRESHOLD),
                ),
            ))
            plotly_mlt(fig_gauge, 260)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with gc2:
            st.markdown('<div class="sec-card"><div class="sec-title">Top Contributing Factors</div>', unsafe_allow_html=True)
            if MODEL_LOADED and coefs is not None:
                try:
                    feats = app_to_features(app)
                    row = {col: feats.get(col, medians.get(col, 0) if col in numeric_cols else modes.get(col, "Unknown"))
                           for col in feature_cols}
                    df_row = pd.DataFrame([row])
                    for col in numeric_cols:
                        if col in df_row: df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(medians.get(col, 0))
                    for col in cat_cols:
                        if col in df_row: df_row[col] = df_row[col].fillna(modes.get(col, "Unknown")).astype(str)
                    X_sc = lasso_pre.transform(df_row[feature_cols])
                    contributions = X_sc[0] * coefs
                    contrib_df = pd.DataFrame({"Feature": feature_cols, "Contribution": contributions})
                    contrib_df = contrib_df[contrib_df["Contribution"].abs() > 0.001]
                    contrib_df = contrib_df.sort_values("Contribution", ascending=True).tail(10)

                    fig_contrib = go.Figure(go.Bar(
                        x=contrib_df["Contribution"], y=contrib_df["Feature"], orientation="h",
                        marker_color=[LIKELIHOOD_COLORS["Green"] if c > 0 else LIKELIHOOD_COLORS["Red"]
                                      for c in contrib_df["Contribution"]],
                    ))
                    fig_contrib.update_layout(xaxis_title="Contribution", yaxis_title="")
                    plotly_mlt(fig_contrib, 260)
                    st.plotly_chart(fig_contrib, use_container_width=True)

                    pos = contrib_df[contrib_df["Contribution"] > 0]["Feature"].tolist()
                    neg = contrib_df[contrib_df["Contribution"] < 0]["Feature"].tolist()
                    if pos: st.markdown(f"✅ **Helping:** {', '.join(pos[:3])}")
                    if neg: st.markdown(f"⚠️ **Hurting:** {', '.join(neg[:3])}")
                except Exception as e:
                    st.info(f"Contribution chart unavailable: {e}")
            else:
                st.info("Coefficient data not available.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Suggested action
        st.markdown(f'<div class="sec-card"><div class="sec-title">Suggested Coach Action</div>'
                    f'<p>{suggest_action(flag)}</p></div>', unsafe_allow_html=True)

        # Coach notes
        st.markdown('<div class="sec-card"><div class="sec-title">Coach Notes</div>', unsafe_allow_html=True)
        note_key = f"detail_note_{app['id']}"
        new_note = st.text_area("Notes", value=app.get('notes',''), key=note_key, height=100, label_visibility="collapsed")
        if new_note != app.get('notes',''):
            real_idx = next((i for i, a in enumerate(st.session_state.applicants) if a['id'] == app['id']), -1)
            if real_idx >= 0:
                st.session_state.applicants[real_idx]['notes'] = new_note
                st.session_state.data_saved = False
        st.caption("Notes auto-persist with Save button in sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — FAIRNESS MONITOR
# ══════════════════════════════════════════════════════════════════════
with tab5:
    scored_apps = [a for a in st.session_state.applicants if a.get('score') is not None]
    apps_with_actual = [a for a in scored_apps if a.get('actual_offer') is not None]

    if len(apps_with_actual) < 20:
        st.info(f"Fairness analysis needs at least 20 applicants with known outcomes (Offered/Denied). "
                f"Currently have **{len(apps_with_actual)}**. Import more data with known statuses.")
    else:
        fair_df = pd.DataFrame([{
            "Actual_Label":          a.get('actual_offer'),
            "Predicted_Label":       1 if (a.get('score') or 0) >= (THRESHOLD * 100) else 0,
            "Predicted_Probability": (a.get('score') or 0) / 100,
            "Gender":                a.get('gender',''),
            "Race":                  a.get('race',''),
            "First Generation College": 'Yes' if a.get('first_gen') else 'No',
            "Designated Low Income": 'Yes' if a.get('low_income') else 'No',
            "Program Track":         a.get('track',''),
            "Pell Grant":            'Pell Recipient' if (a.get('pell') or 0) > 0 else 'No Pell',
        } for a in apps_with_actual])

        st.markdown(legend_html(), unsafe_allow_html=True)

        fairness_groups = {
            "Gender": "Gender", "Race": "Race",
            "First Generation": "First Generation College",
            "Low Income": "Designated Low Income",
            "Program Track": "Program Track",
            "Pell Grant": "Pell Grant",
        }
        avail = {k: v for k, v in fairness_groups.items() if v in fair_df.columns}
        sel_group = st.selectbox("Select Subgroup Category", list(avail.keys()))
        group_col = avail[sel_group]
        fm = compute_fairness(fair_df, group_col)

        if fm is not None:
            # KPI strip
            fk1, fk2, fk3, fk4 = st.columns(4)
            with fk1: st.markdown(kpi("Subgroups", len(fm)), unsafe_allow_html=True)
            with fk2: st.markdown(kpi("Apps Analyzed", fm["Count"].sum()), unsafe_allow_html=True)
            r_spread = fm["Recall"].dropna()
            r_val = round(r_spread.max() - r_spread.min(), 3) if len(r_spread) > 0 else 0
            with fk3: st.markdown(kpi("Recall Spread", r_val, "amber" if r_val > 0.10 else ""), unsafe_allow_html=True)
            fnr_spread = fm["FNR"].dropna()
            f_val = round(fnr_spread.max() - fnr_spread.min(), 3) if len(fnr_spread) > 0 else 0
            with fk4: st.markdown(kpi("FNR Spread", f_val, "red" if f_val > 0.10 else ""), unsafe_allow_html=True)

            # Metrics table
            st.markdown('<div class="sec-card"><div class="sec-title">Fairness Metrics Table</div>', unsafe_allow_html=True)
            st.dataframe(fm, use_container_width=True, hide_index=True, height=350)
            st.markdown("</div>", unsafe_allow_html=True)

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown('<div class="sec-card"><div class="sec-title">Recall & FNR by Subgroup</div>', unsafe_allow_html=True)
                fm_c = fm.dropna(subset=["Recall", "FNR"])
                if len(fm_c) > 0:
                    fig_f1 = go.Figure()
                    fig_f1.add_trace(go.Bar(x=fm_c["Subgroup"], y=fm_c["Recall"], name="Recall", marker_color="#059669"))
                    fig_f1.add_trace(go.Bar(x=fm_c["Subgroup"], y=fm_c["FNR"], name="FNR", marker_color="#DC2626"))
                    fig_f1.update_layout(barmode="group")
                    plotly_mlt(fig_f1, 320)
                    st.plotly_chart(fig_f1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with ch2:
                st.markdown('<div class="sec-card"><div class="sec-title">Avg Predicted Probability by Subgroup</div>', unsafe_allow_html=True)
                fig_f2 = go.Figure(go.Bar(
                    x=fm["Subgroup"], y=fm["Avg Predicted Prob"],
                    marker_color=[LIKELIHOOD_COLORS[assign_likelihood(v)] for v in fm["Avg Predicted Prob"]],
                    text=[f"{v:.1%}" for v in fm["Avg Predicted Prob"]], textposition="auto",
                ))
                plotly_mlt(fig_f2, 320)
                st.plotly_chart(fig_f2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Disparity flags
            st.markdown('<div class="sec-card"><div class="sec-title">Disparity Flags</div>', unsafe_allow_html=True)
            y_act = fair_df["Actual_Label"].astype(int).values
            y_prd = fair_df["Predicted_Label"].astype(int).values
            overall_recall = recall_score(y_act, y_prd, zero_division=0)
            tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_act, y_prd, labels=[0, 1]).ravel()
            overall_fnr = fn_o / (fn_o + tp_o) if (fn_o + tp_o) > 0 else 0
            flags_found = False
            for _, row in fm.dropna(subset=["Recall", "FNR"]).iterrows():
                if abs(row["Recall"] - overall_recall) > 0.10:
                    st.warning(f"⚠️ **{row['Subgroup']}** — Recall ({row['Recall']:.3f}) differs from overall ({overall_recall:.3f}) by >0.10")
                    flags_found = True
                if abs(row["FNR"] - overall_fnr) > 0.10:
                    st.warning(f"⚠️ **{row['Subgroup']}** — FNR ({row['FNR']:.3f}) differs from overall ({overall_fnr:.3f}) by >0.10")
                    flags_found = True
            if not flags_found:
                st.success("✅ No subgroup disparities exceed the 0.10 threshold.")
            st.markdown("</div>", unsafe_allow_html=True)

            fair_csv = fm.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export Fairness Report (.csv)", fair_csv,
                               "mlt_fairness_report.csv", "text/csv")
        else:
            st.info(f"Not enough data to compute fairness for '{sel_group}'.")

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-card"><div class="sec-title">Model Configuration</div>', unsafe_allow_html=True)
    mi1, mi2, mi3 = st.columns(3)
    with mi1:
        st.markdown("**Algorithm:** L1 (LASSO) Logistic Regression")
        st.markdown("**Class Weights:** Balanced")
        st.markdown("**Cross-Validation:** 5-fold, ROC-AUC scoring")
    with mi2:
        st.markdown(f"**Decision Threshold:** {THRESHOLD}")
        st.markdown(f"**Model Status:** {'✅ Loaded' if MODEL_LOADED else '❌ Not loaded'}")
        st.markdown(f"**Features:** {len(feature_cols) if MODEL_LOADED else '—'}")
    with mi3:
        st.markdown("**Train Cohorts:** CP 2018–2023")
        st.markdown("**Validation:** CP 2024")
        st.markdown("**AUC:** 0.903")
    st.markdown("</div>", unsafe_allow_html=True)

    if MODEL_LOADED and len(coef_df) > 0:
        # Coefficient chart
        st.markdown('<div class="sec-card"><div class="sec-title">LASSO Coefficients (Non-Zero Features)</div>', unsafe_allow_html=True)
        chart_df = coef_df.sort_values("Coefficient", ascending=True)
        fig_coef = go.Figure(go.Bar(
            x=chart_df["Coefficient"], y=chart_df["Feature"], orientation="h",
            marker_color=[LIKELIHOOD_COLORS["Green"] if c > 0 else LIKELIHOOD_COLORS["Red"]
                          for c in chart_df["Coefficient"]],
        ))
        fig_coef.update_layout(xaxis_title="Coefficient Value", yaxis_title="")
        plotly_mlt(fig_coef, max(350, len(chart_df) * 28 + 80))
        st.plotly_chart(fig_coef, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Positive / Negative split
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="sec-card"><div class="sec-title">↑ Features Increasing Offer Likelihood</div>', unsafe_allow_html=True)
            pos = coef_df[coef_df["Coefficient"] > 0]
            for _, r in pos.iterrows():
                st.markdown(f"- **{r['Feature']}**: +{r['Coefficient']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="sec-card"><div class="sec-title">↓ Features Decreasing Offer Likelihood</div>', unsafe_allow_html=True)
            neg = coef_df[coef_df["Coefficient"] < 0]
            for _, r in neg.iterrows():
                st.markdown(f"- **{r['Feature']}**: {r['Coefficient']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Coefficient table
        st.markdown('<div class="sec-card"><div class="sec-title">Full Coefficient Table</div>', unsafe_allow_html=True)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        coef_csv = coef_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Coefficients (.csv)", coef_csv,
                           "mlt_lasso_coefficients.csv", "text/csv")
    else:
        st.info("Load the LASSO model files to see coefficient insights.")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2rem; padding:1rem;
            border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.75rem;">
    <strong style="color:#1B2A4A;">Management Leadership for Tomorrow</strong> · 
    Career Prep Job Fit Scorer · LASSO Model v2.0 · AUC 0.903
</div>
""", unsafe_allow_html=True)
